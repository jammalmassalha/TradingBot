#!/usr/bin/env python3
"""Backtest the 1H strategy, then gate Binance connectivity on human approval.

This script is the safety interlock of the whole pipeline. Nothing in this
project may reach the exchange without first passing through it: it replays the
tactical risk layer over real 1H history, prints a full scorecard, and stops
dead unless a human types ``y``.

The bias driving the replay is the deterministic ``TrendBiasModel``, not the
agent graph. Replaying 1440 hourly bars through a multi-agent deliberation
would cost thousands of LLM calls, take hours, and still not reproduce between
runs — so it could not function as a gate. What this validates is the part that
must be deterministic anyway: sizing, ATR brackets, fees, exit precedence and
the metrics. The agents supply direction live; this proves the machinery that
direction flows through.

Examples:
    python run_1h_backtest.py                       # 60 days of $TRADING_SYMBOL 1H
    python run_1h_backtest.py --symbol ETHUSDT      # override the pair
    python run_1h_backtest.py --days 30 --yes       # non-interactive
    python run_1h_backtest.py --csv bars.csv        # offline, from a saved frame
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance_ohlcv import (
    compute_indicators,
    load_1h_ohlcv,
    resolve_market,
    resolve_symbol,
)
from tradingagents.scalping.engine import BacktestConfig, run_backtest
from tradingagents.scalping.metrics import render_report
from tradingagents.scalping.signals import TrendBiasModel

logger = logging.getLogger("run_1h_backtest")

APPROVAL_PROMPT = "Do you approve these backtest metrics to proceed with Binance connection? (y/n): "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1H backtest and Binance approval gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default=None,
                        help="pair to trade; defaults to $TRADING_SYMBOL, else BTCUSDT")
    parser.add_argument("--days", type=int, default=60,
                        help="days of 1H history to replay (30-60 recommended)")
    parser.add_argument("--balance", type=float, default=10_000.0,
                        help="starting balance in USDT")
    parser.add_argument("--risk-pct", type=float, default=2.0,
                        help="percent of equity risked per trade")
    parser.add_argument("--atr-stop", type=float, default=1.5,
                        help="stop distance in ATR multiples")
    parser.add_argument("--atr-target", type=float, default=3.0,
                        help="take-profit distance in ATR multiples")
    parser.add_argument("--fee-bps", type=float, default=10.0,
                        help="per-side fee in basis points of notional")
    parser.add_argument("--market", default=None, choices=["spot", "futures"],
                        help="spot candles or USD-M perp candles; "
                             "defaults to $BINANCE_MARKET, else spot")
    parser.add_argument("--leverage", type=float, default=None,
                        help="futures only; widens the notional ceiling, not the risk")
    parser.add_argument("--funding-bps", dest="funding_bps", type=float, default=1.0,
                        help="futures only; funding paid per 8h, in bps of notional")
    parser.add_argument("--no-short", action="store_true",
                        help="long-only, matching a spot account that cannot short")
    parser.add_argument("--csv", type=Path, default=None,
                        help="replay a saved OHLCV CSV instead of calling Binance")
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive gate (for CI; never for live)")
    parser.add_argument("--env", default=None, choices=["testnet", "live"],
                        help="Binance environment to probe after approval")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    """The indicator-bearing 1H frame, from Binance or a saved CSV.

    The CSV path exists because the live endpoint is unreachable from some
    networks (corporate egress filters and several jurisdictions block Binance
    outright); without it the gate could not be exercised at all there.
    """
    if args.csv is not None:
        if not args.csv.is_file():
            raise FileNotFoundError(f"no such CSV: {args.csv}")
        frame = pd.read_csv(args.csv)
        if "Date" not in frame.columns:
            raise ValueError("CSV must have a 'Date' column of hourly timestamps")
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
        frame = frame.sort_values("Date").reset_index(drop=True)
        return compute_indicators(frame)

    label = "USD-M perp" if args.market == "futures" else "spot"
    print(f"Fetching {args.days} days of {args.symbol} 1H {label} candles from Binance...")
    return load_1h_ohlcv(symbol=args.symbol, days=args.days, market=args.market)


def confirm(prompt: str = APPROVAL_PROMPT) -> bool:
    """Ask for approval. Anything but ``y`` — including EOF — is a refusal.

    A non-interactive stdin raises EOFError; that must read as "no". Defaulting
    to yes there would let a cron job or a piped invocation route live orders
    with no human having seen the numbers.
    """
    try:
        answer = input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() == "y"


def probe_binance(
    symbol: str,
    env: str | None,
    market: str = "spot",
    leverage: float | None = None,
) -> None:
    """Post-approval connectivity probe. Read-only: it places no orders."""
    try:
        from binance_executor import ExecutionRejected, build_executor
    except ImportError as exc:
        print(f"Execution bridge unavailable: {exc}")
        return

    extra = {}
    if market == "futures" and leverage:
        extra["leverage"] = int(leverage)
    try:
        executor = build_executor(symbol=symbol, env=env, market=market, **extra)
    except ValueError as exc:
        print(f"Executor configuration error: {exc}")
        return

    print(f"\nConnecting to Binance ({executor.env}, {executor.market})...")
    if executor.is_live:
        print("  *** LIVE ENVIRONMENT — orders would use real funds ***")
    if market == "futures":
        print("  *** LEVERAGED PERPETUALS — losses can exceed the margin posted ***")
    try:
        for key, value in executor.check_connectivity().items():
            print(f"  {key:>20}: {value}")
        print("\nConnectivity OK. No orders were placed by this script.")
    except ExecutionRejected as exc:
        print(f"  Could not reach Binance: {exc}")
    except Exception as exc:
        print(f"  Unexpected error while probing Binance: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Resolved after .env is loaded so TRADING_SYMBOL is visible, and before
    # any fetch, so a typo fails here rather than as an HTTP 400.
    try:
        args.symbol = resolve_symbol(args.symbol)
        args.market = resolve_market(args.market)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 1

    # Spot is the default everywhere, so a leveraged run has to be asked for
    # twice: once for the data, once for the money.
    if args.leverage and args.market != "futures":
        print("--leverage requires --market futures", file=sys.stderr)
        return 1

    try:
        frame = load_frame(args)
    except (VendorError, FileNotFoundError, ValueError) as exc:
        print(f"Could not load 1H data: {exc}", file=sys.stderr)
        return 1

    is_futures = args.market == "futures"
    config = BacktestConfig(
        initial_balance=args.balance,
        risk_pct=args.risk_pct,
        atr_stop_multiple=args.atr_stop,
        atr_target_multiple=args.atr_target,
        fee_bps=args.fee_bps,
        allow_short=not args.no_short,
        leverage=args.leverage if is_futures and args.leverage else 1.0,
        funding_bps_per_8h=args.funding_bps if is_futures else 0.0,
    )

    try:
        outcome = run_backtest(frame, TrendBiasModel(), config, symbol=args.symbol)
    except ValueError as exc:
        print(f"Backtest could not run: {exc}", file=sys.stderr)
        return 1

    print(render_report(outcome.metrics, outcome.trades, outcome.symbol,
                        outcome.start, outcome.end))
    print(f" Bars replayed: {outcome.bars}   |   Risk per trade: {config.risk_pct:.2f}% "
          f"of equity at {config.atr_stop_multiple}x ATR")
    print(" The bias source here is the deterministic trend model, not the agent graph;")
    print(" these figures validate the risk and execution machinery, not LLM alpha.\n")

    if args.yes:
        print("Approval auto-accepted via --yes.")
    elif not confirm():
        print("Not approved — exiting without connecting to Binance.")
        sys.exit(0)

    probe_binance(args.symbol, args.env, args.market, args.leverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
