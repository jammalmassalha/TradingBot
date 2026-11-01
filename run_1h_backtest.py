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
    python run_1h_backtest.py                       # 60 days of BTCUSDT 1H
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
    parser.add_argument("--symbol", default="BTCUSDT")
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

    print(f"Fetching {args.days} days of {args.symbol} 1H candles from Binance...")
    return load_1h_ohlcv(symbol=args.symbol, days=args.days)


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


def probe_binance(symbol: str, env: str | None) -> None:
    """Post-approval connectivity probe. Read-only: it places no orders."""
    try:
        from binance_executor import BinanceExecutor, ExecutionRejected
    except ImportError as exc:
        print(f"Execution bridge unavailable: {exc}")
        return

    try:
        executor = BinanceExecutor(symbol=symbol, env=env)
    except ValueError as exc:
        print(f"Executor configuration error: {exc}")
        return

    print(f"\nConnecting to Binance ({executor.env})...")
    if executor.is_live:
        print("  *** LIVE ENVIRONMENT — orders would use real funds ***")
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

    try:
        frame = load_frame(args)
    except (VendorError, FileNotFoundError, ValueError) as exc:
        print(f"Could not load 1H data: {exc}", file=sys.stderr)
        return 1

    config = BacktestConfig(
        initial_balance=args.balance,
        risk_pct=args.risk_pct,
        atr_stop_multiple=args.atr_stop,
        atr_target_multiple=args.atr_target,
        fee_bps=args.fee_bps,
        allow_short=not args.no_short,
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

    probe_binance(args.symbol, args.env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
