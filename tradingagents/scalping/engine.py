"""The tactical layer: turn an hourly bias into sized, bracketed, settled trades.

This is a bar-close simulator, not a tick simulator, and the ordering below is
what keeps it honest:

  1. A bar's high/low is first tested against the stop and target of a position
     that was *already* open. Only a position opened on an earlier bar can be
     exited by this bar's range.
  2. The bias for the bar is computed from that bar's close, which is known
     once the bar has closed.
  3. A new entry fills at that same close.

Doing (2) or (3) before (1) would let a position be opened and stopped out
inside the bar that created it, which is look-ahead: the engine would be using
the bar's extremes to trade a signal derived from its close.

Where a bar touches both the stop and the target, the stop is assumed to have
been hit first. Hourly bars hide their internal path, and resolving the
ambiguity in the strategy's favor is the single easiest way to manufacture a
backtest that cannot be reproduced live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tradingagents.dataflows.vendors.binance_ohlcv import DEFAULT_SYMBOL, format_hour
from tradingagents.scalping.metrics import PerformanceMetrics, compute_metrics
from tradingagents.scalping.signals import Bias, BiasModel, StrategicBias

logger = logging.getLogger(__name__)

# Binance spot taker fee, in basis points of notional, charged on both sides.
DEFAULT_FEE_BPS = 10.0


@dataclass
class BacktestConfig:
    """Account rules for a run. The defaults are the mission's risk mandate."""

    initial_balance: float = 10_000.0
    # Fraction of equity risked on the distance to the stop, not of notional.
    risk_pct: float = 2.0
    atr_stop_multiple: float = 1.5
    atr_target_multiple: float = 3.0
    fee_bps: float = DEFAULT_FEE_BPS
    allow_short: bool = True
    # Ceiling on position notional as a percentage of equity. 100 means the
    # simulator never borrows: a low-ATR bar can otherwise imply a position many
    # times the account, which no spot venue would fill.
    max_notional_pct: float = 100.0

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {self.initial_balance}")
        if not 0 < self.risk_pct <= 100:
            raise ValueError(f"risk_pct must be in (0, 100], got {self.risk_pct}")
        if self.atr_stop_multiple <= 0:
            raise ValueError("atr_stop_multiple must be positive")
        if self.atr_target_multiple <= 0:
            raise ValueError("atr_target_multiple must be positive")
        if self.fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")


@dataclass
class Trade:
    """One round trip. Open while ``exit_price`` is None."""

    side: str                      # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    fees: float = 0.0

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity

    @property
    def pnl(self) -> float:
        """Realized PnL net of both legs' fees; 0 while the trade is open."""
        if not self.is_closed:
            return 0.0
        gross = (self.exit_price - self.entry_price) * self.direction * self.quantity
        return gross - self.fees

    @property
    def pnl_pct(self) -> float:
        """Net PnL as a percentage of the capital committed to the position."""
        return (self.pnl / self.notional * 100.0) if self.notional else 0.0

    def unrealized(self, price: float) -> float:
        """Mark-to-market PnL at ``price``, with the entry fee already charged.

        Fees accrue onto ``fees`` as each leg is charged, so this and ``pnl``
        share one definition of cost and the equity curve does not jump when an
        open position settles.
        """
        if self.is_closed:
            return 0.0
        return (price - self.entry_price) * self.direction * self.quantity - self.fees

    @property
    def entry_time_text(self) -> str:
        return format_hour(self.entry_time)

    @property
    def exit_time_text(self) -> str:
        return format_hour(self.exit_time) if self.exit_time is not None else "—"


@dataclass
class BacktestOutcome:
    """Everything a run produced, ready to be scored and rendered."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: PerformanceMetrics | None = None
    symbol: str = ""
    start: str = ""
    end: str = ""
    bars: int = 0


def position_size(
    equity: float,
    atr: float,
    price: float,
    config: BacktestConfig,
) -> float:
    """Units to trade so that a stop-out costs ``risk_pct`` of equity.

    The stop distance is the risk per unit, so ``risk budget / stop distance``
    is the size that makes a stop-out cost exactly the budget. The result is
    then capped by ``max_notional_pct``, because at low volatility the risk
    formula alone can ask for a position larger than the account.
    """
    if equity <= 0 or atr <= 0 or price <= 0:
        return 0.0
    stop_distance = atr * config.atr_stop_multiple
    if stop_distance <= 0:
        return 0.0
    quantity = (equity * config.risk_pct / 100.0) / stop_distance
    max_notional = equity * config.max_notional_pct / 100.0
    return min(quantity, max_notional / price)


def bracket_levels(
    entry_price: float,
    atr: float,
    side: str,
    config: BacktestConfig,
) -> tuple[float, float]:
    """The (stop, target) prices for a new position, in ATR units."""
    stop_distance = atr * config.atr_stop_multiple
    target_distance = atr * config.atr_target_multiple
    if side == "LONG":
        return entry_price - stop_distance, entry_price + target_distance
    return entry_price + stop_distance, entry_price - target_distance


def _fee(notional: float, config: BacktestConfig) -> float:
    return abs(notional) * config.fee_bps / 10_000.0


def _exit_price_for_bar(trade: Trade, bar: pd.Series) -> tuple[float, str] | None:
    """Where this bar closes ``trade``, if it does.

    The stop is tested before the target: an hourly bar does not record the
    order in which its extremes occurred, and assuming the favorable one would
    inflate every result that depends on it.
    """
    high, low = float(bar["High"]), float(bar["Low"])
    if trade.side == "LONG":
        if low <= trade.stop_price:
            return trade.stop_price, "stop"
        if high >= trade.target_price:
            return trade.target_price, "target"
        return None
    if high >= trade.stop_price:
        return trade.stop_price, "stop"
    if low <= trade.target_price:
        return trade.target_price, "target"
    return None


def _first_tradable_index(frame: pd.DataFrame) -> int:
    """The first bar the engine may trade from.

    When the frame carries ATR, trading starts where it warms up, since no
    position can be sized before then. A frame without the column is allowed:
    a bias model may supply its own volatility reading (the live agent layer
    does), and in that case the model — not this function — owns the decision
    to stay flat while it has nothing to size with.
    """
    if "ATR14" not in frame.columns:
        return 0
    warmed = frame["ATR14"].notna().to_numpy().nonzero()[0]
    if warmed.size == 0:
        raise ValueError("ATR14 never warms up in this frame; the window is too short")
    return int(warmed[0])


def run_backtest(
    frame: pd.DataFrame,
    bias_model: BiasModel,
    config: BacktestConfig | None = None,
    symbol: str = DEFAULT_SYMBOL,
) -> BacktestOutcome:
    """Replay ``frame`` bar by bar under ``bias_model`` and the account rules."""
    config = config or BacktestConfig()
    if frame.empty:
        raise ValueError("cannot backtest an empty frame")

    start_index = _first_tradable_index(frame)
    # ``equity`` is the whole account: a position's cost basis is never removed
    # from it, and the position contributes only its mark-to-market PnL. The
    # alternative (debiting the notional on entry and crediting it back on exit)
    # double-counts unless every leg is mirrored exactly, and gives a meaningless
    # ledger for shorts, which receive cash on entry.
    equity = config.initial_balance
    open_trade: Trade | None = None
    trades: list[Trade] = []
    equity_curve: list[float] = []
    bars_in_market = 0

    for index in range(start_index, len(frame)):
        bar = frame.iloc[index]
        close = float(bar["Close"])

        # (1) Resolve an existing position against this bar's range first.
        if open_trade is not None:
            hit = _exit_price_for_bar(open_trade, bar)
            if hit is not None:
                price, reason = hit
                equity += _close_trade(open_trade, bar["Date"], price, reason, config)
                trades.append(open_trade)
                open_trade = None

        # (2) The bias for this bar, known only at its close.
        bias = _safe_bias(bias_model, frame, index)

        # (3) An opposite or flat call closes the position at the close.
        if open_trade is not None and bias.bias.direction != open_trade.direction:
            equity += _close_trade(open_trade, bar["Date"], close, "signal flip", config)
            trades.append(open_trade)
            open_trade = None

        # (4) A directional call with no position opens one at the close.
        if open_trade is None and bias.is_directional and bias.atr > 0:
            side = "LONG" if bias.bias is Bias.LONG else "SHORT"
            if side == "LONG" or config.allow_short:
                open_trade = _open_trade(equity, close, bias, side, bar["Date"], config)

        if open_trade is not None:
            bars_in_market += 1
        marked = equity + (open_trade.unrealized(close) if open_trade else 0.0)
        equity_curve.append(max(marked, 0.0))

    # A position still open at the end is closed at the last close, so its PnL
    # is realized and counted rather than silently dropped from the scorecard.
    if open_trade is not None:
        last = frame.iloc[-1]
        equity += _close_trade(open_trade, last["Date"], float(last["Close"]),
                               "end of data", config)
        trades.append(open_trade)
        if equity_curve:
            equity_curve[-1] = max(equity, 0.0)

    traded = frame.iloc[start_index:]
    first_close, last_close = float(traded["Close"].iloc[0]), float(traded["Close"].iloc[-1])
    buy_hold_pct = (last_close / first_close - 1.0) * 100.0 if first_close else 0.0

    outcome = BacktestOutcome(
        trades=trades,
        equity_curve=equity_curve,
        symbol=symbol,
        start=format_hour(traded["Date"].iloc[0]),
        end=format_hour(traded["Date"].iloc[-1]),
        bars=len(traded),
    )
    outcome.metrics = compute_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=config.initial_balance,
        buy_hold_return_pct=buy_hold_pct,
        bars_in_market=bars_in_market,
        total_bars=len(traded),
    )
    return outcome


def _safe_bias(bias_model: BiasModel, frame: pd.DataFrame, index: int) -> StrategicBias:
    """The model's call for a bar, degrading to FLAT if the model raises.

    A strategic layer that throws (a failed LLM call, a malformed response) must
    not abort a 1440-bar sweep, and must not be read as a direction either.
    """
    try:
        return bias_model.bias_for(frame, index)
    except Exception as exc:
        row = frame.iloc[index]
        logger.warning("Bias model failed at bar %d (%s): %s", index, row.get("Date"), exc)
        return StrategicBias(Bias.FLAT, row["Date"], float(row["Close"]), 0.0,
                             rationale=f"bias model error: {exc}")


def _open_trade(
    cash: float,
    price: float,
    bias: StrategicBias,
    side: str,
    when: pd.Timestamp,
    config: BacktestConfig,
) -> Trade | None:
    """Size and bracket a new position, or None when it would be too small."""
    quantity = position_size(cash, bias.atr, price, config)
    if quantity <= 0:
        return None
    stop, target = bracket_levels(price, bias.atr, side, config)
    # A stop at or through the entry would be filled instantly; such a position
    # has no defined risk, so it is not opened.
    if (side == "LONG" and stop >= price) or (side == "SHORT" and stop <= price):
        return None
    return Trade(
        side=side,
        entry_time=when,
        entry_price=price,
        quantity=quantity,
        stop_price=stop,
        target_price=target,
        fees=_fee(price * quantity, config),
    )


def _close_trade(
    trade: Trade,
    when: pd.Timestamp,
    price: float,
    reason: str,
    config: BacktestConfig,
) -> float:
    """Settle ``trade`` and return the net PnL to credit to equity.

    Only PnL moves the ledger, because the cost basis was never debited from it
    (see ``run_backtest``).
    """
    trade.exit_time = when
    trade.exit_price = price
    trade.exit_reason = reason
    trade.fees += _fee(price * trade.quantity, config)
    return trade.pnl
