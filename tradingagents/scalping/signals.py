"""The strategic layer: what an hourly regime call looks like, and how to make one.

The hierarchy exists because of latency and cost. A full multi-agent
deliberation takes 15-45 seconds and many thousands of tokens, which is
tolerable once an hour and ruinous once a minute. So the LLM never sees a
tick: it emits a ``StrategicBias`` for the hour, and the tactical layer in
``engine.py`` trades inside that bias with arithmetic only.

Two producers implement the same contract:

  * ``TrendBiasModel`` — deterministic, from the 1H indicators. This is what a
    backtest runs on. Replaying 1440 hourly bars through the agent graph would
    cost thousands of LLM calls and still not be reproducible run-to-run, so a
    sweep that is meant to validate *risk mechanics* uses the arithmetic proxy
    and leaves the agents out of the loop.
  * ``AgentBiasModel`` — the real multi-agent graph, used live. One call per
    closed hourly candle.

Because both return the same object, the tactical layer and the executor
cannot tell which one produced a bias, and the backtested risk machinery is
literally the code that runs live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd

from tradingagents.dataflows.vendors.binance_ohlcv import DEFAULT_SYMBOL

logger = logging.getLogger(__name__)


class Bias(StrEnum):
    """The regime call for an hour. ``FLAT`` means hold no directional risk."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @property
    def direction(self) -> int:
        """+1 long, -1 short, 0 flat — the sign used in PnL arithmetic."""
        return {Bias.LONG: 1, Bias.SHORT: -1, Bias.FLAT: 0}[self]


# How a trading decision maps onto a bias, so an LLM's BUY/SELL/HOLD verdict and
# the arithmetic proxy speak the same language.
_DECISION_TO_BIAS = {
    "BUY": Bias.LONG,
    "LONG": Bias.LONG,
    "SELL": Bias.SHORT,
    "SHORT": Bias.SHORT,
    "HOLD": Bias.FLAT,
    "FLAT": Bias.FLAT,
}


def bias_from_decision(decision: str) -> Bias:
    """Map a free-text agent verdict onto a ``Bias``, defaulting to ``FLAT``.

    Unrecognized text becomes ``FLAT`` rather than raising: an unparseable model
    output must not be read as a direction, and refusing to trade is the safe
    reading of "I could not understand the decision".
    """
    if not isinstance(decision, str):
        return Bias.FLAT
    token = decision.strip().upper()
    for key, bias in _DECISION_TO_BIAS.items():
        if token == key or token.startswith(f"{key} ") or f"**{key}**" in token:
            return bias
    for key, bias in _DECISION_TO_BIAS.items():
        if key in token:
            return bias
    logger.warning("Could not read a direction from decision %r; treating as FLAT", decision)
    return Bias.FLAT


@dataclass(frozen=True)
class StrategicBias:
    """One hour's regime call, plus the levels the tactical layer trades against.

    ``atr`` is carried on the bias rather than re-derived downstream so that the
    stop distance, the position size and the take-profit all come from the same
    volatility reading the decision was made under. Recomputing it at execution
    time would silently resize a position when the next bar's ATR differs.
    """

    bias: Bias
    as_of: datetime
    close: float
    atr: float
    invalidation: float | None = None
    support: float | None = None
    resistance: float | None = None
    rationale: str = ""

    @property
    def is_directional(self) -> bool:
        return self.bias is not Bias.FLAT

    def volatility_band(self, multiple: float) -> float:
        """``multiple`` ATRs in price terms — the stop and target distance unit."""
        return self.atr * multiple


class BiasModel(Protocol):
    """Anything that can turn the bar history up to ``index`` into a bias."""

    def bias_for(self, frame: pd.DataFrame, index: int) -> StrategicBias:
        ...


# A reading above this is overbought and below its mirror is oversold; the proxy
# refuses to open a fresh trend position into an exhausted move.
RSI_OVERBOUGHT = 72.0
RSI_OVERSOLD = 28.0


@dataclass
class TrendBiasModel:
    """Deterministic stand-in for the agents, used to backtest the risk layer.

    The rule is a plain 1H trend filter: price and the fast EMA on the same side
    of the slow EMA, MACD confirming, and RSI not already exhausted. It is not
    claimed to be a good alpha model — its job is to produce a realistic stream
    of directional calls so the sizing, stops and metrics can be exercised over
    real price history.
    """

    require_macd: bool = True
    use_rsi_filter: bool = True

    def bias_for(self, frame: pd.DataFrame, index: int) -> StrategicBias:
        row = frame.iloc[index]
        atr = _finite(row.get("ATR14"))
        close = _finite(row.get("Close"))
        ema_fast = _finite(row.get("EMA20"))
        ema_mid = _finite(row.get("EMA50"))
        ema_slow = _finite(row.get("EMA200"))

        # Without a warmed-up ATR there is no stop distance, so there is no
        # position size either; refusing to trade is the only safe answer.
        if atr is None or atr <= 0 or close is None:
            return StrategicBias(Bias.FLAT, row["Date"], close or 0.0, atr or 0.0,
                                 rationale="indicators not warmed up")
        if ema_fast is None or ema_mid is None or ema_slow is None:
            return StrategicBias(Bias.FLAT, row["Date"], close, atr,
                                 rationale="EMA set not warmed up")

        macd_hist = _finite(row.get("MACDHist")) or 0.0
        rsi = _finite(row.get("RSI14"))

        long_trend = close > ema_fast > ema_mid and close > ema_slow
        short_trend = close < ema_fast < ema_mid and close < ema_slow

        if self.require_macd:
            long_trend = long_trend and macd_hist > 0
            short_trend = short_trend and macd_hist < 0
        if self.use_rsi_filter and rsi is not None:
            long_trend = long_trend and rsi < RSI_OVERBOUGHT
            short_trend = short_trend and rsi > RSI_OVERSOLD

        if long_trend:
            bias, rationale = Bias.LONG, "1H uptrend: close>EMA20>EMA50, above EMA200"
        elif short_trend:
            bias, rationale = Bias.SHORT, "1H downtrend: close<EMA20<EMA50, below EMA200"
        else:
            bias, rationale = Bias.FLAT, "no aligned 1H trend"

        # The invalidation level is where the directional thesis is simply wrong,
        # quoted in the same ATR units the tactical layer sizes with.
        invalidation = None
        if bias is Bias.LONG:
            invalidation = min(ema_mid, close - atr)
        elif bias is Bias.SHORT:
            invalidation = max(ema_mid, close + atr)

        return StrategicBias(
            bias=bias,
            as_of=row["Date"],
            close=close,
            atr=atr,
            invalidation=invalidation,
            support=ema_mid if bias is Bias.LONG else None,
            resistance=ema_mid if bias is Bias.SHORT else None,
            rationale=rationale,
        )


@dataclass
class AgentBiasModel:
    """The live strategic layer: one multi-agent graph run per closed 1H candle.

    Deliberately NOT used by the backtest. It is the adapter the live loop uses
    to turn the graph's verdict into the same ``StrategicBias`` the tactical
    layer already understands.
    """

    graph: Any
    symbol: str = DEFAULT_SYMBOL
    asset_type: str = "crypto"

    def bias_for(self, frame: pd.DataFrame, index: int) -> StrategicBias:
        row = frame.iloc[index]
        atr = _finite(row.get("ATR14")) or 0.0
        close = _finite(row.get("Close")) or 0.0
        # The graph keys a run by date; an hourly loop passes the bar's own
        # hourly boundary so successive hours are distinct runs.
        stamp = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
        try:
            _, decision = self.graph.propagate(self.symbol, stamp, self.asset_type)
        except Exception as exc:
            # A failed deliberation must not be read as a direction. The hour is
            # skipped flat rather than inheriting the previous hour's bias.
            logger.error("Strategic layer failed for %s at %s: %s", self.symbol, stamp, exc)
            return StrategicBias(Bias.FLAT, row["Date"], close, atr,
                                 rationale=f"agent run failed: {exc}")

        bias = bias_from_decision(str(decision))
        invalidation = None
        if atr > 0:
            if bias is Bias.LONG:
                invalidation = close - atr
            elif bias is Bias.SHORT:
                invalidation = close + atr
        return StrategicBias(
            bias=bias,
            as_of=row["Date"],
            close=close,
            atr=atr,
            invalidation=invalidation,
            rationale=str(decision)[:500],
        )


def _finite(value: Any) -> float | None:
    """``value`` as a float, or None when it is missing or not finite.

    Indicator columns hold NaN until they warm up, and NaN silently poisons
    every comparison it touches (``NaN > x`` is False), so it is converted to an
    explicit None the callers must handle.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) or number in (float("inf"), float("-inf")) else number
