"""Performance statistics for an hourly equity curve.

Every ratio here has a degenerate case that silently produces a flattering
number — a strategy that never loses has an infinite profit factor, a strategy
that took two trades has a meaningless Sharpe — and this is the module a human
reads before approving live order routing. So degenerate inputs return ``None``
and are rendered as ``n/a`` rather than as a large float.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tradingagents.scalping.engine import Trade

# Hourly bars in a 365-day year. Crypto trades continuously, so unlike equities
# there is no 252-session convention: every hour of the year is a sample.
HOURS_PER_YEAR = 8_760


@dataclass(frozen=True)
class PerformanceMetrics:
    """The scorecard for one backtest run."""

    initial_capital: float
    ending_capital: float
    total_return_pct: float
    buy_hold_return_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float | None
    profit_factor: float | None
    gross_profit: float
    gross_loss: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    exposure_pct: float


def _returns(equity_curve: list[float]) -> list[float]:
    """Bar-over-bar simple returns of an equity curve.

    A non-positive prior equity has no meaningful percentage change (and would
    divide by zero or flip the sign), so such steps are dropped instead of
    contributing a nonsense return.
    """
    out: list[float] = []
    # strict=False is deliberate: the second argument is the same list offset by
    # one, so it is always exactly one shorter.
    for previous, current in zip(equity_curve, equity_curve[1:], strict=False):
        if previous > 0:
            out.append(current / previous - 1.0)
    return out


def _stdev(values: list[float], mean: float) -> float:
    """Sample standard deviation (n-1), the convention Sharpe is quoted with."""
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def sharpe_ratio(returns: list[float], periods_per_year: int = HOURS_PER_YEAR) -> float | None:
    """Annualized Sharpe of hourly returns, or None when it is undefined.

    A zero standard deviation means the curve never moved — dividing by it
    would give infinity, which would read as a spectacular result.
    """
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    deviation = _stdev(returns, mean)
    if deviation == 0:
        return None
    return (mean / deviation) * math.sqrt(periods_per_year)


def sortino_ratio(returns: list[float], periods_per_year: int = HOURS_PER_YEAR) -> float | None:
    """Annualized Sortino: excess return per unit of *downside* deviation.

    The downside deviation is taken over all periods (zeros included for
    non-negative ones), not only the losing subset. Averaging over just the
    losers would shrink the denominator as the strategy improves and overstate
    the ratio. A curve with no losing bar leaves it undefined.
    """
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    downside = [min(value, 0.0) ** 2 for value in returns]
    deviation = math.sqrt(sum(downside) / len(returns))
    if deviation == 0:
        return None
    return (mean / deviation) * math.sqrt(periods_per_year)


def max_drawdown_pct(equity_curve: list[float]) -> float:
    """Deepest peak-to-trough fall of the curve, as a positive percentage."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst * 100.0


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[float],
    initial_capital: float,
    buy_hold_return_pct: float,
    bars_in_market: int = 0,
    total_bars: int = 0,
) -> PerformanceMetrics:
    """Score a finished run. ``trades`` must contain only closed positions."""
    closed = [trade for trade in trades if trade.is_closed]
    ending = equity_curve[-1] if equity_curve else initial_capital

    wins = [trade for trade in closed if trade.pnl > 0]
    losses = [trade for trade in closed if trade.pnl < 0]
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = abs(sum(trade.pnl for trade in losses))

    returns = _returns(equity_curve)
    return PerformanceMetrics(
        initial_capital=initial_capital,
        ending_capital=ending,
        total_return_pct=(ending / initial_capital - 1.0) * 100.0 if initial_capital else 0.0,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=len(closed),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(len(wins) / len(closed) * 100.0) if closed else None,
        # With no losing trade the ratio is infinite, which is a property of the
        # sample rather than of the strategy, so it is reported as undefined.
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else None,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        max_drawdown_pct=max_drawdown_pct(equity_curve),
        sharpe_ratio=sharpe_ratio(returns),
        sortino_ratio=sortino_ratio(returns),
        exposure_pct=(bars_in_market / total_bars * 100.0) if total_bars else 0.0,
    )


def _format(value: float | None, suffix: str = "", digits: int = 2) -> str:
    """Render an optional statistic, showing an undefined one as ``n/a``."""
    return "n/a" if value is None else f"{value:,.{digits}f}{suffix}"


def render_report(
    metrics: PerformanceMetrics,
    trades: list[Trade],
    symbol: str,
    start: str,
    end: str,
    recent: int = 5,
) -> str:
    """The terminal scorecard a human approves (or rejects) before going live."""
    closed = [trade for trade in trades if trade.is_closed]
    lines = [
        "",
        "=" * 78,
        f" 1H BACKTEST — {symbol}",
        f" Window: {start}  ->  {end}",
        "=" * 78,
        "",
        " CAPITAL",
        f"   Initial capital      : ${metrics.initial_capital:,.2f}",
        f"   Ending capital       : ${metrics.ending_capital:,.2f}",
        f"   Total net return     : {metrics.total_return_pct:+.2f}%",
        f"   Buy & hold benchmark : {metrics.buy_hold_return_pct:+.2f}%",
        f"   Excess vs buy & hold : "
        f"{metrics.total_return_pct - metrics.buy_hold_return_pct:+.2f}%",
        "",
        " TRADES",
        f"   Total trades         : {metrics.total_trades}"
        f"  (wins {metrics.wins} / losses {metrics.losses})",
        f"   Win rate             : {_format(metrics.win_rate_pct, '%')}",
        f"   Gross profit         : ${metrics.gross_profit:,.2f}",
        f"   Gross loss           : ${metrics.gross_loss:,.2f}",
        f"   Profit factor        : {_format(metrics.profit_factor)}",
        f"   Time in market       : {metrics.exposure_pct:.1f}% of bars",
        "",
        " RISK",
        f"   Max drawdown         : {metrics.max_drawdown_pct:.2f}%",
        f"   Sharpe (annualized)  : {_format(metrics.sharpe_ratio)}",
        f"   Sortino (annualized) : {_format(metrics.sortino_ratio)}",
        f"   (annualized over {HOURS_PER_YEAR:,} hourly periods)",
        "",
    ]

    lines.append(f" LAST {min(recent, len(closed))} TRADES")
    if not closed:
        lines.append("   (no trades were closed in this window)")
    else:
        header = (
            f"   {'Entry (UTC)':<24}{'Exit (UTC)':<24}{'Side':<6}"
            f"{'Entry':>12}{'Exit':>12}{'Qty':>13}{'PnL $':>12}{'PnL %':>9}  Reason"
        )
        lines.append(header)
        lines.append("   " + "-" * (len(header) - 3))
        for trade in closed[-recent:]:
            lines.append(
                f"   {trade.entry_time_text:<24}{trade.exit_time_text:<24}"
                f"{trade.side:<6}{trade.entry_price:>12,.2f}{trade.exit_price:>12,.2f}"
                f"{trade.quantity:>13,.6f}{trade.pnl:>+12,.2f}{trade.pnl_pct:>+8.2f}%"
                f"  {trade.exit_reason}"
            )
    lines.extend(["", "=" * 78, ""])
    return "\n".join(lines)
