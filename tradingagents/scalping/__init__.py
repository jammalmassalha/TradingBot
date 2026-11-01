"""The two-tier 1H engine: a strategic bias layer and a tactical execution layer.

Separated from ``tradingagents.backtest`` on purpose. That module scores
*decision quality* and its docstring forbids growing a portfolio simulator
inside it — no quantities, no fills, no cash ledger. This package is the
opposite: it is exactly the position/fill/ledger machinery, driven by a
directional bias rather than producing one. Keeping them apart means the
evaluation tool stays an evaluation tool.
"""

from tradingagents.scalping.engine import (
    BacktestConfig,
    BacktestOutcome,
    Trade,
    run_backtest,
)
from tradingagents.scalping.metrics import PerformanceMetrics, compute_metrics
from tradingagents.scalping.signals import (
    Bias,
    StrategicBias,
    TrendBiasModel,
)

__all__ = [
    "BacktestConfig",
    "BacktestOutcome",
    "Bias",
    "PerformanceMetrics",
    "StrategicBias",
    "Trade",
    "TrendBiasModel",
    "compute_metrics",
    "run_backtest",
]
