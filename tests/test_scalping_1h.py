"""Tests for the 1H scalping pipeline: indicators, risk engine and order guardrails.

These cover the arithmetic a human is asked to approve in ``run_1h_backtest.py``
and the limits that stand between a wrong model output and a real order. They
are all offline: no test here touches Binance or an LLM.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.binance_ohlcv import (
    ATR_PERIOD,
    compute_indicators,
    floor_to_hour,
    format_hour,
)
from tradingagents.scalping.engine import (
    BacktestConfig,
    Trade,
    bracket_levels,
    position_size,
    run_backtest,
)
from tradingagents.scalping.metrics import (
    compute_metrics,
    max_drawdown_pct,
    sharpe_ratio,
    sortino_ratio,
)
from tradingagents.scalping.signals import Bias, StrategicBias, bias_from_decision

pytestmark = pytest.mark.unit


def make_frame(closes, highs=None, lows=None, start="2025-01-01"):
    """An hourly OHLCV frame from a list of closes."""
    index = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Date": index,
            "Open": closes,
            "High": highs if highs is not None else [c + 1 for c in closes],
            "Low": lows if lows is not None else [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1.0] * len(closes),
        }
    )


class FixedBias:
    """A bias model that returns the same call on every bar."""

    def __init__(self, bias: Bias, atr: float = 10.0):
        self.bias, self.atr = bias, atr

    def bias_for(self, frame, index):
        row = frame.iloc[index]
        return StrategicBias(self.bias, row["Date"], float(row["Close"]), self.atr)


class ScriptedBias:
    """A bias model driven by a per-bar list, for testing transitions."""

    def __init__(self, biases, atr: float = 10.0):
        self.biases, self.atr = biases, atr

    def bias_for(self, frame, index):
        row = frame.iloc[index]
        bias = self.biases[index] if index < len(self.biases) else Bias.FLAT
        return StrategicBias(bias, row["Date"], float(row["Close"]), self.atr)


# --- indicators -------------------------------------------------------------


def test_emas_match_pandas_reference():
    closes = [100 + i * 0.5 for i in range(260)]
    data = compute_indicators(make_frame(closes))
    for period in (20, 50, 200):
        expected = data["Close"].ewm(span=period, adjust=False, min_periods=period).mean()
        assert (data[f"EMA{period}"] - expected).abs().max() == pytest.approx(0.0, abs=1e-12)


def test_atr_on_constant_range_equals_that_range():
    """With a fixed high-low spread and no gaps, ATR converges to the spread."""
    closes = [100.0] * 60
    data = compute_indicators(make_frame(closes, highs=[101.0] * 60, lows=[99.0] * 60))
    assert data["ATR14"].iloc[-1] == pytest.approx(2.0, abs=1e-9)
    # Before the period fills there is no value at all, rather than a partial one.
    assert data["ATR14"].iloc[: ATR_PERIOD - 1].isna().all()


def test_rsi_is_100_when_every_bar_rises():
    closes = [100 + i for i in range(40)]
    data = compute_indicators(make_frame(closes))
    assert data["RSI14"].iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_is_zero_when_every_bar_falls():
    closes = [200 - i for i in range(40)]
    data = compute_indicators(make_frame(closes))
    assert data["RSI14"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_macd_is_fast_minus_slow_ema():
    closes = [100 + (i % 7) for i in range(80)]
    data = compute_indicators(make_frame(closes))
    fast = data["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    slow = data["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    assert (data["MACD"] - (fast - slow)).abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (data["MACDHist"] - (data["MACD"] - data["MACDSignal"])).abs().max() < 1e-12


def test_hour_formatting_is_utc_and_floored():
    moment = pd.Timestamp("2025-03-04 17:42:09", tz="UTC").to_pydatetime()
    assert format_hour(moment) == "2025-03-04 17:00:00 UTC"
    assert floor_to_hour(moment).minute == 0


# --- position sizing --------------------------------------------------------


def test_position_size_risks_exactly_the_configured_percent():
    """A stop-out must cost risk_pct of equity — the core of the risk mandate.

    The ATR here is wide enough that the risk formula, not the no-leverage
    notional cap, is the binding constraint (the cap is covered separately).
    """
    config = BacktestConfig(initial_balance=10_000, risk_pct=2.0, atr_stop_multiple=1.5)
    atr, price = 1_000.0, 50_000.0
    quantity = position_size(10_000, atr, price, config)
    assert quantity * price <= 10_000, "the cap must not be what binds in this case"
    loss_at_stop = quantity * atr * config.atr_stop_multiple
    assert loss_at_stop == pytest.approx(200.0)


def test_position_size_is_capped_at_no_leverage():
    """Low volatility must not imply a position larger than the account."""
    config = BacktestConfig(initial_balance=10_000, risk_pct=2.0, atr_stop_multiple=1.5)
    quantity = position_size(10_000, atr=0.01, price=100.0, config=config)
    assert quantity * 100.0 <= 10_000 + 1e-9


def test_position_size_refuses_degenerate_inputs():
    config = BacktestConfig()
    assert position_size(0, 10, 100, config) == 0.0
    assert position_size(10_000, 0, 100, config) == 0.0
    assert position_size(10_000, 10, 0, config) == 0.0


def test_bracket_levels_are_symmetric_in_atr_units():
    config = BacktestConfig(atr_stop_multiple=1.5, atr_target_multiple=3.0)
    stop, target = bracket_levels(100.0, atr=10.0, side="LONG", config=config)
    assert (stop, target) == (85.0, 130.0)
    stop, target = bracket_levels(100.0, atr=10.0, side="SHORT", config=config)
    assert (stop, target) == (115.0, 70.0)


# --- engine semantics -------------------------------------------------------


def test_long_target_fill_produces_the_exact_expected_pnl():
    """Hand-computed round trip, fees included."""
    # Flat at 100 until the last bar, which spikes through the 3x ATR target.
    closes = [100.0] * 30 + [100.0]
    frame = make_frame(closes, highs=[100.5] * 30 + [200.0], lows=[99.5] * 31)
    config = BacktestConfig(initial_balance=10_000, risk_pct=2.0, atr_stop_multiple=1.5,
                            atr_target_multiple=3.0, fee_bps=10.0)
    outcome = run_backtest(frame, FixedBias(Bias.LONG, atr=10.0), config)

    trade = outcome.trades[0]
    assert trade.side == "LONG"
    # quantity = (10000 * 2%) / (10 * 1.5) = 13.333..., capped by notional:
    # 13.333 * 100 = 1333 <= 10000, so the risk formula governs.
    assert trade.quantity == pytest.approx(200 / 15)
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(130.0)   # 100 + 3 * 10
    assert trade.exit_reason == "target"

    gross = (130.0 - 100.0) * trade.quantity
    fees = (100.0 * trade.quantity + 130.0 * trade.quantity) * 0.001
    assert trade.pnl == pytest.approx(gross - fees)
    # The bias is still LONG after the target fills, so the engine re-enters at
    # that bar's close and the re-entry settles at end-of-data for a fee-only
    # loss. Ending capital is therefore the sum over every closed trade.
    assert outcome.metrics.ending_capital == pytest.approx(
        10_000 + sum(t.pnl for t in outcome.trades)
    )


def test_stop_is_assumed_hit_before_target_in_the_same_bar():
    """An hourly bar hides its path, so the unfavorable fill must be assumed."""
    closes = [100.0] * 30 + [100.0]
    # The final bar spans both the 85 stop and the 130 target.
    frame = make_frame(closes, highs=[100.5] * 30 + [200.0], lows=[99.5] * 30 + [50.0])
    outcome = run_backtest(frame, FixedBias(Bias.LONG, atr=10.0), BacktestConfig())
    assert outcome.trades[0].exit_reason == "stop"
    assert outcome.trades[0].pnl < 0


def test_a_position_is_never_opened_and_closed_in_the_same_bar():
    """Entry fills at a bar's close, so that bar's range cannot exit it."""
    closes = [100.0] * 20
    # Every bar has a huge range that would hit any stop, yet the entry bar
    # must survive, because the entry happens at its close.
    frame = make_frame(closes, highs=[500.0] * 20, lows=[1.0] * 20)
    outcome = run_backtest(frame, FixedBias(Bias.LONG, atr=10.0), BacktestConfig())
    first = outcome.trades[0]
    assert first.exit_time > first.entry_time


def test_opposite_signal_closes_the_position_at_the_close():
    closes = [100.0] * 40
    frame = make_frame(closes, highs=[100.2] * 40, lows=[99.8] * 40)
    biases = [Bias.LONG] * 20 + [Bias.SHORT] * 20
    outcome = run_backtest(frame, ScriptedBias(biases, atr=10.0), BacktestConfig())
    flips = [t for t in outcome.trades if t.exit_reason == "signal flip"]
    assert flips, "an opposite bias must close the open position"
    assert flips[0].exit_price == pytest.approx(100.0)


def test_flat_signal_closes_and_does_not_reopen():
    closes = [100.0] * 30
    frame = make_frame(closes, highs=[100.2] * 30, lows=[99.8] * 30)
    biases = [Bias.LONG] * 10 + [Bias.FLAT] * 20
    outcome = run_backtest(frame, ScriptedBias(biases, atr=10.0), BacktestConfig())
    assert len(outcome.trades) == 1
    assert outcome.trades[0].exit_reason == "signal flip"


def test_open_position_is_settled_at_the_end_of_data():
    closes = [100.0] * 25
    frame = make_frame(closes, highs=[100.2] * 25, lows=[99.8] * 25)
    outcome = run_backtest(frame, FixedBias(Bias.LONG, atr=10.0), BacktestConfig())
    assert all(t.is_closed for t in outcome.trades)
    assert outcome.trades[-1].exit_reason == "end of data"


def test_short_is_suppressed_when_shorting_is_disabled():
    closes = [100.0] * 25
    frame = make_frame(closes, highs=[100.2] * 25, lows=[99.8] * 25)
    outcome = run_backtest(frame, FixedBias(Bias.SHORT, atr=10.0),
                           BacktestConfig(allow_short=False))
    assert outcome.trades == []


def test_a_failing_bias_model_is_treated_as_flat_not_fatal():
    class Broken:
        def bias_for(self, frame, index):
            raise RuntimeError("LLM exploded")

    closes = [100.0] * 25
    outcome = run_backtest(make_frame(closes), Broken(), BacktestConfig())
    assert outcome.trades == []
    assert outcome.metrics.ending_capital == pytest.approx(10_000.0)


def test_equity_curve_has_one_point_per_replayed_bar():
    closes = [100.0 + i * 0.1 for i in range(60)]
    outcome = run_backtest(make_frame(closes), FixedBias(Bias.LONG, atr=5.0), BacktestConfig())
    assert len(outcome.equity_curve) == outcome.bars


def test_backtest_rejects_an_unwarmed_window():
    with pytest.raises(ValueError, match="too short"):
        run_backtest(compute_indicators(make_frame([100.0] * 5)),
                     FixedBias(Bias.LONG), BacktestConfig())


@pytest.mark.parametrize("field,value", [
    ("initial_balance", 0), ("risk_pct", 0), ("risk_pct", 101),
    ("atr_stop_multiple", 0), ("atr_target_multiple", -1), ("fee_bps", -1),
])
def test_backtest_config_rejects_invalid_settings(field, value):
    with pytest.raises(ValueError):
        BacktestConfig(**{field: value})


# --- metrics ----------------------------------------------------------------


def test_max_drawdown_measures_peak_to_trough():
    assert max_drawdown_pct([100, 120, 60, 80]) == pytest.approx(50.0)
    assert max_drawdown_pct([100, 110, 120]) == pytest.approx(0.0)


def test_ratios_are_undefined_rather_than_infinite():
    """A flat curve has no volatility; reporting a huge Sharpe would mislead."""
    assert sharpe_ratio([0.0, 0.0, 0.0]) is None
    assert sortino_ratio([0.01, 0.02, 0.03]) is None    # no losing period
    assert sharpe_ratio([0.01]) is None                 # too few samples


def test_sharpe_is_annualized_over_8760_hours():
    returns = [0.01, -0.01] * 50
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    expected = (mean / variance**0.5) * (8_760 ** 0.5)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_profit_factor_is_undefined_with_no_losing_trade():
    winner = Trade("LONG", pd.Timestamp("2025-01-01", tz="UTC"), 100.0, 1.0, 90.0, 130.0)
    winner.exit_time, winner.exit_price, winner.exit_reason = (
        pd.Timestamp("2025-01-02", tz="UTC"), 130.0, "target"
    )
    metrics = compute_metrics([winner], [10_000, 10_030], 10_000, 1.0)
    assert metrics.profit_factor is None
    assert metrics.win_rate_pct == pytest.approx(100.0)


def test_metrics_on_a_run_with_no_trades():
    metrics = compute_metrics([], [10_000, 10_000], 10_000, 0.0)
    assert metrics.total_trades == 0
    assert metrics.win_rate_pct is None
    assert metrics.total_return_pct == pytest.approx(0.0)


# --- bias parsing -----------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("BUY", Bias.LONG), ("**BUY**", Bias.LONG), ("SELL", Bias.SHORT),
    ("HOLD", Bias.FLAT), ("buy the dip", Bias.LONG),
    ("", Bias.FLAT), ("unintelligible", Bias.FLAT), (None, Bias.FLAT),
])
def test_bias_from_decision(text, expected):
    assert bias_from_decision(text) is expected


# --- vendor fetch path ------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def kline_rows(start_ms, count, step=3_600_000, price=100.0):
    """Binance-shaped kline arrays: [openTime, O, H, L, C, V, ...]."""
    return [
        [start_ms + i * step, str(price), str(price + 1), str(price - 1),
         str(price), "1.0", start_ms + (i + 1) * step - 1]
        for i in range(count)
    ]


def test_fetch_klines_paginates_past_the_1000_row_cap(monkeypatch):
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor

    pages = []

    def fake_get(url, params=None, timeout=None):
        start = int(params["startTime"])
        # First call returns a full page (so pagination continues), then a short one.
        count = vendor.MAX_KLINES_PER_REQUEST if not pages else 20
        pages.append(start)
        return FakeResponse(kline_rows(start, count))

    monkeypatch.setattr(vendor.requests, "get", fake_get)
    monkeypatch.setattr(vendor.time, "sleep", lambda _: None)
    frame = vendor.fetch_klines("BTCUSDT", days=60)

    assert len(pages) >= 2, "a long window must be paginated"
    assert list(frame.columns) == vendor.OHLCV_COLUMNS
    # Overlapping page boundaries must not produce duplicate bars.
    assert frame["Date"].is_unique
    assert frame["Date"].is_monotonic_increasing


def test_fetch_klines_excludes_the_still_forming_candle(monkeypatch):
    """The in-progress bar has a moving close and must never be tradable."""
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor

    now = vendor.floor_to_hour()

    def fake_get(url, params=None, timeout=None):
        start = int(params["startTime"])
        # Deliberately return bars running past the current hour boundary.
        return FakeResponse(kline_rows(start, 50))

    monkeypatch.setattr(vendor.requests, "get", fake_get)
    monkeypatch.setattr(vendor.time, "sleep", lambda _: None)
    frame = vendor.fetch_klines("BTCUSDT", days=1, end=now)

    newest = pd.Timestamp(frame["Date"].iloc[-1]).to_pydatetime()
    assert newest <= now - pd.Timedelta(hours=1).to_pytimedelta()


@pytest.mark.parametrize("status", [429, 418, 500, 503])
def test_transient_http_failures_map_to_vendor_unavailable(monkeypatch, status):
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor
    from tradingagents.dataflows.errors import VendorUnavailableError

    monkeypatch.setattr(vendor.requests, "get",
                        lambda *a, **k: FakeResponse([], status_code=status))
    with pytest.raises(VendorUnavailableError):
        vendor.fetch_klines("BTCUSDT", days=1)


def test_a_rejected_symbol_is_no_market_data_not_an_outage(monkeypatch):
    """HTTP 400 is permanent, so the router must not retry another vendor for it."""
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor
    from tradingagents.dataflows.errors import NoMarketDataError

    monkeypatch.setattr(vendor.requests, "get",
                        lambda *a, **k: FakeResponse({"code": -1121}, status_code=400))
    with pytest.raises(NoMarketDataError):
        vendor.fetch_klines("NOTACOIN", days=1)


def test_a_connection_error_is_mapped_not_leaked(monkeypatch):
    import requests

    import tradingagents.dataflows.vendors.binance_ohlcv as vendor
    from tradingagents.dataflows.errors import VendorUnavailableError

    def boom(*args, **kwargs):
        raise requests.ConnectionError("DNS failure")

    monkeypatch.setattr(vendor.requests, "get", boom)
    with pytest.raises(VendorUnavailableError, match="request failed"):
        vendor.fetch_klines("BTCUSDT", days=1)


def test_an_empty_window_raises_no_market_data(monkeypatch):
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor
    from tradingagents.dataflows.errors import NoMarketDataError

    monkeypatch.setattr(vendor.requests, "get", lambda *a, **k: FakeResponse([]))
    with pytest.raises(NoMarketDataError):
        vendor.fetch_klines("BTCUSDT", days=1)


def test_fetch_rejects_a_nonsensical_window():
    import tradingagents.dataflows.vendors.binance_ohlcv as vendor

    with pytest.raises(ValueError, match="days must be"):
        vendor.fetch_klines("BTCUSDT", days=0)


def test_report_renders_without_warmed_up_indicators():
    """Early bars have NaN indicators; the prompt must show n/a, not 'nan'."""
    from tradingagents.dataflows.vendors.binance_ohlcv import render_1h_report

    data = compute_indicators(make_frame([100.0] * 20))
    report = render_1h_report(data, "BTCUSDT", bars=3)
    assert "n/a" in report
    assert "EMA200" in report



# --- executor guardrails ----------------------------------------------------


class FakeFilters:
    """A stand-in client exposing only what the executor reads."""

    def __init__(self, step="0.00001", tick="0.01", min_qty="0.00001",
                 min_notional="10", price="50000", free="10000",
                 base_asset="BTC", quote_asset="USDT"):
        self._price, self._free = price, free
        self._info = {
            "baseAsset": base_asset,
            "quoteAsset": quote_asset,
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": step,
                 "minQty": min_qty, "maxQty": "9000"},
                {"filterType": "PRICE_FILTER", "tickSize": tick},
                {"filterType": "NOTIONAL", "minNotional": min_notional},
            ]
        }
        self.balance_calls: list[str] = []

    def get_symbol_info(self, symbol):
        return self._info

    def get_symbol_ticker(self, symbol):
        return {"price": self._price}

    def get_asset_balance(self, asset):
        self.balance_calls.append(asset)
        return {"free": self._free}

    def ping(self):
        return {}


def executor_with(client, symbol="BTCUSDT", **kwargs):
    from binance_executor import BinanceExecutor
    executor = BinanceExecutor(symbol=symbol, env="testnet",
                               api_key="k", api_secret="s", **kwargs)
    executor._client = client
    return executor


# --- choosing the traded pair ----------------------------------------------


def test_symbol_is_normalized_from_common_spellings():
    from tradingagents.dataflows.vendors.binance_ohlcv import normalize_symbol
    for spelling in ("ethusdt", " ETH-USDT ", "eth/usdt", "ETH_USDT"):
        assert normalize_symbol(spelling) == "ETHUSDT"


@pytest.mark.parametrize("bad", ["", "   ", "BTC USDT", "BTC!", "ABC", "x" * 30])
def test_malformed_symbols_are_refused_locally(bad):
    """A typo must fail here, not as an opaque HTTP 400 several calls later."""
    from tradingagents.dataflows.vendors.binance_ohlcv import normalize_symbol
    with pytest.raises(ValueError):
        normalize_symbol(bad)


def test_symbol_falls_back_to_the_environment_then_to_btc(monkeypatch):
    from tradingagents.dataflows.vendors.binance_ohlcv import (
        DEFAULT_SYMBOL,
        resolve_symbol,
    )
    monkeypatch.delenv("TRADING_SYMBOL", raising=False)
    assert resolve_symbol() == DEFAULT_SYMBOL
    monkeypatch.setenv("TRADING_SYMBOL", "solusdt")
    assert resolve_symbol() == "SOLUSDT"
    # An explicit argument always beats the environment.
    assert resolve_symbol("ETHUSDT") == "ETHUSDT"


def test_executor_picks_up_the_environment_symbol(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "ethusdt")
    executor = executor_with(FakeFilters(base_asset="ETH"), symbol=None)
    assert executor.symbol == "ETHUSDT"


def test_base_and_quote_come_from_exchange_info_not_the_ticker():
    """Slicing the ticker would mis-identify the assets on non-USDT pairs."""
    client = FakeFilters(base_asset="SOL", quote_asset="FDUSD")
    executor = executor_with(client, symbol="SOLFDUSD")
    assert executor.base_asset == "SOL"
    assert executor.quote_asset == "FDUSD"
    executor.get_equity()
    executor.get_position_quantity()
    assert client.balance_calls == ["FDUSD", "SOL"]


def test_non_usd_quoted_pair_is_refused():
    """A $1,000 cap on ETHBTC would mean 1,000 BTC, so the pair is refused."""
    from binance_executor import ExecutionRejected
    executor = executor_with(
        FakeFilters(base_asset="ETH", quote_asset="BTC"), symbol="ETHBTC"
    )
    with pytest.raises(ExecutionRejected, match="not a USD stablecoin"):
        executor.load_filters()


def test_backtest_cli_rejects_a_malformed_symbol(capsys, monkeypatch):
    import run_1h_backtest
    monkeypatch.delenv("TRADING_SYMBOL", raising=False)
    assert run_1h_backtest.main(["--symbol", "NOT A PAIR"]) == 1
    assert "Invalid --symbol" in capsys.readouterr().err


def test_notional_is_hard_capped_at_1000_usd():
    """The absolute cap holds however large the requested size is."""
    from binance_executor import MAX_TRADE_NOTIONAL_USD
    executor = executor_with(FakeFilters(free="10000000"))
    price = Decimal("50000")
    quantity = executor.prepare_quantity(price, Decimal("50"))  # 2.5M requested
    assert quantity * price <= MAX_TRADE_NOTIONAL_USD


def test_notional_is_capped_at_20_percent_of_a_small_account():
    """With $1,000 equity the 20% rule binds before the $1,000 absolute cap."""
    executor = executor_with(FakeFilters(free="1000"))
    price = Decimal("50000")
    quantity = executor.prepare_quantity(price, Decimal("1"), equity=Decimal("1000"))
    assert quantity * price <= Decimal("200")


def test_quantity_is_truncated_down_to_the_lot_step():
    executor = executor_with(FakeFilters(step="0.001"))
    quantity = executor.prepare_quantity(Decimal("100"), Decimal("1.23456"))
    assert quantity == quantity.quantize(Decimal("0.001"))
    assert quantity <= Decimal("1.23456")


def test_order_below_min_notional_is_rejected():
    from binance_executor import ExecutionRejected
    executor = executor_with(FakeFilters(min_notional="1000"))
    with pytest.raises(ExecutionRejected, match="below the"):
        executor.prepare_quantity(Decimal("100"), Decimal("0.5"))   # $50 notional


def test_order_below_min_quantity_is_rejected():
    from binance_executor import ExecutionRejected
    executor = executor_with(FakeFilters(step="1", min_qty="1", min_notional="0"))
    with pytest.raises(ExecutionRejected, match="below the"):
        executor.prepare_quantity(Decimal("100"), Decimal("0.4"))   # truncates to 0


def test_price_is_truncated_to_the_tick_size():
    executor = executor_with(FakeFilters(tick="0.05"))
    filters = executor.load_filters()
    assert filters.normalize_price(Decimal("100.07")) == Decimal("100.05")
    assert filters.normalize_price(Decimal("100.04")) == Decimal("100.00")


def test_hold_routes_no_order():
    executor = executor_with(FakeFilters())
    assert executor.execute("HOLD")["status"] == "HOLD"


def test_unknown_action_is_rejected():
    from binance_executor import ExecutionRejected
    executor = executor_with(FakeFilters())
    with pytest.raises(ExecutionRejected, match="unknown action"):
        executor.execute("LIQUIDATE EVERYTHING")


def test_dry_run_buy_passes_guardrails_without_ordering():
    executor = executor_with(FakeFilters())
    plan = executor.execute("BUY", quantity=0.5, stop_price=49000, take_profit=52000,
                            dry_run=True)
    assert plan["status"] == "DRY_RUN"
    assert Decimal(plan["notional"]) <= Decimal("1000")
    assert "entry_order" not in plan


def test_live_environment_must_be_spelled_correctly():
    from binance_executor import BinanceExecutor
    with pytest.raises(ValueError, match="testnet"):
        BinanceExecutor(env="production")
