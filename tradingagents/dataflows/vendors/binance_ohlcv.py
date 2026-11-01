"""Binance 1-Hour OHLCV ingestion and the intraday indicator set.

The rest of the dataflows layer is built for daily bars from Yahoo: one row per
session, a ``Date`` column normalized to midnight, and a look-ahead filter that
compares calendar dates. A 1H scalping loop needs none of that and breaks all of
it, so this vendor is deliberately separate rather than a mode flag on
``yahoo/ohlcv.py``:

  * the bar key is an hourly UTC boundary (``YYYY-MM-DD HH:00:00 UTC``), not a
    date, so the daily staleness and midnight-normalization rules do not apply;
  * Binance serves the pair natively as ``BTCUSDT``, so the Yahoo symbol
    normalization (``BTC-USD``) must NOT be applied here — it would query a
    symbol Binance does not list;
  * the indicator set (EMA/RSI/MACD/ATR) is computed in-frame because the
    tactical layer needs ATR on every bar, not a rendered report string.

Only the keyless public REST endpoint is used (``GET /api/v3/klines``), so this
module never needs an API key and never signs a request. Order placement lives
in ``binance_executor.py`` and is kept strictly out of this file.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

logger = logging.getLogger(__name__)

BINANCE_REST_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"

# Binance caps a single klines response at 1000 rows, so any window longer than
# ~41 days of hourly bars must be paginated.
MAX_KLINES_PER_REQUEST = 1000

HOUR_MS = 3_600_000
REQUEST_TIMEOUT_SECONDS = 20.0

# Hourly bars in a 365-day year, used to annualize Sharpe/Sortino.
HOURS_PER_YEAR = 8_760

# The hourly timestamp format the agents are given, so a 1H run is unambiguous
# about which bar it is reasoning over (daily runs pass a bare YYYY-MM-DD).
HOUR_FORMAT = "%Y-%m-%d %H:00:00 UTC"

# Column order every consumer of this module can rely on.
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Indicator parameters. Centralized so the backtest, the executor and the
# agent-facing report all size positions off exactly the same ATR.
EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14


def floor_to_hour(moment: datetime | None = None) -> datetime:
    """The hourly boundary ``moment`` falls in, as an aware UTC datetime.

    A 1H bar is only final at the top of the next hour, so every timestamp in
    this pipeline is snapped down to its opening boundary. Naive input is
    assumed to be UTC rather than silently taking the host timezone, which would
    shift every bar key on a non-UTC machine.
    """
    moment = moment or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def format_hour(moment: datetime) -> str:
    """Render an hourly boundary as ``YYYY-MM-DD HH:00:00 UTC`` for the agents."""
    return floor_to_hour(moment).strftime(HOUR_FORMAT)


def _request_klines(symbol: str, interval: str, start_ms: int, limit: int) -> list[list[Any]]:
    """One page of raw klines, with transport failures mapped to the vendor taxonomy.

    Binance answers a bad symbol with HTTP 400 and a JSON body, which is a
    permanent "no such instrument" rather than an outage, so it is reported as
    ``NoMarketDataError``; everything else (throttling, 5xx, connection
    failures) is a transient ``VendorUnavailableError`` the caller may retry.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    try:
        response = requests.get(
            f"{BINANCE_REST_BASE}{KLINES_PATH}",
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Binance klines request failed: {exc}") from exc

    if response.status_code == 400:
        raise NoMarketDataError(symbol, symbol, "Binance rejected the symbol or interval")
    if response.status_code in (418, 429):
        raise VendorUnavailableError(
            f"Binance throttled the klines request (HTTP {response.status_code})"
        )
    if response.status_code >= 500:
        raise VendorUnavailableError(
            f"Binance klines endpoint returned HTTP {response.status_code}"
        )
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise VendorUnavailableError(f"Binance klines response was unusable: {exc}") from exc

    if not isinstance(payload, list):
        raise VendorUnavailableError("Binance klines response was not a list of bars")
    return payload


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 60,
    end: datetime | None = None,
) -> pd.DataFrame:
    """``days`` of ``interval`` candles for ``symbol``, oldest first.

    Pages through the 1000-row response cap until the window is covered. Only
    bars that have *closed* are returned: the in-progress candle has a moving
    close that would let a backtest trade on a price that never settled.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")

    end_boundary = floor_to_hour(end)
    start_boundary = end_boundary - timedelta(days=days)
    start_ms = int(start_boundary.timestamp() * 1000)
    # A bar is complete only once its close time has passed, so the newest bar
    # eligible for inclusion is the one that opened an hour before the current
    # boundary.
    last_open_ms = int(end_boundary.timestamp() * 1000) - HOUR_MS

    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= last_open_ms:
        page = _request_klines(symbol, interval, cursor, MAX_KLINES_PER_REQUEST)
        if not page:
            break
        rows.extend(page)
        newest_open = int(page[-1][0])
        if len(page) < MAX_KLINES_PER_REQUEST:
            break
        cursor = newest_open + HOUR_MS
        # Stay well inside the public weight limit when paginating a long window.
        time.sleep(0.12)

    if not rows:
        raise NoMarketDataError(symbol, symbol, f"Binance returned no {interval} bars")

    frame = _to_frame(rows)
    frame = frame[frame["OpenTimeMs"] <= last_open_ms]
    frame = frame.drop(columns=["OpenTimeMs"]).reset_index(drop=True)
    if frame.empty:
        raise NoMarketDataError(symbol, symbol, f"no closed {interval} bars in the window")

    logger.info(
        "Binance %s %s: %d closed bars from %s to %s",
        symbol, interval, len(frame),
        format_hour(frame["Date"].iloc[0]), format_hour(frame["Date"].iloc[-1]),
    )
    return frame


def _to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Raw Binance kline arrays into a typed, de-duplicated OHLCV frame.

    Pagination overlaps by one bar at each page boundary, so duplicate open
    times are dropped here rather than being left to double-count in indicators.
    """
    frame = pd.DataFrame(
        [
            {
                "OpenTimeMs": int(row[0]),
                "Open": row[1],
                "High": row[2],
                "Low": row[3],
                "Close": row[4],
                "Volume": row[5],
            }
            for row in rows
        ]
    )
    frame = frame.drop_duplicates(subset="OpenTimeMs").sort_values("OpenTimeMs")
    frame["Date"] = pd.to_datetime(frame["OpenTimeMs"], unit="ms", utc=True)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    return frame[["OpenTimeMs", *OHLCV_COLUMNS]]


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, the averaging RSI and ATR are defined with.

    ``ewm(alpha=1/period)`` is the recursive form Wilder specified; a plain
    rolling mean gives visibly different values and would make the ATR stops
    disagree with every charting package.
    """
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA(20/50/200), RSI(14), MACD(12/26/9) and ATR(14) to 1H closes.

    Returns a new frame; the input is not mutated. Leading rows where a given
    indicator has not warmed up hold NaN rather than a partial value, so a
    caller cannot accidentally size a position off an ATR built from three bars.
    """
    if frame.empty:
        raise ValueError("cannot compute indicators on an empty frame")
    for column in OHLCV_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"OHLCV frame is missing the {column!r} column")

    data = frame.copy().reset_index(drop=True)
    close = data["Close"]

    for period in EMA_PERIODS:
        data[f"EMA{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), RSI_PERIOD)
    loss = _wilder((-delta).clip(lower=0.0), RSI_PERIOD)
    # A window with no losing bar has infinite relative strength; RSI is 100
    # there by definition, and dividing by zero would otherwise yield NaN.
    relative_strength = gain / loss.replace(0.0, pd.NA)
    data["RSI14"] = (100 - 100 / (1 + relative_strength)).fillna(
        pd.Series(100.0, index=data.index).where(gain > 0)
    )

    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    data["MACD"] = ema_fast - ema_slow
    data["MACDSignal"] = data["MACD"].ewm(
        span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL
    ).mean()
    data["MACDHist"] = data["MACD"] - data["MACDSignal"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR14"] = _wilder(true_range, ATR_PERIOD)

    return data


def load_1h_ohlcv(
    symbol: str = "BTCUSDT",
    days: int = 60,
    end: datetime | None = None,
    with_indicators: bool = True,
) -> pd.DataFrame:
    """The 1H bar series for ``symbol``, indicators attached, oldest first.

    This is the single entry point the strategic (agent) and tactical
    (backtest/executor) layers should both call, so they cannot drift onto
    differently-parameterized ATRs.
    """
    frame = fetch_klines(symbol=symbol, interval="1h", days=days, end=end)
    return compute_indicators(frame) if with_indicators else frame


def latest_bar(frame: pd.DataFrame) -> pd.Series:
    """The most recent closed bar, as a Series."""
    if frame.empty:
        raise ValueError("cannot take the latest bar of an empty frame")
    return frame.iloc[-1]


def render_1h_report(frame: pd.DataFrame, symbol: str = "BTCUSDT", bars: int = 12) -> str:
    """A compact text snapshot of the 1H state for an LLM analyst prompt.

    Kept to the last ``bars`` rows and rounded: the strategic layer needs the
    current regime, not a thousand rows of history it would pay tokens to read.
    """
    if frame.empty:
        return f"No 1H market data available for {symbol}."

    recent = frame.tail(bars)
    current = latest_bar(frame)
    lines = [
        f"# {symbol} — 1H timeframe as of {format_hour(current['Date'])}",
        "",
        f"Close: {current['Close']:.2f}",
        _indicator_line("EMA20", current.get("EMA20")),
        _indicator_line("EMA50", current.get("EMA50")),
        _indicator_line("EMA200", current.get("EMA200")),
        _indicator_line("RSI14", current.get("RSI14")),
        _indicator_line("MACD", current.get("MACD")),
        _indicator_line("MACD signal", current.get("MACDSignal")),
        _indicator_line("ATR14", current.get("ATR14")),
        "",
        f"## Last {len(recent)} hourly bars",
        "| Hour (UTC) | Open | High | Low | Close | RSI14 | ATR14 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in recent.iterrows():
        lines.append(
            f"| {format_hour(row['Date'])} | {row['Open']:.2f} | {row['High']:.2f} | "
            f"{row['Low']:.2f} | {row['Close']:.2f} | "
            f"{_cell(row.get('RSI14'))} | {_cell(row.get('ATR14'))} |"
        )
    return "\n".join(lines)


def _indicator_line(label: str, value: Any) -> str:
    return f"{label}: {_cell(value)}"


def _cell(value: Any) -> str:
    """Format an indicator value, showing a not-yet-warmed-up one as ``n/a``."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"
