#!/usr/bin/env python3
"""Isolated Binance execution bridge for the 1H pipeline.

This is the only module in the project that can move real money, so it is kept
deliberately small, dependency-light and importable on its own. Nothing here
calls an LLM, and nothing here decides direction: it receives a side and a
volatility band and is responsible purely for turning that into an order the
exchange will accept, or refusing to.

The guardrails below are hardcoded rather than configurable because they exist
to bound a *model* failure. A prompt-injected or simply wrong agent can ask for
a 50 BTC position; a limit that the same agent could raise would be no limit at
all. In order of application:

  1. ``exchangeInfo`` filters (LOT_SIZE / PRICE_FILTER / NOTIONAL) are fetched
     from the venue and every quantity and price is truncated to them.
  2. Notional is capped at the lesser of ``MAX_TRADE_NOTIONAL_USD`` and
     ``MAX_EQUITY_FRACTION`` of account equity.
  3. Anything below the venue's ``minNotional`` is rejected rather than sent.

Truncation is always *down* and always in ``Decimal``. Binary floats cannot
represent a 0.00000001 step, and rounding up can produce a quantity larger than
the balance that funds it; both are routine causes of exchange rejections.

Usage:
    python binance_executor.py --check          # connectivity + filters, no orders
    python binance_executor.py --dry-run BUY    # full guardrail pass, nothing sent
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

from tradingagents.dataflows.vendors.binance_ohlcv import (
    FUTURES_MARKET,
    SPOT_MARKET,
    VALID_MARKETS,
    resolve_market,
    resolve_symbol,
)

logger = logging.getLogger("binance_executor")

# --- Hardcoded safety envelope. Not read from config, by design. -------------

# Largest notional a single order may carry, whatever the strategy asks for.
MAX_TRADE_NOTIONAL_USD = Decimal("1000")
# ...and never more than this fraction of account equity, so a small account is
# not fully committed to one trade by the absolute cap above.
MAX_EQUITY_FRACTION = Decimal("0.20")

# --- Leverage envelope (USD-M futures only) ----------------------------------

# On spot, equity and notional are the same currency and the fraction above is
# the whole story. On a perpetual they are not: $13 of margin at 7x controls
# $91 of notional. The fraction is therefore applied to the *margin* a trade
# consumes, and the resulting notional is margin x leverage. Applying it to
# notional instead would make a small account untradeable (20% of $13 is $2.60,
# below every venue minimum) while applying nothing at all would let one trade
# commit the entire wallet.
MAX_LEVERAGE = Decimal("10")
DEFAULT_LEVERAGE = Decimal("5")
# Ceiling on the share of wallet equity one position may post as margin. In
# cross mode the whole wallet backs the position, so committing all of it to a
# single trade means any adverse move liquidates the account.
MAX_MARGIN_FRACTION = Decimal("0.50")
DEFAULT_MARGIN_FRACTION = Decimal("0.20")

# Binance's maintenance-margin rate for the lowest (smallest-notional) tier on
# most USD-M symbols. Used only to estimate how far away liquidation sits, so a
# stop that could never trigger is refused. Deliberately conservative.
MAINTENANCE_MARGIN_RATE = Decimal("0.005")
# A stop must sit comfortably inside the liquidation price, not on top of it:
# at the boundary the position is closed by the venue at a worse price than the
# stop would have achieved, and the "risk" the sizing assumed is fiction.
MAX_STOP_FRACTION_OF_LIQUIDATION = Decimal("0.80")

CROSS_MARGIN = "CROSSED"
ISOLATED_MARGIN = "ISOLATED"

TESTNET_REST_URL = "https://testnet.binance.vision/api"
# USD-M futures testnet is a different host with different keys; spot testnet
# keys are rejected there with -2015, which is the most common setup error.
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

# The notional cap above is denominated in US dollars, and that cap only means
# anything if the pair's quote asset is worth roughly a dollar. On ETHBTC a
# "1000" cap would mean 1000 BTC, so pairs quoted in anything but a USD-pegged
# stablecoin are refused rather than silently handed a millionfold wider
# envelope. Add to this set only after checking the peg.
USD_QUOTE_ASSETS = frozenset({"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "USD"})

# Fallback quote asset, used only before exchangeInfo has been read.
QUOTE_ASSET = "USDT"


class ExecutionRejected(RuntimeError):
    """An order was refused locally and never reached the exchange."""


@dataclass(frozen=True)
class SymbolFilters:
    """The venue's precision and minimum rules for one symbol."""

    symbol: str
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    tick_size: Decimal
    min_notional: Decimal
    base_asset: str = ""
    quote_asset: str = ""

    def normalize_quantity(self, quantity: Decimal) -> Decimal:
        """``quantity`` truncated down to a whole number of lot steps."""
        return _truncate(quantity, self.step_size)

    def normalize_price(self, price: Decimal) -> Decimal:
        """``price`` truncated down to a whole number of ticks."""
        return _truncate(price, self.tick_size)


def _truncate(value: Decimal, step: Decimal) -> Decimal:
    """Largest multiple of ``step`` that is <= ``value``.

    ``quantize`` is not used: it rounds to a number of decimal places, which is
    wrong whenever the step is not a power of ten (a 0.05 tick, say). Flooring
    the quotient handles any step, and the result is normalized so the string
    sent to the API carries no exponent form that the venue would reject.
    """
    if step <= 0:
        return value
    steps = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return (steps * step).normalize()


def _decimal(value: Any, default: str = "0") -> Decimal:
    """Parse an exchange-supplied number, never via float.

    ``Decimal(0.1)`` is 0.1000000000000000055..., which defeats the whole point
    of truncating to a step, so values are stringified first.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _format(value: Decimal) -> str:
    """A plain decimal string for the API (never ``1E-5``)."""
    return format(value.normalize(), "f")


class BinanceExecutor:
    """Places guarded spot orders on Binance testnet or live.

    The client is created lazily so that importing this module — which the
    backtest does, to show the user what *would* be routed — never needs
    credentials or a network call.
    """

    def __init__(
        self,
        symbol: str | None = None,
        env: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.symbol = resolve_symbol(symbol)
        self.env = (env or os.getenv("BINANCE_ENV") or "testnet").strip().lower()
        if self.env not in ("testnet", "live"):
            raise ValueError(f"BINANCE_ENV must be 'testnet' or 'live', got {self.env!r}")
        self._api_key = api_key or os.getenv("BINANCE_API_KEY") or ""
        self._api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY") or ""
        self._client: Any = None
        self._filters: SymbolFilters | None = None
        self.market = SPOT_MARKET

    @property
    def is_live(self) -> bool:
        return self.env == "live"

    @property
    def client(self) -> Any:
        """The python-binance client, created on first use."""
        if self._client is not None:
            return self._client
        try:
            from binance.client import Client
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ExecutionRejected(
                "python-binance is not installed; run: pip install python-binance"
            ) from exc
        if not self._api_key or not self._api_secret:
            raise ExecutionRejected(
                "BINANCE_API_KEY and BINANCE_SECRET_KEY must be set to reach the exchange"
            )
        # testnet=True points python-binance at testnet.binance.vision. Live
        # keys are never sent there, nor testnet keys to production.
        self._client = Client(self._api_key, self._api_secret, testnet=not self.is_live)
        logger.info("Binance client ready (env=%s, symbol=%s)", self.env, self.symbol)
        return self._client

    # --- venue metadata ----------------------------------------------------

    def load_filters(self, refresh: bool = False) -> SymbolFilters:
        """Fetch and cache the symbol's LOT_SIZE / PRICE_FILTER / NOTIONAL rules."""
        if self._filters is not None and not refresh:
            return self._filters
        try:
            info = self.client.get_symbol_info(self.symbol)
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(f"could not read exchangeInfo: {exc}") from exc
        if not info:
            raise ExecutionRejected(f"{self.symbol} is not listed on Binance {self.env}")

        by_type = {entry.get("filterType"): entry for entry in info.get("filters", [])}
        lot = by_type.get("LOT_SIZE", {})
        price = by_type.get("PRICE_FILTER", {})
        # Binance renamed MIN_NOTIONAL to NOTIONAL; both still appear depending
        # on the symbol and environment, so both are accepted.
        notional = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}

        # Base and quote come from exchangeInfo rather than from slicing the
        # ticker: "BTCUSDT".removesuffix("USDT") only works for USDT pairs, and
        # guessing wrong would read the balance of an asset that is not the one
        # being traded.
        quote_asset = str(info.get("quoteAsset") or "").upper()
        base_asset = str(info.get("baseAsset") or "").upper()
        if quote_asset and quote_asset not in USD_QUOTE_ASSETS:
            raise ExecutionRejected(
                f"{self.symbol} is quoted in {quote_asset}, which is not a USD "
                f"stablecoin; the ${MAX_TRADE_NOTIONAL_USD} notional cap would "
                f"not mean dollars. Use a USD-quoted pair "
                f"({', '.join(sorted(USD_QUOTE_ASSETS))})."
            )

        self._filters = SymbolFilters(
            symbol=self.symbol,
            step_size=_decimal(lot.get("stepSize"), "0.00000001"),
            min_qty=_decimal(lot.get("minQty"), "0"),
            max_qty=_decimal(lot.get("maxQty"), "9" * 12),
            tick_size=_decimal(price.get("tickSize"), "0.01"),
            min_notional=_decimal(notional.get("minNotional"), "0"),
            base_asset=base_asset,
            quote_asset=quote_asset,
        )
        logger.info(
            "%s filters: step=%s tick=%s minQty=%s minNotional=%s",
            self.symbol, self._filters.step_size, self._filters.tick_size,
            self._filters.min_qty, self._filters.min_notional,
        )
        return self._filters

    def get_price(self) -> Decimal:
        """Last traded price for the symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return _decimal(ticker["price"])
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(f"could not read the {self.symbol} price: {exc}") from exc

    def get_equity(self) -> Decimal:
        """Free quote-asset balance, used as the equity base for the 20% cap.

        The quote asset comes from exchangeInfo, so a USDC- or FDUSD-quoted
        pair reads the balance it is actually settled in.

        Only the free balance counts: funds already committed to a resting
        order cannot back a new one, and counting them would let the cap be
        breached across several open orders.
        """
        asset = self.quote_asset
        try:
            balance = self.client.get_asset_balance(asset=asset)
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(f"could not read the {asset} balance: {exc}") from exc
        if not balance:
            return Decimal("0")
        return _decimal(balance.get("free"))

    @property
    def base_asset(self) -> str:
        """The asset being bought and sold, per exchangeInfo."""
        filters = self.load_filters()
        return filters.base_asset or self.symbol.removesuffix(filters.quote_asset or QUOTE_ASSET)

    @property
    def quote_asset(self) -> str:
        """The asset the pair is priced and settled in, per exchangeInfo."""
        return self.load_filters().quote_asset or QUOTE_ASSET

    def get_position_quantity(self, base_asset: str | None = None) -> Decimal:
        """Free balance of the base asset — the spot position that can be sold."""
        asset = base_asset or self.base_asset
        try:
            balance = self.client.get_asset_balance(asset=asset)
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(f"could not read the {asset} balance: {exc}") from exc
        return _decimal(balance.get("free")) if balance else Decimal("0")

    # --- the guardrail itself ---------------------------------------------

    def cap_notional(self, requested: Decimal, equity: Decimal | None = None) -> Decimal:
        """``requested`` notional reduced to the hardcoded ceiling.

        Equity is only consulted to make the cap *tighter*. If the balance
        cannot be read, the absolute cap still applies, so a failure here can
        never widen the envelope.
        """
        ceiling = MAX_TRADE_NOTIONAL_USD
        if equity is not None and equity > 0:
            ceiling = min(ceiling, equity * MAX_EQUITY_FRACTION)
        if requested > ceiling:
            logger.warning(
                "Requested notional %s exceeds the safety cap; reducing to %s",
                _format(requested), _format(ceiling),
            )
        return min(requested, ceiling)

    def prepare_quantity(
        self,
        price: Decimal,
        requested_quantity: Decimal,
        equity: Decimal | None = None,
    ) -> Decimal:
        """A venue-legal, capped quantity, or raise ``ExecutionRejected``.

        This is the single chokepoint every order passes through, so the cap
        cannot be bypassed by calling a different order method.
        """
        filters = self.load_filters()
        if price <= 0:
            raise ExecutionRejected(f"price must be positive, got {price}")
        if requested_quantity <= 0:
            raise ExecutionRejected(f"quantity must be positive, got {requested_quantity}")

        capped_notional = self.cap_notional(price * requested_quantity, equity)
        quantity = filters.normalize_quantity(capped_notional / price)

        if quantity > filters.max_qty:
            quantity = filters.normalize_quantity(filters.max_qty)
        if quantity <= 0 or quantity < filters.min_qty:
            raise ExecutionRejected(
                f"quantity {_format(quantity)} is below the {self.symbol} minimum "
                f"{_format(filters.min_qty)} after lot-size truncation"
            )
        # Truncating down can drop the order under the venue minimum even though
        # the pre-truncation notional cleared it, so the check runs on the final
        # quantity rather than the requested one.
        notional = quantity * price
        if filters.min_notional > 0 and notional < filters.min_notional:
            raise ExecutionRejected(
                f"notional {_format(notional)} {QUOTE_ASSET} is below the "
                f"{self.symbol} minimum {_format(filters.min_notional)}"
            )
        return quantity

    # --- order routing -----------------------------------------------------

    def execute(
        self,
        action: str,
        quantity: float | Decimal | None = None,
        stop_price: float | Decimal | None = None,
        take_profit: float | Decimal | None = None,
        limit_price: float | Decimal | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Route a BUY / SELL / HOLD decision.

        BUY opens a long and immediately attaches a bracket; SELL closes the
        spot position; HOLD does nothing but report. ``dry_run`` runs every
        guardrail and returns the order that *would* be sent.
        """
        verdict = (action or "").strip().upper()
        if verdict == "HOLD":
            logger.info("HOLD — no order routed for %s; waiting for the next 1H close",
                        self.symbol)
            return {"status": "HOLD", "symbol": self.symbol}
        if verdict == "BUY":
            return self.open_long(quantity, stop_price, take_profit, limit_price, dry_run)
        if verdict == "SELL":
            return self.close_position(dry_run=dry_run)
        raise ExecutionRejected(f"unknown action {action!r}; expected BUY, SELL or HOLD")

    def open_long(
        self,
        quantity: float | Decimal | None,
        stop_price: float | Decimal | None,
        take_profit: float | Decimal | None = None,
        limit_price: float | Decimal | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Buy, then attach the protective bracket.

        The stop is placed immediately after the fill rather than on the next
        loop iteration: between those two points the position would be
        unprotected, and a 1H loop can be minutes away from its next wake-up.
        """
        filters = self.load_filters()
        price = self.get_price() if limit_price is None else _decimal(limit_price)
        equity = self._equity_or_none()

        requested = _decimal(quantity) if quantity is not None else (
            self.cap_notional(MAX_TRADE_NOTIONAL_USD, equity) / price
        )
        final_quantity = self.prepare_quantity(price, requested, equity)

        plan: dict[str, Any] = {
            "status": "DRY_RUN" if dry_run else "SUBMITTED",
            "env": self.env,
            "symbol": self.symbol,
            "side": "BUY",
            "quantity": _format(final_quantity),
            "reference_price": _format(price),
            "notional": _format(final_quantity * price),
        }
        if stop_price is not None:
            plan["stop_price"] = _format(filters.normalize_price(_decimal(stop_price)))
        if take_profit is not None:
            plan["take_profit"] = _format(filters.normalize_price(_decimal(take_profit)))

        if dry_run:
            logger.info("DRY RUN — would submit: %s", plan)
            return plan

        try:
            if limit_price is None:
                entry = self.client.order_market_buy(
                    symbol=self.symbol, quantity=_format(final_quantity)
                )
            else:
                entry = self.client.order_limit_buy(
                    symbol=self.symbol,
                    quantity=_format(final_quantity),
                    price=_format(filters.normalize_price(price)),
                )
        except Exception as exc:
            raise ExecutionRejected(f"entry order failed: {exc}") from exc

        plan["entry_order"] = entry
        if stop_price is not None:
            plan["bracket"] = self._attach_bracket(
                final_quantity, _decimal(stop_price),
                _decimal(take_profit) if take_profit is not None else None,
            )
        return plan

    def _attach_bracket(
        self,
        quantity: Decimal,
        stop_price: Decimal,
        take_profit: Decimal | None,
    ) -> dict[str, Any]:
        """Protect an open long with an OCO, falling back to a plain stop.

        OCO is preferred because the exchange then cancels the stop when the
        target fills (and vice versa); with two independent orders, one fill
        leaves the other live and able to open an unintended opposite position.
        A venue that refuses the OCO still gets a stop — an unprotected
        position is the one outcome not accepted here.
        """
        filters = self.load_filters()
        stop = filters.normalize_price(stop_price)
        # The stop-limit is placed slightly through the trigger so a gapping
        # market still fills it; a limit exactly at the trigger often does not.
        stop_limit = filters.normalize_price(stop * Decimal("0.999"))

        if take_profit is not None:
            try:
                oco = self.client.create_oco_order(
                    symbol=self.symbol,
                    side="SELL",
                    quantity=_format(quantity),
                    price=_format(filters.normalize_price(take_profit)),
                    stopPrice=_format(stop),
                    stopLimitPrice=_format(stop_limit),
                    stopLimitTimeInForce="GTC",
                )
                return {"type": "OCO", "order": oco}
            except Exception as exc:
                logger.warning("OCO bracket rejected (%s); falling back to a stop-loss", exc)

        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="SELL",
                type="STOP_LOSS_LIMIT",
                timeInForce="GTC",
                quantity=_format(quantity),
                price=_format(stop_limit),
                stopPrice=_format(stop),
            )
            return {"type": "STOP_LOSS_LIMIT", "order": order}
        except Exception as exc:
            # The position is open and unprotected: say so loudly rather than
            # returning a success-shaped result.
            logger.error("STOP-LOSS PLACEMENT FAILED for %s — position is UNPROTECTED: %s",
                         self.symbol, exc)
            raise ExecutionRejected(
                f"entry filled but the stop could not be placed: {exc}"
            ) from exc

    def close_position(self, dry_run: bool = False) -> dict[str, Any]:
        """Cancel resting orders and sell the whole free base-asset balance.

        Open orders are cancelled first: the bracket from the entry still holds
        a claim on the balance, and selling underneath it would either fail for
        insufficient funds or leave a stop order with nothing behind it.
        """
        filters = self.load_filters()
        quantity = filters.normalize_quantity(self.get_position_quantity())
        if quantity <= 0 or quantity < filters.min_qty:
            logger.info("No %s position to close", self.symbol)
            return {"status": "FLAT", "symbol": self.symbol}

        if dry_run:
            return {"status": "DRY_RUN", "side": "SELL", "symbol": self.symbol,
                    "quantity": _format(quantity)}

        self._cancel_open_orders()
        # Cancelling frees the balance the bracket was holding, so the sellable
        # quantity is re-read rather than reused from before the cancel.
        quantity = filters.normalize_quantity(self.get_position_quantity())
        if quantity <= 0 or quantity < filters.min_qty:
            return {"status": "FLAT", "symbol": self.symbol}
        try:
            order = self.client.order_market_sell(
                symbol=self.symbol, quantity=_format(quantity)
            )
        except Exception as exc:
            raise ExecutionRejected(f"close order failed: {exc}") from exc
        return {"status": "CLOSED", "symbol": self.symbol,
                "quantity": _format(quantity), "order": order}

    def _cancel_open_orders(self) -> None:
        """Best-effort cancel of resting orders; a vanished order is not an error."""
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
        except Exception as exc:
            logger.warning("Could not list open %s orders: %s", self.symbol, exc)
            return
        for order in open_orders or []:
            try:
                self.client.cancel_order(symbol=self.symbol, orderId=order["orderId"])
            except Exception as exc:
                logger.warning("Could not cancel order %s: %s", order.get("orderId"), exc)

    def _equity_or_none(self) -> Decimal | None:
        """Account equity, or None when it cannot be read.

        None keeps the absolute notional cap in force while skipping the
        percentage one, so an unreadable balance never widens the envelope.
        """
        try:
            return self.get_equity()
        except ExecutionRejected as exc:
            logger.warning("Could not read equity (%s); the absolute cap still applies", exc)
            return None

    def check_connectivity(self) -> dict[str, Any]:
        """Read-only probe: credentials, filters, price and balance. Places no order."""
        report: dict[str, Any] = {
            "env": self.env, "market": self.market, "symbol": self.symbol,
        }
        self.client.ping()
        filters = self.load_filters()
        report["filters"] = {
            "step_size": _format(filters.step_size),
            "tick_size": _format(filters.tick_size),
            "min_qty": _format(filters.min_qty),
            "min_notional": _format(filters.min_notional),
        }
        report["price"] = _format(self.get_price())
        report["base_asset"] = filters.base_asset or self.base_asset
        report["quote_asset"] = filters.quote_asset or QUOTE_ASSET
        report["equity_quote"] = _format(self.get_equity())
        report["position_base"] = _format(self.get_position_quantity())
        report["max_trade_notional"] = _format(
            self.cap_notional(MAX_TRADE_NOTIONAL_USD, self._equity_or_none())
        )
        return report


class BinanceFuturesExecutor(BinanceExecutor):
    """The same guardrails, routed to USD-M perpetual futures.

    Subclassed rather than branched because every venue-specific call differs
    (``futures_*`` endpoints, a wallet instead of per-asset balances, signed
    positions instead of a base-asset balance). The *policy* — truncate to the
    venue's filters, cap the size, never leave a position unprotected — is
    inherited unchanged, so there is exactly one implementation of it.

    Two differences from spot are not cosmetic:

      * **Leverage.** Equity no longer bounds notional, so the fraction is
        applied to margin and the notional ceiling becomes margin x leverage.
      * **Shorts are real.** On spot, SELL can only mean "close". Here it could
        mean "open a leveraged short", so SELL keeps its spot meaning (close)
        and opening a short requires the explicit ``SHORT`` action. Guessing
        would eventually open an unwanted position.
    """

    def __init__(
        self,
        symbol: str | None = None,
        env: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        leverage: int | Decimal | None = None,
        margin_mode: str | None = None,
        margin_fraction: Decimal | None = None,
    ) -> None:
        super().__init__(symbol=symbol, env=env, api_key=api_key, api_secret=api_secret)
        self.market = FUTURES_MARKET
        self.leverage = _bounded_leverage(
            leverage if leverage is not None else os.getenv("BINANCE_LEVERAGE")
        )
        self.margin_fraction = _bounded_margin_fraction(
            margin_fraction if margin_fraction is not None
            else os.getenv("BINANCE_MARGIN_FRACTION")
        )
        mode = (margin_mode or os.getenv("BINANCE_MARGIN_MODE") or CROSS_MARGIN).upper()
        if mode in ("CROSS", "CROSSED"):
            mode = CROSS_MARGIN
        if mode not in (CROSS_MARGIN, ISOLATED_MARGIN):
            raise ValueError(
                f"BINANCE_MARGIN_MODE must be CROSSED or ISOLATED, got {mode!r}"
            )
        self.margin_mode = mode
        self._account_configured = False

    # --- venue metadata ----------------------------------------------------

    def load_filters(self, refresh: bool = False) -> SymbolFilters:
        """Filters from ``futures_exchange_info``.

        The futures endpoint returns every symbol in one payload rather than
        offering a per-symbol lookup, so the match is done here.
        """
        if self._filters is not None and not refresh:
            return self._filters
        try:
            info = self.client.futures_exchange_info()
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(f"could not read futures exchangeInfo: {exc}") from exc

        entry = next(
            (s for s in (info or {}).get("symbols", []) if s.get("symbol") == self.symbol),
            None,
        )
        if entry is None:
            raise ExecutionRejected(
                f"{self.symbol} is not listed on Binance USD-M futures {self.env}"
            )
        # A delisted or settling contract still appears in exchangeInfo; only
        # TRADING accepts orders.
        status = entry.get("status") or entry.get("contractStatus")
        if status and status != "TRADING":
            raise ExecutionRejected(f"{self.symbol} is not trading (status {status})")

        by_type = {f.get("filterType"): f for f in entry.get("filters", [])}
        lot = by_type.get("MARKET_LOT_SIZE") or by_type.get("LOT_SIZE") or {}
        price = by_type.get("PRICE_FILTER", {})
        notional = by_type.get("MIN_NOTIONAL") or by_type.get("NOTIONAL") or {}

        quote_asset = str(entry.get("quoteAsset") or "").upper()
        if quote_asset and quote_asset not in USD_QUOTE_ASSETS:
            raise ExecutionRejected(
                f"{self.symbol} is quoted in {quote_asset}; only USD-margined "
                f"contracts are supported (COIN-M is a different risk model)"
            )

        self._filters = SymbolFilters(
            symbol=self.symbol,
            step_size=_decimal(lot.get("stepSize"), "0.001"),
            min_qty=_decimal(lot.get("minQty"), "0"),
            max_qty=_decimal(lot.get("maxQty"), "9" * 12),
            tick_size=_decimal(price.get("tickSize"), "0.01"),
            min_notional=_decimal(notional.get("notional") or notional.get("minNotional"), "0"),
            base_asset=str(entry.get("baseAsset") or "").upper(),
            quote_asset=quote_asset,
        )
        logger.info(
            "%s futures filters: step=%s tick=%s minQty=%s minNotional=%s",
            self.symbol, self._filters.step_size, self._filters.tick_size,
            self._filters.min_qty, self._filters.min_notional,
        )
        return self._filters

    def get_price(self) -> Decimal:
        """Mark price, which is what the venue uses for liquidation and PnL.

        The last trade price can be dragged away from the mark by a thin book;
        sizing and stop distances must agree with the number the exchange will
        actually measure them against.
        """
        try:
            mark = self.client.futures_mark_price(symbol=self.symbol)
            if isinstance(mark, list):
                mark = next((m for m in mark if m.get("symbol") == self.symbol), None)
            price = _decimal((mark or {}).get("markPrice"))
            if price > 0:
                return price
        except Exception as exc:
            logger.warning("Mark price unavailable (%s); falling back to last trade", exc)
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            if isinstance(ticker, list):
                ticker = next((t for t in ticker if t.get("symbol") == self.symbol), {})
            return _decimal((ticker or {}).get("price"))
        except Exception as exc:
            raise ExecutionRejected(
                f"could not read the {self.symbol} futures price: {exc}"
            ) from exc

    def get_equity(self) -> Decimal:
        """Available balance of the futures wallet, in the quote asset.

        This is the futures wallet, not the spot wallet: they are funded
        separately, and reading the spot balance here is exactly the mistake
        that produces "Invalid API-key, IP, or permissions" against a
        futures-only key.

        ``availableBalance`` is used rather than ``balance`` because margin
        already committed to open positions cannot back a new one.
        """
        asset = self.quote_asset
        try:
            balances = self.client.futures_account_balance()
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(
                f"could not read the futures {asset} balance: {exc}"
            ) from exc
        for entry in balances or []:
            if str(entry.get("asset", "")).upper() == asset:
                available = entry.get("availableBalance", entry.get("balance"))
                return _decimal(available)
        return Decimal("0")

    def get_position_quantity(self, base_asset: str | None = None) -> Decimal:
        """Signed position size: positive when long, negative when short.

        Spot returns an unsigned balance because a spot position cannot be
        negative. A perpetual can, and collapsing the sign would make a short
        look like a long of the same size.
        """
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
        except ExecutionRejected:
            raise
        except Exception as exc:
            raise ExecutionRejected(
                f"could not read the {self.symbol} position: {exc}"
            ) from exc
        if isinstance(positions, dict):
            positions = [positions]
        total = Decimal("0")
        for entry in positions or []:
            if str(entry.get("symbol", "")).upper() == self.symbol:
                total += _decimal(entry.get("positionAmt"))
        return total

    # --- leverage-aware sizing --------------------------------------------

    def cap_notional(self, requested: Decimal, equity: Decimal | None = None) -> Decimal:
        """``requested`` notional reduced to the leveraged ceiling.

        The margin fraction bounds what fraction of the wallet is at stake;
        leverage then converts that margin into notional. The absolute USD cap
        still applies on top, and an unreadable equity still falls back to it
        alone, so a failed balance read can never widen the envelope.
        """
        ceiling = MAX_TRADE_NOTIONAL_USD
        if equity is not None and equity > 0:
            ceiling = min(ceiling, equity * self.margin_fraction * self.leverage)
        if requested > ceiling:
            logger.warning(
                "Requested notional %s exceeds the futures safety cap; reducing to %s",
                _format(requested), _format(ceiling),
            )
        return min(requested, ceiling)

    def liquidation_distance(self, notional: Decimal, equity: Decimal) -> Decimal | None:
        """Estimated fractional price move from entry to liquidation.

        In cross margin the whole wallet backs the position, so the move that
        exhausts it is ``(equity - maintenance) / notional``. This is an
        estimate — the venue's tiered maintenance table is the authority — and
        is used only to refuse a stop that sits beyond it.
        """
        if notional <= 0 or equity <= 0:
            return None
        maintenance = notional * MAINTENANCE_MARGIN_RATE
        if equity <= maintenance:
            return Decimal("0")
        return (equity - maintenance) / notional

    def check_stop_is_reachable(
        self,
        entry_price: Decimal,
        stop_price: Decimal,
        quantity: Decimal,
        equity: Decimal | None,
    ) -> None:
        """Refuse a stop that sits at or beyond the liquidation price.

        A stop further away than liquidation is not a stop: the venue closes
        the position first, at a worse price and for the whole wallet. Sizing
        that assumed "risk = distance to stop" would be describing a loss that
        cannot happen the way it was modelled.
        """
        if equity is None or entry_price <= 0 or stop_price <= 0:
            return
        distance = self.liquidation_distance(entry_price * quantity, equity)
        if distance is None:
            return
        stop_fraction = abs(entry_price - stop_price) / entry_price
        allowed = distance * MAX_STOP_FRACTION_OF_LIQUIDATION
        if stop_fraction > allowed:
            raise ExecutionRejected(
                f"stop at {_format(stop_price)} is {stop_fraction:.1%} from entry, "
                f"beyond the {allowed:.1%} liquidation-safe limit at "
                f"{_format(self.leverage)}x on {_format(equity)} {self.quote_asset}. "
                f"Reduce leverage, tighten the stop, or fund the wallet."
            )

    def configure_account(self) -> dict[str, Any]:
        """Set margin mode and leverage for the symbol, once per process.

        Both calls are idempotent and both report an error when the setting is
        already in place; that is a success for our purposes, so "no need to
        change margin type" is not treated as a failure.
        """
        if self._account_configured:
            return {"margin_mode": self.margin_mode, "leverage": int(self.leverage)}
        result: dict[str, Any] = {}
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol, marginType=self.margin_mode
            )
            result["margin_mode"] = self.margin_mode
        except Exception as exc:
            if "no need to change" in str(exc).lower() or "-4046" in str(exc):
                result["margin_mode"] = self.margin_mode
            else:
                raise ExecutionRejected(
                    f"could not set {self.margin_mode} margin on {self.symbol}: {exc}"
                ) from exc
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=int(self.leverage)
            )
            result["leverage"] = int(self.leverage)
        except Exception as exc:
            raise ExecutionRejected(
                f"could not set {int(self.leverage)}x leverage on {self.symbol}: {exc}"
            ) from exc
        self._account_configured = True
        logger.info("%s configured: %s margin at %sx",
                    self.symbol, self.margin_mode, int(self.leverage))
        return result

    # --- order routing -----------------------------------------------------

    def execute(
        self,
        action: str,
        quantity: float | Decimal | None = None,
        stop_price: float | Decimal | None = None,
        take_profit: float | Decimal | None = None,
        limit_price: float | Decimal | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Route BUY / LONG / SHORT / SELL / CLOSE / HOLD.

        SELL means *close*, as it does on spot. Opening a short requires the
        explicit SHORT action so that an agent emitting SELL to mean "exit"
        can never be read as "open a leveraged short".
        """
        verdict = (action or "").strip().upper()
        if verdict == "HOLD":
            logger.info("HOLD — no order routed for %s; waiting for the next 1H close",
                        self.symbol)
            return {"status": "HOLD", "symbol": self.symbol, "market": self.market}
        if verdict in ("BUY", "LONG"):
            return self.open_position("BUY", quantity, stop_price, take_profit,
                                      limit_price, dry_run)
        if verdict == "SHORT":
            return self.open_position("SELL", quantity, stop_price, take_profit,
                                      limit_price, dry_run)
        if verdict in ("SELL", "CLOSE", "FLAT"):
            return self.close_position(dry_run=dry_run)
        raise ExecutionRejected(
            f"unknown action {action!r}; expected BUY/LONG, SHORT, SELL/CLOSE or HOLD"
        )

    def open_long(
        self,
        quantity: float | Decimal | None,
        stop_price: float | Decimal | None,
        take_profit: float | Decimal | None = None,
        limit_price: float | Decimal | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Spot-compatible alias for ``open_position('BUY', ...)``."""
        return self.open_position("BUY", quantity, stop_price, take_profit,
                                  limit_price, dry_run)

    def open_position(
        self,
        side: str,
        quantity: float | Decimal | None,
        stop_price: float | Decimal | None,
        take_profit: float | Decimal | None = None,
        limit_price: float | Decimal | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Open a long (BUY) or short (SELL) and attach a reduce-only bracket."""
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ExecutionRejected(f"side must be BUY or SELL, got {side!r}")

        filters = self.load_filters()
        price = self.get_price() if limit_price is None else _decimal(limit_price)
        equity = self._equity_or_none()

        requested = _decimal(quantity) if quantity is not None else (
            self.cap_notional(MAX_TRADE_NOTIONAL_USD, equity) / price
        )
        final_quantity = self.prepare_quantity(price, requested, equity)

        # An existing opposite position would be netted by this order rather
        # than replaced, leaving a size nobody chose. Close first.
        current = self.get_position_quantity()
        if (current > 0 and side == "SELL") or (current < 0 and side == "BUY"):
            raise ExecutionRejected(
                f"an opposite {self.symbol} position of {_format(current)} is open; "
                f"close it before opening the other side"
            )

        stop = filters.normalize_price(_decimal(stop_price)) if stop_price is not None else None
        if stop is not None:
            self._check_stop_side(side, price, stop)
            self.check_stop_is_reachable(price, stop, final_quantity, equity)

        notional = final_quantity * price
        margin = notional / self.leverage if self.leverage > 0 else notional
        plan: dict[str, Any] = {
            "status": "DRY_RUN" if dry_run else "SUBMITTED",
            "env": self.env,
            "market": self.market,
            "symbol": self.symbol,
            "side": side,
            "position": "LONG" if side == "BUY" else "SHORT",
            "quantity": _format(final_quantity),
            "reference_price": _format(price),
            "notional": _format(notional),
            "leverage": int(self.leverage),
            "margin_mode": self.margin_mode,
            "estimated_margin": _format(margin),
        }
        if equity is not None:
            distance = self.liquidation_distance(notional, equity)
            if distance is not None:
                plan["liquidation_distance_pct"] = f"{distance:.2%}"
        if stop is not None:
            plan["stop_price"] = _format(stop)
        if take_profit is not None:
            plan["take_profit"] = _format(filters.normalize_price(_decimal(take_profit)))

        if dry_run:
            logger.info("DRY RUN — would submit: %s", plan)
            return plan

        self.configure_account()
        try:
            entry = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET" if limit_price is None else "LIMIT",
                quantity=_format(final_quantity),
                **({} if limit_price is None else {
                    "price": _format(filters.normalize_price(price)),
                    "timeInForce": "GTC",
                }),
            )
        except Exception as exc:
            raise ExecutionRejected(f"futures entry order failed: {exc}") from exc

        plan["entry_order"] = entry
        if stop is not None:
            plan["bracket"] = self._attach_futures_bracket(
                side, stop,
                filters.normalize_price(_decimal(take_profit))
                if take_profit is not None else None,
            )
        return plan

    @staticmethod
    def _check_stop_side(side: str, price: Decimal, stop: Decimal) -> None:
        """A stop on the wrong side of entry triggers instantly on submission."""
        if side == "BUY" and stop >= price:
            raise ExecutionRejected(
                f"long stop {_format(stop)} is at or above entry {_format(price)}"
            )
        if side == "SELL" and stop <= price:
            raise ExecutionRejected(
                f"short stop {_format(stop)} is at or below entry {_format(price)}"
            )

    def _attach_futures_bracket(
        self,
        entry_side: str,
        stop_price: Decimal,
        take_profit: Decimal | None,
    ) -> dict[str, Any]:
        """Attach STOP_MARKET (and optionally TAKE_PROFIT_MARKET) to the position.

        ``closePosition=true`` makes each order close whatever is open and tells
        the venue to cancel the siblings once one fires, which is the futures
        equivalent of a spot OCO and avoids the leftover-order problem. Both are
        reduce-only by construction, so neither can open a new position.

        MARK_PRICE is the trigger because stops triggered on last price are
        routinely picked off by a single thin-book wick that the mark never
        follows.
        """
        exit_side = "SELL" if entry_side == "BUY" else "BUY"
        result: dict[str, Any] = {}
        try:
            result["stop"] = self.client.futures_create_order(
                symbol=self.symbol,
                side=exit_side,
                type="STOP_MARKET",
                stopPrice=_format(stop_price),
                closePosition=True,
                workingType="MARK_PRICE",
                timeInForce="GTE_GTC",
            )
        except Exception as exc:
            logger.error(
                "STOP PLACEMENT FAILED for %s — position is UNPROTECTED: %s",
                self.symbol, exc,
            )
            raise ExecutionRejected(
                f"entry filled but the stop could not be placed: {exc}"
            ) from exc

        if take_profit is not None:
            try:
                result["take_profit"] = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=exit_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=_format(take_profit),
                    closePosition=True,
                    workingType="MARK_PRICE",
                    timeInForce="GTE_GTC",
                )
            except Exception as exc:
                # The stop is already live, so the position is protected; a
                # missing target costs upside, not the account.
                logger.warning("Take-profit rejected (%s); the stop remains in place", exc)
        return result

    def close_position(self, dry_run: bool = False) -> dict[str, Any]:
        """Flatten the position with a reduce-only market order."""
        filters = self.load_filters()
        position = self.get_position_quantity()
        quantity = filters.normalize_quantity(abs(position))
        if quantity <= 0:
            logger.info("No %s futures position to close", self.symbol)
            return {"status": "FLAT", "symbol": self.symbol, "market": self.market}

        side = "SELL" if position > 0 else "BUY"
        if dry_run:
            return {"status": "DRY_RUN", "side": side, "symbol": self.symbol,
                    "market": self.market, "quantity": _format(quantity)}

        self._cancel_open_orders()
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=_format(quantity),
                reduceOnly=True,
            )
        except Exception as exc:
            raise ExecutionRejected(f"futures close order failed: {exc}") from exc
        return {"status": "CLOSED", "symbol": self.symbol, "market": self.market,
                "side": side, "quantity": _format(quantity), "order": order}

    def _cancel_open_orders(self) -> None:
        """Cancel every resting order on the symbol, including the bracket."""
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol)
        except Exception as exc:
            logger.warning("Could not cancel open %s futures orders: %s", self.symbol, exc)

    def check_connectivity(self) -> dict[str, Any]:
        """Read-only futures probe. Places no order and changes no setting."""
        report: dict[str, Any] = {
            "env": self.env, "market": self.market, "symbol": self.symbol,
        }
        try:
            self.client.futures_ping()
        except Exception as exc:
            raise ExecutionRejected(f"could not reach the futures API: {exc}") from exc

        filters = self.load_filters()
        price = self.get_price()
        equity = self.get_equity()
        report["filters"] = {
            "step_size": _format(filters.step_size),
            "tick_size": _format(filters.tick_size),
            "min_qty": _format(filters.min_qty),
            "min_notional": _format(filters.min_notional),
        }
        report["price"] = _format(price)
        report["base_asset"] = filters.base_asset
        report["quote_asset"] = filters.quote_asset or QUOTE_ASSET
        report["wallet_available"] = _format(equity)
        report["position"] = _format(self.get_position_quantity())
        report["leverage"] = int(self.leverage)
        report["margin_mode"] = self.margin_mode
        max_notional = self.cap_notional(MAX_TRADE_NOTIONAL_USD, equity or None)
        report["max_trade_notional"] = _format(max_notional)

        # The number that actually decides whether this account can trade this
        # contract at all: one minimum lot is often worth more than a small
        # wallet's entire buying power, and the failure otherwise only shows up
        # as a rejected order later.
        smallest = max(filters.min_qty, filters.step_size) * price
        if filters.min_notional > 0:
            smallest = max(smallest, filters.min_notional)
        report["smallest_possible_notional"] = _format(smallest)
        report["tradable"] = "yes" if smallest <= max_notional else "no"
        if smallest > max_notional:
            report["why_not"] = (
                f"the smallest order the venue accepts is {_format(smallest)} "
                f"{filters.quote_asset or QUOTE_ASSET}, above the "
                f"{_format(max_notional)} this wallet may commit at "
                f"{int(self.leverage)}x. Pick a lower-priced contract or add funds."
            )
        return report


def _bounded_leverage(value: Any) -> Decimal:
    """Requested leverage, clamped to ``MAX_LEVERAGE`` and at least 1x."""
    if value in (None, ""):
        return DEFAULT_LEVERAGE
    leverage = _decimal(value, str(DEFAULT_LEVERAGE))
    if leverage < 1:
        leverage = Decimal("1")
    if leverage > MAX_LEVERAGE:
        logger.warning("Leverage %s exceeds the %sx cap; using %s",
                       _format(leverage), _format(MAX_LEVERAGE), _format(MAX_LEVERAGE))
        leverage = MAX_LEVERAGE
    return Decimal(int(leverage))


def _bounded_margin_fraction(value: Any) -> Decimal:
    """Share of the wallet one position may post as margin, clamped to the ceiling."""
    if value in (None, ""):
        return DEFAULT_MARGIN_FRACTION
    fraction = _decimal(value, str(DEFAULT_MARGIN_FRACTION))
    if fraction <= 0:
        return DEFAULT_MARGIN_FRACTION
    if fraction > MAX_MARGIN_FRACTION:
        logger.warning("Margin fraction %s exceeds the %s ceiling; using %s",
                       _format(fraction), _format(MAX_MARGIN_FRACTION),
                       _format(MAX_MARGIN_FRACTION))
        fraction = MAX_MARGIN_FRACTION
    return fraction


def build_executor(
    symbol: str | None = None,
    env: str | None = None,
    market: str | None = None,
    **kwargs: Any,
) -> BinanceExecutor:
    """The executor for the configured market. Spot stays the default."""
    chosen = resolve_market(market)
    if chosen == FUTURES_MARKET:
        return BinanceFuturesExecutor(symbol=symbol, env=env, **kwargs)
    # Leverage settings are meaningless on spot; silently ignoring them would
    # let someone believe they were trading 7x when they were not.
    for key in ("leverage", "margin_mode", "margin_fraction"):
        if kwargs.pop(key, None) is not None:
            raise ValueError(f"{key} is only valid when BINANCE_MARKET=futures")
    return BinanceExecutor(symbol=symbol, env=env, **kwargs)


def _diagnose(message: str, args: argparse.Namespace) -> str:
    """Turn an opaque Binance API error into the setup mistake behind it.

    ``-2015`` in particular has three unrelated causes and the message names
    none of them, so it is worth spelling out rather than making the user guess
    which of key, IP or permission is wrong.
    """
    text = message.lower()
    market = (args.market or os.getenv("BINANCE_MARKET") or SPOT_MARKET).lower()
    env = (args.env or os.getenv("BINANCE_ENV") or "testnet").lower()
    hints: list[str] = []

    if "-2015" in text or "invalid api-key" in text or "permissions for action" in text:
        hints.append(
            "Binance -2015 means the key was rejected, not that it does not exist. "
            "The usual causes, in order:"
        )
        if market != FUTURES_MARKET:
            hints.append(
                "  1. You are querying the SPOT wallet. Funds held in the USD-M "
                "futures wallet are invisible to it. Set BINANCE_MARKET=futures "
                "(or pass --market futures)."
            )
        else:
            hints.append(
                "  1. Futures trading is not enabled on the key. Enable "
                "'Futures' in the key's permissions (it is off by default, and "
                "on live keys it cannot be added after creation — make a new key)."
            )
        if env == "testnet":
            hints.append(
                "  2. Futures testnet keys come from testnet.binancefuture.com. "
                "Spot testnet keys from testnet.binance.vision are rejected "
                "there, and vice versa."
            )
        hints.append(
            "  3. The key has an IP allow-list that does not include this "
            "machine's outbound address."
        )
    elif "-4046" in text or "no need to change margin" in text:
        hints.append("The margin mode was already set; this is not an error.")
    elif "-4131" in text or "-2019" in text or "margin is insufficient" in text:
        hints.append(
            "The wallet cannot fund this order. Lower the notional, raise "
            "leverage (within the cap), or fund the futures wallet."
        )
    elif "-1121" in text or "invalid symbol" in text:
        hints.append(
            f"The symbol is not listed on this venue. A spot pair is not "
            f"automatically a perpetual contract; check it exists on "
            f"{'USD-M futures' if market == FUTURES_MARKET else 'spot'}."
        )
    if not hints:
        return ""
    return "\n" + "\n".join(hints)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance execution bridge (guarded)")
    parser.add_argument("--symbol", default=None,
                        help="pair to trade; defaults to $TRADING_SYMBOL, else BTCUSDT")
    parser.add_argument("--env", default=None, choices=["testnet", "live"])
    parser.add_argument("--check", action="store_true",
                        help="probe connectivity and print filters; places no order")
    parser.add_argument("--market", default=None, choices=list(VALID_MARKETS),
                        help="spot wallet or USD-M perpetual futures; "
                             "defaults to $BINANCE_MARKET, else spot")
    parser.add_argument("--leverage", type=int, default=None,
                        help=f"futures only; capped at {int(MAX_LEVERAGE)}x")
    parser.add_argument("--margin-mode", dest="margin_mode", default=None,
                        choices=["CROSSED", "ISOLATED"],
                        help="futures only; defaults to $BINANCE_MARGIN_MODE, else CROSSED")
    parser.add_argument("--dry-run", dest="dry_run", metavar="ACTION",
                        help="run all guardrails for BUY/SHORT/SELL/HOLD without sending")
    parser.add_argument("--quantity", type=float, default=None)
    parser.add_argument("--stop-price", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    extra: dict[str, Any] = {}
    if resolve_market(args.market) == FUTURES_MARKET:
        extra = {"leverage": args.leverage, "margin_mode": args.margin_mode}

    try:
        executor = build_executor(
            symbol=args.symbol, env=args.env, market=args.market, **extra
        )
        if args.check:
            for key, value in executor.check_connectivity().items():
                print(f"{key:>20}: {value}")
            return 0
        if args.dry_run:
            result = executor.execute(
                args.dry_run, quantity=args.quantity,
                stop_price=args.stop_price, take_profit=args.take_profit,
                dry_run=True,
            )
            for key, value in result.items():
                print(f"{key:>20}: {value}")
            return 0
    except ExecutionRejected as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        print(_diagnose(str(exc), args), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(_diagnose(str(exc), args), file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
