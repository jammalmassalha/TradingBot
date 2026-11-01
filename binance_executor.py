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

from tradingagents.dataflows.vendors.binance_ohlcv import resolve_symbol

logger = logging.getLogger("binance_executor")

# --- Hardcoded safety envelope. Not read from config, by design. -------------

# Largest notional a single order may carry, whatever the strategy asks for.
MAX_TRADE_NOTIONAL_USD = Decimal("1000")
# ...and never more than this fraction of account equity, so a small account is
# not fully committed to one trade by the absolute cap above.
MAX_EQUITY_FRACTION = Decimal("0.20")

TESTNET_REST_URL = "https://testnet.binance.vision/api"

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
        report: dict[str, Any] = {"env": self.env, "symbol": self.symbol}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance execution bridge (guarded)")
    parser.add_argument("--symbol", default=None,
                        help="pair to trade; defaults to $TRADING_SYMBOL, else BTCUSDT")
    parser.add_argument("--env", default=None, choices=["testnet", "live"])
    parser.add_argument("--check", action="store_true",
                        help="probe connectivity and print filters; places no order")
    parser.add_argument("--dry-run", dest="dry_run", metavar="ACTION",
                        help="run all guardrails for BUY/SELL/HOLD without sending")
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

    try:
        executor = BinanceExecutor(symbol=args.symbol, env=args.env)
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
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
