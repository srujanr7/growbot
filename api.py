import time
import logging
import os
import datetime
import pandas as pd
from dotenv import load_dotenv
from growwapi import GrowwAPI as _GrowwSDK

load_dotenv()
logger = logging.getLogger(__name__)


class GrowwAPIWrapper:
    """
    Drop-in replacement for INDstocksAPI, wrapping the Groww Python SDK.

    Segment mapping:
      EQUITY     → CASH
      DERIVATIVE → FNO

    Product mapping:
      MIS    → MIS
      CNC    → CNC
      MTF    → CNC   (closest fit)
      MARGIN → NRML  (FNO overnight)

    scrip_code format used internally:
      NSE_RELIANCE  — equity
      NFO_NIFTY...  — futures/options
      NIDX_NIFTY    — index (LTP-only, no orders)

    security_id = bare trading_symbol for Groww orders (e.g. "RELIANCE")
    """

    def __init__(self):
        token = os.getenv("GROWW_ACCESS_TOKEN")
        if not token:
            raise RuntimeError(
                "GROWW_ACCESS_TOKEN not set in environment. "
                "Generate via GrowwAPI.get_access_token() and "
                "store in .env"
            )
        self._sdk           = _GrowwSDK(token)
        self._token_invalid = False
        logger.info("✅ Groww API initialised")

    # ── Compatibility shim ────────────────────────────────────

    def is_token_valid(self) -> bool:
        return not self._token_invalid

    def refresh_token(self):
        """
        Groww access tokens do not expire daily like INDstocks.
        This is a no-op kept for interface compatibility.
        On a 401, regenerate GROWW_ACCESS_TOKEN in .env and restart.
        """
        logger.info(
            "refresh_token() called — Groww tokens do not expire "
            "daily. No action needed."
        )
        return False

    # ── Segment / product converters ──────────────────────────

    @staticmethod
    def _groww_segment(segment: str) -> str:
        """EQUITY/DERIVATIVE → CASH/FNO"""
        return {
            "EQUITY":     "CASH",
            "DERIVATIVE": "FNO",
            "CASH":       "CASH",
            "FNO":        "FNO",
            "INDEX":      "CASH",   # indices live under CASH segment
        }.get((segment or "EQUITY").upper(), "CASH")

    @staticmethod
    def _api_product(trade_mode: str,
                     segment: str = "EQUITY") -> str:
        seg = (segment or "EQUITY").upper()
        if seg in ("DERIVATIVE", "FNO"):
            return "NRML"
        mode = (trade_mode or "MIS").upper()
        return {
            "MIS":    "MIS",
            "CNC":    "CNC",
            "MTF":    "CNC",
            "MARGIN": "NRML",
        }.get(mode, "MIS")

    # ── Market Data ───────────────────────────────────────────

    def get_ltp(self, scrip_codes: str) -> dict:
        """
        Get last traded prices for one or more instruments.

        scrip_codes: comma-separated, e.g. 'NSE_RELIANCE,NSE_INFY'

        Returns: { 'NSE_RELIANCE': 2500.5, ... }

        Uses SDK get_ltp() which accepts exchange_trading_symbols
        as a string (single) or tuple (multiple), up to 50 per call.
        Index instruments (NIDX_) are routed to CASH segment using
        their bare name as the symbol key.
        """
        codes = [c.strip() for c in scrip_codes.split(",") if c.strip()]

        cash_codes = [c for c in codes
                      if not c.startswith("NFO_")]
        fno_codes  = [c for c in codes
                      if c.startswith("NFO_")]

        result = {}

        # ── CASH (equity + index) ─────────────────────────────
        try:
            if cash_codes:
                symbols = (
                    tuple(cash_codes)
                    if len(cash_codes) > 1
                    else cash_codes[0]
                )
                resp = self._sdk.get_ltp(
                    segment=self._sdk.SEGMENT_CASH,
                    exchange_trading_symbols=symbols
                )
                # SDK returns { "NSE_RELIANCE": {"ltp": 2500.5}, ... }
                # OR { "NSE_RELIANCE": 2500.5 } depending on version.
                # Normalise both shapes.
                if resp:
                    for k, v in resp.items():
                        price = v["ltp"] if isinstance(v, dict) else v
                        result[k] = float(price)
        except Exception as e:
            self._handle_sdk_error(e, "get_ltp CASH")

        # ── FNO ───────────────────────────────────────────────
        try:
            if fno_codes:
                # Groww FNO LTP uses NSE_<trading_symbol> format too
                # e.g. NSE_NIFTY25JUNFUT
                fno_symbols = tuple(
                    c.replace("NFO_", "NSE_") for c in fno_codes
                ) if len(fno_codes) > 1 else fno_codes[0].replace(
                    "NFO_", "NSE_"
                )
                resp = self._sdk.get_ltp(
                    segment=self._sdk.SEGMENT_FNO,
                    exchange_trading_symbols=fno_symbols
                )
                if resp:
                    for k, v in resp.items():
                        price    = v["ltp"] if isinstance(v, dict) else v
                        # Re-key NSE_XXX → NFO_XXX to match callers
                        orig_key = "NFO_" + k.replace("NSE_", "")
                        result[orig_key] = float(price)
        except Exception as e:
            self._handle_sdk_error(e, "get_ltp FNO")

        return result

    def get_full_quote(self, scrip_codes: str) -> dict:
        """
        Returns OHLCV quote for one instrument.
        Maps Groww get_quote() response to the shape bot.py expects.
        """
        codes  = [c.strip() for c in scrip_codes.split(",")]
        code   = codes[0] if codes else ""

        segment = (
            self._sdk.SEGMENT_FNO if code.startswith("NFO_")
            else self._sdk.SEGMENT_CASH
        )
        trading_symbol = (
            code
            .replace("NSE_", "")
            .replace("NFO_", "")
            .replace("BSE_", "")
            .replace("NIDX_", "")
        )

        try:
            resp = self._sdk.get_quote(
                exchange=self._sdk.EXCHANGE_NSE,
                segment=segment,
                trading_symbol=trading_symbol
            )
            if resp:
                ohlc = resp.get("ohlc", {})
                return {
                    code: {
                        "live_price":            resp.get("last_price", 0),
                        "volume":                resp.get("volume", 0),
                        "day_change_percentage": resp.get(
                            "day_change_perc", 0
                        ),
                        "day_high": ohlc.get("high", 0),
                        "day_low":  ohlc.get("low",  0),
                        "open":     ohlc.get("open", 0),
                        "close":    ohlc.get("close", 0),
                    }
                }
        except Exception as e:
            self._handle_sdk_error(e, f"get_full_quote ({code})")
        return {}

    def get_historical(self, scrip_code: str, interval,
                       start_time, end_time) -> pd.DataFrame:
        """
        Fetch OHLCV candles and return a normalised DataFrame.

        interval: INDstocks string format e.g. '5minute', '15minute',
                  '60minute', '1day' — converted to interval_in_minutes.
        start_time / end_time: epoch milliseconds OR
                               "YYYY-MM-DD HH:mm:ss" strings.

        Candle response from Groww SDK (get_historical_candle_data):
          list of [timestamp_string, open, high, low, close, volume]
          where timestamp_string is ISO format: "2025-09-24T10:30:00"

        NOTE: get_historical_candle_data is deprecated by Groww.
        We keep it because get_historical_candles (the new API) uses
        groww_symbol format which requires instrument lookup first.
        Switch to get_historical_candles when you add instrument lookup.
        """
        interval_map = {
            "1minute":   1,   "2minute":  2,   "3minute":  3,
            "5minute":   5,   "10minute": 10,  "15minute": 15,
            "30minute":  30,  "60minute": 60,  "120minute": 120,
            "180minute": 180, "240minute": 240, "1day": 375,
        }
        interval_minutes = interval_map.get(str(interval), 15)

        trading_symbol = (
            scrip_code
            .replace("NSE_",  "")
            .replace("NFO_",  "")
            .replace("NIDX_", "")
            .replace("BSE_",  "")
        )
        segment = (
            self._sdk.SEGMENT_FNO
            if scrip_code.startswith("NFO_")
            else self._sdk.SEGMENT_CASH
        )

        # Accept both epoch-ms integers and "YYYY-MM-DD HH:mm:ss" strings.
        # SDK accepts either format for start_time / end_time.
        def _to_str(t):
            if isinstance(t, (int, float)):
                return datetime.datetime.fromtimestamp(
                    t / 1000
                ).strftime("%Y-%m-%d %H:%M:%S")
            return str(t)

        start_str = _to_str(start_time)
        end_str   = _to_str(end_time)

        try:
            resp = self._sdk.get_historical_candle_data(
                trading_symbol      = trading_symbol,
                exchange            = self._sdk.EXCHANGE_NSE,
                segment             = segment,
                start_time          = start_str,
                end_time            = end_str,
                interval_in_minutes = interval_minutes,
            )
        except Exception as e:
            self._handle_sdk_error(e, f"get_historical ({scrip_code})")
            return None

        if not resp:
            return None

        candles = resp.get("candles")
        if not candles:
            return None

        # Groww candle format:
        # [timestamp_iso_string, open, high, low, close, volume]
        # e.g. ["2025-09-24T10:30:00", 2500.0, 2520.0, 2490.0, 2510.0, 50000]
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Timestamp is an ISO string — do NOT use unit="s"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ── Orders ────────────────────────────────────────────────

    def place_order(self, txn_type: str, security_id: str,
                    qty: int, order_type: str = "MARKET",
                    limit_price: float = None,
                    exchange: str = "NSE",
                    segment: str = "EQUITY",
                    product: str = "CNC",
                    is_amo: bool = False) -> dict:
        """
        Place a standard order.

        security_id: bare trading_symbol string
                     e.g. 'RELIANCE', 'NIFTY25JUNFUT'
        NOTE: algo_id is intentionally NOT a parameter —
              Groww SDK place_order does not accept it.
        """
        groww_segment = self._groww_segment(segment)
        sdk_segment   = (
            self._sdk.SEGMENT_FNO
            if groww_segment == "FNO"
            else self._sdk.SEGMENT_CASH
        )
        sdk_exchange = (
            self._sdk.EXCHANGE_BSE
            if exchange.upper() == "BSE"
            else self._sdk.EXCHANGE_NSE
        )
        sdk_product_map = {
            "MIS":  self._sdk.PRODUCT_MIS,
            "CNC":  self._sdk.PRODUCT_CNC,
            "NRML": self._sdk.PRODUCT_NRML,
        }
        sdk_product = sdk_product_map.get(
            product.upper(), self._sdk.PRODUCT_MIS
        )
        sdk_order_type = (
            self._sdk.ORDER_TYPE_LIMIT
            if order_type.upper() == "LIMIT"
            else self._sdk.ORDER_TYPE_MARKET
        )
        sdk_txn = (
            self._sdk.TRANSACTION_TYPE_BUY
            if txn_type.upper() == "BUY"
            else self._sdk.TRANSACTION_TYPE_SELL
        )

        kwargs = dict(
            trading_symbol   = str(security_id),
            quantity         = int(qty),
            validity         = self._sdk.VALIDITY_DAY,
            exchange         = sdk_exchange,
            segment          = sdk_segment,
            product          = sdk_product,
            order_type       = sdk_order_type,
            transaction_type = sdk_txn,
        )
        if order_type.upper() == "LIMIT" and limit_price:
            kwargs["price"] = round(float(limit_price), 2)

        try:
            resp = self._sdk.place_order(**kwargs)
            if resp and resp.get("groww_order_id"):
                order_id = resp["groww_order_id"]
                logger.info(
                    f"✅ {txn_type} order placed: {order_id}"
                )
                return {
                    "order_id":     order_id,
                    "order_status": resp.get("order_status", "OPEN"),
                }
        except Exception as e:
            self._handle_sdk_error(e, "place_order")

        logger.error(
            f"❌ place_order failed: {txn_type} {security_id} "
            f"qty={qty}"
        )
        return None

    def cancel_order(self, order_id: str,
                     segment: str = "EQUITY") -> dict:
        sdk_segment = (
            self._sdk.SEGMENT_FNO
            if self._groww_segment(segment) == "FNO"
            else self._sdk.SEGMENT_CASH
        )
        try:
            resp = self._sdk.cancel_order(
                segment=sdk_segment,
                groww_order_id=order_id
            )
            return resp or {}
        except Exception as e:
            self._handle_sdk_error(e, "cancel_order")
            return {}

    def get_order_book(self) -> list:
        try:
            resp = self._sdk.get_order_list()
            return resp.get("order_list", []) if resp else []
        except Exception as e:
            self._handle_sdk_error(e, "get_order_book")
            return []

    # ── Portfolio ─────────────────────────────────────────────

    def get_holdings(self) -> list:
        try:
            resp = self._sdk.get_holdings_for_user()
            return resp.get("holdings", []) if resp else []
        except Exception as e:
            self._handle_sdk_error(e, "get_holdings")
            return []

    def get_positions(self, segment: str = "equity",
                      product: str = "cnc") -> dict:
        sdk_segment = (
            self._sdk.SEGMENT_FNO
            if segment.lower() in ("derivative", "fno")
            else self._sdk.SEGMENT_CASH
        )
        try:
            resp = self._sdk.get_positions_for_user(
                segment=sdk_segment
            )
            return (
                {"positions": resp.get("positions", [])}
                if resp else {}
            )
        except Exception as e:
            self._handle_sdk_error(e, "get_positions")
            return {}

    def get_funds(self) -> dict:
        """
        Returns a normalised funds dict that notifier.py and
        bot.py both consume.

        Groww get_available_margin_details() shape:
          equity_margin_details.mis_balance_available
          equity_margin_details.cnc_balance_available
          fno_margin_details.future_balance_available
          fno_margin_details.option_buy_balance_available
          fno_margin_details.option_sell_balance_available
          clear_cash  → opening / settlement cash
        """
        try:
            resp = self._sdk.get_available_margin_details()
            if not resp:
                return {}

            eq  = resp.get("equity_margin_details", {})
            fno = resp.get("fno_margin_details",    {})

            mis_bal  = float(eq.get("mis_balance_available",          0))
            cnc_bal  = float(eq.get("cnc_balance_available",          0))
            fut_bal  = float(fno.get("future_balance_available",      0))
            opt_buy  = float(fno.get("option_buy_balance_available",  0))
            opt_sell = float(fno.get("option_sell_balance_available", 0))
            cash     = float(resp.get("clear_cash", 0))

            return {
                "sod_balance":          cash,
                "funds_added":          0.0,
                "withdrawal_balance":   cash,
                "realized_pnl":         0.0,
                "unrealized_pnl":       0.0,
                "brokerage":            float(
                    resp.get("brokerage_and_charges", 0)
                ),
                "detailed_avl_balance": {
                    "eq_mis":     mis_bal,
                    "eq_cnc":     cnc_bal,
                    "eq_mtf":     cnc_bal,   # MTF not separate on Groww
                    "future":     fut_bal,
                    "option_buy": opt_buy,
                    "option_sell":opt_sell,
                },
            }
        except Exception as e:
            self._handle_sdk_error(e, "get_funds")
            return {}

    def get_true_balance(self, trade_mode: str = "MIS") -> float:
        """Returns the single real balance for the given trade mode."""
        funds = self.get_funds()
        if not funds:
            return 0.0
        avl = funds.get("detailed_avl_balance", {})
        mapping = {
            "MIS":    float(avl.get("eq_mis",  0.0)),
            "CNC":    float(avl.get("eq_cnc",  0.0)),
            "MTF":    float(avl.get("eq_mtf",  0.0)),
            "MARGIN": float(avl.get("future",  0.0)),
        }
        return mapping.get(
            trade_mode.upper(),
            float(avl.get("eq_mis", 0.0))
        )

    def check_margin(self, security_id: str, qty: int,
                     price: float, segment: str = "EQUITY",
                     txn_type: str = "BUY",
                     product: str = "CNC",
                     exchange: str = "NSE") -> dict:
        """
        Calculate margin required for an order.
        Returns dict with total_margin key.

        security_id MUST be a trading_symbol string e.g. "RELIANCE",
        NOT a numeric exchange_token.
        """
        groww_segment = self._groww_segment(segment)
        sdk_segment   = (
            self._sdk.SEGMENT_FNO
            if groww_segment == "FNO"
            else self._sdk.SEGMENT_CASH
        )
        sdk_product_map = {
            "MIS":  self._sdk.PRODUCT_MIS,
            "CNC":  self._sdk.PRODUCT_CNC,
            "NRML": self._sdk.PRODUCT_NRML,
        }
        sdk_product = sdk_product_map.get(
            product.upper(), self._sdk.PRODUCT_MIS
        )
        sdk_txn = (
            self._sdk.TRANSACTION_TYPE_BUY
            if txn_type.upper() == "BUY"
            else self._sdk.TRANSACTION_TYPE_SELL
        )

        order_details = [{
            "trading_symbol":  str(security_id),
            "transaction_type": sdk_txn,
            "quantity":        int(qty),
            "price":           round(float(price), 2),
            "order_type":      self._sdk.ORDER_TYPE_LIMIT,
            "product":         sdk_product,
            "exchange":        self._sdk.EXCHANGE_NSE,
        }]

        try:
            resp = self._sdk.get_order_margin_details(
                segment=sdk_segment,
                orders=order_details
            )
            if resp:
                total  = float(resp.get("total_requirement", 0))
                broker = float(resp.get("brokerage_and_charges", 0))
                return {
                    "total_margin": total,
                    "charges":      {"total_charges": broker},
                }
        except Exception as e:
            self._handle_sdk_error(e, "check_margin")
        return None

    def get_margin_per_unit(self, security_id: str,
                            price: float,
                            segment: str = "EQUITY",
                            txn_type: str = "BUY",
                            product: str = "MIS",
                            exchange: str = "NSE") -> float:
        margin = self.check_margin(
            security_id=security_id,
            qty=1,
            price=price,
            segment=segment,
            txn_type=txn_type,
            product=product,
            exchange=exchange,
        )
        if margin:
            total = float(margin.get("total_margin", 0.0))
            logger.debug(
                f"Margin/unit [{security_id}] @ ₹{price} "
                f"({product}): ₹{total:.2f}"
            )
            return total
        return 0.0

    # ── Error handling ────────────────────────────────────────

    def _handle_sdk_error(self, exc: Exception, context: str):
        err = str(exc)
        if "401" in err or "authentication" in err.lower():
            self._token_invalid = True
            logger.error(
                f"❌ Groww auth error in {context}. "
                "Regenerate GROWW_ACCESS_TOKEN."
            )
        elif "429" in err or "rate" in err.lower():
            logger.warning(
                f"Rate limited in {context} — backing off 2s"
            )
            time.sleep(2)
        else:
            logger.error(f"SDK error in {context}: {exc}")