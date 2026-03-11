import logging
import threading
import time
import os
from growwapi import GrowwFeed, GrowwAPI

def _load_access_token():
    try:
        with open("access_token.txt") as f:
            return f.read().strip()
    except:
        raise Exception("Access token missing. Run generate_token.py")

logger = logging.getLogger(__name__)

class PriceFeed:
    """
    Streams live LTP via Groww SDK GrowwFeed.

    Correct SDK usage (from official docs):
      feed = GrowwFeed(groww_instance)          # pass GrowwAPI instance
      feed.subscribe_ltp(instruments_list,       # equity + FNO
                         on_data_received=cb)    # BLOCKING call
      feed.subscribe_index_value(index_list,     # indices separately
                                 on_data_received=cb)  # BLOCKING call

    Since subscribe_ltp is blocking and subscribe_index_value is
    blocking, we run them in separate daemon threads.

    instruments: list of ws_token strings from watchlist.py
      e.g. ["NSE:2885", "NSE:1594", "NSE:NIFTY"]
      Equity/FNO tokens have numeric exchange_token.
      Index tokens have string exchange_token (e.g. "NIFTY").

    LTP response shape (from feed.get_ltp()):
      {"ltp": {"NSE": {"CASH": {"2885": {"tsInMillis": ..., "ltp": 1419.1}}}}}

    Index value response shape (from feed.get_index_value()):
      {"NSE": {"CASH": {"NIFTY": {"tsInMillis": ..., "value": 24386.7}}}}

    latest_prices is keyed by bare exchange_token string e.g. "2885".
    """

    def __init__(self, instruments: list,
                 mode: str = "ltp",
                 on_tick=None):
        self.instruments   = instruments
        self.mode          = mode
        self.on_tick       = on_tick
        self.latest_prices = {}
        self._running      = False
        self._equity_feed  = None   # GrowwFeed for equity/FNO
        self._index_feed   = None   # GrowwFeed for indices
        self._lock         = threading.Lock()

    # ── Instrument parsing ────────────────────────────────────

    def _split_instruments(self):
        """
        Split ws_tokens into equity/FNO list and index list.

        ws_token format from watchlist.py:
          "NSE:2885"   — numeric token → equity or FNO
          "NSE:NIFTY"  — string token  → index

        Returns (equity_list, index_list) in SDK dict format.
        """
        equity_list = []
        index_list  = []

        for token in self.instruments:
            try:
                parts = token.split(":")
                if len(parts) != 2:
                    continue
                exchange, exch_token = parts[0], parts[1]

                if exch_token.isdigit():
                    equity_list.append({
                        "exchange": "NSE",
                        "segment": "CASH",
                        "exchange_token": exch_token,
                    })
                elif exch_token.isdigit():
                    # Numeric token → equity CASH
                    equity_list.append({
                        "exchange":       exchange,
                        "segment":        "CASH",
                        "exchange_token": exch_token,
                    })
                else:
                    # Non-numeric token → index
                    # e.g. "NIFTY", "BANKNIFTY"
                    index_list.append({
                        "exchange":       exchange,
                        "segment":        "CASH",
                        "exchange_token": exch_token,
                    })
            except Exception as e:
                logger.warning(f"Skipping ws_token {token}: {e}")

        return equity_list, index_list

    # ── Callbacks ─────────────────────────────────────────────

    def _on_equity_data(self, meta):
        """
        Callback for equity/FNO LTP ticks.
        SDK triggers this when data is received.
        Call feed.get_ltp() inside callback to get current data.

        Response shape:
          {"ltp": {"NSE": {"CASH": {"2885": {"tsInMillis": ...,
                                             "ltp": 1419.1}}}}}
        """
        try:
            ltp_data = self._equity_feed.get_ltp()
            ltp_map  = ltp_data.get("ltp", {})
            for exchange, seg_data in ltp_map.items():
                for segment, token_data in seg_data.items():
                    for exch_token, tick in token_data.items():
                        price = tick.get("ltp")
                        if price is None:
                            continue
                        price = float(price)
                        with self._lock:
                            self.latest_prices[exch_token] = price
                        if self.on_tick:
                            self.on_tick(exch_token, price)
        except Exception as e:
            logger.error(f"PriceFeed _on_equity_data error: {e}")

    def _on_index_data(self, meta):
        """
        Callback for index value ticks.
        Call feed.get_index_value() inside callback.

        Response shape:
          {"NSE": {"CASH": {"NIFTY": {"tsInMillis": ...,
                                      "value": 24386.7}}}}
        """
        try:
            idx_data = self._index_feed.get_index_value()
            for exchange, seg_data in idx_data.items():
                for segment, token_data in seg_data.items():
                    for exch_token, tick in token_data.items():
                        price = tick.get("value")
                        if price is None:
                            continue
                        price = float(price)
                        with self._lock:
                            self.latest_prices[exch_token] = price
                        if self.on_tick:
                            self.on_tick(exch_token, price)
        except Exception as e:
            logger.error(f"PriceFeed _on_index_data error: {e}")

    # ── Feed threads ──────────────────────────────────────────

    def _run_equity_feed(self, equity_list: list):
        """
        subscribe_ltp is a BLOCKING call — run in its own thread.
        Reconnects automatically on failure.
        """
        token = _load_access_token()
        while self._running:
            try:
                groww              = GrowwAPI(token)
                self._equity_feed  = GrowwFeed(groww)

                logger.info(
                    f"✅ Equity/FNO feed subscribing: "
                    f"{len(equity_list)} instruments"
                )
                # subscribe_ltp is BLOCKING — nothing after this runs
                self._equity_feed.subscribe_ltp(
                    equity_list,
                    on_data_received=self._on_equity_data
                )
                # If subscribe_ltp returns (connection dropped),
                # fall through to reconnect
                logger.warning(
                    "Equity feed subscribe_ltp returned — reconnecting"
                )

            except Exception as e:
                logger.error(f"Equity feed error: {e}")

            if self._running:
                logger.info("Equity feed reconnecting in 5s...")
                time.sleep(5)

    def _run_index_feed(self, index_list: list):
        """
        subscribe_index_value is a BLOCKING call — separate thread.
        """
        token = _load_access_token()
        while self._running:
            try:
                groww             = GrowwAPI(token)
                self._index_feed  = GrowwFeed(groww)

                logger.info(
                    f"✅ Index feed subscribing: "
                    f"{len(index_list)} indices"
                )
                # subscribe_index_value is BLOCKING
                self._index_feed.subscribe_index_value(
                    index_list,
                    on_data_received=self._on_index_data
                )
                logger.warning(
                    "Index feed subscribe_index_value returned — reconnecting"
                )

            except Exception as e:
                logger.error(f"Index feed error: {e}")

            if self._running:
                logger.info("Index feed reconnecting in 5s...")
                time.sleep(5)

    # ── Public interface ──────────────────────────────────────

    def start(self):
        self._running = True
        equity_list, index_list = self._split_instruments()

        if equity_list:
            t = threading.Thread(
                target=self._run_equity_feed,
                args=(equity_list,),
                daemon=True,
                name="equity-feed"
            )
            t.start()
            logger.info("Equity/FNO price feed thread started")

        if index_list:
            t = threading.Thread(
                target=self._run_index_feed,
                args=(index_list,),
                daemon=True,
                name="index-feed"
            )
            t.start()
            logger.info("Index price feed thread started")

        return None   # no single thread to return; callers don't use it

    def stop(self):
        self._running = False
        # GrowwFeed does not expose a graceful disconnect method —
        # daemon threads will die with the process.
        # Unsubscribe best-effort:
        equity_list, index_list = self._split_instruments()
        if self._equity_feed and equity_list:
            try:
                self._equity_feed.unsubscribe_ltp(equity_list)
            except Exception:
                pass
        if self._index_feed and index_list:
            try:
                self._index_feed.unsubscribe_index_value(index_list)
            except Exception:
                pass

    def restart(self):
        """Restart feeds with current instrument list (called by scheduler)."""
        logger.info("🔄 Restarting price feeds...")
        self.stop()
        time.sleep(2)
        self._equity_feed = None
        self._index_feed  = None
        self._running     = True
        self.start()

    def get_ltp(self, token: str) -> float:
        """
        token: bare exchange_token e.g. "2885" or "NIFTY"
        Returns cached price or None if not yet received.
        """
        with self._lock:
            return self.latest_prices.get(token)


class OrderFeed:
    """
    REST-only order status tracker using Groww SDK.
    Polls get_order_status() until terminal state.
    Interface identical to INDstocks OrderFeed.
    """

    def __init__(self, on_update=None):
        self.on_update    = on_update
        self.order_states = {}
        self._sdk         = None

    def start(self):
        token = _load_access_token()
        self._sdk = GrowwAPI(token)
        logger.info("✅ Order feed started (Groww REST polling)")

    def stop(self):
        pass

    def _fetch_order(self, order_id: str,
                     segment: str = "EQUITY") -> dict:
        if not self._sdk:
            return {}

        sdk_segment = (
            self._sdk.SEGMENT_FNO
            if segment.upper() in ("FNO", "DERIVATIVE")
            else self._sdk.SEGMENT_CASH
        )

        # Groww order status → INDstocks terminal names
        status_map = {
            "EXECUTED":  "SUCCESS",
            "CANCELLED": "CANCELLED",
            "REJECTED":  "FAILED",
            "EXPIRED":   "EXPIRED",
        }

        try:
            resp = self._sdk.get_order_status(
                groww_order_id=order_id,
                segment=sdk_segment,
            )
            if not resp:
                return {}

            raw_status = resp.get("order_status", "")
            status     = status_map.get(raw_status, raw_status)
            filled_qty = int(resp.get("filled_quantity", 0) or 0)

            avg_price = 0.0
            try:
                detail = self._sdk.get_order_detail(
                    groww_order_id=order_id,
                    segment=sdk_segment,
                )
                if detail:
                    avg_price = float(
                        detail.get("average_fill_price", 0) or 0
                    )
            except Exception:
                pass

            result = {
                "order_id":        order_id,
                "order_status":    status,
                "filled_quantity": filled_qty,
                "average_price":   avg_price,
            }

            prev = self.order_states.get(order_id, {})
            if prev.get("order_status") != status:
                self.order_states[order_id] = result
                logger.info(f"Order {order_id} → {status}")
                if self.on_update:
                    self.on_update(result)

            return result

        except Exception as e:
            logger.error(f"Order fetch error ({order_id}): {e}")
            return {}

    def wait_for_fill(self, order_id: str,
                      timeout: int = 30,
                      segment: str = "EQUITY") -> dict:
        terminal = {
            "SUCCESS", "FAILED", "CANCELLED",
            "PARTIALLY_EXECUTED", "EXPIRED", "ABORTED"
        }
        start = time.time()
        while time.time() - start < timeout:
            state = self._fetch_order(order_id, segment)
            if state.get("order_status") in terminal:
                return state
            time.sleep(1.0)
        logger.warning(f"wait_for_fill timeout for order {order_id}")

        return self.order_states.get(order_id, {})

