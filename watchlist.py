import logging
import threading
import time
import os
import pandas as pd
from growwapi import GrowwAPI

logger = logging.getLogger(__name__)

SECTOR_MOMENTUM = {
    "BANK": 0, "FINANCE": 0, "NBFC": 0, "IT": 0,
    "AUTO": 0, "METAL": 0, "PHARMA": 0, "ENERGY": 0,
    "OIL_GAS": 0, "POWER": 0, "INFRA": 0, "REALTY": 0,
    "FMCG": 0, "CHEMICAL": 0, "CEMENT": 0, "TEXTILE": 0,
    "MEDIA": 0, "CONSUMER": 0, "DEFENCE": 0, "TELECOM": 0,
    "LOGISTICS": 0, "AGRI": 0, "OTHER": 0,
}


class FullMarketScanner:
    """
    Two-tier scanner using Groww SDK.

    Instrument dict keys used throughout the bot:
      scrip_code:      "NSE_RELIANCE"   (for api.get_ltp, get_historical)
      security_id:     "RELIANCE"       (for place_order, check_margin)
      ws_token:        "NSE:2885"       (for GrowwFeed subscription)
      exchange_token:  "2885"           (bare, for price_feed.get_ltp lookup)

    Index instruments:
      scrip_code:      "NIDX_NIFTY"
      security_id:     "NIFTY"
      ws_token:        "NSE:NIFTY"      (non-numeric → routed to subscribe_index_value)
      exchange_token:  "NIFTY"

    FNO instruments:
      scrip_code:      "NFO_NIFTY25JUNFUT"
      security_id:     "NIFTY25JUNFUT"
      ws_token:        "NSE:35241"      (numeric exchange_token)
      exchange_token:  "35241"

    Instruments CSV columns (from official docs):
      exchange, exchange_token, trading_symbol, groww_symbol, name,
      instrument_type, segment, series, isin, underlying_symbol,
      underlying_exchange_token, expiry_date, strike_price, lot_size,
      tick_size, freeze_quantity, is_reserved, buy_allowed, sell_allowed
    """

    TIER1_SHORTLIST  = 50
    MIN_PRICE_EQUITY = 50.0
    MIN_PRICE_FNO    = 1.0
    TIER1_INTERVAL   = 30 * 60   # seconds between tier1 refreshes
    BATCH_SIZE       = 50         # Groww LTP/OHLC max per call

    def __init__(self, api,
                 top_n_equity: int = 10,
                 top_n_fno: int    = 5,
                 top_n_index: int  = 2):
        self.api          = api
        self.top_n_equity = top_n_equity
        self.top_n_fno    = top_n_fno
        self.top_n_index  = top_n_index

        self.universe_equity = []
        self.universe_fno    = []
        self.universe_index  = []

        self.shortlist_equity = []
        self.shortlist_fno    = []
        self.shortlist_index  = []

        self._last_tier1 = 0
        self._lock       = threading.Lock()

        def _load_access_token():
            try:
                with open("access_token.txt") as f:
                    return f.read().strip()
            except:
                raise Exception("Access token missing. Run generate_token.py")

    # ── Universe loading ──────────────────────────────────────

    def load_universe(self):
        logger.info(
            "📥 Loading instrument universe from Groww SDK..."
        )
        try:
            df = pd.read_csv("instruments.csv")
            if df is None or df.empty:
                logger.error("Groww instruments DataFrame is empty")
                return

            # Normalise column names to uppercase for consistent access
            df.columns = df.columns.str.strip().str.upper()

            logger.info(
                f"Instruments loaded: {len(df)} rows | "
                f"columns: {df.columns.tolist()}"
            )

            self.universe_equity = self._parse_equity(df)
            self.universe_fno    = self._parse_fno(df)
            self.universe_index  = self._parse_index(df)

            logger.info(
                f"📦 Universe — "
                f"Equity: {len(self.universe_equity)} | "
                f"FNO: {len(self.universe_fno)} | "
                f"Index: {len(self.universe_index)}"
            )
        except Exception as e:
            logger.error(f"load_universe error: {e}", exc_info=True)

    def _parse_equity(self, df: pd.DataFrame) -> list:
        """
        Filter: exchange=NSE, segment=CASH, instrument_type=EQ

        Official CSV columns used:
          EXCHANGE, SEGMENT, INSTRUMENT_TYPE,
          TRADING_SYMBOL, EXCHANGE_TOKEN, TICK_SIZE, LOT_SIZE
        """
        try:
            mask = (
                (df["EXCHANGE"].str.upper()         == "NSE") &
                (df["SEGMENT"].str.upper()          == "CASH") &
                (df["INSTRUMENT_TYPE"].str.upper()  == "EQ")
            )
            filtered = df[mask].copy()
        except KeyError as e:
            logger.error(f"Equity parse: missing column {e}")
            return []

        instruments = []
        for _, row in filtered.iterrows():
            try:
                symbol     = str(row.get("TRADING_SYMBOL", "")).strip()
                exch_token = str(row.get("EXCHANGE_TOKEN", "")).strip()
                if not symbol or not exch_token:
                    continue
                instruments.append({
                    "name":            symbol,
                    "scrip_code":      f"NSE_{symbol}",
                    "security_id":     symbol,         # for orders
                    "ws_token":        f"NSE:{exch_token}",
                    "exchange_token":  exch_token,     # for price_feed lookup
                    "segment":         "EQUITY",
                    "product":         "CNC",
                    "exchange":        "NSE",
                    "instrument_type": "EQUITY",
                    "tick_size":       float(
                        row.get("TICK_SIZE", 0.05) or 0.05
                    ),
                    "lot_units":       int(
                        row.get("LOT_SIZE", 1) or 1
                    ),
                })
            except Exception as e:
                logger.warning(f"Skipping equity row: {e}")
        return instruments

    def _parse_fno(self, df: pd.DataFrame) -> list:
        """
        Filter: segment=FNO, instrument_type=FUT,
                expiry_date >= today (nearest expiry)
        """
        today = pd.Timestamp.today().normalize()

        try:
            expiry_col = pd.to_datetime(
                df["EXPIRY_DATE"], errors="coerce"
            )
            mask = (
                (df["SEGMENT"].str.upper()         == "FNO") &
                (df["INSTRUMENT_TYPE"].str.upper() == "FUT") &
                (expiry_col >= today)
            )
            filtered = df[mask].copy()
        except KeyError as e:
            logger.error(f"FNO parse: missing column {e}")
            return []

        instruments = []
        for _, row in filtered.iterrows():
            try:
                symbol     = str(row.get("TRADING_SYMBOL", "")).strip()
                underlying = str(
                    row.get("UNDERLYING_SYMBOL", symbol)
                ).strip()
                exch_token = str(row.get("EXCHANGE_TOKEN", "")).strip()
                if not symbol or not exch_token:
                    continue
                instruments.append({
                    "name":            f"{underlying} Future",
                    "scrip_code":      f"NFO_{symbol}",
                    "security_id":     symbol,
                    "ws_token":        f"NSE:{exch_token}",
                    "exchange_token":  exch_token,
                    "segment":         "DERIVATIVE",
                    "product":         "NRML",
                    "exchange":        "NSE",
                    "instrument_type": "FUTURES",
                    "lot_size":        int(
                        row.get("LOT_SIZE", 1) or 1
                    ),
                    "expiry":          str(
                        row.get("EXPIRY_DATE", "")
                    ),
                    "tick_size":       float(
                        row.get("TICK_SIZE", 0.05) or 0.05
                    ),
                })
            except Exception as e:
                logger.warning(f"Skipping FNO row: {e}")
        return instruments

    def _parse_index(self, df: pd.DataFrame) -> list:
        """
        Filter: segment=CASH, instrument_type=INDEX

        Index exchange_token from Groww instruments CSV is the
        string name (e.g. "NIFTY"), not a numeric token.
        This is what subscribe_index_value expects.
        """
        try:
            mask = (
                (df["SEGMENT"].str.upper()         == "CASH") &
                (df["INSTRUMENT_TYPE"].str.upper().isin(["INDEX", "INDICES"]))
            )
            filtered = df[mask].copy()
        except KeyError:
            return []

        instruments = []
        for _, row in filtered.iterrows():
            try:
                symbol     = str(row.get("TRADING_SYMBOL", "")).strip()
                # For indices, exchange_token in Groww CSV is the
                # string symbol name (e.g. "NIFTY", "BANKNIFTY")
                exch_token = str(
                    row.get("EXCHANGE_TOKEN", symbol)
                ).strip()
                if not symbol:
                    continue
                instruments.append({
                    "name":            symbol,
                    "scrip_code":      f"NIDX_{symbol}",
                    "security_id":     symbol,
                    "ws_token":        f"NSE:{exch_token}",
                    "exchange_token":  exch_token,
                    "segment":         "INDEX",
                    "product":         None,
                    "exchange":        "NSE",
                    "instrument_type": "INDEX",
                })
            except Exception as e:
                logger.warning(f"Skipping index row: {e}")
        return instruments

    # ── Sector detection ──────────────────────────────────────

    @staticmethod
    def _detect_sector(name: str) -> str:
        n = name.upper()
        if "BANK"    in n: return "BANK"
        if "FINANCE" in n: return "FINANCE"
        if "TECH"    in n or "INFY" in n: return "IT"
        if "AUTO"    in n or "MOTOR" in n: return "AUTO"
        if "STEEL"   in n or "METAL" in n: return "METAL"
        if "PHARMA"  in n: return "PHARMA"
        if "OIL"     in n or "GAS" in n: return "OIL_GAS"
        if "POWER"   in n: return "POWER"
        if "INFRA"   in n: return "INFRA"
        if "REALTY"  in n: return "REALTY"
        if "CEMENT"  in n: return "CEMENT"
        if "DEFENCE" in n: return "DEFENCE"
        if "TELECOM" in n: return "TELECOM"
        return "OTHER"

    # ── Tier 1 scan ───────────────────────────────────────────

    def tier1_scan(self):
        logger.info("🔍 Tier 1 scan starting...")
        start = time.time()

        equity_shortlist = self._score_batch_equity(
            self.universe_equity,
            min_price=self.MIN_PRICE_EQUITY,
            top_n=self.TIER1_SHORTLIST,
        )
        fno_shortlist = self._shortlist_fno(
            self.universe_fno,
            top_n=self.TIER1_SHORTLIST // 5,
        )
        index_shortlist = self.universe_index[:10]

        with self._lock:
            self.shortlist_equity = equity_shortlist
            self.shortlist_fno    = fno_shortlist
            self.shortlist_index  = index_shortlist
            self._last_tier1      = time.time()

        elapsed = time.time() - start
        logger.info(
            f"✅ Tier 1 done in {elapsed:.1f}s | "
            f"Equity: {len(equity_shortlist)} | "
            f"FNO: {len(fno_shortlist)}"
        )

    def _score_batch_equity(self, instruments: list,
                            min_price: float,
                            top_n: int) -> list:
        """
        Batch LTP + OHLC via Groww SDK (50 per call).
        Only instruments passing price and volatility filters are scored.
        """
        if not instruments:
            return []

        # NIFTY benchmark intraday change for relative strength score
        nifty_change = 0.0
        try:
            ohlc = self._sdk.get_ohlc(
                segment=self._sdk.SEGMENT_CASH,
                exchange_trading_symbols="NSE_NIFTY"
            )
            if ohlc and "NSE_NIFTY" in ohlc:
                d          = ohlc["NSE_NIFTY"]
                prev_close = float(d.get("close", 0) or 0)
                curr_open  = float(d.get("open",  0) or 0)
                if prev_close:
                    nifty_change = (
                        (curr_open - prev_close) / prev_close * 100
                    )
        except Exception:
            pass

        scored       = []
        total_batches = (
            len(instruments) + self.BATCH_SIZE - 1
        ) // self.BATCH_SIZE

        for i in range(0, len(instruments), self.BATCH_SIZE):
            batch     = instruments[i:i + self.BATCH_SIZE]
            batch_num = i // self.BATCH_SIZE + 1

            # SDK expects "NSE_SYMBOL" format as a tuple for multi
            symbols = tuple(
                f"NSE_{inst['security_id']}" for inst in batch
            )

            try:
                ltp_resp  = self._sdk.get_ltp(
                    segment=self._sdk.SEGMENT_CASH,
                    exchange_trading_symbols=symbols
                ) or {}
                ohlc_resp = self._sdk.get_ohlc(
                    segment=self._sdk.SEGMENT_CASH,
                    exchange_trading_symbols=symbols
                ) or {}

                hits = 0
                for inst in batch:
                    key = f"NSE_{inst['security_id']}"

                    # Normalise LTP response — SDK may return
                    # {"NSE_RELIANCE": 2500.5} or
                    # {"NSE_RELIANCE": {"ltp": 2500.5}}
                    raw_ltp = ltp_resp.get(key, 0)
                    price   = float(
                        raw_ltp["ltp"]
                        if isinstance(raw_ltp, dict)
                        else raw_ltp
                    )

                    ohlc  = ohlc_resp.get(key, {})
                    high  = float(ohlc.get("high", price) or price)
                    low   = float(ohlc.get("low",  price) or price)

                    if price <= min_price:
                        continue

                    atr_pct = (
                        (high - low) / price * 100
                        if price > 0 else 0
                    )
                    if atr_pct < 1.2:
                        continue

                    vol_score = min(atr_pct * 15, 40)
                    atr_score = min(atr_pct * 10, 40)
                    rs_score  = atr_pct * 10 - nifty_change * 10

                    sector = self._detect_sector(inst["name"])
                    if SECTOR_MOMENTUM.get(sector, 0) < 0:
                        continue

                    scored.append({
                        **inst,
                        "score":  round(
                            vol_score + atr_score + rs_score, 2
                        ),
                        "ltp":    price,
                        "volume": 0,
                        "change": atr_pct,
                    })
                    hits += 1

                logger.debug(
                    f"Equity batch {batch_num}/{total_batches}: "
                    f"{hits}/{len(batch)} scored"
                )

            except Exception as e:
                logger.error(
                    f"Equity batch {batch_num} error: {e}"
                )

            time.sleep(1.5)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    def _shortlist_fno(self, instruments: list,
                       top_n: int) -> list:
        """Prioritise index futures, then stock futures by lot size."""
        if not instruments:
            return []
        index_names = {
            "NIFTY", "BANKNIFTY", "FINNIFTY",
            "MIDCPNIFTY", "SENSEX", "BANKEX",
        }
        index_futs, stock_futs = [], []
        for inst in instruments:
            if any(n in inst["name"].upper() for n in index_names):
                index_futs.append(inst)
            else:
                stock_futs.append(inst)

        stock_futs.sort(
            key=lambda x: x.get("lot_size", 0), reverse=True
        )
        combined = index_futs + stock_futs
        return [
            {**inst, "score": 0.0, "ltp": 0.0,
             "volume": 0, "change": 0.0}
            for inst in combined[:top_n]
        ]

    # ── Public API ────────────────────────────────────────────

    def get_active(self) -> list:
        if time.time() - self._last_tier1 > self.TIER1_INTERVAL:
            threading.Thread(
                target=self.tier1_scan, daemon=True
            ).start()
        with self._lock:
            return (
                self.shortlist_equity[:self.top_n_equity] +
                self.shortlist_fno[:self.top_n_fno]
            )

    def get_ws_tokens(self) -> list:
        """Returns ws_token list for all shortlisted + index instruments."""
        with self._lock:
            all_insts = (
                self.shortlist_equity +
                self.shortlist_fno +
                self.shortlist_index
            )
            return [inst["ws_token"] for inst in all_insts]

    def get_cfg_from_token(self, token: str):
        """
        Returns instrument config using exchange_token.
        Used by signal_worker() when websocket tick arrives.
        """
        with self._lock:
            all_insts = (
                self.shortlist_equity +
                self.shortlist_fno +
                self.shortlist_index
            )
    
            for inst in all_insts:
                if inst["exchange_token"] == token:
                    return inst
    
        return None

    def start_background_refresh(self):
        def _loop():
            while True:
                try:
                    self.tier1_scan()
                except Exception as e:
                    logger.error(f"Tier 1 background error: {e}")
                time.sleep(self.TIER1_INTERVAL)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        logger.info("✅ Background market scanner started")

        return thread





