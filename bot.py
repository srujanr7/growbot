import sys
import time
import logging
import schedule
import threading
from datetime import datetime
from api import GrowwAPIWrapper
from risk import RiskManager
from queue import Queue
from websocket_feed import PriceFeed, OrderFeed
from ml.trainer import AutoTrainer
from ml.trade_memory import TradeMemory
from ml.risk_reward import TrailingStop
from notifier import TelegramNotifier
from watchlist import FullMarketScanner
from position_manager import PositionManager
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import requests as _req
try:
    _ip = _req.get("https://api.ipify.org", timeout=5).text
    logger.info(f"🌐 Outbound IP: {_ip}")
except Exception:
    logger.warning("Could not detect outbound IP")

# ── Runtime config ────────────────────────────────────────────
TRADE_MODE     = "MIS"
TRADE_CAPITAL  = None
MAX_EQUITY_POS = config.MAX_EQUITY_POS
MAX_FNO_POS    = config.MAX_FNO_POS
MIN_CONFIDENCE = config.TRADE_CONFIDENCE

# ── Segment / Exchange / Instrument filters ───────────────────
ACTIVE_SEGMENT    = "BOTH"
ACTIVE_EXCHANGE   = "BOTH"
ACTIVE_INSTRUMENT = "ALL"

# ── Core services ─────────────────────────────────────────────
api = GrowwAPIWrapper()
risk = RiskManager(
    max_position_pct     = config.MAX_POSITION_PCT,
    daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT
)
notifier    = TelegramNotifier()
scanner     = FullMarketScanner(
    api, top_n_equity=10, top_n_fno=5, top_n_index=2
)
posmgr      = PositionManager(
    max_equity     = MAX_EQUITY_POS,
    max_fno        = MAX_FNO_POS,
    min_confidence = MIN_CONFIDENCE
)
trade_memory = TradeMemory()

# ── India VIX ─────────────────────────────────────────────────
_vix_data = {
    "vix":        15.0,
    "multiplier": 1.0,
    "safe":       True,
    "half_size":  False,
    "stop_trade": False,
}


def get_india_vix() -> dict:
    try:
        r = _req.get(
            "https://www.nseindia.com/api/allIndices",
            headers={"User-Agent": "Mozilla/5.0",
                     "Accept":     "application/json"},
            timeout=5,
        )
        vix = next(
            (float(i["last"])
             for i in r.json().get("data", [])
             if i.get("index") == "INDIA VIX"),
            15.0,
        )
        return {
            "vix":        vix,
            "safe":       vix < 20,
            "half_size":  20 <= vix < 25,
            "stop_trade": vix >= 25,
            "multiplier": (
                0.0 if vix >= 25 else
                0.5 if vix >= 20 else
                1.0
            ),
        }
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e} — assuming safe")
        return {
            "vix": 15.0, "safe": True,
            "half_size": False, "stop_trade": False,
            "multiplier": 1.0,
        }


# ── ML Trainers ───────────────────────────────────────────────
trainers      = {}
_trainer_lock = threading.Lock()


def get_trainer(cfg: dict) -> AutoTrainer:
    key = cfg["scrip_code"]
    if key not in trainers:
        with _trainer_lock:
            if key not in trainers:
                trainers[key] = AutoTrainer(
                    scrip_code   = cfg["scrip_code"],
                    security_id  = cfg["security_id"],
                    interval     = config.CANDLE_INTERVAL,
                    retrain_days = config.RETRAIN_DAYS
                )
                trainers[key].start_schedule(
                    retrain_time=config.RETRAIN_TIME
                )
                logger.info(
                    f"🧠 Trainer created: {cfg['name']} | "
                    f"interval={trainers[key].interval}"
                )
                time.sleep(1.0)
    return trainers[key]


# ── WebSocket feeds ───────────────────────────────────────────

def on_price_tick(token, ltp):
    event_queue.put((token, ltp))


def on_order_update(order):
    status   = order.get("order_status", "")
    order_id = order.get("order_id", "")
    logger.info(f"📬 {order_id} → {status}")
    if status == "FAILED":
        notifier.error("OrderFailed", f"Order {order_id} rejected")


price_feed = PriceFeed(
    instruments=[], mode="ltp", on_tick=on_price_tick
)
order_feed = OrderFeed(on_update=on_order_update)

event_queue = Queue()


# ── Helpers ───────────────────────────────────────────────────

def signal_worker():

    while True:

        try:

            token, ltp = event_queue.get()

            cfg = scanner.get_cfg_from_token(token)

            if not cfg:
                continue

            df = get_candles(cfg)

            if df is None:
                continue

            trainer = get_trainer(cfg)

            result = trainer.get_signal(df)

            if result["signal"] == "BUY":
                process_entry(cfg, result, ltp)

        except Exception as e:
            logger.error(f"Signal worker error: {e}")

def is_market_open() -> bool:
    now = datetime.now().strftime("%H:%M")
    return config.MARKET_OPEN <= now <= config.MARKET_CLOSE


def _ws_token_key(cfg: dict) -> str:
    token = cfg.get("ws_token")
    if not token:
        return None

    parts = token.split(":")
    if len(parts) != 2:
        return None

    return parts[1]


def _get_ltp_for_cfg(cfg: dict) -> float:
    """
    Gets LTP, trying WebSocket cache first,
    falling back to REST API.
    """
    ltp = price_feed.get_ltp(_ws_token_key(cfg))
    if ltp is not None:
        return float(ltp)
    # REST fallback
    result = api.get_ltp(cfg["scrip_code"])
    price  = result.get(cfg["scrip_code"]) if result else None
    return float(price) if price is not None else None


CANDLE_CACHE = {}


def get_candles(cfg: dict):
    key = cfg["scrip_code"]

    if key in CANDLE_CACHE:
        ts, data = CANDLE_CACHE[key]
        if time.time() - ts < 60:
            return data

    trainer  = trainers.get(cfg["scrip_code"])
    interval = trainer.interval if trainer else config.CANDLE_INTERVAL
    if interval == "auto":
        interval = "5minute"

    end_time = int(time.time() * 1000)

    if interval in [
        "1minute", "2minute", "3minute", "4minute", "5minute",
        "10minute", "15minute", "30minute"
    ]:
        start_time = end_time - (7 * 24 * 60 * 60 * 1000)
    elif interval in [
        "60minute", "120minute", "180minute", "240minute"
    ]:
        start_time = end_time - (14 * 24 * 60 * 60 * 1000)
    else:
        start_time = end_time - (7 * 24 * 60 * 60 * 1000)

    time.sleep(0.25)

    df = api.get_historical(
        scrip_code = cfg["scrip_code"],
        interval   = interval,
        start_time = start_time,
        end_time   = end_time,
    )

    if df is None:
        return None

    logger.debug(f"{cfg['name']} candles loaded: {len(df)}")
    CANDLE_CACHE[key] = (time.time(), df)
    return df


def get_balance(trade_mode: str = None) -> float:
    mode = trade_mode or TRADE_MODE
    raw  = api.get_true_balance(mode)
    return min(raw, TRADE_CAPITAL) if TRADE_CAPITAL else raw


def get_all_balances() -> dict:
    funds = api.get_funds()
    return funds if funds else {}


def _interval_minutes(interval: str) -> int:
    mapping = {
        "1minute": 1, "2minute": 2, "3minute": 3,
        "5minute": 5, "10minute": 10, "15minute": 15,
        "30minute": 30, "60minute": 60, "1day": 375,
    }
    return mapping.get(interval, 5)


def _effective_segment(cfg: dict) -> str:
    return cfg.get("segment", "EQUITY")


def _passes_filters(cfg: dict) -> tuple:
    seg       = cfg.get("segment",         "EQUITY")
    exchange  = cfg.get("exchange",        "NSE")
    inst_type = cfg.get("instrument_type", "EQUITY")

    if ACTIVE_SEGMENT == "EQUITY" and seg != "EQUITY":
        return False, f"segment filter (EQUITY only, got {seg})"
    if ACTIVE_SEGMENT == "FNO" and seg not in ("DERIVATIVE", "FNO"):
        return False, f"segment filter (FNO only, got {seg})"
    if ACTIVE_EXCHANGE == "NSE" and exchange != "NSE":
        return False, f"exchange filter (NSE only, got {exchange})"
    if ACTIVE_EXCHANGE == "BSE" and exchange != "BSE":
        return False, f"exchange filter (BSE only, got {exchange})"
    if ACTIVE_INSTRUMENT != "ALL" and inst_type != ACTIVE_INSTRUMENT:
        return False, (
            f"instrument filter ({ACTIVE_INSTRUMENT} only, "
            f"got {inst_type})"
        )
    return True, ""


# ── SL/TP Monitor ─────────────────────────────────────────────

def _monitor_open_positions(active: list):
    for scrip_code, pos in list(posmgr.positions.items()):
        meta = pos.get("signal_meta", {})

        cfg = next(
            (s for s in active if s["scrip_code"] == scrip_code),
            None
        )
        if not cfg:
            continue

        ltp = _get_ltp_for_cfg(cfg)
        if ltp is None:
            logger.warning(
                f"LTP missing for {cfg['name']} — "
                "forcing REST fetch"
            )
            result = api.get_ltp(cfg["scrip_code"])
            ltp = (
                result.get(cfg["scrip_code"]) if result else None
            )

        if not ltp:
            logger.warning(
                f"Still no LTP for {cfg['name']} — "
                "skipping exit check"
            )
            continue

        ltp         = float(ltp)
        entry       = pos["entry"]
        stop_loss   = meta.get("stop_loss",   entry * 0.99)
        take_profit = meta.get("take_profit", entry * 1.02)
        trailing    = meta.get("trailing")

        current_dd = max(0.0, (entry - ltp) / entry * 100)
        meta["max_drawdown"] = max(
            meta.get("max_drawdown", 0.0), current_dd
        )

        trail_exit = False
        locked_pnl = 0.0
        if trailing:
            t_info     = trailing.update(ltp)
            trail_exit = t_info["exit_now"]
            locked_pnl = t_info.get("locked_pnl", 0.0)

        hard_sl  = ltp <= stop_loss
        tp_hit   = ltp >= take_profit

        opened_at = datetime.fromisoformat(
            pos.get("opened_at", datetime.now().isoformat())
        )
        held_minutes = (
            datetime.now() - opened_at
        ).total_seconds() / 60
        time_exit = held_minutes > 120

        if hard_sl or tp_hit or trail_exit or time_exit:
            reason = (
                "HARD_STOP_LOSS" if hard_sl    else
                "TAKE_PROFIT"    if tp_hit     else
                "TRAILING_STOP"  if trail_exit else
                "TIME_EXIT_2HR"
            )
            pnl_pct = (ltp - entry) / entry * 100
            logger.info(
                f"🚨 {reason}: {cfg['name']} | "
                f"LTP=₹{ltp} Entry=₹{entry} "
                f"PnL={pnl_pct:+.2f}%"
                + (f" Locked={locked_pnl:+.2f}%"
                   if trail_exit else "")
            )
            process_exit(cfg, ltp, reason=reason)


# ── Core cycle ────────────────────────────────────────────────

def run_cycle():
    if not is_market_open():
        logger.info("⏸ Market closed — skipping cycle.")
        return
    if notifier.bot_paused:
        logger.info("Bot paused via Telegram.")
        return
    if not api.is_token_valid():
        logger.warning("⚠️ Token invalid — skipping cycle.")
        return
    if _vix_data["stop_trade"]:
        logger.info(
            f"🔴 VIX {_vix_data['vix']:.1f} ≥ 25 — trading paused"
        )
        return

    active = scanner.get_active()
    logger.info(
        f"📊 Active: {len(active)} | "
        f"Segment={ACTIVE_SEGMENT} | "
        f"Exchange={ACTIVE_EXCHANGE} | "
        f"Instrument={ACTIVE_INSTRUMENT}"
    )
    if not active:
        logger.warning("No active instruments.")
        return

    _monitor_open_positions(active)

    signals      = []
    exit_signals = []

    def scan_instrument(cfg):
        try:
            passes, filter_reason = _passes_filters(cfg)
            if not passes:
                logger.debug(
                    f"[{cfg['name']}] Filtered — {filter_reason}"
                )
                return

            if posmgr.has_position(cfg["scrip_code"]):
                ltp = _get_ltp_for_cfg(cfg)
                if ltp:
                    df = get_candles(cfg)
                    if df is not None and len(df) >= 30:
                        result = get_trainer(cfg).get_signal(df)
                        if result["signal"] == "SELL":
                            exit_signals.append({
                                "cfg":    cfg,
                                "result": result,
                                "ltp":    ltp,
                                "reason": "MODEL_SELL_SIGNAL",
                            })
                return

            ltp = _get_ltp_for_cfg(cfg)
            if not ltp:
                return

            df = get_candles(cfg)
            if df is None or len(df) < 30:
                time.sleep(1)
                df = get_candles(cfg)

            if df is None or len(df) < 30:
                logger.warning(
                    f"[{cfg['name']}] Insufficient candles, skipping."
                )
                return

            result = get_trainer(cfg).get_signal(df)
            if result["signal"] == "BUY":
                signals.append({
                    "cfg":    cfg,
                    "result": result,
                    "ltp":    ltp,
                })

        except Exception as e:
            logger.error(f"Scan error [{cfg['name']}]: {e}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(scan_instrument, active)

    for entry in exit_signals:
        process_exit(
            entry["cfg"],
            entry["ltp"],
            reason=entry.get("reason", "MODEL_SELL_SIGNAL"),
        )

    ranked = posmgr.rank_signals(signals)
    logger.info(
        f"📊 {len(ranked)} actionable | "
        f"EQ={posmgr.equity_slots_free()} free | "
        f"FNO={posmgr.fno_slots_free()} free"
    )

    for entry in ranked:
        can_enter, reason = posmgr.can_enter(
            entry["cfg"]["segment"],
            entry["result"]["confidence"],
        )
        if not can_enter:
            logger.info(
                f"[{entry['cfg']['name']}] Skipped: {reason}"
            )
            continue
        process_entry(entry["cfg"], entry["result"], entry["ltp"])

    status = posmgr.status()
    logger.info(
        f"📁 Open={status['open_positions']} | "
        f"EQ={status['equity_open']}/{MAX_EQUITY_POS} | "
        f"FNO={status['fno_open']}/{MAX_FNO_POS} | "
        f"WR={status['win_rate']:.1f}%"
    )


# ── Entry ─────────────────────────────────────────────────────

def process_entry(cfg: dict, result: dict, ltp: float):
    passes, filter_reason = _passes_filters(cfg)
    if not passes:
        logger.info(
            f"[{cfg['name']}] Entry blocked — {filter_reason}"
        )
        return

    balance = get_balance()

    if not risk.can_trade(balance):
        notifier.kill_switch(
            risk.daily_pnl, risk.get_daily_limit(balance)
        )
        return

    limit = risk.get_daily_limit(balance)
    if risk.daily_pnl < -(limit * 0.8):
        notifier.risk_warning(risk.daily_pnl, limit, balance)

    segment = _effective_segment(cfg)
    product = api._api_product(TRADE_MODE, segment)

    margin_per_unit = api.get_margin_per_unit(
        security_id = cfg["security_id"],   # trading_symbol string
        price       = ltp,
        segment     = segment,
        txn_type    = "BUY",
        product     = product,
        exchange    = cfg.get("exchange", "NSE"),
    )

    if margin_per_unit <= 0:
        logger.warning(
            f"[{cfg['name']}] Margin API returned 0, "
            "falling back to raw price for sizing"
        )
        margin_per_unit = ltp

    qty = posmgr.position_size(
        balance    = balance,
        price      = margin_per_unit,
        segment    = segment,
        confidence = result["confidence"],
        kelly_pct  = result.get("kelly_pct", 0.0),
        rr_ratio   = result.get("rr_ratio",  0.0),
    )
    qty = risk.apply_per_trade_limit(qty, margin_per_unit)

    if _vix_data["multiplier"] < 1.0:
        original_qty = qty
        qty = max(1, int(qty * _vix_data["multiplier"]))
        logger.info(
            f"⚠️ VIX {_vix_data['vix']:.1f} — "
            f"qty scaled {original_qty} → {qty} "
            f"(×{_vix_data['multiplier']})"
        )

    margin = api.check_margin(
        security_id = cfg["security_id"],
        qty         = qty,
        price       = ltp,
        segment     = segment,
        txn_type    = "BUY",
        product     = product,
        exchange    = cfg.get("exchange", "NSE"),
    )
    if margin:
        required = float(margin.get("total_margin", 0))
        charges  = float(
            margin.get("charges", {}).get("total_charges", 0)
        )
        if required > balance:
            notifier.insufficient_margin(
                cfg["name"], required, balance
            )
            logger.warning(
                f"[{cfg['name']}] Margin check failed: "
                f"need ₹{required:,.0f}, have ₹{balance:,.0f}"
            )
            return
        logger.info(
            f"[{cfg['name']}] Margin OK: "
            f"₹{required:,.0f} / ₹{balance:,.0f} available | "
            f"Charges: ₹{charges:.2f}"
        )

    # NOTE: no algo_id — Groww SDK place_order does not accept it
    order = api.place_order(
        txn_type    = "BUY",
        security_id = cfg["security_id"],
        qty         = qty,
        order_type  = "LIMIT",
        limit_price = round(
            ltp * (1 + config.LIMIT_ORDER_SLIPPAGE), 2
        ),
        segment     = segment,
        product     = product,
        exchange    = cfg.get("exchange", "NSE"),
    )

    if order:
        fill      = order_feed.wait_for_fill(
            order["order_id"],
            timeout = config.ORDER_FILL_TIMEOUT,
            segment = segment,
        )
        avg_price = fill.get("average_price") or ltp

        if fill.get("order_status") == "FAILED":
            notifier.order_failed(
                "BUY", cfg["name"], qty,
                fill.get("reason", "Unknown"),
            )
            return

        stop_loss   = result.get("stop_loss")  or round(
            avg_price * 0.99, 2
        )
        take_profit = result.get("take_profit") or round(
            avg_price * 1.02, 2
        )
        atr_val     = result.get("atr") or avg_price * 0.01

        trailing = TrailingStop(
            entry            = avg_price,
            atr              = atr_val,
            trail_multiplier = 1.5,
        )

        sl_pct = result.get("sl_pct") or (
            (avg_price - stop_loss) / avg_price * 100
        )
        tp_pct = result.get("tp_pct") or (
            (take_profit - avg_price) / avg_price * 100
        )

        posmgr.open_position(
            scrip_code  = cfg["scrip_code"],
            name        = cfg["name"],
            segment     = segment,
            qty         = qty,
            entry_price = avg_price,
            order_id    = order["order_id"],
            signal_meta = {
                **result,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "trailing":     trailing,
                "max_drawdown": 0.0,
            },
        )

        notifier.trade_executed(
            side        = "BUY",
            name        = cfg["name"],
            segment     = segment,
            qty         = qty,
            price       = avg_price,
            confidence  = result["confidence"],
            xgb         = result.get("xgb",     ""),
            rl          = result.get("rl",      ""),
            lgbm        = result.get("lgbm",    ""),
            lstm        = result.get("lstm",    ""),
            pattern     = result.get("pattern", "NONE"),
            regime      = result.get("regime",  ""),
            stop_loss   = stop_loss,
            take_profit = take_profit,
            rr_ratio    = result.get("rr_ratio", 0),
            sl_pct      = sl_pct,
            tp_pct      = tp_pct,
        )


# ── Exit ──────────────────────────────────────────────────────

def process_exit(cfg: dict, ltp: float,
                 reason: str = "MODEL_SELL_SIGNAL"):
    pos = posmgr.positions.get(cfg["scrip_code"])
    if not pos:
        return

    segment = pos["segment"]
    product = api._api_product(TRADE_MODE, segment)

    # NOTE: no algo_id
    order = api.place_order(
        txn_type    = "SELL",
        security_id = cfg["security_id"],
        qty         = pos["qty"],
        order_type  = "MARKET",
        segment     = segment,
        product     = product,
        exchange    = cfg.get("exchange", "NSE"),
    )

    if order:
        fill       = order_feed.wait_for_fill(
            order["order_id"],
            timeout = config.ORDER_FILL_TIMEOUT,
            segment = segment,
        )
        exit_price = fill.get("average_price") or ltp
        closed     = posmgr.close_position(cfg["scrip_code"], exit_price)
        pnl        = closed.get("pnl", 0)
        pnl_pct    = (
            (exit_price - pos["entry"]) / pos["entry"] * 100
            if pos["entry"] > 0 else 0
        )
        risk.update_pnl(pnl)

        meta = pos.get("signal_meta", {})

        opened_at = datetime.fromisoformat(
            pos.get("opened_at", datetime.now().isoformat())
        )
        held_minutes = (
            datetime.now() - opened_at
        ).total_seconds() / 60
        interval_key = (
            trainers[cfg["scrip_code"]].interval
            if cfg["scrip_code"] in trainers
            else "5minute"
        )
        held_candles = max(
            1, int(held_minutes / _interval_minutes(interval_key))
        )

        trainer = trainers.get(cfg["scrip_code"])
        if trainer:
            trainer.record_trade_outcome(
                signal           = "BUY",
                entry            = pos["entry"],
                exit_price       = exit_price,
                xgb_was          = meta.get("xgb",  "HOLD"),
                rl_was           = meta.get("rl",   "HOLD"),
                sentiment        = meta.get("sentiment", 0.0),
                held_candles     = held_candles,
                max_drawdown_pct = meta.get("max_drawdown", 0.0),
                signal_meta      = meta,
            )

        trade_memory.record({
            "scrip_code":       cfg["scrip_code"],
            "name":             cfg["name"],
            "segment":          segment,
            "side":             "BUY",
            "entry":            pos["entry"],
            "exit":             exit_price,
            "qty":              pos["qty"],
            "pnl":              pnl,
            "pnl_pct":          round(pnl_pct, 3),
            "profitable":       pnl >= 0,
            "held_candles":     held_candles,
            "max_drawdown_pct": round(meta.get("max_drawdown", 0.0), 3),
            "exit_reason":      reason,
            "stop_loss":        meta.get("stop_loss"),
            "take_profit":      meta.get("take_profit"),
            "confidence":       meta.get("confidence", 0),
            "xgb":              meta.get("xgb",     ""),
            "lgbm":             meta.get("lgbm",    ""),
            "lstm":             meta.get("lstm",    ""),
            "rl":               meta.get("rl",      ""),
            "pattern":          meta.get("pattern", ""),
            "regime":           meta.get("regime",  ""),
            "sentiment":        meta.get("sentiment", 0),
            "rr_ratio":         meta.get("rr_ratio",  0),
            "ev_quality":       meta.get("ev_quality", ""),
            "interval":         interval_key,
            "vix":              _vix_data["vix"],
            "exchange":         cfg.get("exchange", ""),
            "instrument_type":  cfg.get("instrument_type", "EQUITY"),
        })

        notifier.trade_closed(
            name       = cfg["name"],
            qty        = pos["qty"],
            entry      = pos["entry"],
            exit_price = exit_price,
            pnl        = pnl,
            daily_pnl  = risk.daily_pnl,
            reason     = reason,
        )


# ── Scheduled jobs ────────────────────────────────────────────

def square_off_all():
    logger.info("⏰ Auto square-off triggered")
    notifier.squareoff_alert(posmgr.positions)
    for scrip_code, pos in list(posmgr.positions.items()):
        cfg = next(
            (s for s in
             scanner.universe_equity + scanner.universe_fno
             if s["scrip_code"] == scrip_code),
            None
        )
        if cfg:
            ltp = _get_ltp_for_cfg(cfg) or pos["entry"]
            process_exit(cfg, ltp, reason="AUTO_SQUAREOFF")


def daily_reset():
    global _vix_data
    risk.reset_daily()

    _vix_data = get_india_vix()
    logger.info(
        f"🌡️ India VIX: {_vix_data['vix']:.1f} | "
        f"{'✅ Safe' if _vix_data['safe'] else '⚠️ Caution' if _vix_data['half_size'] else '🔴 STOP'}"
    )

    if _vix_data["stop_trade"]:
        notifier.send(
            f"🔴 <b>VIX ALERT — Trading Paused Today</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"India VIX : {_vix_data['vix']:.1f}\n"
            f"Threshold : 25.0\n"
            f"Action    : Bot will NOT trade today\n\n"
            f"Send /resume to override."
        )
        notifier.bot_paused = True
    elif _vix_data["half_size"]:
        notifier.send(
            f"⚠️ <b>VIX WARNING — Reduced Sizes</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"India VIX  : {_vix_data['vix']:.1f}\n"
            f"Action     : All position sizes halved today"
        )
    else:
        logger.info(
            f"VIX normal ({_vix_data['vix']:.1f}) — "
            "full size trading"
        )
    logger.info("🔄 Daily counters reset")


def send_daily_summary():
    balance = get_balance()
    status  = posmgr.status()
    notifier.daily_summary(
        total_pnl = risk.daily_pnl,
        trades    = risk.trades_today,
        wins      = status["win_count"],
        balance   = balance,
        vix       = _vix_data["vix"],
    )


def refresh_ws_subscriptions():
    """
    Restart price feeds with the current scanner shortlist.
    Called every 5 minutes by scheduler.
    Uses price_feed.restart() — NOT price_feed.ws (which doesn't exist).
    """
    new_tokens = scanner.get_ws_tokens()
    price_feed.instruments = new_tokens
    price_feed.restart()
    logger.info(
        f"🔄 WS feeds restarted: {len(new_tokens)} instruments"
    )


# ── Boot ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Booting algo bot — Full Market Mode...")

    scanner.load_universe()
    scanner.tier1_scan()

    price_feed.instruments = scanner.get_ws_tokens()
    price_feed.start()
    order_feed.start()

    threading.Thread(target=signal_worker, daemon=True).start()
    time.sleep(2)

    _vix_data = get_india_vix()
    logger.info(
        f"🌡️ Boot VIX check: {_vix_data['vix']:.1f} | "
        f"{'✅ Safe' if _vix_data['safe'] else '⚠️ Caution' if _vix_data['half_size'] else '🔴 STOP TRADING'}"
    )
    if _vix_data["stop_trade"]:
        logger.warning(
            "🔴 VIX >= 25 at boot — bot will not trade today"
        )
        notifier.send(
            f"🔴 <b>Bot started but VIX = {_vix_data['vix']:.1f}</b>\n"
            f"Trading paused. Send /resume to override."
        )
        notifier.bot_paused = True

    scanner.start_background_refresh()
    notifier.start_command_listener(bot_ref=sys.modules[__name__])

    schedule.every(config.CYCLE_INTERVAL_SECONDS).seconds.do(run_cycle)
    schedule.every(5).minutes.do(refresh_ws_subscriptions)
    schedule.every().day.at(config.DAILY_RESET_AT).do(daily_reset)
    schedule.every().day.at(config.SQUAREOFF_AT).do(square_off_all)
    schedule.every().day.at("15:30").do(send_daily_summary)
    schedule.every().day.at("00:01").do(scanner.load_universe)

    notifier.bot_started()

    logger.info("✅ All systems running")
    logger.info("⏳ Waiting 30 seconds for initial model warmup...")
    time.sleep(30)

    logger.info("🚀 Running first trading cycle...")
    run_cycle()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            notifier.bot_stopped("Manual shutdown")
            price_feed.stop()
            order_feed.stop()
            notifier.stop_listener()
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            notifier.error("MainLoopError", str(e))

            time.sleep(5)
