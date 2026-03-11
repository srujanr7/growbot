import requests
import logging
import threading
import time
from datetime import datetime, timedelta
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramNotifier:

    BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    def __init__(self):
        self.chat_id      = TELEGRAM_CHAT_ID
        self.bot_paused   = False
        self._last_update = 0
        self._running     = False
        self._bot_ref     = None

    def send(self, message: str):
        self._send(message)

    def _send(self, message: str):
        def _do():
            try:
                r = requests.post(
                    f"{self.BASE_URL}/sendMessage",
                    json={
                        "chat_id":    self.chat_id,
                        "text":       message,
                        "parse_mode": "HTML"
                    },
                    timeout=10
                )
                if r.status_code != 200:
                    logger.error(f"Telegram send failed: {r.text}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")
        threading.Thread(target=_do, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # ALERT TYPES
    # ─────────────────────────────────────────────────────────

    def bot_started(self):
        self._send(
            "🚀 <b>BOT STARTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time   : {self._now()}\n"
            "Status : All systems running\n\n"
            "<b>📊 Info</b>\n"
            "/status      → system health + filters\n"
            "/risk        → risk dashboard\n"
            "/funds       → balance breakdown\n"
            "/pnl         → today's PnL\n"
            "/positions   → open positions\n"
            "/models      → model status\n"
            "/performance → all-time stats\n"
            "/intervals   → active intervals\n"
            "/vix         → India VIX status\n"
            "/filters     → active trading filters\n\n"
            "<b>⚙️ Controls</b>\n"
            "/stop      → pause trading\n"
            "/resume    → resume trading\n"
            "/pause 30  → pause N minutes\n\n"
            "<b>💰 Risk Controls</b>\n"
            "/setmode MIS|CNC|MTF|MARGIN\n"
            "/setcapital 2000\n"
            "/setlimit 5000\n"
            "/setdaily 3000\n"
            "/setslots 3 2\n"
            "/settrades 10\n\n"
            "<b>🔍 Market Filters</b>\n"
            "/setsegment EQUITY|FNO|BOTH\n"
            "/setexchange NSE|BSE|BOTH\n"
            "/setinstrument EQUITY|FUTURES|OPTIONS|ALL\n\n"
            "Send /help for full list"
        )

    def bot_stopped(self, reason: str = "Manual shutdown"):
        self._send(
            "🔴 <b>BOT STOPPED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason : {reason}\n"
            f"Time   : {self._now()}"
        )

    def trade_executed(self, side: str, name: str,
                       segment: str, qty: int, price: float,
                       confidence: float, xgb: str, rl: str,
                       lgbm: str = "", lstm: str = "",
                       pattern: str = "", regime: str = "",
                       stop_loss: float = 0,
                       take_profit: float = 0,
                       rr_ratio: float = 0,
                       sl_pct: float = 0,
                       tp_pct: float = 0):
        emoji = "🟢" if side == "BUY" else "🔴"
        extra = ""
        if lgbm:
            extra += f"LGBM       : {lgbm}\n"
        if lstm:
            extra += f"LSTM       : {lstm}\n"
        if pattern and pattern != "NONE":
            extra += f"Pattern    : {pattern}\n"
        if regime:
            extra += f"Regime     : {regime}\n"
        sl_tp = ""
        if stop_loss and take_profit:
            sl_tp = (
                f"Stop Loss  : ₹{stop_loss:,.2f} "
                f"(-{sl_pct:.2f}%)\n"
                f"Take Profit: ₹{take_profit:,.2f} "
                f"(+{tp_pct:.2f}%)\n"
                f"RR Ratio   : {rr_ratio:.1f}:1\n"
            )
        self._send(
            f"{emoji} <b>{side} EXECUTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Segment    : {segment}\n"
            f"Qty        : {qty}\n"
            f"Price      : ₹{price:,.2f}\n"
            f"XGB        : {xgb}\n"
            f"RL         : {rl}\n"
            f"{extra}"
            f"{sl_tp}"
            f"Confidence : {confidence*100:.1f}%\n"
            f"Time       : {self._now()}"
        )

    def trade_closed(self, name: str, qty: int,
                     entry: float, exit_price: float,
                     pnl: float, daily_pnl: float,
                     reason: str = ""):
        emoji       = "✅" if pnl >= 0 else "❌"
        pct         = (exit_price - entry) / entry * 100
        reason_line = f"Exit Reason: {reason}\n" if reason else ""
        self._send(
            f"{emoji} <b>POSITION CLOSED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Qty        : {qty}\n"
            f"Entry      : ₹{entry:,.2f}\n"
            f"Exit       : ₹{exit_price:,.2f}\n"
            f"{reason_line}"
            f"PnL        : ₹{pnl:+,.2f} ({pct:+.2f}%)\n"
            f"Daily PnL  : ₹{daily_pnl:+,.2f}\n"
            f"Time       : {self._now()}"
        )

    def order_failed(self, side: str, name: str,
                     qty: int, reason: str):
        self._send(
            "🚨 <b>ORDER FAILED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Side   : {side}\n"
            f"Name   : {name}\n"
            f"Qty    : {qty}\n"
            f"Reason : {reason}\n"
            f"Time   : {self._now()}"
        )

    def insufficient_margin(self, name: str,
                             required: float,
                             available: float):
        self._send(
            "⚠️ <b>INSUFFICIENT MARGIN</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Required   : ₹{required:,.2f}\n"
            f"Available  : ₹{available:,.2f}\n"
            f"Action     : Skipped\n"
            f"Time       : {self._now()}"
        )

    def risk_warning(self, daily_pnl: float,
                     limit: float, balance: float):
        pct = abs(daily_pnl) / balance * 100 if balance else 0
        self._send(
            "⚠️ <b>RISK WARNING</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss : ₹{daily_pnl:,.2f} ({pct:.1f}%)\n"
            f"Limit      : ₹{limit:,.2f}\n"
            f"Time       : {self._now()}"
        )

    def kill_switch(self, daily_pnl: float, limit: float):
        self._send(
            "🔴 <b>KILL SWITCH ACTIVATED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss : ₹{daily_pnl:,.2f}\n"
            f"Limit Hit  : ₹{limit:,.2f}\n"
            f"Action     : Trading PAUSED\n"
            f"Time       : {self._now()}\n\n"
            "Send /resume tomorrow after reset."
        )

    def squareoff_alert(self, positions: dict):
        lines = ""
        for key, pos in positions.items():
            if pos:
                lines += (
                    f"  {key.upper()} — "
                    f"{pos['qty']} @ ₹{pos['entry']:,.2f}\n"
                )
        if not lines:
            lines = "  No open positions\n"
        self._send(
            "⏰ <b>AUTO SQUARE-OFF</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time   : {self._now()}\n"
            f"Closed :\n{lines}"
        )

    def model_retrained(self, instrument: str, samples: int,
                        duration_seconds: float,
                        next_retrain: str):
        self._send(
            "🧠 <b>MODEL RETRAINED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument   : {instrument}\n"
            f"Samples      : {samples:,}\n"
            f"Duration     : {duration_seconds:.0f}s\n"
            f"Next Retrain : {next_retrain}\n"
            f"Time         : {self._now()}"
        )

    def daily_summary(self, total_pnl: float, trades: int,
                      wins: int, balance: float,
                      vix: float = 0.0):
        losses   = trades - wins
        win_rate = wins / trades * 100 if trades > 0 else 0
        emoji    = "✅" if total_pnl >= 0 else "❌"
        vix_line = ""
        if vix > 0:
            vix_icon = (
                "🔴" if vix >= 25 else
                "⚠️" if vix >= 20 else
                "✅"
            )
            vix_line = f"India VIX  : {vix:.1f} {vix_icon}\n"
        self._send(
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Total PnL  : ₹{total_pnl:+,.2f}\n"
            f"Trades     : {trades}\n"
            f"Wins       : {wins}  |  Losses: {losses}\n"
            f"Win Rate   : {win_rate:.1f}%\n"
            f"Balance    : ₹{balance:,.2f}\n"
            f"{vix_line}"
            f"Date       : {datetime.now().strftime('%d %b %Y')}"
        )

    def error(self, error_type: str, message: str):
        self._send(
            "🚨 <b>ERROR</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Type    : <code>{error_type}</code>\n"
            f"Message : {message}\n"
            f"Time    : {self._now()}"
        )

    # ─────────────────────────────────────────────────────────
    # COMMAND LISTENER
    # ─────────────────────────────────────────────────────────

    def start_command_listener(self, bot_ref=None):
        self._bot_ref = bot_ref
        self._running = True

        def _poll():
            while self._running:
                try:
                    r = requests.get(
                        f"{self.BASE_URL}/getUpdates",
                        params={
                            "offset":  self._last_update + 1,
                            "timeout": 2
                        },
                        timeout=10
                    )
                    if r.status_code == 200:
                        for update in r.json().get("result", []):
                            self._last_update = update["update_id"]
                            msg  = update.get("message", {})
                            text = msg.get("text", "").strip()
                            if text:
                                self._handle_command(text)
                except Exception as e:
                    logger.error(f"Command poll error: {e}")
                time.sleep(3)

        threading.Thread(target=_poll, daemon=True).start()
        logger.info("✅ Telegram command listener started")

    def _handle_command(self, text: str):
        parts = text.strip().split()
        cmd   = parts[0].lower() if parts else ""
        args  = parts[1:]
        ref   = self._bot_ref

        # ── /help ─────────────────────────────────────────────
        if cmd == "/help":
            self._send(
                "ℹ️ <b>AVAILABLE COMMANDS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>📊 Info</b>\n"
                "/status       → system health + filters\n"
                "/risk         → risk dashboard\n"
                "/funds        → balance breakdown\n"
                "/pnl          → today's PnL\n"
                "/positions    → open positions + SL/TP\n"
                "/models       → model status + weights\n"
                "/performance  → all-time stats\n"
                "/intervals    → active intervals\n"
                "/vix          → India VIX live status\n"
                "/filters      → current trading filters\n\n"
                "<b>⚙️ Controls</b>\n"
                "/stop             → pause trading\n"
                "/resume           → resume trading\n"
                "/pause 30         → pause N minutes\n\n"
                "<b>💰 Risk Controls</b>\n"
                "/setmode MIS      → intraday equity\n"
                "/setmode CNC      → delivery equity\n"
                "/setmode MTF      → leveraged delivery\n"
                "/setmode MARGIN   → futures &amp; options\n"
                "/setcapital 2000  → limit capital ₹\n"
                "/setcapital 0     → remove limit\n"
                "/setlimit 5000    → max ₹ per trade\n"
                "/setlimit 0       → remove limit\n"
                "/setdaily 3000    → max daily loss ₹\n"
                "/setdaily 0       → revert to 3% auto\n"
                "/setslots 3 2     → equity/FNO slots\n"
                "/settrades 10     → max trades today\n\n"
                "<b>🔍 Market Filters</b>\n"
                "/setsegment EQUITY    → stocks only\n"
                "/setsegment FNO       → futures &amp; options only\n"
                "/setsegment BOTH      → all segments (default)\n"
                "/setexchange NSE      → NSE only\n"
                "/setexchange BSE      → BSE only\n"
                "/setexchange BOTH     → all exchanges (default)\n"
                "/setinstrument EQUITY   → stocks only\n"
                "/setinstrument FUTURES  → futures only\n"
                "/setinstrument OPTIONS  → options only\n"
                "/setinstrument ALL      → everything (default)"
            )

        # ── /vix ──────────────────────────────────────────────
        elif cmd == "/vix":
            if not ref:
                return
            vix_info = ref.get_india_vix()
            vix      = vix_info["vix"]

            if vix_info["stop_trade"]:
                status_line = "🔴 TRADING STOPPED (VIX ≥ 25)"
                action_line = "No new positions today."
            elif vix_info["half_size"]:
                status_line = "⚠️ REDUCED SIZES (VIX 20–24)"
                action_line = "Position sizes halved today."
            else:
                status_line = "✅ NORMAL (VIX < 20)"
                action_line = "Full position sizes active."

            filled = min(10, int(vix / 4))
            bar    = "█" * filled + "░" * (10 - filled)

            self._send(
                "🌡️ <b>INDIA VIX STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Current VIX : {vix:.2f}\n"
                f"Gauge       : [{bar}]\n\n"
                f"Status  : {status_line}\n"
                f"Action  : {action_line}\n\n"
                "<b>Thresholds</b>\n"
                "  &lt; 20  → ✅ Trade normally\n"
                "  20–24 → ⚠️ Half position size\n"
                "  ≥ 25  → 🔴 No trading\n\n"
                f"Multiplier : {vix_info['multiplier']:.1f}×\n"
                f"Bot Paused : {'Yes ⏸' if self.bot_paused else 'No ✅'}\n"
                f"Time       : {self._now()}"
            )

        # ── /filters ──────────────────────────────────────────
        elif cmd == "/filters":
            if not ref:
                return
            seg  = getattr(ref, "ACTIVE_SEGMENT",    "BOTH")
            exch = getattr(ref, "ACTIVE_EXCHANGE",   "BOTH")
            inst = getattr(ref, "ACTIVE_INSTRUMENT", "ALL")

            seg_desc = (
                "Stocks only (NSE/BSE)"         if seg == "EQUITY" else
                "Futures &amp; Options only"    if seg == "FNO"    else
                "All segments"
            )
            exch_desc = (
                "NSE only" if exch == "NSE" else
                "BSE only" if exch == "BSE" else
                "NSE + BSE"
            )
            inst_desc = (
                "Equity stocks only" if inst == "EQUITY"  else
                "Futures only"       if inst == "FUTURES" else
                "Options only"       if inst == "OPTIONS" else
                "All instruments"
            )

            self._send(
                "🔍 <b>ACTIVE TRADING FILTERS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Segment    : {seg} — {seg_desc}\n"
                f"Exchange   : {exch} — {exch_desc}\n"
                f"Instrument : {inst} — {inst_desc}\n\n"
                "<b>To change:</b>\n"
                "/setsegment EQUITY|FNO|BOTH\n"
                "/setexchange NSE|BSE|BOTH\n"
                "/setinstrument EQUITY|FUTURES|OPTIONS|ALL\n\n"
                f"Time : {self._now()}"
            )

        # ── /status ───────────────────────────────────────────
        elif cmd == "/status":
            if not ref:
                return
            st   = ref.posmgr.status()
            vix  = ref._vix_data
            seg  = getattr(ref, "ACTIVE_SEGMENT",    "BOTH")
            exch = getattr(ref, "ACTIVE_EXCHANGE",   "BOTH")
            inst = getattr(ref, "ACTIVE_INSTRUMENT", "ALL")

            vix_line = (
                f"VIX        : {vix['vix']:.1f} "
                + ("✅ Safe"       if vix["safe"]      else
                   "⚠️ Half-size" if vix["half_size"] else
                   "🔴 STOPPED")
                + "\n"
            )
            self._send(
                "📊 <b>BOT STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode       : {'⏸ Paused' if self.bot_paused else '✅ Live'}\n"
                f"Trade Mode : {ref.TRADE_MODE}\n"
                f"Positions  : {st['open_positions']} open\n"
                f"Equity     : {st['equity_open']}/{ref.MAX_EQUITY_POS}\n"
                f"FNO        : {st['fno_open']}/{ref.MAX_FNO_POS}\n"
                f"Win Rate   : {st['win_rate']:.1f}%\n"
                f"Daily PnL  : ₹{ref.risk.daily_pnl:+,.0f}\n"
                f"Trades     : {ref.risk.trades_today}/{ref.risk.max_trades_per_day}\n"
                f"{vix_line}"
                f"Segment    : {seg}\n"
                f"Exchange   : {exch}\n"
                f"Instrument : {inst}\n"
                f"Time       : {self._now()}"
            )

        # ── /risk ─────────────────────────────────────────────
        elif cmd == "/risk":
            if not ref:
                return
            bal   = ref.get_balance(ref.TRADE_MODE)
            limit = ref.risk.get_daily_limit(bal)
            used  = abs(min(ref.risk.daily_pnl, 0))
            pct   = used / limit * 100 if limit else 0
            st    = ref.posmgr.status()
            self._send(
                "💰 <b>RISK DASHBOARD</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Trade Mode       : {ref.TRADE_MODE}\n"
                f"Active Balance   : ₹{bal:,.0f}\n"
                f"Capital Limit    : "
                + (f"₹{ref.TRADE_CAPITAL:,.0f}"
                   if ref.TRADE_CAPITAL else "No limit") + "\n"
                f"Daily Loss Limit : ₹{limit:,.0f} "
                + ("(custom)" if ref.risk.daily_loss_cap
                   else "(3% auto)") + "\n"
                f"Daily Loss Used  : ₹{used:,.0f} ({pct:.0f}%)\n"
                f"Per Trade Limit  : "
                + (f"₹{ref.risk.per_trade_limit:,.0f}"
                   if ref.risk.per_trade_limit else "No limit") + "\n"
                f"Max Trades/Day   : {ref.risk.max_trades_per_day}\n"
                f"Trades Today     : {ref.risk.trades_today}\n\n"
                f"Equity : {st['equity_open']}/{ref.MAX_EQUITY_POS} slots\n"
                f"FNO    : {st['fno_open']}/{ref.MAX_FNO_POS} slots\n\n"
                f"Status : {'⏸ Paused' if self.bot_paused else '✅ Active'}\n"
                f"Time   : {self._now()}"
            )

        # ── /funds ────────────────────────────────────────────
        elif cmd == "/funds":
            if not ref:
                return

            # get_all_balances() calls api.get_funds() which calls
            # get_available_margin_details() — already has all we need.
            # No additional API calls needed here.
            f = ref.get_all_balances()
            if not f:
                self._send("❌ Could not fetch funds.")
                return

            avl        = f.get("detailed_avl_balance", {})
            trade_mode = ref.TRADE_MODE

            mode_balance_map = {
                "MIS":    ("eq_mis",  "Intraday Equity (MIS)"),
                "CNC":    ("eq_cnc",  "Delivery Equity (CNC)"),
                "MTF":    ("eq_mtf",  "Leveraged Delivery (MTF)"),
                "MARGIN": ("future",  "Futures Margin"),
            }
            active_field, active_label = mode_balance_map.get(
                trade_mode, ("eq_mis", "Intraday Equity (MIS)")
            )
            true_balance = float(avl.get(active_field, 0.0))

            cap_line = (
                f"\n⚠️ Capped at: ₹{ref.TRADE_CAPITAL:,.2f}"
                if ref.TRADE_CAPITAL
                and true_balance > ref.TRADE_CAPITAL
                else ""
            )

            # ── Buying power derived from margin data ─────────
            # Groww's get_available_margin_details() returns both
            # cnc_balance_available (full cash, 1x) and
            # mis_balance_available (leveraged, e.g. 5x).
            # We derive the multiplier from the ratio of the two.
            # This is correct and requires NO extra API call.
            buying_power_line = ""
            try:
                eq_cnc = float(avl.get("eq_cnc", 0.0))
                eq_mis = float(avl.get("eq_mis", 0.0))

                if trade_mode == "MIS" and eq_cnc > 0 and eq_mis > 0:
                    # MIS leverage = MIS balance / CNC balance
                    leverage     = round(eq_mis / eq_cnc, 1)
                    buying_power = true_balance  # already the leveraged balance
                    buying_power_line = (
                        f"\n<b>📐 Buying Power</b>\n"
                        f"Leverage     : ~{leverage:.1f}x "
                        f"(MIS ÷ CNC)\n"
                        f"Buying Power : ₹{buying_power:,.2f}\n"
                        f"Cash Base    : ₹{eq_cnc:,.2f}\n"
                    )
                elif trade_mode == "CNC":
                    buying_power_line = (
                        f"\n<b>📐 Buying Power</b>\n"
                        f"Leverage     : 1.0x (delivery)\n"
                        f"Buying Power : ₹{true_balance:,.2f}\n"
                    )
                elif trade_mode == "MARGIN":
                    fut = float(avl.get("future", 0.0))
                    buying_power_line = (
                        f"\n<b>📐 Buying Power</b>\n"
                        f"Futures Margin : ₹{fut:,.2f}\n"
                        f"(Actual buying power depends on contract)\n"
                    )
            except Exception as e:
                logger.warning(f"/funds buying power calc failed: {e}")
                buying_power_line = ""

            self._send(
                "💵 <b>FUNDS BREAKDOWN</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"<b>✅ Active Balance ({trade_mode})</b>\n"
                f"{active_label} : ₹{true_balance:,.2f}"
                f"{cap_line}\n"
                f"{buying_power_line}\n"
                "<b>📈 All Equity Limits</b>\n"
                f"Intraday (MIS)   : ₹{avl.get('eq_mis', 0):,.2f}\n"
                f"Delivery (CNC)   : ₹{avl.get('eq_cnc', 0):,.2f}\n"
                f"Leveraged (MTF)  : ₹{avl.get('eq_mtf', 0):,.2f}\n\n"
                "<b>📊 Derivatives Limits</b>\n"
                f"Futures (MARGIN) : ₹{avl.get('future', 0):,.2f}\n"
                f"Options Buy      : ₹{avl.get('option_buy', 0):,.2f}\n"
                f"Options Sell     : ₹{avl.get('option_sell', 0):,.2f}\n\n"
                "ℹ️ <i>All above = same cash, split by segment.\n"
                "Bot uses only the active mode balance.</i>\n\n"
                "<b>💰 Account</b>\n"
                f"Opening Balance  : ₹{f.get('sod_balance', 0):,.2f}\n"
                f"Funds Added      : ₹{f.get('funds_added', 0):,.2f}\n"
                f"Withdrawal Avail : ₹{f.get('withdrawal_balance', 0):,.2f}\n\n"
                "<b>📉 Today</b>\n"
                f"Realized PnL     : ₹{f.get('realized_pnl', 0):+,.2f}\n"
                f"Unrealized PnL   : ₹{f.get('unrealized_pnl', 0):+,.2f}\n"
                f"Brokerage        : ₹{f.get('brokerage', 0):,.2f}\n\n"
                f"Time : {self._now()}"
            )

        # ── /pnl ──────────────────────────────────────────────
        elif cmd == "/pnl":
            if not ref:
                return
            emoji = "✅" if ref.risk.daily_pnl >= 0 else "❌"
            f     = ref.get_all_balances()
            self._send(
                f"{emoji} <b>TODAY'S PnL</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Daily PnL      : ₹{ref.risk.daily_pnl:+,.2f}\n"
                f"Realized PnL   : ₹{f.get('realized_pnl', 0):+,.2f}\n"
                f"Unrealized PnL : ₹{f.get('unrealized_pnl', 0):+,.2f}\n"
                f"Brokerage      : ₹{f.get('brokerage', 0):,.2f}\n"
                f"Trades         : {ref.risk.trades_today}\n"
                f"Time           : {self._now()}"
            )

        # ── /positions ────────────────────────────────────────
        elif cmd == "/positions":
            if not ref:
                return
            positions = ref.posmgr.positions
            if not positions:
                self._send("📭 No open positions")
                return
            lines = [
                "📋 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━"
            ]
            for sc, pos in positions.items():
                meta    = pos.get("signal_meta", {})
                sl      = meta.get("stop_loss")
                tp      = meta.get("take_profit")
                pattern = meta.get("pattern", "")
                regime  = meta.get("regime",  "")
                sl_line = (
                    f"  SL / TP : ₹{sl:,.2f} / ₹{tp:,.2f}\n"
                    if sl and tp else ""
                )
                lines.append(
                    f"<b>{pos['name']}</b>\n"
                    f"  Qty     : {pos['qty']}\n"
                    f"  Entry   : ₹{pos['entry']:,.2f}\n"
                    f"  Segment : {pos['segment']}\n"
                    f"  Conf    : {meta.get('confidence', 0):.1%}\n"
                    f"{sl_line}"
                    + (f"  Pattern : {pattern}\n"
                       if pattern and pattern != "NONE" else "")
                    + (f"  Regime  : {regime}\n" if regime else "")
                )
            lines.append(f"Time : {self._now()}")
            self._send("\n\n".join(lines))

        # ── /models ───────────────────────────────────────────
        elif cmd == "/models":
            if not ref:
                return
            import os
            model_dir = "ml/models"
            if not os.path.exists(model_dir):
                self._send("No models trained yet.")
                return
            files = os.listdir(model_dir)
            xgb   = len([f for f in files if f.startswith("xgb")])
            lgbm  = len([f for f in files if f.startswith("lgbm")])
            lstm  = len([f for f in files if f.startswith("lstm")])
            ppo   = len([f for f in files if f.startswith("ppo")])
            from ml.meta_model import MetaModel
            weights = MetaModel().weights

            gate_lines = ""
            for sc, t in ref.trainers.items():
                gate_lines += (
                    f"  {sc} → gate={t._confidence_gate:.0%}\n"
                )

            self._send(
                "🤖 <b>MODEL STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"XGBoost  : {xgb} instruments\n"
                f"LightGBM : {lgbm} instruments\n"
                f"LSTM     : {lstm} instruments\n"
                f"PPO      : {ppo} instruments\n\n"
                "<b>Meta Weights (auto-updating)</b>\n"
                f"XGB  : {weights.get('xgb', 0):.3f}\n"
                f"PPO  : {weights.get('ppo', 0):.3f}\n"
                f"A2C  : {weights.get('a2c', 0):.3f}\n"
                f"Sent : {weights.get('sentiment', 0):.3f}\n\n"
                "<b>Adaptive Confidence Gates</b>\n"
                f"{gate_lines or '  No trainers yet' + chr(10)}"
                f"Retrain : Every Sunday 18:30\n"
                f"Time    : {self._now()}"
            )

        # ── /intervals ────────────────────────────────────────
        elif cmd == "/intervals":
            if not ref:
                return
            lines = [
                "⏱ <b>ACTIVE INTERVALS</b>\n━━━━━━━━━━━━━━━━━━━━"
            ]
            for sc, trainer in ref.trainers.items():
                lines.append(f"  {sc} → {trainer.interval}")
            lines.append(f"\nTime : {self._now()}")
            self._send("\n".join(lines))

        # ── /performance ──────────────────────────────────────
        elif cmd == "/performance":
            if not ref:
                return
            from ml.trade_memory import TradeMemory
            stats = TradeMemory().get_stats()
            pf    = stats.get("profit_factor", 0)

            from ml.trade_memory import TradeMemory as TM
            df = TM().df
            reason_lines = ""
            if not df.empty and "exit_reason" in df.columns:
                counts = df["exit_reason"].value_counts()
                for reason, cnt in counts.items():
                    reason_lines += f"  {reason}: {cnt}\n"

            self._send(
                "📈 <b>ALL-TIME PERFORMANCE</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Total Trades   : {stats['total']}\n"
                f"Wins           : {stats['wins']}\n"
                f"Losses         : {stats['losses']}\n"
                f"Win Rate       : {stats['win_rate']}%\n"
                f"Avg Win        : ₹{stats['avg_win']:+,.2f}\n"
                f"Avg Loss       : ₹{stats['avg_loss']:+,.2f}\n"
                f"Profit Factor  : {pf:.2f}x\n"
                f"Best Trade     : ₹{stats['best_trade']:+,.2f}\n"
                f"Worst Trade    : ₹{stats['worst_trade']:+,.2f}\n"
                f"Total PnL      : ₹{stats['total_pnl']:+,.2f}\n\n"
                + (
                    "<b>Exit Reasons</b>\n" + reason_lines
                    if reason_lines else ""
                )
                + f"Time           : {self._now()}"
            )

        # ── /stop ─────────────────────────────────────────────
        elif cmd == "/stop":
            self.bot_paused = True
            self._send(
                "⛔ <b>TRADING PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "No new orders will be placed.\n"
                f"Time : {self._now()}\n\n"
                "Send /resume to restart."
            )

        # ── /resume ───────────────────────────────────────────
        elif cmd == "/resume":
            self.bot_paused = False
            self._send(
                "✅ <b>TRADING RESUMED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Time : {self._now()}"
            )

        # ── /pause ────────────────────────────────────────────
        elif cmd == "/pause":
            minutes   = (
                int(args[0]) if args and args[0].isdigit() else 30
            )
            self.bot_paused = True
            resume_at = datetime.now() + timedelta(minutes=minutes)
            self._send(
                "⏸ <b>BOT PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Duration : {minutes} minutes\n"
                f"Resumes  : {resume_at.strftime('%I:%M %p')}\n"
                f"Time     : {self._now()}"
            )
            def _auto_resume():
                time.sleep(minutes * 60)
                self.bot_paused = False
                self._send(
                    "▶️ <b>AUTO-RESUMED</b>\n"
                    f"Time : {self._now()}"
                )
            threading.Thread(
                target=_auto_resume, daemon=True
            ).start()

        # ── /setmode ──────────────────────────────────────────
        elif cmd == "/setmode":
            if not args:
                self._send(
                    "Usage: <code>/setmode MIS</code>\n\n"
                    "MIS    → intraday equity\n"
                    "CNC    → delivery equity\n"
                    "MTF    → leveraged delivery\n"
                    "MARGIN → futures &amp; options"
                )
                return
            mode = args[0].upper()
            if mode not in ("MIS", "CNC", "MTF", "MARGIN"):
                self._send("❌ Valid: MIS, CNC, MTF, MARGIN")
                return
            ref.TRADE_MODE = mode
            bal = ref.api.get_true_balance(mode)
            self._send(
                f"✅ <b>Mode → {mode}</b>\n"
                f"Available : ₹{bal:,.2f}"
            )

        # ── /setcapital ───────────────────────────────────────
        elif cmd == "/setcapital":
            if not args:
                self._send(
                    "Usage: <code>/setcapital 2000</code> "
                    "or 0 to remove"
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.TRADE_CAPITAL = None
                    self._send("✅ Capital limit removed.")
                else:
                    ref.TRADE_CAPITAL = amount
                    self._send(
                        f"✅ Capital limit: ₹{amount:,.0f}"
                    )
            except ValueError:
                self._send(
                    "❌ Invalid. Usage: "
                    "<code>/setcapital 2000</code>"
                )

        # ── /setlimit ─────────────────────────────────────────
        elif cmd == "/setlimit":
            if not args:
                self._send(
                    "Usage: <code>/setlimit 5000</code> "
                    "or 0 to remove"
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.per_trade_limit = None
                    self._send("✅ Per-trade limit removed.")
                else:
                    ref.risk.per_trade_limit = amount
                    self._send(
                        f"✅ Per-trade limit: ₹{amount:,.0f}"
                    )
            except ValueError:
                self._send(
                    "❌ Invalid. Usage: "
                    "<code>/setlimit 5000</code>"
                )

        # ── /setdaily ─────────────────────────────────────────
        elif cmd == "/setdaily":
            if not args:
                self._send(
                    "Usage: <code>/setdaily 3000</code> "
                    "or 0 for 3% auto"
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.daily_loss_cap = None
                    self._send(
                        "✅ Daily cap removed — reverting to 3%."
                    )
                else:
                    ref.risk.daily_loss_cap = amount
                    self._send(
                        f"✅ Daily loss cap: ₹{amount:,.0f}"
                    )
            except ValueError:
                self._send(
                    "❌ Invalid. Usage: "
                    "<code>/setdaily 3000</code>"
                )

        # ── /setslots ─────────────────────────────────────────
        elif cmd == "/setslots":
            if len(args) < 2:
                self._send(
                    "Usage: <code>/setslots 3 2</code>"
                )
                return
            try:
                eq  = int(args[0])
                fno = int(args[1])
                ref.posmgr.max_equity = eq
                ref.posmgr.max_fno    = fno
                ref.MAX_EQUITY_POS    = eq
                ref.MAX_FNO_POS       = fno
                self._send(
                    f"✅ Slots: Equity={eq} | FNO={fno}"
                )
            except ValueError:
                self._send(
                    "❌ Invalid. Usage: "
                    "<code>/setslots 3 2</code>"
                )

        # ── /settrades ────────────────────────────────────────
        elif cmd == "/settrades":
            if not args:
                self._send(
                    "Usage: <code>/settrades 10</code>"
                )
                return
            try:
                n = int(args[0])
                ref.risk.max_trades_per_day = n
                self._send(
                    f"✅ Max trades/day: {n}\n"
                    f"Used today: {ref.risk.trades_today}"
                )
            except ValueError:
                self._send(
                    "❌ Invalid. Usage: "
                    "<code>/settrades 10</code>"
                )

        # ── /setsegment ───────────────────────────────────────
        elif cmd == "/setsegment":
            if not args:
                self._send(
                    "Usage: <code>/setsegment EQUITY</code>\n\n"
                    "EQUITY → stocks only (NSE/BSE)\n"
                    "FNO    → futures &amp; options only\n"
                    "BOTH   → all segments (default)"
                )
                return
            val = args[0].upper()
            if val not in ("EQUITY", "FNO", "BOTH"):
                self._send("❌ Valid values: EQUITY, FNO, BOTH")
                return
            ref.ACTIVE_SEGMENT = val
            desc = (
                "Stocks only (NSE/BSE)"         if val == "EQUITY" else
                "Futures &amp; Options only"    if val == "FNO"    else
                "All segments"
            )
            self._send(
                f"✅ <b>Segment filter → {val}</b>\n"
                f"Trading   : {desc}\n"
                f"Time      : {self._now()}"
            )

        # ── /setexchange ──────────────────────────────────────
        elif cmd == "/setexchange":
            if not args:
                self._send(
                    "Usage: <code>/setexchange NSE</code>\n\n"
                    "NSE  → NSE listed only\n"
                    "BSE  → BSE listed only\n"
                    "BOTH → all exchanges (default)"
                )
                return
            val = args[0].upper()
            if val not in ("NSE", "BSE", "BOTH"):
                self._send("❌ Valid values: NSE, BSE, BOTH")
                return
            ref.ACTIVE_EXCHANGE = val
            desc = (
                "NSE only"  if val == "NSE" else
                "BSE only"  if val == "BSE" else
                "NSE + BSE"
            )
            self._send(
                f"✅ <b>Exchange filter → {val}</b>\n"
                f"Trading   : {desc}\n"
                f"Time      : {self._now()}"
            )

        # ── /setinstrument ────────────────────────────────────
        elif cmd == "/setinstrument":
            if not args:
                self._send(
                    "Usage: <code>/setinstrument FUTURES</code>\n\n"
                    "EQUITY  → equity stocks only\n"
                    "FUTURES → futures contracts only\n"
                    "OPTIONS → options contracts only\n"
                    "ALL     → all instruments (default)"
                )
                return
            val = args[0].upper()
            if val not in ("EQUITY", "FUTURES", "OPTIONS", "ALL"):
                self._send(
                    "❌ Valid values: EQUITY, FUTURES, OPTIONS, ALL"
                )
                return
            ref.ACTIVE_INSTRUMENT = val
            desc = (
                "Equity stocks only" if val == "EQUITY"  else
                "Futures only"       if val == "FUTURES" else
                "Options only"       if val == "OPTIONS" else
                "All instruments"
            )
            self._send(
                f"✅ <b>Instrument filter → {val}</b>\n"
                f"Trading   : {desc}\n"
                f"Time      : {self._now()}"
            )

        # ── unknown command ───────────────────────────────────
        elif cmd.startswith("/"):
            self._send(
                f"❓ Unknown: <code>{cmd}</code>\n"
                "Send /help to see all commands."
            )

    def stop_listener(self):
        self._running = False

    def _now(self) -> str:
        return datetime.now().strftime("%d %b %Y %H:%M:%S")