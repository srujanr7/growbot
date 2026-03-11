import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MEMORY_FILE = "data/trade_history.csv"

COLUMNS = [
    "timestamp", "scrip_code", "name", "segment",
    "side", "entry", "exit", "qty", "pnl", "pnl_pct",
    "hold_minutes", "confidence", "xgb_signal", "rl_signal",
    "sentiment", "rsi_at_entry", "macd_at_entry",
    "volume_ratio_at_entry", "market_trend",
    "outcome"
]


class TradeMemory:

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        if os.path.exists(MEMORY_FILE):
            self.df = pd.read_csv(MEMORY_FILE)
        else:
            self.df = pd.DataFrame(columns=COLUMNS)
            self.df.to_csv(MEMORY_FILE, index=False)
        logger.info(f"📚 Trade memory loaded: {len(self.df)} trades")

    def record(self, trade: dict):
        pnl     = trade.get("pnl", 0)
        pnl_pct = trade.get("pnl_pct", 0)

        if pnl_pct > 0.2:
            outcome = "WIN"
        elif pnl_pct < -0.1:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        row = {
            "timestamp":             datetime.now().isoformat(),
            "scrip_code":            trade.get("scrip_code", ""),
            "name":                  trade.get("name", ""),
            "segment":               trade.get("segment", ""),
            "side":                  trade.get("side", "BUY"),
            "entry":                 trade.get("entry", 0),
            "exit":                  trade.get("exit", 0),
            "qty":                   trade.get("qty", 0),
            "pnl":                   round(pnl, 2),
            "pnl_pct":               round(pnl_pct, 4),
            "hold_minutes":          trade.get("hold_minutes", 0),
            "confidence":            trade.get("confidence", 0),
            "xgb_signal":            trade.get("xgb", ""),
            "rl_signal":             trade.get("rl", ""),
            "sentiment":             trade.get("sentiment", 0),
            "rsi_at_entry":          trade.get("rsi", 0),
            "macd_at_entry":         trade.get("macd", 0),
            "volume_ratio_at_entry": trade.get("volume_ratio", 1),
            "market_trend":          trade.get("market_trend", "NEUTRAL"),
            "outcome":               outcome
        }

        self.df = pd.concat(
            [self.df, pd.DataFrame([row])],
            ignore_index=True
        )
        self.df.to_csv(MEMORY_FILE, index=False)
        logger.info(
            f"📝 Trade recorded: {trade.get('name')} | "
            f"{outcome} | ₹{pnl:+.2f}"
        )

    def get_last_n(self, scrip_code: str, n: int = 3) -> list:
        """
        Returns last N trades for a specific instrument
        as a list of dicts. Used by adaptive confidence gate
        in trainer.py to detect consecutive loss streaks.
        FIXED: was missing entirely — caused AttributeError
        on every trade close.
        """
        if self.df.empty:
            return []
        sub = self.df[self.df["scrip_code"] == scrip_code]
        if sub.empty:
            return []
        last_n = sub.tail(n)
        records = []
        for _, row in last_n.iterrows():
            records.append({
                "scrip_code": row.get("scrip_code", ""),
                "pnl_pct":    row.get("pnl_pct", 0),
                "outcome":    row.get("outcome", ""),
                "profitable": row.get("outcome", "") == "WIN",
            })
        return records

    def get_stats(self) -> dict:
        if len(self.df) == 0:
            return {
                "total": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "best_trade": 0,
                "worst_trade": 0, "total_pnl": 0
            }
        wins   = self.df[self.df["outcome"] == "WIN"]
        losses = self.df[self.df["outcome"] == "LOSS"]
        return {
            "total":         len(self.df),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / len(self.df) * 100, 1),
            "avg_win":       round(wins["pnl"].mean(), 2) if len(wins) else 0,
            "avg_loss":      round(losses["pnl"].mean(), 2) if len(losses) else 0,
            "profit_factor": round(
                wins["pnl"].sum() / abs(losses["pnl"].sum()), 2
            ) if len(losses) and losses["pnl"].sum() != 0 else 0,
            "best_trade":    round(self.df["pnl"].max(), 2),
            "worst_trade":   round(self.df["pnl"].min(), 2),
            "total_pnl":     round(self.df["pnl"].sum(), 2)
        }

    def get_instrument_stats(self, scrip_code: str) -> dict:
        sub = self.df[self.df["scrip_code"] == scrip_code]
        if len(sub) == 0:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0}
        wins = sub[sub["outcome"] == "WIN"]
        return {
            "trades":    len(sub),
            "wins":      len(wins),
            "win_rate":  round(len(wins) / len(sub) * 100, 1),
            "total_pnl": round(sub["pnl"].sum(), 2),
            "avg_pnl":   round(sub["pnl"].mean(), 2)
        }

    def get_training_data(self, min_trades: int = 20) -> pd.DataFrame:
        return self.df if len(self.df) >= min_trades else pd.DataFrame()