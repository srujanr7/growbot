import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class IntervalScore:
    interval: str
    win_rate: float
    avg_pnl:  float
    trades:   int
    score:    float

INTERVALS = ["1minute", "2minute", "3minute", "5minute",
             "10minute", "15minute", "30minute", "60minute"]


class IntervalSelector:
    """
    Automatically picks the best candle interval
    for each instrument based on backtest performance.

    Logic:
    - Short intervals (1m, 2m): more signals, more noise, 
      higher transaction cost impact
    - Long intervals (15m, 30m): fewer signals, cleaner trends
    - Best interval = highest risk-adjusted return

    Usage:
        selector = IntervalSelector(api, scrip_code, security_id)
        best     = selector.find_best()
        # → "5minute"
    """

    def __init__(self, api, scrip_code: str, security_id: str):
        self.api         = api
        self.scrip_code  = scrip_code
        self.security_id = security_id

    def _fetch(self, interval: str, days: int = 30) -> pd.DataFrame:
        """Fetch candles for a given interval."""
        import time
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        try:
            df = self.api.get_historical(
                self.scrip_code, interval,
                start_ms, end_ms
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Fetch failed for {interval}: {e}")
            return pd.DataFrame()

    def _backtest(self, df: pd.DataFrame) -> dict:
        """
        Simple EMA crossover backtest to score an interval.
        Returns win_rate, avg_pnl, trade_count.
        """
        if len(df) < 50:
            return {"win_rate": 0.0, "avg_pnl": 0.0, "trades": 0}

        df       = df.copy().reset_index(drop=True)
        df["f"]  = df["close"].ewm(span=9).mean()   # fast EMA
        df["s"]  = df["close"].ewm(span=21).mean()  # slow EMA
        df.dropna(inplace=True)

        trades = []
        pos    = 0
        entry  = 0.0

        for i in range(1, len(df)):
            prev_cross = df["f"].iloc[i-1] > df["s"].iloc[i-1]
            curr_cross = df["f"].iloc[i]   > df["s"].iloc[i]

            # Golden cross → BUY
            if not prev_cross and curr_cross and pos == 0:
                pos   = 1
                entry = df["close"].iloc[i]

            # Death cross → SELL/EXIT
            elif prev_cross and not curr_cross and pos == 1:
                pnl = (df["close"].iloc[i] - entry) / entry * 100
                trades.append(pnl)
                pos = 0

        if not trades:
            return {"win_rate": 0.0, "avg_pnl": 0.0, "trades": 0}

        arr      = np.array(trades)
        win_rate = float(np.mean(arr > 0))
        avg_pnl  = float(np.mean(arr))
        return {
            "win_rate": win_rate,
            "avg_pnl":  avg_pnl,
            "trades":   len(trades)
        }

    def _score(self, stats: dict, interval: str) -> float:
        """
        Score formula:
          base    = win_rate × avg_pnl
          penalty = too few trades (unreliable stats)
          penalty = very short intervals (noise + costs)
        """
        if stats["trades"] < 5:
            return 0.0

        base = stats["win_rate"] * max(stats["avg_pnl"], 0)

        # Penalize noisy short intervals unless they truly outperform
        noise_penalty = {
            "1minute":  0.70,
            "2minute":  0.80,
            "3minute":  0.85,
            "5minute":  0.92,
            "10minute": 0.97,
            "15minute": 1.00,
            "30minute": 1.00,
            "60minute": 0.95,   # too slow for intraday
        }
        multiplier = noise_penalty.get(interval, 1.0)

        # Bonus for having more trades (statistical confidence)
        trade_bonus = min(stats["trades"] / 50, 1.5)

        return round(base * multiplier * trade_bonus, 4)

    def find_best(self, candidates=None) -> str:
        """
        Test all intervals, return the best one.
        Falls back to "5minute" if nothing works.
        """
        candidates = candidates or INTERVALS
        results: Dict[str, IntervalScore] = {}

        logger.info(f"Testing {len(candidates)} intervals "
                    f"for {self.scrip_code}...")

        for interval in candidates:
            df    = self._fetch(interval)
            stats = self._backtest(df)
            score = self._score(stats, interval)

            results[interval] = IntervalScore(
                interval = interval,
                win_rate = stats["win_rate"],
                avg_pnl  = stats["avg_pnl"],
                trades   = stats["trades"],
                score    = score
            )
            logger.info(
                f"  {interval:10s} → "
                f"WR={stats['win_rate']:.0%}  "
                f"PnL={stats['avg_pnl']:+.3f}%  "
                f"Trades={stats['trades']}  "
                f"Score={score:.4f}"
            )

        if not results:
            return "5minute"

        best = max(results.values(), key=lambda x: x.score)

        if best.score == 0.0:
            logger.warning("No interval scored > 0, defaulting to 5minute")
            return "5minute"

        logger.info(
            f"✅ Best interval for {self.scrip_code}: "
            f"{best.interval} (score={best.score:.4f})"
        )
        return best.interval

    def find_best_per_session(self) -> dict:
        """
        Different intervals may work better at different times.
        Returns a schedule:
          { "morning": "3minute", "midday": "15minute",
            "afternoon": "5minute" }
        Useful if you want to auto-switch during the day.
        """
        # For now, return same interval for all sessions
        # You can extend this to fetch session-specific data
        best = self.find_best()
        return {
            "morning":   best,
            "midday":    best,
            "afternoon": best
        }