import logging
import pandas as pd

logger = logging.getLogger(__name__)


class PatternMemory:
    """
    Tracks historical win rate per combination of:
      scrip_code + pattern + regime + interval

    After enough trades it answers:
      "Hammer on RELIANCE in TRENDING_UP on 5min
       has historically won 81% of the time"

    This overrides hardcoded pattern confidence values
    once enough data (≥5 trades) exists for a setup.
    """

    MIN_SAMPLES = 5       # ignore setups with < 5 trades
    BOOST_THRESHOLD = 0.75
    REDUCE_THRESHOLD = 0.40

    def __init__(self, memory_ref=None):
        # Accepts a TradeMemory instance or imports lazily
        self._memory_ref = memory_ref

    def _get_df(self) -> pd.DataFrame:
        if self._memory_ref is not None:
            return self._memory_ref.df
        try:
            from ml.trade_memory import TradeMemory
            return TradeMemory().df
        except Exception:
            return pd.DataFrame()

    def get_historical_winrate(self,
                               scrip_code: str,
                               pattern:    str,
                               regime:     str,
                               interval:   str = "") -> float:
        """
        Returns win rate float 0.0–1.0.
        Returns 0.5 (neutral) if insufficient data.
        """
        if pattern in ("NONE", "", None):
            return 0.5

        df = self._get_df()
        if df.empty:
            return 0.5

        required = {"scrip_code", "pattern",
                    "regime", "profitable"}
        if not required.issubset(df.columns):
            return 0.5

        mask = (
            (df["scrip_code"] == scrip_code) &
            (df["pattern"]    == pattern)    &
            (df["regime"]     == regime)
        )
        if interval and "interval" in df.columns:
            mask &= df["interval"] == interval

        subset = df[mask]
        if len(subset) < self.MIN_SAMPLES:
            return 0.5   # not enough data yet

        return float(subset["profitable"].mean())

    def adjust_confidence(self,
                          signal:     dict,
                          scrip_code: str) -> dict:
        """
        Boosts or reduces signal confidence based on
        historical performance of this exact setup.
        Returns updated signal dict (does not mutate original).
        """
        signal = signal.copy()
        pattern  = signal.get("pattern", "NONE")
        regime   = signal.get("regime",  "UNKNOWN")
        interval = signal.get("interval", "")

        if pattern in ("NONE", "", None):
            return signal

        wr = self.get_historical_winrate(
            scrip_code, pattern, regime, interval
        )

        # Neutral — not enough history
        if wr == 0.5:
            return signal

        if wr >= self.BOOST_THRESHOLD:
            # Strong historical edge — boost confidence
            signal["confidence"] = min(
                1.0, signal["confidence"] * 1.20
            )
            signal["pattern_wr"] = round(wr, 3)
            signal["reason"]     = (
                f"Historical WR={wr:.0%} for "
                f"{pattern} in {regime}"
            )
            logger.debug(
                f"PatternMemory BOOST: {pattern} "
                f"on {scrip_code} → WR={wr:.0%}"
            )

        elif wr <= self.REDUCE_THRESHOLD:
            # Poor historical edge — reduce confidence
            signal["confidence"] *= 0.70
            signal["pattern_wr"] = round(wr, 3)
            signal["reason"]     = (
                f"Low historical WR={wr:.0%} for "
                f"{pattern} in {regime}"
            )
            logger.debug(
                f"PatternMemory REDUCE: {pattern} "
                f"on {scrip_code} → WR={wr:.0%}"
            )

        return signal

    def summary(self, scrip_code: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame summary of all pattern win rates.
        Useful for /performance Telegram command.
        """
        df = self._get_df()
        if df.empty:
            return pd.DataFrame()

        required = {"scrip_code", "pattern",
                    "regime", "profitable"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        if scrip_code:
            df = df[df["scrip_code"] == scrip_code]

        group_cols = ["scrip_code", "pattern", "regime"]
        if "interval" in df.columns:
            group_cols.append("interval")

        summary = (
            df.groupby(group_cols)["profitable"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "win_rate",
                              "count": "trades"})
            .reset_index()
        )
        summary = summary[
            summary["trades"] >= self.MIN_SAMPLES
        ]
        summary["win_rate"] = summary["win_rate"].round(3)
        return summary.sort_values(
            "win_rate", ascending=False
        )