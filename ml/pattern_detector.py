import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects candlestick patterns with high win rates.
    Each pattern returns a signal and confidence.
    """

    def detect_all(self, df: pd.DataFrame) -> dict:
        if len(df) < 5:
            return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

        patterns = []

        # Run all detectors
        patterns.append(self._hammer(df))
        patterns.append(self._engulfing(df))
        patterns.append(self._doji(df))
        patterns.append(self._morning_star(df))
        patterns.append(self._three_white_soldiers(df))

        # Filter out NONE patterns
        active = [p for p in patterns if p["pattern"] != "NONE"]

        if not active:
            return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

        # Return highest confidence pattern
        best = max(active, key=lambda x: x["confidence"])
        logger.debug(f"Pattern: {best['pattern']} → {best['signal']}")
        return best

    def _hammer(self, df: pd.DataFrame) -> dict:
        c = df.iloc[-1]
        body      = abs(c["close"] - c["open"])
        lower_wick = c["open"] - c["low"] if c["close"] > c["open"] else c["close"] - c["low"]
        upper_wick = c["high"] - max(c["close"], c["open"])

        if (lower_wick > body * 2 and
                upper_wick < body * 0.3 and
                body > 0):
            return {
                "pattern":    "HAMMER",
                "signal":     "BUY",
                "confidence": 0.72
            }
        return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

    def _engulfing(self, df: pd.DataFrame) -> dict:
        if len(df) < 2:
            return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Bullish engulfing
        if (prev["close"] < prev["open"] and
                curr["close"] > curr["open"] and
                curr["open"] < prev["close"] and
                curr["close"] > prev["open"]):
            return {
                "pattern":    "BULLISH_ENGULFING",
                "signal":     "BUY",
                "confidence": 0.75
            }

        # Bearish engulfing
        if (prev["close"] > prev["open"] and
                curr["close"] < curr["open"] and
                curr["open"] > prev["close"] and
                curr["close"] < prev["open"]):
            return {
                "pattern":    "BEARISH_ENGULFING",
                "signal":     "SELL",
                "confidence": 0.75
            }

        return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

    def _doji(self, df: pd.DataFrame) -> dict:
        c    = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        rng  = c["high"] - c["low"]

        if rng > 0 and body / rng < 0.1:
            return {
                "pattern":    "DOJI",
                "signal":     "HOLD",   # indecision
                "confidence": 0.60
            }
        return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

    def _morning_star(self, df: pd.DataFrame) -> dict:
        if len(df) < 3:
            return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]

        if (c1["close"] < c1["open"] and
                abs(c2["close"] - c2["open"]) < (c1["open"] - c1["close"]) * 0.3 and
                c3["close"] > c3["open"] and
                c3["close"] > (c1["open"] + c1["close"]) / 2):
            return {
                "pattern":    "MORNING_STAR",
                "signal":     "BUY",
                "confidence": 0.80
            }
        return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

    def _three_white_soldiers(self, df: pd.DataFrame) -> dict:
        if len(df) < 3:
            return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}

        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]

        if (c1["close"] > c1["open"] and
                c2["close"] > c2["open"] and
                c3["close"] > c3["open"] and
                c2["close"] > c1["close"] and
                c3["close"] > c2["close"] and
                c2["open"] > c1["open"] and
                c3["open"] > c2["open"]):
            return {
                "pattern":    "THREE_WHITE_SOLDIERS",
                "signal":     "BUY",
                "confidence": 0.82
            }
        return {"pattern": "NONE", "signal": "HOLD", "confidence": 0.0}