import pandas as pd
import pandas_ta as ta

class MACrossRSIStrategy:
    """
    Signal logic:
      BUY  — fast EMA crosses above slow EMA AND RSI < 60
      SELL — fast EMA crosses below slow EMA OR RSI > 75
    """
    def __init__(self, fast=9, slow=21, rsi_period=14):
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period

    def generate_signal(self, df: pd.DataFrame) -> str:
        if df is None or len(df) < self.slow + 5:
            return "HOLD"

        df["ema_fast"] = ta.ema(df["close"], length=self.fast)
        df["ema_slow"] = ta.ema(df["close"], length=self.slow)
        df["rsi"]      = ta.rsi(df["close"], length=self.rsi_period)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Golden cross + RSI not overbought
        if (prev["ema_fast"] < prev["ema_slow"] and
                curr["ema_fast"] > curr["ema_slow"] and
                curr["rsi"] < 60):
            return "BUY"

        # Death cross OR overbought
        if (prev["ema_fast"] > prev["ema_slow"] and
                curr["ema_fast"] < curr["ema_slow"]) or curr["rsi"] > 75:
            return "SELL"

        return "HOLD"