import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import os
import joblib
import ta                          # ← FIXED: was pandas_ta

logger = logging.getLogger(__name__)


class LGBMModel:
    MODEL_DIR = "ml/models"

    def __init__(self, scrip_code: str):
        self.scrip_code = scrip_code
        self.model      = None
        self.trained    = False
        self._load()

    def _path(self):
        return f"{self.MODEL_DIR}/lgbm_{self.scrip_code}.pkl"

    def _load(self):
        if os.path.exists(self._path()):
            try:
                self.model   = joblib.load(self._path())
                self.trained = True
                logger.info(f"✅ LGBM loaded for {self.scrip_code}")
            except Exception as e:
                logger.warning(f"LGBM load error: {e}")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Trend ─────────────────────────────────────────────
        df["ema_9"]     = ta.trend.ema_indicator(df["close"], window=9)
        df["ema_21"]    = ta.trend.ema_indicator(df["close"], window=21)
        df["ema_50"]    = ta.trend.ema_indicator(df["close"], window=50)
        df["ema_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)

        # ── Momentum ──────────────────────────────────────────
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["roc"] = df["close"].pct_change(10) * 100          # ← FIXED: ta has no roc()
        df["mom"] = df["close"].diff(10)                       # ← FIXED: ta has no mom()

        # ── MACD ──────────────────────────────────────────────
        macd_ind     = ta.trend.MACD(df["close"])
        df["macd"]   = macd_ind.macd()
        df["macd_s"] = macd_ind.macd_signal()
        df["macd_h"] = macd_ind.macd_diff()

        # ── Bollinger Bands ───────────────────────────────────
        bb             = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        bb_upper       = bb.bollinger_hband()
        bb_lower       = bb.bollinger_lband()
        bb_range       = bb_upper - bb_lower
        df["bb_pct"]   = (df["close"] - bb_lower) / (bb_range + 1e-9)

        # ── ATR ───────────────────────────────────────────────
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # ── Volume ────────────────────────────────────────────
        df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
        df["obv"]       = ta.volume.on_balance_volume(df["close"], df["volume"])

        # ── Stochastic ────────────────────────────────────────
        stoch         = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"]
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ── Price ratios ──────────────────────────────────────
        df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
        df["oc_ratio"] = (df["close"] - df["open"]) / df["close"]

        # ── Target ────────────────────────────────────────────
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        df.dropna(inplace=True)
        return df

    FEATURE_COLS = [
        "ema_cross", "rsi", "roc", "mom",
        "macd", "macd_s", "macd_h",
        "bb_pct", "atr", "vol_ratio",
        "stoch_k", "stoch_d",
        "hl_ratio", "oc_ratio"
    ]

    def train(self, df: pd.DataFrame):
        if len(df) < 50:
            return
        try:
            df = self._build_features(df)
            X  = df[self.FEATURE_COLS].values
            y  = df["target"].values

            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            self.model.fit(X, y)

            os.makedirs(self.MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, self._path())
            self.trained = True
            logger.info(f"✅ LGBM trained for {self.scrip_code}")

        except Exception as e:
            logger.error(f"LGBM train error: {e}")

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.trained or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0}

        try:
            df   = self._build_features(df)
            X    = df[self.FEATURE_COLS].values[-1:]
            prob = self.model.predict_proba(X)[0]
            conf = float(max(prob))
            pred = int(self.model.predict(X)[0])

            if conf < 0.60:
                return {"signal": "HOLD", "confidence": conf}

            signal = "BUY" if pred == 1 else "SELL"
            return {"signal": signal, "confidence": conf}

        except Exception as e:
            logger.error(f"LGBM predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}

    def save(self):
        joblib.dump(self.model, self.model_path)

    def load(self):
        try:
            self.model = joblib.load(self.model_path)
        except:

            pass
