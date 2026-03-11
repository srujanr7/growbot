import numpy as np
import pandas as pd
import logging
import os
import ta                          # ← FIXED: moved to top-level, was pandas_ta inside methods

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMNet(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size=10, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMModel:
    SEQ_LEN   = 20
    MODEL_DIR = "ml/models"

    def __init__(self, scrip_code: str):
        self.scrip_code = scrip_code
        self.model      = None
        self.trained    = False
        self.scaler_min = None
        self.scaler_max = None

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — LSTM disabled")
            return

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self._load()

    def _path(self):
        return f"{self.MODEL_DIR}/lstm_{self.scrip_code}.pt"

    def _load(self):
        if not TORCH_AVAILABLE:
            return
        if os.path.exists(self._path()):
            try:
                checkpoint       = torch.load(self._path())
                self.model       = LSTMNet()
                self.model.load_state_dict(checkpoint["model"])
                self.scaler_min  = checkpoint["scaler_min"]
                self.scaler_max  = checkpoint["scaler_max"]
                self.trained     = True
                logger.info(f"✅ LSTM loaded for {self.scrip_code}")
            except Exception as e:
                logger.warning(f"LSTM load error: {e}")

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        # ← FIXED: was `import pandas_ta as ta` inline here
        df       = df.copy()
        df["r"]  = df["close"].pct_change()
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        macd_ind  = ta.trend.MACD(df["close"])
        df["macd"] = macd_ind.macd()

        df["vol_r"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

        bb             = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        bb_upper       = bb.bollinger_hband()
        bb_lower       = bb.bollinger_lband()
        bb_range       = bb_upper - bb_lower
        df["bb_p"]     = (df["close"] - bb_lower) / (bb_range + 1e-9)

        df["atr"] = (
            ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=14
            ) / df["close"]
        )

        df.dropna(inplace=True)

        cols = ["r", "rsi", "macd", "vol_r", "bb_p", "atr",
                "open", "high", "low", "close"]
        return df[cols].values

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_min is None:
            self.scaler_min = X.min(axis=0)
            self.scaler_max = X.max(axis=0)
        rng = self.scaler_max - self.scaler_min
        rng[rng == 0] = 1
        return (X - self.scaler_min) / rng

    def train(self, df: pd.DataFrame, epochs: int = 30):
        if not TORCH_AVAILABLE or len(df) < self.SEQ_LEN + 10:
            return

        try:
            feats = self._get_features(df)
            feats = self._normalize(feats)

            X, y = [], []
            for i in range(len(feats) - self.SEQ_LEN - 1):
                X.append(feats[i:i + self.SEQ_LEN])
                curr  = feats[i + self.SEQ_LEN - 1, -1]
                nxt   = feats[i + self.SEQ_LEN, -1]
                chg   = (nxt - curr) / curr if curr != 0 else 0
                label = 1 if chg > 0.002 else 2 if chg < -0.002 else 0
                y.append(label)

            X = torch.FloatTensor(np.array(X))
            y = torch.LongTensor(np.array(y))

            self.model = LSTMNet(input_size=X.shape[2])
            optimizer  = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion  = nn.CrossEntropyLoss()

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                out  = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            torch.save({
                "model":      self.model.state_dict(),
                "scaler_min": self.scaler_min,
                "scaler_max": self.scaler_max
            }, self._path())

            self.trained = True
            logger.info(f"✅ LSTM trained for {self.scrip_code}")

        except Exception as e:
            logger.error(f"LSTM train error: {e}")

    def predict(self, df: pd.DataFrame) -> dict:
        if not TORCH_AVAILABLE or not self.trained or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0}

        try:
            feats = self._get_features(df)
            feats = self._normalize(feats)

            if len(feats) < self.SEQ_LEN:
                return {"signal": "HOLD", "confidence": 0.0}

            seq  = feats[-self.SEQ_LEN:]
            X    = torch.FloatTensor(seq).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                out   = self.model(X)
                probs = torch.softmax(out, dim=1)[0]
                pred  = int(torch.argmax(probs))
                conf  = float(probs[pred])

            label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal    = label_map.get(pred, "HOLD")

            if conf < 0.55:
                signal = "HOLD"

            return {"signal": signal, "confidence": conf}

        except Exception as e:
            logger.error(f"LSTM predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}

    def save(self):
    torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except:
            pass