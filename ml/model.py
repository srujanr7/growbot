import numpy as np
import pandas as pd
import joblib
import logging
import os
import gymnasium as gym
from gymnasium import spaces
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
    logger.info("PPO/RL libraries loaded successfully")
except ImportError:
    RL_AVAILABLE = False
    logger.warning("stable-baselines3 not installed. Running XGBoost only.")


class XGBSignalModel:
    """
    Predicts BUY (1) / SELL (0) signal from features.
    Uses TimeSeriesSplit to avoid lookahead bias.
    """

    def __init__(self, model_path="ml/xgb_model.pkl"):
        self.model_path   = model_path
        self.scaler       = StandardScaler()
        self.model        = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        self.feature_cols = None
        self.is_trained   = False

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train on historical features with time-series CV."""
        df = features.copy()
        df["_label"] = labels
        df = df.dropna()

        X = df.drop(columns=["_label"])
        y = df["_label"].astype(int)

        self.feature_cols = X.columns.tolist()
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr,  X_val = X.iloc[train_idx],  X.iloc[val_idx]
            y_tr,  y_val = y.iloc[train_idx],  y.iloc[val_idx]
            X_tr_sc      = self.scaler.fit_transform(X_tr)
            X_val_sc     = self.scaler.transform(X_val)

            self.model.fit(
                X_tr_sc, y_tr,
                eval_set=[(X_val_sc, y_val)],
                verbose=False
            )
            preds = self.model.predict(X_val_sc)
            logger.info(
                f"Fold {fold+1}:\n"
                f"{classification_report(y_val, preds)}"
            )

        self.is_trained = True
        self.save()
        logger.info(f"XGB model trained on {len(X)} samples")

    def predict(self, features: pd.DataFrame) -> dict:
        """Returns signal and confidence. HOLD if not trained or feature mismatch."""
        if not self.is_trained or self.feature_cols is None:
            return {"signal": "HOLD", "confidence": 0.0}

        try:
            # Guard: only keep columns the model was trained on
            available = [c for c in self.feature_cols
                         if c in features.columns]
            if len(available) < len(self.feature_cols):
                missing = set(self.feature_cols) - set(available)
                logger.warning(
                    f"XGB predict: missing features {missing}, "
                    f"returning HOLD"
                )
                return {"signal": "HOLD", "confidence": 0.0}

            row        = features[self.feature_cols].iloc[[-1]]
            row_sc     = self.scaler.transform(row)
            proba      = self.model.predict_proba(row_sc)[0]
            confidence = float(max(proba))
            label      = int(self.model.predict(row_sc)[0])

            if confidence < 0.60:
                return {"signal": "HOLD", "confidence": confidence}

            return {
                "signal":     "BUY" if label == 1 else "SELL",
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"XGB predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model":        self.model,
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols
        }, self.model_path)
        joblib.dump(self.model, self.model_path)
        logger.info(f"XGB model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            try:
                data              = joblib.load(self.model_path)
                self.model        = data["model"]
                self.scaler       = data["scaler"]
                self.feature_cols = data["feature_cols"]
                self.is_trained   = True
                logger.info(f"XGB model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"XGB load error: {e}")
        else:
            logger.warning(
                f"No XGB model at {self.model_path}. "
                f"Will train from scratch."
            )


class TradingEnv(gym.Env):
    """
    Custom Gym environment for RL trading.
    State:   feature vector + position flag + unrealized PnL
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame,
                 prices: pd.Series,
                 initial_balance: float = 100_000):
        super().__init__()
        self.features  = features.reset_index(drop=True)
        self.prices    = prices.reset_index(drop=True)
        self.n_steps   = len(features)
        self.init_bal  = initial_balance

        n_features = features.shape[1] + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _obs(self):
        row   = self.features.iloc[
            self.current_step
        ].values.astype(np.float32)
        price = self.prices.iloc[self.current_step]
        unrealized = (
            (price - self.entry_price) / self.entry_price
            if self.position else 0.0
        )
        return np.append(row, [float(self.position), unrealized])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position     = 0
        self.entry_price  = 0.0
        self.balance      = self.init_bal
        self.hold_count   = 0
        return self._obs(), {}

    def step(self, action):
        price  = self.prices.iloc[self.current_step]
        reward = 0.0

        if action == 1 and self.position == 0:
            self.position    = 1
            self.entry_price = price
            self.hold_count  = 0

        elif action == 2 and self.position == 1:
            pnl              = (price - self.entry_price) / self.entry_price
            reward           = pnl * 100
            self.balance    *= (1 + pnl)
            self.position    = 0
            self.entry_price = 0.0
            self.hold_count  = 0

        elif action == 0 and self.position == 1:
            self.hold_count += 1
            reward           = -0.001 * self.hold_count

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        return self._obs(), reward, done, False, {}


class RLAgent:
    """
    PPO Reinforcement Learning agent.
    Returns HOLD safely if PPO not available.
    """

    def __init__(self, model_path="ml/ppo_model"):
        self.model_path = model_path
        self.model      = None
        self.available  = RL_AVAILABLE

    def train(self, features: pd.DataFrame,
              prices: pd.Series,
              timesteps: int = 300_000):
        if not self.available:
            logger.warning(
                "Skipping RL training — "
                "stable-baselines3 not installed"
            )
            return
        try:
            env = DummyVecEnv([
                lambda: TradingEnv(features, prices)
            ])
            self.model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1
            )
            self.model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False
            )
            self.model.save(self.model_path)
            logger.info(f"PPO model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"RL training error: {e}")

    def load(self):
        if not self.available:
            return
        try:
            if os.path.exists(f"{self.model_path}.zip"):
                self.model = PPO.load(self.model_path)
                logger.info(
                    f"PPO model loaded from {self.model_path}"
                )
            else:
                logger.warning(
                    f"No PPO model at {self.model_path}"
                )
        except Exception as e:
            logger.error(f"PPO load error: {e}")

    def predict(self, obs: np.ndarray) -> int:
        if not self.available or self.model is None:
            return 0
        try:
            action, _ = self.model.predict(
                obs, deterministic=True
            )
            return int(action)
        except Exception as e:
            logger.error(f"PPO predict error: {e}")
            return 0