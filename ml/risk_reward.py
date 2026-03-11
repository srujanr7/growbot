import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RiskRewardCalculator:
    """
    Calculates ATR-based dynamic stop loss and take profit.
    NOT fixed percentages — adapts to how volatile
    the stock actually is today.

    Default multipliers:
      SL  = entry - (ATR × 1.5)   → tight but not too tight
      TP  = entry + (ATR × 3.0)   → 2:1 reward/risk minimum
    """

    def __init__(self, sl_multiplier: float = 1.5,
                 tp_multiplier: float = 3.0,
                 min_rr_ratio:  float = 2.0):
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.min_rr_ratio  = min_rr_ratio

    def calculate(self, ltp: float,
                  df: pd.DataFrame) -> dict:
        """
        Returns SL, TP, risk/reward per share, and
        whether this trade meets the minimum RR ratio.

        Falls back to fixed 1%/2% if ATR unavailable.
        """
        try:
            atr = float(df["atr"].iloc[-1])
            if np.isnan(atr) or atr <= 0:
                raise ValueError("ATR invalid")
        except Exception:
            # Fallback: 1% SL, 2% TP
            atr = ltp * 0.01 / self.sl_multiplier
            logger.warning(
                "ATR unavailable — using price-based fallback"
            )

        stop_loss   = ltp - (atr * self.sl_multiplier)
        take_profit = ltp + (atr * self.tp_multiplier)

        risk_per_share   = ltp - stop_loss
        reward_per_share = take_profit - ltp

        rr_ratio = (
            reward_per_share / risk_per_share
            if risk_per_share > 0 else 0.0
        )

        sl_pct = risk_per_share   / ltp * 100
        tp_pct = reward_per_share / ltp * 100

        return {
            "stop_loss":        round(stop_loss,   2),
            "take_profit":      round(take_profit, 2),
            "risk_per_share":   round(risk_per_share,   2),
            "reward_per_share": round(reward_per_share, 2),
            "rr_ratio":         round(rr_ratio, 2),
            "sl_pct":           round(sl_pct, 3),
            "tp_pct":           round(tp_pct, 3),
            "atr":              round(atr, 2),
            "acceptable":       rr_ratio >= self.min_rr_ratio,
        }


class TrailingStop:
    """
    Moves stop loss UP as price rises.
    Locks in profit while letting winners run.

    Example on RELIANCE entry ₹2800, ATR ₹25:
      trail_distance = 25 × 1.5 = ₹37.50
      Price → ₹2840 → stop moves to ₹2802.50 (locks +0.09%)
      Price → ₹2875 → stop moves to ₹2837.50 (locks +1.34%)
      Price → ₹2835 → STOP HIT → exit at ₹2837.50
    """

    def __init__(self, entry: float, atr: float,
                 trail_multiplier: float = 1.5):
        self.entry          = entry
        self.atr            = atr
        self.trail_distance = atr * trail_multiplier
        self.highest_price  = entry
        self.current_stop   = entry - self.trail_distance

    def update(self, current_price: float) -> dict:
        """
        Call every cycle with current LTP.
        Returns whether to exit and current locked PnL.
        """
        # Ratchet stop up as price rises — never move it down
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.current_stop  = (
                self.highest_price - self.trail_distance
            )

        stop_hit    = current_price <= self.current_stop
        unreal_pnl  = (current_price - self.entry) / self.entry * 100
        locked_pnl  = (self.current_stop - self.entry) / self.entry * 100

        return {
            "exit_now":       stop_hit,
            "current_stop":   round(self.current_stop,  2),
            "highest_price":  round(self.highest_price, 2),
            "current_price":  round(current_price,      2),
            "unrealized_pnl": round(unreal_pnl, 3),
            "locked_pnl":     round(locked_pnl, 3),
        }


class ExpectedValueFilter:
    """
    EV = (win_rate × reward) − (loss_rate × risk)
    Must be positive to take the trade.
    Negative EV = guaranteed long-term loss regardless of win rate.
    """

    def calculate(self, win_rate: float,
                  risk_amount: float,
                  reward_amount: float) -> dict:
        if risk_amount <= 0:
            return {
                "expected_value": 0.0,
                "ev_pct": 0.0,
                "positive": False,
                "quality": "INVALID"
            }

        loss_rate = 1.0 - win_rate
        ev        = (win_rate * reward_amount) - \
                    (loss_rate * risk_amount)
        ev_pct    = ev / risk_amount * 100

        quality = (
            "EXCELLENT" if ev_pct > 50 else
            "GOOD"      if ev_pct > 20 else
            "MARGINAL"  if ev_pct > 0  else
            "NEGATIVE"
        )

        return {
            "expected_value": round(ev,     2),
            "ev_pct":         round(ev_pct, 2),
            "positive":       ev > 0,
            "quality":        quality,
        }


class KellyCriterion:
    """
    Calculates optimal position size from win rate and RR ratio.
    Always uses Half Kelly to reduce variance.
    Caps at max_pct regardless.

    Negative Kelly = edge is against you = don't trade.
    """

    def calculate(self, win_rate: float,
                  rr_ratio: float,
                  balance: float,
                  max_pct: float = 0.05) -> dict:
        if rr_ratio <= 0 or win_rate <= 0:
            return {
                "full_kelly_pct": 0.0,
                "half_kelly_pct": 0.0,
                "safe_pct":       0.0,
                "allocation":     0.0,
            }

        full_kelly = win_rate - (1.0 - win_rate) / rr_ratio
        half_kelly = full_kelly / 2.0
        safe_pct   = max(0.005, min(half_kelly, max_pct))

        # Negative Kelly = no edge → don't trade
        if full_kelly <= 0:
            safe_pct = 0.0

        return {
            "full_kelly_pct": round(full_kelly * 100, 2),
            "half_kelly_pct": round(half_kelly * 100, 2),
            "safe_pct":       round(safe_pct   * 100, 2),
            "allocation":     round(balance * safe_pct, 2),
        }