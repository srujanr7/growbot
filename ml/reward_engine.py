import logging

logger = logging.getLogger(__name__)


class RewardEngine:
    """
    Converts real trade outcomes into RL reward signals.

    IMPORTANT: Returns a dict, not a float.
    Callers access reward["reward"], reward["quality"] etc.

    Reward rules:
      Base              = PnL %
      Quick win  ×1.5   — capital used efficiently (≤3 candles)
      Trending win ×1.2 — correctly rode the trend
      Volatile win ×1.3 — bonus for trading hard conditions well
      Clean win  ×1.2   — profitable AND low drawdown (<0.5%)
      Slow loss  ×2.0   — held loser >10 candles, punish hard
      Volatile loss ×1.5 — should have stayed out
      Drawdown penalty  — (drawdown - 2%) × 2 subtracted
      Clamped to [-10, +10]
    """

    def calculate(self,
                  entry:            float,
                  exit_price:       float,
                  held_candles:     int,
                  signal:           dict,
                  max_drawdown_pct: float = 0.0) -> dict:

        if entry <= 0:
            return self._zero_result()

        # Direction-aware PnL
        sig = "BUY"
        if isinstance(signal, dict):
            sig = signal.get("signal", "BUY")

        if sig == "SELL":
            pnl_pct = (entry - exit_price) / entry * 100
        else:
            pnl_pct = (exit_price - entry) / entry * 100

        reward = pnl_pct
        regime = signal.get("regime", "") if isinstance(signal, dict) else ""

        # ── Bonuses ───────────────────────────────────────────

        # Quick win: in and out fast = efficient capital use
        if pnl_pct > 0 and held_candles <= 3:
            reward *= 1.5
            logger.debug("Quick win bonus ×1.5")

        # Won in a trending market — riding the wave correctly
        if pnl_pct > 0 and "TRENDING" in regime:
            reward *= 1.2
            logger.debug("Trending win bonus ×1.2")

        # Won in volatile conditions — harder to do, so reward more
        if pnl_pct > 0 and regime == "VOLATILE":
            reward *= 1.3
            logger.debug("Volatile win bonus ×1.3")

        # Clean win: profitable AND very low drawdown
        if pnl_pct > 0 and max_drawdown_pct < 0.5:
            reward *= 1.2
            logger.debug("Clean win bonus ×1.2")

        # ── Penalties ─────────────────────────────────────────

        # Held a loser too long — punish hard to teach early exits
        if pnl_pct < 0 and held_candles > 10:
            reward *= 2.0
            logger.debug("Slow loss penalty ×2.0")

        # Lost in volatile market — should have stayed out
        if pnl_pct < 0 and regime == "VOLATILE":
            reward *= 1.5
            logger.debug("Volatile loss penalty ×1.5")

        # High drawdown even on a winner — too much risk taken
        if max_drawdown_pct > 2.0:
            drawdown_penalty = (max_drawdown_pct - 2.0) * 2
            reward -= drawdown_penalty
            logger.debug(
                f"Drawdown penalty: -{drawdown_penalty:.2f} "
                f"(drawdown={max_drawdown_pct:.1f}%)"
            )

        # ── Clamp ─────────────────────────────────────────────
        reward = max(-10.0, min(10.0, reward))

        quality = (
            "EXCELLENT" if reward >  3.0 else
            "GOOD"      if reward >  1.0 else
            "NEUTRAL"   if reward > -1.0 else
            "BAD"       if reward > -3.0 else
            "TERRIBLE"
        )

        logger.debug(
            f"RewardEngine: pnl={pnl_pct:+.2f}% "
            f"candles={held_candles} "
            f"drawdown={max_drawdown_pct:.2f}% "
            f"regime={regime} "
            f"→ reward={reward:.3f} ({quality})"
        )

        return {
            "reward":       round(reward, 4),
            "pnl_pct":      round(pnl_pct, 3),
            "held_candles": held_candles,
            "profitable":   pnl_pct > 0,
            "quality":      quality,
        }

    @staticmethod
    def _zero_result() -> dict:
        return {
            "reward":       0.0,
            "pnl_pct":      0.0,
            "held_candles": 0,
            "profitable":   False,
            "quality":      "NEUTRAL",
        }
