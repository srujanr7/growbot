import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Tracks daily PnL, trade count, and enforces all risk limits.
    Limits can be adjusted live via Telegram commands.
    """

    def __init__(self, max_position_pct: float = 0.02,
                 daily_loss_limit_pct: float = 0.03):
        self.max_position_pct     = max_position_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct

        self.daily_pnl          = 0.0
        self.trades_today       = 0
        self.max_trades_per_day = 10

        # Runtime overrides set via Telegram
        self.per_trade_limit = None   # ₹ max per trade
        self.daily_loss_cap  = None   # ₹ absolute daily loss cap

    def can_trade(self, balance: float) -> bool:
        limit = self.get_daily_limit(balance)
        if self.daily_pnl < -limit:
            logger.warning("🔴 Daily loss limit hit — kill switch.")
            return False
        if self.trades_today >= self.max_trades_per_day:
            logger.warning("🔴 Max daily trades reached.")
            return False
        return True

    def get_daily_limit(self, balance: float) -> float:
        if self.daily_loss_cap is not None:
            return self.daily_loss_cap
        return balance * self.daily_loss_limit_pct

    def apply_per_trade_limit(self, qty: int, ltp: float) -> int:
        if self.per_trade_limit is None:
            return qty
        max_qty = int(self.per_trade_limit / ltp)
        clamped = max(1, min(qty, max_qty))
        if clamped < qty:
            logger.info(
                f"Per-trade limit ₹{self.per_trade_limit:,.0f} "
                f"applied: qty {qty} → {clamped}"
            )
        return clamped

    def position_size(self, balance: float, price: float) -> int:
        allocation = balance * self.max_position_pct
        return max(1, int(allocation / price))

    def update_pnl(self, pnl: float):
        self.daily_pnl    += pnl
        self.trades_today += 1
        logger.info(
            f"Daily PnL: ₹{self.daily_pnl:.2f} | "
            f"Trades: {self.trades_today}"
        )

    def reset_daily(self):
        self.daily_pnl    = 0.0
        self.trades_today = 0
        logger.info("🔄 Risk counters reset")