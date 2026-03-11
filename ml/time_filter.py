from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimeFilter:
    """
    Filters trades based on time of day.
    NSE has predictable patterns:
    - First 15 min: very volatile, avoid
    - 10:00-11:30: best momentum window
    - 12:00-13:30: lunch lull, low volume
    - 14:00-15:00: second best window
    - Last 10 min: avoid (square-off chaos)
    """

    WINDOWS = [
        # (start, end, multiplier, label)
        ("09:15", "09:30", 0.0,  "AVOID — opening volatility"),
        ("09:30", "10:00", 0.6,  "CAUTION — settling"),
        ("10:00", "11:30", 1.0,  "BEST — morning momentum"),
        ("11:30", "12:00", 0.8,  "GOOD — pre-lunch"),
        ("12:00", "13:30", 0.5,  "LOW — lunch lull"),
        ("13:30", "14:00", 0.7,  "MODERATE — resuming"),
        ("14:00", "15:00", 0.9,  "GOOD — afternoon momentum"),
        ("15:00", "15:10", 0.3,  "AVOID — pre-close"),
    ]

    def get_multiplier(self) -> dict:
        now = datetime.now().strftime("%H:%M")
        for start, end, mult, label in self.WINDOWS:
            if start <= now < end:
                return {
                    "multiplier": mult,
                    "label":      label,
                    "should_trade": mult > 0.0
                }
        return {"multiplier": 0.0, "label": "MARKET CLOSED",
                "should_trade": False}