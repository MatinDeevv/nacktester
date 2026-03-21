"""
Example: London Session Breakout
Trade the breakout of the Asian session range at London open.
This is based on the structural edge from session timing patterns.
"""

from strategy_runtime import Strategy


class LondonBreakout(Strategy):
    name = "London Breakout"

    # ── Parameters ──
    size = 0.01
    asian_start_hour = 0   # UTC hour for Asian session start
    asian_end_hour = 7     # UTC hour for Asian session end
    london_open_hour = 8   # UTC hour for London open
    entry_window_hours = 2 # How many hours after London open to enter
    sl_buffer_pct = 0.001  # SL buffer beyond Asian range (0.1%)
    tp_rr = 2.0            # Take profit as multiple of risk

    def on_init(self, ctx):
        self._asian_high = 0
        self._asian_low = 999999
        self._in_asian = False
        self._breakout_ready = False
        self._traded_today = False

    def on_bar(self, ctx):
        hour = ctx.bar.name.hour
        price = ctx.bar["close"]
        high = ctx.bar["high"]
        low = ctx.bar["low"]

        # New day reset (rough: when we enter Asian session)
        if hour == self.asian_start_hour and not self._in_asian:
            self._asian_high = 0
            self._asian_low = 999999
            self._in_asian = True
            self._breakout_ready = False
            self._traded_today = False

        # Build Asian range
        if self._in_asian and self.asian_start_hour <= hour < self.asian_end_hour:
            self._asian_high = max(self._asian_high, high)
            self._asian_low = min(self._asian_low, low)

        # Mark breakout ready at London open
        if hour == self.london_open_hour and self._in_asian:
            self._in_asian = False
            self._breakout_ready = True

        # Trade the breakout
        if (self._breakout_ready and not self._traded_today and
            not ctx.has_position and
            self.london_open_hour <= hour < self.london_open_hour + self.entry_window_hours):

            rng = self._asian_high - self._asian_low
            if rng < 0.5:  # Skip tiny ranges
                return

            buffer = price * self.sl_buffer_pct

            # Breakout long
            if high > self._asian_high + buffer:
                sl = self._asian_low - buffer
                risk = price - sl
                tp = price + risk * self.tp_rr
                ctx.buy(size=self.size, sl=sl, tp=tp)
                self._traded_today = True

            # Breakout short
            elif low < self._asian_low - buffer:
                sl = self._asian_high + buffer
                risk = sl - price
                tp = price - risk * self.tp_rr
                ctx.sell(size=self.size, sl=sl, tp=tp)
                self._traded_today = True

        # End of day close
        if ctx.has_position and hour >= 20:
            ctx.close("end_of_day")
