"""
SuperTrend Dual RMA Strategy
Python port of the Pine Script v6 strategy.

Interfaces expected by bot.py:
    StrategyConfig  — dataclass with all tunable parameters
    ChannelTrendATR — strategy class with .warmup(df) and .next(candle) -> dict
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Literal, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("strategy")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # ── SuperTrend Dual RMA ──────────────────
    rma1_length: int   = 13
    rma2_length: int   = 24
    atr_period:  int   = 4
    atr_mult:    float = 3.0

    # "with_trend" | "against_trend" | "both"
    trade_dir:   str   = "both"

    # "on_touch" | "on_return"
    entry_mode:  str   = "on_touch"

    # ── Risk Management ──────────────────────
    sl_perc:     float = 0.75    # Stop Loss %
    tp_mode:     str   = "fixed"  # "fixed" | "channel_border" | "channel_middle"
    tp_perc:     float = 0.75    # TP % (used only when tp_mode == "fixed")
    use_pyramid: bool  = True   # max 2 entries per direction
    cooldown_bars: int = 2      # min bars between entries
    touch_tolerance: float = 0.0  # price units; keep 0.0 for exact Pine on identical data

    # ── ATR Filter ───────────────────────────
    use_atr_filter: bool  = True
    atr_flt_len:    int   = 14
    atr_flt_smooth: str   = "ema"   # "rma" | "sma" | "ema" | "wma"
    atr_flt_min:    float = 0.14     # price units; entries blocked below this

    # ── Time Filter ──────────────────────────
    use_time_filter: bool  = True
    time_start:      dtime = field(default_factory=lambda: dtime(10, 0))
    time_end:        dtime = field(default_factory=lambda: dtime(23, 0))
    block_weekends:  bool  = True

    # ── Derived (read-only, used by bot.py) ──
    @property
    def length(self) -> int:
        """Minimum bars required before strategy is ready."""
        return max(self.rma1_length, self.rma2_length, self.atr_period, self.atr_flt_len) + 1


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rma(series: np.ndarray, length: int) -> np.ndarray:
    """Wilder's RMA (EMA with alpha = 1/length), matches Pine ta.rma()."""
    alpha = 1.0 / length
    out = np.full_like(series, np.nan)
    valid_idx = np.flatnonzero(~np.isnan(series))
    if len(valid_idx) < length:
        return out
    # Pine ta.rma() ignores leading na values and seeds with SMA of first valid values.
    start = valid_idx[length - 1]
    seed_values = series[valid_idx[:length]]
    out[start] = np.mean(seed_values)
    for i in range(start + 1, len(series)):
        if np.isnan(series[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def _smooth(series: np.ndarray, length: int, method: str) -> np.ndarray:
    m = method.lower()
    if m == "sma":
        return pd.Series(series).rolling(length).mean().to_numpy()
    if m == "ema":
        return pd.Series(series).ewm(span=length, adjust=False).mean().to_numpy()
    if m == "wma":
        weights = np.arange(1, length + 1, dtype=float)
        weights /= weights.sum()
        return np.convolve(series, weights[::-1], mode="full")[: len(series)]
    return _rma(series, length)  # default: rma


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int) -> np.ndarray:
    """True Range → RMA (matches Pine ta.atr())."""
    tr = np.empty_like(high)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    return _rma(tr, period)


def _vwap_rma(hlc3: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    """Volume-weighted RMA: rma(hlc3*vol, n) / rma(vol, n) — matches Pine."""
    num = _rma(hlc3 * volume, length)
    den = _rma(volume,        length)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den != 0, num / den, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL STATE  (one value updated per bar, O(1) per tick)
# ─────────────────────────────────────────────────────────────────────────────

class _RmaInc:
    """Incremental Wilder RMA."""
    def __init__(self, length: int):
        self.alpha = 1.0 / length
        self.length = length
        self.value: Optional[float] = None
        self._buf: deque = deque(maxlen=length)

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._buf.append(x)
            if len(self._buf) == self.length:
                self.value = float(np.mean(self._buf))
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class _AtrInc:
    """Incremental ATR (True Range → RMA)."""
    def __init__(self, period: int):
        self._rma = _RmaInc(period)
        self._prev_close: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev_close is None:
            self.value = self._rma.update(high - low)
            self._prev_close = close
            return self.value
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low  - self._prev_close),
        )
        self._prev_close = close
        self.value = self._rma.update(tr)
        return self.value


class _SmoothInc:
    """Incremental smoothed ATR for the filter (RMA/EMA; SMA/WMA approximate)."""
    def __init__(self, length: int, method: str):
        self.method = method.lower()
        self.length = length
        self._rma = _RmaInc(length)
        # EMA
        self._ema_val: Optional[float] = None
        self._ema_alpha = 2.0 / (length + 1)
        # SMA / WMA buffer
        self._buf: deque = deque(maxlen=length)
        self.value: Optional[float] = None

    def update(self, tr: float) -> Optional[float]:
        if self.method == "rma":
            self.value = self._rma.update(tr)
        elif self.method == "ema":
            if self._ema_val is None:
                self._buf.append(tr)
                if len(self._buf) == self.length:
                    self._ema_val = float(np.mean(self._buf))
                    self.value = self._ema_val
            else:
                self._ema_val = self._ema_alpha * tr + (1 - self._ema_alpha) * self._ema_val
                self.value = self._ema_val
        elif self.method in ("sma", "wma"):
            self._buf.append(tr)
            if len(self._buf) == self.length:
                arr = np.array(self._buf)
                if self.method == "sma":
                    self.value = float(arr.mean())
                else:
                    w = np.arange(1, self.length + 1, dtype=float)
                    self.value = float((arr * w).sum() / w.sum())
        return self.value


class _AtrFilterInc:
    """Incremental ATR filter value (tr smoothed by chosen method)."""
    def __init__(self, length: int, method: str):
        self._smooth = _SmoothInc(length, method)
        self._prev_close: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev_close is None:
            self.value = self._smooth.update(high - low)
            self._prev_close = close
            return self.value
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low  - self._prev_close),
        )
        self._prev_close = close
        self.value = self._smooth.update(tr)
        return self.value


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STRATEGY CLASS
# ─────────────────────────────────────────────────────────────────────────────

Signal = Literal["long", "short"]


class ChannelTrendATR:
    """
    SuperTrend Dual RMA strategy.

    Usage:
        cfg = StrategyConfig()
        st  = ChannelTrendATR(cfg)
        st.warmup(df)           # df: DataFrame with OHLCV columns
        result = st.next(candle)  # candle: dict with OHLCV + timestamp
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

        # ── incremental indicators ─────────────
        # RMA numerators / denominators for vwap-rma
        self._rma1_num = _RmaInc(cfg.rma1_length)
        self._rma1_den = _RmaInc(cfg.rma1_length)
        self._rma2_num = _RmaInc(cfg.rma2_length)
        self._rma2_den = _RmaInc(cfg.rma2_length)

        self._atr     = _AtrInc(cfg.atr_period)
        self._atr_flt = _AtrFilterInc(cfg.atr_flt_len, cfg.atr_flt_smooth)

        # ── state ──────────────────────────────
        self._trend_dir: float = 1.0          # +1 = bullish, -1 = bearish

        self._upper_band: Optional[float] = None
        self._lower_band: Optional[float] = None
        self._basis:      Optional[float] = None
        self._rma1_val:   Optional[float] = None
        self._rma2_val:   Optional[float] = None

        # entry signal state
        self._pending_long:  bool = False
        self._pending_short: bool = False

        # pyramiding counters (reset when position closes)
        self._long_count:  int = 0
        self._short_count: int = 0

        # cooldown
        self._last_entry_bar: int = -cfg.cooldown_bars - 1
        self._bar_index:      int = 0

        # deduplication (same as bot.py self.last_signal)
        self._last_signal: Optional[str] = None

        self._ready: bool = False

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def warmup(self, df: pd.DataFrame) -> None:
        """
        Feed historical OHLCV DataFrame to initialise all indicators.
        Columns required: open, high, low, close, volume, timestamp.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # batch-compute everything for the full history
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        close  = df["close"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        hlc3   = (high + low + close) / 3.0

        rma1 = _vwap_rma(hlc3, volume, self.cfg.rma1_length)
        rma2 = _vwap_rma(hlc3, volume, self.cfg.rma2_length)
        atr  = _atr(high, low, close, self.cfg.atr_period)

        # replay to set incremental state
        for i in range(len(df)):
            hlc3_i = hlc3[i]
            vol_i  = volume[i]

            self._rma1_num.update(hlc3_i * vol_i)
            self._rma1_den.update(vol_i)
            self._rma2_num.update(hlc3_i * vol_i)
            self._rma2_den.update(vol_i)
            self._atr.update(high[i], low[i], close[i])
            self._atr_flt.update(high[i], low[i], close[i])

        # set final band / trend state from batch arrays
        last = len(df) - 1
        if not np.isnan(rma1[last]) and not np.isnan(rma2[last]) and not np.isnan(atr[last]):
            self._rma1_val  = rma1[last]
            self._rma2_val  = rma2[last]
            self._basis     = (rma1[last] + rma2[last]) / 2.0
            self._upper_band = self._basis + self.cfg.atr_mult * atr[last]
            self._lower_band = self._basis - self.cfg.atr_mult * atr[last]

            # replay trend direction over full history
            trend = 1.0
            for i in range(last + 1):
                if np.isnan(rma1[i]) or np.isnan(atr[i]):
                    continue
                b  = (rma1[i] + rma2[i]) / 2.0
                ub = b + self.cfg.atr_mult * atr[i]
                lb = b - self.cfg.atr_mult * atr[i]
                if close[i] > ub:
                    trend = 1.0
                elif close[i] < lb:
                    trend = -1.0
            self._trend_dir = trend
            self._ready = True
            log.info("✅ Strategy warmed up | trend=%+d | upper=%.5f | lower=%.5f",
                     int(self._trend_dir), self._upper_band, self._lower_band)
        else:
            log.warning("⚠️ Warmup incomplete — not enough data for indicators")

        self._bar_index = len(df)

    def notify_position_closed(self) -> None:
        """Call from bot when a position is fully closed to reset pyramid counters."""
        self._long_count  = 0
        self._short_count = 0
        self._last_signal = None

    def next(self, candle: dict) -> dict:
        """
        Process one completed candle and return a signal dict:
        {
            "signal":   "long" | "short" | None,
            "trend":    +1 | -1,
            "atr_ok":   bool,
            "time_ok":  bool,
            "upper":    float,
            "lower":    float,
            "basis":    float,
            "sl":       float | None,
            "tp":       float | None,
        }
        """
        o = float(candle["open"])
        h = float(candle["high"])
        lo = float(candle["low"])
        c = float(candle["close"])
        vol = float(candle["volume"])
        ts: datetime = candle["timestamp"]

        self._bar_index += 1

        # ── update indicators ──────────────────────────────────────────────
        hlc3 = (h + lo + c) / 3.0

        n1 = self._rma1_num.update(hlc3 * vol)
        d1 = self._rma1_den.update(vol)
        n2 = self._rma2_num.update(hlc3 * vol)
        d2 = self._rma2_den.update(vol)

        atr_val     = self._atr.update(h, lo, c)
        atr_flt_val = self._atr_flt.update(h, lo, c)

        if None in (n1, d1, n2, d2, atr_val) or d1 == 0 or d2 == 0:
            return self._empty_result()

        rma1 = n1 / d1
        rma2 = n2 / d2

        basis      = (rma1 + rma2) / 2.0
        upper_band = basis + self.cfg.atr_mult * atr_val
        lower_band = basis - self.cfg.atr_mult * atr_val

        # ── trend direction ───────────────────────────────────────────────
        if c > upper_band:
            self._trend_dir = 1.0
        elif c < lower_band:
            self._trend_dir = -1.0

        self._upper_band = upper_band
        self._lower_band = lower_band
        self._basis      = basis
        self._rma1_val   = rma1
        self._rma2_val   = rma2
        self._ready      = True

        # ── ATR filter ────────────────────────────────────────────────────
        atr_ok = (
            not self.cfg.use_atr_filter
            or atr_flt_val is None
            or atr_flt_val >= self.cfg.atr_flt_min
        )

        # ── Time filter ───────────────────────────────────────────────────
        time_ok = self._check_time(ts)

        # ── entry signal ──────────────────────────────────────────────────
        signal = self._compute_signal(h, lo, c, upper_band, lower_band,
                                      atr_ok, time_ok)

        # ── TP / SL levels ────────────────────────────────────────────────
        sl, tp = None, None
        if signal is not None:
            sl = c * (1.0 - self.cfg.sl_perc / 100.0) if signal == "long" \
                 else c * (1.0 + self.cfg.sl_perc / 100.0)
            tp = self._tp_level(signal, c, upper_band, lower_band, basis)

        return {
            "signal":  signal,
            "trend":   int(self._trend_dir),
            "atr_ok":  atr_ok,
            "time_ok": time_ok,
            "upper":   upper_band,
            "lower":   lower_band,
            "basis":   basis,
            "sl":      sl,
            "tp":      tp,
        }

    # ── PRIVATE ────────────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "signal": None, "trend": int(self._trend_dir),
            "atr_ok": True, "time_ok": True,
            "upper": self._upper_band, "lower": self._lower_band,
            "basis": self._basis, "sl": None, "tp": None,
        }

    def _check_time(self, ts: datetime) -> bool:
        if not self.cfg.use_time_filter:
            return True

        # weekends
        if self.cfg.block_weekends and ts.weekday() >= 5:  # 5=Sat, 6=Sun
            return False

        # time range
        t = ts.time().replace(second=0, microsecond=0)
        return self.cfg.time_start <= t < self.cfg.time_end

    def _compute_signal(
        self,
        high: float, low: float, close: float,
        upper_band: float, lower_band: float,
        atr_ok: bool, time_ok: bool,
    ) -> Optional[str]:

        cfg = self.cfg

        # ── touch / return detection ──────────────────────────────────────
        tol = cfg.touch_tolerance

        if low <= lower_band + tol:
            self._pending_long = True
        if high >= upper_band - tol:
            self._pending_short = True

        if cfg.entry_mode == "on_touch":
            long_signal_base  = low  <= lower_band + tol
            short_signal_base = high >= upper_band - tol
        else:  # on_return
            long_signal_base  = self._pending_long  and close > lower_band and low  > lower_band
            short_signal_base = self._pending_short and close < upper_band and high < upper_band

        if long_signal_base:
            self._pending_long  = False
        if short_signal_base:
            self._pending_short = False

        # ── direction filter ──────────────────────────────────────────────
        td = self.cfg.trade_dir
        long_dir_ok  = (td == "both"
                        or (td == "with_trend"    and self._trend_dir ==  1.0)
                        or (td == "against_trend" and self._trend_dir == -1.0))
        short_dir_ok = (td == "both"
                        or (td == "with_trend"    and self._trend_dir == -1.0)
                        or (td == "against_trend" and self._trend_dir ==  1.0))

        # ── cooldown ──────────────────────────────────────────────────────
        cool_ok = (self._bar_index - self._last_entry_bar) >= cfg.cooldown_bars

        # ── pyramiding ────────────────────────────────────────────────────
        pyr_long_ok  = not cfg.use_pyramid or self._long_count  < 2
        pyr_short_ok = not cfg.use_pyramid or self._short_count < 2

        # ── final gates ───────────────────────────────────────────────────
        long_entry  = (long_signal_base  and long_dir_ok  and
                       atr_ok and time_ok and cool_ok and pyr_long_ok)
        short_entry = (short_signal_base and short_dir_ok and
                       atr_ok and time_ok and cool_ok and pyr_short_ok)

        if long_entry:
            self._long_count      += 1
            self._last_entry_bar   = self._bar_index
            return "long"

        if short_entry:
            self._short_count     += 1
            self._last_entry_bar   = self._bar_index
            return "short"

        return None

    def _tp_level(
        self,
        signal: str,
        close: float,
        upper_band: float,
        lower_band: float,
        basis: float,
    ) -> float:
        mode = self.cfg.tp_mode
        if signal == "long":
            if mode == "fixed":
                return close * (1.0 + self.cfg.tp_perc / 100.0)
            if mode == "channel_border":
                return upper_band
            return basis  # channel_middle
        else:  # short
            if mode == "fixed":
                return close * (1.0 - self.cfg.tp_perc / 100.0)
            if mode == "channel_border":
                return lower_band
            return basis
