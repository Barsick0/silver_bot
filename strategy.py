import numpy as np
from dataclasses import dataclass
from datetime import time as dtime


# =========================================================
# ⚙️ ПАРАМЕТРЫ (КАК В TRADINGVIEW — МЕНЯЙ ЗДЕСЬ)
# =========================================================

@dataclass
class StrategyConfig:

    # === MACHINE LEARNING CORE ===
    length: int = 30          # Lookback Window
    h_param: float = 80       # Smoothness (Bandwidth)
    r_param: float = 3        # Regression Alpha

    # === CHANNEL WIDTH ===
    mult_inner: float = 3
    mult_outer: float = 3

    # === ENTRY ===
    use_close_cross: bool = False

    # === TREND FILTER ===
    use_trend_filter: bool = True
    prd: int = 24
    atr_factor: float = 5
    atr_pd: int = 5

    # === ATR FILTER ===
    use_atr_filter: bool = True
    atr_filter_len: int = 40
    atr_ma_len: int = 100
    atr_mult_thresh: float = 1

    # === SESSION ===
    use_time_filter: bool = True
    session_start: dtime = dtime(9, 30)
    session_end: dtime = dtime(20, 0)
    trade_weekends: bool = True

    # === RISK ===
    sl_perc: float = 0.55
    tp_perc: float = 0.75


# =========================================================
# ATR (как в Pine)
# =========================================================

def calc_atr(high, low, close, period):
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    tr = np.insert(tr, 0, high[0] - low[0])

    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr


# =========================================================
# STRATEGY
# =========================================================

class ChannelTrendATR:

    def __init__(self, cfg=StrategyConfig()):
        self.cfg = cfg

        self.weights = np.array([
            (1 + (i**2)/(2*cfg.r_param*(cfg.h_param**2)))**(-cfg.r_param)
            for i in range(cfg.length)
        ])

        self.w_sum = self.weights.sum()

        # состояние
        self.center = np.nan
        self.t_up = np.nan
        self.t_down = np.nan
        self.trend = 0

        # буфер
        self.h, self.l, self.c, self.src, self.ts = [], [], [], [], []

    # =========================================================

    def next(self, candle):

        cfg = self.cfg

        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        ts = candle["timestamp"]

        src = (h + l + c) / 3

        self.h.append(h)
        self.l.append(l)
        self.c.append(c)
        self.src.append(src)
        self.ts.append(ts)

        if len(self.c) < cfg.length + 2:
            return {"signal": None}

        high = np.array(self.h)
        low = np.array(self.l)
        close = np.array(self.c)
        src_a = np.array(self.src)

        # =====================================================
        # KERNEL (как Pine)
        # =====================================================

        window = src_a[-cfg.length:][::-1]
        y_hat = np.dot(window, self.weights) / self.w_sum

        error = np.mean(np.abs(src_a[-cfg.length:] - y_hat))
        atr = calc_atr(high, low, close, cfg.length)[-1]

        vol = (error + atr) / 2

        lower_outer = y_hat - vol * cfg.mult_outer
        upper_outer = y_hat + vol * cfg.mult_outer

        # =====================================================
        # ПРЕДЫДУЩИЙ КАНАЛ (ВАЖНО!)
        # =====================================================

        window_prev = src_a[-cfg.length-1:-1][::-1]
        y_hat_prev = np.dot(window_prev, self.weights) / self.w_sum

        error_prev = np.mean(np.abs(src_a[-cfg.length-1:-1] - y_hat_prev))
        atr_prev = calc_atr(high[:-1], low[:-1], close[:-1], cfg.length)[-1]

        vol_prev = (error_prev + atr_prev) / 2

        lower_outer_prev = y_hat_prev - vol_prev * cfg.mult_outer
        upper_outer_prev = y_hat_prev + vol_prev * cfg.mult_outer

        # =====================================================
        # ATR FILTER
        # =====================================================

        atr_ok = True
        if cfg.use_atr_filter:
            atr_raw = calc_atr(high, low, close, cfg.atr_filter_len)
            atr_ma = np.mean(atr_raw[-cfg.atr_ma_len:])
            ratio = atr_raw[-1] / atr_ma if atr_ma > 0 else 0
            atr_ok = ratio >= cfg.atr_mult_thresh

        # =====================================================
        # SESSION
        # =====================================================

        time_ok = True
        if cfg.use_time_filter:
            t = ts.time()
            in_session = cfg.session_start <= t <= cfg.session_end
            is_weekend = ts.weekday() >= 5
            time_ok = in_session and (cfg.trade_weekends or not is_weekend)

        # =====================================================
        # SUPERTREND (ПОЛНЫЙ КЛОН)
        # =====================================================

        atr_st = calc_atr(high, low, close, cfg.atr_pd)[-1]

        if len(high) >= cfg.prd * 2 + 1:
            window_h = high[-cfg.prd*2-1:]
            window_l = low[-cfg.prd*2-1:]

            ph = max(window_h[:-cfg.prd]) if window_h[cfg.prd] == max(window_h) else np.nan
            pl = min(window_l[:-cfg.prd]) if window_l[cfg.prd] == min(window_l) else np.nan

            lastpp = ph if not np.isnan(ph) else pl

            if not np.isnan(lastpp):
                self.center = lastpp if np.isnan(self.center) else (self.center * 2 + lastpp) / 3

        up = self.center - cfg.atr_factor * atr_st if not np.isnan(self.center) else np.nan
        dn = self.center + cfg.atr_factor * atr_st if not np.isnan(self.center) else np.nan

        prev_close = close[-2]

        t_up = max(up, self.t_up) if not np.isnan(self.t_up) and prev_close > self.t_up else up
        t_down = min(dn, self.t_down) if not np.isnan(self.t_down) and prev_close < self.t_down else dn

        if not np.isnan(self.t_down) and c > self.t_down:
            trend = 1
        elif not np.isnan(self.t_up) and c < self.t_up:
            trend = -1
        else:
            trend = self.trend if self.trend != 0 else 1

        self.t_up = t_up
        self.t_down = t_down
        self.trend = trend

        # =====================================================
        # ENTRY (100% как Pine)
        # =====================================================

        if cfg.use_close_cross:
            long_cond = c < lower_outer
            short_cond = c > upper_outer
        else:
            long_cond = l < lower_outer
            short_cond = h > upper_outer

        long_prev = low[-2] < lower_outer_prev
        short_prev = high[-2] > upper_outer_prev

        go_long = (
            long_cond and not long_prev
            and atr_ok and time_ok
            and (trend == 1 if cfg.use_trend_filter else True)
        )

        go_short = (
            short_cond and not short_prev
            and atr_ok and time_ok
            and (trend == -1 if cfg.use_trend_filter else True)
        )

        signal = "long" if go_long else "short" if go_short else None

        # =====================================================
        # SL / TP
        # =====================================================

        sl = None
        tp = None

        if signal == "long":
            sl = c * (1 - cfg.sl_perc / 100)
            tp = c * (1 + cfg.tp_perc / 100)

        if signal == "short":
            sl = c * (1 + cfg.sl_perc / 100)
            tp = c * (1 - cfg.tp_perc / 100)

        return {
            "signal": signal,
            "sl": sl,
            "tp": tp,
            "trend": trend
        }