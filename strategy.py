"""
strategy.py — Полный порт Pine Script стратегии "Channel Trend ATR"
Все параметры соответствуют настройкам для SV1! на M1
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import time as dtime


# ─────────────────────────────────────────────
# ПАРАМЕТРЫ СТРАТЕГИИ (из настроек для SV1! M1)
# ─────────────────────────────────────────────
@dataclass
class StrategyConfig:
    # Machine Learning Core
    length:     int   = 22
    h_param:    float = 30
    r_param:    float = 3

    # Channel Width
    mult_inner: float = 3
    mult_outer: float = 3

    # Entry Condition
    use_close_cross: bool = False   # True = вход по закрытию за каналом

    # Trend Filter — Pivot Point SuperTrend
    use_trend_filter: bool  = True
    prd:              int   = 5
    atr_factor:       float = 3
    atr_pd:           int   = 6

    # ATR Volatility Filter
    use_atr_filter:   bool  = True
    atr_filter_len:   int   = 20
    atr_ma_len:       int   = 25
    atr_mult_thresh:  float = 1.0

    # Session Time Filter (московское время биржи)
    use_time_filter:  bool  = True
    session_start:    dtime = dtime(9, 30)
    session_end:      dtime = dtime(20, 0)
    trade_weekends:   bool  = False

    # Risk Management
    sl_perc: float = 0.55
    tp_perc: float = 0.75


# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def _true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    hl  = high - low
    hcp = np.abs(high - prev_close)
    lcp = np.abs(low  - prev_close)
    return np.maximum(hl, np.maximum(hcp, lcp))


def _rma(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder RMA — точный аналог ta.atr() в Pine Script."""
    result = np.full(len(series), np.nan)
    alpha  = 1.0 / period
    # Первое значение — простая средняя за period баров
    if len(series) < period:
        return result
    result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = _true_range(high, low, prev_close)
    tr[0] = high[0] - low[0]
    return _rma(tr, period)


def kernel_weight(idx: int, h: float, r: float) -> float:
    """Rational Quadratic kernel weight."""
    d = float(idx ** 2)
    return (1.0 + d / (2.0 * r * h ** 2)) ** (-r)


def pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    """Точный аналог ta.pivothigh(left, right) из Pine Script."""
    result = np.full(len(series), np.nan)
    for i in range(left + right, len(series)):
        window = series[i - left - right: i + 1]
        peak   = series[i - right]
        if not np.isnan(peak) and peak == np.nanmax(window):
            result[i] = peak
    return result


def pivot_low(series: np.ndarray, left: int, right: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    for i in range(left + right, len(series)):
        window = series[i - left - right: i + 1]
        trough = series[i - right]
        if not np.isnan(trough) and trough == np.nanmin(window):
            result[i] = trough
    return result


# ─────────────────────────────────────────────
# ОСНОВНОЙ КЛАСС СТРАТЕГИИ
# ─────────────────────────────────────────────

class ChannelTrendATR:
    """
    Полный порт Pine Script стратегии.
    Работает в двух режимах:
      - batch(df)  : векторный расчёт по истории (бэктест / прогрев)
      - update(candle): пошаговое добавление одной свечи (live-торговля)
    """

    def __init__(self, cfg: StrategyConfig = StrategyConfig()):
        self.cfg = cfg
        self._weights = np.array(
            [kernel_weight(i, cfg.h_param, cfg.r_param) for i in range(cfg.length)]
        )
        self._w_sum = self._weights.sum()

        # Состояние SuperTrend (обновляется пошагово)
        self._center   = np.nan
        self._t_up     = np.nan
        self._t_down   = np.nan
        self._trend    = 0
        self._prev_trend = 0

        # Буфер последних свечей для инкрементального расчёта
        self._buf_high  : list = []
        self._buf_low   : list = []
        self._buf_close : list = []
        self._buf_src   : list = []
        self._buf_ts    : list = []   # datetime timestamps

    # ── Публичный метод: прогрев по истории ──────────────────────────────

    def warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Принимает DataFrame с колонками: open, high, low, close, timestamp.
        Возвращает df с колонками индикаторов и сигналами.
        Используется для первоначального прогрева состояния.
        """
        df = df.copy().reset_index(drop=True)
        n  = len(df)

        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        src   = (high + low + close) / 3.0   # hlc3

        cfg = self.cfg

        # ── Kernel Regression ────────────────────────────────────────────
        y_hat = np.full(n, np.nan)
        for idx in range(cfg.length - 1, n):
            window = src[idx - cfg.length + 1: idx + 1][::-1]
            y_hat[idx] = np.dot(window, self._weights) / self._w_sum

        # ── Volatility ────────────────────────────────────────────────────
        atr_ch   = calc_atr(high, low, close, cfg.length)
        mean_dev = np.full(n, np.nan)
        for idx in range(cfg.length - 1, n):
            mean_dev[idx] = np.mean(np.abs(src[idx - cfg.length + 1: idx + 1] - y_hat[idx]))
        volatility = (mean_dev + atr_ch) / 2.0

        # ── Channels ──────────────────────────────────────────────────────
        df["y_hat"]       = y_hat
        df["upper_inner"] = y_hat + volatility * cfg.mult_inner
        df["lower_inner"] = y_hat - volatility * cfg.mult_inner
        df["upper_outer"] = y_hat + volatility * cfg.mult_outer
        df["lower_outer"] = y_hat - volatility * cfg.mult_outer

        # ── ATR Volatility Filter ─────────────────────────────────────────
        atr_raw      = calc_atr(high, low, close, cfg.atr_filter_len)
        atr_baseline = pd.Series(atr_raw).rolling(cfg.atr_ma_len).mean().values
        atr_ratio    = np.where(atr_baseline > 0, atr_raw / atr_baseline, np.nan)
        df["atr_ratio"] = atr_ratio
        df["atr_ok"]    = (atr_ratio >= cfg.atr_mult_thresh) if cfg.use_atr_filter else True

        # ── Pivot Point SuperTrend ────────────────────────────────────────
        ph = pivot_high(high, cfg.prd, cfg.prd)
        pl = pivot_low(low,   cfg.prd, cfg.prd)

        center_arr = np.full(n, np.nan)
        c = np.nan
        for i in range(n):
            lastpp = ph[i] if not np.isnan(ph[i]) else (pl[i] if not np.isnan(pl[i]) else np.nan)
            if not np.isnan(lastpp):
                c = lastpp if np.isnan(c) else (c * 2 + lastpp) / 3.0
            center_arr[i] = c
        df["center"] = center_arr

        atr_st  = calc_atr(high, low, close, cfg.atr_pd)
        up_band = center_arr - cfg.atr_factor * atr_st
        dn_band = center_arr + cfg.atr_factor * atr_st

        t_up    = np.full(n, np.nan)
        t_down  = np.full(n, np.nan)
        trend   = np.zeros(n, dtype=int)

        for i in range(1, n):
            pc = close[i - 1]
            t_up[i]   = max(up_band[i], t_up[i-1])   if (not np.isnan(t_up[i-1])   and pc > t_up[i-1])   else up_band[i]
            t_down[i] = min(dn_band[i], t_down[i-1]) if (not np.isnan(t_down[i-1]) and pc < t_down[i-1]) else dn_band[i]
            prev_t = trend[i-1] if trend[i-1] != 0 else 1
            if   not np.isnan(t_down[i-1]) and close[i] > t_down[i-1]: trend[i] = 1
            elif not np.isnan(t_up[i-1])   and close[i] < t_up[i-1]:   trend[i] = -1
            else: trend[i] = prev_t

        df["t_up"]    = t_up
        df["t_down"]  = t_down
        df["trend"]   = trend
        df["trailing_sl"] = np.where(trend == 1, t_up, t_down)

        # Сохраняем последнее состояние в объект для live-режима
        last = -1
        self._center   = center_arr[last]
        self._t_up     = t_up[last]
        self._t_down   = t_down[last]
        self._trend    = int(trend[last])
        self._prev_trend = int(trend[-2]) if n >= 2 else 0

        # Буфер для инкрементального расчёта (последние length баров)
        buf_len = max(cfg.length, cfg.atr_filter_len, cfg.atr_ma_len,
                      cfg.atr_pd, cfg.prd * 2 + 1) + 10
        self._buf_high  = list(high[-buf_len:])
        self._buf_low   = list(low[-buf_len:])
        self._buf_close = list(close[-buf_len:])
        self._buf_src   = list(src[-buf_len:])
        self._buf_ts    = list(df["timestamp"].iloc[-buf_len:])

        # ── Сигналы (для бэктеста) ────────────────────────────────────────
        prev_close       = pd.Series(close).shift(1).values
        prev_lower_outer = df["lower_outer"].shift(1).values
        prev_upper_outer = df["upper_outer"].shift(1).values

        if cfg.use_close_cross:
            long_touch  = close < df["lower_outer"].values
            short_touch = close > df["upper_outer"].values
        else:
            long_touch  = low  < df["lower_outer"].values
            short_touch = high > df["upper_outer"].values

        long_touch_prev  = np.concatenate([[False], long_touch[:-1]])
        short_touch_prev = np.concatenate([[False], short_touch[:-1]])

        long_signal  = long_touch  & ~long_touch_prev
        short_signal = short_touch & ~short_touch_prev

        trend_long  = (trend == 1)  if cfg.use_trend_filter else np.ones(n, bool)
        trend_short = (trend == -1) if cfg.use_trend_filter else np.ones(n, bool)

        df["go_long"]  = long_signal  & df["atr_ok"] & trend_long
        df["go_short"] = short_signal & df["atr_ok"] & trend_short

        return df

    # ── Публичный метод: одна новая свеча (live) ──────────────────────────

    def next(self, candle: dict) -> dict:
        """
        Принимает словарь одной завершённой свечи:
          { "timestamp": datetime, "open": float, "high": float,
            "low": float, "close": float }

        Возвращает словарь:
          { "signal": "long" | "short" | None,
            "trend": 1 | -1 | 0,
            "atr_ok": bool,
            "time_ok": bool,
            "lower_outer": float,
            "upper_outer": float,
            "trailing_sl": float }
        """
        cfg = self.cfg
        h, l, c = float(candle["high"]), float(candle["low"]), float(candle["close"])
        ts       = candle["timestamp"]
        src_val  = (h + l + c) / 3.0

        # Обновляем буферы
        self._buf_high.append(h);  self._buf_low.append(l)
        self._buf_close.append(c); self._buf_src.append(src_val)
        self._buf_ts.append(ts)

        max_buf = max(cfg.length, cfg.atr_filter_len, cfg.atr_ma_len,
                      cfg.atr_pd, cfg.prd * 2 + 1) + 10
        if len(self._buf_high) > max_buf:
            self._buf_high  = self._buf_high[-max_buf:]
            self._buf_low   = self._buf_low[-max_buf:]
            self._buf_close = self._buf_close[-max_buf:]
            self._buf_src   = self._buf_src[-max_buf:]
            self._buf_ts    = self._buf_ts[-max_buf:]

        high_a  = np.array(self._buf_high,  dtype=float)
        low_a   = np.array(self._buf_low,   dtype=float)
        close_a = np.array(self._buf_close, dtype=float)
        src_a   = np.array(self._buf_src,   dtype=float)
        n       = len(close_a)

        if n < cfg.length:
            return {"signal": None, "trend": 0, "atr_ok": False,
                    "time_ok": False, "lower_outer": np.nan,
                    "upper_outer": np.nan, "trailing_sl": np.nan}

        # ── Kernel Regression ─────────────────────────────────────────────
        window_src = src_a[-cfg.length:][::-1]
        y_hat_cur  = float(np.dot(window_src, self._weights) / self._w_sum)

        # ── Volatility ────────────────────────────────────────────────────
        atr_ch_a   = calc_atr(high_a, low_a, close_a, cfg.length)
        mean_dev   = float(np.mean(np.abs(src_a[-cfg.length:] - y_hat_cur)))
        vol        = (mean_dev + atr_ch_a[-1]) / 2.0

        lower_outer = y_hat_cur - vol * cfg.mult_outer
        upper_outer = y_hat_cur + vol * cfg.mult_outer

        # ── ATR Volatility Filter ─────────────────────────────────────────
        atr_ok = True
        if cfg.use_atr_filter and n >= cfg.atr_filter_len + cfg.atr_ma_len:
            atr_raw_a    = calc_atr(high_a, low_a, close_a, cfg.atr_filter_len)
            atr_baseline = float(np.nanmean(atr_raw_a[-cfg.atr_ma_len:]))
            atr_ratio    = atr_raw_a[-1] / atr_baseline if atr_baseline > 0 else 0.0
            atr_ok       = atr_ratio >= cfg.atr_mult_thresh

        # ── SuperTrend (инкрементально) ───────────────────────────────────
        if n >= cfg.atr_pd:
            atr_st_a = calc_atr(high_a, low_a, close_a, cfg.atr_pd)
            atr_st   = atr_st_a[-1]
        else:
            atr_st = vol  # запасной вариант

        # Обновляем center по новым pivot'ам
        if n >= cfg.prd * 2 + 1:
            ph_val = pivot_high(high_a, cfg.prd, cfg.prd)[-1]
            pl_val = pivot_low(low_a,   cfg.prd, cfg.prd)[-1]
            lastpp = ph_val if not np.isnan(ph_val) else (pl_val if not np.isnan(pl_val) else np.nan)
            if not np.isnan(lastpp):
                self._center = lastpp if np.isnan(self._center) \
                    else (self._center * 2 + lastpp) / 3.0

        up_band = self._center - cfg.atr_factor * atr_st if not np.isnan(self._center) else np.nan
        dn_band = self._center + cfg.atr_factor * atr_st if not np.isnan(self._center) else np.nan

        prev_close = close_a[-2] if n >= 2 else np.nan

        # TUp
        if not np.isnan(self._t_up) and not np.isnan(prev_close) and prev_close > self._t_up:
            new_t_up = max(up_band, self._t_up) if not np.isnan(up_band) else self._t_up
        else:
            new_t_up = up_band if not np.isnan(up_band) else self._t_up

        # TDown
        if not np.isnan(self._t_down) and not np.isnan(prev_close) and prev_close < self._t_down:
            new_t_down = min(dn_band, self._t_down) if not np.isnan(dn_band) else self._t_down
        else:
            new_t_down = dn_band if not np.isnan(dn_band) else self._t_down

        # Trend
        prev_t = self._trend if self._trend != 0 else 1
        if   not np.isnan(self._t_down) and c > self._t_down: new_trend = 1
        elif not np.isnan(self._t_up)   and c < self._t_up:   new_trend = -1
        else: new_trend = prev_t

        self._prev_trend = self._trend
        self._trend      = new_trend
        self._t_up       = new_t_up
        self._t_down     = new_t_down

        trailing_sl = new_t_up if new_trend == 1 else new_t_down

        # ── Time Filter ───────────────────────────────────────────────────
        time_ok = True
        if cfg.use_time_filter:
            bar_time = ts.time() if hasattr(ts, "time") else ts
            in_sess  = cfg.session_start <= bar_time <= cfg.session_end
            is_wknd  = ts.weekday() >= 5 if hasattr(ts, "weekday") else False
            wknd_ok  = True if cfg.trade_weekends else not is_wknd
            time_ok  = in_sess and wknd_ok

        # ── Entry Conditions ──────────────────────────────────────────────
        prev_c = close_a[-2] if n >= 2 else c

        if cfg.use_close_cross:
            long_touch_cur  = c   < lower_outer
            long_touch_prev = prev_c < (y_hat_cur - (mean_dev + atr_ch_a[-2]) / 2.0 * cfg.mult_outer) \
                              if n >= 2 else False
            short_touch_cur  = c   > upper_outer
            short_touch_prev = prev_c > (y_hat_cur + (mean_dev + atr_ch_a[-2]) / 2.0 * cfg.mult_outer) \
                               if n >= 2 else False
        else:
            # wick mode: low/high коснулись канала
            long_touch_cur   = l < lower_outer
            long_touch_prev  = np.array(self._buf_low)[-2] < lower_outer if n >= 2 else False
            short_touch_cur  = h > upper_outer
            short_touch_prev = np.array(self._buf_high)[-2] > upper_outer if n >= 2 else False

        # Первый бар пробоя (не повторный)
        long_signal  = long_touch_cur  and not long_touch_prev
        short_signal = short_touch_cur and not short_touch_prev

        trend_ok_long  = (new_trend == 1)  if cfg.use_trend_filter else True
        trend_ok_short = (new_trend == -1) if cfg.use_trend_filter else True

        go_long  = long_signal  and atr_ok and time_ok and trend_ok_long
        go_short = short_signal and atr_ok and time_ok and trend_ok_short

        signal = "long" if go_long else ("short" if go_short else None)

        return {
            "signal":       signal,
            "trend":        new_trend,
            "atr_ok":       atr_ok,
            "time_ok":      time_ok,
            "lower_outer":  lower_outer,
            "upper_outer":  upper_outer,
            "trailing_sl":  float(trailing_sl) if not np.isnan(trailing_sl) else np.nan,
            "y_hat":        y_hat_cur,
        }
