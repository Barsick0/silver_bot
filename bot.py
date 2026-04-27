import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from strategy import ChannelTrendATR, StrategyConfig
from broker import TInvestBroker

# ─────────────────────────────────────────────
# ⚙️ CORE CONFIG (НАСТРАИВАЕМОЕ ЯДРО)
# ─────────────────────────────────────────────

API_TOKEN = os.getenv("TINVEST_TOKEN")
ACCOUNT_ID = os.getenv("TINVEST_ACCOUNT")
FIGI = os.getenv("TINVEST_FIGI")

LOT_QTY = 1

# 🔥 контроль
MAX_TRADE_DURATION_MIN = 140
EMERGENCY_SL_EXTRA = 0.2

SANDBOX = False
WARMUP_BARS = 300

STRATEGY_CFG = StrategyConfig()

# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)

log = logging.getLogger("bot")


class TradingBot:

    def __init__(self):
        self.strategy = ChannelTrendATR(STRATEGY_CFG)
        self.broker = TInvestBroker(
            token=API_TOKEN,
            account_id=ACCOUNT_ID,
            figi=FIGI,
            lot_size=LOT_QTY,
            sandbox=SANDBOX,
        )

        self._last_candle_time = None
        self.position_open_time = None
        self.last_signal = None

    async def warmup(self):
        log.info("🔄 Warming up strategy...")
        candles = await self.broker.get_candles(count=WARMUP_BARS)

        if len(candles) < STRATEGY_CFG.length:
            log.error("❌ Not enough candles")
            return False

        df = pd.DataFrame(candles)
        self.strategy.warmup(df)

        self._last_candle_time = candles[-1]["timestamp"]
        log.info("✅ Warmup complete")
        return True

    async def process_tick(self):
        candle = await self.broker.get_last_candle()
        if candle is None:
            return

        if candle["timestamp"] == self._last_candle_time:
            return

        self._last_candle_time = candle["timestamp"]

        log.info(
            "🕯️ %s | O=%.4f H=%.4f L=%.4f C=%.4f",
            candle["timestamp"].strftime("%H:%M"),
            candle["open"], candle["high"],
            candle["low"], candle["close"],
        )

        result = self.strategy.next(candle)
        signal = result["signal"]

        log.info(
            "📊 trend=%+d atr=%s time=%s signal=%s",
            result["trend"], result["atr_ok"], result["time_ok"],
            signal or "—",
        )

        pos = await self.broker.get_position()

        # ─────────── ЕСЛИ ЕСТЬ ПОЗИЦИЯ ───────────
        if pos["side"] is not None:

            if self.position_open_time is None:
                self.position_open_time = datetime.now()

            # ⏱ время сделки
            minutes = (datetime.now() - self.position_open_time).total_seconds() / 60

            if minutes > MAX_TRADE_DURATION_MIN:
                log.warning("⏰ EXIT: TIMEOUT (%.1f min)", minutes)
                await self.broker.close_position()
                await self.broker.cancel_all_orders()
                self.position_open_time = None
                return

            # 💣 аварийный убыток
            entry = pos["avg_price"]
            close = candle["close"]
            side = pos["side"]

            if side == "long":
                loss_pct = ((entry - close) / entry) * 100
            else:  # short
                loss_pct = ((close - entry) / entry) * 100

            emergency_sl = STRATEGY_CFG.sl_perc + EMERGENCY_SL_EXTRA

            if loss_pct >= emergency_sl:
                log.warning("💣 EXIT: LOSS (%.3f%%) [%s]", loss_pct, side)
                await self.broker.close_position()
                await self.broker.cancel_all_orders()
                self.position_open_time = None
                return

            # 🛡 восстановление стопов
            stops = await self.broker.get_stop_orders()

            if not stops:
                log.warning("⚠️ restoring stops")

                if side == "long":
                    sl = entry * (1 - STRATEGY_CFG.sl_perc / 100)
                    tp = entry * (1 + STRATEGY_CFG.tp_perc / 100)
                    sl_dir = "sell"
                    tp_dir = "sell"
                else:  # short
                    sl = entry * (1 + STRATEGY_CFG.sl_perc / 100)
                    tp = entry * (1 - STRATEGY_CFG.tp_perc / 100)
                    sl_dir = "buy"
                    tp_dir = "buy"

                await self.broker.place_stop_order(sl_dir, sl, pos["qty"], False)
                await self.broker.place_stop_order(tp_dir, tp, pos["qty"], True)

            return

        # ─────────── ВХОД ───────────
        if signal is None:
            self.last_signal = None
            return

        if signal == self.last_signal:
            return

        await self.broker.enter_trade(
            signal=signal,
            entry_price=candle["close"],
            sl_perc=STRATEGY_CFG.sl_perc,
            tp_perc=STRATEGY_CFG.tp_perc,
            qty=LOT_QTY,
        )

        self.position_open_time = datetime.now()
        self.last_signal = signal

    async def run(self):
        log.info("🚀 Bot starting")
        await self.broker.connect()

        try:
            if not await self.warmup():
                return

            log.info("✅ Live")

            while True:
                try:
                    await self.process_tick()
                except Exception as e:
                    log.exception("⚠️ %s", e)

                await asyncio.sleep(15)

        finally:
            await self.broker.disconnect()


if __name__ == "__main__":
    asyncio.run(TradingBot().run())
