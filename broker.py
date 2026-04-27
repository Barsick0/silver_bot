import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from t_tech.invest import (
    AsyncClient,
    CandleInterval,
    OrderDirection,
    OrderType,
    StopOrderDirection,
    StopOrderType,
    StopOrderExpirationType,
    Quotation,
)
from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from t_tech.invest.utils import quotation_to_decimal, decimal_to_quotation

log = logging.getLogger("broker")


# ─────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────

def q_to_float(q) -> float:
    if q is None:
        return 0.0
    return float(quotation_to_decimal(q))


def float_to_q(value: float) -> Quotation:
    return decimal_to_quotation(Decimal(str(round(value, 9))))


# ─────────────────────────────────────────────
# БРОКЕР
# ─────────────────────────────────────────────

class TInvestBroker:

    CANDLE_INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN

    def __init__(self, token: str, account_id: str, figi: str,
                 lot_size: int = 1, sandbox: bool = False):
        self.token = token
        self.account_id = account_id
        self.figi = figi
        self.lot_size = lot_size
        self.sandbox = sandbox
        self._client = None

    # ── Подключение ─────────────────────────

    async def connect(self):
        target = INVEST_GRPC_API_SANDBOX if self.sandbox else INVEST_GRPC_API
        self._client = await AsyncClient(self.token, target=target).__aenter__()
        log.info("✅ T-Invest API connected")

    async def disconnect(self):
        if self._client:
            await self._client.close()
            log.info("🔌 Disconnected")

    # ── Свечи ───────────────────────────────

    async def get_candles(self, count: int = 500):
        now = datetime.now(timezone.utc)
        from_ = now - timedelta(minutes=count + 10)

        resp = await self._client.market_data.get_candles(
            figi=self.figi,
            from_=from_,
            to=now,
            interval=self.CANDLE_INTERVAL,
        )

        candles = []
        for c in resp.candles:
            if c.is_complete:
                candles.append({
                    "timestamp": c.time.astimezone(tz=None),
                    "open": q_to_float(c.open),
                    "high": q_to_float(c.high),
                    "low": q_to_float(c.low),
                    "close": q_to_float(c.close),
                    "volume": c.volume,
                })

        candles.sort(key=lambda x: x["timestamp"])
        return candles[-count:]

    async def get_last_candle(self):
        candles = await self.get_candles(count=3)
        return candles[-1] if candles else None

    # ── Позиция ─────────────────────────────

    async def get_position(self):
        resp = await self._client.operations.get_portfolio(
            account_id=self.account_id
        )
        for pos in resp.positions:
            if pos.figi == self.figi:
                qty = int(quotation_to_decimal(pos.quantity))
                if qty > 0:
                    return {
                        "side": "long",
                        "qty": qty,
                        "avg_price": q_to_float(pos.average_position_price),
                    }
        return {"side": None, "qty": 0, "avg_price": 0.0}

    async def get_stop_orders(self):
        resp = await self._client.stop_orders.get_stop_orders(
            account_id=self.account_id
        )
        return [o for o in resp.stop_orders if o.figi == self.figi]

    # ── Ордера ──────────────────────────────

    async def place_market_order(self, direction: str, qty: int):
        order_dir = (
            OrderDirection.ORDER_DIRECTION_BUY
            if direction == "buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

        resp = await self._client.orders.post_order(
            figi=self.figi,
            quantity=qty,
            direction=order_dir,
            account_id=self.account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )

        log.info("📤 Market order placed")
        return resp.order_id

    async def place_stop_order(self, direction: str, stop_price: float,
                               qty: int, is_tp: bool = False):

        stop_dir = (
            StopOrderDirection.STOP_ORDER_DIRECTION_BUY
            if direction == "buy"
            else StopOrderDirection.STOP_ORDER_DIRECTION_SELL
        )

        stop_type = (
            StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT
            if is_tp
            else StopOrderType.STOP_ORDER_TYPE_STOP_LOSS
        )

        resp = await self._client.stop_orders.post_stop_order(
            figi=self.figi,
            quantity=qty,
            price=float_to_q(stop_price) if is_tp else None,
            stop_price=float_to_q(stop_price),
            direction=stop_dir,
            account_id=self.account_id,
            stop_order_type=stop_type,
            expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
            expire_date=None,
        )

        log.info("🛡️ Stop order (%s) placed", "TP" if is_tp else "SL")
        return resp.stop_order_id

    # ── Управление ──────────────────────────

    async def close_position(self):
        pos = await self.get_position()
        if pos["side"] == "long":
            await self.place_market_order("sell", pos["qty"])

    async def enter_trade(self, signal, entry_price, sl_perc, tp_perc, qty):

        if signal != "long":
            return

        sl = entry_price * (1 - sl_perc / 100)
        tp = entry_price * (1 + tp_perc / 100)

        log.info("📥 Opening position...")

        await self.place_market_order("buy", qty)

        # Ждем подтверждение позиции
        for _ in range(10):
            await asyncio.sleep(0.5)
            pos = await self.get_position()
            if pos["side"] == "long":
                log.info("✅ Position confirmed")
                break
        else:
            log.error("❌ Position NOT found after entry!")
            return

        # Ставим защиту
        log.info("🛡️ Placing SL/TP...")

        await self.place_stop_order("sell", sl, qty, False)
        await self.place_stop_order("sell", tp, qty, True)

        log.info("✅ Trade entered with protection")

    async def cancel_all_orders(self):
        await self._client.cancel_all_orders(account_id=self.account_id)
