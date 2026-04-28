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

    CANDLE_INTERVAL = CandleInterval.CANDLE_INTERVAL_5_MIN

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
        # 5-minute candles: count * 5 minutes
        from_ = now - timedelta(minutes=count * 5 + 10)

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
                    side = "long"
                    avg_price = q_to_float(pos.average_position_price)
                    # Check for SHORT position using various possible attributes
                    try:
                        # Try position_direction attribute (T-Invest API v2)
                        if hasattr(pos, "position_direction") and pos.position_direction:
                            direction_str = str(pos.position_direction)
                            if "SHORT" in direction_str.upper():
                                side = "short"
                        # Fallback: check if average_price is None but qty > 0 might still be SHORT
                        # Also check quantity_lots which might indicate SHORT
                        elif hasattr(pos, "quantity_lots") and pos.quantity_lots:
                            lots = int(quotation_to_decimal(pos.quantity_lots))
                            if lots < 0:
                                side = "short"
                    except Exception as e:
                        log.warning("⚠️ Position direction detection: %s", e)
                    
                    return {
                        "side": side,
                        "qty": qty,
                        "avg_price": avg_price,
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
        elif pos["side"] == "short":
            await self.place_market_order("buy", pos["qty"])

    async def enter_trade(self, signal, entry_price, sl_perc, tp_perc, qty):

        if signal == "long":
            sl = entry_price * (1 - sl_perc / 100)
            tp = entry_price * (1 + tp_perc / 100)
            direction = "buy"
            stop_dir = "sell"
        elif signal == "short":
            sl = entry_price * (1 + sl_perc / 100)
            tp = entry_price * (1 - tp_perc / 100)
            direction = "sell"
            stop_dir = "buy"
        else:
            return

        # Check for existing position (pyramiding)
        existing_pos = await self.get_position()
        is_pyramiding = existing_pos["side"] is not None and existing_pos["qty"] > 0

        if is_pyramiding:
            log.info("📐 Pyramiding: existing %s %d @ %.4f", 
                   existing_pos["side"], existing_pos["qty"], existing_pos["avg_price"])
            
            # Save current stop order prices before attempting new entry
            current_stops = await self.get_stop_orders()
            saved_sl = None
            saved_tp = None
            for stop in current_stops:
                price = q_to_float(stop.stop_price)
                if "take" in str(stop.stop_order_type).lower():
                    saved_tp = price
                else:
                    saved_sl = price
            
            # Calculate new average price for SL move
            if existing_pos["side"] == signal:  # Same direction - average in
                total_qty = existing_pos["qty"] + qty
                new_avg = (
                    (existing_pos["avg_price"] * existing_pos["qty"] + entry_price * qty) / total_qty
                )
                
                # Recalculate SL to new average
                if signal == "long":
                    sl = new_avg * (1 - sl_perc / 100)
                else:
                    sl = new_avg * (1 + sl_perc / 100)
                
                log.info("📐 New avg price: %.4f, new SL: %.4f", new_avg, sl)
        
        log.info("📥 Opening %s position...", signal)

        await self.place_market_order(direction, qty)

        for _ in range(10):
            await asyncio.sleep(0.5)
            pos = await self.get_position()
            if (signal == "long" and pos["side"] == "long") or \
               (signal == "short" and pos["side"] == "short"):
                log.info("✅ Position confirmed")
                break
        else:
            log.error("❌ Position NOT found after entry!")
            return

        log.info("🛡️ Placing SL/TP...")

        # Cancel old stop orders (only if we had existing position with stops)
        if is_pyramiding:
            await self.cancel_all_orders()

        await self.place_stop_order(stop_dir, sl, pos["qty"], False)
        await self.place_stop_order(stop_dir, tp, pos["qty"], True)

        log.info("✅ Trade entered with protection")

    async def verify_and_restore_stops(self, pos: dict, sl_perc: float, tp_perc: float):
        """Verify SL/TP are correct, restore if missing or wrong."""
        current_stops = await self.get_stop_orders()
        entry = pos["avg_price"]
        side = pos["side"]
        qty = pos["qty"]
        
        if side == "long":
            expected_sl = entry * (1 - sl_perc / 100)
            expected_tp = entry * (1 + tp_perc / 100)
            sl_dir = "sell"
            tp_dir = "sell"
        else:
            expected_sl = entry * (1 + sl_perc / 100)
            expected_tp = entry * (1 - tp_perc / 100)
            sl_dir = "buy"
            tp_dir = "buy"
        
        sl_missing = True
        tp_missing = True
        sl_wrong = False
        tp_wrong = False
        
        for stop in current_stops:
            price = q_to_float(stop.stop_price)
            stop_type = str(stop.stop_order_type)
            
            if "take" in stop_type.lower():
                tp_missing = False
                if abs(price - expected_tp) > 0.01:
                    tp_wrong = True
            else:
                sl_missing = False
                if abs(price - expected_sl) > 0.01:
                    sl_wrong = True
        
        if sl_missing or tp_missing or sl_wrong or tp_wrong:
            log.warning("⚠️ Stops verification failed: SL=%s TP=%s. Restoring...",
                       "missing" if sl_missing else ("wrong" if sl_wrong else "OK"),
                       "missing" if tp_missing else ("wrong" if tp_wrong else "OK"))
            
            # Cancel all and place correct stops
            await self.cancel_all_orders()
            await self.place_stop_order(sl_dir, expected_sl, qty, False)
            await self.place_stop_order(tp_dir, expected_tp, qty, True)
            log.info("✅ Stops restored: SL=%.4f TP=%.4f", expected_sl, expected_tp)
            return True
        
        return False

    async def cancel_all_orders(self):
        await self._client.cancel_all_orders(account_id=self.account_id)
