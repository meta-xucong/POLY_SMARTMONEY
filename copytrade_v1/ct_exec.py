from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ct_utils import round_to_tick, safe_float


logger = logging.getLogger(__name__)


def _mid_price(orderbook: Dict[str, Optional[float]]) -> Optional[float]:
    bid = orderbook.get("best_bid")
    ask = orderbook.get("best_ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def _best_from_levels(levels: Iterable[Any], pick_max: bool) -> Optional[float]:
    prices: List[float] = []
    for level in levels:
        if isinstance(level, Mapping):
            candidate = safe_float(level.get("price"))
            if candidate is not None:
                prices.append(candidate)
        elif isinstance(level, (list, tuple)) and level:
            candidate = safe_float(level[0])
            if candidate is not None:
                prices.append(candidate)
    if not prices:
        return None
    return max(prices) if pick_max else min(prices)


def _normalize_orderbook_payload(book: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(book, Mapping):
        return book
    if hasattr(book, "dict"):
        payload = book.dict()
        if isinstance(payload, Mapping):
            return payload
    if hasattr(book, "__dict__"):
        payload = dict(book.__dict__)
        if isinstance(payload, Mapping):
            return payload
    return None


def get_orderbook(client: Any, token_id: str) -> Dict[str, Optional[float]]:
    tid = str(token_id)

    best_ask: Optional[float] = None
    best_bid: Optional[float] = None

    try:
        price = client.get_price(tid, side="BUY")
        if isinstance(price, dict):
            best_ask = safe_float(price.get("price"))
        else:
            best_ask = safe_float(price)
    except Exception:
        pass
    try:
        price = client.get_price(tid, side="SELL")
        if isinstance(price, dict):
            best_bid = safe_float(price.get("price"))
        else:
            best_bid = safe_float(price)
    except Exception:
        pass

    if best_ask is not None or best_bid is not None:
        return {"best_bid": best_bid, "best_ask": best_ask}

    try:
        book = client.get_order_book(tid)
        payload: Any = book
        if hasattr(book, "dict"):
            payload = book.dict()
        elif isinstance(book, dict):
            payload = book

        bids = payload.get("bids", []) if isinstance(payload, dict) else getattr(book, "bids", [])
        asks = payload.get("asks", []) if isinstance(payload, dict) else getattr(book, "asks", [])

        def _best(levels: Any, pick_max: bool) -> Optional[float]:
            prices: list[float] = []
            if isinstance(levels, list):
                for level in levels:
                    if isinstance(level, dict):
                        price = safe_float(level.get("price"))
                    elif isinstance(level, (list, tuple)) and level:
                        price = safe_float(level[0])
                    else:
                        price = None
                    if price is not None:
                        prices.append(float(price))
            if not prices:
                return None
            return max(prices) if pick_max else min(prices)

        best_bid = _best(bids, pick_max=True)
        best_ask = _best(asks, pick_max=False)
        return {"best_bid": best_bid, "best_ask": best_ask}
    except Exception:
        return {"best_bid": None, "best_ask": None}


def reconcile_one(
    token_id: str,
    desired_shares: float,
    my_shares: float,
    orderbook: Dict[str, Optional[float]],
    open_orders: List[Dict[str, Any]],
    now_ts: int,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    deadband = float(cfg.get("deadband_shares") or 0)
    delta = desired_shares - my_shares
    if abs(delta) <= deadband and not open_orders:
        return actions

    abs_delta = abs(delta)

    mode = str(cfg.get("order_size_mode") or "fixed_shares").lower()
    size: float = 0.0
    target_order_usd: Optional[float] = None

    if mode == "auto_usd":
        ref_price = _mid_price(orderbook)
        if ref_price is None or ref_price <= 0:
            return actions

        min_usd = float(cfg.get("min_order_usd") or 5.0)
        max_usd = float(cfg.get("max_order_usd") or 25.0)
        if max_usd < min_usd:
            max_usd = min_usd

        k = float(cfg.get("_auto_order_k") or 0.3)

        delta_usd = abs_delta * ref_price
        order_usd = delta_usd * k
        if order_usd < min_usd:
            order_usd = min_usd
        if order_usd > max_usd:
            order_usd = max_usd

        target_order_usd = order_usd
        size = order_usd / ref_price
    else:
        slice_min = float(cfg.get("slice_min") or 0)
        slice_max = float(cfg.get("slice_max") or abs_delta)
        if slice_max <= 0:
            slice_max = abs_delta

        size = min(abs_delta, slice_max)
        if slice_min > 0 and abs_delta > slice_min and size < slice_min:
            size = slice_min

    side = "BUY" if delta > 0 else "SELL"
    price: Optional[float] = None
    best_bid = orderbook.get("best_bid")
    best_ask = orderbook.get("best_ask")
    tick_size = float(cfg.get("tick_size") or 0)
    if side == "BUY":
        if best_bid is not None:
            price = best_bid
        elif best_ask is not None:
            price = best_ask - tick_size
        if price is not None:
            price = round_to_tick(price, tick_size, direction="down")
    else:
        if best_ask is not None:
            price = best_ask
        elif best_bid is not None:
            price = best_bid + tick_size
        if price is not None:
            price = round_to_tick(price, tick_size, direction="up")

    maker_only = bool(cfg.get("maker_only"))
    if maker_only and tick_size and tick_size > 0:
        if side == "BUY" and best_ask is not None and price is not None and price >= best_ask:
            price = round_to_tick(best_ask - tick_size, tick_size, direction="down")
        if side == "SELL" and best_bid is not None and price is not None and price <= best_bid:
            price = round_to_tick(best_bid + tick_size, tick_size, direction="up")

        if price is None or price <= 0:
            return actions

    if price is None or price <= 0:
        return actions

    if mode == "auto_usd" and target_order_usd is not None:
        size = target_order_usd / price

    max_shares_cap = float(cfg.get("max_order_shares_cap") or 5000.0)
    if size > max_shares_cap:
        size = max_shares_cap

    allow_short = bool(cfg.get("allow_short"))
    if side == "SELL" and not allow_short:
        size = min(size, my_shares)

    if size <= 0:
        return actions

    if open_orders:
        active_order: Optional[Dict[str, Any]] = None
        if side == "BUY":
            active_order = max(open_orders, key=lambda order: float(order.get("price") or 0))
        else:
            active_order = min(open_orders, key=lambda order: float(order.get("price") or 0))
        if active_order:
            active_price = safe_float(active_order.get("price"))
            last_reprice_ts = int(
                state.setdefault("last_reprice_ts_by_token", {}).get(token_id) or 0
            )
            reprice_ticks = int(cfg.get("reprice_ticks") or cfg.get("reprice_min_ticks") or 1)
            cooldown_sec = int(cfg.get("reprice_cooldown_sec") or 0)
            cooldown_ok = cooldown_sec <= 0 or (now_ts - last_reprice_ts) >= cooldown_sec
            if active_price is not None and tick_size > 0 and cooldown_ok:
                if side == "BUY" and best_bid is not None:
                    trigger = best_bid >= active_price + tick_size * reprice_ticks
                elif side == "SELL" and best_ask is not None:
                    trigger = best_ask <= active_price - tick_size * reprice_ticks
                else:
                    trigger = False
                if trigger:
                    logger.info(
                        "[REPRICE] token_id=%s side=%s active_price=%s ideal_price=%s "
                        "best_bid=%s best_ask=%s reprice_ticks=%s cooldown_sec=%s since_last=%s",
                        token_id,
                        side,
                        active_price,
                        price,
                        best_bid,
                        best_ask,
                        reprice_ticks,
                        cooldown_sec,
                        now_ts - last_reprice_ts,
                    )
                    actions.append({"type": "cancel", "order_id": active_order.get("order_id")})
                    actions.append(
                        {
                            "type": "place",
                            "token_id": token_id,
                            "side": side,
                            "price": price,
                            "size": size,
                            "ts": now_ts,
                        }
                    )
                    state.setdefault("last_reprice_ts_by_token", {})[token_id] = now_ts
                    return actions
        return actions

    actions.append(
        {
            "type": "place",
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
            "ts": now_ts,
        }
    )
    return actions


def _extract_order_id(response: object) -> Optional[str]:
    candidates = (
        "order_id",
        "orderId",
        "orderID",
        "id",
        "orderHash",
        "order_hash",
        "hash",
    )

    visited: set[int] = set()

    def walk(obj: object) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)
            for key in candidates:
                if key in obj and obj[key] is not None:
                    return str(obj[key])
            for value in obj.values():
                nested = walk(value)
                if nested:
                    return nested
        if isinstance(obj, (list, tuple)):
            for item in obj:
                nested = walk(item)
                if nested:
                    return nested
        return None

    return walk(response)


def cancel_order(client: Any, order_id: str) -> Optional[object]:
    if not order_id:
        return None
    if callable(getattr(client, "cancel", None)):
        return client.cancel(order_id=order_id)
    if callable(getattr(client, "cancel_order", None)):
        return client.cancel_order(order_id)
    if callable(getattr(client, "cancel_orders", None)):
        return client.cancel_orders([order_id])

    private = getattr(client, "private", None)
    if private is not None:
        if callable(getattr(private, "cancel", None)):
            return private.cancel(order_id=order_id)
        if callable(getattr(private, "cancel_order", None)):
            return private.cancel_order(order_id)
        if callable(getattr(private, "cancel_orders", None)):
            return private.cancel_orders([order_id])

    return None


def place_order(client: Any, token_id: str, side: str, price: float, size: float) -> Dict[str, Any]:
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL

    side_const = BUY if side.upper() == "BUY" else SELL
    order_args = OrderArgs(token_id=str(token_id), side=side_const, price=float(price), size=float(size))
    signed = client.create_order(order_args)
    response = client.post_order(signed, OrderType.GTC)
    order_id = _extract_order_id(response)
    result: Dict[str, Any] = {"response": response}
    if order_id:
        result["order_id"] = order_id
    return result


def apply_actions(
    client: Any,
    actions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    now_ts: int,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    updated = [dict(order) for order in open_orders]
    for action in actions:
        if action.get("type") == "cancel":
            order_id = action.get("order_id")
            if not order_id:
                continue
            if dry_run:
                updated = [o for o in updated if str(o.get("order_id")) != str(order_id)]
                continue
            try:
                cancel_order(client, str(order_id))
                updated = [o for o in updated if str(o.get("order_id")) != str(order_id)]
            except Exception as exc:
                logger.warning("cancel_order failed order_id=%s: %s", order_id, exc)
            continue

    for action in actions:
        if action.get("type") != "place":
            continue
        if dry_run:
            updated.append(
                {
                    "order_id": "dry_run",
                    "side": action.get("side"),
                    "price": action.get("price"),
                    "size": action.get("size"),
                    "ts": now_ts,
                }
            )
            continue
        try:
            response = place_order(
                client,
                token_id=str(action.get("token_id")),
                side=str(action.get("side")),
                price=float(action.get("price")),
                size=float(action.get("size")),
            )
        except Exception as exc:
            logger.warning("place_order failed token_id=%s: %s", action.get("token_id"), exc)
            continue
        order_id = response.get("order_id")
        if order_id:
            updated.append(
                {
                    "order_id": order_id,
                    "side": action.get("side"),
                    "price": action.get("price"),
                    "size": action.get("size"),
                    "ts": now_ts,
                }
            )
    return updated




def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return None


def _coerce_list(payload: Any) -> List[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "orders", "result", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    for key in ("data", "orders", "result", "items"):
        value = getattr(payload, key, None)
        if isinstance(value, list):
            return value
    return []


def _parse_created_ts(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if parsed > 0 else None
        if isinstance(value, str):
            text = value.strip()
            num = safe_float(text)
            if num is not None and num > 0:
                numeric = int(num)
                return numeric // 1000 if numeric > 10_000_000_000 else numeric
            from datetime import datetime, timezone

            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed_dt = datetime.fromisoformat(text)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            return int(parsed_dt.timestamp())
    except Exception:
        return None
    return None


def _normalize_open_order(order: Any) -> Optional[Dict[str, Any]]:
    data = _as_dict(order)
    if not data:
        return None

    order_id = (
        data.get("id")
        or data.get("order_id")
        or data.get("orderId")
        or data.get("orderID")
        or data.get("order_hash")
        or data.get("orderHash")
    )
    token_id = (
        data.get("asset_id")
        or data.get("assetId")
        or data.get("token_id")
        or data.get("tokenId")
        or data.get("clobTokenId")
        or data.get("clob_token_id")
    )
    if not order_id or not token_id:
        return None

    side = data.get("side") or data.get("taker_side") or data.get("maker_side")
    side_norm = side.upper() if isinstance(side, str) else str(side).upper()

    price = safe_float(data.get("price") or data.get("limit_price") or data.get("limitPrice"))
    size = safe_float(
        data.get("size")
        or data.get("original_size")
        or data.get("originalSize")
        or data.get("remaining_size")
        or data.get("remainingSize")
        or data.get("amount")
    )

    created_ts = _parse_created_ts(data.get("created_at") or data.get("createdAt") or data.get("timestamp"))
    return {
        "order_id": str(order_id),
        "token_id": str(token_id),
        "side": side_norm,
        "price": price,
        "size": size,
        "created_ts": created_ts,
    }


def fetch_open_orders_norm(client: Any) -> tuple[list[dict[str, Any]], bool, str | None]:
    from py_clob_client.clob_types import OpenOrderParams

    try:
        payload = client.get_orders(OpenOrderParams())
    except Exception as exc:
        try:
            payload = client.get_orders()
        except Exception as exc2:
            return [], False, str(exc2 or exc)

    orders = _coerce_list(payload)
    normalized: List[Dict[str, Any]] = []
    for order in orders:
        parsed = _normalize_open_order(order)
        if not parsed:
            continue
        normalized.append(
            {
                "order_id": parsed["order_id"],
                "token_id": parsed["token_id"],
                "side": parsed["side"],
                "price": float(parsed["price"]) if parsed["price"] is not None else None,
                "size": float(parsed["size"]) if parsed["size"] is not None else None,
                "ts": parsed["created_ts"],
            }
        )
    filtered = [item for item in normalized if item["price"] is not None and item["size"] is not None]
    return filtered, True, None
