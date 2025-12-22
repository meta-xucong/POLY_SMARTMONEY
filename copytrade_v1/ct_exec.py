from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from ct_utils import clamp, round_to_tick, safe_float


class PriceSample:
    def __init__(self, price: float) -> None:
        self.price = float(price)


def _extract_best_price(payload: Any, side: str) -> Optional[PriceSample]:
    numeric = safe_float(payload)
    if numeric is not None:
        return PriceSample(numeric)

    if isinstance(payload, Mapping):
        primary_keys = {
            "bid": (
                "best_bid",
                "bestBid",
                "bid",
                "highestBid",
                "bestBidPrice",
                "bidPrice",
                "buy",
            ),
            "ask": (
                "best_ask",
                "bestAsk",
                "ask",
                "offer",
                "best_offer",
                "bestOffer",
                "lowestAsk",
                "sell",
            ),
        }[side]
        for key in primary_keys:
            if key in payload:
                extracted = _extract_best_price(payload[key], side)
                if extracted is not None:
                    return extracted

        ladder_keys = {
            "bid": ("bids", "bid_levels", "buy_orders", "buyOrders"),
            "ask": ("asks", "ask_levels", "sell_orders", "sellOrders", "offers"),
        }[side]
        for key in ladder_keys:
            if key in payload:
                ladder = payload[key]
                if isinstance(ladder, Iterable) and not isinstance(ladder, (str, bytes, bytearray)):
                    for entry in ladder:
                        if isinstance(entry, Mapping) and "price" in entry:
                            candidate = safe_float(entry.get("price"))
                            if candidate is not None:
                                return PriceSample(candidate)
                        extracted = _extract_best_price(entry, side)
                        if extracted is not None:
                            return extracted

        for value in payload.values():
            extracted = _extract_best_price(value, side)
            if extracted is not None:
                return extracted
        return None

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            extracted = _extract_best_price(item, side)
            if extracted is not None:
                return extracted
        return None

    return None


def _fetch_best_price(client: Any, token_id: str, side: str) -> Optional[PriceSample]:
    method_candidates = (
        ("get_market_orderbook", {"market": token_id}),
        ("get_market_orderbook", {"token_id": token_id}),
        ("get_market_orderbook", {"market_id": token_id}),
        ("get_order_book", {"market": token_id}),
        ("get_order_book", {"token_id": token_id}),
        ("get_orderbook", {"market": token_id}),
        ("get_orderbook", {"token_id": token_id}),
        ("get_market", {"market": token_id}),
        ("get_market", {"token_id": token_id}),
        ("get_market_data", {"market": token_id}),
        ("get_market_data", {"token_id": token_id}),
        ("get_ticker", {"market": token_id}),
        ("get_ticker", {"token_id": token_id}),
    )

    for name, kwargs in method_candidates:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        try:
            resp = fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        payload = resp
        if isinstance(resp, tuple) and len(resp) == 2:
            payload = resp[1]
        if isinstance(payload, Mapping) and {"data", "status"} <= set(payload.keys()):
            payload = payload.get("data")

        best = _extract_best_price(payload, side)
        if best is not None:
            return best
    return None


def get_orderbook(client: Any, token_id: str) -> Dict[str, Optional[float]]:
    best_bid = _fetch_best_price(client, token_id, "bid")
    best_ask = _fetch_best_price(client, token_id, "ask")
    return {
        "best_bid": best_bid.price if best_bid is not None else None,
        "best_ask": best_ask.price if best_ask is not None else None,
    }


def reconcile_one(
    token_id: str,
    desired_shares: float,
    my_shares: float,
    orderbook: Dict[str, Optional[float]],
    open_orders: List[Dict[str, Any]],
    now_ts: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    deadband = float(cfg.get("deadband_shares") or 0)
    delta = desired_shares - my_shares
    if abs(delta) <= deadband:
        return actions

    ttl_sec = int(cfg.get("order_ttl_sec") or 0)
    remaining_orders: List[Dict[str, Any]] = []
    for order in open_orders:
        ts = int(order.get("ts") or 0)
        if ttl_sec > 0 and now_ts - ts > ttl_sec:
            actions.append({"type": "cancel", "order_id": order.get("order_id")})
        else:
            remaining_orders.append(order)

    if remaining_orders:
        return actions

    slice_min = float(cfg.get("slice_min") or 0)
    slice_max = float(cfg.get("slice_max") or abs(delta))
    size = clamp(abs(delta), slice_min, slice_max)

    best_bid = orderbook.get("best_bid")
    best_ask = orderbook.get("best_ask")
    tick_size = float(cfg.get("tick_size") or 0)

    side = "BUY" if delta > 0 else "SELL"
    price: Optional[float] = None
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

    if price is None or price <= 0:
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
    candidates = []
    for attr in ("cancel", "cancel_order", "cancel_orders"):
        method = getattr(client, attr, None)
        if callable(method):
            candidates.append(method)
    private = getattr(client, "private", None)
    if private is not None:
        for attr in ("cancel", "cancel_order", "cancel_orders"):
            method = getattr(private, attr, None)
            if callable(method):
                candidates.append(method)

    last_error: Optional[Exception] = None
    for method in candidates:
        try:
            return method(order_id)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
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
            if not dry_run:
                cancel_order(client, str(order_id))
            updated = [o for o in updated if str(o.get("order_id")) != str(order_id)]

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
        response = place_order(
            client,
            token_id=str(action.get("token_id")),
            side=str(action.get("side")),
            price=float(action.get("price")),
            size=float(action.get("size")),
        )
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


def cancel_expired_only(
    client: Any,
    open_orders: Dict[str, List[Dict[str, Any]]],
    now_ts: int,
    ttl_sec: int,
    dry_run: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    updated: Dict[str, List[Dict[str, Any]]] = {}
    for token_id, orders in open_orders.items():
        keep: List[Dict[str, Any]] = []
        for order in orders:
            ts = int(order.get("ts") or 0)
            if ttl_sec > 0 and now_ts - ts > ttl_sec:
                order_id = order.get("order_id")
                if order_id and not dry_run:
                    cancel_order(client, str(order_id))
                continue
            keep.append(order)
        if keep:
            updated[token_id] = keep
    return updated
