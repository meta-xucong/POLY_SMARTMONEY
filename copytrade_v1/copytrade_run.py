from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smartmoney_query.poly_martmoney_query.api_client import DataApiClient

from ct_data import fetch_positions_norm
from ct_exec import (
    apply_actions,
    cancel_expired_only,
    fetch_open_orders_norm,
    get_orderbook,
    reconcile_one,
)
from ct_resolver import resolve_token_id
from ct_risk import risk_check
from ct_state import load_state, save_state


DEFAULT_CONFIG_PATH = Path(__file__).with_name("copytrade_config.json")
DEFAULT_STATE_PATH = Path(__file__).with_name("state.json")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件必须为 JSON dict")
    return payload


def _normalize_privkey(key: str) -> str:
    return key[2:] if key.startswith(("0x", "0X")) else key


def init_clob_client():
    from py_clob_client.client import ClobClient

    host = os.getenv("POLY_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLY_CHAIN_ID", "137"))
    signature_type = int(os.getenv("POLY_SIGNATURE", "2"))
    key = _normalize_privkey(os.environ["POLY_KEY"])
    funder = os.environ["POLY_FUNDER"]

    client = ClobClient(
        host,
        key=key,
        chain_id=chain_id,
        signature_type=signature_type,
        funder=funder,
    )
    api_creds = client.create_or_derive_api_creds()
    client.set_api_creds(api_creds)
    try:
        setattr(client, "api_creds", api_creds)
    except Exception:
        pass
    return client


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Copytrade v1")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--target", dest="target_address")
    parser.add_argument("--my", dest="my_address")
    parser.add_argument("--ratio", type=float, dest="follow_ratio")
    parser.add_argument("--poll", type=int, dest="poll_interval_sec")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sync_open_orders_state(
    prev: Dict[str, Any],
    remote: list[dict],
    now_ts: int,
) -> Dict[str, Any]:
    existing_by_id: Dict[str, dict] = {}
    for orders in (prev or {}).values():
        for order in orders or []:
            oid = str(order.get("order_id") or "")
            if oid:
                existing_by_id[oid] = order

    new_map: Dict[str, list[dict]] = {}
    for order in remote:
        token_id = str(order.get("token_id") or "")
        oid = str(order.get("order_id") or "")
        if not token_id or not oid:
            continue
        prev_order = existing_by_id.get(oid, {})
        ts = int(prev_order.get("ts") or order.get("created_ts") or now_ts)
        new_map.setdefault(token_id, []).append(
            {
                "order_id": oid,
                "side": order.get("side") or prev_order.get("side"),
                "price": order.get("price") if order.get("price") is not None else prev_order.get("price"),
                "size": order.get("size") if order.get("size") is not None else prev_order.get("size"),
                "ts": ts,
            }
        )
    return new_map


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))
    for key in ("target_address", "my_address", "follow_ratio", "poll_interval_sec"):
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            cfg[key] = arg_val
    if not cfg.get("target_address") or not cfg.get("my_address"):
        raise ValueError("target_address / my_address 未配置")

    state = load_state(args.state)
    state["target"] = cfg.get("target_address")
    state["my_address"] = cfg.get("my_address")
    state["follow_ratio"] = cfg.get("follow_ratio")

    data_client = DataApiClient()
    clob_client = init_clob_client()

    poll_interval = int(cfg.get("poll_interval_sec") or 20)
    size_threshold = float(cfg.get("size_threshold") or 0)

    while True:
        now_ts = int(time.time())
        if not args.dry_run:
            try:
                remote_orders = fetch_open_orders_norm(clob_client)
                state["open_orders"] = _sync_open_orders_state(
                    state.get("open_orders", {}),
                    remote_orders,
                    now_ts,
                )
            except Exception as exc:
                print(f"[WARN] sync open orders failed: {exc}")

        target_pos, target_info = fetch_positions_norm(
            data_client,
            cfg["target_address"],
            size_threshold,
        )
        if not target_info.get("ok") or target_info.get("incomplete"):
            print("[SAFE] target positions 不完整，仅撤单")
            state["open_orders"] = cancel_expired_only(
                clob_client,
                state.get("open_orders", {}),
                now_ts,
                int(cfg.get("order_ttl_sec") or 0),
                args.dry_run,
            )
            save_state(args.state, state)
            time.sleep(poll_interval)
            continue

        my_pos, my_info = fetch_positions_norm(
            data_client,
            cfg["my_address"],
            0.0,
        )
        if not my_info.get("ok") or my_info.get("incomplete"):
            print("[SAFE] my positions 不完整，仅撤单")
            state["open_orders"] = cancel_expired_only(
                clob_client,
                state.get("open_orders", {}),
                now_ts,
                int(cfg.get("order_ttl_sec") or 0),
                args.dry_run,
            )
            save_state(args.state, state)
            time.sleep(poll_interval)
            continue

        desired_by_token_key: Dict[str, float] = {}
        for pos in target_pos:
            token_key = pos["token_key"]
            desired_by_token_key[token_key] = float(cfg["follow_ratio"]) * float(pos["size"])

        desired_by_token_id: Dict[str, float] = {}
        token_key_by_token_id: Dict[str, str] = {}
        for pos in target_pos:
            token_key = pos["token_key"]
            try:
                token_id = resolve_token_id(token_key, pos, state["token_map"])
            except Exception as exc:
                print(f"[WARN] resolver 失败: {token_key} -> {exc}")
                continue
            desired_by_token_id[token_id] = desired_by_token_key[token_key]
            token_key_by_token_id[token_id] = token_key

        my_by_token_id: Dict[str, float] = {}
        for pos in my_pos:
            token_key = pos["token_key"]
            try:
                token_id = resolve_token_id(token_key, pos, state["token_map"])
            except Exception as exc:
                print(f"[WARN] resolver 失败(自身): {token_key} -> {exc}")
                continue
            my_by_token_id[token_id] = float(pos["size"])
            token_key_by_token_id.setdefault(token_id, token_key)

        reconcile_set: Set[str] = set(desired_by_token_id)
        reconcile_set.update(my_by_token_id)
        reconcile_set.update(state.get("open_orders", {}).keys())

        orderbooks: Dict[str, Dict[str, Optional[float]]] = {}
        total_notional = 0.0
        if float(cfg.get("max_notional_total") or 0) > 0:
            for token_id in reconcile_set:
                ob = get_orderbook(clob_client, token_id)
                orderbooks[token_id] = ob
                ref_price = _mid_price(ob)
                if ref_price is None:
                    continue
                desired = desired_by_token_id.get(token_id, 0.0)
                total_notional += abs(desired) * ref_price
            cfg["_total_notional"] = total_notional

        for token_id in reconcile_set:
            desired = desired_by_token_id.get(token_id, 0.0)
            my_shares = my_by_token_id.get(token_id, 0.0)

            if abs(desired - my_shares) <= float(cfg.get("deadband_shares") or 0):
                continue

            if token_id in orderbooks:
                ob = orderbooks[token_id]
            else:
                ob = get_orderbook(clob_client, token_id)

            ref_price = _mid_price(ob)
            if ref_price is None:
                print(f"[WARN] 无法获取盘口: token_id={token_id}")
                continue

            token_key = token_key_by_token_id.get(token_id, f"token:{token_id}")
            ok, reason = risk_check(token_key, desired, my_shares, ref_price, cfg)
            if not ok:
                print(f"[RISK] {token_key} 拒绝: {reason}")
                continue

            actions = reconcile_one(
                token_id,
                desired,
                my_shares,
                ob,
                state.get("open_orders", {}).get(token_id, []),
                now_ts,
                cfg,
            )
            if not actions:
                continue
            print(f"[ACTION] token_id={token_id} -> {actions}")

            updated_orders = apply_actions(
                clob_client,
                actions,
                state.get("open_orders", {}).get(token_id, []),
                now_ts,
                args.dry_run,
            )
            if updated_orders:
                state.setdefault("open_orders", {})[token_id] = updated_orders
            else:
                state.get("open_orders", {}).pop(token_id, None)

        state["last_sync_ts"] = now_ts
        save_state(args.state, state)
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
