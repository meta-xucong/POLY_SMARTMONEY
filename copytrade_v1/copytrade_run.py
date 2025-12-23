from __future__ import annotations

import argparse
import json
import os
import re
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
from ct_resolver import (
    gamma_fetch_markets_by_clob_token_ids,
    market_is_tradeable,
    resolve_token_id,
)
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


_EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _is_placeholder_addr(value: Optional[str]) -> bool:
    if not value:
        return True
    text = value.strip()
    if text.lower() in ("0x...", "0x…", "0x"):
        return True
    if "..." in text or "…" in text:
        return True
    return False


def _is_evm_address(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(_EVM_ADDR_RE.match(value.strip()))


def _get_env_first(keys: list[str]) -> Optional[str]:
    for key in keys:
        env_value = os.getenv(key)
        if env_value and env_value.strip():
            return env_value.strip()
    return None


def _resolve_addr(name: str, current: Optional[str], env_keys: list[str]) -> str:
    if _is_placeholder_addr(current):
        current = _get_env_first(env_keys)

    if not _is_evm_address(current):
        raise ValueError(
            f"{name} 未配置或格式不合法：{current!r}。需要 0x + 40 位十六进制地址。"
            f" 你可以在 copytrade_config.json 里填 {name}，或设置环境变量：{env_keys}"
        )
    return current.strip()


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

    cfg["my_address"] = _resolve_addr(
        "my_address",
        cfg.get("my_address"),
        env_keys=[
            "POLY_FUNDER",
            "POLY_MY_ADDRESS",
            "MY_ADDRESS",
        ],
    )
    cfg["target_address"] = _resolve_addr(
        "target_address",
        cfg.get("target_address"),
        env_keys=[
            "COPYTRADE_TARGET",
            "CT_TARGET",
            "POLY_TARGET_ADDRESS",
            "TARGET_ADDRESS",
        ],
    )

    state = load_state(args.state)
    print(
        "[CFG] target="
        f"{cfg['target_address']} my={cfg['my_address']} ratio={cfg.get('follow_ratio')}"
    )
    state["target"] = cfg.get("target_address")
    state["my_address"] = cfg.get("my_address")
    state["follow_ratio"] = cfg.get("follow_ratio")
    state.setdefault("seen_tokens", [])
    state.setdefault("tracked_tokens", [])
    state.setdefault("ignored_tokens", {})
    state.setdefault("market_status_cache", {})
    state.setdefault("bootstrapped", False)
    if not isinstance(state.get("seen_tokens"), list):
        state["seen_tokens"] = []
    if not isinstance(state.get("tracked_tokens"), list):
        state["tracked_tokens"] = []
    if not isinstance(state.get("ignored_tokens"), dict):
        state["ignored_tokens"] = {}
    if not isinstance(state.get("market_status_cache"), dict):
        state["market_status_cache"] = {}
    if not isinstance(state.get("bootstrapped"), bool):
        state["bootstrapped"] = False

    data_client = DataApiClient()
    clob_client = init_clob_client()

    poll_interval = int(cfg.get("poll_interval_sec") or 20)
    size_threshold = float(cfg.get("size_threshold") or 0)
    follow_new_only = bool(cfg.get("follow_new_topics_only", False))
    bootstrap_skip = bool(cfg.get("bootstrap_skip_first_cycle", True))
    skip_closed = bool(cfg.get("skip_closed_markets", True))
    refresh_sec = int(cfg.get("market_status_refresh_sec") or 300)

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

        seen = set(state["seen_tokens"])
        tracked = set(state["tracked_tokens"])
        target_tokens_now = set(desired_by_token_id.keys())

        if follow_new_only and not state["bootstrapped"]:
            seen |= target_tokens_now
            state["seen_tokens"] = sorted(seen)
            state["tracked_tokens"] = sorted(tracked)
            state["bootstrapped"] = True
            print(
                "[BOOT] follow_new_topics_only=1，已忽略启动时存量 topics:"
                f" {len(target_tokens_now)} 个 token"
            )
            save_state(args.state, state)
            if bootstrap_skip:
                time.sleep(poll_interval)
                continue

        if follow_new_only:
            new_tokens = target_tokens_now - seen
            if new_tokens:
                print(f"[NEW] 发现新 topics: {len(new_tokens)} 个 token，将开始跟随")
            seen |= new_tokens
            tracked |= new_tokens
            state["seen_tokens"] = sorted(seen)
            state["tracked_tokens"] = sorted(tracked)

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
        reconcile_set.update(state.get("open_orders", {}).keys())
        if follow_new_only:
            reconcile_set = (reconcile_set & tracked) | set(state.get("open_orders", {}).keys())
        else:
            reconcile_set.update(my_by_token_id)

        ignored = state["ignored_tokens"]
        status_cache = state["market_status_cache"]
        if skip_closed:
            need_query = []
            for token_id in reconcile_set:
                if token_id in ignored:
                    continue
                cached = status_cache.get(token_id)
                if not cached or now_ts - int(cached.get("ts") or 0) >= refresh_sec:
                    need_query.append(token_id)

            if need_query:
                meta_map = gamma_fetch_markets_by_clob_token_ids(need_query)
                for token_id in need_query:
                    meta = meta_map.get(token_id)
                    tradeable = market_is_tradeable(meta) if meta else False
                    status_cache[token_id] = {"ts": now_ts, "tradeable": tradeable, "meta": meta}

        orderbooks: Dict[str, Dict[str, Optional[float]]] = {}
        total_notional = 0.0
        if float(cfg.get("max_notional_total") or 0) > 0:
            for token_id in reconcile_set:
                if skip_closed:
                    cached = status_cache.get(token_id)
                    if token_id in ignored or (cached and cached.get("tradeable") is False):
                        continue
                ob = get_orderbook(clob_client, token_id)
                orderbooks[token_id] = ob
                ref_price = _mid_price(ob)
                if ref_price is None:
                    continue
                desired = desired_by_token_id.get(token_id, 0.0)
                total_notional += abs(desired) * ref_price
            cfg["_total_notional"] = total_notional

        for token_id in reconcile_set:
            if skip_closed and token_id in ignored:
                if token_id in state.get("open_orders", {}):
                    cancels = [
                        {"type": "cancel", "order_id": o.get("order_id")}
                        for o in state["open_orders"].get(token_id, [])
                        if o.get("order_id")
                    ]
                    if cancels:
                        updated_orders = apply_actions(
                            clob_client,
                            cancels,
                            state["open_orders"].get(token_id, []),
                            now_ts,
                            args.dry_run,
                        )
                        if updated_orders:
                            state.setdefault("open_orders", {})[token_id] = updated_orders
                        else:
                            state.get("open_orders", {}).pop(token_id, None)
                continue

            if skip_closed:
                cached = status_cache.get(token_id)
                if cached and cached.get("tradeable") is False:
                    ignored[token_id] = {"ts": now_ts, "reason": "closed_or_inactive"}
                    if token_id in state.get("open_orders", {}):
                        cancels = [
                            {"type": "cancel", "order_id": o.get("order_id")}
                            for o in state["open_orders"].get(token_id, [])
                            if o.get("order_id")
                        ]
                        if cancels:
                            updated_orders = apply_actions(
                                clob_client,
                                cancels,
                                state["open_orders"].get(token_id, []),
                                now_ts,
                                args.dry_run,
                            )
                            if updated_orders:
                                state.setdefault("open_orders", {})[token_id] = updated_orders
                            else:
                                state.get("open_orders", {}).pop(token_id, None)
                    if follow_new_only and token_id in tracked:
                        tracked.discard(token_id)
                        state["tracked_tokens"] = sorted(tracked)
                    meta = (cached or {}).get("meta") or {}
                    slug = meta.get("slug") or meta.get("marketSlug") or ""
                    print(f"[SKIP] market closed/inactive token_id={token_id} slug={slug}")
                    continue

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
