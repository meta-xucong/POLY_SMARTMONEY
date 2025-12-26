from __future__ import annotations

import argparse
import json
import logging
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

from ct_data import fetch_positions_norm, fetch_target_actions_since
from ct_exec import (
    apply_actions,
    cancel_expired_only,
    fetch_open_orders_norm,
    get_orderbook,
    reconcile_one,
)
from ct_resolver import (
    gamma_fetch_markets_by_clob_token_ids,
    market_tradeable_state,
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


def _shorten_address(address: str) -> str:
    text = address.strip()
    if len(text) <= 12:
        return text
    return f"{text[:6]}..{text[-4:]}"


def _setup_logging(cfg: Dict[str, Any], target_address: str) -> logging.Logger:
    log_dir = Path(cfg.get("log_dir") or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    short = _shorten_address(target_address)
    log_path = log_dir / f"copytrade_{short}_{timestamp}_pid{pid}.log"

    level_name = str(cfg.get("log_level") or "INFO").upper()
    level = logging.INFO
    if level_name in logging._nameToLevel:
        level = logging._nameToLevel[level_name]

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info("日志初始化完成: %s", log_path)
    return logger


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


def _calc_used_notional_total(
    my_by_token_id: Dict[str, float],
    open_orders_by_token_id: Dict[str, list[dict]],
    mid_cache: Dict[str, float],
    max_position_usd_per_token: float,
) -> float:
    pos_total = 0.0
    for token_id, shares in my_by_token_id.items():
        mid = float(mid_cache.get(token_id, 0.5))
        pos_usd = abs(shares) * mid
        if max_position_usd_per_token > 0:
            pos_usd = min(pos_usd, max_position_usd_per_token)
        pos_total += pos_usd

    buy_orders_total = 0.0
    for orders in open_orders_by_token_id.values():
        for order in orders or []:
            side = str(order.get("side") or "").upper()
            if side != "BUY":
                continue
            size = float(order.get("size") or 0.0)
            price = float(order.get("price") or 0.0)
            buy_orders_total += abs(size) * price

    return pos_total + buy_orders_total


def _collect_order_ids(open_orders_by_token_id: Dict[str, list[dict]]) -> set[str]:
    order_ids: set[str] = set()
    for orders in open_orders_by_token_id.values():
        for order in orders or []:
            order_id = order.get("order_id")
            if order_id:
                order_ids.add(str(order_id))
    return order_ids


def _action_identity(action: Dict[str, object]) -> str:
    raw = action.get("raw") or {}
    if isinstance(raw, dict):
        tx_hash = raw.get("txHash") or raw.get("tx_hash") or raw.get("transactionHash")
        log_index = raw.get("logIndex") or raw.get("log_index")
        fill_id = raw.get("fillId") or raw.get("fill_id")
        if tx_hash and log_index is not None:
            return f"tx:{tx_hash}:{log_index}"
        if fill_id is not None:
            return f"fill:{fill_id}"
        if tx_hash:
            return f"tx:{tx_hash}"
    token_id = action.get("token_id") or ""
    side = action.get("side") or ""
    size = action.get("size") or ""
    ts = action.get("timestamp")
    action_ms = int(ts.timestamp() * 1000) if ts else 0
    price = ""
    if isinstance(raw, dict):
        price = raw.get("price") or raw.get("fillPrice") or raw.get("avgPrice") or ""
    return f"fallback:{token_id}:{side}:{size}:{price}:{action_ms}"


def _prune_order_ts_by_id(state: Dict[str, Any]) -> None:
    order_ts_by_id = state.get("order_ts_by_id")
    if not isinstance(order_ts_by_id, dict):
        state["order_ts_by_id"] = {}
        return
    active_ids = _collect_order_ids(state.get("open_orders", {}))
    for order_id in list(order_ts_by_id.keys()):
        if str(order_id) not in active_ids:
            order_ts_by_id.pop(order_id, None)


def _maybe_update_target_last(
    state: Dict[str, Any],
    token_id: str,
    t_now: Optional[float],
    should_update: bool,
) -> None:
    if should_update and t_now is not None:
        state.setdefault("target_last_shares", {})[token_id] = float(t_now)


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

    logger = _setup_logging(cfg, cfg["target_address"])

    state = load_state(args.state)
    state.setdefault("sizing", {})
    state["sizing"].setdefault("ema_delta_usd", None)
    logger.info(
        "[CFG] target=%s my=%s ratio=%s",
        cfg["target_address"],
        cfg["my_address"],
        cfg.get("follow_ratio"),
    )
    state["target"] = cfg.get("target_address")
    state["my_address"] = cfg.get("my_address")
    state["follow_ratio"] = cfg.get("follow_ratio")
    state.setdefault("open_orders", {})
    state.setdefault("token_map", {})
    state.setdefault("bootstrapped", False)
    state.setdefault("boot_token_ids", [])
    state.setdefault("probed_token_ids", [])
    state.setdefault("ignored_tokens", {})
    state.setdefault("market_status_cache", {})
    state.setdefault("target_last_shares", {})
    state.setdefault("target_last_seen_ts", {})
    state.setdefault("target_missing_streak", {})
    state.setdefault("cooldown_until", {})
    state.setdefault("target_last_event_ts", {})
    state.setdefault("target_actions_cursor_ms", 0)
    state.setdefault("last_mid_price_by_token_id", {})
    state.setdefault("order_ts_by_id", {})
    state.setdefault("seen_action_ids", [])
    if not isinstance(state.get("open_orders"), dict):
        state["open_orders"] = {}
    if not isinstance(state.get("token_map"), dict):
        state["token_map"] = {}
    if not isinstance(state.get("bootstrapped"), bool):
        state["bootstrapped"] = False
    if not isinstance(state.get("boot_token_ids"), list):
        state["boot_token_ids"] = []
    if not isinstance(state.get("probed_token_ids"), list):
        state["probed_token_ids"] = []
    if not isinstance(state.get("ignored_tokens"), dict):
        state["ignored_tokens"] = {}
    if not isinstance(state.get("market_status_cache"), dict):
        state["market_status_cache"] = {}
    if not isinstance(state.get("target_last_shares"), dict):
        state["target_last_shares"] = {}
    if not isinstance(state.get("target_last_seen_ts"), dict):
        state["target_last_seen_ts"] = {}
    if not isinstance(state.get("target_missing_streak"), dict):
        state["target_missing_streak"] = {}
    if not isinstance(state.get("cooldown_until"), dict):
        state["cooldown_until"] = {}
    if not isinstance(state.get("target_last_event_ts"), dict):
        state["target_last_event_ts"] = {}
    if not isinstance(state.get("target_actions_cursor_ms"), (int, float)):
        state["target_actions_cursor_ms"] = 0
    if not isinstance(state.get("last_mid_price_by_token_id"), dict):
        state["last_mid_price_by_token_id"] = {}
    if not isinstance(state.get("order_ts_by_id"), dict):
        state["order_ts_by_id"] = {}
    if not isinstance(state.get("seen_action_ids"), list):
        state["seen_action_ids"] = []

    data_client = DataApiClient()
    clob_client = init_clob_client()

    poll_interval = int(cfg.get("poll_interval_sec") or 20)
    size_threshold = float(cfg.get("size_threshold") or 0)
    skip_closed = bool(cfg.get("skip_closed_markets", True))
    refresh_sec = int(cfg.get("market_status_refresh_sec") or 300)
    positions_limit = int(cfg.get("positions_limit") or 500)
    positions_max_pages = int(cfg.get("positions_max_pages") or 20)
    actions_page_size = int(cfg.get("actions_page_size") or 300)
    actions_max_offset = int(cfg.get("actions_max_offset") or 10000)

    if int(state.get("target_actions_cursor_ms") or 0) <= 0:
        state["target_actions_cursor_ms"] = int(time.time() * 1000)

    while True:
        now_ts = int(time.time())
        ttl_sec = int(cfg.get("order_ttl_sec") or 0)
        try:
            remote_orders, ok, err = fetch_open_orders_norm(clob_client)
            if ok:
                remote_by_token: Dict[str, list[dict]] = {}
                order_ts_by_id = state.setdefault("order_ts_by_id", {})
                remote_order_ids: set[str] = set()
                for order in remote_orders:
                    order_id = str(order["order_id"])
                    ts = order.get("ts") or order_ts_by_id.get(order_id) or now_ts
                    if order_id not in order_ts_by_id:
                        order_ts_by_id[order_id] = int(ts)
                    remote_order_ids.add(order_id)
                    remote_by_token.setdefault(order["token_id"], []).append(
                        {
                            "order_id": order_id,
                            "side": order["side"],
                            "price": order["price"],
                            "size": order["size"],
                            "ts": int(ts),
                        }
                    )
                for order_id in list(order_ts_by_id.keys()):
                    if str(order_id) not in remote_order_ids:
                        order_ts_by_id.pop(order_id, None)
                state["open_orders"] = remote_by_token
            else:
                logger.warning("[WARN] sync open orders failed: %s", err)
        except Exception as exc:
            logger.exception("[ERR] sync open orders failed: %s", exc)

        state["open_orders"] = cancel_expired_only(
            clob_client,
            state.get("open_orders", {}),
            now_ts,
            ttl_sec,
            args.dry_run,
        )
        _prune_order_ts_by_id(state)

        actions_delta_by_token_id: Dict[str, float] = {}
        actions_info: Dict[str, object] = {"ok": True, "incomplete": False}
        actions_list: list[Dict[str, object]] = []
        actions_cursor_ms = int(state.get("target_actions_cursor_ms") or 0)
        try:
            actions_list, actions_info = fetch_target_actions_since(
                data_client,
                cfg["target_address"],
                actions_cursor_ms,
                page_size=actions_page_size,
                max_offset=actions_max_offset,
            )
            seen_action_ids = state.setdefault("seen_action_ids", [])
            seen_action_set = {str(item) for item in seen_action_ids}
            filtered_actions: list[Dict[str, object]] = []
            for action in actions_list:
                action_id = _action_identity(action)
                if action_id in seen_action_set:
                    continue
                filtered_actions.append(action)
                seen_action_ids.append(action_id)
                seen_action_set.add(action_id)
            if len(seen_action_ids) > 2000:
                del seen_action_ids[:-2000]
            actions_list = filtered_actions

            for action in actions_list:
                token_id = action.get("token_id")
                token_key = action.get("token_key")
                side = str(action.get("side") or "").upper()
                size = float(action.get("size") or 0.0)
                if token_id:
                    tid = str(token_id)
                    delta = size if side == "BUY" else -size
                    actions_delta_by_token_id[tid] = actions_delta_by_token_id.get(tid, 0.0) + delta
            latest_action_ms = int(actions_info.get("latest_ms") or 0)
            if actions_info.get("ok") and not actions_info.get("incomplete"):
                if latest_action_ms > actions_cursor_ms:
                    state["target_actions_cursor_ms"] = latest_action_ms
            else:
                logger.warning(
                    "[WARN] actions incomplete or failed; keep cursor_ms=%s", actions_cursor_ms
                )
        except Exception as exc:
            logger.exception("[ERR] fetch target actions failed: %s", exc)

        target_pos, target_info = fetch_positions_norm(
            data_client,
            cfg["target_address"],
            size_threshold,
            positions_limit=positions_limit,
            positions_max_pages=positions_max_pages,
        )
        hard_cap = positions_limit * positions_max_pages
        if len(target_pos) >= hard_cap:
            target_info["incomplete"] = True
            logger.info("[SAFE] target positions 可能截断(len>=hard_cap=%s), 仅撤单", hard_cap)

        my_pos, my_info = fetch_positions_norm(
            data_client,
            cfg["my_address"],
            0.0,
            positions_limit=positions_limit,
            positions_max_pages=positions_max_pages,
        )
        if len(my_pos) >= hard_cap:
            my_info["incomplete"] = True
            logger.info("[SAFE] my positions 可能截断(len>=hard_cap=%s), 仅撤单", hard_cap)

        logger.info(
            "[POS] target_count=%s my_count=%s target_incomplete=%s my_incomplete=%s",
            len(target_pos),
            len(my_pos),
            bool(target_info.get("incomplete")),
            bool(my_info.get("incomplete")),
        )
        if target_info.get("incomplete"):
            logger.info(
                "[POS] target positions info limit=%s total=%s max_pages=%s",
                target_info.get("limit"),
                target_info.get("total"),
                target_info.get("max_pages"),
            )

        if not target_info.get("ok") or target_info.get("incomplete"):
            logger.warning("[SAFE] target positions 不完整，仅撤单")
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

        if not my_info.get("ok") or my_info.get("incomplete"):
            logger.warning("[SAFE] my positions 不完整，仅撤单")
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

        token_key_by_token_id: Dict[str, str] = {
            token_id: token_key for token_key, token_id in state.get("token_map", {}).items()
        }

        target_shares_now_by_token_id: Dict[str, float] = {}
        for pos in target_pos:
            token_key = pos["token_key"]
            try:
                token_id = resolve_token_id(token_key, pos, state["token_map"])
            except Exception as exc:
                logger.warning("[WARN] resolver 失败: %s -> %s", token_key, exc)
                continue
            target_shares_now_by_token_id[token_id] = float(pos["size"])
            token_key_by_token_id[token_id] = token_key

        my_by_token_id: Dict[str, float] = {}
        for pos in my_pos:
            token_key = pos["token_key"]
            try:
                token_id = resolve_token_id(token_key, pos, state["token_map"])
            except Exception as exc:
                logger.warning("[WARN] resolver 失败(自身): %s -> %s", token_key, exc)
                continue
            my_by_token_id[token_id] = float(pos["size"])
            token_key_by_token_id.setdefault(token_id, token_key)

        for action in actions_list:
            token_id = action.get("token_id")
            token_key = action.get("token_key")
            if token_id:
                token_key_by_token_id.setdefault(str(token_id), str(token_key or ""))
                continue
            if not token_key:
                continue
            try:
                token_id = resolve_token_id(
                    token_key,
                    {
                        "token_key": token_key,
                        "condition_id": action.get("condition_id"),
                        "outcome_index": action.get("outcome_index"),
                        "slug": None,
                        "raw": action.get("raw") or {},
                    },
                    state["token_map"],
                )
            except Exception as exc:
                logger.warning("[WARN] resolver 失败(actions): %s -> %s", token_key, exc)
                continue
            side = str(action.get("side") or "").upper()
            size = float(action.get("size") or 0.0)
            delta = size if side == "BUY" else -size
            tid = str(token_id)
            actions_delta_by_token_id[tid] = actions_delta_by_token_id.get(tid, 0.0) + delta
            token_key_by_token_id.setdefault(tid, str(token_key))

        boot_sync_mode = str(cfg.get("boot_sync_mode") or "baseline_only").lower()
        if not state.get("bootstrapped") and boot_sync_mode == "baseline_only":
            boot_token_ids = sorted(target_shares_now_by_token_id.keys())
            state["boot_token_ids"] = boot_token_ids
            for token_id, t_now in target_shares_now_by_token_id.items():
                state.setdefault("target_last_shares", {})[token_id] = float(t_now)
                state.setdefault("target_last_seen_ts", {})[token_id] = now_ts
                state.setdefault("target_missing_streak", {})[token_id] = 0
            state["bootstrapped"] = True
            logger.info("[BOOT] baseline_only: baseline_tokens=%s", len(boot_token_ids))
            save_state(args.state, state)
            time.sleep(poll_interval)
            continue

        reconcile_set: Set[str] = set(target_shares_now_by_token_id)
        reconcile_set.update(state.get("target_last_shares", {}).keys())
        reconcile_set.update(my_by_token_id)
        reconcile_set.update(state.get("open_orders", {}).keys())
        reconcile_set.update(actions_delta_by_token_id.keys())

        ignored = state["ignored_tokens"]
        expired_ignored = [
            token_id
            for token_id, meta in ignored.items()
            if isinstance(meta, dict)
            and meta.get("expires_at")
            and now_ts >= int(meta.get("expires_at") or 0)
        ]
        for token_id in expired_ignored:
            ignored.pop(token_id, None)
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
                    tradeable = market_tradeable_state(meta)
                    status_cache[token_id] = {"ts": now_ts, "tradeable": tradeable, "meta": meta}

        orderbooks: Dict[str, Dict[str, Optional[float]]] = {}

        mode = str(cfg.get("order_size_mode") or "fixed_shares").lower()
        min_usd = float(cfg.get("min_order_usd") or 5.0)
        max_usd = float(cfg.get("max_order_usd") or 25.0)
        target_mid_usd = (min_usd + max_usd) / 2.0
        max_position_usd_per_token = float(cfg.get("max_position_usd_per_token") or 0.0)
        cooldown_sec = int(cfg.get("cooldown_sec_per_token") or 0)
        missing_timeout_sec = int(cfg.get("missing_timeout_sec") or 0)
        missing_to_zero_rounds = int(cfg.get("missing_to_zero_rounds") or 2)
        orphan_cancel_rounds = int(cfg.get("orphan_cancel_rounds") or 3)
        orphan_ignore_sec = int(cfg.get("orphan_ignore_sec") or 120)
        debug_token_ids = {str(token_id) for token_id in (cfg.get("debug_token_ids") or [])}
        eps = float(cfg.get("delta_eps") or 1e-9)

        ema = state.get("sizing", {}).get("ema_delta_usd")
        if ema is None or ema <= 0:
            ema = target_mid_usd * 3.0

        k = target_mid_usd / max(ema, 1e-9)
        k = max(0.002, min(1.2, k))

        cfg["_auto_order_k"] = k

        delta_usd_samples = []

        planned_total_notional = _calc_used_notional_total(
            my_by_token_id,
            state.get("open_orders", {}),
            state.get("last_mid_price_by_token_id", {}),
            max_position_usd_per_token,
        )

        for token_id in reconcile_set:
            if cooldown_sec > 0:
                cooldown_until = int(state.get("cooldown_until", {}).get(token_id) or 0)
                if now_ts < cooldown_until:
                    logger.info(
                        "[COOLDOWN] token_id=%s until=%s",
                        token_id,
                        cooldown_until,
                    )
                    continue
            if skip_closed:
                if token_id in ignored:
                    if token_id in state.get("open_orders", {}) and state["open_orders"].get(token_id):
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
                            _prune_order_ts_by_id(state)
                            planned_total_notional = _calc_used_notional_total(
                                my_by_token_id,
                                state.get("open_orders", {}),
                                state.get("last_mid_price_by_token_id", {}),
                                max_position_usd_per_token,
                            )
                            if cooldown_sec > 0:
                                state.setdefault("cooldown_until", {})[token_id] = (
                                    now_ts + cooldown_sec
                                )
                    continue

                cached = status_cache.get(token_id) or {}
                tradeable = cached.get("tradeable")

                if tradeable is False:
                    ignored[token_id] = {"ts": now_ts, "reason": "closed_or_not_tradeable"}
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
                            _prune_order_ts_by_id(state)
                            planned_total_notional = _calc_used_notional_total(
                                my_by_token_id,
                                state.get("open_orders", {}),
                                state.get("last_mid_price_by_token_id", {}),
                                max_position_usd_per_token,
                            )
                            if cooldown_sec > 0:
                                state.setdefault("cooldown_until", {})[token_id] = (
                                    now_ts + cooldown_sec
                                )
                    meta = cached.get("meta") or {}
                    slug = meta.get("slug") or ""
                    logger.info("[SKIP] closed/inactive token_id=%s slug=%s", token_id, slug)
                    continue

                if tradeable is None:
                    logger.warning("[WARN] market 状态未知(稍后重试): token_id=%s", token_id)
                    continue

            t_now_present = token_id in target_shares_now_by_token_id
            t_now = target_shares_now_by_token_id.get(token_id) if t_now_present else None
            t_last = state.get("target_last_shares", {}).get(token_id)
            my_shares = my_by_token_id.get(token_id, 0.0)
            open_orders = state.get("open_orders", {}).get(token_id, [])
            open_orders_count = len(open_orders)
            missing_streak = int(state.get("target_missing_streak", {}).get(token_id) or 0)
            last_seen_ts = int(state.get("target_last_seen_ts", {}).get(token_id) or 0)
            treating_missing_as_zero = False
            action_delta = actions_delta_by_token_id.get(token_id)

            if action_delta is None and not t_now_present:
                missing_streak += 1
                state.setdefault("target_missing_streak", {})[token_id] = missing_streak
                missing_timeout = (
                    missing_timeout_sec > 0
                    and last_seen_ts > 0
                    and now_ts - last_seen_ts >= missing_timeout_sec
                )
                if (
                    (missing_streak >= missing_to_zero_rounds or missing_timeout)
                    and t_last is not None
                    and float(t_last) > 0
                ):
                    t_now_present = True
                    t_now = 0.0
                    treating_missing_as_zero = True
                    state.setdefault("target_missing_streak", {})[token_id] = 0
                missing = t_now is None
                if (missing and (my_shares > 0 or open_orders_count > 0)) or (
                    token_id in debug_token_ids
                ):
                    legacy_desired = float(cfg.get("follow_ratio") or 0.0) * (
                        t_now or 0.0
                    )
                    legacy_delta = legacy_desired - my_shares
                    logger.info(
                        "[DBG] token_id=%s missing=%s missing_streak=%s t_now=%s t_last=%s "
                        "my_shares=%s open_orders_count=%s",
                        token_id,
                        missing,
                        missing_streak,
                        t_now,
                        t_last,
                        my_shares,
                        open_orders_count,
                    )
                    logger.info(
                        "[DBG] token_id=%s legacy_desired=%s legacy_delta=%s "
                        "old logic would SELL → churn risk",
                        token_id,
                        legacy_desired,
                        legacy_delta,
                    )
                if (
                    missing
                    and t_now is None
                    and my_shares == 0
                    and open_orders_count > 0
                    and missing_streak >= orphan_cancel_rounds
                ):
                    cancel_actions = [
                        {"type": "cancel", "order_id": order.get("order_id")}
                        for order in open_orders
                        if order.get("order_id")
                    ]
                    if cancel_actions:
                        logger.info(
                            "[ORPHAN] token_id=%s canceled=%s missing_streak=%s",
                            token_id,
                            len(cancel_actions),
                            missing_streak,
                        )
                        updated_orders = apply_actions(
                            clob_client,
                            cancel_actions,
                            open_orders,
                            now_ts,
                            args.dry_run,
                        )
                        if updated_orders:
                            state.setdefault("open_orders", {})[token_id] = updated_orders
                        else:
                            state.get("open_orders", {}).pop(token_id, None)
                        _prune_order_ts_by_id(state)
                    else:
                        state.get("open_orders", {}).pop(token_id, None)
                        _prune_order_ts_by_id(state)
                    if orphan_ignore_sec > 0:
                        state.setdefault("ignored_tokens", {})[token_id] = {
                            "expires_at": now_ts + orphan_ignore_sec,
                            "reason": "orphan_canceled",
                        }
                if not treating_missing_as_zero:
                    continue

            if action_delta is not None:
                state.setdefault("target_missing_streak", {})[token_id] = 0
                if t_now_present:
                    state.setdefault("target_last_seen_ts", {})[token_id] = now_ts
            elif not treating_missing_as_zero:
                state.setdefault("target_missing_streak", {})[token_id] = 0
                state.setdefault("target_last_seen_ts", {})[token_id] = now_ts

            should_update_last = t_now_present
            if t_last is None and action_delta is None:
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                should_probe = (
                    bool(state.get("bootstrapped"))
                    and token_id not in set(state.get("boot_token_ids", []))
                    and bool(cfg.get("probe_buy_on_first_seen", True))
                    and t_now is not None
                    and float(t_now) > 0
                    and token_id not in set(state.get("probed_token_ids", []))
                    and my_shares <= 0
                )
                has_buy_open = any(
                    str(order.get("side") or "").upper() == "BUY" for order in open_orders or []
                )
                if should_probe and has_buy_open and ttl_sec > 0:
                    stale_buy_orders = [
                        order
                        for order in open_orders
                        if str(order.get("side") or "").upper() == "BUY"
                        and now_ts - int(order.get("ts") or now_ts) >= ttl_sec
                    ]
                    if stale_buy_orders:
                        cancel_actions = [
                            {"type": "cancel", "order_id": order.get("order_id")}
                            for order in stale_buy_orders
                            if order.get("order_id")
                        ]
                        if cancel_actions:
                            logger.info(
                                "[PROBE] token_id=%s stale_buy_cancel=%s",
                                token_id,
                                len(cancel_actions),
                            )
                            updated_orders = apply_actions(
                                clob_client,
                                cancel_actions,
                                open_orders,
                                now_ts,
                                args.dry_run,
                            )
                            if updated_orders:
                                state.setdefault("open_orders", {})[token_id] = updated_orders
                                open_orders = updated_orders
                            else:
                                state.get("open_orders", {}).pop(token_id, None)
                                open_orders = []
                            _prune_order_ts_by_id(state)
                            has_buy_open = any(
                                str(order.get("side") or "").upper() == "BUY"
                                for order in open_orders or []
                            )
                if should_probe and not has_buy_open:
                    if token_id in orderbooks:
                        ob = orderbooks[token_id]
                    else:
                        ob = get_orderbook(clob_client, token_id)
                        orderbooks[token_id] = ob

                    ref_price = _mid_price(ob)
                    if ref_price is None:
                        logger.warning("[WARN] 无法获取盘口(探针): token_id=%s", token_id)
                        continue

                    state.setdefault("last_mid_price_by_token_id", {})[token_id] = float(
                        ref_price
                    )
                    probe_usd = float(
                        cfg.get("probe_order_usd") or cfg.get("min_order_usd") or 5.0
                    )
                    if probe_usd <= 0:
                        probe_usd = float(cfg.get("min_order_usd") or 5.0)
                    probe_shares = probe_usd / ref_price

                    cap_shares = float("inf")
                    if max_position_usd_per_token > 0:
                        cap_shares = max_position_usd_per_token / ref_price

                    my_target = min(my_shares + probe_shares, cap_shares)
                    delta = my_target - my_shares
                    if delta <= eps:
                        continue

                    desired_side = "BUY"
                    open_orders_for_reconcile = open_orders
                    if open_orders:
                        opposite_orders = [
                            order
                            for order in open_orders
                            if str(order.get("side") or "").upper() != desired_side
                        ]
                        if opposite_orders:
                            cancel_actions = [
                                {"type": "cancel", "order_id": order.get("order_id")}
                                for order in opposite_orders
                                if order.get("order_id")
                            ]
                            if cancel_actions:
                                logger.info(
                                    "[REVERSE] token_id=%s side=%s cancel_open_orders=%s",
                                    token_id,
                                    desired_side,
                                    len(cancel_actions),
                                )
                                updated_orders = apply_actions(
                                    clob_client,
                                    cancel_actions,
                                    open_orders,
                                    now_ts,
                                    args.dry_run,
                                )
                                if updated_orders:
                                    state.setdefault("open_orders", {})[
                                        token_id
                                    ] = updated_orders
                                    open_orders = updated_orders
                                else:
                                    state.get("open_orders", {}).pop(token_id, None)
                                    open_orders = []
                                _prune_order_ts_by_id(state)
                                planned_total_notional = _calc_used_notional_total(
                                    my_by_token_id,
                                    state.get("open_orders", {}),
                                    state.get("last_mid_price_by_token_id", {}),
                                    max_position_usd_per_token,
                                )
                                if cooldown_sec > 0:
                                    state.setdefault("cooldown_until", {})[token_id] = (
                                        now_ts + cooldown_sec
                                    )
                        open_orders_for_reconcile = [
                            order
                            for order in open_orders
                            if str(order.get("side") or "").upper() == desired_side
                        ]

                    token_key = token_key_by_token_id.get(token_id, f"token:{token_id}")
                    actions = reconcile_one(
                        token_id,
                        my_target,
                        my_shares,
                        ob,
                        open_orders_for_reconcile,
                        now_ts,
                        cfg,
                    )
                    if not actions:
                        continue
                    filtered_actions = []
                    for act in actions:
                        if act.get("type") != "place":
                            filtered_actions.append(act)
                            continue
                        side = str(act.get("side") or "")
                        price = float(act.get("price") or ref_price or 0.0)
                        size = float(act.get("size") or 0.0)
                        ok, reason = risk_check(
                            token_key,
                            size,
                            my_shares,
                            price,
                            cfg,
                            side=side,
                            planned_total_notional=planned_total_notional,
                        )
                        if not ok:
                            if reason == "max_notional_total":
                                max_total = float(cfg.get("max_notional_total") or 0.0)
                                order_notional = abs(size) * price
                                logger.warning(
                                    "[RISK] %s 拒绝: %s token=%s side=%s planned_total=%s "
                                    "order_notional=%s max_total=%s",
                                    token_key,
                                    reason,
                                    token_id,
                                    side,
                                    planned_total_notional,
                                    order_notional,
                                    max_total,
                                )
                            else:
                                logger.warning("[RISK] %s 拒绝: %s", token_key, reason)
                            continue
                        filtered_actions.append(act)
                        if act.get("type") == "place" and str(side).upper() == "BUY":
                            planned_total_notional += abs(size) * price
                    if not filtered_actions:
                        continue
                    actions = filtered_actions
                    logger.info("[ACTION] token_id=%s -> %s", token_id, actions)

                    updated_orders = apply_actions(
                        clob_client,
                        actions,
                        open_orders,
                        now_ts,
                        args.dry_run,
                    )
                    if updated_orders:
                        state.setdefault("open_orders", {})[token_id] = updated_orders
                    else:
                        state.get("open_orders", {}).pop(token_id, None)
                    _prune_order_ts_by_id(state)
                    planned_total_notional = _calc_used_notional_total(
                        my_by_token_id,
                        state.get("open_orders", {}),
                        state.get("last_mid_price_by_token_id", {}),
                        max_position_usd_per_token,
                    )

                    if cooldown_sec > 0 and actions:
                        state.setdefault("cooldown_until", {})[token_id] = (
                            now_ts + cooldown_sec
                        )

                    probed = set(state.get("probed_token_ids", []))
                    probed.add(token_id)
                    state["probed_token_ids"] = sorted(probed)
                continue

            if t_now is None and action_delta is None:
                continue

            d_target = action_delta if action_delta is not None else float(t_now) - float(t_last)
            if abs(d_target) <= eps:
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue

            if token_id in orderbooks:
                ob = orderbooks[token_id]
            else:
                ob = get_orderbook(clob_client, token_id)

            ref_price = _mid_price(ob)
            if ref_price is None:
                logger.warning("[WARN] 无法获取盘口: token_id=%s", token_id)
                logger.info("[NOOP] token_id=%s reason=orderbook_empty", token_id)
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue
            state.setdefault("last_mid_price_by_token_id", {})[token_id] = float(ref_price)

            cap_shares = float("inf")
            if max_position_usd_per_token > 0:
                cap_shares = max_position_usd_per_token / ref_price

            d_my = float(cfg.get("follow_ratio") or 0.0) * d_target
            if d_target > 0:
                logger.info(
                    "[SIGNAL] BUY token_id=%s d_target=%s d_my=%s my_shares=%s",
                    token_id,
                    d_target,
                    d_my,
                    my_shares,
                )
            elif d_target < 0:
                logger.info(
                    "[SIGNAL] SELL token_id=%s d_target=%s d_my=%s my_shares=%s",
                    token_id,
                    d_target,
                    d_my,
                    my_shares,
                )
            my_target = my_shares + d_my
            if my_target < 0:
                my_target = 0.0
            if my_target > cap_shares:
                my_target = cap_shares
            delta = my_target - my_shares
            if abs(delta) <= eps:
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue
            deadband_shares = float(cfg.get("deadband_shares") or 0.0)
            if abs(delta) <= deadband_shares:
                logger.info(
                    "[NOOP] token_id=%s reason=deadband delta=%s deadband=%s",
                    token_id,
                    delta,
                    deadband_shares,
                )
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue

            desired_side = "BUY" if delta > 0 else "SELL"
            open_orders_for_reconcile = open_orders
            if open_orders:
                opposite_orders = [
                    order
                    for order in open_orders
                    if str(order.get("side") or "").upper() != desired_side
                ]
                if opposite_orders:
                    cancel_actions = [
                        {"type": "cancel", "order_id": order.get("order_id")}
                        for order in opposite_orders
                        if order.get("order_id")
                    ]
                    if cancel_actions:
                        logger.info(
                            "[REVERSE] token_id=%s side=%s cancel_open_orders=%s",
                            token_id,
                            desired_side,
                            len(cancel_actions),
                        )
                        updated_orders = apply_actions(
                            clob_client,
                            cancel_actions,
                            open_orders,
                            now_ts,
                            args.dry_run,
                        )
                        if updated_orders:
                            state.setdefault("open_orders", {})[token_id] = updated_orders
                            open_orders = updated_orders
                        else:
                            state.get("open_orders", {}).pop(token_id, None)
                            open_orders = []
                        _prune_order_ts_by_id(state)
                        planned_total_notional = _calc_used_notional_total(
                            my_by_token_id,
                            state.get("open_orders", {}),
                            state.get("last_mid_price_by_token_id", {}),
                            max_position_usd_per_token,
                        )
                        if cooldown_sec > 0:
                            state.setdefault("cooldown_until", {})[token_id] = (
                                now_ts + cooldown_sec
                            )
                open_orders_for_reconcile = [
                    order
                    for order in open_orders
                    if str(order.get("side") or "").upper() == desired_side
                ]

            state.setdefault("target_last_event_ts", {})[token_id] = now_ts

            if mode == "auto_usd":
                delta_shares = abs(my_target - my_shares)
                delta_usd_samples.append(delta_shares * ref_price)

            token_key = token_key_by_token_id.get(token_id, f"token:{token_id}")
            actions = reconcile_one(
                token_id,
                my_target,
                my_shares,
                ob,
                open_orders_for_reconcile,
                now_ts,
                cfg,
            )
            if not actions:
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue
            filtered_actions = []
            blocked_reasons: set[str] = set()
            had_place_action = False
            for act in actions:
                if act.get("type") != "place":
                    filtered_actions.append(act)
                    continue
                had_place_action = True
                side = str(act.get("side") or "")
                price = float(act.get("price") or ref_price or 0.0)
                size = float(act.get("size") or 0.0)
                ok, reason = risk_check(
                    token_key,
                    size,
                    my_shares,
                    price,
                    cfg,
                    side=side,
                    planned_total_notional=planned_total_notional,
                )
                if not ok:
                    if reason == "max_notional_total":
                        max_total = float(cfg.get("max_notional_total") or 0.0)
                        order_notional = abs(size) * price
                        logger.warning(
                            "[RISK] %s 拒绝: %s token=%s side=%s planned_total=%s "
                            "order_notional=%s max_total=%s",
                            token_key,
                            reason,
                            token_id,
                            side,
                            planned_total_notional,
                            order_notional,
                            max_total,
                        )
                    else:
                        logger.warning("[RISK] %s 拒绝: %s", token_key, reason)
                    blocked_reasons.add(reason or "risk_check")
                    continue
                filtered_actions.append(act)
                if act.get("type") == "place" and str(side).upper() == "BUY":
                    planned_total_notional += abs(size) * price
            if not filtered_actions:
                if had_place_action:
                    reason_text = ",".join(sorted(blocked_reasons)) if blocked_reasons else "risk_check"
                    logger.info("[NOOP] token_id=%s reason=%s", token_id, reason_text)
                _maybe_update_target_last(state, token_id, t_now, should_update_last)
                continue
            actions = filtered_actions
            logger.info("[ACTION] token_id=%s -> %s", token_id, actions)

            updated_orders = apply_actions(
                clob_client,
                actions,
                open_orders,
                now_ts,
                args.dry_run,
            )
            if updated_orders:
                state.setdefault("open_orders", {})[token_id] = updated_orders
            else:
                state.get("open_orders", {}).pop(token_id, None)
            _prune_order_ts_by_id(state)
            planned_total_notional = _calc_used_notional_total(
                my_by_token_id,
                state.get("open_orders", {}),
                state.get("last_mid_price_by_token_id", {}),
                max_position_usd_per_token,
            )

            if cooldown_sec > 0 and actions:
                state.setdefault("cooldown_until", {})[token_id] = now_ts + cooldown_sec

            _maybe_update_target_last(state, token_id, t_now, should_update_last)

        if mode == "auto_usd" and delta_usd_samples:
            delta_usd_samples.sort()
            mid = delta_usd_samples[len(delta_usd_samples) // 2]
            alpha = 0.2
            new_ema = (1 - alpha) * ema + alpha * mid
            state.setdefault("sizing", {})["ema_delta_usd"] = new_ema
            state["sizing"]["last_k"] = cfg.get("_auto_order_k")

        state["last_sync_ts"] = now_ts
        save_state(args.state, state)
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
