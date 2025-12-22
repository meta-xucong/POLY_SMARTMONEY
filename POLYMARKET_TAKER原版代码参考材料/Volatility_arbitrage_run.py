# Volatility_arbitrage_run.py
# -*- coding: utf-8 -*-
"""
运行入口（循环策略版）：
- 事件页 /event/<slug>：列出子问题并选择（与老版一致）。
- 新增：
  1) 交互输入：买入份数（留空按 $1 反推）、跌幅窗口/阈值、盈利百分比、可选买入触发价；
  2) 基于 `VolArbStrategy` 的循环状态机：跌幅触发买入 → 成交确认 → 盈利达标卖出；
  3) 成交回调推进状态机，可重复执行买卖循环；
  4) 支持 stop 指令和市场关闭检测，安全退出。
"""
from __future__ import annotations
import sys
import os
import time
import threading
import re
import hmac
import hashlib
import json
from queue import Queue, Empty
from typing import Dict, Any, Tuple, List, Optional
from decimal import Decimal, ROUND_UP, ROUND_DOWN
import requests
from datetime import datetime, timezone
from Volatility_arbitrage_strategy import (
    StrategyConfig,
    VolArbStrategy,
    ActionType,
    Action,
)
from Volatility_buy import execute_auto_buy  # BUY 规范化逻辑统一交由执行器实现
from Volatility_sell import execute_auto_sell

# ========== 1) Client：优先 ws 版，回退 rest 版 ==========
def _get_client():
    try:
        from Volatility_arbitrage_main_ws import get_client  # 优先
        return get_client()
    except Exception as e1:
        try:
            from Volatility_arbitrage_main_rest import get_client  # 退回
            return get_client()
        except Exception as e2:
            print("[ERR] 无法导入 get_client：", e1, "|", e2)
            sys.exit(1)

# ========== 2) 保留 price_watch 的单市场解析函数（先尝试） ==========
try:
    from Volatility_arbitrage_price_watch import resolve_token_ids
except Exception as e:
    print("[ERR] 无法从 Volatility_arbitrage_price_watch 导入 resolve_token_ids：", e)
    sys.exit(1)

# ========== 3) 行情订阅（未动） ==========
try:
    from Volatility_arbitrage_main_ws import ws_watch_by_ids
except Exception as e:
    print("[ERR] 无法从 Volatility_arbitrage_main_ws 导入 ws_watch_by_ids：", e)
    sys.exit(1)

CLOB_API_HOST = "https://clob.polymarket.com"
GAMMA_ROOT = "https://gamma-api.polymarket.com"

# ===== 旧版解析器（复刻 + 极小修正） =====
def _parse_yes_no_ids_literal(source: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [x.strip() for x in source.split(",")]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    return None, None

def _extract_event_slug(s: str) -> str:
    m = re.search(r"/event/([^/?#]+)", s)
    if m: return m.group(1)
    s = s.strip()
    if s and ("/" not in s) and ("?" not in s) and ("&" not in s):
        return s
    return ""


def _extract_market_slug(s: str) -> str:
    m = re.search(r"/market/([^/?#]+)", s)
    if m:
        return m.group(1)
    s = s.strip()
    if s and ("/" not in s) and ("?" not in s) and ("&" not in s):
        return s
    return ""


def _parse_timestamp(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e12:
            ts = ts / 1000.0
        return ts
    if isinstance(val, str):
        raw = val.strip()
        if not raw:
            return None
        try:
            ts = float(raw)
            if ts > 1e12:
                ts = ts / 1000.0
            return ts
        except ValueError:
            pass
        iso = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
    return None


def _market_meta_from_obj(m: dict) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not isinstance(m, dict):
        return meta
    meta["slug"] = m.get("slug") or m.get("marketSlug") or m.get("market_slug")
    meta["market_id"] = (
        m.get("marketId")
        or m.get("id")
        or m.get("market_id")
        or m.get("conditionId")
        or m.get("condition_id")
    )

    end_keys = (
        "endDate",
        "endTime",
        "closeTime",
        "closeDate",
        "closedTime",
        "expiry",
        "expirationTime",
    )
    for key in end_keys:
        ts = _parse_timestamp(m.get(key))
        if ts:
            meta["end_ts"] = ts
            break

    resolve_keys = (
        "resolvedTime",
        "resolutionTime",
        "resolveTime",
        "resolvedAt",
        "finalizationTime",
        "finalizedTime",
        "settlementTime",
    )
    for key in resolve_keys:
        ts = _parse_timestamp(m.get(key))
        if ts:
            meta["resolved_ts"] = ts
            break

    if "end_ts" not in meta and "resolved_ts" in meta:
        meta["end_ts"] = meta["resolved_ts"]

    meta["raw"] = m
    return meta


def _maybe_fetch_market_meta_from_source(source: str) -> Dict[str, Any]:
    slug = _extract_market_slug(source)
    if not slug:
        return {}
    m = _fetch_market_by_slug(slug)
    if m:
        return _market_meta_from_obj(m)
    return {}


def _market_has_ended(meta: Dict[str, Any], now: Optional[float] = None) -> bool:
    if not meta:
        return False
    if now is None:
        now = time.time()
    candidates: List[float] = []
    for key in ("resolved_ts", "end_ts"):
        ts = meta.get(key)
        if isinstance(ts, (int, float)):
            candidates.append(float(ts))
    if not candidates:
        return False
    return now >= min(candidates)


def _extract_position_size(status: Dict[str, Any]) -> float:
    if not isinstance(status, dict):
        return 0.0
    for key in ("position_size", "position", "size"):
        val = status.get(key)
        if val is None:
            continue
        try:
            size = float(val)
            if size > 0:
                return size
        except (TypeError, ValueError):
            continue
    return 0.0


def _should_attempt_claim(
    meta: Dict[str, Any],
    status: Dict[str, Any],
    closed_by_ws: bool,
) -> bool:
    pos_size = _extract_position_size(status)
    if pos_size <= 0:
        return False
    if closed_by_ws:
        return True
    return _market_has_ended(meta)


def _resolve_client_host(client) -> str:
    env_host = os.getenv("POLY_HOST")
    if isinstance(env_host, str) and env_host.strip():
        return env_host.strip().rstrip("/")

    for attr in ("host", "_host", "base_url", "api_url"):
        val = getattr(client, attr, None)
        if isinstance(val, str) and val.strip():
            host = val.strip().rstrip("/")
            if "gamma-api" in host:
                return host.replace("gamma-api", "clob")
            return host

    return CLOB_API_HOST


def _extract_api_creds(client) -> Optional[Dict[str, str]]:
    def _pair_from_mapping(mp: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not isinstance(mp, dict):
            return None
        key_keys = ("key", "apiKey", "api_key", "id", "apiId", "api_id")
        secret_keys = ("secret", "apiSecret", "api_secret", "apiSecretKey")
        key_val = next((mp.get(k) for k in key_keys if mp.get(k)), None)
        secret_val = next((mp.get(k) for k in secret_keys if mp.get(k)), None)
        if key_val and secret_val:
            return {"key": str(key_val), "secret": str(secret_val)}
        return None

    def _pair_from_object(obj: Any) -> Optional[Dict[str, str]]:
        if obj is None:
            return None
        # 对部分库返回的命名元组/数据类做兼容
        for attr_key in ("key", "apiKey", "api_key", "id", "apiId", "api_id"):
            key_val = getattr(obj, attr_key, None)
            if key_val:
                break
        else:
            key_val = None
        for attr_secret in ("secret", "apiSecret", "api_secret", "apiSecretKey"):
            secret_val = getattr(obj, attr_secret, None)
            if secret_val:
                break
        else:
            secret_val = None
        if key_val and secret_val:
            return {"key": str(key_val), "secret": str(secret_val)}
        if hasattr(obj, "to_dict"):
            try:
                return _pair_from_mapping(obj.to_dict())
            except Exception:
                return None
        return None

    def _pair_from_sequence(seq: Any) -> Optional[Dict[str, str]]:
        if not isinstance(seq, (list, tuple)) or len(seq) < 2:
            return None
        key_val, secret_val = seq[0], seq[1]
        if key_val and secret_val:
            return {"key": str(key_val), "secret": str(secret_val)}
        return None

    candidates = [
        getattr(client, "api_creds", None),
        getattr(client, "_api_creds", None),
    ]
    getter = getattr(client, "get_api_creds", None)
    if callable(getter):
        try:
            candidates.append(getter())
        except Exception:
            pass
    key = getattr(client, "api_key", None)
    secret = getattr(client, "api_secret", None)
    if key and secret:
        candidates.append({"key": key, "secret": secret})
    # 兼容直接从环境变量注入 API key/secret 的场景
    env_key = os.getenv("POLY_API_KEY")
    env_secret = os.getenv("POLY_API_SECRET")
    if env_key and env_secret:
        candidates.append({"key": env_key, "secret": env_secret})

    for cand in candidates:
        if cand is None:
            continue
        pair = None
        if isinstance(cand, dict):
            pair = _pair_from_mapping(cand)
        elif isinstance(cand, (list, tuple)):
            pair = _pair_from_sequence(cand)
        else:
            pair = _pair_from_object(cand)
        if pair:
            return pair
    return None


def _sign_payload(secret: str, timestamp: str, method: str, path: str, body: str) -> str:
    payload = f"{timestamp}{method.upper()}{path}{body}"
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _claim_via_http(client, market_id: str, token_id: Optional[str]) -> bool:
    creds = _extract_api_creds(client)
    if not creds:
        print("[CLAIM] 当前客户端缺少 API 凭证信息，无法调用 HTTP claim 接口。")
        return False

    host = _resolve_client_host(client)
    path = "/v1/user/clob/positions/claim"
    url = f"{host}{path}"
    payload: Dict[str, Any] = {"market": market_id}
    if token_id:
        payload["tokenIds"] = [token_id]

    body = json.dumps(payload, separators=(",", ":"))
    ts = str(int(time.time() * 1000))
    signature = _sign_payload(creds["secret"], ts, "POST", path, body)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": creds["key"],
        "X-API-Signature": signature,
        "X-API-Timestamp": ts,
    }

    try:
        resp = requests.post(url, data=body, headers=headers, timeout=10)
    except Exception as exc:
        print(f"[CLAIM] 请求 {url} 时出现异常：{exc}")
        return False

    if resp.status_code == 404:
        print("[CLAIM] 目标 claim 接口返回 404，请确认所使用的 Clob API 版本是否支持自动 claim。")
        return False
    if resp.status_code >= 500:
        print(f"[CLAIM] 服务端 {resp.status_code} 错误：{resp.text}")
        return False
    if resp.status_code in (401, 403):
        print(f"[CLAIM] 接口拒绝访问（{resp.status_code}）：{resp.text}")
        return False

    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    print(f"[CLAIM] HTTP {path} 返回状态 {resp.status_code}，响应：{data}")
    if isinstance(data, dict) and data.get("error"):
        return False
    return resp.ok


def _extract_positions_from_data_api_response(payload: Any) -> Optional[List[dict]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        return None
    return None


def _fetch_positions_from_data_api(client) -> Tuple[List[dict], bool, str]:
    address = None
    for attr in ("funder", "owner", "address", "wallet"):
        cand = getattr(client, attr, None)
        if cand:
            address = cand
            break

    if not address:
        return [], False, "缺少地址，无法从数据接口拉取持仓。"

    limit = 500
    offset = 0
    collected: List[dict] = []
    total_records: Optional[int] = None
    url = f"{GAMMA_ROOT}/positions"

    try:
        while True:
            params = {"address": address, "limit": limit, "offset": offset}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            positions = _extract_positions_from_data_api_response(payload)
            if positions is None:
                return [], False, "数据接口返回格式异常，缺少 data 字段。"
            collected.extend(positions)
            meta = payload.get("meta") if isinstance(payload, dict) else {}
            if isinstance(meta, dict):
                raw_total = meta.get("total") or meta.get("count")
                try:
                    if raw_total is not None:
                        total_records = int(raw_total)
                except (TypeError, ValueError):
                    total_records = None
            if not positions or (total_records is not None and len(collected) >= total_records):
                break
            offset += len(positions)
    except requests.RequestException as exc:
        return [], False, f"请求失败：{exc}"

    total = total_records if total_records is not None else len(collected)
    origin = f"data-api positions(limit={limit}, total={total})"
    return collected, True, origin


def _attempt_claim(client, meta: Dict[str, Any], token_id: str) -> None:
    market_id = meta.get("market_id") if isinstance(meta, dict) else None
    print(f"[CLAIM] 检测到需处理的未平仓仓位，token_id={token_id}，开始尝试 claim…")
    if not market_id:
        print("[CLAIM] 未找到 market_id，无法自动 claim，请手动处理。")
        return

    claim_fn = getattr(client, "claim_positions", None)
    if callable(claim_fn):
        claim_kwargs = {"market": market_id}
        if token_id:
            claim_kwargs["token_ids"] = [token_id]
        try:
            print(f"[CLAIM] 尝试调用 claim_positions({claim_kwargs})…")
            resp = claim_fn(**claim_kwargs)
            print(f"[CLAIM] 响应: {resp}")
            return
        except TypeError as exc:
            print(f"[CLAIM] claim_positions 参数不匹配: {exc}，改用 HTTP 接口。")
        except Exception as exc:
            print(f"[CLAIM] 调用 claim_positions 失败: {exc}，改用 HTTP 接口。")

    if _claim_via_http(client, market_id, token_id):
        return

    print("[CLAIM] 未找到可用的 claim 方法，请手动处理。")

def _http_json(url: str, params=None) -> Optional[Any]:
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _list_markets_under_event(event_slug: str) -> List[dict]:
    if not event_slug:
        return []
    # A) /events?slug=<slug>
    data = _http_json(f"{GAMMA_ROOT}/events", params={"slug": event_slug, "closed": "false"})
    evs = []
    if isinstance(data, dict) and "data" in data:
        evs = data["data"]
    elif isinstance(data, list):
        evs = data
    if isinstance(evs, list):
        for ev in evs:
            mkts = ev.get("markets") or []
            if mkts:
                return mkts
    # B) /markets?search=<slug> 精确过滤 eventSlug
    data = _http_json(f"{GAMMA_ROOT}/markets", params={"limit": 200, "active": "true", "search": event_slug})
    mkts = []
    if isinstance(data, dict) and "data" in data:
        mkts = data["data"]
    elif isinstance(data, list):
        mkts = data
    if isinstance(mkts, list):
        return [m for m in mkts if str(m.get("eventSlug") or "") == str(event_slug)]
    return []

def _fetch_market_by_slug(market_slug: str) -> Optional[dict]:
    return _http_json(f"{GAMMA_ROOT}/markets/slug/{market_slug}")

def _pick_market_subquestion(markets: List[dict]) -> dict:
    print("[CHOICE] 该事件下存在多个子问题，请选择其一，或直接粘贴具体子问题URL：")
    for i, m in enumerate(markets):
        title = m.get("title") or m.get("question") or m.get("slug")
        end_ts = m.get("endDate") or m.get("endTime") or ""
        mslug = m.get("slug") or ""
        url = f"https://polymarket.com/market/{mslug}" if mslug else "(no slug)"
        print(f"  [{i}] {title}  (end={end_ts})  -> {url}")
    while True:
        s = input("请输入序号或粘贴URL：").strip()
        if s.startswith(("http://", "https://")):
            return {"__direct_url__": s}
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(markets):
                return markets[idx]
        print("请输入有效序号或URL。")

def _tokens_from_market_obj(m: dict) -> Tuple[str, str, str]:
    title = m.get("title") or m.get("question") or m.get("slug") or ""
    yes_id = no_id = ""
    ids = m.get("clobTokenIds") or m.get("clobTokens")
    # 兼容字符串形式的 clobTokenIds
    if isinstance(ids, str):
        try:
            import json as _json
            ids = _json.loads(ids)
        except Exception:
            ids = None
    if isinstance(ids, (list, tuple)) and len(ids) >= 2:
        return str(ids[0]), str(ids[1]), title
    # 兼容 tokens/outcomes 结构
    outcomes = m.get("outcomes") or m.get("tokens") or []
    if isinstance(outcomes, list) and outcomes and isinstance(outcomes[0], dict):
        for o in outcomes:
            name = (o.get("name") or o.get("outcome") or o.get("position") or "").strip().lower()
            tid = o.get("tokenId") or o.get("clobTokenId") or o.get("token_id") or o.get("id") or ""
            if not tid:
                continue
            if name in ("yes", "y", "true", "yes token", "yes_token"):
                yes_id = str(tid)
            elif name in ("no", "n", "false", "no token", "no_token"):
                no_id = str(tid)
        if yes_id and no_id:
            return yes_id, no_id, title
    # 兼容直接字段
    y = m.get("yesTokenId") or m.get("yes_token_id")
    n = m.get("noTokenId") or m.get("no_token_id")
    if y and n:
        return str(y), str(n), title
    return yes_id, no_id, title

def _resolve_with_fallback(source: str) -> Tuple[str, str, str, Dict[str, Any]]:
    # 1) "YES_id,NO_id"
    y, n = _parse_yes_no_ids_literal(source)
    if y and n:
        return y, n, "(Manual IDs)", {}
    # 2) 先尝试旧解析器（单一市场 URL/slug）
    try:
        y1, n1, title1, raw1 = resolve_token_ids(source)
        if y1 and n1:
            meta = _market_meta_from_obj(raw1 or {}) if raw1 else {}
            if not meta:
                meta = _maybe_fetch_market_meta_from_source(source)
            return y1, n1, title1, meta
    except Exception:
        pass
    # 2.5) 若上一步失败：把输入当作可能的 market slug（含 /event 路由别名）
    cand_slugs: List[str] = []
    ms = _extract_market_slug(source)
    if ms:
        cand_slugs.append(ms)
    es = _extract_event_slug(source)
    if es and es not in cand_slugs:
        cand_slugs.append(es)
    for slug in cand_slugs:
        # A) 直接按 /markets/slug/<slug>
        m = _fetch_market_by_slug(slug)
        if isinstance(m, dict):
            yx, nx, tx = _tokens_from_market_obj(m)
            if yx and nx:
                return yx, nx, tx or (m.get("title") or m.get("question") or slug), _market_meta_from_obj(m)
        # B) 用 /markets?search=<slug> 兜底（先 active，再放宽）
        for params in ({"limit": 200, "search": slug, "active": "true"}, {"limit": 200, "search": slug}):
            data = _http_json(f"{GAMMA_ROOT}/markets", params=params)
            mkts = []
            if isinstance(data, dict) and "data" in data:
                mkts = data["data"]
            elif isinstance(data, list):
                mkts = data
            if isinstance(mkts, list) and mkts:
                hit = None
                # 优先 slug 精确命中
                for m2 in mkts:
                    if str(m2.get("slug") or "") == slug:
                        hit = m2; break
                # 其次 eventSlug 命中
                if not hit:
                    for m2 in mkts:
                        if str(m2.get("eventSlug") or "") == slug:
                            hit = m2; break
                # 仍无则取第一条
                if not hit:
                    hit = mkts[0]
                if hit:
                    yx, nx, tx = _tokens_from_market_obj(hit)
                    if yx and nx:
                        return yx, nx, tx, _market_meta_from_obj(hit)
    # 3) 事件页/事件 slug 回退链路
    event_slug = _extract_event_slug(source)
    if not event_slug:
        raise ValueError("无法从输入中提取事件 slug，且直接解析失败。")
    mkts = _list_markets_under_event(event_slug)
    if not mkts:
        raise ValueError(f"未在事件 {event_slug} 下检索到子问题列表。")
    chosen = _pick_market_subquestion(mkts)
    if "__direct_url__" in chosen:
        y2, n2, title2, raw2 = resolve_token_ids(chosen["__direct_url__"])
        if y2 and n2:
            meta = _market_meta_from_obj(raw2 or {}) if raw2 else {}
            if not meta:
                meta = _maybe_fetch_market_meta_from_source(chosen["__direct_url__"])
            return y2, n2, title2, meta
        raise ValueError("无法从粘贴的URL解析出 tokenId。")
    y3, n3, title3 = _tokens_from_market_obj(chosen)
    if y3 and n3:
        meta = _market_meta_from_obj(chosen)
        return y3, n3, title3, meta
    slug2 = chosen.get("slug") or ""
    if slug2:
        # 兜底：拉完整市场详情；若还不行，再把 /market/<slug> 丢给旧解析器
        m_full = _fetch_market_by_slug(slug2)
        if m_full:
            y4, n4, title4 = _tokens_from_market_obj(m_full)
            if y4 and n4:
                meta = _market_meta_from_obj(m_full)
                return y4, n4, title4, meta
        y5, n5, title5, raw5 = resolve_token_ids(f"https://polymarket.com/market/{slug2}")
        if y5 and n5:
            meta = _market_meta_from_obj(raw5 or {}) if raw5 else {}
            if not meta:
                meta = _maybe_fetch_market_meta_from_source(f"https://polymarket.com/market/{slug2}")
            return y5, n5, title5, meta
    raise ValueError("子问题未包含 tokenId，且兜底解析失败。")

# ====== 下单执行工具 ======
def _floor(x: float, dp: int) -> float:
    q = Decimal(str(x)).quantize(Decimal("1." + "0"*dp), rounding=ROUND_DOWN)
    return float(q)

def _normalize_sell_pair(price: float, size: float) -> Tuple[float, float]:
    # 价格 4dp；份数 2dp（下单时再 floor 一次，确保不超）
    return _floor(price, 4), _floor(size, 2)

def _place_buy_fak(client, token_id: str, price: float, size: float) -> Dict[str, Any]:
    return execute_auto_buy(client=client, token_id=token_id, price=price, size=size)

def _place_sell_fok(client, token_id: str, price: float, size: float) -> Dict[str, Any]:
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import SELL
    eff_p, eff_s = _normalize_sell_pair(price, size)
    order = OrderArgs(token_id=str(token_id), side=SELL, price=float(eff_p), size=float(eff_s))
    signed = client.create_order(order)
    return client.post_order(signed, OrderType.FOK)

# ===== 主流程 =====
def main():
    client = _get_client()
    creds_check = _extract_api_creds(client)
    if not creds_check or not creds_check.get("key") or not creds_check.get("secret"):
        print("[ERR] 无法获取完整 API 凭证，请检查配置后重试。")
        return
    print("[INIT] API 凭证已验证。")
    print("[INIT] ClobClient 就绪。")
    print('请输入 Polymarket 市场 URL，或 "YES_id,NO_id"：')
    source = input().strip()
    if not source:
        print("[ERR] 未输入，退出。")
        return
    try:
        yes_id, no_id, title, market_meta = _resolve_with_fallback(source)
    except Exception as e:
        print("[ERR] 无法解析目标：", e)
        return
    market_meta = market_meta or {}
    print(f"[INFO] 市场/子问题标题: {title}")
    print(f"[INFO] 解析到 tokenIds: YES={yes_id} | NO={no_id}")

    def _fmt_ts(ts_val: Optional[float]) -> Optional[str]:
        if ts_val is None:
            return None
        try:
            ts_f = float(ts_val)
        except (TypeError, ValueError):
            return None
        dt = datetime.fromtimestamp(ts_f, tz=timezone.utc)
        return dt.isoformat()

    end_ts = market_meta.get("end_ts") if isinstance(market_meta, dict) else None
    resolved_ts = market_meta.get("resolved_ts") if isinstance(market_meta, dict) else None
    if end_ts or resolved_ts:
        end_str = _fmt_ts(end_ts)
        resolve_str = _fmt_ts(resolved_ts)
        if end_str:
            print(f"[INFO] 市场计划截止时间 (UTC): {end_str}")
        if resolve_str and resolve_str != end_str:
            print(f"[INFO] 市场预计结算时间 (UTC): {resolve_str}")

    def _calc_deadline(meta: Dict[str, Any]) -> Optional[float]:
        candidates: List[float] = []
        if isinstance(meta, dict):
            for key in ("end_ts", "resolved_ts"):
                ts_val = meta.get(key)
                if isinstance(ts_val, (int, float)):
                    candidates.append(float(ts_val))
        return min(candidates) if candidates else None

    market_deadline_ts = _calc_deadline(market_meta)
    if market_deadline_ts:
        dt_deadline = datetime.fromtimestamp(market_deadline_ts, tz=timezone.utc)
        print(
            "[INFO] 监控目标结束时间 (UTC): "
            f"{dt_deadline.isoformat()}"
        )
    else:
        print("[ERR] 未能获取市场结束时间，程序终止。")
        return

    sell_only_start_ts: Optional[float] = None
    if market_deadline_ts:
        print(
            "请输入倒计时开始时间（UTC）。"
            "可输入：\n"
            "  - 绝对时间，如 2024-01-01 12:30:00 或 ISO8601；\n"
            "  - 提前的分钟数，如输入 30 表示截止前 30 分钟进入仅卖出模式；\n"
            "留空表示不启用倒计时卖出保护。"
        )
        countdown_in = input().strip()
        if countdown_in:
            parsed_ts: Optional[float] = None
            used_minutes = False
            if re.search(r"[A-Za-z:/-]", countdown_in):
                parsed_ts = _parse_timestamp(countdown_in)
            else:
                try:
                    minutes_before = float(countdown_in)
                    parsed_ts = market_deadline_ts - minutes_before * 60.0
                    used_minutes = True
                except Exception:
                    parsed_ts = _parse_timestamp(countdown_in)
            if not parsed_ts:
                print("[ERR] 无法解析倒计时开始时间，程序终止。")
                return
            if parsed_ts >= market_deadline_ts:
                print("[ERR] 倒计时开始时间必须早于市场结束时间，程序终止。")
                return
            sell_only_start_ts = parsed_ts
            if used_minutes:
                print(
                    f"[INFO] 倒计时卖出模式将在市场结束前 {countdown_in} 分钟开启。"
                )
            else:
                dt_start = datetime.fromtimestamp(parsed_ts, tz=timezone.utc)
                print(
                    "[INFO] 倒计时卖出模式将在 UTC 时间 "
                    f"{dt_start.isoformat()} 开启。"
                )

    print('请选择方向（YES/NO），回车确认：')
    side = input().strip().upper()
    if side not in ("YES", "NO"):
        print("[ERR] 方向非法，退出。")
        return
    token_id = yes_id if side == "YES" else no_id

    print("请输入买入份数（留空=按 $1 反推）：")
    size_in = input().strip()
    print("请输入买入触发价（对标 ask，如 0.35，留空表示仅依赖跌幅触发）：")
    buy_px_in = input().strip()
    buy_threshold = None
    if buy_px_in:
        try:
            buy_threshold = float(buy_px_in)
        except Exception:
            print("[ERR] 触发价非法，退出。")
            return

    print("请输入跌幅窗口分钟数（默认 10）：")
    drop_window_in = input().strip()
    try:
        drop_window = float(drop_window_in) if drop_window_in else 10.0
    except Exception:
        print("[ERR] 跌幅窗口非法，退出。")
        return

    print("请输入跌幅触发百分比（默认 5 表示 5%）：")
    drop_pct_in = input().strip()
    try:
        drop_pct = float(drop_pct_in) / 100.0 if drop_pct_in else 0.05
    except Exception:
        print("[ERR] 跌幅百分比非法，退出。")
        return

    print("请输入卖出盈利百分比（默认 5 表示 +5%）：")
    profit_in = input().strip()
    try:
        profit_pct = float(profit_in) / 100.0 if profit_in else 0.05
    except Exception:
        print("[ERR] 盈利百分比非法，退出。")
        return

    sell_only_start_ts: Optional[float] = None
    if market_deadline_ts:
        print(
            "请输入倒计时开始时间（UTC）。"
            "可输入：\n"
            "  - 绝对时间，如 2024-01-01 12:30:00 或 ISO8601；\n"
            "  - 提前的分钟数，如输入 30 表示截止前 30 分钟进入仅卖出模式；\n"
            "留空表示不启用倒计时卖出保护。"
        )
        countdown_in = input().strip()
        if countdown_in:
            parsed_ts: Optional[float] = None
            used_minutes = False
            if re.search(r"[A-Za-z:/-]", countdown_in):
                parsed_ts = _parse_timestamp(countdown_in)
            else:
                try:
                    minutes_before = float(countdown_in)
                    parsed_ts = market_deadline_ts - minutes_before * 60.0
                    used_minutes = True
                except Exception:
                    parsed_ts = _parse_timestamp(countdown_in)
            if not parsed_ts:
                print("[ERR] 无法解析倒计时开始时间，程序终止。")
                return
            if parsed_ts >= market_deadline_ts:
                print("[ERR] 倒计时开始时间必须早于市场结束时间，程序终止。")
                return
            sell_only_start_ts = parsed_ts
            if used_minutes:
                print(
                    f"[INFO] 倒计时卖出模式将在市场结束前 {countdown_in} 分钟开启。"
                )
            else:
                dt_start = datetime.fromtimestamp(parsed_ts, tz=timezone.utc)
                print(
                    "[INFO] 倒计时卖出模式将在 UTC 时间 "
                    f"{dt_start.isoformat()} 开启。"
                )

    cfg = StrategyConfig(
        token_id=token_id,
        buy_price_threshold=buy_threshold,
        drop_window_minutes=drop_window,
        drop_pct=drop_pct,
        profit_pct=profit_pct,
    )
    strategy = VolArbStrategy(cfg)

    latest: Dict[str, Dict[str, Any]] = {}
    action_queue: Queue[Action] = Queue()
    stop_event = threading.Event()
    sell_only_event = threading.Event()
    market_closed_detected = False

    slug_for_refresh = ""
    if isinstance(market_meta, dict):
        slug_for_refresh = (
            str(market_meta.get("slug") or "")
            or str(market_meta.get("market_slug") or "")
        )
        if not slug_for_refresh:
            raw_meta = market_meta.get("raw") if isinstance(market_meta, dict) else {}
            if isinstance(raw_meta, dict):
                slug_for_refresh = str(raw_meta.get("slug") or "")
    if not slug_for_refresh:
        slug_for_refresh = _extract_market_slug(source)
    unable_to_refresh_logged = False

    def _refresh_market_meta() -> Dict[str, Any]:
        nonlocal market_meta, market_deadline_ts, unable_to_refresh_logged
        slug = slug_for_refresh
        if not slug:
            if not unable_to_refresh_logged:
                print("[COUNTDOWN] 无市场 slug，无法刷新事件状态，仅依赖本地信息。")
                unable_to_refresh_logged = True
            return market_meta
        m_obj = _fetch_market_by_slug(slug)
        if isinstance(m_obj, dict):
            refreshed = _market_meta_from_obj(m_obj)
            if refreshed:
                market_meta = refreshed
                new_deadline = _calc_deadline(market_meta)
                if new_deadline:
                    market_deadline_ts = new_deadline
        return market_meta

    def _calc_size_by_1dollar(ask_px: float) -> float:
        if not ask_px or ask_px <= 0:
            return 1.0
        s = 1.0 / ask_px
        return float(Decimal(str(s)).quantize(Decimal("1"), rounding=ROUND_UP))

    def _extract_ts(raw: Optional[Any]) -> float:
        if raw is None:
            return time.time()
        try:
            ts = float(raw)
        except Exception:
            return time.time()
        if ts > 1e12:
            ts = ts / 1000.0
        return ts

    def _is_market_closed(payload: Dict[str, Any]) -> bool:
        status_keys = ["status", "market_status", "marketStatus"]
        for key in status_keys:
            val = payload.get(key)
            if isinstance(val, str) and val.lower() in {"closed", "settled", "resolved", "expired"}:
                return True
        bool_keys = ["is_closed", "market_closed", "closed", "isMarketClosed"]
        for key in bool_keys:
            val = payload.get(key)
            if isinstance(val, bool) and val:
                return True
            if isinstance(val, str) and val.strip().lower() in {"true", "1", "yes"}:
                return True
        return False

    def _event_indicates_market_closed(ev: Dict[str, Any]) -> bool:
        if not isinstance(ev, dict):
            return False

        if _is_market_closed(ev):
            return True

        queue: List[Dict[str, Any]] = []
        for key in ("market", "market_state", "marketState", "marketStatus", "data", "payload"):
            val = ev.get(key)
            if isinstance(val, dict):
                queue.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        queue.append(item)

        while queue:
            item = queue.pop()
            if _is_market_closed(item):
                return True
            for key, val in item.items():
                if isinstance(val, dict):
                    queue.append(val)
                elif isinstance(val, list):
                    for sub in val:
                        if isinstance(sub, dict):
                            queue.append(sub)
        return False

    def _extract_field(resp: Any, key: str) -> Any:
        if isinstance(resp, dict):
            return resp.get(key)
        if hasattr(resp, key):
            return getattr(resp, key)
        return None

    def _extract_price(resp: Any, fallback: float) -> float:
        for key in (
            "avg_price",
            "avgPrice",
            "filled_avg_price",
            "filledAvgPrice",
            "price",
            "last_price",
            "lastPrice",
            "limit_price",
            "limitPrice",
        ):
            val = _extract_field(resp, key)
            if val is None:
                continue
            try:
                return float(val)
            except Exception:
                continue
        return float(fallback)

    def _extract_filled(resp: Any) -> Optional[float]:
        for key in (
            "filled",
            "filled_size",
            "filledSize",
            "filled_amount",
            "filledAmount",
            "makingAmount",
        ):
            val = _extract_field(resp, key)
            if val is None:
                continue
            try:
                return float(val)
            except Exception:
                continue
        return None

    def _extract_size(resp: Any, fallback: float) -> float:
        filled = _extract_filled(resp)
        if filled is not None:
            return float(filled)
        return float(fallback)

    def _extract_remaining(resp: Any) -> Optional[float]:
        val = _extract_field(resp, "remaining")
        if val is not None:
            try:
                return max(float(val), 0.0)
            except (TypeError, ValueError):
                pass
        requested_val = _extract_field(resp, "requested")
        filled_val = _extract_filled(resp)
        try:
            requested = float(requested_val) if requested_val is not None else None
        except (TypeError, ValueError):
            requested = None
        try:
            filled = float(filled_val) if filled_val is not None else None
        except (TypeError, ValueError):
            filled = None
        if requested is not None and filled is not None:
            return max(requested - filled, 0.0)
        return None

    def _status_lower(resp: Any) -> str:
        val = _extract_field(resp, "status")
        if isinstance(val, str):
            return val.lower()
        if isinstance(resp, str):
            return resp.lower()
        return ""

    def _parse_price_change(pc: Dict[str, Any]) -> Tuple[float, float, float]:
        def _to_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        price_fields = (
            "last_trade_price",
            "last_price",
            "mark_price",
            "price",
        )

        bid = _to_float(pc.get("best_bid"))
        ask = _to_float(pc.get("best_ask"))

        price_val: Optional[float] = None
        for key in price_fields:
            price_val = _to_float(pc.get(key))
            if price_val is not None:
                break

        if price_val is None:
            if bid is not None and ask is not None:
                price_val = (bid + ask) / 2.0
            elif bid is not None:
                price_val = bid
            elif ask is not None:
                price_val = ask
            else:
                price_val = 0.0

        return (
            bid or 0.0,
            ask or 0.0,
            price_val,
        )

    def _on_event(ev: Dict[str, Any]):
        nonlocal market_closed_detected
        if stop_event.is_set():
            return
        if not isinstance(ev, dict):
            return
        if _event_indicates_market_closed(ev):
            print("[MARKET] 收到市场关闭事件，准备退出…")
            market_closed_detected = True
            strategy.stop("market closed")
            stop_event.set()
            return

        if ev.get("event_type") == "price_change":
            pcs = ev.get("price_changes", [])
        elif "price_changes" in ev:
            pcs = ev.get("price_changes", [])
        else:
            return
        ts = _extract_ts(ev.get("timestamp") or ev.get("ts") or ev.get("time"))
        for pc in pcs:
            if str(pc.get("asset_id")) != str(token_id):
                continue
            bid, ask, last = _parse_price_change(pc)
            latest[token_id] = {"price": last, "best_bid": bid, "best_ask": ask}
            action = strategy.on_tick(best_ask=ask, best_bid=bid, ts=ts)
            if action:
                action_queue.put(action)
            if _is_market_closed(pc):
                print("[MARKET] 检测到市场关闭信号，准备退出…")
                market_closed_detected = True
                strategy.stop("market closed")
                stop_event.set()
                break

    def _confirm_market_closed():
        nonlocal market_closed_detected
        attempt = 0
        while not stop_event.is_set():
            refreshed_meta = _refresh_market_meta()
            attempt += 1
            now = time.time()
            if _market_has_ended(refreshed_meta, now):
                print("[MARKET] 已确认市场结束，可进行后续处理。")
                market_closed_detected = True
                strategy.stop("market ended confirmed")
                stop_event.set()
                return
            if attempt == 1:
                print("[MARKET] 倒计时结束但市场尚未标记结束，10 秒后再次检查…")
            else:
                print(
                    f"[MARKET] 第 {attempt} 次检查仍未确认结束，10 秒后再次重试…"
                )
            for _ in range(10):
                if stop_event.is_set():
                    return
                time.sleep(1)

    def _activate_sell_only(reason: str) -> None:
        if not sell_only_event.is_set():
            sell_only_event.set()
            strategy.enable_sell_only(reason)
            print("[COUNTDOWN] 已进入仅卖出模式：倒计时窗口内不再买入。")

    def _countdown_monitor():
        if not market_deadline_ts:
            return
        last_display: Optional[int] = None
        sell_only_warn_logged = False
        while not stop_event.is_set():
            now = time.time()
            if sell_only_start_ts and not sell_only_event.is_set():
                until_sell_only = sell_only_start_ts - now
                if until_sell_only <= 0:
                    _activate_sell_only("countdown window")
                elif until_sell_only <= 300 and not sell_only_warn_logged:
                    mins = int(max(until_sell_only, 0) // 60)
                    secs = int(max(until_sell_only, 0) % 60)
                    print(
                        "[COUNTDOWN] 距离仅卖出模式开启还剩 "
                        f"{mins:02d}:{secs:02d}。"
                    )
                    sell_only_warn_logged = True
            remaining = market_deadline_ts - now
            if remaining <= 0:
                if last_display != 0:
                    print("[COUNTDOWN] 距离市场结束还剩 00:00")
                print("[COUNTDOWN] 倒计时结束，开始确认市场状态…")
                _confirm_market_closed()
                return
            if remaining <= 300:
                secs_left = int(remaining)
                if secs_left != last_display:
                    mm = secs_left // 60
                    ss = secs_left % 60
                    print(
                        f"[COUNTDOWN] 距离市场结束还剩 {mm:02d}:{ss:02d}"
                    )
                    last_display = secs_left
                for _ in range(5):
                    if stop_event.is_set():
                        return
                    time.sleep(0.2)
            else:
                wait = min(remaining - 300, 60)
                if wait <= 0:
                    wait = 1
                for _ in range(int(wait)):
                    if stop_event.is_set():
                        return
                    time.sleep(1)

    ws_thread = threading.Thread(
        target=ws_watch_by_ids,
        kwargs={
            "asset_ids": [token_id],
            "label": f"{title} ({side})",
            "on_event": _on_event,
            "verbose": False,
        },
        daemon=True,
    )
    ws_thread.start()

    print("[RUN] 监听行情中… 输入 stop / exit 可手动停止。")

    if sell_only_start_ts and time.time() >= sell_only_start_ts:
        _activate_sell_only("countdown window")

    if market_deadline_ts:
        threading.Thread(target=_countdown_monitor, daemon=True).start()

    start_wait = time.time()
    while not latest.get(token_id) and not stop_event.is_set():
        if time.time() - start_wait > 5:
            print("[WAIT] 尚未收到行情，继续等待…")
            start_wait = time.time()
        time.sleep(0.2)

    if stop_event.is_set():
        print("[EXIT] 已终止。")
        return

    def _input_listener():
        while not stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd in {"stop", "exit", "quit"}:
                print("[CMD] 收到停止指令，准备退出…")
                strategy.stop("manual stop")
                stop_event.set()
                break

    threading.Thread(target=_input_listener, daemon=True).start()

    def _fmt_price(val: Optional[Any]) -> str:
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return "-"

    def _fmt_pct(val: Optional[Any]) -> str:
        try:
            return f"{float(val) * 100.0:.2f}%"
        except (TypeError, ValueError):
            return "-"

    def _fmt_minutes(seconds: Optional[Any]) -> str:
        try:
            sec = float(seconds)
        except (TypeError, ValueError):
            return "-"
        return f"{sec / 60.0:.1f}m"

    success_status = {
        "success",
        "matched",
        "filled",
        "complete",
        "completed",
        "executed",
        "accepted",
        "partial",
    }
    position_size: Optional[float] = None
    last_order_size: Optional[float] = None
    last_log = 0.0
    buy_cooldown_until: float = 0.0
    buy_cooldown_reason: Optional[str] = None
    pending_buy: Optional[Action] = None
    short_buy_cooldown = 1.0

    try:
        while not stop_event.is_set():
            now = time.time()
            if pending_buy is not None and now >= buy_cooldown_until:
                if sell_only_event.is_set():
                    print("[COUNTDOWN] 仍在仅卖出模式内，丢弃待执行的买入信号。")
                    strategy.on_reject("sell-only window active")
                    pending_buy = None
                    buy_cooldown_reason = None
                else:
                    label = "卖出" if buy_cooldown_reason == "sell" else "买入"
                    print(f"[COOLDOWN] {label}冷却结束，重新尝试买入…")
                    action_queue.put(pending_buy)
                    pending_buy = None
                    buy_cooldown_reason = None

            if now - last_log >= 1.0:
                snap = latest.get(token_id) or {}
                bid = float(snap.get("best_bid") or 0.0)
                ask = float(snap.get("best_ask") or 0.0)
                last_px = float(snap.get("price") or 0.0)
                st = strategy.status()
                awaiting = st.get("awaiting")
                awaiting_s = awaiting.value if hasattr(awaiting, "value") else awaiting
                entry_price = st.get("entry_price")
                print(
                    f"[PX] bid={bid:.4f} ask={ask:.4f} last={last_px:.4f} | "
                    f"state={st.get('state')} awaiting={awaiting_s} entry={entry_price}"
                )

                extra_lines: List[str] = []

                drop_stats = st.get("drop_stats") or {}
                config_snapshot = st.get("config") or {}
                history_len = st.get("price_history_len")
                history_display = history_len if history_len is not None else "-"
                window_seconds = drop_stats.get("window_seconds")
                extra_lines.append(
                    "    时间窗口: "
                    f"{_fmt_minutes(window_seconds)} | 采样点数: {history_display}"
                )

                drop_line = (
                    "    窗口跌幅: 当前 "
                    f"{_fmt_pct(drop_stats.get('current_drop_ratio'))} / 最大 "
                    f"{_fmt_pct(drop_stats.get('max_drop_ratio'))} / 阈值 "
                    f"{_fmt_pct(config_snapshot.get('drop_pct'))}"
                )
                extra_lines.append(drop_line)

                price_line = (
                    "    窗口价格: 高 "
                    f"{_fmt_price(drop_stats.get('window_high'))} / 低 "
                    f"{_fmt_price(drop_stats.get('window_low'))}"
                )
                extra_lines.append(price_line)

                if st.get("state") == "LONG":
                    sell_target = st.get("sell_trigger")
                    if sell_target is not None:
                        extra_lines.append(f"    目标卖出价格: {float(sell_target):.4f}")
                    else:
                        extra_lines.append("    目标卖出价格: -")

                if st.get("sell_only"):
                    extra_lines.append("    状态：倒计时仅卖出模式（禁止买入）")
                for line in extra_lines:
                    print(line)
                last_log = now

            try:
                action = action_queue.get(timeout=0.5)
            except Empty:
                continue

            if stop_event.is_set():
                break

            snap = latest.get(token_id) or {}
            bid = float(snap.get("best_bid") or 0.0)
            ask = float(snap.get("best_ask") or 0.0)

            if (
                not market_closed_detected
                and market_meta
                and _market_has_ended(market_meta, now)
            ):
                print("[MARKET] 达到市场截止时间，准备退出…")
                market_closed_detected = True
                strategy.stop("market ended")
                stop_event.set()
                continue

            if action.action == ActionType.BUY:
                if sell_only_event.is_set():
                    print("[COUNTDOWN] 当前处于倒计时仅卖出模式，忽略买入信号。")
                    strategy.on_reject("sell-only window active")
                    continue
                now_for_buy = time.time()
                if now_for_buy < buy_cooldown_until:
                    remaining = buy_cooldown_until - now_for_buy
                    if buy_cooldown_reason == "sell":
                        print(
                            f"[COOLDOWN] 最近一次卖出后尚在冷却中，剩余 {remaining:.1f}s 再尝试买入。"
                        )
                    else:
                        print(
                            f"[COOLDOWN] 买入冷却中，剩余 {remaining:.1f}s 再尝试买入。"
                        )
                    pending_buy = action
                    continue

                ref_price = action.ref_price or ask or float(snap.get("price") or 0.0)
                if size_in:
                    try:
                        order_size = float(size_in)
                    except Exception:
                        print("[ERR] 份数非法，终止。")
                        strategy.stop("invalid size")
                        stop_event.set()
                        break
                else:
                    order_size = _calc_size_by_1dollar(ref_price)
                    print(f"[HINT] 未指定份数，按 $1 反推 -> size={order_size}")

                try:
                    resp = execute_auto_buy(
                        client=client,
                        token_id=token_id,
                        price=ref_price,
                        size=order_size,
                    )
                except Exception as exc:
                    print(f"[ERR] 买入下单异常：{exc}")
                    strategy.on_reject(str(exc))
                    buy_cooldown_until = time.time() + short_buy_cooldown
                    buy_cooldown_reason = "buy"
                    continue
                print(f"[TRADE][BUY] resp={resp}")
                status = _status_lower(resp)
                status_display = status.upper() if status else "N/A"
                filled_amt = _extract_size(resp, 0.0)
                success = status in success_status or (filled_amt and filled_amt > 0)
                if success and filled_amt > 0:
                    fill_px = _extract_price(resp, ref_price)
                    position_size = float(filled_amt)
                    last_order_size = float(filled_amt)
                    strategy.on_buy_filled(avg_price=fill_px, size=position_size)
                    print(
                        f"[STATE] 买入成交 -> status={status_display} price={fill_px:.4f} size={position_size:.4f}"
                    )
                else:
                    reason_raw = _extract_field(resp, "message")
                    if reason_raw is None and isinstance(resp, dict):
                        reason_raw = resp.get("message")
                    reason_text = str(reason_raw if reason_raw is not None else resp)
                    print(f"[WARN] 买入未成交(status={status_display})：{reason_text}")
                    strategy.on_reject(reason_text)
                buy_cooldown_until = time.time() + short_buy_cooldown
                buy_cooldown_reason = "buy"

            elif action.action == ActionType.SELL:
                ref_price = action.ref_price or bid or float(snap.get("price") or 0.0)
                sell_size = position_size
                if sell_size is None:
                    try:
                        st_now = strategy.status()
                        ps = st_now.get("position_size")
                        sell_size = float(ps) if ps is not None else None
                    except Exception:
                        sell_size = None
                if sell_size is None or sell_size <= 0:
                    sell_size = last_order_size
                if sell_size is None or sell_size <= 0:
                    print("[WARN] 无可用持仓，忽略卖出信号。")
                    strategy.on_reject("empty position")
                    continue

                try:
                    resp = execute_auto_sell(
                        client=client,
                        token_id=token_id,
                        price=ref_price,
                        size=sell_size,
                    )
                except Exception as exc:
                    print(f"[ERR] 卖出下单异常：{exc}")
                    strategy.on_reject(str(exc))
                    continue
                print(f"[TRADE][SELL] resp={resp}")
                status = _status_lower(resp)
                filled_amt = max(_extract_size(resp, 0.0), 0.0)
                remaining_amt = _extract_remaining(resp)
                base_remaining = None
                if remaining_amt is None:
                    base_remaining = sell_size if sell_size is not None else position_size
                    if base_remaining is None:
                        requested_val = _extract_field(resp, "requested")
                        if requested_val is not None:
                            try:
                                base_remaining = float(requested_val)
                            except (TypeError, ValueError):
                                base_remaining = None
                    if base_remaining is not None:
                        remaining_amt = max(float(base_remaining) - filled_amt, 0.0)

                eps = 1e-4
                if remaining_amt is not None and remaining_amt <= eps:
                    remaining_amt = 0.0

                sold_success = status in success_status or filled_amt > eps
                if sold_success:
                    fill_px = _extract_price(resp, ref_price)
                    strategy.on_sell_filled(
                        avg_price=fill_px if filled_amt > 0 else None,
                        size=filled_amt if filled_amt > 0 else None,
                        remaining=remaining_amt,
                    )
                    if remaining_amt and remaining_amt > eps:
                        position_size = remaining_amt
                        last_order_size = remaining_amt
                        print(
                            "[STATE] 卖出部分成交 -> "
                            f"price={fill_px:.4f} sold={filled_amt:.4f} remaining={remaining_amt:.4f}"
                        )
                    else:
                        position_size = None
                        last_order_size = None
                        buy_cooldown_until = time.time() + 15.0
                        buy_cooldown_reason = "sell"
                        sold_display = filled_amt if filled_amt > 0 else (sell_size or 0.0)
                        print(
                            "[STATE] 卖出成交 -> "
                            f"price={fill_px:.4f} size={sold_display:.4f}"
                        )
                else:
                    reason = resp.get("message") if isinstance(resp, dict) else str(resp)
                    print(f"[WARN] 卖出未成交：{reason}")
                    strategy.on_reject(reason if isinstance(reason, str) else None)

    except KeyboardInterrupt:
        print("[CMD] 捕获到 Ctrl+C，准备退出…")
        strategy.stop("keyboard interrupt")
        stop_event.set()

    finally:
        stop_event.set()
        final_status = strategy.status()
        print(f"[EXIT] 最终状态: {final_status}")
        try:
            if _should_attempt_claim(market_meta, final_status, market_closed_detected):
                _attempt_claim(client, market_meta, token_id)
            else:
                print("[CLAIM] 未检测到需要 claim 的仓位，脚本结束。")
        except Exception as claim_exc:
            print(f"[CLAIM] 自动 claim 过程出现异常: {claim_exc}")


if __name__ == "__main__":
    main()