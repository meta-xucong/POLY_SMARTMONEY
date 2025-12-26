from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple

import requests

from smartmoney_query.poly_martmoney_query.api_client import DataApiClient
from smartmoney_query.poly_martmoney_query.models import Position


def _normalize_position(pos: Position) -> Dict[str, object] | None:
    if pos.outcome_index is None:
        return None
    token_key = f"{pos.condition_id}:{pos.outcome_index}"
    end_date = pos.end_date.isoformat() if pos.end_date is not None else None
    return {
        "token_key": token_key,
        "condition_id": pos.condition_id,
        "outcome_index": int(pos.outcome_index),
        "size": float(pos.size),
        "avg_price": float(pos.avg_price),
        "slug": pos.slug,
        "title": pos.title,
        "end_date": end_date,
        "raw": pos.raw,
    }


def fetch_positions_norm(
    client: DataApiClient,
    user: str,
    size_threshold: float,
    positions_limit: int = 500,
    positions_max_pages: int = 20,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    positions, info = fetch_positions_all(
        client,
        user,
        size_threshold,
        positions_limit=positions_limit,
        positions_max_pages=positions_max_pages,
    )
    normalized: List[Dict[str, object]] = []
    for pos in positions:
        normalized_pos = _normalize_position(pos)
        if normalized_pos is None:
            continue
        normalized.append(normalized_pos)
    info.setdefault("limit", positions_limit)
    info.setdefault("max_pages", positions_max_pages)
    info.setdefault("total", len(positions))
    return normalized, info


def fetch_positions_all(
    client: DataApiClient,
    user: str,
    size_threshold: float,
    *,
    positions_limit: int = 500,
    positions_max_pages: int = 20,
) -> Tuple[List[Position], Dict[str, object]]:
    positions, info = client.fetch_positions(
        user,
        size_threshold=size_threshold,
        page_size=positions_limit,
        max_pages=positions_max_pages,
        return_info=True,
    )
    info.setdefault("limit", positions_limit)
    info.setdefault("max_pages", positions_max_pages)
    info.setdefault("total", len(positions))
    return positions, info


def _parse_timestamp(value: object) -> dt.datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value > 1e12:
                value /= 1000.0
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = dt.datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
    except Exception:
        return None
    return None


def _normalize_action(raw: Dict[str, object]) -> Dict[str, object] | None:
    side = str(raw.get("side") or raw.get("action") or raw.get("type") or "").upper()
    event_type = str(
        raw.get("eventType")
        or raw.get("event_type")
        or raw.get("activityType")
        or raw.get("activity_type")
        or raw.get("type")
        or ""
    ).upper()
    if side not in ("BUY", "SELL"):
        return None
    if event_type and event_type not in ("TRADE", "FILL", "BUY", "SELL"):
        return None

    size = raw.get("size") or raw.get("amount") or raw.get("quantity") or raw.get("fillSize")
    try:
        size_val = float(size or 0.0)
    except Exception:
        return None
    if size_val <= 0:
        return None

    token_id = (
        raw.get("tokenId")
        or raw.get("token_id")
        or raw.get("clobTokenId")
        or raw.get("clob_token_id")
        or raw.get("assetId")
        or raw.get("asset_id")
        or raw.get("outcomeTokenId")
        or raw.get("outcome_token_id")
    )
    token_id_text = str(token_id).strip() if token_id is not None else ""
    condition_id = raw.get("conditionId") or raw.get("condition_id") or raw.get("marketId")
    outcome_index = raw.get("outcomeIndex") or raw.get("outcome_index")
    token_key = None
    if condition_id is not None and outcome_index is not None:
        try:
            token_key = f"{condition_id}:{int(outcome_index)}"
        except Exception:
            token_key = None

    ts = _parse_timestamp(raw.get("timestamp") or raw.get("time") or raw.get("createdAt"))
    if ts is None:
        return None

    return {
        "token_id": token_id_text or None,
        "token_key": token_key,
        "condition_id": str(condition_id) if condition_id is not None else None,
        "outcome_index": int(outcome_index) if outcome_index is not None else None,
        "side": side,
        "size": size_val,
        "timestamp": ts,
        "raw": raw,
    }


def fetch_target_actions_since(
    client: DataApiClient,
    user: str,
    since_ms: int,
    *,
    page_size: int = 300,
    max_offset: int = 10000,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    start_time = dt.datetime.fromtimestamp(since_ms / 1000.0, tz=dt.timezone.utc)
    end_time = dt.datetime.now(tz=dt.timezone.utc)
    records, info = _fetch_activity_actions(
        client,
        user,
        start_time=start_time,
        end_time=end_time,
        page_size=page_size,
        max_offset=max_offset,
    )

    normalized: List[Dict[str, object]] = []
    latest_ms = 0
    for raw in records:
        action = _normalize_action(raw)
        if action is None:
            continue
        action_ms = int(action["timestamp"].timestamp() * 1000)
        if action_ms <= since_ms:
            continue
        latest_ms = max(latest_ms, action_ms)
        normalized.append(action)

    incomplete = bool(info.get("incomplete"))
    total_records = int(info.get("total") or len(records))
    maxed_offset = bool(info.get("max_offset_reached") or info.get("reached_max_offset"))
    if not incomplete and (maxed_offset or total_records >= max_offset):
        incomplete = True

    info.setdefault("ok", True)
    info.setdefault("limit", page_size)
    info.setdefault("total", total_records)
    info.setdefault("normalized", len(normalized))
    info["latest_ms"] = latest_ms
    info["incomplete"] = incomplete
    return normalized, info


def _fetch_activity_actions(
    client: DataApiClient,
    user: str,
    *,
    start_time: dt.datetime,
    end_time: dt.datetime,
    page_size: int,
    max_offset: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    fetcher = getattr(client, "fetch_activity_actions", None)
    if callable(fetcher):
        return fetcher(
            user,
            start_time=start_time,
            end_time=end_time,
            page_size=page_size,
            max_offset=max_offset,
            return_info=True,
        )
    return _fetch_activity_actions_fallback(
        client,
        user,
        start_time=start_time,
        end_time=end_time,
        page_size=page_size,
        max_offset=max_offset,
    )


def _fetch_activity_actions_fallback(
    client: DataApiClient,
    user: str,
    *,
    start_time: dt.datetime,
    end_time: dt.datetime,
    page_size: int,
    max_offset: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    host = getattr(client, "host", "https://data-api.polymarket.com").rstrip("/")
    session = getattr(client, "session", None) or requests.Session()
    url = f"{host}/activity"
    page_size = max(1, min(int(page_size), 500))
    max_offset = max(0, min(int(max_offset), 10000))
    start_ts_sec = int(start_time.timestamp())
    end_ts_sec = int(end_time.timestamp())
    ok = True
    incomplete = False
    hit_max_pages = False
    last_error = None
    pages_fetched = 0
    cursor_end_sec = end_ts_sec or int(dt.datetime.now(tz=dt.timezone.utc).timestamp())
    offset = 0
    results: List[Dict[str, object]] = []

    while True:
        params = {
            "user": user,
            "type": "TRADE",
            "limit": page_size,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
            "end": int(cursor_end_sec),
            "start": int(start_ts_sec),
        }
        try:
            resp = session.get(url, params=params, timeout=15.0)
            resp.raise_for_status()
        except requests.RequestException as exc:
            incomplete = True
            ok = False
            last_error = str(exc)
            break

        try:
            payload = resp.json()
        except Exception:
            incomplete = True
            ok = False
            last_error = "invalid_json"
            break

        raw_items = []
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("data") or payload.get("activity") or []
        if not isinstance(raw_items, list):
            incomplete = True
            ok = False
            last_error = "invalid_payload"
            break
        if not raw_items:
            break

        min_ts_sec = None
        for item in raw_items:
            ts = _parse_timestamp(item.get("timestamp") or item.get("time") or item.get("createdAt"))
            if ts is None:
                continue
            ts_sec = int(ts.timestamp())
            if ts_sec < start_ts_sec or ts_sec > end_ts_sec:
                continue
            results.append(item)
            if min_ts_sec is None or ts_sec < min_ts_sec:
                min_ts_sec = ts_sec

        pages_fetched += 1
        if len(raw_items) < page_size:
            break

        offset += len(raw_items)
        if offset >= max_offset:
            if min_ts_sec is None:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = "missing_timestamps_for_window_shift"
                break
            if min_ts_sec >= cursor_end_sec:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = f"hit_max_offset_same_end={max_offset}"
                break
            cursor_end_sec = int(min_ts_sec)
            offset = 0

        if cursor_end_sec < start_ts_sec:
            break

    info = {
        "ok": ok,
        "incomplete": incomplete,
        "error_msg": last_error,
        "hit_max_pages": hit_max_pages,
        "pages_fetched": pages_fetched,
        "cursor_end_sec": cursor_end_sec,
        "total": len(results),
        "limit": page_size,
        "source": "fallback_activity_http",
    }
    return results, info
