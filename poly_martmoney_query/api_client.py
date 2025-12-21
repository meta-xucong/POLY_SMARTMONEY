from __future__ import annotations

import datetime as dt
import email.utils
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from .models import ClosedPosition, Position, Trade, TradeAction

MAX_BACKOFF_SECONDS = float(os.environ.get("SMART_QUERY_MAX_BACKOFF", "60"))
MAX_REQUESTS_PER_SECOND = float(os.environ.get("SMART_QUERY_MAX_RPS", "2"))
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND if MAX_REQUESTS_PER_SECOND > 0 else 0.0
BASE_PAGE_SLEEP = float(os.environ.get("SMART_QUERY_BASE_SLEEP", "0.3"))


class RateLimiter:
    """简单的全局限速器，用于跨用户共享节流。"""

    def __init__(self, rps: float) -> None:
        self.rps = max(rps, 0.1)
        self.min_interval = 1.0 / self.rps
        self._next_ts = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_ts:
            time.sleep(self._next_ts - now)
        self._next_ts = max(self._next_ts, now) + self.min_interval


_GLOBAL_LIMITER = RateLimiter(MAX_REQUESTS_PER_SECOND if MIN_REQUEST_INTERVAL > 0 else 1000.0)


def _sleep_with_jitter(wait: float, jitter_ratio: float = 0.1) -> None:
    if wait <= 0:
        return
    jitter = min(wait * jitter_ratio, 1.0)
    time.sleep(wait + random.random() * jitter)


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    if not header_value:
        return None
    try:
        return float(header_value)
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return max(parsed.timestamp() - dt.datetime.now(tz=dt.timezone.utc).timestamp(), 0.0)


def _request_with_backoff(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 2.0,
    max_backoff: float = MAX_BACKOFF_SECONDS,
    session: Optional[requests.Session] = None,
    limiter: Optional[RateLimiter] = None,
) -> tuple[Optional[requests.Response], Optional[str]]:
    """复用原版脚本的指数回退 + 抖动请求封装。"""

    attempt = 1
    client = session or requests
    limiter = limiter or _GLOBAL_LIMITER
    while True:
        try:
            limiter.wait()
            resp = client.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp, None
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = None
            try:
                if exc.response is not None:
                    body = exc.response.text
            except Exception:
                pass

            retryable_statuses = {408, 429}
            if status is not None and 500 <= status < 600:
                retryable_statuses.add(status)

            if status is not None and 400 <= status < 500 and status not in retryable_statuses:
                extra = f" | body={body[:200]}" if body else ""
                msg = f"HTTP {status} {exc}{extra}"
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {msg}",
                )
                return None, msg

            if attempt >= retries:
                extra = f" | body={body[:200]}" if body else ""
                msg = f"HTTP {status} {exc}{extra}"
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {msg}",
                )
                return None, msg

            retry_after = None
            if status == 429 and exc.response is not None:
                retry_after = _parse_retry_after(exc.response.headers.get("Retry-After"))
            wait = (
                max(retry_after, backoff * (2 ** (attempt - 1)))
                if retry_after is not None
                else min(max_backoff, backoff * (2 ** (attempt - 1)))
            )
            _sleep_with_jitter(wait)
            attempt += 1
        except requests.RequestException as exc:
            if attempt >= retries:
                msg = str(exc)
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {msg}",
                )
                return None, msg

            wait = min(max_backoff, backoff * (2 ** (attempt - 1)))
            _sleep_with_jitter(wait)
            attempt += 1


def _to_timestamp(value: Optional[dt.datetime]) -> Optional[float]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def _shift_before(timestamp: dt.datetime) -> dt.datetime:
    return timestamp - dt.timedelta(microseconds=1)


class DataApiClient:
    """Polymarket Data-API 客户端，覆盖 leaderboard 与 trades。"""

    def __init__(
        self,
        host: str = "https://data-api.polymarket.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.session = session or requests.Session()

    def fetch_leaderboard(
        self,
        *,
        period: str = "ALL",
        order_by: str = "vol",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        url = f"{self.host}/v1/leaderboard"
        period_key = period.upper()
        period_aliases = {
            "MONTHLY": "MONTH",
            "WEEKLY": "WEEK",
            "DAILY": "DAY",
        }
        order_key = order_by.strip().lower()
        order_aliases = {
            "profit": "PNL",
            "pnl": "PNL",
            "vol": "VOL",
            "volume": "VOL",
        }
        limit = max(1, min(limit, 50))
        params = {
            "timePeriod": period_aliases.get(period_key, period_key),
            "orderBy": order_aliases.get(order_key, order_by.strip().upper()),
            "limit": limit,
            "offset": offset,
        }
        resp, _ = _request_with_backoff(url, params=params, session=self.session)
        if resp is None:
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            payload = data.get("data") or data.get("leaderboard")
            return payload if isinstance(payload, list) else []
        return []

    def iter_leaderboard(
        self,
        *,
        period: str = "ALL",
        order_by: str = "vol",
        page_size: int = 100,
        max_pages: Optional[int] = 20,
    ) -> Iterable[Dict[str, Any]]:
        page = 0
        offset = 0
        page_size = max(1, min(page_size, 50))
        while True:
            batch = self.fetch_leaderboard(
                period=period, order_by=order_by, limit=page_size, offset=offset
            )
            if not batch:
                break
            for item in batch:
                yield item
            offset += len(batch)
            page += 1
            if max_pages is not None and page >= max_pages:
                break
            if len(batch) < page_size:
                break

    def fetch_trades(
        self,
        user: str,
        *,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        page_size: int = 100,
        max_pages: Optional[int] = 200,
        taker_only: bool = False,
    ) -> List[Trade]:
        url = f"{self.host}/trades"
        offset = 0
        page = 0
        results: List[Trade] = []
        start_ts = start_time.timestamp() if start_time else None
        end_ts = end_time.timestamp() if end_time else None

        while True:
            params = {
                "user": user,
                "limit": page_size,
                "offset": offset,
                "takerOnly": taker_only,
            }
            resp, _ = _request_with_backoff(url, params=params, session=self.session)
            if resp is None:
                break

            try:
                payload = resp.json()
            except Exception:
                break

            raw_trades = []
            if isinstance(payload, list):
                raw_trades = payload
            elif isinstance(payload, dict):
                raw_trades = payload.get("data") or payload.get("trades") or []
            if not isinstance(raw_trades, list):
                break

            parsed_batch: List[Trade] = []
            reached_earliest = False
            for item in raw_trades:
                trade = Trade.from_api(item)
                if trade is None:
                    continue
                ts = trade.timestamp.timestamp()
                if end_ts is not None and ts > end_ts:
                    continue
                if start_ts is not None and ts < start_ts:
                    reached_earliest = True
                    continue
                parsed_batch.append(trade)

            results.extend(parsed_batch)

            if reached_earliest or len(raw_trades) < page_size:
                break

            offset += page_size
            page += 1
            if max_pages is not None and page >= max_pages:
                break

        results.sort(key=lambda t: t.timestamp)
        return results

    def fetch_trade_actions_window(
        self,
        user: str,
        *,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        taker_only: bool = False,
        return_info: bool = False,
    ) -> List[TradeAction] | tuple[List[TradeAction], Dict[str, object]]:
        url = f"{self.host}/trades"
        start_ts = _to_timestamp(start_time)
        end_ts = _to_timestamp(end_time)
        cursor_end = end_time
        actions: Dict[str, dt.datetime] = {}
        ok = True
        incomplete = False
        hit_max_pages = False
        last_error: Optional[str] = None
        pages_fetched = 0

        while True:
            params = {
                "user": user,
                "limit": page_size,
                "offset": 0,
                "takerOnly": taker_only,
            }
            if start_ts is not None:
                params["start"] = start_ts
            if cursor_end is not None:
                params["end"] = _to_timestamp(cursor_end)

            resp, error_msg = _request_with_backoff(url, params=params, session=self.session)
            if resp is None:
                error_msg = error_msg or "request_failed"
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, error_msg)
                break

            try:
                payload = resp.json()
            except Exception:
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_json")
                break

            raw_trades = []
            if isinstance(payload, list):
                raw_trades = payload
            elif isinstance(payload, dict):
                raw_trades = payload.get("data") or payload.get("trades") or []
            if not isinstance(raw_trades, list):
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_payload")
                break

            if not raw_trades:
                break

            reached_earliest = False
            min_ts: Optional[dt.datetime] = None
            for item in raw_trades:
                trade = Trade.from_api(item)
                if trade is None:
                    continue
                ts = trade.timestamp
                if end_ts is not None and ts.timestamp() > end_ts:
                    continue
                if start_ts is not None and ts.timestamp() < start_ts:
                    reached_earliest = True
                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    continue
                existing = actions.get(trade.tx_hash)
                if existing is None or ts < existing:
                    actions[trade.tx_hash] = ts
                min_ts = ts if min_ts is None else min(min_ts, ts)

            pages_fetched += 1
            if reached_earliest or min_ts is None:
                break

            cursor_end = _shift_before(min_ts)
            if max_pages is not None and pages_fetched >= max_pages:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, f"hit_max_pages={max_pages}")
                print(
                    f"[WARN] trades 分页被截断：user={user} hit max_pages={max_pages}",
                    flush=True,
                )
                break

            _sleep_with_jitter(BASE_PAGE_SLEEP)

        action_list = [
            TradeAction(tx_hash=tx_hash, timestamp=timestamp)
            for tx_hash, timestamp in actions.items()
        ]
        action_list.sort(key=lambda t: t.timestamp)
        info = {
            "ok": ok,
            "incomplete": incomplete,
            "error_msg": last_error,
            "hit_max_pages": hit_max_pages,
            "pages_fetched": pages_fetched,
        }

        return (action_list, info) if return_info else action_list

    def fetch_positions(
        self,
        user: str,
        *,
        size_threshold: float = 0.0,
        page_size: int = 200,
        max_pages: Optional[int] = 50,
        sort_by: str = "TOKENS",
        sort_dir: str = "DESC",
        return_info: bool = False,
    ) -> List[Position] | tuple[List[Position], Dict[str, object]]:
        url = f"{self.host}/positions"
        offset = 0
        page = 0
        results: List[Position] = []
        ok = True
        incomplete = False
        hit_max_pages = False
        last_error: Optional[str] = None
        pages_fetched = 0

        while True:
            params = {
                "user": user,
                "limit": page_size,
                "offset": offset,
                "sizeThreshold": size_threshold,
                "sortBy": sort_by,
                "sortDirection": sort_dir,
            }
            resp, error_msg = _request_with_backoff(url, params=params, session=self.session)
            if resp is None:
                error_msg = error_msg or "request_failed"
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, error_msg)
                break

            try:
                payload = resp.json()
            except Exception:
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_json")
                break

            raw_positions = []
            if isinstance(payload, list):
                raw_positions = payload
            elif isinstance(payload, dict):
                raw_positions = payload.get("data") or payload.get("positions") or []
            if not isinstance(raw_positions, list):
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_payload")
                break

            for item in raw_positions:
                position = Position.from_api(item, user=user)
                if position is None:
                    continue
                results.append(position)

            pages_fetched += 1
            if len(raw_positions) < page_size:
                break

            offset += page_size
            page += 1
            if max_pages is not None and page >= max_pages:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, f"hit_max_pages={max_pages}")
                print(
                    f"[WARN] positions 分页被截断：user={user} hit max_pages={max_pages}",
                    flush=True,
                )
                break

        info = {
            "ok": ok,
            "incomplete": incomplete,
            "error_msg": last_error,
            "hit_max_pages": hit_max_pages,
            "pages_fetched": pages_fetched,
        }

        return (results, info) if return_info else results

    def fetch_closed_positions(
        self,
        user: str,
        *,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        page_size: int = 100,
        max_pages: Optional[int] = 2000,
        sort_by: str = "TIMESTAMP",
        sort_dir: str = "DESC",
        return_info: bool = False,
    ) -> List[ClosedPosition] | tuple[List[ClosedPosition], Dict[str, object]]:
        url = f"{self.host}/closed-positions"
        offset = 0
        page = 0
        results: List[ClosedPosition] = []
        start_ts = start_time.timestamp() if start_time else None
        end_ts = end_time.timestamp() if end_time else None
        ok = True
        incomplete = False
        hit_max_pages = False
        last_error: Optional[str] = None
        pages_fetched = 0

        while True:
            params = {
                "user": user,
                "limit": page_size,
                "offset": offset,
                "sortBy": sort_by,
                "sortDirection": sort_dir,
            }
            resp, error_msg = _request_with_backoff(url, params=params, session=self.session)
            if resp is None:
                error_msg = error_msg or "request_failed"
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, error_msg)
                break

            try:
                payload = resp.json()
            except Exception:
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_json")
                break

            raw_positions = []
            if isinstance(payload, list):
                raw_positions = payload
            elif isinstance(payload, dict):
                raw_positions = payload.get("data") or payload.get("positions") or []
            if not isinstance(raw_positions, list):
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_payload")
                break

            reached_earliest = False
            for item in raw_positions:
                position = ClosedPosition.from_api(item, user=user)
                if position is None:
                    continue
                ts = position.timestamp.timestamp()
                if end_ts is not None and ts > end_ts:
                    continue
                if start_ts is not None and ts < start_ts:
                    reached_earliest = True
                    continue
                results.append(position)

            pages_fetched += 1
            if reached_earliest or len(raw_positions) < page_size:
                break

            offset += page_size
            page += 1
            if max_pages is not None and page >= max_pages:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, f"hit_max_pages={max_pages}")
                print(
                    f"[WARN] closed-positions 分页被截断：user={user} hit max_pages={max_pages}",
                    flush=True,
                )
                break

        results.sort(key=lambda p: p.timestamp)
        info = {
            "ok": ok,
            "incomplete": incomplete,
            "error_msg": last_error,
            "hit_max_pages": hit_max_pages,
            "pages_fetched": pages_fetched,
        }

        return (results, info) if return_info else results

    def fetch_closed_positions_window(
        self,
        user: str,
        *,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        page_size: int = 50,
        max_pages: Optional[int] = None,
        sort_by: str = "TIMESTAMP",
        sort_dir: str = "DESC",
        return_info: bool = False,
    ) -> List[ClosedPosition] | tuple[List[ClosedPosition], Dict[str, object]]:
        url = f"{self.host}/closed-positions"
        start_ts = _to_timestamp(start_time)
        end_ts = _to_timestamp(end_time)
        cursor_end = end_time
        results: List[ClosedPosition] = []
        ok = True
        incomplete = False
        hit_max_pages = False
        last_error: Optional[str] = None
        pages_fetched = 0

        while True:
            params = {
                "user": user,
                "limit": page_size,
                "offset": 0,
                "sortBy": sort_by,
                "sortDirection": sort_dir,
            }
            if start_ts is not None:
                params["start"] = start_ts
            if cursor_end is not None:
                params["end"] = _to_timestamp(cursor_end)

            resp, error_msg = _request_with_backoff(url, params=params, session=self.session)
            if resp is None:
                error_msg = error_msg or "request_failed"
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, error_msg)
                break

            try:
                payload = resp.json()
            except Exception:
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_json")
                break

            raw_positions = []
            if isinstance(payload, list):
                raw_positions = payload
            elif isinstance(payload, dict):
                raw_positions = payload.get("data") or payload.get("positions") or []
            if not isinstance(raw_positions, list):
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, "invalid_payload")
                break

            if not raw_positions:
                break

            reached_earliest = False
            min_ts: Optional[dt.datetime] = None
            for item in raw_positions:
                position = ClosedPosition.from_api(item, user=user)
                if position is None:
                    continue
                ts = position.timestamp
                if end_ts is not None and ts.timestamp() > end_ts:
                    continue
                if start_ts is not None and ts.timestamp() < start_ts:
                    reached_earliest = True
                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    continue
                results.append(position)
                min_ts = ts if min_ts is None else min(min_ts, ts)

            pages_fetched += 1
            if reached_earliest or min_ts is None:
                break

            cursor_end = _shift_before(min_ts)
            if max_pages is not None and pages_fetched >= max_pages:
                hit_max_pages = True
                incomplete = True
                ok = False
                last_error = _combine_error(last_error, f"hit_max_pages={max_pages}")
                print(
                    f"[WARN] closed-positions 分页被截断：user={user} hit max_pages={max_pages}",
                    flush=True,
                )
                break

            _sleep_with_jitter(BASE_PAGE_SLEEP)

        results.sort(key=lambda p: p.timestamp)
        info = {
            "ok": ok,
            "incomplete": incomplete,
            "error_msg": last_error,
            "hit_max_pages": hit_max_pages,
            "pages_fetched": pages_fetched,
        }

        return (results, info) if return_info else results


def _combine_error(existing: Optional[str], new_msg: Optional[str]) -> Optional[str]:
    if not new_msg:
        return existing
    if not existing:
        return new_msg
    if new_msg in existing:
        return existing
    return f"{existing}; {new_msg}"
