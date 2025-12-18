from __future__ import annotations

import datetime as dt
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from .models import Trade

MAX_BACKOFF_SECONDS = float(os.environ.get("SMART_QUERY_MAX_BACKOFF", "60"))
MAX_REQUESTS_PER_SECOND = float(os.environ.get("SMART_QUERY_MAX_RPS", "2"))
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND if MAX_REQUESTS_PER_SECOND > 0 else 0.0


def _respect_rate_limit() -> None:
    """简单的全局限速，复用参考代码的节奏控制。"""

    if MIN_REQUEST_INTERVAL <= 0:
        return

    now = time.monotonic()
    elapsed = now - _respect_rate_limit._last_request_time  # type: ignore[attr-defined]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    _respect_rate_limit._last_request_time = time.monotonic()  # type: ignore[attr-defined]


_respect_rate_limit._last_request_time = 0.0  # type: ignore[attr-defined]


def _request_with_backoff(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 2.0,
    max_backoff: float = MAX_BACKOFF_SECONDS,
):
    """复用原版脚本的指数回退 + 抖动请求封装。"""

    attempt = 1
    while True:
        try:
            _respect_rate_limit()
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = None
            try:
                if exc.response is not None:
                    body = exc.response.text
            except Exception:
                pass

            if status is not None and 400 <= status < 500 and status != 429:
                extra = f" | body={body[:200]}" if body else ""
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {exc}{extra}",
                )
                return None

            if attempt >= retries:
                extra = f" | body={body[:200]}" if body else ""
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {exc}{extra}",
                )
                return None

            wait = min(max_backoff, backoff * (2 ** (attempt - 1)))
            jitter = min(wait * 0.1, 1.0)
            time.sleep(wait + random.random() * jitter)
            attempt += 1
        except requests.RequestException as exc:
            if attempt >= retries:
                print(
                    f"[WARN] 请求失败（{attempt}/{retries}）：{url} params={params} -> {exc}",
                )
                return None

            wait = min(max_backoff, backoff * (2 ** (attempt - 1)))
            jitter = min(wait * 0.1, 1.0)
            time.sleep(wait + random.random() * jitter)
            attempt += 1


class DataApiClient:
    """Polymarket Data-API 客户端，覆盖 leaderboard 与 trades。"""

    def __init__(self, host: str = "https://data-api.polymarket.com") -> None:
        self.host = host.rstrip("/")

    def fetch_leaderboard(
        self,
        *,
        period: str = "ALL",
        order_by: str = "vol",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        url = f"{self.host}/v1/leaderboard"
        params = {
            "timePeriod": period.upper(),
            "orderBy": order_by.upper(),
            "limit": limit,
            "offset": offset,
        }
        resp = _request_with_backoff(url, params=params)
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
            resp = _request_with_backoff(url, params=params)
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
