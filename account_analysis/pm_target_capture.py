from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from smartmoney_query.poly_martmoney_query.api_client import (
        RateLimiter,
        _parse_timestamp,
        _request_with_backoff,
    )
except Exception:  # pragma: no cover - fallback for direct execution
    RateLimiter = None
    _parse_timestamp = None
    _request_with_backoff = None

DATA_API_ROOT = os.getenv("POLY_DATA_API_ROOT", "https://data-api.polymarket.com")
GAMMA_ROOT = os.getenv("POLY_GAMMA_ROOT", "https://gamma-api.polymarket.com")

_DEFAULT_LIMITER = RateLimiter(2.0) if RateLimiter else None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _to_ts(value: dt.datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return int(value.timestamp())


def _jsonl_dump(path: Path, rows: Iterable[Dict[str, Any]], mode: str = "a") -> int:
    count = 0
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    if _request_with_backoff:
        resp, _ = _request_with_backoff(
            url, params=params, session=None, limiter=_DEFAULT_LIMITER
        )
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception:
            return None
    try:
        resp = requests.get(url, params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def resolve_target(target: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Resolve @handle to proxyWallet using gamma public-search."""
    target = target.strip()
    if not target:
        raise ValueError("target 不能为空")
    if target.lower().startswith("0x") and len(target) >= 8:
        profile = _request_json(
            f"{GAMMA_ROOT}/public-profile", params={"address": target}
        )
        return target, profile if isinstance(profile, dict) else None

    handle = target[1:] if target.startswith("@") else target
    data = _request_json(
        f"{GAMMA_ROOT}/public-search",
        params={
            "q": handle,
            "search_profiles": "true",
            "limit_per_type": 10,
        },
    )
    profiles: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        raw_profiles = data.get("profiles") or data.get("data") or []
        if isinstance(raw_profiles, list):
            profiles = [p for p in raw_profiles if isinstance(p, dict)]

    handle_lower = handle.lower()
    exact = [p for p in profiles if str(p.get("name", "")).lower() == handle_lower]
    match = exact or [p for p in profiles if handle_lower in str(p.get("name", "")).lower()]
    if not match:
        raise ValueError(f"未找到 handle={target} 的 profile")

    profile = match[0]
    proxy_wallet = str(profile.get("proxyWallet") or profile.get("proxy_wallet") or "").strip()
    if not proxy_wallet:
        raise ValueError(f"handle={target} 返回数据缺少 proxyWallet")
    return proxy_wallet, profile


def _iter_time_windows(
    start: dt.datetime, end: dt.datetime, window_hours: int
) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    cursor = start
    delta = dt.timedelta(hours=window_hours)
    while cursor < end:
        window_end = min(cursor + delta, end)
        yield cursor, window_end
        cursor = window_end


def fetch_activity(
    proxy_wallet: str,
    start: dt.datetime,
    end: dt.datetime,
    out_dir: Path,
    window_hours: int = 6,
    page_size: int = 500,
) -> List[Path]:
    """Fetch activity in windowed slices and dump to daily JSONL files."""
    _ensure_dir(out_dir)
    raw_dir = out_dir / "raw"
    _ensure_dir(raw_dir)

    day_files: Dict[str, Path] = {}
    created: List[Path] = []

    for window_start, window_end in _iter_time_windows(start, end, window_hours):
        day_key = window_start.strftime("%Y%m%d")
        file_path = day_files.get(day_key)
        if file_path is None:
            file_path = raw_dir / f"activity_{day_key}.jsonl"
            day_files[day_key] = file_path
            created.append(file_path)

        offset = 0
        while True:
            params = {
                "user": proxy_wallet,
                "type": "TRADE",
                "start": _to_ts(window_start),
                "end": _to_ts(window_end),
                "limit": min(page_size, 500),
                "offset": offset,
            }
            payload = _request_json(f"{DATA_API_ROOT}/activity", params=params)
            rows: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = payload.get("data") or payload.get("activity") or []
            if not isinstance(rows, list) or not rows:
                break

            _jsonl_dump(file_path, rows, mode="a")
            offset += len(rows)
            if len(rows) < params["limit"]:
                break
    return created


def fetch_trades(
    proxy_wallet: str,
    taker_only: bool,
    out_file: Path,
    page_size: int = 500,
    max_pages: int = 200,
) -> int:
    _ensure_dir(out_file.parent)
    offset = 0
    page = 0
    total = 0
    with out_file.open("w", encoding="utf-8") as handle:
        while True:
            params = {
                "user": proxy_wallet,
                "limit": min(page_size, 10000),
                "offset": offset,
                "takerOnly": "true" if taker_only else "false",
            }
            payload = _request_json(f"{DATA_API_ROOT}/trades", params=params)
            rows: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = payload.get("data") or payload.get("trades") or []
            if not isinstance(rows, list) or not rows:
                break

            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            offset += len(rows)
            page += 1
            if max_pages is not None and page >= max_pages:
                break
            if len(rows) < params["limit"]:
                break
    return total


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_key_value(value: Optional[float], decimals: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def _trade_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    tx_hash = str(row.get("transactionHash") or row.get("txHash") or row.get("tx_hash") or "").strip()
    asset = str(row.get("asset") or row.get("assetId") or row.get("tokenId") or "").strip()
    side = str(row.get("side") or row.get("action") or "").strip().upper()
    size = _normalize_key_value(_coerce_float(row.get("size") or row.get("amount")))
    price = _normalize_key_value(_coerce_float(row.get("price") or row.get("rate")))
    ts_value = row.get("timestamp") or row.get("time") or row.get("createdAt")
    ts = _parse_timestamp(ts_value) if _parse_timestamp else None
    ts_key = ""
    if ts is not None:
        ts_key = str(int(ts.timestamp()))
    elif isinstance(ts_value, (int, float)):
        ts_key = str(int(ts_value))
    else:
        ts_key = str(ts_value or "").strip()
    return tx_hash, asset, side, size, price, ts_key


def _extract_trade_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = row.get("timestamp") or row.get("time") or row.get("createdAt")
    if _parse_timestamp:
        parsed_ts = _parse_timestamp(timestamp)
        timestamp = parsed_ts.isoformat() if parsed_ts else timestamp
    return {
        "conditionId": row.get("conditionId") or row.get("condition_id"),
        "asset": row.get("asset") or row.get("assetId") or row.get("tokenId"),
        "outcomeIndex": row.get("outcomeIndex") or row.get("outcome_index"),
        "side": row.get("side") or row.get("action"),
        "price": _coerce_float(row.get("price") or row.get("rate")),
        "size": _coerce_float(row.get("size") or row.get("amount")),
        "usdcSize": _coerce_float(row.get("usdcSize") or row.get("usdc_size")),
        "timestamp": timestamp,
        "transactionHash": row.get("transactionHash") or row.get("txHash") or row.get("tx_hash"),
    }


def enrich_and_dedupe(
    activity_files: List[Path],
    trades_true_file: Path,
    trades_false_file: Path,
    out_dir: Path,
) -> Path:
    _ensure_dir(out_dir)
    derived_dir = out_dir / "derived"
    _ensure_dir(derived_dir)

    taker_rows = _load_jsonl(trades_true_file)
    taker_keys = {_trade_key(row) for row in taker_rows}

    activity_rows: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
    for file_path in activity_files:
        for row in _load_jsonl(file_path):
            key = _trade_key(row)
            activity_rows[key] = row

    enriched_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str, str, str, str, str]] = set()
    for row in _load_jsonl(trades_false_file):
        key = _trade_key(row)
        seen_keys.add(key)
        merged = dict(_extract_trade_fields(row))
        activity_row = activity_rows.get(key)
        if activity_row:
            merged.update({k: v for k, v in _extract_trade_fields(activity_row).items() if v})
        merged["is_taker"] = key in taker_keys
        merged["is_maker"] = key not in taker_keys
        enriched_rows.append(merged)

    for key, row in activity_rows.items():
        if key in seen_keys:
            continue
        merged = dict(_extract_trade_fields(row))
        merged["is_taker"] = key in taker_keys
        merged["is_maker"] = key not in taker_keys
        enriched_rows.append(merged)

    output_jsonl = derived_dir / "trades_enriched.jsonl"
    _jsonl_dump(output_jsonl, enriched_rows, mode="w")
    _maybe_write_parquet(enriched_rows, derived_dir / "trades_enriched.parquet")
    return output_jsonl


def _maybe_write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
    except Exception:
        return


def analyze_legs(
    trades_enriched_file: Path, out_dir: Path, dt_list: List[int]
) -> Path:
    _ensure_dir(out_dir)
    derived_dir = out_dir / "derived"
    _ensure_dir(derived_dir)
    trades = _load_jsonl(trades_enriched_file)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in trades:
        condition = str(row.get("conditionId") or "").strip()
        if not condition:
            continue
        grouped.setdefault(condition, []).append(row)

    pairs: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "pair_rates": {},
        "median_leg_delay_sec": {},
    }

    for threshold in dt_list:
        total_trades = sum(len(rows) for rows in grouped.values())
        paired_trades = 0
        delays: List[float] = []
        for condition_id, rows in grouped.items():
            rows_sorted = sorted(rows, key=lambda r: r.get("timestamp") or "")
            used = set()
            for i, row in enumerate(rows_sorted):
                if i in used:
                    continue
                ts1 = _parse_timestamp(row.get("timestamp")) if _parse_timestamp else None
                if ts1 is None:
                    continue
                outcome1 = row.get("outcomeIndex")
                for j in range(i + 1, len(rows_sorted)):
                    if j in used:
                        continue
                    row2 = rows_sorted[j]
                    outcome2 = row2.get("outcomeIndex")
                    if outcome1 == outcome2:
                        continue
                    ts2 = _parse_timestamp(row2.get("timestamp")) if _parse_timestamp else None
                    if ts2 is None:
                        continue
                    delay = abs((ts2 - ts1).total_seconds())
                    if delay <= threshold:
                        used.add(i)
                        used.add(j)
                        paired_trades += 2
                        delays.append(delay)
                        pairs.append(
                            {
                                "conditionId": condition_id,
                                "outcomeIndex_a": outcome1,
                                "outcomeIndex_b": outcome2,
                                "timestamp_a": row.get("timestamp"),
                                "timestamp_b": row2.get("timestamp"),
                                "delay_sec": delay,
                                "threshold_sec": threshold,
                            }
                        )
                        break

        pair_rate = paired_trades / total_trades if total_trades else 0.0
        summary["pair_rates"][str(threshold)] = pair_rate
        if delays:
            delays_sorted = sorted(delays)
            mid = len(delays_sorted) // 2
            median = (
                delays_sorted[mid]
                if len(delays_sorted) % 2 == 1
                else (delays_sorted[mid - 1] + delays_sorted[mid]) / 2
            )
            summary["median_leg_delay_sec"][str(threshold)] = median
        else:
            summary["median_leg_delay_sec"][str(threshold)] = None

    pairs_file = derived_dir / "legs_pairs.jsonl"
    _jsonl_dump(pairs_file, pairs, mode="w")
    _maybe_write_parquet(pairs, derived_dir / "legs_pairs.parquet")

    summary_path = derived_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary_path


def _target_dir(base: Path, target: str, proxy_wallet: str) -> Path:
    name = target.strip().lstrip("@")
    if not name:
        name = proxy_wallet
    return base / name


def _write_profile(out_dir: Path, target: str, proxy_wallet: str, profile: Optional[Dict[str, Any]]) -> None:
    meta_dir = out_dir / "meta"
    _ensure_dir(meta_dir)
    payload = {
        "target": target,
        "proxy_wallet": proxy_wallet,
        "profile": profile,
        "fetched_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
    }
    with (meta_dir / "profile.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_backfill(args: argparse.Namespace) -> None:
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if start is None or end is None:
        raise ValueError("start/end 需要 ISO 日期或时间，如 2025-12-01 或 2025-12-01T00:00:00Z")
    if start >= end:
        raise ValueError("start 必须早于 end")

    proxy_wallet, profile = resolve_target(args.target)
    out_dir = _target_dir(Path(args.out), args.target, proxy_wallet)
    _ensure_dir(out_dir)
    _write_profile(out_dir, args.target, proxy_wallet, profile)

    activity_files = fetch_activity(
        proxy_wallet,
        start,
        end,
        out_dir,
        window_hours=args.window_hours,
        page_size=args.page_size,
    )

    raw_dir = out_dir / "raw"
    trades_true = raw_dir / "trades_takerOnly_true.jsonl"
    trades_false = raw_dir / "trades_takerOnly_false.jsonl"

    fetch_trades(
        proxy_wallet,
        taker_only=True,
        out_file=trades_true,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    fetch_trades(
        proxy_wallet,
        taker_only=False,
        out_file=trades_false,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )

    trades_enriched = enrich_and_dedupe(activity_files, trades_true, trades_false, out_dir)
    analyze_legs(trades_enriched, out_dir, dt_list=args.dt_list)


def run_live(args: argparse.Namespace) -> None:
    proxy_wallet, profile = resolve_target(args.target)
    out_dir = _target_dir(Path(args.out), args.target, proxy_wallet)
    _ensure_dir(out_dir)
    _write_profile(out_dir, args.target, proxy_wallet, profile)

    raw_dir = out_dir / "raw"
    _ensure_dir(raw_dir)
    live_file = raw_dir / "activity_live.jsonl"

    last_ts = None
    print("[INFO] 启动 live 模式，按 Ctrl+C 退出。")
    try:
        while True:
            end_ts = int(dt.datetime.now(tz=dt.timezone.utc).timestamp())
            params = {
                "user": proxy_wallet,
                "type": "TRADE",
                "limit": 500,
                "offset": 0,
                "end": end_ts,
            }
            if last_ts is not None:
                params["start"] = last_ts
            payload = _request_json(f"{DATA_API_ROOT}/activity", params=params)
            rows: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = payload.get("data") or payload.get("activity") or []
            if rows:
                _jsonl_dump(live_file, rows, mode="a")
                for row in rows:
                    ts = row.get("timestamp") or row.get("time") or row.get("createdAt")
                    parsed = _parse_timestamp(ts) if _parse_timestamp else None
                    if parsed:
                        last_ts = max(last_ts or 0, int(parsed.timestamp()))
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[INFO] live 模式已停止。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymarket 单账户数据分析脚本")
    parser.add_argument("--target", required=True, help="@handle 或 0x 地址")
    parser.add_argument("--out", default="data", help="输出根目录")
    parser.add_argument("--mode", choices=["backfill", "live"], default="backfill")
    parser.add_argument("--start", help="回溯起始时间 (ISO)")
    parser.add_argument("--end", help="回溯结束时间 (ISO)")
    parser.add_argument("--window-hours", type=int, default=6, help="activity 分片窗口小时数")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--poll", type=int, default=3, help="live 模式轮询间隔秒")
    parser.add_argument("--dt-list", type=int, nargs="*", default=[5, 10, 30])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "backfill":
        if not args.start or not args.end:
            parser.error("backfill 模式需要 --start/--end")
        run_backfill(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
