from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


GAMMA_API_ROOT = "https://gamma-api.polymarket.com"
DATA_API_ROOT = "https://data-api.polymarket.com"

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "nba_ncaab": {
        "category": "Sports",
        "keywords": [
            "nba",
            "ncaab",
            "ncaa basketball",
            "college basketball",
            "march madness",
            "final four",
            "playoffs",
        ],
        "exclude_keywords": [
            "nfl",
            "mlb",
            "nhl",
            "golf",
            "soccer",
            "tennis",
            "ufc",
            "crypto",
            "bitcoin",
            "election",
            "trump",
        ],
    },
    "politics_event": {
        "category": "Politics",
        "keywords": [
            "election",
            "trump",
            "biden",
            "harris",
            "iran",
            "israel",
            "ukraine",
            "tariff",
            "ceasefire",
            "regime",
            "president",
            "senate",
            "congress",
            "putin",
            "zelensky",
        ],
        "exclude_keywords": [
            "nba",
            "ncaab",
            "nfl",
            "mlb",
            "nhl",
            "bitcoin",
            "ethereum",
        ],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover candidate users by starting from topic-specific Polymarket markets.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS.keys()),
        required=True,
        help="Topic preset used to discover markets and users.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="CSV file path for discovered users. Defaults under data/.",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="JSON metadata path. Defaults next to output file.",
    )
    parser.add_argument(
        "--events-limit",
        type=int,
        default=1200,
        help="Maximum number of Gamma events to scan.",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=150,
        help="Maximum matched markets kept for user discovery.",
    )
    parser.add_argument(
        "--min-event-volume",
        type=float,
        default=1000.0,
        help="Minimum event volume before a matched event is kept.",
    )
    parser.add_argument(
        "--min-market-volume",
        type=float,
        default=300.0,
        help="Minimum market volume before a matched market is kept.",
    )
    parser.add_argument(
        "--trade-page-size",
        type=int,
        default=200,
        help="Trades page size per market.",
    )
    parser.add_argument(
        "--trade-pages-per-market",
        type=int,
        default=5,
        help="How many latest trade pages to inspect per market.",
    )
    parser.add_argument(
        "--max-position-users-per-token",
        type=int,
        default=40,
        help="How many top holders per token to ingest from market-positions.",
    )
    parser.add_argument(
        "--min-user-score",
        type=float,
        default=3.0,
        help="Minimum discovery score required to keep a user in output.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=2000,
        help="Maximum discovered users written to output.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=20.0,
        help="HTTP request timeout.",
    )
    return parser.parse_args()


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_float(value: object) -> float:
    try:
        text = str(value).strip().replace(",", "")
        if text == "":
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value: object) -> Optional[dt.datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float,
    retries: int = 3,
) -> Any:
    wait = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 8.0)
    raise RuntimeError("unreachable")


def _event_matches(
    event: Dict[str, Any],
    *,
    keywords: List[str],
    exclude_keywords: List[str],
) -> bool:
    text_blob = " ".join(
        [
            _normalize_text(event.get("title")),
            _normalize_text(event.get("slug")),
            _normalize_text(event.get("description")),
            _normalize_text(event.get("category")),
        ]
    ).strip()
    if not text_blob:
        return False
    if exclude_keywords and any(item in text_blob for item in exclude_keywords):
        return False
    return any(item in text_blob for item in keywords)


def _fetch_matching_events(
    session: requests.Session,
    *,
    category: str,
    keywords: List[str],
    exclude_keywords: List[str],
    events_limit: int,
    min_event_volume: float,
    timeout: float,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    offset = 0
    page_size = 100
    while offset < max(events_limit, page_size):
        payload = _request_json(
            session,
            f"{GAMMA_API_ROOT}/events",
            params={
                "limit": page_size,
                "offset": offset,
                "category": category,
                "archived": "false",
            },
            timeout=timeout,
        )
        if not isinstance(payload, list) or not payload:
            break
        for event in payload:
            if not isinstance(event, dict):
                continue
            if not _event_matches(event, keywords=keywords, exclude_keywords=exclude_keywords):
                continue
            if _parse_float(event.get("volume")) < min_event_volume:
                continue
            matched.append(event)
        offset += len(payload)
        if offset >= events_limit or len(payload) < page_size:
            break
    return matched


def _extract_markets_from_events(
    events: Iterable[Dict[str, Any]],
    *,
    min_market_volume: float,
    max_markets: int,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    markets: List[Dict[str, Any]] = []
    for event in events:
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            condition_id = _normalize_text(market.get("conditionId"))
            if not condition_id or condition_id in seen:
                continue
            market_volume = _parse_float(market.get("volume"))
            if market_volume < min_market_volume:
                continue
            seen.add(condition_id)
            markets.append(
                {
                    "condition_id": condition_id,
                    "slug": market.get("slug") or "",
                    "question": market.get("question") or "",
                    "event_slug": event.get("slug") or "",
                    "event_title": event.get("title") or "",
                    "event_id": str(event.get("id") or "").strip(),
                    "event_volume": _parse_float(event.get("volume")),
                    "market_volume": market_volume,
                }
            )
    markets.sort(key=lambda item: (item["event_volume"], item["market_volume"]), reverse=True)
    return markets[:max_markets]


def _update_user_stats_from_trade(
    user_stats: Dict[str, Dict[str, float]],
    market: Dict[str, Any],
    trade: Dict[str, Any],
) -> None:
    user = _normalize_text(trade.get("proxyWallet"))
    if not user:
        return
    stats = user_stats[user]
    stats["trade_count"] += 1.0
    stats["trade_volume"] += _parse_float(trade.get("price")) * _parse_float(trade.get("size"))
    stats["market_trade_hits"] += 1.0
    stats.setdefault("markets_touched", set()).add(market["condition_id"])


def _update_user_stats_from_position(
    user_stats: Dict[str, Dict[str, float]],
    market: Dict[str, Any],
    position: Dict[str, Any],
) -> None:
    user = _normalize_text(position.get("proxyWallet"))
    if not user:
        return
    stats = user_stats[user]
    stats["position_market_hits"] += 1.0
    stats["position_value"] += _parse_float(position.get("currentValue"))
    stats["position_realized_pnl"] += _parse_float(position.get("realizedPnl"))
    stats.setdefault("markets_touched", set()).add(market["condition_id"])


def _fetch_market_trades(
    session: requests.Session,
    *,
    market_condition_id: str,
    page_size: int,
    pages_per_market: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    for _ in range(max(1, pages_per_market)):
        payload = _request_json(
            session,
            f"{DATA_API_ROOT}/trades",
            params={
                "market": market_condition_id,
                "limit": page_size,
                "offset": offset,
            },
            timeout=timeout,
        )
        if not isinstance(payload, list) or not payload:
            break
        out.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < page_size:
            break
        offset += len(payload)
    return out


def _fetch_market_positions(
    session: requests.Session,
    *,
    market_condition_id: str,
    timeout: float,
) -> List[Dict[str, Any]]:
    payload = _request_json(
        session,
        f"{DATA_API_ROOT}/v1/market-positions",
        params={"market": market_condition_id},
        timeout=timeout,
    )
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for token_entry in payload:
        if not isinstance(token_entry, dict):
            continue
        positions = token_entry.get("positions") or []
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, dict):
                out.append(position)
    return out


def _score_user(stats: Dict[str, float]) -> float:
    markets_touched = float(len(stats.get("markets_touched", set())))
    trade_hits = float(stats.get("market_trade_hits", 0.0))
    position_hits = float(stats.get("position_market_hits", 0.0))
    trade_volume = float(stats.get("trade_volume", 0.0))
    position_value = float(stats.get("position_value", 0.0))
    return (
        markets_touched * 1.6
        + trade_hits * 0.08
        + position_hits * 0.12
        + min(trade_volume / 2000.0, 12.0)
        + min(position_value / 1000.0, 8.0)
    )


def _serialize_rows(
    user_stats: Dict[str, Dict[str, float]],
    *,
    min_user_score: float,
    max_users: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for user, stats in user_stats.items():
        score = _score_user(stats)
        if score < min_user_score:
            continue
        rows.append(
            {
                "user": user,
                "discovery_score": round(score, 6),
                "topic_markets_touched": len(stats.get("markets_touched", set())),
                "topic_trade_count": int(stats.get("trade_count", 0.0)),
                "topic_trade_volume": round(float(stats.get("trade_volume", 0.0)), 6),
                "topic_position_hits": int(stats.get("position_market_hits", 0.0)),
                "topic_position_value": round(float(stats.get("position_value", 0.0)), 6),
                "topic_position_realized_pnl": round(
                    float(stats.get("position_realized_pnl", 0.0)), 6
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            item["discovery_score"],
            item["topic_markets_touched"],
            item["topic_trade_volume"],
        ),
        reverse=True,
    )
    return rows[:max_users]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user",
        "discovery_score",
        "topic_markets_touched",
        "topic_trade_count",
        "topic_trade_volume",
        "topic_position_hits",
        "topic_position_value",
        "topic_position_realized_pnl",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    preset = PRESET_CONFIGS[args.preset]
    base_dir = Path(__file__).resolve().parent

    output_file = (
        Path(args.output_file)
        if args.output_file
        else base_dir / "data" / f"topic_seed_users_{args.preset}.csv"
    )
    if not output_file.is_absolute():
        output_file = (base_dir / output_file).resolve()

    metadata_file = (
        Path(args.metadata_file)
        if args.metadata_file
        else output_file.with_suffix(".metadata.json")
    )
    if not metadata_file.is_absolute():
        metadata_file = (base_dir / metadata_file).resolve()

    session = requests.Session()
    timeout = float(args.request_timeout_sec)

    print(
        f"[INFO] Discovering topic users: preset={args.preset} category={preset['category']}",
        flush=True,
    )
    events = _fetch_matching_events(
        session,
        category=str(preset["category"]),
        keywords=list(preset["keywords"]),
        exclude_keywords=list(preset["exclude_keywords"]),
        events_limit=max(1, int(args.events_limit)),
        min_event_volume=float(args.min_event_volume),
        timeout=timeout,
    )
    print(f"[INFO] Matched events: {len(events)}", flush=True)

    markets = _extract_markets_from_events(
        events,
        min_market_volume=float(args.min_market_volume),
        max_markets=max(1, int(args.max_markets)),
    )
    print(f"[INFO] Kept markets: {len(markets)}", flush=True)

    user_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "trade_count": 0.0,
            "trade_volume": 0.0,
            "market_trade_hits": 0.0,
            "position_market_hits": 0.0,
            "position_value": 0.0,
            "position_realized_pnl": 0.0,
            "markets_touched": set(),
        }
    )

    for index, market in enumerate(markets, start=1):
        condition_id = market["condition_id"]
        print(
            f"[INFO] ({index}/{len(markets)}) market={condition_id} slug={market['slug']}",
            flush=True,
        )
        try:
            trades = _fetch_market_trades(
                session,
                market_condition_id=condition_id,
                page_size=max(1, int(args.trade_page_size)),
                pages_per_market=max(1, int(args.trade_pages_per_market)),
                timeout=timeout,
            )
            for trade in trades:
                _update_user_stats_from_trade(user_stats, market, trade)

            positions = _fetch_market_positions(
                session,
                market_condition_id=condition_id,
                timeout=timeout,
            )
            positions.sort(
                key=lambda item: _parse_float(item.get("currentValue"))
                + _parse_float(item.get("realizedPnl")),
                reverse=True,
            )
            for position in positions[: max(1, int(args.max_position_users_per_token))]:
                _update_user_stats_from_position(user_stats, market, position)
        except requests.RequestException as exc:
            print(f"[WARN] market discovery failed for {condition_id}: {exc}", flush=True)

    rows = _serialize_rows(
        user_stats,
        min_user_score=float(args.min_user_score),
        max_users=max(1, int(args.max_users)),
    )
    _write_csv(output_file, rows)

    metadata = {
        "preset": args.preset,
        "output_file": str(output_file),
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "matched_events": len(events),
        "kept_markets": len(markets),
        "discovered_users": len(rows),
        "events_limit": args.events_limit,
        "max_markets": args.max_markets,
        "min_event_volume": args.min_event_volume,
        "min_market_volume": args.min_market_volume,
        "trade_pages_per_market": args.trade_pages_per_market,
        "max_position_users_per_token": args.max_position_users_per_token,
        "min_user_score": args.min_user_score,
        "sample_markets": markets[:20],
        "top_users": rows[:20],
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[INFO] Topic discovery complete: users={len(rows)} output={output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
