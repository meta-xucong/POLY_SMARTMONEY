from __future__ import annotations

from typing import Dict, List, Tuple

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
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    positions, info = client.fetch_positions(
        user,
        size_threshold=size_threshold,
        return_info=True,
    )
    normalized: List[Dict[str, object]] = []
    for pos in positions:
        normalized_pos = _normalize_position(pos)
        if normalized_pos is None:
            continue
        normalized.append(normalized_pos)
    return normalized, info
