from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Trade:
    """标准化的成交记录，来源于 Polymarket `/trades` 接口。"""

    tx_hash: str
    market_id: str
    outcome: Optional[str]
    side: str
    price: float
    size: float
    cost: float
    timestamp: dt.datetime
    market_slug: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return abs(self.price * self.size)

    @classmethod
    def from_api(cls, raw: Dict[str, Any]) -> Optional["Trade"]:
        """从 `/trades` 返回的原始字典构建 Trade，兼容常见字段。"""

        tx_hash = str(raw.get("tx_hash") or raw.get("txHash") or raw.get("transactionHash") or "").strip()
        market_id = str(
            raw.get("conditionId")
            or raw.get("market")
            or raw.get("marketId")
            or raw.get("market_id")
            or ""
        ).strip()
        outcome = raw.get("outcome") or raw.get("tokenName") or raw.get("outcome_name")
        side = (raw.get("type") or raw.get("side") or "").upper()
        price = _coerce_float(raw.get("price") or raw.get("avgPrice") or raw.get("fillPrice"))
        size = _coerce_float(raw.get("size") or raw.get("amount") or raw.get("quantity"))
        cost = _coerce_float(raw.get("cost") or raw.get("value"))
        ts = _parse_timestamp(raw.get("timestamp") or raw.get("time") or raw.get("created_at"))
        slug = raw.get("marketSlug") or raw.get("slug")

        if not tx_hash or not market_id or price is None or size is None or ts is None:
            return None

        if cost is None:
            cost = price * size

        return cls(
            tx_hash=tx_hash,
            market_id=market_id,
            outcome=str(outcome) if outcome is not None else None,
            side=side or "",
            price=price,
            size=size,
            cost=cost,
            timestamp=ts,
            market_slug=str(slug) if slug is not None else None,
            raw=raw,
        )


def _parse_timestamp(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            # `/trades` 时间戳通常为秒或毫秒
            if value > 1e12:
                value /= 1000.0
            return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return dt.datetime.fromisoformat(text)
    except Exception:
        return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()
            if cleaned == "":
                return None
            return float(cleaned)
    except Exception:
        return None
    return None


@dataclass
class MarketAggregation:
    market_id: str
    slug: Optional[str]
    resolved_outcome: Optional[str]
    volume: float
    cash_flow: float
    remaining_positions: Dict[str, float]
    pnl: Optional[float]
    resolved: bool
    win: Optional[bool]
    trades_count: int
    first_trade_at: dt.datetime
    last_trade_at: dt.datetime


@dataclass
class AggregatedStats:
    user: str
    start_time: Optional[dt.datetime]
    end_time: Optional[dt.datetime]
    total_volume: float
    resolved_pnl: float
    win_rate: Optional[float]
    resolved_markets: int
    unresolved_markets: int
    markets: List[MarketAggregation]
