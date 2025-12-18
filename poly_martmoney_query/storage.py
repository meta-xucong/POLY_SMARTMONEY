from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from .models import AggregatedStats, MarketAggregation, Trade


def append_trades_csv(path: Path, trades: Iterable[Trade]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_hashes = set()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = row.get("tx_hash")
                if tx:
                    existing_hashes.add(tx)

    fieldnames = [
        "tx_hash",
        "market_id",
        "market_slug",
        "outcome",
        "side",
        "price",
        "size",
        "cost",
        "timestamp",
    ]

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        for trade in trades:
            if trade.tx_hash in existing_hashes:
                continue
            writer.writerow(
                {
                    "tx_hash": trade.tx_hash,
                    "market_id": trade.market_id,
                    "market_slug": trade.market_slug or "",
                    "outcome": trade.outcome or "",
                    "side": trade.side,
                    "price": f"{trade.price:.6f}",
                    "size": f"{trade.size:.6f}",
                    "cost": f"{trade.cost:.6f}",
                    "timestamp": trade.timestamp.isoformat(),
                }
            )
            existing_hashes.add(trade.tx_hash)


def write_market_stats_csv(path: Path, stats: AggregatedStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "market_id",
        "slug",
        "resolved",
        "resolved_outcome",
        "win",
        "pnl",
        "volume",
        "cash_flow",
        "remaining_positions",
        "trades_count",
        "first_trade_at",
        "last_trade_at",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in stats.markets:
            writer.writerow(
                {
                    "market_id": m.market_id,
                    "slug": m.slug or "",
                    "resolved": m.resolved,
                    "resolved_outcome": m.resolved_outcome or "",
                    "win": m.win if m.win is not None else "",
                    "pnl": f"{m.pnl:.6f}" if m.pnl is not None else "",
                    "volume": f"{m.volume:.6f}",
                    "cash_flow": f"{m.cash_flow:.6f}",
                    "remaining_positions": _format_positions(m.remaining_positions),
                    "trades_count": m.trades_count,
                    "first_trade_at": m.first_trade_at.isoformat(),
                    "last_trade_at": m.last_trade_at.isoformat(),
                }
            )

    summary_path = path.with_name(path.stem + "_summary" + path.suffix)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user",
                "start_time",
                "end_time",
                "total_volume",
                "resolved_pnl",
                "win_rate",
                "resolved_markets",
                "unresolved_markets",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "user": stats.user,
                "start_time": stats.start_time.isoformat() if stats.start_time else "",
                "end_time": stats.end_time.isoformat() if stats.end_time else "",
                "total_volume": f"{stats.total_volume:.6f}",
                "resolved_pnl": f"{stats.resolved_pnl:.6f}",
                "win_rate": f"{stats.win_rate:.4f}" if stats.win_rate is not None else "",
                "resolved_markets": stats.resolved_markets,
                "unresolved_markets": stats.unresolved_markets,
            }
        )


def _format_positions(positions: Dict[str, float]) -> str:
    parts: List[str] = []
    for outcome, size in positions.items():
        parts.append(f"{outcome}:{size:.4f}")
    return ";".join(parts)
