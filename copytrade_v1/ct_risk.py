from __future__ import annotations

from typing import Dict, Optional, Tuple


def risk_check(
    token_key: str,
    order_shares: float,
    my_shares: float,
    ref_price: float,
    cfg: Dict[str, object],
    side: Optional[str] = None,
    planned_total_notional: Optional[float] = None,
) -> Tuple[bool, str]:
    blacklist = cfg.get("blacklist_token_keys") or []
    if token_key in blacklist:
        return False, "blacklist"

    max_per_token = float(cfg.get("max_notional_per_token") or 0)
    order_notional = abs(order_shares) * ref_price if ref_price else 0.0
    if max_per_token > 0 and order_notional > max_per_token:
        return False, "max_notional_per_token"

    max_total = float(cfg.get("max_notional_total") or 0)
    if max_total > 0 and planned_total_notional is not None and side is not None:
        if str(side).upper() == "BUY":
            delta = abs(order_shares) * ref_price
            if planned_total_notional + delta > max_total:
                return False, "max_notional_total"

    return True, "ok"
