from __future__ import annotations

from typing import Dict, Tuple


def risk_check(
    token_key: str,
    order_shares: float,
    my_shares: float,
    ref_price: float,
    cfg: Dict[str, object],
) -> Tuple[bool, str]:
    blacklist = cfg.get("blacklist_token_keys") or []
    if token_key in blacklist:
        return False, "blacklist"

    max_per_token = float(cfg.get("max_notional_per_token") or 0)
    order_notional = abs(order_shares) * ref_price if ref_price else 0.0
    if max_per_token > 0 and order_notional > max_per_token:
        return False, "max_notional_per_token"

    return True, "ok"
