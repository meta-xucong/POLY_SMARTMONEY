from __future__ import annotations

from typing import Dict, Tuple


def risk_check(
    token_key: str,
    desired_shares: float,
    my_shares: float,
    ref_price: float,
    cfg: Dict[str, object],
) -> Tuple[bool, str]:
    blacklist = cfg.get("blacklist_token_keys") or []
    if token_key in blacklist:
        return False, "blacklist"

    max_per_token = float(cfg.get("max_notional_per_token") or 0)
    est_notional = abs(desired_shares) * ref_price if ref_price else 0.0
    if max_per_token > 0 and est_notional > max_per_token:
        return False, "max_notional_per_token"

    max_total = float(cfg.get("max_notional_total") or 0)
    total_notional = float(cfg.get("_total_notional") or 0)
    if max_total > 0 and total_notional > max_total:
        return False, "max_notional_total"

    return True, "ok"
