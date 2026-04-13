from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether existing smartmoney_query data is sufficient for offline screening.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory produced by poly_martmoney_query_run.py",
    )
    parser.add_argument(
        "--min-account-age-days",
        type=float,
        default=300.0,
        help="Required account age threshold in days",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.51,
        help="Required win rate threshold",
    )
    parser.add_argument(
        "--min-lifetime-pnl",
        type=float,
        default=0.0,
        help="Required lifetime pnl lower bound",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def main() -> None:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    users_summary_path = data_dir / "users_summary.csv"
    rows = _read_csv(users_summary_path)

    if not rows:
        raise SystemExit(f"users_summary.csv not found or empty: {users_summary_path}")

    total = len(rows)
    lifetime_ok = 0
    lifetime_missing = 0
    age_ok = 0
    age_missing = 0
    win_ok = 0
    win_missing = 0
    human_like = 0
    hft_flagged = 0
    fully_ready = 0
    summary_dirs_missing = 0

    missing_lifetime_users: List[str] = []
    users_dir = data_dir / "users"

    for row in rows:
        user = (row.get("user") or "").strip()
        if not user:
            continue

        lifetime_status = (row.get("lifetime_status") or "").strip().lower()
        lifetime_pnl = _to_float(row.get("lifetime_realized_pnl_sum"))
        age_days = _to_float(row.get("account_age_days"))
        win_rate = _to_float(row.get("win_rate_no_flat"))
        suspected_hft = _to_bool(row.get("suspected_hft"))

        user_dir = users_dir / user
        if not user_dir.exists():
            summary_dirs_missing += 1

        lifetime_is_ready = lifetime_status == "ok" and lifetime_pnl is not None
        if lifetime_is_ready:
            lifetime_ok += 1
        else:
            lifetime_missing += 1
            missing_lifetime_users.append(user)

        if age_days is not None:
            if age_days >= args.min_account_age_days:
                age_ok += 1
        else:
            age_missing += 1

        if win_rate is not None:
            if win_rate >= args.min_win_rate:
                win_ok += 1
        else:
            win_missing += 1

        if suspected_hft is True:
            hft_flagged += 1
        else:
            human_like += 1

        if (
            lifetime_is_ready
            and lifetime_pnl is not None
            and lifetime_pnl > args.min_lifetime_pnl
            and age_days is not None
            and age_days >= args.min_account_age_days
            and win_rate is not None
            and win_rate >= args.min_win_rate
            and suspected_hft is not True
        ):
            fully_ready += 1

    report = {
        "data_dir": str(data_dir),
        "users_summary_path": str(users_summary_path),
        "total_users": total,
        "lifetime_ok_count": lifetime_ok,
        "lifetime_missing_or_not_ok_count": lifetime_missing,
        "account_age_ge_threshold_count": age_ok,
        "account_age_missing_count": age_missing,
        "win_rate_ge_threshold_count": win_ok,
        "win_rate_missing_count": win_missing,
        "non_hft_count": human_like,
        "hft_flagged_count": hft_flagged,
        "user_dir_missing_count": summary_dirs_missing,
        "fully_ready_for_offline_screening_count": fully_ready,
        "can_enforce_overall_positive_pnl_offline": lifetime_missing == 0,
        "recommended_next_action": (
            "offline_screening_ok"
            if lifetime_missing == 0
            else "backfill_lifetime_for_missing_users"
        ),
        "sample_missing_lifetime_users": missing_lifetime_users[:20],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
