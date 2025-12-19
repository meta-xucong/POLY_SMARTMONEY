"""
读取 poly_martmoney_query_run.py 输出的 CSV，生成用户特征表与候选名单。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen Polymarket smart money users")
    parser.add_argument(
        "--config",
        default="screen_users_config.json",
        help="配置文件路径（默认 screen_users_config.json）",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_datetime(value: str) -> Optional[dt.datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(values_sorted) - 1)
    if lower == upper:
        return values_sorted[lower]
    weight = idx - lower
    return values_sorted[lower] * (1 - weight) + values_sorted[upper] * weight


def _mean(values: Iterable[float]) -> Optional[float]:
    values_list = list(values)
    if not values_list:
        return None
    return sum(values_list) / len(values_list)


def _median(values: List[float]) -> Optional[float]:
    return _percentile(values, 0.5)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _load_user_summary_map(path: Path) -> Dict[str, Dict[str, str]]:
    summaries = {}
    for row in _read_csv(path):
        user = row.get("user")
        if user:
            summaries[user] = row
    return summaries


def _extract_summary_times(summary: Dict[str, str]) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    start_time = _parse_datetime(summary.get("start_time", ""))
    end_time = _parse_datetime(summary.get("end_time", ""))
    return start_time, end_time


def _calculate_window_days(
    start_time: Optional[dt.datetime],
    end_time: Optional[dt.datetime],
    default_days: float,
) -> float:
    if start_time and end_time:
        delta = end_time - start_time
        days = max(delta.total_seconds() / 86400, 0.0)
        if days > 0:
            return days
    return float(default_days)


def _collect_daily_counts(timestamps: List[dt.datetime]) -> Dict[dt.date, int]:
    daily_counts: Dict[dt.date, int] = {}
    for ts in timestamps:
        day = ts.date()
        daily_counts[day] = daily_counts.get(day, 0) + 1
    return daily_counts


def _compute_burstiness(daily_counts: Dict[dt.date, int]) -> Optional[float]:
    if not daily_counts:
        return None
    counts = list(daily_counts.values())
    mean_daily = _mean(counts)
    if mean_daily in (None, 0):
        return None
    return max(counts) / mean_daily


def _compute_intervals_minutes(timestamps: List[dt.datetime]) -> List[float]:
    if len(timestamps) < 2:
        return []
    timestamps_sorted = sorted(timestamps)
    intervals = []
    for prev, nxt in zip(timestamps_sorted, timestamps_sorted[1:]):
        delta = nxt - prev
        intervals.append(delta.total_seconds() / 60)
    return intervals


def _normalize(value: Optional[float], clamp: Optional[float]) -> float:
    if value is None:
        return 0.0
    if clamp is None or clamp <= 0:
        return value
    return max(min(value, clamp), 0.0) / clamp


def _compute_copy_score(metrics: Dict[str, Optional[float]], config: Dict[str, Any]) -> float:
    weights = config.get("score_weights", {})
    clamps = config.get("score_clamps", {})
    score = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        norm_value = _normalize(value, clamps.get(key))
        score += weight * norm_value
    return score


def _apply_filters(metrics: Dict[str, Optional[float]], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    filters = config.get("filters", {})
    failures = []

    def _check_min(key: str, label: str) -> None:
        threshold = filters.get(key)
        value = metrics.get(label)
        if threshold is None:
            return
        if value is None or value < threshold:
            failures.append(f"{label}<{threshold}")

    def _check_max(key: str, label: str) -> None:
        threshold = filters.get(key)
        value = metrics.get(label)
        if threshold is None:
            return
        if value is None or value > threshold:
            failures.append(f"{label}>{threshold}")

    _check_min("min_closed_count", "closed_count")
    _check_max("max_trades_per_day", "trades_per_day")
    _check_max("max_daily_trades", "max_trades_per_day")
    _check_max("max_p90_cost", "p90_cost")
    _check_max("max_cost", "max_cost")
    _check_max("max_open_exposure", "open_exposure")

    max_loss_threshold = filters.get("max_loss")
    max_loss = metrics.get("max_loss")
    if max_loss_threshold is not None:
        if max_loss is None:
            failures.append(f"max_loss<{max_loss_threshold}")
        elif max_loss < max_loss_threshold:
            failures.append(f"max_loss<{max_loss_threshold}")

    return (len(failures) == 0), failures


def _build_features(
    user: str,
    closed_rows: List[Dict[str, str]],
    open_rows: List[Dict[str, str]],
    summary_row: Optional[Dict[str, str]],
    config: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    flat_eps = float(config.get("flat_pnl_epsilon", 1e-9))
    min_cost_for_roi = float(config.get("min_cost_for_roi", 1.0))
    bayes_alpha = float(config.get("bayes_alpha", 2.0))
    bayes_beta = float(config.get("bayes_beta", 2.0))

    timestamps: List[dt.datetime] = []
    pnls: List[float] = []
    costs: List[float] = []
    roi_values: List[float] = []

    win_count = 0
    loss_count = 0
    flat_count = 0

    for row in closed_rows:
        pnl = _parse_float(row.get("realized_pnl", ""))
        avg_price = _parse_float(row.get("avg_price", ""))
        total_bought = _parse_float(row.get("total_bought", ""))
        ts = _parse_datetime(row.get("timestamp", ""))

        if pnl is not None:
            pnls.append(pnl)
            if pnl > flat_eps:
                win_count += 1
            elif pnl < -flat_eps:
                loss_count += 1
            else:
                flat_count += 1

        if avg_price is not None and total_bought is not None:
            cost = avg_price * total_bought
            costs.append(cost)
            if pnl is not None and cost >= min_cost_for_roi:
                roi_values.append(pnl / cost)

        if ts is not None:
            timestamps.append(ts)

    closed_count = len(closed_rows)
    win_rate_no_flat = None
    if win_count + loss_count > 0:
        win_rate_no_flat = win_count / (win_count + loss_count)

    bayes_win_rate = None
    if win_count + loss_count > 0:
        bayes_win_rate = (win_count + bayes_alpha) / (
            win_count + loss_count + bayes_alpha + bayes_beta
        )

    window_days = float(config.get("window_days_default", 30))
    asof_time = None
    if summary_row:
        start_time, end_time = _extract_summary_times(summary_row)
        window_days = _calculate_window_days(start_time, end_time, window_days)
        asof_time = _parse_datetime(summary_row.get("asof_time", ""))

    trades_per_day = None
    if window_days > 0:
        trades_per_day = closed_count / window_days

    daily_counts = _collect_daily_counts(timestamps)
    max_trades_per_day = max(daily_counts.values()) if daily_counts else None
    p95_trades_per_day = _percentile(list(daily_counts.values()), 0.95)
    burstiness = _compute_burstiness(daily_counts)

    intervals_minutes = _compute_intervals_minutes(timestamps)
    interval_p10 = _percentile(intervals_minutes, 0.1)
    interval_median = _median(intervals_minutes)

    mean_pnl = _mean(pnls)
    median_pnl = _median(pnls)
    max_loss = min(pnls) if pnls else None

    loss_values = [p for p in pnls if p < 0]
    p95_loss = _percentile(loss_values, 0.95)

    mean_cost = _mean(costs)
    median_cost = _median(costs)
    p90_cost = _percentile(costs, 0.9)
    max_cost = max(costs) if costs else None
    sum_cost = sum(costs) if costs else None

    mean_roi = _mean(roi_values)
    median_roi = _median(roi_values)

    win_pnl_sum = sum(p for p in pnls if p > 0)
    loss_pnl_sum = sum(p for p in pnls if p < 0)
    profit_factor = None
    if loss_pnl_sum < 0:
        profit_factor = win_pnl_sum / abs(loss_pnl_sum)

    open_values: List[float] = []
    open_end_dates: List[dt.datetime] = []
    for row in open_rows:
        current_value = _parse_float(row.get("current_value", ""))
        if current_value is not None:
            open_values.append(current_value)
        end_date = _parse_datetime(row.get("end_date", ""))
        if end_date is not None:
            open_end_dates.append(end_date)

    open_exposure = sum(open_values) if open_values else 0.0
    open_count = len(open_rows)
    top1_current_value = max(open_values) if open_values else None
    concentration = (
        _safe_ratio(top1_current_value, open_exposure) if open_exposure > 0 else None
    )

    near_expiry_days = float(config.get("near_expiry_days", 3))
    if asof_time is None:
        asof_time = dt.datetime.now(tz=dt.timezone.utc)

    near_expiry_value = 0.0
    for row in open_rows:
        end_date = _parse_datetime(row.get("end_date", ""))
        current_value = _parse_float(row.get("current_value", ""))
        if end_date is None or current_value is None:
            continue
        seconds_to_expiry = (end_date - asof_time).total_seconds()
        if 0 <= seconds_to_expiry <= near_expiry_days * 86400:
            near_expiry_value += current_value

    near_expiry_ratio = (
        near_expiry_value / open_exposure if open_exposure > 0 else None
    )

    metrics: Dict[str, Optional[float]] = {
        "closed_count": float(closed_count),
        "win_count": float(win_count),
        "loss_count": float(loss_count),
        "flat_count": float(flat_count),
        "win_rate_no_flat": win_rate_no_flat,
        "bayes_win_rate": bayes_win_rate,
        "trades_per_day": trades_per_day,
        "max_trades_per_day": float(max_trades_per_day) if max_trades_per_day else None,
        "p95_trades_per_day": p95_trades_per_day,
        "burstiness": burstiness,
        "interval_p10_minutes": interval_p10,
        "interval_median_minutes": interval_median,
        "mean_pnl": mean_pnl,
        "median_pnl": median_pnl,
        "max_loss": max_loss,
        "p95_loss": p95_loss,
        "mean_cost": mean_cost,
        "median_cost": median_cost,
        "p90_cost": p90_cost,
        "max_cost": max_cost,
        "sum_cost": sum_cost,
        "mean_roi": mean_roi,
        "median_roi": median_roi,
        "profit_factor": profit_factor,
        "open_exposure": open_exposure,
        "open_count": float(open_count),
        "concentration": concentration,
        "near_expiry_ratio": near_expiry_ratio,
    }

    return metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parent
    config_path = (base_dir / args.config).resolve()
    config = _load_config(config_path)

    data_dir = (base_dir / config.get("data_dir", "data")).resolve()
    users_dir = (base_dir / config.get("users_dir", "data/users")).resolve()
    output_dir = (base_dir / config.get("output_dir", "data")).resolve()

    features_filename = config.get("features_filename", "users_features.csv")
    candidates_filename = config.get("candidates_filename", "candidates.csv")
    metadata_filename = config.get("metadata_filename", "screening_metadata.json")

    summary_map = _load_user_summary_map(data_dir / "users_summary.csv")

    features_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for user_dir in sorted(users_dir.iterdir() if users_dir.exists() else []):
        if not user_dir.is_dir():
            continue
        user = user_dir.name
        closed_rows = _read_csv(user_dir / "closed_positions.csv")
        open_rows = _read_csv(user_dir / "positions.csv")
        summary_row = None
        if (user_dir / "summary.csv").exists():
            summary_rows = _read_csv(user_dir / "summary.csv")
            if summary_rows:
                summary_row = summary_rows[0]
        elif user in summary_map:
            summary_row = summary_map[user]

        metrics = _build_features(user, closed_rows, open_rows, summary_row, config)
        row: Dict[str, Any] = {"user": user}
        row.update(metrics)
        row["copy_score"] = _compute_copy_score(row, config)

        passed, failures = _apply_filters(row, config)
        row["passed_filter"] = passed
        row["filter_failures"] = ";".join(failures)

        features_rows.append(row)
        if passed:
            candidate_rows.append(row)

    features_rows = sorted(features_rows, key=lambda row: row.get("copy_score", 0), reverse=True)
    candidate_rows = sorted(candidate_rows, key=lambda row: row.get("copy_score", 0), reverse=True)

    _write_csv(output_dir / features_filename, features_rows)
    _write_csv(output_dir / candidates_filename, candidate_rows)

    metadata = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "config": config,
        "features_file": str((output_dir / features_filename).resolve()),
        "candidates_file": str((output_dir / candidates_filename).resolve()),
        "users_count": len(features_rows),
        "candidates_count": len(candidate_rows),
    }
    with (output_dir / metadata_filename).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"[INFO] 完成筛选：全量={len(features_rows)}，候选={len(candidate_rows)}，"
        f"输出目录={output_dir}"
    )


if __name__ == "__main__":
    main()
