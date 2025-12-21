"""
一键执行示例：使用 Data API 拉取 closed-positions + positions，统计交易笔数、
每笔盈亏、总盈亏与胜率，并写入 data/ 目录。
"""
import argparse
import csv
import datetime as dt
import math
from pathlib import Path
from typing import Iterable, List, Optional

from poly_martmoney_query.api_client import DataApiClient
from poly_martmoney_query.models import UserSummary
from poly_martmoney_query.processors import summarize_user
from poly_martmoney_query.storage import (
    write_closed_positions_csv,
    write_positions_csv,
    write_trade_actions_csv,
    write_user_summaries_csv,
    write_user_summary_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket smart money query runner")
    parser.add_argument("--user", help="单地址模式，指定要查询的钱包地址")
    parser.add_argument("--days", type=int, default=30, help="统计区间天数（默认 30）")
    parser.add_argument("--top", type=int, default=50, help="批量模式取 leaderboard 前 N 名")
    parser.add_argument("--period", default="MONTH", help="leaderboard 时间维度（默认 MONTH）")
    parser.add_argument("--order-by", default="pnl", help="leaderboard 排序字段（默认 pnl）")
    parser.add_argument(
        "--size-threshold",
        type=float,
        default=0.0,
        help="positions 的 sizeThreshold（默认 0）",
    )
    parser.add_argument(
        "--min-leaderboard-pnl",
        type=float,
        default=None,
        help="仅保留 leaderboard 已实现盈亏 >= 阈值的地址（默认不过滤）",
    )
    parser.add_argument(
        "--min-leaderboard-vol",
        type=float,
        default=None,
        help="仅保留 leaderboard 成交量 >= 阈值的地址（默认不过滤）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 data/users/<addr>/summary.csv 已存在则跳过该地址",
    )
    return parser.parse_args()


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric(item: dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in item:
            parsed = _to_float(item.get(key))
            if parsed is not None:
                return parsed
    return None


def _collect_users(client: DataApiClient, args: argparse.Namespace) -> List[str]:
    if args.user:
        return [args.user]

    leaderboard_users = []
    seen = set()
    print(f"[INFO] 获取 leaderboard（{args.period}，按 {args.order_by}）……", flush=True)
    page_size = 50
    target = max(1, args.top)
    max_pages = math.ceil(target / page_size) + 2
    for item in client.iter_leaderboard(
        period=args.period,
        order_by=args.order_by,
        page_size=page_size,
        max_pages=max_pages,
    ):
        min_pnl = args.min_leaderboard_pnl
        if min_pnl is not None:
            pnl_value = _extract_metric(item, ("pnl", "profit", "PNL"))
            if pnl_value is not None and pnl_value < min_pnl:
                continue
        min_vol = args.min_leaderboard_vol
        if min_vol is not None:
            vol_value = _extract_metric(item, ("volume", "vol", "VOL"))
            if vol_value is not None and vol_value < min_vol:
                continue
        addr = item.get("proxyWallet") or item.get("address")
        if addr:
            normalized = addr.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            leaderboard_users.append(addr)
        if len(leaderboard_users) >= target:
            break

    if not leaderboard_users:
        raise SystemExit("未获取到 leaderboard 用户，检查网络或 API 可用性")
    if len(leaderboard_users) < target:
        print(
            f"[WARN] leaderboard 去重后仅获取到 {len(leaderboard_users)} 个地址，"
            f"未达目标 {target}。",
            flush=True,
        )

    return leaderboard_users


def _parse_datetime(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_existing_summary(path: Path) -> Optional[UserSummary]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if not row:
                return None
            return UserSummary(
                user=row.get("user", ""),
                start_time=_parse_datetime(row.get("start_time", "")),
                end_time=_parse_datetime(row.get("end_time", "")),
                account_start_time=_parse_datetime(row.get("account_start_time", "")),
                account_age_days=_to_float(row.get("account_age_days")),
                lifetime_realized_pnl_sum=_parse_float(
                    row.get("lifetime_realized_pnl_sum", "0")
                ),
                closed_count=_parse_int(row.get("closed_count", "0")),
                closed_realized_pnl_sum=_parse_float(row.get("closed_realized_pnl_sum", "0")),
                win_count=_parse_int(row.get("win_count", "0")),
                loss_count=_parse_int(row.get("loss_count", "0")),
                flat_count=_parse_int(row.get("flat_count", "0")),
                win_rate_all=_to_float(row.get("win_rate_all")),
                win_rate_no_flat=_to_float(row.get("win_rate_no_flat")),
                open_count=_parse_int(row.get("open_count", "0")),
                open_unrealized_pnl_sum=_parse_float(row.get("open_unrealized_pnl_sum", "0")),
                open_realized_pnl_sum=_parse_float(row.get("open_realized_pnl_sum", "0")),
                asof_time=_parse_datetime(row.get("asof_time", "")) or dt.datetime.now(
                    tz=dt.timezone.utc
                ),
                status=row.get("status") or None,
            )
    except Exception:
        return None


def main() -> None:
    args = _parse_args()
    client = DataApiClient()
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    users_dir = data_dir / "users"
    data_dir.mkdir(exist_ok=True)
    users_dir.mkdir(exist_ok=True)

    now = dt.datetime.now(tz=dt.timezone.utc)
    start = now - dt.timedelta(days=args.days) if args.days > 0 else None
    end = now

    users = _collect_users(client, args)
    summaries = []
    report_rows: List[dict] = []

    for idx, addr in enumerate(users, start=1):
        print(f"[INFO] ({idx}/{len(users)}) 抓取地址 {addr} 的仓位数据……", flush=True)
        user_dir = users_dir / addr
        summary_path = user_dir / "summary.csv"
        if args.resume and summary_path.exists():
            existing_summary = _load_existing_summary(summary_path)
            if existing_summary is not None and existing_summary.status == "ok":
                summaries.append(existing_summary)
                report_rows.append(
                    {
                        "user": addr,
                        "status": "skipped",
                        "ok": True,
                        "incomplete": False,
                        "closed_count": existing_summary.closed_count,
                        "open_count": existing_summary.open_count,
                        "closed_incomplete": False,
                        "open_incomplete": False,
                        "closed_hit_max_pages": False,
                        "open_hit_max_pages": False,
                        "error_msg": "resume-skip",
                    }
                )
                print(f"[INFO] 地址 {addr} 已存在 summary.csv，跳过。", flush=True)
                continue
            if existing_summary is None:
                print(
                    f"[WARN] 地址 {addr} summary.csv 无法解析，将重新抓取。",
                    flush=True,
                )
            else:
                print(
                    f"[WARN] 地址 {addr} summary.csv 状态为 {existing_summary.status or 'unknown'}，将重新抓取。",
                    flush=True,
                )

        try:
            closed_positions, closed_info = client.fetch_closed_positions_window(
                user=addr,
                start_time=start,
                end_time=end,
                return_info=True,
            )
            trade_actions, trade_info = client.fetch_trade_actions_window_from_activity(
                user=addr,
                start_time=start,
                end_time=end,
                return_info=True,
            )
            lifetime_closed_positions, lifetime_info = client.fetch_closed_positions_window(
                user=addr,
                return_info=True,
            )
            open_positions, open_info = client.fetch_positions(
                user=addr,
                size_threshold=args.size_threshold,
                return_info=True,
            )
            account_start_time = client.fetch_account_start_time_from_activity(user=addr)
            lifetime_realized_pnl_sum = sum(
                pos.realized_pnl for pos in lifetime_closed_positions
            )
            summary = summarize_user(
                closed_positions,
                open_positions,
                user=addr,
                start_time=start,
                end_time=end,
                asof_time=now,
                account_start_time=account_start_time,
                lifetime_realized_pnl_sum=lifetime_realized_pnl_sum,
            )
            incomplete = (
                bool(closed_info["incomplete"])
                or bool(open_info["incomplete"])
                or bool(lifetime_info["incomplete"])
                or bool(trade_info["incomplete"])
            )
            status = "ok" if not incomplete else "incomplete"
            summary.status = status
            summaries.append(summary)

            write_closed_positions_csv(user_dir / "closed_positions.csv", closed_positions)
            write_trade_actions_csv(user_dir / "trade_actions.csv", trade_actions)
            write_positions_csv(user_dir / "positions.csv", open_positions)
            write_user_summary_csv(summary_path, summary)

            error_msg = "; ".join(
                msg
                for msg in [
                    closed_info.get("error_msg"),
                    trade_info.get("error_msg"),
                    open_info.get("error_msg"),
                    lifetime_info.get("error_msg"),
                ]
                if msg
            )
            report_rows.append(
                {
                    "user": addr,
                    "status": status,
                    "ok": not incomplete,
                    "incomplete": incomplete,
                    "closed_count": summary.closed_count,
                    "open_count": summary.open_count,
                    "closed_incomplete": bool(closed_info["incomplete"]),
                    "open_incomplete": bool(open_info["incomplete"]),
                    "closed_hit_max_pages": bool(closed_info["hit_max_pages"]),
                    "open_hit_max_pages": bool(open_info["hit_max_pages"]),
                    "error_msg": error_msg,
                }
            )

            win_rate_text = (
                f"{summary.win_rate_all:.2%}" if summary.win_rate_all is not None else "N/A"
            )
            account_days_text = (
                f"{summary.account_age_days:.1f}天"
                if summary.account_age_days is not None
                else "N/A"
            )
            print(
                f"[INFO] 地址 {addr}：已平仓={summary.closed_count}，"
                f"已实现盈亏={summary.closed_realized_pnl_sum:.4f}，"
                f"持仓已实现={summary.open_realized_pnl_sum:.4f}，"
                f"持仓浮盈浮亏={summary.open_unrealized_pnl_sum:.4f}，"
                f"胜率={win_rate_text}，"
                f"账号年龄={account_days_text}，"
                f"历史总收益={summary.lifetime_realized_pnl_sum:.4f}",
                flush=True,
            )
            if incomplete:
                print(
                    f"[WARN] 地址 {addr} 数据不完整：{error_msg or 'unknown_error'}",
                    flush=True,
                )
        except Exception as exc:
            report_rows.append(
                {
                    "user": addr,
                    "status": "error",
                    "ok": False,
                    "incomplete": True,
                    "closed_count": "",
                    "open_count": "",
                    "closed_incomplete": True,
                    "open_incomplete": True,
                    "closed_hit_max_pages": False,
                    "open_hit_max_pages": False,
                    "error_msg": str(exc),
                }
            )
            print(f"[WARN] 地址 {addr} 处理失败：{exc}", flush=True)

    write_user_summaries_csv(data_dir / "users_summary.csv", summaries)
    report_path = data_dir / "run_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "user",
            "status",
            "ok",
            "incomplete",
            "closed_count",
            "open_count",
            "closed_incomplete",
            "open_incomplete",
            "closed_hit_max_pages",
            "open_hit_max_pages",
            "error_msg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"[INFO] 完成：已写入 {data_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
