"""
一键执行示例：使用 Data API 拉取 closed-positions + positions，统计交易笔数、
每笔盈亏、总盈亏与胜率，并写入 data/ 目录。
"""
import argparse
import datetime as dt
from pathlib import Path
from typing import List

from poly_martmoney_query.api_client import DataApiClient
from poly_martmoney_query.processors import summarize_user
from poly_martmoney_query.storage import (
    write_closed_positions_csv,
    write_positions_csv,
    write_user_summaries_csv,
    write_user_summary_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket smart money query runner")
    parser.add_argument("--user", help="单地址模式，指定要查询的钱包地址")
    parser.add_argument("--days", type=int, default=30, help="统计区间天数（默认 30）")
    parser.add_argument("--top", type=int, default=50, help="批量模式取 leaderboard 前 N 名")
    parser.add_argument("--period", default="ALL", help="leaderboard 时间维度（默认 ALL）")
    parser.add_argument("--order-by", default="vol", help="leaderboard 排序字段（默认 vol）")
    parser.add_argument(
        "--size-threshold",
        type=float,
        default=0.0,
        help="positions 的 sizeThreshold（默认 0）",
    )
    return parser.parse_args()


def _collect_users(client: DataApiClient, args: argparse.Namespace) -> List[str]:
    if args.user:
        return [args.user]

    leaderboard_users = []
    print(f"[INFO] 获取 leaderboard（{args.period}，按 {args.order_by}）……", flush=True)
    for item in client.iter_leaderboard(
        period=args.period,
        order_by=args.order_by,
        page_size=100,
        max_pages=20,
    ):
        addr = item.get("proxyWallet") or item.get("address")
        if addr:
            leaderboard_users.append(addr)
        if len(leaderboard_users) >= args.top:
            break

    if not leaderboard_users:
        raise SystemExit("未获取到 leaderboard 用户，检查网络或 API 可用性")

    return leaderboard_users


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

    for idx, addr in enumerate(users, start=1):
        print(f"[INFO] ({idx}/{len(users)}) 抓取地址 {addr} 的仓位数据……", flush=True)
        closed_positions = client.fetch_closed_positions(
            user=addr,
            start_time=start,
            end_time=end,
        )
        open_positions = client.fetch_positions(
            user=addr,
            size_threshold=args.size_threshold,
        )
        summary = summarize_user(
            closed_positions,
            open_positions,
            user=addr,
            start_time=start,
            end_time=end,
            asof_time=now,
        )
        summaries.append(summary)

        user_dir = users_dir / addr
        write_closed_positions_csv(user_dir / "closed_positions.csv", closed_positions)
        write_positions_csv(user_dir / "positions.csv", open_positions)
        write_user_summary_csv(user_dir / "summary.csv", summary)

        win_rate_text = (
            f"{summary.win_rate_all:.2%}" if summary.win_rate_all is not None else "N/A"
        )
        print(
            f"[INFO] 地址 {addr}：已平仓={summary.closed_count}，"
            f"已实现盈亏={summary.closed_realized_pnl_sum:.4f}，"
            f"胜率={win_rate_text}",
            flush=True,
        )

    write_user_summaries_csv(data_dir / "users_summary.csv", summaries)
    print(f"[INFO] 完成：已写入 {data_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
