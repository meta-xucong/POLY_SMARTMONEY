"""
一键执行的示例脚本：从 Data API 拉取 leaderboard 用户，抓取 30 天内成交，
并将成交明细与市场聚合结果写入 data/ 目录。
"""
import datetime as dt
from pathlib import Path

from poly_martmoney_query.api_client import DataApiClient
from poly_martmoney_query.processors import aggregate_markets
from poly_martmoney_query.storage import append_trades_csv, write_market_stats_csv


def main() -> None:
    client = DataApiClient()

    leaderboard_users = []
    for item in client.iter_leaderboard(period="ALL", order_by="vol", page_size=100, max_pages=2):
        addr = item.get("proxyWallet") or item.get("address")
        if addr:
            leaderboard_users.append(addr)

    if not leaderboard_users:
        raise SystemExit("未获取到 leaderboard 用户，检查网络或 API 可用性")

    user = leaderboard_users[0]
    start = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=30)

    trades = client.fetch_trades(user=user, start_time=start)
    stats = aggregate_markets(trades, user=user, start_time=start, end_time=None, resolutions={})

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    append_trades_csv(data_dir / "trades_raw.csv", trades)
    write_market_stats_csv(data_dir / "market_stats.csv", stats)

    print(
        f"完成：地址 {user} 的成交明细与市场统计已落盘到 {data_dir.resolve()}/，"
        f"成交数={len(trades)}，市场数={len(stats.markets)}"
    )


if __name__ == "__main__":
    main()
