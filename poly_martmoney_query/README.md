# poly_martmoney_query 使用说明

本目录实现了基于 `smartmoney_query_plan.md` 的「聪明钱」监控脚本，涵盖排行榜扫描、成交抓取、指标聚合与 CSV 持久化。以下说明面向希望复用脚本的同事，示例均使用官方 Data API，尽量减少自定义兜底逻辑。

## 目录结构
- `api_client.py`：Data API 客户端，支持 leaderboard 分页扫描与按时间截断的成交翻页获取。
- `models.py`：Trade、MarketAggregation、AggregatedStats 数据模型与字段规范。
- `processors.py`：交易过滤、市场层聚合与胜率/PNL/成交量统计。
- `storage.py`：CSV 工具，支持成交明细去重追加、市场统计与摘要写盘。
- `__init__.py`：便捷导出。

## 环境依赖
- Python 3.9+，第三方依赖仅包含 `requests`。
- 可通过环境变量调整请求节奏：
  - `SMART_QUERY_MAX_BACKOFF`：指数回退的最大等待秒数，默认 60。
  - `SMART_QUERY_MAX_RPS`：全局限速（每秒请求数），默认 2。

## 快速开始
下面示例演示从排行榜发现用户、抓取成交、按时间窗口聚合并写入 CSV：

```bash
pip install requests
```

```python
import datetime as dt
from pathlib import Path

from poly_martmoney_query.api_client import DataApiClient
from poly_martmoney_query.processors import aggregate_markets
from poly_martmoney_query.storage import append_trades_csv, write_market_stats_csv

client = DataApiClient()

# 1) 扫描 leaderboard（ALL 周期，按成交量排序），取前 200 个用户
leaderboard_users = []
for item in client.iter_leaderboard(period="ALL", order_by="vol", page_size=100, max_pages=2):
    addr = item.get("proxyWallet") or item.get("address")
    if addr:
        leaderboard_users.append(addr)

# 2) 针对单个地址抓取成交，并按时间截断
user = leaderboard_users[0]
start = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=30)
trades = client.fetch_trades(user=user, start_time=start)

# 3) 聚合市场表现（若有结算结果，可在 resolutions 中填入 market_id -> outcome）
stats = aggregate_markets(trades, user=user, start_time=start, end_time=None, resolutions={})

# 4) 写入 CSV：成交明细与市场聚合
append_trades_csv(Path("data/trades_raw.csv"), trades)
write_market_stats_csv(Path("data/market_stats.csv"), stats)
```

## 常见用法
- **按时间窗口重算胜率/PNL**：调用 `aggregate_markets` 时传入 `start_time`/`end_time`，函数会先过滤交易再计算汇总。
- **增量抓取**：`fetch_trades` 内置 offset 翻页，可结合本地最新成交时间作为 `start_time` 做“截断式”更新。
- **调节请求节奏**：无需改代码，设置环境变量即可控制全局限速与回退等待。
- **数据落盘格式**：
  - `append_trades_csv` 以 `tx_hash` 去重写入标准字段，便于后续增量同步。
  - `write_market_stats_csv` 生成市场级明细与 `_summary` 汇总两份文件，便于排名或看板展示。

## 与参考材料的衔接
仓库中的《POLYMARKET_MAKER_REVERSE-原版代码参考材料》提供了构建连接、成交查询、条件筛选等已验证逻辑。本套脚本在请求节奏控制与数据解析上延续了原版做法，可直接替换或嵌入到原有自动化流程中，仅需按上述示例导入对应函数即可。

## 运行自检
如需快速校验代码可运行：

```bash
python -m compileall -q poly_martmoney_query
```

若命令通过则表示当前 Python 环境能成功解析本目录脚本。
