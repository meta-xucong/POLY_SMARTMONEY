# 聪明钱账户查询功能需求与实现思路（简化版）

## 目标
- 支持按自定义时间区间（任意起止日期或近 N 天）查询指定账户的历史表现：胜率、盈亏（PNL）、成交量（VOL）。
- 从排行榜用户池中自动发现候选账户，再按条件筛选并排名。
- 提供简单的查询接口（API/前端可复用）。

## 数据来源与约束
- **Leaderboard**：`GET https://data-api.polymarket.com/v1/leaderboard`，仅支持 DAY/WEEK/MONTH/ALL，适合做用户发现；地址字段以 `proxyWallet` 为准。
- **Trades Data-API**：`GET https://data-api.polymarket.com/trades`，可按 `user` 拉取成交明细但无时间过滤；需要客户端翻页并自行按时间截断。
- **Subgraph (GraphQL)**：适合自定义时间范围、聚合统计与获取市场结算信息，能减少 /trades 全量翻页的开销。

## 核心指标口径
- **胜率（推荐市场级）**：按市场聚合，已结算市场中最终 PNL>0 记为一次胜；未结算市场不计入胜率。
- **PNL**：按市场构建持仓（BUY 增仓、SELL 减仓），结合结算结果计算最终盈亏；如有子图可直接取已实现/最终 PNL 字段。
- **成交量**：以 USDC 口径累加交易规模；可同时记录现金口径与份额口径方便换算。

## 数据流程
1. **候选用户池**：定期扫描 leaderboard（ALL/MONTH，orderBy=VOL 或 PNL，分页到允许的 offset），收集去重的 `proxyWallet`。
2. **交易拉取**：对候选用户调用 `/trades?user=...&limit=100&offset=...` 分页抓取；按时间窗口截断或基于最后一笔交易做增量。
3. **结算补全**：通过 Subgraph 获取市场结算状态与胜出 outcome，供 PNL/胜率计算使用。
4. **指标计算**：按市场聚合成交，得出净持仓、成本、收入、最终 PNL、胜负标记；再汇总到用户-时间窗统计（volume、pnl、win_rate、市场数）。

## 数据存储建议
- `trades_raw(proxy_wallet, condition_id, outcome, side, price, size, timestamp, tx_hash, title/slug, ingested_at)`，索引 `(proxy_wallet, timestamp)`。
- `market_pnl_user(proxy_wallet, condition_id, buy_cost, sell_proceeds, net_shares, final_pnl, win_flag, first_trade_ts, last_trade_ts)`。
- `user_window_stats(proxy_wallet, window_start, window_end, total_volume_usdc, total_pnl_usdc, win_rate_market, markets_resolved_cnt, markets_traded_cnt, updated_at)`。

## 接口与筛选示例
- **单用户查询**：`GET /users/{addr}/stats?start=YYYY-MM-DD&end=YYYY-MM-DD`，返回 volume、pnl、win_rate 等；优先查缓存表，无则触发即时计算或返回“构建中”。
- **排名筛选**：`GET /rank?start=...&end=...&minVol=...&maxVol=...&minWinRate=...&sort=pnl_desc`，从用户池 + `user_window_stats` 过滤并排序，得到符合条件的账户列表。

## 工程化实现要点
- 采集层：并发抓取 leaderboard 与 trades，做好限速与重试；子图客户端用于结算信息与聚合查询。
- 处理层：标准化成交数据，按市场构建持仓并计算 PNL/胜率/成交量。
- 存储层：Postgres 或 DuckDB；定时作业增量更新 trades、每日重算近 90 天窗口。
- API 层：FastAPI/Flask 等轻量接口，前端可直接调用；结果可增加市场明细说明（如 Top 盈利市场）。

## 风险与对策
- **/trades 翻页慢**：优先用 Subgraph 做时间范围查询或先全量同步再做增量。
- **胜率依赖结算信息**：必须从子图或市场接口补齐 resolved + outcome，只用已结算市场计算胜率。
- **地址口径**：系统内部统一用 `proxyWallet`；如需 EOA 映射，另建对应表。
