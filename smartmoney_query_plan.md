# 聪明钱账户查询功能需求与实现思路（简化版）

## 目标
- 支持按自定义时间区间（任意起止日期或近 N 天）查询指定账户的历史表现：胜率、盈亏（PNL）、成交量（VOL）。
- 从排行榜用户池中自动发现候选账户，再按条件筛选并排名。
- 提供简单的查询接口（API/前端可复用）。

## 数据来源与约束
- **Leaderboard（只做用户发现）**：`GET https://data-api.polymarket.com/v1/leaderboard`，仅支持 DAY/WEEK/MONTH/ALL，用于挖掘候选账户；地址字段以 `proxyWallet` 为准。
- **Trades Data-API（核心数据抓取）**：`GET https://data-api.polymarket.com/trades`，按 `user` 拉成交明细但无时间过滤；需要客户端翻页并自行按时间截断。后续要做“定期监控 + 及时跟单”时，可直接复用同一套拉取逻辑。
- **Subgraph (GraphQL，选配)**：可用于补充市场结算信息，减少 /trades 翻页开销；如想极简，可暂时不接入，后续再加。

## 核心指标口径
- **胜率（推荐市场级）**：按市场聚合，已结算市场中最终 PNL>0 记为一次胜；未结算市场不计入胜率。
- **PNL**：按市场构建持仓（BUY 增仓、SELL 减仓），结合结算结果计算最终盈亏；如有子图可直接取已实现/最终 PNL 字段。
- **成交量**：以 USDC 口径累加交易规模；可同时记录现金口径与份额口径方便换算。

## 数据流程
1. **候选用户池**：定期扫描 leaderboard（ALL/MONTH，orderBy=VOL 或 PNL，分页到允许的 offset），只拿 `proxyWallet` 作为监控/查询对象。
2. **交易拉取（核心）**：对候选用户调用 `/trades?user=...&limit=100&offset=...` 分页抓取；按时间窗口截断或基于最后一笔交易做增量，为未来的“持续监控 + 跟单”打好复用基础。
3. **结算补全（选配）**：若需要胜率与最终 PNL，可通过 Subgraph 获取市场结算状态与胜出 outcome。
4. **指标计算**：按市场聚合成交，得出净持仓、成本、收入、最终 PNL、胜负标记；再汇总到用户-时间窗统计（volume、pnl、win_rate、市场数）。

## 数据存储建议（极简，无数据库）
- **CSV/Parquet 文件即可**：按功能拆分小文件，方便追加/更新：
  - `users.csv`：从 leaderboard 去重得到的 `proxyWallet` 列表，附上抓取时间。
  - `trades_raw_{user}.csv` 或分区文件：分页写入交易明细，新增时直接 append，遇到重复 `tx_hash` 去重即可。
  - `market_stats_{user}.csv`：按市场聚合后的 PNL/胜率/成交量等结果，用于查询和后续跟单策略分析。
- **定期更新策略**：用简单的定时脚本（如 cron + Python）拉增量 trades，写入 CSV 并重算最近 N 天的窗口；后续若数据量增大再考虑迁移到数据库。

## 接口与筛选示例
- **单用户查询**：`GET /users/{addr}/stats?start=YYYY-MM-DD&end=YYYY-MM-DD`，返回 volume、pnl、win_rate 等；可先读本地聚合 CSV/Parquet 缓存，无则即时计算或提示“构建中”。
- **排名筛选**：`GET /rank?start=...&end=...&minVol=...&maxVol=...&minWinRate=...&sort=pnl_desc`，从 `users.csv` 取用户池，再读取对应用户的聚合统计文件进行过滤与排序。

## 工程化实现要点
- 采集层：并发抓取 leaderboard 与 trades，做好限速与重试；子图客户端仅在需要结算信息时调用。
- 处理层：标准化成交数据，按市场构建持仓并计算 PNL/胜率/成交量。
- 存储层：使用 CSV/Parquet 本地文件，按用户或时间分片；定时作业增量更新 trades 并重算近 90 天窗口。
- API 层：FastAPI/Flask 等轻量接口，直接读取聚合文件返回结果；可附带市场明细用于解释策略来源。

## 风险与对策
- **/trades 翻页慢**：如需要可引入 Subgraph 做时间范围查询，或先全量同步再做增量。
- **胜率依赖结算信息**：必须从子图或市场接口补齐 resolved + outcome，只用已结算市场计算胜率。
- **地址口径**：系统内部统一用 `proxyWallet`；如需 EOA 映射，另建对应表。
