# Magic vs Kings 跟单异常分析（2026-02-18）

## 结论

1. **“10U目标单只跟到约5U”是风控上限导致的，不是下单失败。**
   - 该 token 目标仓位信号为 `d_target=40` shares，按价格 `0.25` 约等于 **10 USD**。
   - 跟单程序在该 token 上持续出现 `NOOP reason=max_position_usd_per_token`，说明触发了单 token 最大仓位限额，后续被拦截。

2. **“有的账户没有探针单，只出现两次 10 shares”是账户启动时序 + 轮询切换导致的正常现象。**
   - 日志显示该轮运行从 6 账户切换为 5 账户（重启后少了 `maker_test`）。
   - 在目标单出现时，不同账户进入该 token 的状态不同：
     - 部分账户先经历了 `TOPIC PROBE` 再下单。
     - 部分账户直接进入 `LONG`/`ACTION`，因此只看到两次 10 shares。

3. **该 token 实际下单切片是 10 shares/笔（2.5U/笔），并非整单一次打满。**
   - 多个账户对同一 token 连续出现 `size=10.0` 的 ACTION；
   - 累计美元从 `2.5` 增到 `6.0204`（含探针 + 正式单），或只到 `5.0`（两笔 10 shares，无探针）。

## 关键日志证据（便于复盘）

- 目标仓位信号和探针：
  - `SIGNAL BUY ... d_target=40.0`
  - `TOPIC PROBE ... target=4.0816`
  - 随后 `ACTION ... size=10.0`
- 风控拦截：
  - `NOOP ... reason=max_position_usd_per_token`
- 账户数量变化：
  - 18:06 初始化 6 个账户；
  - 19:54 重启后仅加载 5 个账户并开始轮询。

## 建议

1. 若希望单账户可完整跟到 10U，请提高 `max_position_usd_per_token`（并同步检查 `max_order_usd`、账户级 `max_notional_per_token`）。
2. 若希望“每个账户都一定先探针再加仓”，需要统一首见判定（避免重启/轮询时有的账户直接继承为 LONG）。
3. 建议在日志中增加“配置快照输出”（当前生效的关键限额），便于排查“本地配置与运行配置不一致”。
