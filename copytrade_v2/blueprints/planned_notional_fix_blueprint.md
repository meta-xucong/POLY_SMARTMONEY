# Copytrade v2 风控失效修复蓝图（planned=0）

> 目标：在不牺牲性能的前提下，恢复单 token 风控上限的可靠性；先修复 planned=0 的根因，再加保守兜底与故障提示。

## 背景与问题概述
- 现象：单 token 上限失效，持续买入直至耗尽资金。
- 核心原因：d07cee4 后风控基数完全依赖 planned notional，累计兜底被移除；而 planned 在多种场景下被低估为 0。
- 关键触发链路：
  1) target positions 无法解析 token_id（POSMAP pending=206）；
  2) t_now 缺失导致盘口不拉取 / mid 不更新；
  3) `_calc_used_notional_totals` 在 mid 缺失时按 0 估值 → planned 恒为 0；
  4) 风控基数为 0 → `max_notional_per_token` 不触发。

## 方案总览
1) **恢复 target token_id 解析能力**：在 target positions loop 中，当 token_map 与 raw 都无法提供 token_id 时，启用 resolver（带预算/限流）。
2) **修复 planned=0 的主因**：确保 mid_cache 在必要时可被更新；在 token_id 恢复后，planned 应能正常增长。
3) **planned=0 的兜底与告警**：当计划值异常为 0 且持仓不为空时，启用保守估值并输出明确故障提示。
4) **补充诊断链路**：actions/trades 侧 token_id 缺失采样、orderbook 空的告警聚合。

---

## 详细改动计划

### A. 解析层：positions raw 字段缺失与 token_id 解析
**问题**：positions raw 中缺乏 tokenId/clobTokenId 等字段，fast-path 无法解析 token_id。

**修改思路**：
- 在 target positions 构建时，当 `token_map` 和 `_extract_token_id_from_raw` 失败时调用 `resolve_token_id`。
- 引入预算/限流以避免大账户阻塞（例如 `max_resolve_target_positions_per_loop`）。
- 成功解析后写回 `token_map`，形成持久缓存。

**实施要点**：
- 位置：`copytrade_v2/copytrade_run.py` target positions loop。
- 增加日志统计：resolved_by_cache/raw/resolver/pending。


### B. planned notional 主修复
**问题**：planned=0 的主因是 token_id 解析失败 + mid 缺失。

**修改思路**：
- 先解决 token_id 解析（A 方案）。
- 保证拉到盘口后写入 `last_mid_price_by_token_id`，让 planned 基于真实 mid 估值。
- 如 orderbook 经常为空，考虑使用 `avg_price` 或上次 mid 作为临时 fallback。


### C. planned=0 的保守兜底与故障提示
**问题**：planned=0 时风控完全失效。

**修改思路**：
- 当 `mid<=0` 且 `shares>0` 时，保守估值（默认 1.0，或新配置项 `missing_mid_fallback_price`）。
- 若 `planned_total_notional==0` 且 `my_by_token_id` 非空，输出高优先级告警并记录连续次数。

**示例告警**：
```
[ALERT] planned_notional_zero my_positions=30 reason=missing_mid_or_token_map
```


### D. 诊断补强（不可见问题补齐）
1) **actions/trades token_id 缺失采样**
   - 当 action/trade raw 解析失败，输出 raw key 采样。
   - 高比例缺失时，触发 resolver + 缓存写回。

2) **orderbook 空告警聚合**
   - 对相同 token 连续出现 `orderbook_empty` 进行聚合告警。
   - 若超过阈值，提示数据源或市场不可交易。

---

## 风险与影响评估
- **性能风险**：恢复 target resolver 会增加 Gamma API 调用。
  - 缓解：加入 per-loop 预算与缓存写回，避免全量解析。
- **误报风险**：planned=0 兜底估值过保守可能影响交易效率。
  - 缓解：仅在 `shares>0` 且 `mid<=0` 时触发，并加入可配置项。

---

## 验证建议
- 观察 `POSMAP pending` 是否下降；token_map 是否持续增长。
- `RISK_SUMMARY used_total` 是否恢复为非零。
- 触发 `planned_notional_zero` 告警次数是否下降。
- 在大账户场景下，验证 resolver 预算是否避免阻塞。

---

## 推荐改动路径（执行顺序）
1) 恢复 target resolver（带预算） → 解决 token_id 解析失败。
2) 确认 planned 正常增长（RISK_SUMMARY 非 0）。
3) 增加 planned=0 兜底 + 告警。
4) 补充 actions/trades 与 orderbook 空诊断。

