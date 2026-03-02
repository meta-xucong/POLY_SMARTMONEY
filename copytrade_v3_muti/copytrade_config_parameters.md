# Copytrade v3 Muti 配置参数说明

本文档详细说明 `copytrade_config.json` 中所有可用的配置参数。

## 核心架构

- **多账户支持 (Multi-Account)**: 支持从 `accounts.json` 加载多个跟单账户，按轮询方式依次处理
- **多目标支持 (Multi-Target)**: 支持同时跟踪多个目标地址的持仓，取各 token 的最大持仓作为目标
- **Accumulator 风控**: 独立于 API 的本地累计买入额限制，作为第一道防线

---

## 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_addresses` | array | [] | **目标钱包地址列表** (多目标模式)。支持同时跟踪多个地址的持仓，同一 token 取最大持仓 |
| `target_address` | string | "" | 单一目标地址（向后兼容，优先使用 target_addresses） |
| `accounts_file` | string | "accounts.json" | 账户配置文件路径 |
| `poly_host` | string | "https://clob.polymarket.com" | Polymarket CLOB API 地址 |
| `poly_chain_id` | int | 137 | 链 ID (Polygon 主网为 137) |
| `poly_signature` | int | 2 | 签名类型 |

---

## LowP 防护（低价 token 特殊配置）

当 token 价格低于阈值时，自动切换到低风险参数集。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lowp_guard_enabled` | bool | true | 是否启用 LowP 防护 |
| `lowp_price_threshold` | float | 0.05 | 价格阈值（USD），低于此值触发 LowP 模式 |
| `lowp_follow_ratio_mult` | float | 0.02 | LowP 模式下跟单比例乘数（基础比例 * 此值） |
| `lowp_min_order_usd` | float | 1 | LowP 模式最小订单金额（USD） |
| `lowp_max_order_usd` | float | 2 | LowP 模式最大订单金额（USD） |
| `lowp_probe_order_usd` | float | 1 | LowP 模式探针订单金额 |
| `lowp_max_notional_per_token` | float | 2 | LowP 模式每 token 最大名义价值 |

---

## 轮询与刷新

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `poll_interval_sec` | int | 20 | 正常轮询间隔（秒） |
| `poll_interval_sec_exiting` | int | 3 | 退出/减仓时的轮询间隔（秒），更快响应 |
| `config_reload_sec` | int | 600 | 配置热重载检查间隔（秒） |
| `target_positions_refresh_sec` | int | 25 | 目标持仓缓存刷新间隔（秒） |
| `heartbeat_interval_sec` | int | 600 | 心跳日志输出间隔（秒） |
| `api_timeout_sec` | float | 15.0 | API 调用超时时间（秒） |

---

## 订单管理

### 订单大小模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_size_mode` | string | "auto_usd" | 订单大小模式：`auto_usd` 自动计算 / `fixed_shares` 固定股数 |
| `min_order_usd` | float | 1 | 最小订单金额（USD） |
| `max_order_usd` | float | 6 | 最大订单金额（USD） |
| `min_order_shares` | float | 5 | 最小订单股数 |
| `max_order_shares_cap` | float | 5000 | 单次订单最大股数限制 |
| `deadband_shares` | float | 1.0 | 死区股数，低于此值不调整 |
| `slice_min` | float | 5 | 分批下单最小股数 |
| `slice_max` | float | 25 | 分批下单最大股数 |
| `tick_size` | float | 0 | 价格最小变动单位（0=自动从市场获取） |
| `min_price` | float | 0.01 | 最低下单价格 |

### Maker/Taker 设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `maker_only` | bool | true | 仅使用 Maker 订单（吃单） |
| `taker_enabled` | bool | true | 是否允许 Taker 订单（挂单） |
| `taker_spread_threshold` | float | 0.011 | 触发 Taker 的价差阈值 |
| `taker_order_type` | string | "FAK" | Taker 订单类型：`FAK` (Fill and Kill) / `FOK` (Fill or Kill) |
| `maker_max_wait_sec` | int | 0 | Maker 订单最长等待时间（秒），超时自动转 Taker |
| `maker_to_taker_enabled` | bool | true | 是否启用 Maker 转 Taker 机制 |

### 重新定价 (Reprice)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_reprice` | bool | true | 是否启用自动重新定价 |
| `reprice_ticks` | int | 1 | 价格偏离多少 tick 时触发 reprice |
| `reprice_cooldown_sec` | int | 3 | 重新定价冷却时间（秒） |

---

## 风险控制

### 持仓限制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_position_usd_per_token` | float | 10 | 每 token 最大持仓价值（USD），0=无限制 |
| `max_notional_per_token` | float | 0 | 每 token 最大名义价值（USD），0=无限制 |
| `max_notional_total` | float | 0 | 总持仓名义价值上限（USD），0=无限制 |
| `allow_short` | bool | false | 是否允许做空 |

### Accumulator 独立风控

Accumulator 是独立于 API 的第一道防线，用于防止 API 数据同步延迟导致的超仓。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `accumulator_max_total_usd` | float | 500 | Accumulator 总买入上限（USD），作为硬限制 |
| `accumulator_stale_reset_sec` | int | 7200 | Accumulator 条目过期重置时间（秒），持仓为空且过期后自动清零 |
| `shadow_buy_ttl_sec` | int | 900 | Shadow buy 订单存活时间（秒） |

### 买入窗口限制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `buy_window_sec` | int | 120 | 买入窗口时间（秒） |
| `buy_window_max_usd_per_token` | float | 8 | 窗口期内每 token 最大买入金额 |
| `buy_window_max_usd_total` | float | 20 | 窗口期内总买入金额上限 |

---

## 启动同步 (Boot Sync)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `boot_sync_mode` | string | "baseline_only" | 启动同步模式：<br>- `baseline_only`: 仅同步当前持仓<br>- `baseline_replay`: 同步持仓 + 回放历史交易<br>- `replay_24h`: 回放最近 24 小时<br>- `replay_actions`: 回放历史操作 |
| `fresh_boot_on_start` | bool | false | 每次启动是否进行全新 bootstrap |
| `ignore_boot_tokens` | bool | true | 是否忽略启动时的已有 token |
| `ignore_boot_tokens_scope` | string | "probe_only" | 忽略范围：`probe_only` 仅阻止探针 / `all` 阻止所有买入 |
| `probe_order_usd` | float | 1 | 首次发现 token 时的探针订单金额 |
| `probe_buy_on_first_seen` | bool | true | 是否在新 token 首次发现时下探针单 |
| `follow_new_topics_only` | bool | false | 仅跟随新的 topic（忽略所有启动前 token） |
| `adopt_existing_orders_on_boot` | bool | true | 启动时接管已存在的订单 |
| `actions_replay_window_sec` | int | 86400 | 交易回放窗口（秒），默认 24 小时 |

---

## 缺失数据处理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `missing_timeout_sec` | int | 600 | 目标持仓缺失超时时间（秒） |
| `missing_to_zero_rounds` | int | 0 | 连续多少轮缺失后视为清仓（0=禁用） |
| `missing_freeze_streak` | int | 5 | 触发缺失冻结的连续缺失轮数 |
| `missing_freeze_min_sec` | int | 600 | 缺失冻结最短时长（秒） |
| `missing_freeze_max_sec` | int | 1800 | 缺失冻结最长时长（秒） |

---

## 订单可见性与同步

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_visibility_grace_sec` | int | 180 | 订单在远程不可见的宽限期（秒） |
| `orphan_cancel_rounds` | int | 3 | 孤儿订单取消轮数阈值 |
| `orphan_ignore_sec` | int | 120 | 孤儿订单取消后忽略时间（秒） |
| `dedupe_place` | bool | true | 是否启用下单去重 |
| `dedupe_place_price_eps` | float | 1e-6 | 去重价格精度 |
| `dedupe_place_size_rel_eps` | float | 1e-6 | 去重大小相对精度 |

---

## 卖出确认机制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sell_confirm_max` | int | 5 | 最大卖出确认次数 |
| `sell_confirm_window_sec` | int | 300 | 卖出确认窗口时间（秒） |
| `sell_confirm_force_ratio` | float | 0.5 | 强制卖出触发比例（目标减持比例） |
| `sell_confirm_force_shares` | float | 0.0 | 强制卖出触发股数阈值 |

---

## 动作 (Action) 数据获取

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actions_source` | string | "trades" | 动作数据源：`trades` 交易记录 / `activity` 活动流 |
| `actions_page_size` | int | 300 | 每页获取动作数量 |
| `actions_max_offset` | int | 10000 | 最大偏移量（上限 3000） |
| `actions_taker_only` | bool | false | 仅获取 Taker 交易 |
| `actions_lag_threshold_sec` | int | 180 | 动作延迟告警阈值（秒） |
| `actions_unreliable_hold_sec` | int | 120 | 动作数据不可靠时冻结买入时长（秒） |
| `lag_replay_window_sec` | int | 120 | 延迟触发回放的窗口（秒） |
| `lag_replay_cooldown_sec` | int | 120 | 延迟回放冷却时间（秒） |
| `seen_action_ids_cap` | int | 5000 | 已见动作 ID 缓存上限 |
| `seen_action_ids_cap_replay` | int | 50000 | 回放模式下已见动作 ID 缓存上限 |

---

## Token 解析与映射

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_resolve_target_positions_per_loop` | int | 20 | 每轮最大解析目标持仓数量 |
| `max_resolve_actions_per_loop` | int | 20 | 每轮最大解析动作数量 |
| `max_resolve_actions_on_missing` | int | 60 | 缺失率高时 boosted 解析上限 |
| `max_resolve_trades_per_loop` | int | 10 | 每轮最大解析交易数量 |
| `resolver_fail_cooldown_sec` | int | 300 | 解析失败冷却时间（秒） |
| `resolve_actions_missing_ratio` | float | 0.3 | 触发 boosted 解析的缺失率阈值 |

---

## 市场状态与过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_closed_markets` | bool | true | 是否跳过已关闭市场 |
| `block_on_unknown_market_state` | bool | false | 市场状态未知时是否阻塞交易 |
| `closed_ignore_min_ttl_sec` | int | 86400 | 关闭市场忽略最小时长（秒） |
| `market_status_refresh_sec` | int | 300 | 市场状态刷新间隔（秒） |
| `orderbook_empty_close_streak` | int | 3 | 订单簿为空多少轮后关闭 token |

---

## 头寸与日志

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `positions_limit` | int | 500 | 每次获取持仓数量限制 |
| `positions_max_pages` | int | 20 | 获取持仓最大页数 |
| `my_positions_force_http` | bool | false | 强制使用 HTTP 获取自身持仓 |
| `target_cache_bust_mode` | string | "bucket" | 目标持仓缓存破坏模式：`bucket` / `sec` / `ms` / `nonce` |
| `log_positions_cache_headers` | bool | false | 是否记录持仓缓存头部信息 |
| `positions_cache_header_keys` | array | ["Age", "CF-Cache-Status", "X-Cache", "Via", "Cache-Control"] | 缓存头部字段 |

---

## 日志配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_dir` | string | "logs" | 日志目录 |
| `log_level` | string | "INFO" | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `log_retention_days` | int | 7 | 日志保留天数 |
| `log_cleanup_hour` | int | 12 | 日志清理执行小时（UTC） |
| `log_dedup_window_sec` | float | 300.0 | 重复日志抑制窗口（秒） |

---

## Topic 生命周期管理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topic_cycle_mode` | bool | true | 是否启用 topic 生命周期模式 |
| `topic_entry_settle_sec` | int | 60 | 建仓结算期（秒） |
| `exit_full_sell` | bool | true | 退出时是否全额卖出 |
| `exit_ignore_cooldown` | bool | true | 退出时是否忽略冷却 |
| `cooldown_sec_per_token` | int | 20 | 每 token 操作冷却时间（秒） |
| `dust_exit_eps` | float | 0.0 |  dust 退出阈值（股数） |

---

## 黑名单

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `blacklist_token_keys` | array | [] | 黑名单 token key 列表（支持部分匹配） |

---

## 重试与退避

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retry_on_insufficient_balance` | bool | false | 余额不足时是否重试 |
| `retry_shrink_factor` | float | 0.5 | 重试时订单金额缩减因子 |
| `place_fail_backoff_base_sec` | float | 2 | 下单失败退避基础时间（秒） |
| `place_fail_backoff_cap_sec` | float | 60 | 下单失败退避上限（秒） |

---

## 杂项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `allow_partial` | bool | true | 是否允许部分成交 |
| `delta_eps` | float | 1e-9 | 偏差比较精度 |
| `debug_token_ids` | array | [] | 调试追踪的 token ID 列表 |
| `risk_summary_interval_sec` | int | 120 | 风险摘要输出间隔（秒） |
| `missing_mid_fallback_price` | float | 1.0 | 缺失中间价时的回退价格 |
| `size_threshold` | float | 0 | 持仓大小过滤阈值 |

---

## 环境变量

以下配置也可通过环境变量设置（优先级高于配置文件）：

| 环境变量 | 说明 |
|----------|------|
| `POLY_HOST` | CLOB API 地址 |
| `POLY_CHAIN_ID` | 链 ID |
| `POLY_SIGNATURE` | 签名类型 |
| `COPYTRADE_TARGET` / `CT_TARGET` / `POLY_TARGET_ADDRESS` / `TARGET_ADDRESS` | 目标地址 |

---

## 账户配置 (accounts.json)

每个账户支持以下配置：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | 必填 | 账户名称/标识 |
| `my_address` | string | 必填 | 钱包地址（0x...） |
| `private_key` | string | 必填 | 私钥（建议从环境变量读取） |
| `follow_ratio` | float | 0.05 | 该账户的跟单比例 |
| `enabled` | bool | true | 是否启用该账户 |
| `max_notional_per_token` | float | null | 该账户每 token 限制（覆盖全局） |
| `max_notional_total` | float | null | 该账户总持仓限制（覆盖全局） |

---

## 配置示例

```json
{
  "target_addresses": ["0xaDd407264226afaD5536752765C6F98b2a8a51Da"],
  "poly_host": "https://clob.polymarket.com",
  "poly_chain_id": 137,
  
  "lowp_guard_enabled": true,
  "lowp_price_threshold": 0.05,
  "lowp_follow_ratio_mult": 0.02,
  
  "poll_interval_sec": 20,
  "poll_interval_sec_exiting": 3,
  
  "order_size_mode": "auto_usd",
  "min_order_usd": 1,
  "max_order_usd": 6,
  "max_position_usd_per_token": 10,
  
  "accumulator_max_total_usd": 500,
  
  "maker_only": true,
  "taker_enabled": true,
  "taker_spread_threshold": 0.011,
  
  "enable_reprice": true,
  "reprice_ticks": 1,
  
  "boot_sync_mode": "baseline_only",
  "actions_replay_window_sec": 86400,
  "follow_new_topics_only": false,
  
  "blacklist_token_keys": ["Bitcoin"]
}
```

---

## 注意事项

1. **Accumulator 风控优先**: `accumulator_max_total_usd` 是硬限制，即使 API 数据延迟也能防止超仓
2. **多账户延迟**: N 个账户时，每个账户实际轮询间隔约为 `N * poll_interval_sec`
3. **LowP 模式**: 低价 token 风险高，建议使用更低的仓位限制
4. **状态文件**: 每个账户有独立的状态文件，勿手动删除
5. **日志轮转**: 日志按天轮转，保留 `log_retention_days` 天
