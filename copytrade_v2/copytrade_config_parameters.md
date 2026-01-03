# copytrade_config.json 参数说明

本文档解释 `copytrade_v2/copytrade_config.json` 中每个参数的作用。

## 账户与跟随比例

- `target_address`: 被跟随的目标地址。
- `my_address`: 跟随者自身地址。
- `follow_ratio`: 跟随比例（按目标仓位/下单规模的比例进行跟随）。

## 低价市场（Low P）保护

- `lowp_guard_enabled`: 是否开启低价市场保护逻辑。
- `lowp_price_threshold`: 低价阈值（价格低于该值视为低价市场）。
- `lowp_follow_ratio_mult`: 低价市场时的跟随比例倍数（用于降低跟随规模）。

- `lowp_min_order_usd`: 低价市场最小下单金额（USD）。
- `lowp_max_order_usd`: 低价市场最大下单金额（USD）。
- `lowp_probe_order_usd`: 低价市场探测单金额（USD）。
- `lowp_max_notional_per_token`: 低价市场单个 token 最大名义金额（USD）。

## 轮询与缓存

- `poll_interval_sec`: 正常状态轮询间隔（秒）。
- `poll_interval_sec_exiting`: 退出/减仓状态轮询间隔（秒）。
- `config_reload_sec`: 配置文件热加载间隔（秒）。
- `size_threshold`: 规模阈值，小于该值的变动可忽略。
- `target_positions_refresh_sec`: 目标仓位刷新间隔（秒）。
- `log_positions_cache_headers`: 是否记录目标仓位请求的缓存响应头。
- `positions_cache_header_keys`: 需要记录的缓存响应头字段列表。
- `target_cache_bust_mode`: 目标仓位请求的缓存绕过模式（如 `sec`）。
- `my_positions_force_http`: 是否强制以 HTTP 获取自身仓位（默认使用 HTTPS）。

## 仓位同步与死区

- `deadband_shares`: 死区阈值（份额差异小于该值时不调整）。

## 下单切片与规模控制

- `slice_min`: 单次切片最小份额。
- `slice_max`: 单次切片最大份额。
- `order_size_mode`: 下单规模模式（如 `auto_usd`）。
- `min_order_usd`: 最小下单金额（USD）。
- `min_order_shares`: 最小下单份额。
- `dust_exit_eps`: 视为尾仓/粉尘仓位的阈值。
- `max_order_usd`: 单笔最大下单金额（USD）。
- `max_position_usd_per_token`: 单个 token 最大持仓金额（USD）。
- `boot_sync_mode`: 启动同步模式（如 `baseline_only`）。
- `fresh_boot_on_start`: 启动时是否重新进行首次同步。
- `ignore_boot_tokens`: 启动同步时是否忽略 token。
- `ignore_boot_tokens_scope`: 忽略 token 的范围（如 `probe_only`）。
- `probe_buy_on_first_seen`: 首次看到 token 是否下探测买单。
- `probe_order_usd`: 探测单金额（USD）。
- `follow_new_topics_only`: 是否只跟随新话题/新市场。
- `adopt_existing_orders_on_boot`: 启动时是否接管已有挂单。

## 价格与撮合

- `tick_size`: 价格最小变动单位。

- `maker_only`: 是否仅挂 maker 单。
- `reprice_ticks`: 改价偏移的 tick 数。
- `reprice_cooldown_sec`: 改价冷却时间（秒）。
- `enable_reprice`: 是否启用自动改价。
- `exit_full_sell`: 退出时是否全量卖出。
- `exit_ignore_cooldown`: 退出时是否忽略冷却时间。

## 风控与容错

- `cooldown_sec_per_token`: 每个 token 的冷却时间（秒）。
- `missing_timeout_sec`: 目标仓位缺失超时时间（秒）。
- `missing_to_zero_rounds`: 目标仓位缺失达到次数后是否视为零仓位。
- `debug_token_ids`: 调试用 token id 列表。
- `log_dir`: 日志目录。
- `log_level`: 日志级别（如 `INFO`）。
- `skip_closed_markets`: 是否跳过已关闭市场。
- `block_on_unknown_market_state`: 是否在未知市场状态时阻塞操作。

## 额度上限

- `max_notional_per_token`: 单个 token 最大名义金额（USD）。
- `max_notional_total`: 总名义金额上限（USD）。

## 下单失败重试与去重

- `allow_partial`: 是否允许部分成交。
- `taker_enabled`: 是否启用 taker（吃单）逻辑。
- `taker_spread_threshold`: 触发 taker 的价差阈值。
- `taker_order_type`: taker 下单类型（如 `FAK`）。
- `dedupe_place`: 是否对重复下单请求进行去重。
- `dedupe_place_price_eps`: 价格去重容差。
- `dedupe_place_size_rel_eps`: 份额去重的相对容差。
- `order_visibility_grace_sec`: 订单可见的宽限时间（秒），用于等待挂单出现在订单列表。
- `retry_on_insufficient_balance`: 余额不足时是否重试。
- `retry_shrink_factor`: 重试时缩小下单规模的系数。
- `place_fail_backoff_base_sec`: 下单失败退避基准时间（秒）。
- `place_fail_backoff_cap_sec`: 下单失败退避上限时间（秒）。
- `place_fail_backoff_sec`: 下单失败固定退避时间（秒）。

## 黑名单

- `blacklist_token_keys`: 黑名单 token 关键字列表。
