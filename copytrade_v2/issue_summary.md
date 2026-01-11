# copytrade_v2 不完整问题总结与修改清单

## 问题总结

- **positions 拉取不完整的直接触发点**：目标与自身地址在 `positions` 拉取时被标记为 `incomplete=True` 并跳过执行；日志显示请求没有拿到有效 HTTP 响应（`t_http=None`、headers 为空），属于请求失败路径。
- **根因一：HTTP 直连缺少重试/退避**：`fetch_positions_norm` 在 `force_http=True` 时使用 `_fetch_positions_norm_http`，该路径在请求失败时直接 `incomplete=True`，没有使用 `DataApiClient` 中的 backoff 逻辑。
- **根因二：高频 cache bust 放大请求频率**：有新 actions 时强制 `nonce` 模式，会持续打穿缓存；配合 20s 轮询与 25s refresh，容易触发限流或连接异常。
- **根因三：my positions 同样走 HTTP 直连**：`my_positions_force_http=True` 使自身地址也进入无重试路径，导致“双倍失败概率”。
- **相关异常**：open orders 同步失败与 positions 失败同一类型（请求异常），说明整体请求稳定性不足。

## 修改清单（按优先级）

1. **引入重试/退避**
   - 将 `_fetch_positions_norm_http` 接入与 `DataApiClient` 一致的 backoff 逻辑，或复用 `DataApiClient` 的 `session` 与 `_request_with_backoff`。
   - 对于网络错误（超时、连接失败、429/5xx），进行指数退避重试，而不是一次失败即标记 `incomplete`。

2. **降低 cache bust 强度**
   - `has_new_actions` 时不强制 `nonce`，改为 `bucket` 或 `sec`，或在短窗口内仅允许少量 nonce。
   - 将 `target_positions_refresh_sec` 与 `poll_interval_sec` 进行协调（例如 refresh >= poll * 2），避免频繁重复请求。

3. **恢复 my positions 的稳定路径**
   - 将默认 `my_positions_force_http` 设置为 `False`，优先使用 `DataApiClient.fetch_positions` 的带退避实现。

4. **增加错误可观测性**
   - 在 `copytrade_run.py` 输出 `target_info['error_msg']` / `my_info['error_msg']`，以便快速判断是限流、超时还是 payload 错误。

5. **请求负载调优**
   - 在高频场景下考虑减少 `positions_max_pages` 或仅拉取增量（如引入 cursor/last update 时间），降低每轮请求成本。

## 验证要点

- `target_incomplete` / `my_incomplete` 告警数量显著下降。
- `t_http` 能稳定返回状态码（200/304），`cache_headers` 正常出现。
- open orders 同步失败告警减少。
- `planned_notional_zero` 告警次数下降。
