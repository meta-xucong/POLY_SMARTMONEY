# Polymarket Copytrade v1.0 功能设计蓝图（功能优先｜严谨版）

> 目标：尽快实现“能跑起来的跟单”，优质跟单优化放到 v1.1+。  
> 核心原则：**不做逐笔成交复刻**，只做**仓位目标跟踪（Position Targeting）**。

---

## 0. v1.0 定义与边界

### v1.0 必须实现
1. 轮询目标地址 `target` 的 open positions（Data-API `/positions`）。
2. 计算目标仓位跟随目标：`desired_shares = follow_ratio * target_size`。
3. 轮询你自己的 open positions（同样 `/positions`）作为“当前仓位”。
4. 对每个 outcome 执行“仓位差修正下单”：**小单分批、maker 挂单、TTL 到期撤挂重挂**。
5. 支持 `--dry-run` 影子模式：只打印决策、不下单。
6. 崩溃可恢复：`state.json` 持久化（open_orders / token_map / 基本游标）。

### v1.0 不做（明确不在范围内）
- 不做逐笔 trade/activity mirroring（因为拆分成交/部分成交/薄盘口会导致混乱）。
- 不做 “Cycle 第一轮/多轮” 跟随识别（v1.1 再加）。
- 不上 websocket（先轮询稳定）。
- 不做多目标融合（先单目标）。

---

## 1. 总体架构（最小可维护拆分）

建议新增目录：`POLY_SMARTMONEY/copytrade_v1/`

```
copytrade_v1/
  copytrade_run.py            # 主循环入口（轮询 + 对账 + 下单）
  copytrade_config.json       # 配置
  state.json                  # 自动生成，断点恢复

  ct_data.py                  # DataApiClient positions 拉取 + 归一化
  ct_resolver.py              # token_key -> token_id 解析 + cache
  ct_exec.py                  # 下单/撤单/重挂（maker + slice + TTL）
  ct_risk.py                  # 最小风控（3条硬规则）
  ct_state.py                 # state load/save
  ct_utils.py                 # clamp/round_tick/logging
```

---

## 2. 信号源与可信度策略（v1.0 关键）

### 2.1 唯一信号源：Data-API `/positions`
- 复用你现有工程里的 `DataApiClient.fetch_positions(user=..., return_info=True)`。
- 每次拉取 positions 必须拿 `info`：
  - `ok == True` 且 `incomplete == False` 才允许执行“加仓/开仓”动作。
  - 否则进入 **安全模式**：仅允许撤单（尤其撤过期单），**禁止加仓**。

> 这条保证：即使 Data-API 分页 hit 上限 / 返回不完整，也不会乱跟。

### 2.2 为什么 v1.0 不用 trades/activity
- 在你当前筛查链路里，trade_actions 更偏统计用途，并不保证有 side/price/size/token 等能直接复刻的字段；
- 即使有，薄盘口+拆分成交会让逐笔复刻在工程上复杂、在收益上不稳定。

---

## 3. 数据模型（codex 固定 schema，避免字段歧义）

### 3.1 内部主键：token_key
v1.0 内部使用：
- `token_key = "{condition_id}:{outcome_index}"`

### 3.2 归一化 Position（target 与 my 一致）
```python
Pos = {
  "token_key": str,          # condition_id:outcome_index
  "condition_id": str,
  "outcome_index": int,
  "size": float,             # shares
  "avg_price": float,        # 仅日志参考
  "slug": str|None,          # resolver 可用（如果 API 返回）
  "title": str|None,
  "end_date": str|None
}
```

### 3.3 Desired（目标仓位）
```python
Desired = {
  "token_key": str,
  "desired_shares": float    # = follow_ratio * target.size (之后再做 clamp)
}
```

### 3.4 State（断点恢复）
```json
{
  "target": "0x...",
  "my_address": "0x...",
  "follow_ratio": 0.2,
  "token_map": {
    "condition:idx": "token_id"
  },
  "open_orders": {
    "token_id": [
      {"order_id":"...", "side":"BUY", "price":0.631, "size":10, "ts": 1766...}
    ]
  },
  "last_sync_ts": 0
}
```

---

## 4. 配置（copytrade_config.json）建议默认值

```json
{
  "target_address": "0x...",
  "my_address": "0x...",
  "follow_ratio": 0.2,

  "poll_interval_sec": 20,
  "size_threshold": 0.0,

  "deadband_shares": 3.0,

  "slice_min": 5,
  "slice_max": 25,

  "order_ttl_sec": 90,
  "tick_size": 0.001,

  "maker_only": true,

  "max_notional_per_token": 500,
  "max_notional_total": 2000,

  "blacklist_token_keys": []
}
```

说明：
- `deadband_shares`：仓位差小于该值不纠偏（防抖动）。
- `slice_min/slice_max`：分批下单的最小/最大份数。
- `order_ttl_sec`：挂单超时撤挂重挂，提高吃单概率。
- `maker_only=true`：v1.0 强烈建议默认只做 maker（不追价、先跑通）。

---

## 5. 模块接口（codex 按函数签名实现）

### 5.1 ct_data.py（positions 拉取 + 归一化）
```python
def fetch_positions_norm(client, user: str, size_threshold: float) -> tuple[list[dict], dict]:
    # 返回 (positions_norm, info)
    # - positions_norm: list[Pos]（按本蓝图 Pos schema 归一化）
    # - info: fetch_positions(return_info=True) 的 info（至少包含 ok/incomplete）
    ...
```

### 5.2 ct_resolver.py（token_key -> token_id）【v1.0 唯一必做新点】
Positions 里通常没有直接可下单的 token_id（或 clob asset_id），所以必须有 resolver：
```python
def resolve_token_id(token_key: str, pos: dict, cache: dict) -> str:
    # 先查 cache[token_key]，命中直接返回；
    # miss 时调用你现有的“市场信息/代币信息”获取 token_id，然后写入 cache。
    ...
```

> 实现要点：解析一次后写入 state.token_map，后续轮询无需重复解析。

### 5.3 ct_exec.py（执行：每 token 最多 1 活跃单，TTL 重挂）
```python
def reconcile_one(
  token_id: str,
  desired_shares: float,
  my_shares: float,
  orderbook: dict,              # {"best_bid":..., "best_ask":...}
  open_orders: list[dict],       # state.open_orders[token_id]
  now_ts: int,
  cfg: dict
) -> list[dict]:
    # 返回 actions: [{'type':'cancel','order_id':...}, {'type':'place',...}]
    # place action 必须包含: side/price/size/token_id
    ...
```

执行规则（v1.0 固定）：
1. `delta = desired - my`  
2. `abs(delta) <= deadband_shares`：不交易  
3. 取消过期订单：`now - order.ts > order_ttl_sec` → cancel  
4. 每轮最多新挂 1 笔（防止 spam）  
5. `slice = clamp(abs(delta), slice_min, slice_max)`  
6. maker 价格（保证不跨价）：  
   - BUY：挂 `best_bid`（或 `best_ask - tick_size`，再做 tick rounding）  
   - SELL：挂 `best_ask`（或 `best_bid + tick_size`，再做 tick rounding）  
7. 写入 state.open_orders  

**注意**：v1.0 不追“目标均价”；你只追“仓位差”。

### 5.4 ct_risk.py（v1.0 最小风控：3条硬规则）
```python
def risk_check(token_key: str, desired_shares: float, my_shares: float, ref_price: float, cfg: dict) -> tuple[bool, str]:
    # 返回 (ok, reason)
    # 必须检查：
    # 1) blacklist_token_keys
    # 2) max_notional_per_token (粗估：abs(desired_shares) * ref_price)
    # 3) max_notional_total     (粗估：所有 token 的 notional 求和)
    ...
```

`ref_price` 可用：mid=(best_bid+best_ask)/2 或 avg_price（粗估即可）。

### 5.5 ct_state.py（断点恢复）
- `load_state(path) -> dict`
- `save_state(path, state)`
- 初次运行自动创建空 state（含 token_map/open_orders）。

---

## 6. 主循环 copytrade_run.py（严格流程）

### 6.1 运行方式（CLI）
- 实盘：
```bash
python3 copytrade_run.py --target 0x... --ratio 0.2 --poll 20
```
- 影子模式：
```bash
python3 copytrade_run.py --target 0x... --ratio 0.2 --poll 20 --dry-run
```

### 6.2 主循环伪代码（建议 codex 按此写）
```text
load cfg
load state
init DataApiClient
init CLOB client (下单/撤单/盘口/余额/仓位)

while True:
  now_ts = time()

  # A) 拉 target positions（带 info）
  target_pos, target_info = fetch_positions_norm(client, target, size_threshold)
  if not target_info.ok or target_info.incomplete:
      cancel_expired_only()  # 安全模式：只撤过期单
      save state; sleep; continue

  # B) 拉 my positions（带 info）
  my_pos, my_info = fetch_positions_norm(client, my_addr, size_threshold=0)
  if not my_info.ok or my_info.incomplete:
      cancel_expired_only()
      save state; sleep; continue

  # C) desired_by_token_key（只对 target_open_positions）
  desired_by_token_key = {token_key: follow_ratio * pos.size for pos in target_pos}

  # D) 解析 token_id，映射到 desired_by_token_id
  desired_by_token_id = {}
  for pos in target_pos:
      token_id = resolve_token_id(pos.token_key, pos, state.token_map)
      desired_by_token_id[token_id] = clamp(desired_by_token_key[pos.token_key])

  # E) my_by_token_id（同样需要 resolver）
  my_by_token_id = {}
  for pos in my_pos:
      token_id = resolve_token_id(pos.token_key, pos, state.token_map)
      my_by_token_id[token_id] = pos.size

  # F) reconcile_set = union(desired_by_token_id, my_by_token_id, state.open_orders)
  for token_id in reconcile_set:
      desired = desired_by_token_id.get(token_id, 0)
      my = my_by_token_id.get(token_id, 0)

      if abs(desired - my) <= deadband: continue

      ob = get_orderbook(token_id)  # best_bid/best_ask
      ref_price = (ob.best_bid + ob.best_ask)/2

      ok, reason = risk_check(token_key?, desired, my, ref_price, cfg)
      if not ok: continue

      actions = reconcile_one(token_id, desired, my, ob, state.open_orders.get(token_id, []), now_ts, cfg)

      if dry_run:
          print(actions)
      else:
          execute(actions)           # cancel/place
          update state.open_orders   # 写入新的 order_id & ts

  save state
  sleep(poll_interval_sec)
```

建议：`reconcile_set` 还要包含你“已挂单但 desired 已为 0”的 token，这样才能及时撤单/对齐。

---

## 7. v1.0 的三条“防混乱硬规则”（必须保留）
1. **Deadband**：`abs(desired - my) <= deadband` 不交易（薄盘口防抖）。  
2. **TTL 撤挂重挂**：挂单超过 `order_ttl_sec` 未吃到就撤掉重挂（提高成交率）。  
3. **1 token ≤ 1 活跃单**：避免多订单状态爆炸、撤单/部分成交导致混乱。  

---

## 8. v1.1+ 升级方向（不要塞进 v1.0）
- Cycle（只跟第一轮）：用 target_pos 从 0→阈值→回到 0 判定一轮。
- taker fallback：连续 N 次 maker 未成交再允许有限滑点打单。
- websocket：盘口/成交更实时，减少轮询与无意义重挂。
- 多目标融合与冲突净额。

---

## 9. codex 实现提示（避免返工）
- v1.0 信号源固定用 positions（汇总态），不要用 activity/trades 去复刻拆分成交。
- resolver 必须做：positions 没 token_id 就无法下单；解析后写入 cache/state。
- 先跑通“影子模式 + 真实下单（maker-only）”，再考虑优化成交质量与延迟。
