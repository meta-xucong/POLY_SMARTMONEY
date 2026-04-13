# Copytrade v1 使用说明

本目录提供基于 `copytrade_v1_blueprint.md` 的最小可运行版本，实现“仓位目标跟踪（Position Targeting）”的跟单脚本。

> 重点：v1.0 仅跟踪目标仓位差，不做逐笔成交复刻；默认 maker-only 挂单，支持买一/卖一移动时撤单重挂。

## 1. 依赖与环境

- Python 3.9+
- 已安装 `py_clob_client`（用于下单）与 `requests`
- 需要可用的 Polymarket API 凭证

### 必需环境变量

- `POLY_KEY`：私钥（hex 字符串，0x 前缀可选）
- `POLY_FUNDER`：代理钱包 / 充值地址

### 可选环境变量

- `POLY_HOST`：默认 `https://clob.polymarket.com`
- `POLY_CHAIN_ID`：默认 `137`
- `POLY_SIGNATURE`：默认 `2`

## 2. 配置文件

编辑 `copytrade_config.json`：

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
  "tick_size": 0.001,
  "maker_only": true,
  "reprice_ticks": 1,
  "reprice_cooldown_sec": 15,
  "max_notional_per_token": 500,
  "max_notional_total": 2000,
  "blacklist_token_keys": []
}
```

配置含义说明：

- `target_address`：被跟单的地址
- `my_address`：你的地址
- `follow_ratio`：跟随比例（目标仓位 * 比例）
- `poll_interval_sec`：轮询间隔（秒）
- `size_threshold`：positions 拉取过滤阈值
- `deadband_shares`：仓位差小于该值不纠偏
- `slice_min/slice_max`：分批下单最小/最大份数
- `tick_size`：价格最小跳动
- `maker_only`：v1.0 默认只做 maker
- `reprice_ticks`：买一/卖一移动触发撤单重挂的最小 tick 数
- `reprice_cooldown_sec`：追价撤单的冷却时间（秒）
- `max_notional_per_token`：单 token 最大名义金额
- `max_notional_total`：总名义金额上限
- `blacklist_token_keys`：黑名单（格式 `condition_id:outcome_index`）

## 3. 运行方式

进入目录后执行：

```bash
cd copytrade_v1
python3 copytrade_run.py
```

### 覆盖配置（可选）

```bash
python3 copytrade_run.py \
  --target 0xTARGET \
  --my 0xMYADDR \
  --ratio 0.2 \
  --poll 20
```

### 影子模式（不下单）

```bash
python3 copytrade_run.py --dry-run
```

## 4. 状态文件

脚本会在同目录生成 `state.json`，用于断点恢复与缓存 token 映射：

- `token_map`：`token_key -> token_id` 解析缓存
- `open_orders`：当前挂单列表（单 token 最多 1 笔）
- `last_sync_ts`：最后同步时间

> 如需重置运行状态，可删除 `state.json` 后重新启动。

## 5. 运行逻辑简述

1. 轮询 `target` 和 `my` 的 open positions（Data-API `/positions`）
2. 计算目标仓位：`desired_shares = follow_ratio * target_size`
3. 解析 `token_id`（基于 `condition_id`/`slug` 走 gamma-api）
4. 对每个 token 执行 `reconcile_one`：
   - deadband 不交易
   - 买一/卖一移动触发撤单重挂
   - 同侧仅保留 1 笔挂单
   - maker-only 挂单
5. 持久化状态并进入下一轮

## 6. 常见问题

- **Q: 为什么只用 positions 作为信号源？**
  - A: v1.0 仅做仓位目标跟踪，positions 是稳定的汇总态数据源。

- **Q: resolver 失败怎么办？**
  - A: 确认 `condition_id`/`slug` 是否正确，或检查网络能否访问 gamma-api。

- **Q: 为什么只做 maker，不吃单？**
  - A: v1.0 先保证稳定性与可控性，taker 逻辑留待 v1.1+。
