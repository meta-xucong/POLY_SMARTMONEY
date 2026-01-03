# 单个账户数据分析脚本使用说明

本目录包含 `pm_target_capture.py`，用于抓取 Polymarket 单个账户的 Activity/Trades 数据并做 maker/taker 推断与补腿配对分析。

## 运行前准备

- Python 3.9+（建议使用虚拟环境）
- 依赖：`requests`（可选 `pandas` 用于导出 parquet）

示例安装：

```bash
pip install requests pandas
```

> 没安装 `pandas` 也能运行，只是不会导出 parquet 文件。

## 主要功能

- 解析 `@handle` → `proxyWallet`（Gamma public-search / public-profile）
- 分段抓取 Activity（/activity，带 start/end）
- 拉取 Trades（/trades，takerOnly=true/false）
- 合并去重并推断 maker/taker
- 计算补腿配对率与延迟统计

## 输出目录结构（示例）

```
account_analysis/
  pm_target_capture.py
  README.md

data/RN1/
  meta/profile.json
  raw/activity_20260101.jsonl
  raw/activity_20260102.jsonl
  raw/trades_takerOnly_true.jsonl
  raw/trades_takerOnly_false.jsonl
  raw/activity_live.jsonl
  derived/trades_enriched.jsonl
  derived/trades_enriched.parquet
  derived/legs_pairs.jsonl
  derived/legs_pairs.parquet
  derived/summary.json
```

## 回溯抓取（backfill）

```bash
python account_analysis/pm_target_capture.py \
  --target @RN1 \
  --start 2025-12-01 \
  --end 2026-01-03 \
  --out data \
  --mode backfill
```

常用参数：

- `--window-hours`：Activity 时间窗口（默认 6 小时）
- `--page-size`：分页大小（默认 500）
- `--max-pages`：/trades 最大页数（默认 200）
- `--dt-list`：补腿配对阈值列表（默认 `5 10 30` 秒）

## 实时采集（live）

```bash
python account_analysis/pm_target_capture.py \
  --target @RN1 \
  --out data \
  --mode live \
  --poll 3
```

说明：

- live 会不断抓取最新的 Activity 记录并追加到 `raw/activity_live.jsonl`。
- 结束请按 `Ctrl+C`。

## 环境变量（可选）

- `POLY_DATA_API_ROOT`：默认 `https://data-api.polymarket.com`
- `POLY_GAMMA_ROOT`：默认 `https://gamma-api.polymarket.com`

示例：

```bash
export POLY_DATA_API_ROOT=https://data-api.polymarket.com
export POLY_GAMMA_ROOT=https://gamma-api.polymarket.com
```

## 常见问题

1. **抓取量太大/翻页受限**：建议缩小 `--start/--end` 时间范围或减少 `--window-hours`。
2. **maker/taker 推断不确定**：脚本基于 `takerOnly` 集合差做统计推断，非逐笔强保证。
3. **parquet 未生成**：确认安装 `pandas`。

## 备注

- 本脚本尽量使用官方公开 Data/Gamma API，避免依赖网页 Activity 壳。
- 统计结果用于行为模式分析，无法替代链上或撮合层的严格判定。
