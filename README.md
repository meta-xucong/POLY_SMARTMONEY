# POLY_SMARTMONEY

## 一键更新（仅干净代码，不涉及私钥）

说明：
- 下面命令会自动识别当前 Git 仓库根目录，不写死本地路径，适配 VPS。
- 只更新 `v3/v4` 代码和 `copytrade_config.json`。
- 不会改动 `accounts.json`、`state_*.json`、`logs/`。

前提：
- 你在本仓库内任意目录执行命令（`git rev-parse --show-toplevel` 能返回路径）。

### 更新 v3（干净代码）

```bash
repo="$(git rev-parse --show-toplevel)" && \
git -C "$repo" fetch origin main && \
git -C "$repo" checkout origin/main -- \
  copytrade_v3_muti/*.py \
  copytrade_v3_muti/check_*.py \
  copytrade_v3_muti/test_*.py \
  copytrade_v3_muti/copytrade_config.json
```

```powershell
$repo=(git rev-parse --show-toplevel).Trim(); `
git -C $repo fetch origin main; `
git -C $repo checkout origin/main -- `
  copytrade_v3_muti/*.py `
  copytrade_v3_muti/check_*.py `
  copytrade_v3_muti/test_*.py `
  copytrade_v3_muti/copytrade_config.json
```

### 更新 v4（干净代码）

```bash
repo="$(git rev-parse --show-toplevel)" && \
git -C "$repo" fetch origin main && \
git -C "$repo" checkout origin/main -- \
  copytrade_v4_muti/*.py \
  copytrade_v4_muti/check_*.py \
  copytrade_v4_muti/test_*.py \
  copytrade_v4_muti/copytrade_config.json
```

```powershell
$repo=(git rev-parse --show-toplevel).Trim(); `
git -C $repo fetch origin main; `
git -C $repo checkout origin/main -- `
  copytrade_v4_muti/*.py `
  copytrade_v4_muti/check_*.py `
  copytrade_v4_muti/test_*.py `
  copytrade_v4_muti/copytrade_config.json
```

### 可选：更新后快速自检

```bash
repo="$(git rev-parse --show-toplevel)" && \
pytest -q "$repo/copytrade_v3_muti" && \
pytest -q "$repo/copytrade_v4_muti"
```

```powershell
$repo=(git rev-parse --show-toplevel).Trim(); `
pytest -q (Join-Path $repo "copytrade_v3_muti"); `
pytest -q (Join-Path $repo "copytrade_v4_muti")
```

## 备注

- 私钥请手动维护在你本地的 `accounts.json`。
- 如果只想更新某一个版本，只执行对应那一段命令即可。
