# POLY_SMARTMONEY

## 更新说明（只更新干净代码，不包含私钥）

下面所有命令都不会主动改动以下文件：
- `accounts.json`
- `state_*.json`
- `logs/`

你只需要按你的部署方式选一个章节执行，不要混用。

## 1) 有 `.git` 的机器（标准 Git 更新）

前提：
- 在仓库目录内执行（`git rev-parse --show-toplevel` 能返回路径）。

### 更新 v3（只执行一段）

VPS / Linux / macOS:

```bash
repo="$(git rev-parse --show-toplevel)" && \
git -C "$repo" fetch origin main && \
git -C "$repo" checkout origin/main -- \
  copytrade_v3_muti/*.py \
  copytrade_v3_muti/check_*.py \
  copytrade_v3_muti/test_*.py \
  copytrade_v3_muti/copytrade_config.json \
  systemd/install_service_v3.sh
```

Windows PowerShell:

```powershell
$repo=(git rev-parse --show-toplevel).Trim(); `
git -C $repo fetch origin main; `
git -C $repo checkout origin/main -- `
  copytrade_v3_muti/*.py `
  copytrade_v3_muti/check_*.py `
  copytrade_v3_muti/test_*.py `
  copytrade_v3_muti/copytrade_config.json `
  systemd/install_service_v3.sh
```

更新后可直接重装并启动 v3 服务：
```bash
repo="$(git rev-parse --show-toplevel)"
sudo bash "$repo/systemd/install_service_v3.sh" "$repo" root auto -
sudo systemctl restart polysmart-copytrade-v3.service
```

### 更新 v4（只执行一段）

VPS / Linux / macOS:

```bash
repo="$(git rev-parse --show-toplevel)" && \
git -C "$repo" fetch origin main && \
git -C "$repo" checkout origin/main -- \
  copytrade_v4_muti/*.py \
  copytrade_v4_muti/check_*.py \
  copytrade_v4_muti/test_*.py \
  copytrade_v4_muti/copytrade_config.json \
  systemd/install_service_v4.sh
```

Windows PowerShell:

```powershell
$repo=(git rev-parse --show-toplevel).Trim(); `
git -C $repo fetch origin main; `
git -C $repo checkout origin/main -- `
  copytrade_v4_muti/*.py `
  copytrade_v4_muti/check_*.py `
  copytrade_v4_muti/test_*.py `
  copytrade_v4_muti/copytrade_config.json `
  systemd/install_service_v4.sh
```

更新后可直接重装并启动 v4 服务：
```bash
repo="$(git rev-parse --show-toplevel)"
sudo bash "$repo/systemd/install_service_v4.sh" "$repo" root auto -
sudo systemctl restart polysmart-copytrade-v4.service
```

v4 并发与限速（避免 429）：
- `copytrade_v4_muti/copytrade_config.json` 默认 `account_workers=2`
- 全局总速率：`global_data_api_rps` / `global_data_http_rps` / `global_clob_api_rps`
- 程序会按 `account_workers` 自动均分到每个 worker
- 若 VPS 资源紧张，可把 `account_workers` 改回 `1`

## 2) 无 `.git` 的机器（宝塔/直接复制部署）

你的实际路径：
- `v3`: `/home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v3_muti`
- `v4`: `/home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v4_muti`

### 仅更新 v3

```bash
set -euo pipefail
dst="/home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v3_muti"
base="$(dirname "$dst")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -L "https://github.com/meta-xucong/POLY_SMARTMONEY/archive/refs/heads/main.tar.gz" | tar -xz -C "$tmp"
src="$tmp/POLY_SMARTMONEY-main/copytrade_v3_muti"

cp -f "$src"/*.py "$dst"/
cp -f "$src"/check_*.py "$dst"/
cp -f "$src"/test_*.py "$dst"/
cp -f "$src"/copytrade_config.json "$dst"/
mkdir -p "$base/systemd"
cp -f "$tmp/POLY_SMARTMONEY-main/systemd/install_service_v3.sh" "$base/systemd"/
sed -i 's/\r$//' "$base/systemd/install_service_v3.sh" || true
sed -i '1s/^\xEF\xBB\xBF//' "$base/systemd/install_service_v3.sh" || true
sudo bash "$base/systemd/install_service_v3.sh" "$base" root auto -
sudo systemctl restart polysmart-copytrade-v3.service

echo "v3 update done: $dst"
```

### 仅更新 v4

```bash
set -euo pipefail
dst="/home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v4_muti"
base="$(dirname "$dst")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -L "https://github.com/meta-xucong/POLY_SMARTMONEY/archive/refs/heads/main.tar.gz" | tar -xz -C "$tmp"
src="$tmp/POLY_SMARTMONEY-main/copytrade_v4_muti"

cp -f "$src"/*.py "$dst"/
cp -f "$src"/check_*.py "$dst"/
cp -f "$src"/test_*.py "$dst"/
cp -f "$src"/copytrade_config.json "$dst"/
mkdir -p "$base/systemd"
cp -f "$tmp/POLY_SMARTMONEY-main/systemd/install_service_v4.sh" "$base/systemd"/
sed -i 's/\r$//' "$base/systemd/install_service_v4.sh" || true
sed -i '1s/^\xEF\xBB\xBF//' "$base/systemd/install_service_v4.sh" || true
sudo bash "$base/systemd/install_service_v4.sh" "$base" root auto -
sudo systemctl restart polysmart-copytrade-v4.service

echo "v4 update done: $dst"
```

## 3) 空白 VPS 一键部署（不含私钥，部署后手动填写）

用途：
- 新 VPS 首次安装（无代码、无环境）。
- 脚本会拉取最新代码，创建虚拟环境，安装依赖，并生成 `accounts.json` 模板。

执行命令：

```bash
set -euo pipefail

BASE="/home/trader/polymarket_api/POLY_SMARTMONEY"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl tar

mkdir -p "$(dirname "$BASE")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -L "https://github.com/meta-xucong/POLY_SMARTMONEY/archive/refs/heads/main.tar.gz" | tar -xz -C "$tmp"
src="$tmp/POLY_SMARTMONEY-main"

mkdir -p "$BASE"
cp -a "$src/copytrade_v1" "$BASE/" 2>/dev/null || true
cp -a "$src/copytrade_v2" "$BASE/" 2>/dev/null || true
cp -a "$src/copytrade_v3_muti" "$BASE/"
cp -a "$src/copytrade_v4_muti" "$BASE/"
cp -a "$src/smartmoney_query" "$BASE/"
cp -a "$src/systemd" "$BASE/"
cp -a "$src/check_key_address.py" "$BASE/" 2>/dev/null || true
cp -a "$src/README.md" "$BASE/" 2>/dev/null || true

python3 -m venv "$BASE/.venv"
"$BASE/.venv/bin/pip" install -U pip wheel
"$BASE/.venv/bin/pip" install py-clob-client requests eth-account

mkdir -p "$BASE/copytrade_v3_muti/logs" "$BASE/copytrade_v4_muti/logs"
[ -f "$BASE/copytrade_v3_muti/accounts.json" ] || cp "$BASE/copytrade_v3_muti/accounts.example.json" "$BASE/copytrade_v3_muti/accounts.json"
[ -f "$BASE/copytrade_v4_muti/accounts.json" ] || cp "$BASE/copytrade_v4_muti/accounts.example.json" "$BASE/copytrade_v4_muti/accounts.json"

sed -i 's/\r$//' "$BASE/systemd/install_service_v3.sh" "$BASE/systemd/install_service_v4.sh" 2>/dev/null || true
sed -i '1s/^\xEF\xBB\xBF//' "$BASE/systemd/install_service_v3.sh" "$BASE/systemd/install_service_v4.sh" 2>/dev/null || true

echo "deploy done: $BASE"
echo "next: edit accounts.json manually"
echo "v3 account file: $BASE/copytrade_v3_muti/accounts.json"
echo "v4 account file: $BASE/copytrade_v4_muti/accounts.json"
```

部署后可直接一键启服务（推荐 `auto` 自动找 Python）：
```bash
sudo bash /home/trader/polymarket_api/POLY_SMARTMONEY/systemd/install_service_v4.sh /home/trader/polymarket_api/POLY_SMARTMONEY root auto -
```

部署后快速验证（任选一个版本）：

```bash
/home/trader/polymarket_api/POLY_SMARTMONEY/.venv/bin/python /home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v3_muti/copytrade_run.py --dry-run
```

```bash
/home/trader/polymarket_api/POLY_SMARTMONEY/.venv/bin/python /home/trader/polymarket_api/POLY_SMARTMONEY/copytrade_v4_muti/copytrade_run.py --dry-run
```
