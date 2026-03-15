#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCREEN_CONFIG="${TOPIC_SCREEN_CONFIG:-topics/nba_ncaab/screen_users_config_human_nba_ncaab_relaxed.json}"

cd "$ROOT_DIR"
python3 screen_users.py --config "$SCREEN_CONFIG"
