#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCREEN_CONFIG="${TOPIC_SCREEN_CONFIG:-topics/politics_event/screen_users_config_politics_event_specialist_refined.json}"

cd "$ROOT_DIR"
python3 screen_users.py --config "$SCREEN_CONFIG"
