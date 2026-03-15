#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TOPIC_DIR="$ROOT_DIR/topics/politics_event"
DATA_DIR="$TOPIC_DIR/data"
SEED_FILE="$DATA_DIR/topic_seed_users_politics_event.csv"
META_FILE="$DATA_DIR/topic_seed_users_politics_event.metadata.json"
SCREEN_CONFIG="${TOPIC_SCREEN_CONFIG:-topics/politics_event/screen_users_config_politics_event_specialist_refined.json}"

mkdir -p "$DATA_DIR"
cd "$ROOT_DIR"

python3 discover_topic_users.py \
  --preset politics_event \
  --output-file "$SEED_FILE" \
  --metadata-file "$META_FILE"

python3 poly_martmoney_query_rerun_candidates.py \
  --config "$SCREEN_CONFIG" \
  --users-file "$SEED_FILE" \
  --user-column user \
  --days 60 \
  --lifetime-mode all \
  --rerun-screen
