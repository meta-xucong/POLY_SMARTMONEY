#!/usr/bin/env bash
set -euo pipefail

LEGACY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "${LEGACY_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TOP="${TOP:-5000}"
PERIOD="${PERIOD:-ALL}"
ORDER_BY="${ORDER_BY:-vol}"
DAYS="${DAYS:-365}"
LIFETIME_MODE="${LIFETIME_MODE:-all}"
TRADE_ACTIONS_PAGE_SIZE="${TRADE_ACTIONS_PAGE_SIZE:-300}"
SCREEN_CONFIG="${SCREEN_CONFIG:-legacy/screen_users_config_human_politics_actionable.json}"

cd "${SCRIPT_DIR}"

exec "${PYTHON_BIN}" "${LEGACY_DIR}/poly_martmoney_query_run.py" \
  --top "${TOP}" \
  --period "${PERIOD}" \
  --order-by "${ORDER_BY}" \
  --days "${DAYS}" \
  --lifetime-mode "${LIFETIME_MODE}" \
  --trade-actions-page-size "${TRADE_ACTIONS_PAGE_SIZE}" \
  --screen-config "${SCREEN_CONFIG}"
