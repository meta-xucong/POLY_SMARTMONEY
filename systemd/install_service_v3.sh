#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <repo_root> <run_user> <python_bin|auto|-> <env_file|- for none>" >&2
  exit 1
fi

REPO_ROOT="$1"
RUN_USER="$2"
PYTHON_BIN="$3"
ENV_FILE="$4"

resolve_python_bin() {
  local requested="$1"
  local candidate=""
  if [ "$requested" = "-" ] || [ "$requested" = "auto" ]; then
    for candidate in \
      "${REPO_ROOT}/.venv/bin/python" \
      "$(command -v python3 2>/dev/null || true)" \
      "/usr/bin/python3"
    do
      if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        echo "$candidate"
        return 0
      fi
    done
    echo "ERROR: no usable python found. Tried .venv/bin/python and system python3." >&2
    return 1
  fi
  if [ ! -x "$requested" ]; then
    echo "ERROR: python executable not found or not executable: $requested" >&2
    return 1
  fi
  echo "$requested"
}

PYTHON_BIN="$(resolve_python_bin "$PYTHON_BIN")"

ACCOUNTS_FILE="${REPO_ROOT}/copytrade_v3_muti/accounts.json"
CONFIG_FILE="${REPO_ROOT}/copytrade_v3_muti/copytrade_config.json"
LOG_DIR="${REPO_ROOT}/copytrade_v3_muti/logs"
OUT_LOG="${LOG_DIR}/systemd.out.log"
ERR_LOG="${LOG_DIR}/systemd.err.log"

if [ ! -f "$ACCOUNTS_FILE" ]; then
  echo "ERROR: accounts.json not found: ${ACCOUNTS_FILE}" >&2
  echo "Please create it before installing the V3 service." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$OUT_LOG" "$ERR_LOG"

# Normalize potential Windows BOM/CRLF to avoid JSON parse/runtime issues.
sed -i '1s/^\xEF\xBB\xBF//' "$ACCOUNTS_FILE" 2>/dev/null || true
sed -i 's/\r$//' "$ACCOUNTS_FILE" 2>/dev/null || true
if [ -f "$CONFIG_FILE" ]; then
  sed -i '1s/^\xEF\xBB\xBF//' "$CONFIG_FILE" 2>/dev/null || true
  sed -i 's/\r$//' "$CONFIG_FILE" 2>/dev/null || true
fi

SERVICE_NAME="polysmart-copytrade-v3.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

ENV_LINE=""
if [ "$ENV_FILE" != "-" ]; then
  ENV_LINE="EnvironmentFile=${ENV_FILE}"
fi

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=POLY_SMARTMONEY Copytrade V3 Multi
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} ${REPO_ROOT}/copytrade_v3_muti/copytrade_run.py
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=20

StandardOutput=append:${REPO_ROOT}/copytrade_v3_muti/logs/systemd.out.log
StandardError=append:${REPO_ROOT}/copytrade_v3_muti/logs/systemd.err.log

[Install]
WantedBy=multi-user.target
EOF

if [ -n "$ENV_LINE" ]; then
  sudo sed -i "s|^Environment=PYTHONUNBUFFERED=1$|$ENV_LINE\\nEnvironment=PYTHONUNBUFFERED=1|" "$SERVICE_PATH"
fi

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Installed and started: $SERVICE_NAME (python=${PYTHON_BIN})"
