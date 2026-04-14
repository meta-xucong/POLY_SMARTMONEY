#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <repo_root> <run_user> <python_bin> <env_file|- for none>" >&2
  exit 1
fi

REPO_ROOT="$1"
RUN_USER="$2"
PYTHON_BIN="$3"
ENV_FILE="$4"

ACCOUNTS_FILE="${REPO_ROOT}/copytrade_v4_muti/accounts.json"

if [ ! -f "$ACCOUNTS_FILE" ]; then
  echo "ERROR: accounts.json not found: ${ACCOUNTS_FILE}" >&2
  echo "Please create it before installing the V4 service." >&2
  exit 1
fi

SERVICE_NAME="polysmart-copytrade-v4.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

ENV_LINE=""
if [ "$ENV_FILE" != "-" ]; then
  ENV_LINE="EnvironmentFile=${ENV_FILE}"
fi

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=POLY_SMARTMONEY Copytrade V4 Multi
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} ${REPO_ROOT}/copytrade_v4_muti/copytrade_run.py
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=20

StandardOutput=append:${REPO_ROOT}/copytrade_v4_muti/logs/systemd.out.log
StandardError=append:${REPO_ROOT}/copytrade_v4_muti/logs/systemd.err.log

[Install]
WantedBy=multi-user.target
EOF

if [ -n "$ENV_LINE" ]; then
  sudo sed -i "s|^Environment=PYTHONUNBUFFERED=1$|$ENV_LINE\\nEnvironment=PYTHONUNBUFFERED=1|" "$SERVICE_PATH"
fi

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Installed and started: $SERVICE_NAME"
