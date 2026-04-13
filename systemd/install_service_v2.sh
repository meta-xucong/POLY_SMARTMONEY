#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <repo_root> <run_user> <python_bin> <env_file>" >&2
  exit 1
fi

REPO_ROOT="$1"
RUN_USER="$2"
PYTHON_BIN="$3"
ENV_FILE="$4"

SERVICE_NAME="polysmart-copytrade-v2.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=POLY_SMARTMONEY Copytrade V2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=${ENV_FILE}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} ${REPO_ROOT}/copytrade_v2/copytrade_run.py
Restart=always
RestartSec=5

StandardOutput=append:${REPO_ROOT}/copytrade_v2/logs/systemd.out.log
StandardError=append:${REPO_ROOT}/copytrade_v2/logs/systemd.err.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Installed and started: $SERVICE_NAME"
