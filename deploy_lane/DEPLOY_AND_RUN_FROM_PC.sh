#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <BOARD_IP> [BOARD_USER]"
  echo "Example: $0 192.168.1.120 root"
  exit 1
fi

BOARD_IP="$1"
BOARD_USER="${2:-root}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/root/deploy_lane"

echo "[1/3] Copying files to ${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR} ..."
ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p ${REMOTE_DIR}"
scp -r "${LOCAL_DIR}/." "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

echo "[2/3] Running benchmark on board ..."
ssh "${BOARD_USER}@${BOARD_IP}" "cd ${REMOTE_DIR} && chmod +x run.sh luckfox_benchmark && ./run.sh"

echo "[3/3] Pulling result CSV back to laptop ..."
scp "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/luckfox_results.csv" "${LOCAL_DIR}/luckfox_results.csv"

echo "Done. Result saved at: ${LOCAL_DIR}/luckfox_results.csv"
