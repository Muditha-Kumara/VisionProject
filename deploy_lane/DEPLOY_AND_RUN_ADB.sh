#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${LOCAL_DIR}/dataset"
PROJECT_ROOT="$(cd "${LOCAL_DIR}/.." && pwd)"
POLY_SRC="${PROJECT_ROOT}/lane_polygons.csv"
POLY_DST="${LOCAL_DIR}/lane_polygons.csv"

if [[ -f "${POLY_SRC}" ]]; then
  if [[ ! -f "${POLY_DST}" || "${POLY_SRC}" -nt "${POLY_DST}" ]]; then
    cp "${POLY_SRC}" "${POLY_DST}"
    echo "Updated deploy polygon CSV from project root: ${POLY_SRC} -> ${POLY_DST}"
  fi
fi

dataset_fingerprint() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "missing"
    return 0
  fi
  (
    cd "$dir"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

DATASET_HASH_BEFORE="$(dataset_fingerprint "$DATASET_DIR")"
if [[ $# -ge 1 ]]; then
  REMOTE_DIR="$1"
else
  if adb shell "[ -d /userdata ]" >/dev/null 2>&1; then
    REMOTE_DIR="/userdata/deploy_lane"
  else
    REMOTE_DIR="/root/deploy_lane"
  fi
fi

echo "[1/5] Checking ADB connection..."
adb devices

adb_wait() {
  adb wait-for-device >/dev/null 2>&1 || true
  if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
    return 1
  fi
  return 0
}

adb_retry() {
  local n=1
  local max=6
  local delay=2
  while true; do
    if adb_wait && "$@"; then
      return 0
    fi
    if [[ $n -ge $max ]]; then
      echo "ADB command failed after ${max} attempts: $*"
      return 1
    fi
    echo "ADB transient failure, retry ${n}/${max} ..."
    n=$((n + 1))
    sleep $delay
  done
}

if ! adb_wait; then
  echo "ADB device not ready. Check cable/authorization and retry."
  exit 1
fi

echo "[2/5] Preparing remote directory: ${REMOTE_DIR}"
adb_retry adb shell "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}/lib'"

echo "[3/5] Pushing files..."
adb_retry adb push "${LOCAL_DIR}/luckfox_benchmark" "${REMOTE_DIR}/luckfox_benchmark"
adb_retry adb push "${LOCAL_DIR}/yolov8n.rknn" "${REMOTE_DIR}/yolov8n.rknn"
adb_retry adb push "${LOCAL_DIR}/lane_polygons.csv" "${REMOTE_DIR}/lane_polygons.csv"
adb_retry adb push "${LOCAL_DIR}/run.sh" "${REMOTE_DIR}/run.sh"
adb_retry adb push "${LOCAL_DIR}/lib/librknnmrt.so" "${REMOTE_DIR}/lib/librknnmrt.so"
adb_retry adb push "${LOCAL_DIR}/lib/librga.so" "${REMOTE_DIR}/lib/librga.so"
if [[ -f "${LOCAL_DIR}/lib/librknn_api.so" ]]; then
  adb_retry adb push "${LOCAL_DIR}/lib/librknn_api.so" "${REMOTE_DIR}/lib/librknn_api.so"
fi
adb_retry adb push "${LOCAL_DIR}/dataset" "${REMOTE_DIR}/dataset"

echo "[4/5] Running benchmark on device..."
adb_retry adb shell "cd '${REMOTE_DIR}' && chmod +x luckfox_benchmark run.sh && ./run.sh"

echo "[5/5] Pulling result CSV back..."
adb_retry adb pull "${REMOTE_DIR}/luckfox_results.csv" "${LOCAL_DIR}/luckfox_results.csv"

if adb_retry adb shell "[ -d '${REMOTE_DIR}/output_images' ]"; then
  rm -rf "${LOCAL_DIR}/output_images"
  adb_retry adb pull "${REMOTE_DIR}/output_images" "${LOCAL_DIR}/output_images"
  echo "Annotated images saved at: ${LOCAL_DIR}/output_images"
fi

echo "Done. Result file: ${LOCAL_DIR}/luckfox_results.csv"

DATASET_HASH_AFTER="$(dataset_fingerprint "$DATASET_DIR")"
if [[ "$DATASET_HASH_BEFORE" != "$DATASET_HASH_AFTER" ]]; then
  echo "ERROR: Local dataset changed during script execution."
  echo "Before: ${DATASET_HASH_BEFORE}"
  echo "After : ${DATASET_HASH_AFTER}"
  exit 1
fi
echo "Dataset integrity verified (unchanged)."
