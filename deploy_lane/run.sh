#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:${LD_LIBRARY_PATH:-}"
chmod +x "$SCRIPT_DIR/luckfox_benchmark"
./luckfox_benchmark
