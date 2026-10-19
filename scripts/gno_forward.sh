#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec python evaluate_gno.py \
  --config configs/gno_ema999.yaml "$@"
