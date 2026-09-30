#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" exec python train.py \
    --config configs/meanflow_improved_ema999.yaml "$@"
