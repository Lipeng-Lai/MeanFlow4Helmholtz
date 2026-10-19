#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec python dataset/generate/generate_dataset.py \
    --config configs/generate_dataset.yaml "$@"
