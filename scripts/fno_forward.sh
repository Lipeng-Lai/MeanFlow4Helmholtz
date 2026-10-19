#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec python evaluate_fno.py --config configs/fno.yaml "$@"
