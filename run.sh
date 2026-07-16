#!/usr/bin/env bash
# Launch Rep-LDM Studio.
set -e
cd "$(dirname "$0")"
PYTHON="${REPLDM_PYTHON:-$HOME/anaconda3/envs/di/bin/python}"
exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
