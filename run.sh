#!/usr/bin/env bash
# Launch Rep-LDM Studio.
set -e
cd "$(dirname "$0")"
# Uses the active Python by default; override with REPLDM_PYTHON=/path/to/python
PYTHON="${REPLDM_PYTHON:-python}"
exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
