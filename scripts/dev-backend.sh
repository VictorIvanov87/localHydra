#!/usr/bin/env bash
# Start the loopback backend (127.0.0.1:8765). Extra args are passed through, e.g.
#   scripts/dev-backend.sh --config config/localhydra.json
set -euo pipefail
cd "$(dirname "$0")/.."
exec backend/.venv/bin/python -m localhydra "$@"
