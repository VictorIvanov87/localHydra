#!/usr/bin/env bash
# contracts.sh generate | check  -- keeps schemas, fixtures and TS types in sync (docs/contracts.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=backend/.venv/bin/python

case "${1:-}" in
  generate)
    "$PY" -m localhydra.contracts.registry
    "$PY" contracts/tools/build_fixtures.py
    npm run --silent --prefix frontend contracts:generate
    ;;
  check)
    "$PY" -m localhydra.contracts.registry --check
    "$PY" contracts/tools/build_fixtures.py --check
    npm run --silent --prefix frontend contracts:check
    ;;
  *)
    echo "usage: scripts/contracts.sh generate|check" >&2
    exit 64
    ;;
esac
