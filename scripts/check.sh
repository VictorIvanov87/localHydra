#!/usr/bin/env bash
# Full local verification: contract sync, backend tests, frontend typecheck/tests/build.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== contracts in sync (schemas, fixtures, generated TS types) =="
scripts/contracts.sh check
echo "== backend tests =="
backend/.venv/bin/python -m pytest backend -q
echo "== frontend typecheck =="
npm run --silent --prefix frontend typecheck
echo "== frontend contract fixture validation =="
npm run --silent --prefix frontend test
echo "== frontend build =="
npm run --silent --prefix frontend build
echo "check OK"
