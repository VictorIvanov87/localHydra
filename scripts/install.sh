#!/usr/bin/env bash
# Clean install of project-local dependencies (backend/.venv, frontend/node_modules).
# Dependency downloads happen here only; the application itself never downloads anything.
set -euo pipefail
cd "$(dirname "$0")/.."

PY_VERSION="${PY_VERSION:-3.12}" # supported: 3.11+; checks were run on 3.12

if command -v uv >/dev/null 2>&1; then
  uv venv backend/.venv --python "$PY_VERSION"
  uv pip install --python backend/.venv/bin/python -c backend/constraints.txt -e "backend[dev]"
else
  "python$PY_VERSION" -m venv backend/.venv
  backend/.venv/bin/python -m pip install -c backend/constraints.txt -e "backend[dev]"
fi

npm ci --prefix frontend
echo "install OK"
