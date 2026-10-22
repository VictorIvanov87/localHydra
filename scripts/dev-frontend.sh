#!/usr/bin/env bash
# Start the Vite dev server on 127.0.0.1:5173, proxying /api to the loopback backend.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
exec npm run dev
