#!/usr/bin/env bash
# Real loopback smoke test: starts the backend and the built frontend (vite preview) as separate
# processes, then checks health, models, the frontend shell, the proxy, bind addresses and the
# Host-header guard. Requires `npm run build` (scripts/check.sh does it) and the example config.
set -euo pipefail
cd "$(dirname "$0")/.."

BP="${BACKEND_PORT:-8765}"
FP="${FRONTEND_PORT:-4173}"
PY=backend/.venv/bin/python
LOGDIR="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

fail() { echo "SMOKE FAIL: $*" >&2; echo "--- backend log"; cat "$LOGDIR/backend.log" >&2 || true; exit 1; }

[ -d frontend/dist ] || fail "frontend/dist missing; run npm run build in frontend/ first"

"$PY" -m localhydra --config config/localhydra.example.json --port "$BP" >"$LOGDIR/backend.log" 2>&1 &
PIDS+=($!)
(cd frontend && LOCALHYDRA_BACKEND_URL="http://127.0.0.1:$BP" exec npx vite preview --port "$FP" >"$LOGDIR/frontend.log" 2>&1) &
PIDS+=($!)

B="http://127.0.0.1:$BP"
F="http://127.0.0.1:$FP"
curl -fsS --retry 30 --retry-delay 1 --retry-connrefused -o /dev/null "$B/api/health" || fail "backend did not come up"
curl -fsS --retry 30 --retry-delay 1 --retry-connrefused -o /dev/null "$F/" || fail "frontend did not come up"

echo "-- backend /api/health"
curl -fsS -D - "$B/api/health" | tee "$LOGDIR/health.txt"
echo
"$PY" - "$LOGDIR/health.txt" <<'EOF' || fail "health payload"
import json, sys
# Text-mode reads normalize HTTP CRLF to LF; split the normalized separator.
body = open(sys.argv[1]).read().split("\n\n", 1)[1]
h = json.loads(body)
assert h["status"] == "ok" and h["local_only"] is True and h["bind_host"] == "127.0.0.1", h
assert h["models_configured"] == 4, h
print("health payload OK")
EOF

echo "-- backend /api/models"
curl -fsS "$B/api/models" | "$PY" -c '
import json, sys
m = json.load(sys.stdin)
assert m["configured"] and m["auto_download"] is False and len(m["models"]) == 4, m
assert {s["availability"] for s in m["models"]} == {"unknown"}, m
print("models payload OK: 4 configured, availability unknown (no provider probe yet), auto_download false")
'

echo "-- error envelope + Host guard"
code=$(curl -s -o "$LOGDIR/nf.json" -w '%{http_code}' "$B/api/nope")
[ "$code" = 404 ] && grep -q '"not_found"' "$LOGDIR/nf.json" || fail "404 envelope ($code)"
code=$(curl -s -o "$LOGDIR/host.json" -w '%{http_code}' -H 'Host: evil.example.com' "$B/api/health")
[ "$code" = 400 ] && grep -q '"host_not_allowed"' "$LOGDIR/host.json" || fail "host guard ($code)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$B/openapi.json")
[ "$code" = 404 ] || fail "openapi/docs must be disabled (CDN assets), got $code"
echo "envelope, host guard, docs-disabled OK"

echo "-- frontend shell"
curl -fsS "$F/" >"$LOGDIR/index.html"
grep -q '<div id="root">' "$LOGDIR/index.html" || fail "no #root in index.html"
if grep -Eq '(src|href)="https?://' "$LOGDIR/index.html"; then fail "index.html references an external URL"; fi
asset=$(grep -Eo '/assets/[^"]+\.js' "$LOGDIR/index.html" | head -n1)
[ -n "$asset" ] || fail "no JS asset in index.html"
curl -fsS -o "$LOGDIR/app.js" -w 'asset %{url_effective} -> %{http_code}, %{size_download} bytes\n' "$F$asset"
echo "-- frontend -> backend proxy"
curl -fsS "$F/api/health" | grep -q '"local_only":true' || fail "proxy /api/health"
echo "proxy OK"

echo "-- listening sockets (must be loopback only)"
for port in "$BP" "$FP"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN | tee "$LOGDIR/lsof.$port"
  if grep -Eq '\*:'"$port"'|0\.0\.0\.0:'"$port" "$LOGDIR/lsof.$port"; then fail "port $port is not loopback-only"; fi
  grep -Eq '127\.0\.0\.1:'"$port"'|\[::1\]:'"$port" "$LOGDIR/lsof.$port" || fail "port $port not on loopback"
done

echo "SMOKE OK"
