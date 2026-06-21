#!/usr/bin/env bash
# Boot the whole app. Usage:
#   ./run.sh           start the server (uses the existing web build)
#   ./run.sh --build   rebuild the React HUD first, then start
set -e
cd "$(dirname "$0")"

source .venv/bin/activate

if [ "$1" = "--build" ]; then
  echo "▸ building web HUD…"
  (cd web && npm run build)
fi

# Start the agentic intent router (TS) so perform_action intents actually
# execute. Needs `npm install` once. We keep its PID and stop it on exit.
ROUTER_PID=""
if [ -d node_modules ]; then
  lsof -ti tcp:8788 | xargs kill -9 2>/dev/null || true
  echo "▸ starting intent router on :8788  (logs → /tmp/intent-router.log)"
  npm run serve >/tmp/intent-router.log 2>&1 &
  ROUTER_PID=$!
  trap '[ -n "$ROUTER_PID" ] && kill "$ROUTER_PID" 2>/dev/null || true' EXIT
else
  echo "▸ skipping intent router — run \`npm install\` to enable real execution"
fi

# kill any stale server holding the port
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true

echo "▸ open http://localhost:8000"
python server.py
