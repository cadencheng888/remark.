#!/usr/bin/env bash
# Boot the whole app in one terminal.
# Usage:
#   ./run.sh           start everything (uses existing web build)
#   ./run.sh --build   rebuild the React HUD first, then start
set -e
cd "$(dirname "$0")"

source .venv/bin/activate

if [ "$1" = "--build" ]; then
  echo "▸ building web HUD…"
  (cd web && npm run build)
fi

# ── kill stale processes ──────────────────────────────────────────────────────
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
lsof -ti tcp:8788 | xargs kill -9 2>/dev/null || true

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done }
trap cleanup EXIT

# ── intent router (TypeScript, port 8788) ────────────────────────────────────
if [ -d node_modules ]; then
  echo "▸ intent router      :8788  (logs → /tmp/intent-router.log)"
  npm run serve >/tmp/intent-router.log 2>&1 &
  PIDS+=($!)
else
  echo "▸ skipping intent router — run \`npm install\` to enable"
fi

# ── mark-this listener ───────────────────────────────────────────────────────
mkfifo /tmp/intention.pipe 2>/dev/null || true
if [ -f node_modules/.bin/tsx ]; then
  echo "▸ mark-this listener         (output below)"
  ./listener.sh &
  PIDS+=($!)
else
  echo "▸ skipping listener — run \`npm install\` to enable"
fi

# ── main HUD (port 8000, foreground) ─────────────────────────────────────────
echo ""
echo "  👓 Glasses HUD   → http://localhost:8000"
echo "  📱 iPhone/glasses → ws://<your-ip>:8000/ws/iphone"
echo ""
python server.py
