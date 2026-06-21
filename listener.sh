#!/usr/bin/env bash
# Terminal 3 — listens for "mark this" commands from server.py and runs the
# full agentic router (src/cli.ts) so you see the complete reasoning trace.
set -e
cd "$(dirname "$0")"

PIPE=/tmp/intention.pipe

[ -p "$PIPE" ] || mkfifo "$PIPE"

echo "👂  Waiting for 'mark this' commands on $PIPE …"
echo "    (say 'mark this, <anything>' while the glasses are streaming)"
echo ""

while true; do
    # Blocks here until server.py writes a command to the FIFO
    read -r intent < "$PIPE"
    [ -z "$intent" ] && continue

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯  mark this → $intent"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    node_modules/.bin/tsx --env-file=.env src/cli.ts "$intent"
    echo ""
done
