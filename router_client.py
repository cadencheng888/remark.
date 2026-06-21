"""
Bridge from the audio pipeline to the agentic intent router (src/server.ts).

Claude (agent.py) turns conversation into an *intent string* — a single natural
-language instruction like "add Nike Air Forces to your Amazon cart". This module
hands that string to route() over HTTP, which tries, in order: a specialized
Fetch.ai agent, the calendar API tier, then a Browserbase web agent that can
accomplish essentially any web task. Calendar is just one tier — any intent works.

The web agent can take 30-90s, so we fire on a background thread. The caller
shows an optimistic card immediately; the real outcome arrives later via `sink`.
"""
import json
import os
import threading
import urllib.error
import urllib.request

import geo

# Where src/server.ts listens. Override with ROUTER_URL in .env.
ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:8788").rstrip("/")
_TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT", "180"))  # seconds; web agent is slow

_SOURCE_LABEL = {
    "agentverse": "Fetch.ai agent",
    "calendar": "Calendar",
    "browserbase": "Web agent",
    "none": "Router",
}


def _with_location(intent: str) -> str:
    """Bake the wearer's current location into the intent string so the web
    agent can handle "near me" / directions / delivery / local availability —
    e.g. "find EV charging stations near me" goes out with the actual city +
    coordinates appended. No-op if location is unavailable."""
    loc = geo.get_location()
    if not loc or not loc.get("label"):
        return intent
    coords = ""
    if loc.get("lat") is not None and loc.get("lng") is not None:
        coords = f", lat {loc['lat']}, lng {loc['lng']}"
    return (f"{intent} (the user's current location is {loc['label']}{coords} — "
            f"use it for any location-dependent step: nearby places, directions, "
            f"delivery, local availability)")


def _format_result(intent: str, result: dict) -> str:
    """Turn a RouteResult into a HUD-ready card line (web/src/agent.js parses
    the ✅ / 🟡 / ⚠️ prefixes)."""
    source = result.get("source", "none")
    status = result.get("status", "failed")
    label = _SOURCE_LABEL.get(source, source)
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    detail = (payload.get("outcome") or payload.get("message")
              or payload.get("details") or "").strip()

    if status == "success":
        return f"✅ {detail or intent}  ·  [{label}]"
    if status == "partial":
        stopped = (payload.get("stoppedBecause") or "").strip()
        tail = f" — {stopped}" if stopped else ""
        return f"🟡 {detail or intent}{tail}  ·  [{label}]"
    return f"⚠️ couldn't complete: {intent}  ·  [{label}]"


def dispatch(intent: str, sink=None) -> None:
    """Send `intent` to the router on a background thread and stream events back.

    sink is called with a dict, thread-safely (it runs on a worker thread):
        {"kind": "thinking", "text": <one reasoning line>}   — many, live
        {"kind": "result",   "text": <final HUD card line>}  — once, at the end

    server.py turns these into {"type":"thinking"} / {"type":"action"} broadcasts.
    No-op if intent is empty.
    """
    intent = (intent or "").strip()
    if not intent:
        return

    def _emit_result(text):
        if sink:
            sink({"kind": "result", "text": text})

    outgoing = _with_location(intent)  # what the router/web agent actually sees

    def _run():
        try:
            req = urllib.request.Request(
                f"{ROUTER_URL}/route",
                data=json.dumps({"intent": outgoing}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                for raw in resp:  # NDJSON — one event per line, as it streams in
                    line = raw.decode().strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "trace" and sink:
                        sink({"kind": "thinking", "text": msg.get("line", "")})
                    elif msg.get("type") == "result":
                        _emit_result(_format_result(intent, msg.get("result", {})))
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            print(f"[router] offline/unreachable ({reason}) for: {intent!r}")
            _emit_result(f"⚠️ agent router offline — couldn't run: {intent}")
        except Exception as e:  # noqa: BLE001 — never let a worker thread crash silently
            print(f"[router] error ({e!r}) for: {intent!r}")
            _emit_result(f"⚠️ agent router error — couldn't run: {intent}")

    threading.Thread(target=_run, daemon=True).start()


def health() -> bool:
    """Quick check used at startup to tell the user if the router is up."""
    try:
        with urllib.request.urlopen(f"{ROUTER_URL}/health", timeout=2) as resp:
            return json.loads(resp.read().decode()).get("ok", False)
    except Exception:
        return False
