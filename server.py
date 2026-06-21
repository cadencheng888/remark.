"""
Web "glasses HUD" for the audio agent.

    python server.py        # then open http://localhost:8000

Wraps the existing pipeline (transcribe.py + agent.py) — no changes to that
logic. Streams live captions and the actions Claude takes to the browser over a
WebSocket, and embeds your Google Calendar so events appear on screen.

Calendar mode is auto-detected:
  - credentials.json present  -> LIVE  (writes to your real calendar)
  - credentials.json missing  -> MOCK  (fake calendar so it still demos)
"""

import asyncio
import os
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import agent
import calendar_tool
import geo
import router_client
from demo import CONVO
from face_gate import FaceGate
from transcribe import stream_microphone

# --- Calendar mode: real if credentials exist, otherwise a mock so the HUD
#     demos immediately (before Google is connected). ---
# LIVE only when BOTH the client file AND a completed authorization exist —
# otherwise scheduling would block on the OAuth browser flow. Until you've
# authorized (token.json), we run MOCK so the HUD/Actions still work.
MODE = "live" if (os.path.exists("credentials.json") and os.path.exists("token.json")) else "mock"
if MODE == "mock":
    _n = [0]

    def _fake_create(title, start_iso, duration_minutes=60, location=None, notes=None):
        _n[0] += 1
        return f"📅 created '{title}' at {start_iso} [id:mock{_n[0]}]"

    calendar_tool.create_event = _fake_create
    calendar_tool.update_event = (
        lambda eid, start_iso=None, duration_minutes=None, **k: f"🔁 moved → {start_iso}"
    )
    calendar_tool.delete_event = lambda eid: "🗑️ cancelled"

# Your calendar, embedded. Override with the GCAL_EMBED env var if needed.
CAL_EMBED = os.environ.get(
    "GCAL_EMBED",
    "https://calendar.google.com/calendar/embed"
    "?src=cadencheng888%40gmail.com&mode=DAY",
)

app = FastAPI()
clients: set[WebSocket] = set()

# The running event loop, captured at startup so background threads (the agentic
# router's executor) can push result cards back onto the loop thread-safely.
_main_loop: asyncio.AbstractEventLoop | None = None

# Capture modes (distinct from MODE, which is the calendar mock/live badge):
#   conversation — passive, only captures when a face is in view (FaceGate)
#   solo         — only acts on commands prefixed with the wake phrase
CAPTURE_MODE = "conversation"
WAKE_PHRASE = "mark this"
face_gate = FaceGate()

# Solo-mode: a command ending on one of these (a transitive verb or a dangling
# preposition/article) is probably mid-sentence — wait a little longer for the
# rest before acting, so "mark this, find … <pause> … EV chargers near me"
# doesn't fire as just "find".
SOLO_INCOMPLETE_GRACE = 4.0  # seconds of extra patience for an unfinished command
_DANGLING_TAIL = {
    "find", "search", "play", "add", "order", "text", "call", "get", "buy",
    "remind", "email", "send", "navigate", "set", "make", "show", "look",
    "book", "schedule", "to", "for", "a", "an", "the", "me", "my", "up", "on",
    "with", "and", "of", "near",
}

# --- pipeline buffer (ephemeral: held only until it acts or the moment passes) ---
SILENCE_FLUSH_SECONDS = 1.5     # quiet gap before we send a flush to the agent
IDLE_CLEAR_SECONDS = 20         # clear the buffer ~20s after the last speech
MAX_AGE_SECONDS = 20            # hard guarantee: nothing held older than ~20s
# Short rolling window so recent clean speech dominates and old chatter doesn't
# linger and poison new requests.
MAX_BUFFER_CHARS = 600
_transcript: list[dict] = []   # each: {"t": text, "ts": monotonic}
_last_speech = 0.0
_dirty = False
_mic_task: asyncio.Task | None = None


async def broadcast(msg: dict):
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.discard(ws)


def _emit_router_event(ev: dict):
    """Thread-safe sink for the agentic router. agent.py's background dispatch
    threads call this with the router's live reasoning ('thinking') and its final
    result; we hop back onto the event loop and broadcast to the HUD."""
    if _main_loop is None:
        return
    kind = ev.get("kind")
    if kind == "thinking":
        msg = {"type": "thinking", "text": ev.get("text", "")}
    elif kind == "result":
        msg = {"type": "action", "text": ev.get("text", "")}
    else:
        return
    asyncio.run_coroutine_threadsafe(broadcast(msg), _main_loop)


@app.on_event("startup")
async def _on_startup():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    agent.set_action_sink(_emit_router_event)  # stream router thinking + results
    # Warm the location cache off the event loop so the first intent isn't
    # stalled by the IP-geolocation lookup.
    loc = await asyncio.to_thread(geo.get_location)
    if loc and loc.get("label"):
        print("  📍 location:", loc["label"])
    if router_client.health():
        print("  ✅ agentic router reachable at", router_client.ROUTER_URL)
    else:
        print("  ⚠️  agentic router NOT reachable at", router_client.ROUTER_URL,
              "— start it with `npm run serve` (perform_action cards still show,",
              "but won't execute).")


def on_final(text: str):
    global _last_speech, _dirty
    if CAPTURE_MODE == "conversation" and not face_gate.is_present():
        return  # no face in view — not capturing
    now = time.monotonic()
    _transcript.append({"t": text, "ts": now})
    _last_speech = now
    _dirty = True
    asyncio.create_task(broadcast({"type": "caption", "final": True, "text": text}))


def on_interim(text: str):
    if CAPTURE_MODE == "conversation" and not face_gate.is_present():
        return
    asyncio.create_task(broadcast({"type": "caption", "final": False, "text": text}))


def on_level(level: float):
    asyncio.create_task(broadcast({"type": "level", "value": round(level, 3)}))


def on_entities(entities):
    agent.add_entities(entities)  # cache for pronoun resolution ("buy them")
    vals = [e.get("value") for e in entities if e.get("value")]
    if vals:
        asyncio.create_task(broadcast({"type": "entities", "values": vals}))


async def _process_and_broadcast(conversation: str) -> bool:
    print(f"→ sending to Claude: {conversation!r}")
    await broadcast({"type": "status", "text": "thinking"})
    acted = False
    try:
        results = await asyncio.to_thread(agent.process_transcript, conversation)
        print(f"← Claude result: {results}")
        if results:
            for line in results:
                if line.startswith("❓CLARIFY|"):
                    parts = line.split("|", 2)
                    q = parts[1] if len(parts) > 1 else "Which one did you mean?"
                    opts = [o for o in (parts[2].split("||") if len(parts) > 2 else []) if o]
                    await broadcast({"type": "clarify", "question": q, "options": opts})
                else:
                    acted = True
                    await broadcast({"type": "action", "text": line})
        else:
            await broadcast({"type": "action", "text": "💬 (no action — chit-chat)", "muted": True})
    except Exception as e:
        print(f"✗ error: {e!r}")
        await broadcast({"type": "action", "text": f"⚠️ {e}", "muted": True})
    await broadcast({"type": "status", "text": "listening"})
    return acted


async def flusher():
    global _dirty, _transcript
    while True:
        await asyncio.sleep(0.25)
        now = time.monotonic()

        # Backstop: drop any utterance older than the max age.
        if _transcript:
            kept = [e for e in _transcript if now - e["ts"] < MAX_AGE_SECONDS]
            if len(kept) != len(_transcript):
                _transcript = kept
                if not _transcript:  # buffer just emptied — tell the HUD to clear
                    await broadcast({"type": "forgotten"})

        # Constraint 2 — clear after a lull: nothing pending and the room has
        # gone quiet, so the moment passed. Forget what was said.
        if _transcript and not _dirty and now - _last_speech > IDLE_CLEAR_SECONDS:
            _transcript = []
            await broadcast({"type": "forgotten"})
            continue

        if not _dirty:
            continue
        if now - _last_speech < SILENCE_FLUSH_SECONDS:
            continue
        _dirty = False

        # Rolling context so a request split across pauses is seen as one whole.
        conversation = " ".join(e["t"] for e in _transcript)[-MAX_BUFFER_CHARS:]
        if len(_transcript) > 10:
            _transcript = _transcript[-10:]

        if CAPTURE_MODE == "solo":
            # Only act on a command prefixed with the wake phrase ("mark this, …").
            i = conversation.lower().rfind(WAKE_PHRASE)
            if i == -1:
                continue  # no trigger yet — keep listening (buffer ages out on its own)
            command = conversation[i + len(WAKE_PHRASE):].lstrip(" ,.:;—-").strip()
            if not command:
                # heard "mark this" but the command hasn't been spoken yet —
                # DON'T clear; wait so the next utterance combines with the trigger.
                continue
            # The command can be split across a mid-sentence pause. If it still
            # looks unfinished (one word, or ends on a transitive verb /
            # preposition), keep waiting a few extra seconds for the rest rather
            # than firing "find" alone — but give up after the grace so a truly
            # terse command still acts.
            words = command.split()
            tail = words[-1].lower().strip(",.?!;:")
            unfinished = len(words) < 2 or tail in _DANGLING_TAIL
            if unfinished and (now - _last_speech) < SOLO_INCOMPLETE_GRACE:
                _dirty = True  # re-check next tick; combine with any further speech
                continue
            _transcript = []  # consume the whole trigger + command
            # "mark this" is an explicit do-something signal, so treat whatever
            # follows as an imperative and let Claude infer a missing verb/app
            # (e.g. "Judy Hopps plushie Amazon wishlist" -> add_to_wishlist).
            directive = (
                "Direct command from the wearer — perform it now, inferring the "
                "action verb and app if implied: " + command
            )
            await _process_and_broadcast(directive)
            continue

        await _process_and_broadcast(conversation)
        # NOT cleared instantly on action — the buffer lingers ~20s (see
        # IDLE_CLEAR_SECONDS / MAX_AGE_SECONDS) so quick follow-ups like
        # "actually move it to 8" keep context. Dedup stops the lingering text
        # from re-firing the same action.


async def _face_status_loop():
    last = None
    while True:
        await asyncio.sleep(0.4)
        state = ("off" if not face_gate.camera_ok()
                 else "present" if face_gate.is_present() else "absent")
        if state != last:
            last = state
            await broadcast({"type": "face", "state": state})


async def mic_loop():
    if CAPTURE_MODE == "conversation":
        face_gate.start()  # on-device camera gate (Conversation mode only)
    asyncio.create_task(_face_status_loop())
    await broadcast({"type": "status", "text": "listening"})
    try:
        await asyncio.gather(
            stream_microphone(
                on_final, on_interim, on_level=on_level, on_entities=on_entities
            ),
            flusher(),
        )
    except Exception as e:
        await broadcast({"type": "status", "text": "mic error"})
        await broadcast({"type": "action", "text": f"⚠️ mic: {e}", "muted": True})


async def run_demo():
    """Replay the scripted conversation through the real Claude pipeline."""
    agent._recent_events.clear()
    agent._recent_actions.clear()
    await broadcast({"type": "status", "text": "demo running"})
    for line in CONVO:
        # Stream word-by-word so captions look live.
        partial = ""
        for w in line.split():
            partial += (" " if partial else "") + w
            await broadcast({"type": "caption", "final": False, "text": partial})
            await asyncio.sleep(0.04)
        await broadcast({"type": "caption", "final": True, "text": line})
        await _process_and_broadcast(line)
        await asyncio.sleep(1.1)
    await broadcast({"type": "status", "text": "demo complete"})


@app.get("/")
async def index():
    return FileResponse("web/dist/index.html")


@app.get("/config")
async def config():
    loc = geo.get_location()
    return {
        "mode": MODE,
        "cal_embed": CAL_EMBED,
        "location": loc.get("label") if loc else None,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _mic_task, _dirty, CAPTURE_MODE
    await ws.accept()
    clients.add(ws)
    await ws.send_json({"type": "status", "text": "idle", "mode": MODE})
    await ws.send_json({"type": "capturemode", "mode": CAPTURE_MODE})
    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("cmd")
            if cmd == "demo":
                asyncio.create_task(run_demo())
            elif cmd == "mic":
                if _mic_task is None or _mic_task.done():
                    _mic_task = asyncio.create_task(mic_loop())
            elif cmd == "capturemode":
                m = data.get("mode")
                if m in ("conversation", "solo"):
                    CAPTURE_MODE = m
                    _transcript.clear()
                    _dirty = False
                    if m == "solo":
                        face_gate.stop()  # release the camera in Solo mode
                    elif _mic_task and not _mic_task.done():
                        face_gate.start()  # back to Conversation while miced → camera on
                    await broadcast({"type": "capturemode", "mode": CAPTURE_MODE})
            elif cmd == "answer":
                # User picked a clarification option — resolve it through the
                # agent (the entity cache still holds the referenced item).
                ans = (data.get("text") or "").strip()
                if ans:
                    await broadcast({"type": "caption", "final": True, "text": ans})
                    asyncio.create_task(_process_and_broadcast(ans))
            elif cmd == "reset":
                agent._recent_events.clear()
                agent._recent_actions.clear()
                agent._entity_cache.clear()
                _transcript.clear()
                _dirty = False
                await broadcast({"type": "reset"})
                await broadcast({"type": "status", "text": "idle"})
    except WebSocketDisconnect:
        clients.discard(ws)


app.mount("/assets", StaticFiles(directory="web/dist/assets"), name="assets")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  👓 Glasses HUD  →  http://localhost:{port}   (calendar mode: {MODE.upper()})")
    if MODE == "mock" and os.path.exists("credentials.json"):
        print("  ℹ️  Running MOCK (Actions work, but not written to real calendar).")
        print("      Authorize once with `python test_calendar.py` to switch to LIVE.\n")
    else:
        print()
    uvicorn.run(app, host="0.0.0.0", port=port)
