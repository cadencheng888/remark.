"""
ambient_awareness.py

Fuses visual_context.py (camera + vision) and transcribe.py (mic + STT) into
a single loop that outputs a one-sentence situational summary every ~15 seconds.

Two windows open:
  • "Ambient Awareness" — camera feed with YOLO boxes + pose skeleton +
    synthesis summary overlaid at the bottom
  • "Transcript"        — live speaker-labeled transcript with interim text

    python ambient_awareness.py

Prerequisites: all deps from both source files, plus .env with:
    ANTHROPIC_API_KEY=sk-ant-...
    DEEPGRAM_API_KEY=...

Optional flags:
    --camera-index N      (default 0; try 1 or 2 for Continuity Camera / iPhone)
    --no-captioning       skip Claude vision captioning (saves API calls)
    --no-visual           skip camera entirely (audio + transcript window only)
    --no-audio            skip mic entirely (visual only)
    --audio-device N      sounddevice input device index (default = system default)
"""

import argparse
import asyncio
import json
import os
import threading
import time
from collections import deque
from typing import Optional

import anthropic
import cv2
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

from visual_context import (
    CAPTION_INTERVAL_SEC,
    DEFAULT_MODEL_ID,
    DEFAULT_POSE_MODEL_PATH,
    DEFAULT_SERVER_URL,
    MAX_POSES,
    POSE_EVERY_N_FRAMES,
    PROCESS_EVERY_N_FRAMES,
    ImageCaptioner,
    ObjectDetector,
    PoseEstimator,
    VisualContextEngine,
    draw_overlay,
)
from transcribe import CHANNELS, SAMPLE_RATE

load_dotenv()

SYNTHESIS_INTERVAL_SEC = 15
TRANSCRIPT_WINDOW_SEC = 30
SYNTHESIS_MODEL = "claude-haiku-4-5-20251001"

# BGR colors for up to 6 speakers
_SPEAKER_COLORS = [
    (100, 200, 255),  # orange-ish
    (100, 255, 150),  # green
    (255, 150, 100),  # blue
    (200, 100, 255),  # purple
    (80, 220, 220),   # yellow
    (255, 180, 180),  # pink-blue
]

_SYNTHESIS_PROMPT = """\
You are an ambient awareness assistant. Based on the sensor data below, write \
ONE concise sentence (max 30 words) describing what is currently happening in \
the surrounding environment. Be specific and direct. No preamble.

VISUAL SUMMARY (last {window}s):
{visual}

SCENE DESCRIPTION (camera):
{caption}

AUDIO TRANSCRIPT (last {window}s):
{transcript}"""


# ---------------------------------------------------------------------------
# Shared context store
# ---------------------------------------------------------------------------

class ContextStore:
    """Thread-safe state shared between the main display loop and async workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self.visual_summary: Optional[dict] = None
        self.latest_caption: Optional[dict] = None
        self.latest_synthesis: str = ""
        self._transcript: deque = deque()   # (timestamp, speaker_id | None, text)
        self._interim: str = ""

    def add_transcript(self, text: str, speaker: Optional[int] = None):
        with self._lock:
            self._transcript.append((time.time(), speaker, text))
            cutoff = time.time() - TRANSCRIPT_WINDOW_SEC * 2
            while self._transcript and self._transcript[0][0] < cutoff:
                self._transcript.popleft()

    def set_interim(self, text: str):
        with self._lock:
            self._interim = text

    def set_visual_summary(self, summary: dict):
        with self._lock:
            self.visual_summary = summary

    def set_caption(self, caption: dict):
        with self._lock:
            self.latest_caption = caption

    def set_synthesis(self, text: str):
        with self._lock:
            self.latest_synthesis = text

    def snapshot(self) -> dict:
        """Snapshot for synthesis prompt building."""
        now = time.time()
        with self._lock:
            cutoff = now - TRANSCRIPT_WINDOW_SEC
            return {
                "visual_summary": self.visual_summary,
                "latest_caption": self.latest_caption,
                "transcript": [
                    (ts, spk, txt)
                    for ts, spk, txt in self._transcript
                    if ts >= cutoff
                ],
            }

    def display_state(self) -> dict:
        """Snapshot for rendering — avoids holding the lock during drawing."""
        with self._lock:
            return {
                "lines": list(self._transcript)[-18:],
                "interim": self._interim,
                "synthesis": self.latest_synthesis,
            }


# ---------------------------------------------------------------------------
# Transcript window renderer
# ---------------------------------------------------------------------------

_TRANSCRIPT_W = 640
_TRANSCRIPT_H = 480
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _speaker_color(speaker_id: Optional[int]) -> tuple:
    if speaker_id is None:
        return (200, 200, 200)
    return _SPEAKER_COLORS[speaker_id % len(_SPEAKER_COLORS)]


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Very simple word-wrap."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).lstrip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_transcript_window(state: dict) -> np.ndarray:
    canvas = np.zeros((_TRANSCRIPT_H, _TRANSCRIPT_W, 3), dtype=np.uint8)

    # Header
    cv2.putText(canvas, "Transcript", (12, 28), _FONT, 0.65, (200, 200, 200), 1)
    cv2.line(canvas, (0, 38), (_TRANSCRIPT_W, 38), (60, 60, 60), 1)

    # Finals — render bottom-up so newest is always near the bottom
    bottom_y = _TRANSCRIPT_H - 50  # leave room for interim
    lines_to_draw: list[tuple[tuple, list[str]]] = []
    for _, spk, txt in reversed(state["lines"]):
        color = _speaker_color(spk)
        prefix = f"[S{spk}] " if spk is not None else "[?] "
        wrapped = _wrap_text(prefix + txt, max_chars=62)
        lines_to_draw.append((color, list(reversed(wrapped))))

    y = bottom_y
    for color, wrapped_reversed in lines_to_draw:
        for line in wrapped_reversed:
            if y < 48:
                break
            cv2.putText(canvas, line, (12, y), _FONT, 0.42, color, 1)
            y -= 18
        y -= 4  # gap between utterances
        if y < 48:
            break

    # Interim text (gray, bottom)
    interim = state.get("interim", "")
    if interim:
        cv2.line(canvas, (0, _TRANSCRIPT_H - 42), (_TRANSCRIPT_W, _TRANSCRIPT_H - 42), (40, 40, 40), 1)
        cv2.putText(canvas, f"… {interim[:72]}", (12, _TRANSCRIPT_H - 20),
                    _FONT, 0.4, (110, 110, 110), 1)

    return canvas


# ---------------------------------------------------------------------------
# Synthesis overlay on camera frame
# ---------------------------------------------------------------------------

def _draw_synthesis(frame, text: str):
    if not text:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 52), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Word-wrap to fit the frame width (approx 55 chars per line at this font size)
    wrapped = _wrap_text(text, max_chars=int(w / 11))
    y_start = h - 44 + max(0, (2 - len(wrapped)) * 13)
    for i, line in enumerate(wrapped[:2]):
        cv2.putText(frame, line, (10, y_start + i * 20),
                    _FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Visual + display loop — MUST run on the main thread (macOS OpenCV constraint)
# ---------------------------------------------------------------------------

def visual_and_display_loop(
    store: ContextStore,
    stop_event: threading.Event,
    camera_index: int,
    server_url: str,
    model_id: str,
    pose_model_path: str,
    max_poses: int,
    enable_captioning: bool,
    show_visual: bool,
):
    detector: Optional[ObjectDetector] = None
    pose_estimator: Optional[PoseEstimator] = None
    captioner: Optional[ImageCaptioner] = None
    cap: Optional[cv2.VideoCapture] = None

    if show_visual:
        detector = ObjectDetector(server_url=server_url, model_id=model_id)

        try:
            pose_estimator = PoseEstimator(model_path=pose_model_path, max_poses=max_poses)
        except Exception as e:
            print(f"[visual] pose unavailable ({e}); object detection only")

        if enable_captioning:
            try:
                captioner = ImageCaptioner()
            except Exception as e:
                print(f"[visual] captioning unavailable ({e})")

        cap = cv2.VideoCapture(camera_index)
        assert cap is not None
        if not cap.isOpened():
            print(f"[visual] could not open camera {camera_index}")
            cap = None
            show_visual = False

    engine = VisualContextEngine(summary_interval_sec=SYNTHESIS_INTERVAL_SEC)

    frame_count = 0
    latest_detections = []
    latest_poses = []
    last_caption_submit = 0.0
    last_seen_caption_ts = 0.0
    frame: Optional[np.ndarray] = None

    if show_visual:
        print(f"[visual] camera {camera_index}  model={model_id}  "
              f"pose={'on' if pose_estimator else 'off'}  "
              f"captions={'on' if captioner else 'off'}")

    try:
        while not stop_event.is_set():
            # ---- camera processing ----------------------------------------
            if show_visual and cap:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                now = time.time()

                if pose_estimator and frame_count % POSE_EVERY_N_FRAMES == 0:
                    try:
                        latest_poses = pose_estimator.estimate(frame)
                    except Exception:
                        pass

                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    try:
                        latest_detections = detector.detect(frame) if detector else []
                    except Exception:
                        pass
                    engine.update(latest_detections, poses=latest_poses)

                if captioner and (now - last_caption_submit) >= CAPTION_INTERVAL_SEC:
                    if captioner.submit_frame(frame):
                        last_caption_submit = now

                if captioner:
                    cap_result = captioner.get_latest()
                    if cap_result and cap_result["timestamp"] > last_seen_caption_ts:
                        last_seen_caption_ts = cap_result["timestamp"]
                        store.set_caption(cap_result)

                summary = engine.maybe_summarize(
                    latest_caption=captioner.get_latest() if captioner else None
                )
                if summary:
                    store.set_visual_summary(summary)

            # ---- display --------------------------------------------------
            state = store.display_state()

            if show_visual and cap and frame is not None:
                display_frame = frame.copy()
                draw_overlay(display_frame, latest_detections, latest_poses)
                _draw_synthesis(display_frame, state["synthesis"])
                cv2.imshow("Ambient Awareness", display_frame)

            transcript_frame = render_transcript_window(state)
            cv2.imshow("Transcript", transcript_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

            if not show_visual:
                time.sleep(0.033)  # ~30fps tick for transcript window refresh

    finally:
        if cap:
            cap.release()
        if pose_estimator:
            pose_estimator.close()
        if captioner:
            captioner.close()
        cv2.destroyAllWindows()
        print("[visual] stopped")


# ---------------------------------------------------------------------------
# Audio loop with speaker diarization — async
# ---------------------------------------------------------------------------

def _diarized_url() -> str:
    params = {
        "model": "nova-3",
        "language": "en-US",
        "encoding": "linear16",
        "sample_rate": str(SAMPLE_RATE),
        "channels": str(CHANNELS),
        "smart_format": "true",
        "punctuate": "true",
        "interim_results": "true",
        "diarize": "true",
        "endpointing": "300",
    }
    return "wss://api.deepgram.com/v1/listen?" + "&".join(f"{k}={v}" for k, v in params.items())


def _parse_diarized(data: dict) -> Optional[tuple[int, str]]:
    """Return (dominant_speaker_id, full_text) from a diarized final result."""
    try:
        words = data["channel"]["alternatives"][0].get("words", [])
        if not words:
            return None
        speaker_votes: dict[int, int] = {}
        parts = []
        for w in words:
            parts.append(w.get("punctuated_word", w.get("word", "")))
            spk = w.get("speaker", 0)
            speaker_votes[spk] = speaker_votes.get(spk, 0) + 1
        dominant = max(speaker_votes, key=lambda s: speaker_votes[s])
        return dominant, " ".join(parts).strip()
    except (KeyError, IndexError):
        return None


async def audio_loop(store: ContextStore, device: Optional[int] = None):
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("[audio] DEEPGRAM_API_KEY not set — skipping audio stream")
        return

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def audio_callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    try:
        ws_ctx = websockets.connect(
            _diarized_url(), additional_headers={"Authorization": f"Token {api_key}"}
        )
    except TypeError:
        ws_ctx = websockets.connect(
            _diarized_url(), extra_headers={"Authorization": f"Token {api_key}"}
        )

    async with ws_ctx as ws:
        print("[audio] connected to Deepgram (diarization on)")

        async def sender():
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=4000,
                device=device,
                callback=audio_callback,
            )
            with stream:
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)

        async def receiver():
            async for message in ws:
                data = json.loads(message)
                if data.get("type") != "Results":
                    continue
                alt = data["channel"]["alternatives"][0]
                text = alt.get("transcript", "").strip()
                if not text:
                    continue
                if data.get("is_final"):
                    store.set_interim("")
                    result = _parse_diarized(data)
                    if result:
                        speaker, final_text = result
                        store.add_transcript(final_text, speaker=speaker)
                else:
                    store.set_interim(text)

        async def keepalive():
            while True:
                await asyncio.sleep(8)
                await ws.send(json.dumps({"type": "KeepAlive"}))

        await asyncio.gather(sender(), receiver(), keepalive())


# ---------------------------------------------------------------------------
# Synthesis loop — async, fires every SYNTHESIS_INTERVAL_SEC
# ---------------------------------------------------------------------------

def _format_prompt(snap: dict) -> str:
    visual = snap["visual_summary"]
    caption = snap["latest_caption"]
    transcript = snap["transcript"]

    visual_text = visual["text"] if visual else "no visual data yet"
    caption_text = caption["text"] if caption else "no scene description yet"
    if transcript:
        lines = [
            f"[Speaker {spk}]: {txt}" if spk is not None else f"[Speaker]: {txt}"
            for _, spk, txt in transcript
        ]
        transcript_text = "\n".join(lines)
    else:
        transcript_text = "no speech detected"

    return _SYNTHESIS_PROMPT.format(
        window=SYNTHESIS_INTERVAL_SEC,
        visual=visual_text,
        caption=caption_text,
        transcript=transcript_text,
    )


async def synthesis_loop(store: ContextStore):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[synthesis] ANTHROPIC_API_KEY not set — skipping synthesis")
        return

    claude = anthropic.Anthropic(api_key=api_key)
    await asyncio.sleep(SYNTHESIS_INTERVAL_SEC)  # let sensors warm up

    while True:
        tick_start = time.monotonic()
        snap = store.snapshot()
        try:
            prompt = _format_prompt(snap)
            response = await asyncio.to_thread(
                lambda p=prompt: claude.messages.create(
                    model=SYNTHESIS_MODEL,
                    max_tokens=80,
                    messages=[{"role": "user", "content": p}],
                )
            )
            text = response.content[0].text.strip()
            store.set_synthesis(text)
            print(f"\n{'─' * 60}\n🌍  {text}\n{'─' * 60}\n")
        except Exception as e:
            print(f"[synthesis] error: {e}")

        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, SYNTHESIS_INTERVAL_SEC - elapsed))


# ---------------------------------------------------------------------------
# Async worker thread
# ---------------------------------------------------------------------------

def _run_async_workers(store: ContextStore, args, stop_event: threading.Event):
    """Audio + synthesis run in a background thread's event loop."""
    async def _main():
        tasks = []
        if not args.no_audio:
            tasks.append(asyncio.create_task(
                audio_loop(store, device=args.audio_device)
            ))
        tasks.append(asyncio.create_task(synthesis_loop(store)))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()
        stop_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ambient environment awareness pipeline")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL_PATH)
    parser.add_argument("--max-poses", type=int, default=MAX_POSES)
    parser.add_argument("--no-captioning", action="store_true",
                        help="Disable Claude vision captioning")
    parser.add_argument("--no-visual", action="store_true",
                        help="Skip camera (transcript window still shows)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip microphone")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="sounddevice input device index (run with --list-devices to find)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    store = ContextStore()
    stop_event = threading.Event()

    async_thread = threading.Thread(
        target=_run_async_workers,
        args=(store, args, stop_event),
        daemon=True,
    )
    async_thread.start()

    try:
        # OpenCV GUI must run on the main thread on macOS
        visual_and_display_loop(
            store=store,
            stop_event=stop_event,
            camera_index=args.camera_index,
            server_url=args.server_url,
            model_id=args.model_id,
            pose_model_path=args.pose_model,
            max_poses=args.max_poses,
            enable_captioning=not args.no_captioning,
            show_visual=not args.no_visual,
        )
    except KeyboardInterrupt:
        print("\n👋 stopping…")
        stop_event.set()

    async_thread.join(timeout=3)
    print("done")


if __name__ == "__main__":
    main()
