"""
visual_context.py

Captures frames from a camera (e.g. iPhone via Continuity Camera), runs
local object detection via Roboflow's inference server, and emits two
kinds of text context:

1. Lightweight continuous events  -> emitted whenever the detected scene
   changes (objects appear / disappear / move significantly)
2. Periodic rich summaries        -> emitted every SUMMARY_INTERVAL_SEC,
   rolling up everything seen in that window into a natural-language-
   ready context blob

Both are designed to be cheap, agent-ready "context objects" (dicts) that
you can hand off to an LLM later -- no raw frames or video leave this
process.

Prereqs (run on your Mac, NOT in this sandbox):
    pip install inference opencv-python mediapipe anthropic
    # Roboflow inference server needs Docker running:
    inference server start
    # MediaPipe pose model (one-time download):
    curl -L -o pose_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
    # Image captioning needs a Claude API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Then run this script in another terminal:
    python visual_context.py --camera-index 0
"""

import argparse
import base64
import math
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
from inference_sdk import InferenceHTTPClient
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DEFAULT_MODEL_ID = "yolov8n-640"          # general pretrained COCO model
DEFAULT_SERVER_URL = "http://localhost:9001"  # local inference server
DEFAULT_POSE_MODEL_PATH = "pose_landmarker.task"
CAPTION_MODEL = "claude-sonnet-4-6"
CONFIDENCE_THRESHOLD = 0.45
PROCESS_EVERY_N_FRAMES = 5                # throttle object detection vs. camera fps
POSE_EVERY_N_FRAMES = 1                   # pose is cheap; run every frame for smooth skeleton
MAX_POSES = 1                            # max people to track at once (cap 3-5 recommended) # 1 for performance
SUMMARY_INTERVAL_SEC = 15                 # how often to emit a rich summary
CAPTION_INTERVAL_SEC = 15                 # how often to sample a frame for VLM captioning
MIN_EVENT_GAP_SEC = 1.0                   # don't spam lightweight events


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PoseInfo:
    """Derived high-level features from one person's pose landmarks."""
    person_id: int        # stable-ish ID assigned via nearest-centroid tracking
    posture: str           # "standing" | "sitting" | "unknown"
    hands_raised: bool
    hand_near_face: bool
    landmarks: object = None  # raw NormalizedLandmark list, for drawing

    @property
    def centroid(self) -> tuple[float, float]:
        """Rough body center (hip midpoint) used for frame-to-frame ID matching."""
        if not self.landmarks:
            return (0.5, 0.5)
        l_hip, r_hip = self.landmarks[_L_HIP], self.landmarks[_R_HIP]
        return ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)

    def descriptor(self) -> str:
        """Compact string used for diffing this person's state across frames."""
        parts = [self.posture]
        if self.hands_raised:
            parts.append("hands_raised")
        if self.hand_near_face:
            parts.append("hand_near_face")
        return "|".join(parts)

    def human_text(self) -> str:
        """Human-readable description for context emission."""
        bits = []
        if self.posture != "unknown":
            bits.append(self.posture)
        if self.hands_raised:
            bits.append("hands raised")
        if self.hand_near_face:
            bits.append("hand near face")
        return ", ".join(bits) if bits else "person detected"


@dataclass
class SceneState:
    """Tracks the most recent detection snapshot so we can diff."""
    counts: Counter = field(default_factory=Counter)
    detections: list[Detection] = field(default_factory=list)
    pose_descriptors: dict[int, str] = field(default_factory=dict)  # person_id -> descriptor
    timestamp: float = 0.0


# ----------------------------------------------------------------------
# Detection client
# ----------------------------------------------------------------------

class ObjectDetector:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL, model_id: str = DEFAULT_MODEL_ID):
        self.client = InferenceHTTPClient(api_url=server_url, api_key=None)
        self.model_id = model_id

    def detect(self, frame) -> list[Detection]:
        # inference_sdk expects either a path or a numpy array (BGR is fine,
        # it handles conversion internally for local inference)
        result = self.client.infer(frame, model_id=self.model_id)

        detections = []
        for pred in result.get("predictions", []):
            if pred["confidence"] < CONFIDENCE_THRESHOLD:
                continue
            detections.append(
                Detection(
                    label=pred["class"],
                    confidence=pred["confidence"],
                    x=pred["x"],
                    y=pred["y"],
                    width=pred["width"],
                    height=pred["height"],
                )
            )
        return detections


# ----------------------------------------------------------------------
# Periodic image captioning (Claude vision, background thread)
# ----------------------------------------------------------------------

class ImageCaptioner:
    """
    Periodically sends a single sampled frame to Claude for an open-ended
    natural-language scene description, complementing the structured
    object/pose signals with something that can describe *activity* and
    *context* ("looks like they're cooking pasta") that geometry alone
    can't express.

    Runs in a background thread so a slow API round-trip (typically
    1-3s) never blocks the camera capture loop. Only ever has at most
    one request in flight -- if a new frame is submitted while a caption
    is still being generated, it's dropped (we only care about "what's
    happening roughly now", not a backlog of stale frames).
    """

    PROMPT = (
        "You are an ambient assistant observing a room through a camera. "
        "Describe in 1-2 concise sentences what is happening in this scene "
        "right now -- focus on activity and context (what the person/people "
        "appear to be doing), not just a list of objects. Be direct and "
        "factual, no preamble."
    )

    def __init__(self, model: str = CAPTION_MODEL, api_key: Optional[str] = None):
        # Imported lazily so the rest of the script still works if the
        # `anthropic` package or API key isn't set up yet.
        import anthropic
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Run `export ANTHROPIC_API_KEY=sk-ant-...` "
                "or pass api_key= explicitly."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._latest_caption: Optional[dict] = None
        self._lock = threading.Lock()
        self._busy = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame_bgr) -> bool:
        """
        Queue a frame for captioning. Non-blocking: if a caption request
        is already in flight, this drops the new frame and returns False
        rather than piling up a backlog of stale requests.
        """
        if self._busy:
            return False
        # Encode here (cheap, on the caller's thread) so the worker thread
        # only does the network call.
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return False
        b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
        try:
            self._inbox.put_nowait(b64)
            return True
        except queue.Full:
            return False

    def get_latest(self) -> Optional[dict]:
        """Returns the most recent caption result dict, or None if none yet."""
        with self._lock:
            return self._latest_caption

    def close(self):
        self._stop.set()
        # unblock the worker if it's waiting on the queue
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)

    # -- worker thread -----------------------------------------------------

    def _worker(self):
        while not self._stop.is_set():
            b64 = self._inbox.get()
            if b64 is None:  # shutdown sentinel
                return
            self._busy = True
            try:
                caption_text = self._call_api(b64)
                with self._lock:
                    self._latest_caption = {
                        "type": "caption",
                        "timestamp": time.time(),
                        "text": caption_text,
                    }
            except Exception as e:
                print(f"[visual_context] captioning error: {e}")
            finally:
                self._busy = False

    def _call_api(self, b64_jpeg: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_jpeg,
                        },
                    },
                    {"type": "text", "text": self.PROMPT},
                ],
            }],
        )
        # response.content is a list of blocks; we only sent a text prompt
        # so we expect a single text block back.
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return " ".join(parts).strip()


# ----------------------------------------------------------------------
# Pose estimator (MediaPipe Tasks API)
# ----------------------------------------------------------------------

# MediaPipe pose landmark indices (subset we use)
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28


class PoseEstimator:
    """
    Wraps MediaPipe Pose Landmarker (Tasks API, VIDEO mode) and turns
    raw 33 landmarks into a small set of high-level features the agent
    can actually reason about. Supports multiple people (up to MAX_POSES)
    with a lightweight nearest-centroid ID assignment across frames --
    not a real tracker, but enough to keep "person 2" referring to
    roughly the same person between calls as long as they don't cross
    paths or leave/re-enter rapidly.
    """

    VISIBILITY_THRESHOLD = 0.5
    MAX_MATCH_DISTANCE = 0.25  # normalized coords; beyond this, treat as a new person

    def __init__(self, model_path: str = DEFAULT_POSE_MODEL_PATH, max_poses: int = MAX_POSES):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=max_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.time()

        # id tracking state
        self._next_id = 0
        self._prev_centroids: dict[int, tuple[float, float]] = {}

    def estimate(self, frame_bgr) -> list[PoseInfo]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            self._prev_centroids = {}
            return []

        # Build raw (centroid, landmarks) pairs for this frame first,
        # then assign IDs by matching against last frame's centroids.
        raw = []
        for landmarks in result.pose_landmarks:
            l_hip, r_hip = landmarks[_L_HIP], landmarks[_R_HIP]
            centroid = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)
            raw.append((centroid, landmarks))

        assigned_ids = self._match_ids(raw)

        poses = []
        for person_id, (centroid, landmarks) in zip(assigned_ids, raw):
            poses.append(PoseInfo(
                person_id=person_id,
                posture=self._infer_posture(landmarks),
                hands_raised=self._hands_raised(landmarks),
                hand_near_face=self._hand_near_face(landmarks),
                landmarks=landmarks,
            ))

        self._prev_centroids = {pid: c for pid, (c, _) in zip(assigned_ids, raw)}
        return poses

    def _match_ids(self, raw: list[tuple[tuple[float, float], object]]) -> list[int]:
        """Greedy nearest-centroid matching against last frame's people."""
        available_prev = dict(self._prev_centroids)  # id -> centroid, consumed as matched
        assigned: list[Optional[int]] = [None] * len(raw)

        # Compute all (frame_idx, prev_id, distance) pairs, sort by distance,
        # greedily assign closest pairs first so two new detections don't
        # both fight over the same previous ID.
        candidates = []
        for i, (centroid, _) in enumerate(raw):
            for pid, prev_centroid in available_prev.items():
                d = math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1])
                if d <= self.MAX_MATCH_DISTANCE:
                    candidates.append((d, i, pid))
        candidates.sort(key=lambda c: c[0])

        used_prev_ids = set()
        for d, i, pid in candidates:
            if assigned[i] is not None or pid in used_prev_ids:
                continue
            assigned[i] = pid
            used_prev_ids.add(pid)

        # Anything unmatched gets a fresh ID
        for i in range(len(raw)):
            if assigned[i] is None:
                assigned[i] = self._next_id
                self._next_id += 1

        return assigned

    # -- derived features -------------------------------------------------

    def _infer_posture(self, lm) -> str:
        """Infer standing vs sitting from knee joint angle (hip-knee-ankle)."""
        for hip_i, knee_i, ankle_i in [(_R_HIP, _R_KNEE, _R_ANKLE),
                                        (_L_HIP, _L_KNEE, _L_ANKLE)]:
            hip, knee, ankle = lm[hip_i], lm[knee_i], lm[ankle_i]
            if min(hip.visibility, knee.visibility, ankle.visibility) < self.VISIBILITY_THRESHOLD:
                continue
            ang = _angle_deg(hip, knee, ankle)
            if ang > 150:
                return "standing"
            if ang < 120:
                return "sitting"
            return "unknown"
        # Lower body not visible (typical when sitting at a desk) -> can't tell
        return "unknown"

    def _hands_raised(self, lm) -> bool:
        """A hand is 'raised' if its wrist is clearly above its shoulder."""
        # MediaPipe normalized coords: y grows downward, so smaller y = higher up.
        for wrist_i, shoulder_i in [(_L_WRIST, _L_SHOULDER), (_R_WRIST, _R_SHOULDER)]:
            w, s = lm[wrist_i], lm[shoulder_i]
            if w.visibility < self.VISIBILITY_THRESHOLD or s.visibility < self.VISIBILITY_THRESHOLD:
                continue
            if w.y < s.y - 0.1:
                return True
        return False

    def _hand_near_face(self, lm) -> bool:
        nose = lm[_NOSE]
        if nose.visibility < self.VISIBILITY_THRESHOLD:
            return False
        for wrist_i in [_L_WRIST, _R_WRIST]:
            w = lm[wrist_i]
            if w.visibility < self.VISIBILITY_THRESHOLD:
                continue
            d = math.hypot(w.x - nose.x, w.y - nose.y)
            if d < 0.15:
                return True
        return False

    def close(self):
        self.landmarker.close()


def _angle_deg(a, b, c) -> float:
    """Interior angle at point b formed by a-b-c, in degrees."""
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    dot = v1x * v2x + v1y * v2y
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos))


# ----------------------------------------------------------------------
# Context generation
# ----------------------------------------------------------------------

class VisualContextEngine:
    """
    Owns scene-state diffing and summary rollups. Pure logic, no I/O,
    so it's easy to unit test and easy to swap the detector out later.
    """

    def __init__(self, summary_interval_sec: float = SUMMARY_INTERVAL_SEC):
        self.prev_state = SceneState()
        self.summary_interval_sec = summary_interval_sec
        self.window_start = time.time()
        self.window_label_counts: Counter = Counter()
        self.window_event_log: list[str] = []
        self.last_event_time = 0.0

    # -- lightweight continuous events -----------------------------------

    def update(self, detections: list[Detection],
               poses: Optional[list[PoseInfo]] = None) -> list[dict]:
        """
        Call this on every processed frame. Returns a list of events for
        meaningful changes in the scene since the last call: object diff
        first, then one pose-state diff per person (appeared, changed, or
        left). The list is empty if nothing changed.
        """
        poses = poses or []
        now = time.time()
        counts = Counter(d.label for d in detections)
        self.window_label_counts.update(counts)

        new_events: list[dict] = []

        obj_event = self._diff(self.prev_state.counts, counts, now)
        if obj_event:
            new_events.append(obj_event)

        new_descriptors: dict[int, str] = {p.person_id: p.descriptor() for p in poses}
        poses_by_id = {p.person_id: p for p in poses}

        all_ids = set(self.prev_state.pose_descriptors) | set(new_descriptors)
        for pid in sorted(all_ids):
            prev_desc = self.prev_state.pose_descriptors.get(pid)
            curr_desc = new_descriptors.get(pid)
            pose_event = self._diff_pose(prev_desc, curr_desc, poses_by_id.get(pid), pid, now)
            if pose_event:
                new_events.append(pose_event)

        self.prev_state = SceneState(
            counts=counts,
            detections=detections,
            pose_descriptors=new_descriptors,
            timestamp=now,
        )

        # Always log real changes into the window for the periodic summary,
        # even if we throttle the live emissions below.
        emitted: list[dict] = []
        # Snapshot the gap check once per update() call: events that fire
        # together on the same frame should all emit (or all be throttled)
        # together, rather than the first event eating the slot.
        gap_ok = (now - self.last_event_time) >= MIN_EVENT_GAP_SEC
        for ev in new_events:
            self.window_event_log.append(ev["text"])
            if gap_ok:
                emitted.append(ev)
        if emitted:
            self.last_event_time = now
        return emitted

    def _diff(self, prev: Counter, curr: Counter, now: float) -> Optional[dict]:
        appeared = curr - prev   # labels with higher count now
        disappeared = prev - curr  # labels with higher count before

        if not appeared and not disappeared:
            return None

        parts = []
        if appeared:
            parts.append("appeared: " + ", ".join(f"{n}x {label}" for label, n in appeared.items()))
        if disappeared:
            parts.append("left: " + ", ".join(f"{n}x {label}" for label, n in disappeared.items()))

        text = "; ".join(parts)
        return {
            "type": "event",
            "timestamp": now,
            "text": text,
            "appeared": dict(appeared),
            "disappeared": dict(disappeared),
            "current_scene": dict(curr),
        }

    def _diff_pose(self, prev: Optional[str], curr: Optional[str],
                   pose: Optional[PoseInfo], person_id: int, now: float) -> Optional[dict]:
        if curr == prev:
            return None
        label = f"person {person_id}"
        if curr is None:
            # Person stopped being detected -- only emit if previously
            # there *was* a non-trivial pose state
            if prev is None:
                return None
            return {
                "type": "pose_event",
                "timestamp": now,
                "person_id": person_id,
                "text": f"{label} no longer detected",
                "from": prev,
                "to": None,
            }
        # Suppress transitions in/out of "unknown" -- they're usually
        # visibility flicker (e.g. lower body briefly occluded), not a
        # real state change.
        if prev is not None and ("unknown" in prev or "unknown" in curr):
            # still update internal state, but don't emit a noisy event
            return None
        text = (
            f"{label} posture: {pose.human_text()}"
            if prev is None
            else f"{label} changed: {prev} -> {pose.human_text()}"
        )
        return {
            "type": "pose_event",
            "timestamp": now,
            "person_id": person_id,
            "text": text,
            "from": prev,
            "to": curr,
        }

    # -- periodic rich summary -------------------------------------------

    def maybe_summarize(self, latest_caption: Optional[dict] = None) -> Optional[dict]:
        now = time.time()
        if now - self.window_start < self.summary_interval_sec:
            return None

        summary = self._build_summary(now, latest_caption)
        self.window_start = now
        self.window_label_counts = Counter()
        self.window_event_log = []
        return summary

    def _build_summary(self, now: float, latest_caption: Optional[dict] = None) -> dict:
        if self.window_label_counts:
            top_objects = self.window_label_counts.most_common()
            objects_text = ", ".join(f"{label} (seen {n}x)" for label, n in top_objects)
        else:
            objects_text = "no objects detected"

        current_scene = dict(self.prev_state.counts)
        current_text = (
            ", ".join(f"{n} {label}" for label, n in current_scene.items())
            if current_scene else "scene currently empty"
        )

        if self.prev_state.pose_descriptors:
            people_bits = [
                f"person {pid}: {desc.replace('|', ', ')}"
                for pid, desc in sorted(self.prev_state.pose_descriptors.items())
            ]
            pose_text = " Currently tracked: " + "; ".join(people_bits) + "."
        else:
            pose_text = ""

        narrative = (
            f"Over the last {self.summary_interval_sec:.0f}s: {objects_text}. "
            f"Currently in frame: {current_text}.{pose_text}"
        )
        if self.window_event_log:
            narrative += " Notable changes: " + " | ".join(self.window_event_log)

        caption_text = None
        if latest_caption:
            caption_text = latest_caption["text"]
            caption_age = now - latest_caption["timestamp"]
            narrative += f" Scene description: {caption_text} (captured {caption_age:.0f}s ago)."

        return {
            "type": "summary",
            "timestamp": now,
            "window_sec": self.summary_interval_sec,
            "text": narrative,
            "object_counts_in_window": dict(self.window_label_counts),
            "current_scene": current_scene,
            "current_poses": dict(self.prev_state.pose_descriptors),
            "caption": caption_text,
            "events_in_window": list(self.window_event_log),
        }


# ----------------------------------------------------------------------
# Capture loop
# ----------------------------------------------------------------------

def run(camera_index: int, server_url: str, model_id: str,
        pose_model_path: str, max_poses: int, enable_captioning: bool,
        show_preview: bool):
    detector = ObjectDetector(server_url=server_url, model_id=model_id)
    pose_estimator: Optional[PoseEstimator] = None
    try:
        pose_estimator = PoseEstimator(model_path=pose_model_path, max_poses=max_poses)
    except Exception as e:
        # Don't kill the whole run if mediapipe / model file isn't set up.
        # Object detection still works on its own.
        print(f"[visual_context] pose estimator unavailable ({e}); "
              f"continuing with object detection only")

    captioner: Optional[ImageCaptioner] = None
    if enable_captioning:
        try:
            captioner = ImageCaptioner()
        except Exception as e:
            print(f"[visual_context] captioning unavailable ({e}); "
                  f"continuing without scene captions")

    engine = VisualContextEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            f"On Mac, check System Settings > camera permissions, and try "
            f"other indices (0, 1, 2...) -- Continuity Camera doesn't "
            f"always land on 0."
        )

    frame_count = 0
    latest_detections: list[Detection] = []
    latest_poses: list[PoseInfo] = []
    last_caption_submit_time = 0.0
    last_seen_caption_ts = 0.0  # to detect when a new caption result lands
    print(f"[visual_context] capturing from camera {camera_index}, "
          f"model={model_id}, server={server_url}, "
          f"pose={'on (max ' + str(max_poses) + ' people)' if pose_estimator else 'off'}, "
          f"captioning={'on' if captioner else 'off'}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[visual_context] frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()

            # Pose runs more often than object detection (it's cheap, and
            # smoother skeleton tracking looks better on the preview).
            if pose_estimator and frame_count % POSE_EVERY_N_FRAMES == 0:
                try:
                    latest_poses = pose_estimator.estimate(frame)
                except Exception as e:
                    print(f"[visual_context] pose error: {e}")

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    latest_detections = detector.detect(frame)
                except Exception as e:
                    print(f"[visual_context] detection error: {e}")
                    # keep last good detections rather than blanking

                events = engine.update(latest_detections, poses=latest_poses)
                for event in events:
                    emit(event)

            # Submit a frame for captioning on its own cadence. submit_frame()
            # is non-blocking and silently drops the request if the previous
            # one is still in flight, so this is safe to call every loop.
            if captioner and (now - last_caption_submit_time) >= CAPTION_INTERVAL_SEC:
                if captioner.submit_frame(frame):
                    last_caption_submit_time = now

            # Surface a new caption result as a standalone event the moment
            # it lands (independent of the summary cadence), since the API
            # call can complete at any point in the window.
            latest_caption = captioner.get_latest() if captioner else None
            if latest_caption and latest_caption["timestamp"] > last_seen_caption_ts:
                last_seen_caption_ts = latest_caption["timestamp"]
                emit(latest_caption)

            summary = engine.maybe_summarize(latest_caption=latest_caption)
            if summary:
                emit(summary)

            if show_preview:
                # Redraw every frame using the most recent state so neither
                # boxes nor skeletons flicker between inference frames.
                draw_overlay(frame, latest_detections, latest_poses)
                cv2.imshow("visual_context preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if pose_estimator:
            pose_estimator.close()
        if captioner:
            captioner.close()
        if show_preview:
            cv2.destroyAllWindows()


# Subset of MediaPipe pose connections (33 landmarks, indices listed at top).
# Skipping facial/finger connections since we only care about body posture.
_POSE_CONNECTIONS = [
    (11, 12),                     # shoulders
    (11, 13), (13, 15),           # left arm
    (12, 14), (14, 16),           # right arm
    (11, 23), (12, 24), (23, 24), # torso
    (23, 25), (25, 27),           # left leg
    (24, 26), (26, 28),           # right leg
]

# Distinct colors (BGR) so each tracked person is visually distinguishable
_PERSON_COLORS = [
    (255, 180, 0),   # orange-blue
    (0, 220, 255),   # yellow
    (255, 0, 200),   # magenta
    (0, 255, 120),   # green
    (255, 120, 120), # light blue
]


def _color_for(person_id: int) -> tuple[int, int, int]:
    return _PERSON_COLORS[person_id % len(_PERSON_COLORS)]


def draw_overlay(frame, detections: list[Detection], poses: Optional[list[PoseInfo]] = None):
    # object boxes
    for d in detections:
        x1 = int(d.x - d.width / 2)
        y1 = int(d.y - d.height / 2)
        x2 = int(d.x + d.width / 2)
        y2 = int(d.y + d.height / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # pose skeletons, one color per tracked person
    h, w = frame.shape[:2]
    for i, pose in enumerate(poses or []):
        if not pose.landmarks:
            continue
        lm = pose.landmarks
        color = _color_for(pose.person_id)
        for p in lm:
            if p.visibility < 0.5:
                continue
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 3, color, -1)
        for a, b in _POSE_CONNECTIONS:
            pa, pb = lm[a], lm[b]
            if pa.visibility < 0.5 or pb.visibility < 0.5:
                continue
            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)
            cv2.line(frame, (ax, ay), (bx, by), color, 2)

        # label near this person's head (nose landmark), falling back to
        # the top-left stacked by index if the nose isn't visible
        nose = lm[_NOSE]
        label = f"#{pose.person_id} {pose.human_text()}"
        if nose.visibility >= 0.5:
            lx, ly = int(nose.x * w), max(20, int(nose.y * h) - 20)
        else:
            lx, ly = 10, 30 + i * 25
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)


def emit(context: dict):
    """
    This is the hand-off point to your agent. For now, just print.
    Swap this out for: a queue, a websocket send, a function call into
    your agent loop, etc.
    """
    tag_map = {
        "event": "EVENT  ",
        "pose_event": "POSE   ",
        "caption": "CAPTION",
        "summary": "SUMMARY",
    }
    tag = tag_map.get(context["type"], "OTHER  ")
    print(f"[{tag}] {context['text']}")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL_PATH,
                        help="Path to MediaPipe pose_landmarker.task file. "
                             "Use the lite variant for real-time.")
    parser.add_argument("--max-poses", type=int, default=MAX_POSES,
                        help="Max number of people to track at once (recommended 3-5).")
    parser.add_argument("--no-captioning", action="store_true",
                        help="Disable periodic Claude vision captioning "
                             "(useful if you don't have ANTHROPIC_API_KEY set).")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    run(
        camera_index=args.camera_index,
        server_url=args.server_url,
        model_id=args.model_id,
        pose_model_path=args.pose_model,
        max_poses=args.max_poses,
        enable_captioning=not args.no_captioning,
        show_preview=not args.no_preview,
    )