from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk

from user_in_the_loop_training import FeedbackManager
from utils.affect import AFFECT_COLUMNS, AFFECT_DISPLAY_NAMES, AFFECT_LEVELS


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEFAULT_SEQ_LEN = 30
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 4

WINDOW_MIN_WIDTH = 620
WINDOW_MIN_HEIGHT = 620
WINDOW_EDGE_MARGIN = 16
WINDOW_VERTICAL_MARGIN = 24
CAMERA_PANEL_WIDTH = 760
ENGAGEMENT_PANEL_WIDTH = 530
PREVIEW_STAGE_HEIGHT = 412
PREVIEW_STAGE_COMPACT_HEIGHT = 336
ENGAGEMENT_PANEL_HEIGHT = 532
PRIMARY_DECISION_BOX_HEIGHT = 260
STAT_BLOCK_HEIGHT = 96
SPOTLIGHT_BOX_HEIGHT = 132
POMODORO_CARD_HEIGHT = 182
TOP_BAR_STACK_BREAKPOINT = 1240
MAIN_STACK_BREAKPOINT = 1180
TILE_MIN_WIDTH = 220

SPOTLIGHT_THRESHOLD = 0.48
PRIMARY_CONFIDENCE_THRESHOLD = 0.58
INFERENCE_INTERVAL_SEC = 0.18
DISPLAY_BLEND = 0.28
OUTPUT_HISTORY_WINDOW = 8
STATE_SWITCH_PATIENCE = 4
SPOTLIGHT_SWITCH_PATIENCE = 4
MIRROR_PREVIEW = True
CAMERA_READ_FAILURE_PATIENCE = 8
CAMERA_RECOVERY_DELAY_MS = 180
INFERENCE_STALE_FRAME_TOLERANCE = 18
SUMMARY_WINDOWS_MINUTES = (4, 8)
SUMMARY_MAX_WINDOW_SEC = SUMMARY_WINDOWS_MINUTES[-1] * 60
POMODORO_TOTAL_MINUTES = 24
POMODORO_BLOCK_MINUTES = 8
POMODORO_TOTAL_SECONDS = POMODORO_TOTAL_MINUTES * 60
POMODORO_BLOCK_SECONDS = POMODORO_BLOCK_MINUTES * 60
POMODORO_FRAME_SAMPLE_INTERVAL_SEC = 4.0
POMODORO_FEEDBACK_SOURCE = "pomodoro_checkin"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MULTI_HEAD_NAMES = list(AFFECT_COLUMNS)

MODEL_VARIANTS = {
    "engagement": {"stem": "mobilenetv2_tcn_distilled", "num_heads": 1},
    "multiaffect": {"stem": "mobilenetv2_tcn_multiaffect_distilled", "num_heads": len(DEFAULT_MULTI_HEAD_NAMES)},
}

COLORS = {
    "bg": "#09111f",
    "panel": "#0d1729",
    "panel_alt": "#111f35",
    "panel_soft": "#15233d",
    "border": "#223655",
    "border_soft": "#1a2a45",
    "text": "#edf3fb",
    "text_soft": "#93a8c5",
    "text_muted": "#6f84a3",
    "preview": "#040916",
    "green": "#31c48d",
    "green_soft": "#18372f",
    "amber": "#f6c453",
    "amber_soft": "#382d14",
    "red": "#ff7070",
    "red_soft": "#3a1d24",
    "blue": "#68b5ff",
    "blue_soft": "#162d47",
    "orange": "#ff9d5c",
    "orange_soft": "#382311",
    "purple": "#b892ff",
    "purple_soft": "#2d2141",
}

HEAD_PALETTES = {
    "engagement": {"accent": COLORS["green"], "surface": COLORS["green_soft"]},
    "not_engaged": {"accent": COLORS["red"], "surface": COLORS["red_soft"]},
    "boredom": {"accent": COLORS["amber"], "surface": COLORS["amber_soft"]},
    "confusion": {"accent": COLORS["blue"], "surface": COLORS["blue_soft"]},
    "frustration": {"accent": COLORS["orange"], "surface": COLORS["orange_soft"]},
    "default": {"accent": COLORS["purple"], "surface": COLORS["purple_soft"]},
}

STATE_STYLES = {
    "idle": {"accent": COLORS["text_soft"], "surface": COLORS["panel_soft"], "badge": "Idle"},
    "warming_up": {"accent": COLORS["amber"], "surface": COLORS["amber_soft"], "badge": "Warming Up"},
    "live_engaged": {"accent": COLORS["green"], "surface": COLORS["green_soft"], "badge": "Live / Engaged"},
    "live_not_engaged": {"accent": COLORS["red"], "surface": COLORS["red_soft"], "badge": "Live / Not Engaged"},
    "live_mixed": {"accent": COLORS["blue"], "surface": COLORS["blue_soft"], "badge": "Live / Mixed"},
    "error": {"accent": COLORS["red"], "surface": COLORS["red_soft"], "badge": "Error"},
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def mix_color(source: str, target: str, amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    s_r, s_g, s_b = _hex_to_rgb(source)
    t_r, t_g, t_b = _hex_to_rgb(target)
    return "#{:02x}{:02x}{:02x}".format(
        int(s_r + (t_r - s_r) * amount),
        int(s_g + (t_g - s_g) * amount),
        int(s_b + (t_b - s_b) * amount),
    )


def _normalize_head_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _display_head_name(name: str) -> str:
    cleaned = str(name).strip()
    return AFFECT_DISPLAY_NAMES.get(cleaned, cleaned.replace("_", " ").title())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(exp, 1.0 / len(exp), dtype=np.float32)
    return (exp / total).astype(np.float32)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _onnx_meta_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(f"{onnx_path.stem}_onnx_meta.json")


def _load_onnx_export_meta(onnx_path: Path) -> dict[str, Any]:
    return _load_json(_onnx_meta_path(onnx_path))


def _variant_artifacts_exist(variant: str) -> bool:
    stem = MODEL_VARIANTS[variant]["stem"]
    for suffix in (".onnx", ".ts", ".pt"):
        if (MODEL_DIR / f"{stem}{suffix}").exists():
            return True
    return False


def _resolve_model_variant(variant: str = "auto") -> str:
    if variant != "auto":
        return variant
    if _variant_artifacts_exist("multiaffect"):
        return "multiaffect"
    return "engagement"


def ensure_onnx_model(variant: str = "auto") -> tuple[Path, str]:
    resolved_variant = _resolve_model_variant(variant)
    stem = MODEL_VARIANTS[resolved_variant]["stem"]
    onnx_path = MODEL_DIR / f"{stem}.onnx"
    if onnx_path.exists():
        return onnx_path, resolved_variant

    previous = os.environ.get("REQUIRE_CUDA")
    os.environ["REQUIRE_CUDA"] = "0"
    try:
        from onxx_port import export_onnx

        export_onnx(onnx_path, variant=resolved_variant)
    finally:
        if previous is None:
            os.environ.pop("REQUIRE_CUDA", None)
        else:
            os.environ["REQUIRE_CUDA"] = previous

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export failed for {resolved_variant}: {onnx_path}")
    return onnx_path, resolved_variant


def _available_providers() -> tuple[list[str], str]:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA"
    return ["CPUExecutionProvider"], "CPU"


def load_model(variant: str = "auto") -> dict[str, Any]:
    onnx_path, resolved_variant = ensure_onnx_model(variant)
    meta = _load_onnx_export_meta(onnx_path)
    variant_name = str(meta.get("variant") or resolved_variant)
    providers, device_label = _available_providers()
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    try:
        session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
    except Exception:
        session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        device_label = "CPU"

    input_name = str(meta.get("input_name") or session.get_inputs()[0].name)
    output_name = str(meta.get("output_name") or session.get_outputs()[0].name)
    class_count = int(meta.get("num_classes") or DEFAULT_NUM_CLASSES)
    seq_len = int(meta.get("seq_len") or DEFAULT_SEQ_LEN)
    img_size = int(meta.get("img_size") or DEFAULT_IMG_SIZE)
    head_count = int(meta.get("num_heads") or MODEL_VARIANTS.get(variant_name, MODEL_VARIANTS["engagement"])["num_heads"])
    default_head_names = ["Engagement"] if head_count == 1 else DEFAULT_MULTI_HEAD_NAMES[:head_count]
    head_names = [str(name) for name in (meta.get("head_names") or default_head_names)][:head_count]
    while len(head_names) < head_count:
        head_names.append(f"Head {len(head_names) + 1}")

    stem = MODEL_VARIANTS.get(variant_name, MODEL_VARIANTS[resolved_variant])["stem"]
    metrics = _load_json(MODEL_DIR / f"{stem}_metrics.json")

    return {
        "session": session,
        "onnx_path": onnx_path,
        "variant": variant_name,
        "metrics": metrics,
        "meta": meta,
        "device_label": device_label,
        "providers": session.get_providers(),
        "input_name": input_name,
        "output_name": output_name,
        "head_count": head_count,
        "head_names": head_names,
        "class_count": class_count,
        "seq_len": seq_len,
        "img_size": img_size,
    }


def preprocess(frames: list[np.ndarray], img_size: int) -> np.ndarray:
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(np.transpose(normalized, (2, 0, 1)))
    return np.expand_dims(np.stack(processed, axis=0).astype(np.float32), axis=0)


def predict_output(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    frames: list[np.ndarray],
    img_size: int,
    class_count: int,
    head_count: int,
) -> np.ndarray:
    inputs = preprocess(frames, img_size)
    raw_output = np.asarray(session.run([output_name], {input_name: inputs})[0])

    if raw_output.ndim == 3:
        logits = raw_output[0]
    elif raw_output.ndim == 2:
        if raw_output.shape[0] == 1 and raw_output.shape[1] in {class_count, head_count * class_count}:
            logits = raw_output.reshape(-1, class_count)
        else:
            logits = raw_output
    elif raw_output.ndim == 1:
        if raw_output.size % class_count != 0:
            raise ValueError(f"Unsupported logits size: {raw_output.shape}")
        logits = raw_output.reshape(-1, class_count)
    else:
        raise ValueError(f"Unsupported ONNX output shape: {raw_output.shape}")

    if logits.ndim != 2 or logits.shape[1] != class_count:
        raise ValueError(f"Unexpected logits shape after reshape: {logits.shape}")

    probabilities = np.stack([_softmax(row) for row in logits], axis=0)
    return probabilities.astype(np.float32)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    capture: cv2.VideoCapture | None = None
    for backend in backends:
        candidate = cv2.VideoCapture(index, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(index)
        if candidate and candidate.isOpened():
            capture = candidate
            break
        if candidate:
            candidate.release()

    if capture is None:
        capture = cv2.VideoCapture(index)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class EngagementApp:
    def __init__(self, root: tk.Tk, runtime: dict[str, Any]):
        self.root = root
        self.session = runtime["session"]
        self.onnx_path = runtime["onnx_path"]
        self.model_variant = runtime["variant"]
        self.metrics = runtime["metrics"]
        self.meta = runtime["meta"]
        self.device_label = runtime["device_label"]
        self.providers = runtime["providers"]
        self.input_name = runtime["input_name"]
        self.output_name = runtime["output_name"]
        self.head_count = runtime["head_count"]
        self.head_names = runtime["head_names"]
        self.class_count = runtime["class_count"]
        self.seq_len = runtime["seq_len"]
        self.img_size = runtime["img_size"]
        self.camera_index = int(runtime.get("camera_index", 0))
        self.feedback_manager = FeedbackManager(
            base_primary_threshold=PRIMARY_CONFIDENCE_THRESHOLD,
            base_spotlight_threshold=SPOTLIGHT_THRESHOLD,
        )

        self.engagement_head_index = next(
            (index for index, name in enumerate(self.head_names) if _normalize_head_name(name) == "engagement"),
            0,
        )
        split = max(1, self.class_count // 2)
        self.negative_class_indices = tuple(range(split))
        self.positive_class_indices = tuple(range(split, self.class_count))

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.capture: cv2.VideoCapture | None = None
        self.running = False
        self.closing = False
        self.capture_after_id: str | None = None
        self.ui_after_id: str | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_frame: np.ndarray | None = None
        self.last_prediction_time = 0.0
        self.last_output_time = 0.0
        self.inference_busy = False
        self.active_inference_id: tuple[int, int] | None = None
        self.session_token = 0
        self.checkin_dialog: tk.Toplevel | None = None
        self.frame_counter = 0
        self.capture_failures = 0

        self.output_lock = threading.Lock()
        self.pending_output: dict[str, Any] | None = None
        self.pending_error: str | None = None
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self.output_history: deque[np.ndarray] = deque(maxlen=OUTPUT_HISTORY_WINDOW)
        self.primary_transition = {"current": None, "candidate": None, "count": 0}
        self.spotlight_transition = {"current": None, "candidate": None, "count": 0}
        self.recent_engagement_samples: deque[tuple[float, float]] = deque()
        self.session_engagement_total = 0.0
        self.session_engagement_count = 0
        self.session_first_sample_time: float | None = None

        self.head_specs = self._build_head_specs()
        self.secondary_specs = [spec for spec in self.head_specs if spec["index"] != self.engagement_head_index]
        self.pomodoro_specs = self._build_pomodoro_specs()
        self.pomodoro_supported = len(self.pomodoro_specs) == 4
        self.signal_tiles: dict[str, dict[str, Any]] = {}
        self.signal_tile_order: list[str] = []
        self.summary_blocks: dict[str, dict[str, Any]] = {}
        self.state = "idle"
        self.scroll_canvas: tk.Canvas | None = None
        self.scroll_canvas_window: int | None = None
        self.content_frame: tk.Frame | None = None
        self.layout_after_id: str | None = None
        self.pomodoro_progress_canvas: tk.Canvas | None = None

        self.pomodoro_active = False
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_completed_blocks = 0
        self.pomodoro_current_block_index = 0
        self.pomodoro_remaining_seconds = float(POMODORO_TOTAL_SECONDS)
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_last_tick_monotonic: float | None = None
        self.pomodoro_session_start_epoch: float | None = None
        self.pomodoro_block_start_epoch: float | None = None
        self.pomodoro_pending_window_end_epoch: float | None = None
        self.pomodoro_last_frame_sample_at = 0.0
        self.pomodoro_phase = "idle" if self.pomodoro_supported else "unavailable"
        self.pomodoro_status_reason = ""
        self.pomodoro_block_outputs: list[tuple[float, np.ndarray]] = []
        self.pomodoro_block_frames: list[tuple[float, np.ndarray]] = []

        self.status_var = tk.StringVar(value=STATE_STYLES["idle"]["badge"])
        self.preview_badge_var = tk.StringVar(value="Idle")
        self.prediction_var = tk.StringVar(value="Camera Idle")
        self.confidence_var = tk.StringVar(value="Waiting for live input")
        self.summary_var = tk.StringVar(value=self._idle_summary())
        self.footer_var = tk.StringVar(value=self._idle_footer())
        self.meta_var = tk.StringVar(value=self._meta_text())
        self.spotlight_label_var = tk.StringVar(value="Secondary Spotlight")
        self.spotlight_value_var = tk.StringVar(value=self._spotlight_idle_title())
        self.spotlight_detail_var = tk.StringVar(value=self._spotlight_idle_detail())
        self.feedback_insight_var = tk.StringVar(value="")
        self.feedback_status_var = tk.StringVar(value="")
        self.engagement_summary_note_var = tk.StringVar(value=self._idle_engagement_summary_note())
        self.pomodoro_status_var = tk.StringVar(value="")
        self.pomodoro_time_var = tk.StringVar(value="")
        self.pomodoro_block_var = tk.StringVar(value="")
        self.pomodoro_next_var = tk.StringVar(value="")
        self.pomodoro_note_var = tk.StringVar(value="")

        self._set_initial_window()
        self._build_root_scaffold()
        self._build_ui()
        self._refresh_feedback_insight()
        self._reset_engagement_summary()
        self._reset_pomodoro_state()
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self._apply_state_palette("idle")
        self._sync_control_states()
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.after_idle(self._apply_responsive_layout)

    def _reset_temporal_smoothing(self) -> None:
        self.output_history.clear()
        self.primary_transition = {"current": None, "candidate": None, "count": 0}
        self.spotlight_transition = {"current": None, "candidate": None, "count": 0}

    def _idle_engagement_summary_note(self) -> str:
        return "Engagement averages update from smoothed live predictions once the camera is running."

    def _set_summary_block(self, key: str, value: str, detail: str) -> None:
        block = self.summary_blocks.get(key)
        if block is None:
            return
        block["value"].configure(text=value)
        block["detail"].configure(text=detail)

    def _reset_engagement_summary(self) -> None:
        self.recent_engagement_samples.clear()
        self.session_engagement_total = 0.0
        self.session_engagement_count = 0
        self.session_first_sample_time = None
        self._set_summary_block("last_4", "--", "Waiting for live predictions")
        self._set_summary_block("last_8", "--", "Waiting for live predictions")
        self._set_summary_block("session", "--", "Start the camera to build a session baseline")
        self.engagement_summary_note_var.set(self._idle_engagement_summary_note())

    def _format_summary_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, remainder = divmod(total_seconds, 60)
        if minutes <= 0:
            return f"{remainder}s"
        return f"{minutes}m {remainder:02d}s"

    def _engagement_summary_label(self, score: float) -> str:
        if score >= 0.7:
            return "Strong engagement"
        if score >= 0.55:
            return "Mostly engaged"
        if score >= 0.45:
            return "Mixed engagement"
        return "Low engagement"

    def _window_average(self, now: float, seconds: int) -> tuple[float | None, float]:
        cutoff = now - seconds
        values = [score for timestamp, score in self.recent_engagement_samples if timestamp >= cutoff]
        if not values:
            return None, 0.0
        earliest = next(timestamp for timestamp, _ in self.recent_engagement_samples if timestamp >= cutoff)
        covered_seconds = max(INFERENCE_INTERVAL_SEC, now - earliest)
        return float(sum(values) / len(values)), min(float(seconds), covered_seconds)

    def _refresh_engagement_summary(self, now: float | None = None) -> None:
        now = float(now if now is not None else time.time())
        four_min_avg, four_min_coverage = self._window_average(now, SUMMARY_WINDOWS_MINUTES[0] * 60)
        eight_min_avg, eight_min_coverage = self._window_average(now, SUMMARY_WINDOWS_MINUTES[1] * 60)

        if four_min_avg is None:
            self._set_summary_block("last_4", "--", "Waiting for live predictions")
        else:
            detail = (
                f"Full {SUMMARY_WINDOWS_MINUTES[0]}m window"
                if four_min_coverage >= SUMMARY_WINDOWS_MINUTES[0] * 60 - 5
                else f"Using {self._format_summary_duration(four_min_coverage)} of data"
            )
            self._set_summary_block("last_4", f"{four_min_avg * 100:.0f}%", detail)

        if eight_min_avg is None:
            self._set_summary_block("last_8", "--", "Waiting for live predictions")
        else:
            detail = (
                f"Full {SUMMARY_WINDOWS_MINUTES[1]}m window"
                if eight_min_coverage >= SUMMARY_WINDOWS_MINUTES[1] * 60 - 5
                else f"Using {self._format_summary_duration(eight_min_coverage)} of data"
            )
            self._set_summary_block("last_8", f"{eight_min_avg * 100:.0f}%", detail)

        if self.session_engagement_count <= 0:
            self._set_summary_block("session", "--", "Start the camera to build a session baseline")
            self.engagement_summary_note_var.set(self._idle_engagement_summary_note())
            return

        session_avg = self.session_engagement_total / self.session_engagement_count
        live_seconds = 0.0 if self.session_first_sample_time is None else max(INFERENCE_INTERVAL_SEC, now - self.session_first_sample_time)
        self._set_summary_block("session", f"{session_avg * 100:.0f}%", f"{self._engagement_summary_label(session_avg)} trend")
        trend_note = self._engagement_summary_label(four_min_avg if four_min_avg is not None else session_avg)
        self.engagement_summary_note_var.set(
            f"Summary: {trend_note.lower()} over {self._format_summary_duration(live_seconds)} of live predictions."
        )

    def _record_engagement_sample(self, output: np.ndarray, timestamp: float | None = None) -> None:
        now = float(timestamp if timestamp is not None else time.time())
        engaged, _, _ = self._engagement_scores(output)
        if self.session_first_sample_time is None:
            self.session_first_sample_time = now
        self.recent_engagement_samples.append((now, engaged))
        self.session_engagement_total += engaged
        self.session_engagement_count += 1

        cutoff = now - SUMMARY_MAX_WINDOW_SEC
        while self.recent_engagement_samples and self.recent_engagement_samples[0][0] < cutoff:
            self.recent_engagement_samples.popleft()

        self._refresh_engagement_summary(now)

    def _neutral_output(self) -> np.ndarray:
        return np.full((self.head_count, self.class_count), 1.0 / max(1, self.class_count), dtype=np.float32)

    def _idle_summary(self) -> str:
        if self.model_variant == "multiaffect":
            return "Start the camera to monitor engagement with contextual boredom, confusion, and frustration."
        return "Start the camera to monitor engagement from the distilled student model."

    def _idle_footer(self) -> str:
        return f"Ready. Live decisions begin after {self.seq_len} buffered frames."

    def _spotlight_idle_title(self) -> str:
        if self.secondary_specs:
            return "Secondary signals stable"
        return "Engagement-only model active"

    def _spotlight_idle_detail(self) -> str:
        if self.secondary_specs:
            return "No non-engagement spotlight is active until live inference begins."
        return "This ONNX export exposes only the engagement head."

    def _metric_summary(self) -> str:
        if self.model_variant == "multiaffect":
            score = self.metrics.get("best_mean_accuracy")
            if score is not None:
                return f"Mean val {float(score) * 100:.1f}%"
        score = self.metrics.get("val_accuracy")
        if score is not None:
            return f"Val {float(score) * 100:.1f}%"
        return "Metrics loaded"

    def _meta_text(self) -> str:
        return (
            f"{self.model_variant.title()} model | {self.head_count} head{'s' if self.head_count != 1 else ''} | "
            f"Seq {self.seq_len} | {self.img_size}px | {self._metric_summary()}"
        )

    def _build_head_specs(self) -> list[dict[str, Any]]:
        specs = []
        for index, name in enumerate(self.head_names):
            key = _normalize_head_name(name)
            palette = HEAD_PALETTES.get(key, HEAD_PALETTES["default"])
            specs.append(
                {
                    "index": index,
                    "name": str(name),
                    "label": _display_head_name(name),
                    "accent": palette["accent"],
                    "surface": palette["surface"],
                }
            )
        return specs

    def _build_pomodoro_specs(self) -> list[dict[str, Any]]:
        ordered_keys = ("engagement", "boredom", "confusion", "frustration")
        ordered_specs: list[dict[str, Any]] = []
        for key in ordered_keys:
            spec = next((item for item in self.head_specs if _normalize_head_name(item["name"]) == key), None)
            if spec is None:
                return []
            ordered_specs.append(spec)
        return ordered_specs

    def _idle_pomodoro_note(self) -> str:
        if not self.pomodoro_supported:
            return "Pomodoro check-ins need the multi-affect model with engagement, boredom, confusion, and frustration heads."
        return "Start Pomodoro to begin a 24-minute focus block with self-checks every 8 minutes."

    def _format_clock(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, remainder = divmod(total_seconds, 60)
        return f"{minutes:02d}:{remainder:02d}"

    def _reset_pomodoro_block_capture(self, start_epoch: float | None = None) -> None:
        self.pomodoro_block_outputs.clear()
        self.pomodoro_block_frames.clear()
        self.pomodoro_block_start_epoch = start_epoch
        self.pomodoro_pending_window_end_epoch = None
        self.pomodoro_last_frame_sample_at = 0.0

    def _reset_pomodoro_state(self, *, phase: str | None = None, reason: str = "") -> None:
        self.pomodoro_active = False
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_completed_blocks = 0
        self.pomodoro_current_block_index = 0
        self.pomodoro_remaining_seconds = float(POMODORO_TOTAL_SECONDS)
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_last_tick_monotonic = None
        self.pomodoro_session_start_epoch = None
        self.pomodoro_status_reason = reason
        self._reset_pomodoro_block_capture()
        self.pomodoro_phase = phase or ("idle" if self.pomodoro_supported else "unavailable")
        self._refresh_pomodoro_ui()

    def _draw_pomodoro_progress(self, completed_blocks: int, current_progress: float, accent: str) -> None:
        if self.pomodoro_progress_canvas is None:
            return
        canvas = self.pomodoro_progress_canvas
        canvas.delete("all")
        width = max(24, canvas.winfo_width() or 240)
        height = max(12, canvas.winfo_height() or 22)
        gap = 10
        total_gap = gap * 2
        block_width = max(24.0, (width - total_gap) / 3.0)
        track = COLORS["panel_soft"]
        current_progress = max(0.0, min(1.0, current_progress))

        for index in range(3):
            x0 = index * (block_width + gap)
            x1 = x0 + block_width
            canvas.create_rectangle(x0, 0, x1, height, fill=track, outline=track)

            fill_fraction = 0.0
            if index < completed_blocks:
                fill_fraction = 1.0
            elif index == completed_blocks:
                fill_fraction = current_progress
            if fill_fraction > 0:
                canvas.create_rectangle(x0, 0, x0 + (block_width * fill_fraction), height, fill=accent, outline=accent)

            canvas.create_text(
                (x0 + x1) / 2,
                height / 2,
                text=f"{index + 1}",
                fill=COLORS["text"],
                font=("Segoe UI Semibold", 9),
            )

    def _refresh_pomodoro_ui(self) -> None:
        if not self.pomodoro_supported:
            status = "Unavailable"
            time_text = self._format_clock(POMODORO_TOTAL_SECONDS)
            block_text = "Needs multi-affect model"
            next_text = "Engagement-only runtime cannot open 4-head self-checks."
            note_text = self.pomodoro_status_reason or self._idle_pomodoro_note()
            accent = COLORS["text_soft"]
            surface = COLORS["panel_soft"]
            completed_blocks = 0
            current_progress = 0.0
        elif self.pomodoro_phase == "running":
            block_remaining = max(0.0, POMODORO_BLOCK_SECONDS - self.pomodoro_block_elapsed_seconds)
            status = "Focus Live"
            time_text = self._format_clock(self.pomodoro_remaining_seconds)
            block_text = f"Block {self.pomodoro_current_block_index + 1}/3"
            next_text = f"Next check-in in {self._format_clock(block_remaining)}"
            note_text = "Live monitoring stays on while the timer runs. The timer pauses during each self-check."
            accent = COLORS["blue"]
            surface = COLORS["blue_soft"]
            completed_blocks = self.pomodoro_completed_blocks
            current_progress = self.pomodoro_block_elapsed_seconds / POMODORO_BLOCK_SECONDS
        elif self.pomodoro_phase == "paused":
            status = "Check-In"
            time_text = self._format_clock(self.pomodoro_remaining_seconds)
            block_text = f"Block {self.pomodoro_current_block_index + 1}/3 complete"
            next_text = f"Review the last {POMODORO_BLOCK_MINUTES} minutes to continue."
            note_text = "Answer how engaged, bored, confused, and frustrated you felt. The next block starts after submit or skip."
            accent = COLORS["amber"]
            surface = COLORS["amber_soft"]
            completed_blocks = self.pomodoro_completed_blocks
            current_progress = 1.0
        elif self.pomodoro_phase == "complete":
            status = "Complete"
            time_text = self._format_clock(0)
            block_text = "3 blocks finished"
            next_text = "The 24-minute Pomodoro is complete."
            note_text = self.pomodoro_status_reason or "Three 8-minute self-check windows were captured for learning."
            accent = COLORS["green"]
            surface = COLORS["green_soft"]
            completed_blocks = 3
            current_progress = 0.0
        elif self.pomodoro_phase == "stopped":
            status = "Stopped"
            time_text = self._format_clock(self.pomodoro_remaining_seconds)
            block_text = "Pomodoro ended early"
            next_text = "Start again for a fresh 24-minute block."
            note_text = self.pomodoro_status_reason or "Pomodoro stopped before the third self-check."
            accent = COLORS["red"]
            surface = COLORS["red_soft"]
            completed_blocks = 0
            current_progress = 0.0
        else:
            status = "Idle"
            time_text = self._format_clock(POMODORO_TOTAL_SECONDS)
            block_text = "Block 1/3"
            next_text = f"Next check-in in {self._format_clock(POMODORO_BLOCK_SECONDS)}"
            note_text = self.pomodoro_status_reason or self._idle_pomodoro_note()
            accent = COLORS["text_soft"]
            surface = COLORS["panel_soft"]
            completed_blocks = 0
            current_progress = 0.0

        self.pomodoro_status_var.set(status)
        self.pomodoro_time_var.set(time_text)
        self.pomodoro_block_var.set(block_text)
        self.pomodoro_next_var.set(next_text)
        self.pomodoro_note_var.set(note_text)

        if hasattr(self, "pomodoro_chip"):
            self.pomodoro_chip.configure(text=status, bg=surface, fg=accent)
        if hasattr(self, "pomodoro_card"):
            self.pomodoro_card.configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
        self._draw_pomodoro_progress(completed_blocks, current_progress, accent)

    def _sync_control_states(self) -> None:
        if hasattr(self, "start_button"):
            if self.running:
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
            else:
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled", bg=COLORS["panel_alt"], fg=COLORS["text"])

        if not hasattr(self, "start_pomodoro_button"):
            return

        pomodoro_open = self.pomodoro_active or self.checkin_dialog is not None
        if self.pomodoro_supported and not pomodoro_open:
            self.start_pomodoro_button.configure(state="normal", bg=COLORS["blue_soft"], fg=COLORS["blue"])
        else:
            self.start_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

        if pomodoro_open:
            self.stop_pomodoro_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
        else:
            self.stop_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

    def _record_pomodoro_frame_sample(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        if not self.pomodoro_active or self.pomodoro_paused or not self.pomodoro_supported:
            return
        now = float(timestamp if timestamp is not None else time.time())
        if self.pomodoro_last_frame_sample_at and (now - self.pomodoro_last_frame_sample_at) < POMODORO_FRAME_SAMPLE_INTERVAL_SEC:
            return
        resized = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        self.pomodoro_block_frames.append((now, resized.copy()))
        self.pomodoro_last_frame_sample_at = now

    def _record_pomodoro_output_sample(self, output: np.ndarray, timestamp: float | None = None) -> None:
        if not self.pomodoro_active or self.pomodoro_paused or not self.pomodoro_supported:
            return
        now = float(timestamp if timestamp is not None else time.time())
        self.pomodoro_block_outputs.append((now, np.asarray(output, dtype=np.float32).copy()))

    def _resample_pomodoro_frames(self) -> list[np.ndarray]:
        frames = [frame.copy() for _, frame in self.pomodoro_block_frames]
        if not frames and self.last_frame is not None:
            resized = cv2.resize(self.last_frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frames = [resized]
        if not frames:
            return []
        if len(frames) >= self.seq_len:
            indices = np.linspace(0, len(frames) - 1, self.seq_len)
            return [frames[int(round(index))].copy() for index in indices]
        padded = [frame.copy() for frame in frames]
        while len(padded) < self.seq_len:
            padded.append(padded[-1].copy())
        return padded

    def _derive_internal_rating(self, output: np.ndarray, corrected_labels: list[int | None], known_mask: list[bool]) -> int:
        predicted_labels = [int(np.argmax(head)) for head in np.asarray(output, dtype=np.float32)]
        head_distance = max(1, self.class_count - 1)
        agreement_scores: list[float] = []
        for index, predicted in enumerate(predicted_labels):
            if index >= len(known_mask) or not known_mask[index]:
                continue
            corrected = corrected_labels[index]
            if corrected is None:
                continue
            distance = abs(predicted - int(corrected))
            agreement_scores.append(max(0.0, 1.0 - (distance / head_distance)))
        if not agreement_scores:
            return 3
        agreement = float(sum(agreement_scores) / len(agreement_scores))
        return max(1, min(5, int(round(1 + (4 * agreement)))))

    def _build_feedback_snapshot(
        self,
        *,
        frames: list[np.ndarray] | None = None,
        output: np.ndarray | None = None,
        summary_override: str | None = None,
        feedback_source: str = "manual_review",
        window_start_epoch: float | None = None,
        window_end_epoch: float | None = None,
        derived_rating: int | None = None,
    ) -> dict[str, Any] | None:
        source_frames = list(self.frame_buffer) if frames is None else [frame.copy() for frame in frames]
        if not source_frames:
            return None
        if frames is None and (len(self.frame_buffer) < self.seq_len or len(self.output_history) == 0):
            return None

        resolved_output = self.display_output.copy().astype(np.float32) if output is None else np.asarray(output, dtype=np.float32).copy()
        engaged, not_engaged, _ = self._engagement_scores(resolved_output)
        primary_confidence = max(engaged, not_engaged)
        spotlight_key = self.spotlight_transition.get("current")
        if not isinstance(spotlight_key, str) or not spotlight_key.startswith("head:"):
            spotlight_key = None
        active_signal = self._signal_for_key(resolved_output, spotlight_key) if spotlight_key else None
        spotlight_confidence = float(active_signal["elevated"]) if active_signal is not None else 0.0
        primary_threshold, spotlight_threshold = self._effective_thresholds()
        return self.feedback_manager.build_review_snapshot(
            frames=source_frames,
            output=resolved_output,
            model_variant=self.model_variant,
            head_names=self.head_names,
            class_count=self.class_count,
            seq_len=self.seq_len,
            img_size=self.img_size,
            state=self.state,
            headline=self.prediction_var.get(),
            confidence_text=self.confidence_var.get(),
            summary=summary_override or self.summary_var.get(),
            primary_confidence=primary_confidence,
            spotlight_key=spotlight_key,
            spotlight_confidence=spotlight_confidence,
            primary_threshold=primary_threshold,
            spotlight_threshold=spotlight_threshold,
            feedback_source=feedback_source,
            window_start_epoch=window_start_epoch,
            window_end_epoch=window_end_epoch,
            derived_rating=derived_rating,
        )

    def _advance_pomodoro_after_checkin(self, status_reason: str) -> None:
        self.pomodoro_prompt_pending = False
        self.pomodoro_paused = False
        self.pomodoro_completed_blocks += 1
        if self.pomodoro_completed_blocks >= 3:
            self.pomodoro_active = False
            self.pomodoro_phase = "complete"
            self.pomodoro_status_reason = status_reason
            self.pomodoro_remaining_seconds = 0.0
            self.pomodoro_block_elapsed_seconds = float(POMODORO_BLOCK_SECONDS)
            self.pomodoro_current_block_index = 2
            self._reset_pomodoro_block_capture()
            self._refresh_pomodoro_ui()
            self._sync_control_states()
            return

        self.pomodoro_current_block_index = self.pomodoro_completed_blocks
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_last_tick_monotonic = time.monotonic()
        self.pomodoro_phase = "running"
        self.pomodoro_status_reason = status_reason
        self._reset_pomodoro_block_capture(time.time())
        self._refresh_pomodoro_ui()
        self._sync_control_states()

    def _close_checkin_dialog(self) -> None:
        if self.checkin_dialog is None:
            return
        try:
            self.checkin_dialog.grab_release()
        except tk.TclError:
            pass
        try:
            self.checkin_dialog.destroy()
        except tk.TclError:
            pass
        self.checkin_dialog = None
        self._sync_control_states()

    def _skip_pomodoro_checkin(self) -> None:
        block_number = self.pomodoro_current_block_index + 1
        self._close_checkin_dialog()
        self.feedback_status_var.set(f"Latest: Block {block_number} self-check skipped.")
        status_reason = (
            "Pomodoro finished. The last self-check was skipped."
            if block_number >= 3
            else f"Block {block_number} skipped. Continue when ready."
        )
        self._advance_pomodoro_after_checkin(status_reason)

    def _submit_pomodoro_checkin(self, selections: dict[int, tk.StringVar]) -> None:
        block_number = self.pomodoro_current_block_index + 1
        if not self.pomodoro_block_outputs:
            self._close_checkin_dialog()
            self.feedback_status_var.set("Latest: Not enough live data was captured for that self-check window.")
            self._advance_pomodoro_after_checkin(f"Block {block_number} had insufficient live data.")
            return

        corrected_labels: list[int | None] = [None] * self.head_count
        known_mask: list[bool] = [False] * self.head_count
        for spec in self.pomodoro_specs:
            choice = selections[spec["index"]].get()
            if choice == "Not sure":
                corrected_labels[spec["index"]] = None
                known_mask[spec["index"]] = False
            else:
                corrected_labels[spec["index"]] = AFFECT_LEVELS.index(choice)
                known_mask[spec["index"]] = True

        aggregated_output = np.stack([sample for _, sample in self.pomodoro_block_outputs], axis=0).mean(axis=0).astype(np.float32)
        derived_rating = self._derive_internal_rating(aggregated_output, corrected_labels, known_mask)
        summary = f"Pomodoro self-check for block {block_number} covering the last {POMODORO_BLOCK_MINUTES} minutes."
        snapshot = self._build_feedback_snapshot(
            frames=self._resample_pomodoro_frames(),
            output=aggregated_output,
            summary_override=summary,
            feedback_source=POMODORO_FEEDBACK_SOURCE,
            window_start_epoch=self.pomodoro_block_start_epoch,
            window_end_epoch=self.pomodoro_pending_window_end_epoch or time.time(),
            derived_rating=derived_rating,
        )
        if snapshot is None:
            self._close_checkin_dialog()
            self.feedback_status_var.set("Latest: The self-check could not be saved because the Pomodoro window was empty.")
            self._advance_pomodoro_after_checkin(f"Block {block_number} could not be saved.")
            return

        self.feedback_manager.submit_feedback(
            snapshot,
            rating=derived_rating,
            corrected_labels=corrected_labels,
            known_mask=known_mask,
        )
        self._refresh_feedback_insight()
        if self.running and len(self.output_history) > 0:
            self._update_primary_view(self.display_output)
        self._close_checkin_dialog()
        status_reason = (
            "Pomodoro complete. Three 8-minute windows are ready for learning."
            if block_number >= 3
            else f"Block {block_number} saved. Timer resumed for the next focus window."
        )
        self._advance_pomodoro_after_checkin(status_reason)

    def _open_pomodoro_checkin(self) -> None:
        if self.checkin_dialog is not None:
            self.checkin_dialog.lift()
            self.checkin_dialog.focus_force()
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Pomodoro Check-In")
        dialog.configure(bg=COLORS["panel"])
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        self.checkin_dialog = dialog
        self._sync_control_states()

        container = tk.Frame(dialog, bg=COLORS["panel"])
        container.pack(fill="both", expand=True, padx=22, pady=22)

        tk.Label(container, text="8-Minute Self-Check", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18)).pack(anchor="w")
        tk.Label(
            container,
            text=f"How engaged, bored, confused, and frustrated did you feel in the last {POMODORO_BLOCK_MINUTES} minutes?",
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            justify="left",
            wraplength=760,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 16))

        choice_values = [*AFFECT_LEVELS, "Not sure"]
        selections: dict[int, tk.StringVar] = {}
        for spec in self.pomodoro_specs:
            row = self._create_card(container, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
            row.pack(fill="x", pady=6)
            tk.Label(row, text=spec["label"], bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Segoe UI Semibold", 10)).pack(anchor="w", padx=16, pady=(14, 8))
            button_row = tk.Frame(row, bg=COLORS["panel_alt"])
            button_row.pack(fill="x", padx=12, pady=(0, 14))
            choice_var = tk.StringVar(value="Not sure")
            selections[spec["index"]] = choice_var
            for column, choice in enumerate(choice_values):
                button_row.grid_columnconfigure(column, weight=1)
                tk.Radiobutton(
                    button_row,
                    text=choice,
                    variable=choice_var,
                    value=choice,
                    indicatoron=False,
                    bg=COLORS["panel_soft"],
                    fg=COLORS["text"],
                    activebackground=mix_color(spec["surface"], "#ffffff", 0.06),
                    activeforeground=COLORS["text"],
                    selectcolor=spec["surface"],
                    relief="flat",
                    bd=0,
                    padx=10,
                    pady=10,
                    font=("Segoe UI Semibold", 9),
                    highlightthickness=0,
                    cursor="hand2",
                ).grid(row=0, column=column, sticky="ew", padx=4)

        action_row = tk.Frame(container, bg=COLORS["panel"])
        action_row.pack(fill="x", pady=(18, 0))
        tk.Button(
            action_row,
            text="Submit",
            command=lambda: self._submit_pomodoro_checkin(selections),
            bg=COLORS["green"],
            fg=COLORS["text"],
            activebackground=mix_color(COLORS["green"], "#ffffff", 0.12),
            activeforeground=COLORS["text"],
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        ).pack(side="left")
        tk.Button(
            action_row,
            text="Skip",
            command=self._skip_pomodoro_checkin,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            activebackground=mix_color(COLORS["panel_alt"], "#ffffff", 0.08),
            activeforeground=COLORS["text"],
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        ).pack(side="left", padx=(10, 0))
        tk.Button(
            action_row,
            text="Stop Pomodoro",
            command=self.stop_pomodoro,
            bg=COLORS["red_soft"],
            fg=COLORS["red"],
            activebackground=mix_color(COLORS["red_soft"], "#ffffff", 0.08),
            activeforeground=COLORS["red"],
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        ).pack(side="right")

        dialog.protocol("WM_DELETE_WINDOW", self._skip_pomodoro_checkin)

    def _update_pomodoro_timer(self) -> None:
        if not self.pomodoro_active:
            return
        if self.pomodoro_paused:
            self._refresh_pomodoro_ui()
            return

        now = time.monotonic()
        if self.pomodoro_last_tick_monotonic is None:
            self.pomodoro_last_tick_monotonic = now
        delta = max(0.0, now - self.pomodoro_last_tick_monotonic)
        self.pomodoro_last_tick_monotonic = now
        if delta <= 0.0:
            self._refresh_pomodoro_ui()
            return

        self.pomodoro_remaining_seconds = max(0.0, self.pomodoro_remaining_seconds - delta)
        self.pomodoro_block_elapsed_seconds = min(float(POMODORO_BLOCK_SECONDS), self.pomodoro_block_elapsed_seconds + delta)
        self.pomodoro_phase = "running"
        if self.pomodoro_block_elapsed_seconds >= POMODORO_BLOCK_SECONDS and not self.pomodoro_prompt_pending:
            self.pomodoro_paused = True
            self.pomodoro_prompt_pending = True
            self.pomodoro_phase = "paused"
            self.pomodoro_pending_window_end_epoch = time.time()
            self._refresh_pomodoro_ui()
            self._open_pomodoro_checkin()
            return

        self._refresh_pomodoro_ui()

    def start_pomodoro(self) -> None:
        if not self.pomodoro_supported:
            self.feedback_status_var.set("Latest: Pomodoro check-ins need the multi-affect model.")
            self._refresh_pomodoro_ui()
            return
        if self.pomodoro_active or self.checkin_dialog is not None:
            return
        if not self.running:
            self.start()
            if not self.running:
                return

        self.pomodoro_active = True
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_phase = "running"
        self.pomodoro_status_reason = ""
        self.pomodoro_completed_blocks = 0
        self.pomodoro_current_block_index = 0
        self.pomodoro_remaining_seconds = float(POMODORO_TOTAL_SECONDS)
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_session_start_epoch = time.time()
        self.pomodoro_last_tick_monotonic = time.monotonic()
        self._reset_pomodoro_block_capture(self.pomodoro_session_start_epoch)
        self.feedback_status_var.set("Latest: Pomodoro started. The first self-check opens in 8 minutes.")
        self._refresh_pomodoro_ui()
        self._sync_control_states()

    def stop_pomodoro(self) -> None:
        if not (self.pomodoro_active or self.checkin_dialog is not None or self.pomodoro_prompt_pending):
            return
        self._close_checkin_dialog()
        self._reset_pomodoro_state(phase="stopped", reason="Pomodoro stopped. Start again for a fresh 24-minute focus block.")
        self.feedback_status_var.set("Latest: Pomodoro stopped before completion.")
        if self.running:
            self.stop()
        else:
            self._sync_control_states()

    def _set_initial_window(self) -> None:
        self.root.title("Live Engagement Monitor")
        self.root.configure(bg=COLORS["bg"])
        self.root.option_add("*Font", "{Segoe UI} 10")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        usable_w = max(WINDOW_MIN_WIDTH, screen_w - (WINDOW_EDGE_MARGIN * 2))
        usable_h = max(WINDOW_MIN_HEIGHT, screen_h - (WINDOW_VERTICAL_MARGIN * 2))
        self.root.minsize(min(WINDOW_MIN_WIDTH, usable_w), min(WINDOW_MIN_HEIGHT, usable_h))
        width = min(usable_w, max(WINDOW_MIN_WIDTH, int(usable_w / 2)))
        height = usable_h
        x_pos = max(0, int((screen_w - usable_w) / 2))
        y_pos = max(0, int((screen_h - usable_h) / 2))
        self.root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def _build_root_scaffold(self) -> None:
        self.scroll_canvas = tk.Canvas(
            self.root,
            bg=COLORS["bg"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.content_frame = tk.Frame(self.scroll_canvas, bg=COLORS["bg"])
        self.scroll_canvas_window = self.scroll_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self.scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        self.content_frame.bind("<Configure>", self._on_content_frame_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _on_scroll_canvas_configure(self, event) -> None:
        if self.scroll_canvas is None or self.scroll_canvas_window is None:
            return
        self.scroll_canvas.itemconfigure(self.scroll_canvas_window, width=event.width)
        self._schedule_responsive_layout()

    def _on_content_frame_configure(self, _event) -> None:
        if self.scroll_canvas is None:
            return
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_mousewheel(self, event) -> None:
        if self.scroll_canvas is None:
            return
        if self.checkin_dialog is not None:
            widget = getattr(event, "widget", None)
            if widget is not None:
                try:
                    if widget.winfo_toplevel() is self.checkin_dialog:
                        return
                except tk.TclError:
                    return

        scroll_region = self.scroll_canvas.bbox("all")
        if not scroll_region:
            return
        if (scroll_region[3] - scroll_region[1]) <= self.scroll_canvas.winfo_height():
            return

        steps = 0
        delta = getattr(event, "delta", 0)
        if delta:
            steps = -int(delta / 120)
            if steps == 0:
                steps = -1 if delta > 0 else 1
        else:
            event_num = getattr(event, "num", None)
            if event_num == 4:
                steps = -1
            elif event_num == 5:
                steps = 1

        if steps == 0:
            return
        self.scroll_canvas.yview_scroll(steps, "units")

    def _on_root_configure(self, event) -> None:
        if event.widget is self.root:
            self._schedule_responsive_layout()

    def _schedule_responsive_layout(self) -> None:
        if self.layout_after_id is not None:
            self.root.after_cancel(self.layout_after_id)
        self.layout_after_id = self.root.after(60, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self.layout_after_id = None
        if self.content_frame is None:
            return

        content_width = max(1, self.content_frame.winfo_width() or self.root.winfo_width())
        top_bar_compact = content_width < TOP_BAR_STACK_BREAKPOINT
        main_stacked = content_width < MAIN_STACK_BREAKPOINT
        tile_columns = max(1, min(len(self.signal_tile_order), content_width // TILE_MIN_WIDTH))

        self._layout_top_bar(top_bar_compact)
        self._layout_main_area(main_stacked)
        self._layout_bottom_header(main_stacked)
        self._layout_signal_tiles(tile_columns)
        self._update_wraplengths(main_stacked)

    def _layout_top_bar(self, compact: bool) -> None:
        self.title_frame.grid_forget()
        self.chip_frame.grid_forget()
        self.button_frame.grid_forget()

        if compact:
            self.top_bar.grid_columnconfigure(0, weight=1)
            self.top_bar.grid_columnconfigure(1, weight=0)
            self.top_bar.grid_columnconfigure(2, weight=0)
            self.title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
            self.chip_frame.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 8))
            self.button_frame.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 18))
            return

        self.top_bar.grid_columnconfigure(0, weight=0)
        self.top_bar.grid_columnconfigure(1, weight=1)
        self.top_bar.grid_columnconfigure(2, weight=0)
        self.title_frame.grid(row=0, column=0, sticky="w", padx=20, pady=18)
        self.chip_frame.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=18)
        self.button_frame.grid(row=0, column=2, sticky="e", padx=20, pady=18)

    def _layout_main_area(self, stacked: bool) -> None:
        self.preview_card.grid_forget()
        self.decision_card.grid_forget()

        if stacked:
            self.main_area.grid_columnconfigure(0, weight=1, minsize=0)
            self.main_area.grid_columnconfigure(1, weight=0, minsize=0)
            self.main_area.grid_rowconfigure(0, weight=1)
            self.main_area.grid_rowconfigure(1, weight=0)
            self.preview_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
            self.decision_card.grid(row=1, column=0, sticky="ew")
            return

        self.main_area.grid_columnconfigure(0, weight=3, minsize=0)
        self.main_area.grid_columnconfigure(1, weight=2, minsize=0)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(1, weight=0)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.decision_card.grid(row=0, column=1, sticky="nsew")

    def _layout_bottom_header(self, compact: bool) -> None:
        self.bottom_title_label.grid_forget()
        self.bottom_meta_label.grid_forget()

        if compact:
            self.bottom_header.grid_columnconfigure(0, weight=1)
            self.bottom_title_label.grid(row=0, column=0, sticky="w")
            self.bottom_meta_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
            return

        self.bottom_header.grid_columnconfigure(0, weight=1)
        self.bottom_title_label.grid(row=0, column=0, sticky="w")
        self.bottom_meta_label.grid(row=0, column=1, sticky="e")

    def _layout_signal_tiles(self, columns: int) -> None:
        if not self.signal_tile_order:
            return

        columns = max(1, min(columns, len(self.signal_tile_order)))
        for index in range(len(self.signal_tile_order)):
            self.tiles_frame.grid_columnconfigure(index, weight=0, uniform="")
            self.tiles_frame.grid_rowconfigure(index, weight=0)

        for column in range(columns):
            self.tiles_frame.grid_columnconfigure(column, weight=1, uniform="signal_tile")

        for index, key in enumerate(self.signal_tile_order):
            tile = self.signal_tiles[key]["frame"]
            row = index // columns
            column = index % columns
            tile.grid_forget()
            tile.grid(row=row, column=column, sticky="ew", padx=6, pady=(0 if row == 0 else 6, 0))

    def _update_wraplengths(self, main_stacked: bool) -> None:
        decision_width = max(360, self.decision_card.winfo_width())
        preview_width = max(360, self.preview_card.winfo_width())
        bottom_width = max(360, self.bottom_band.winfo_width())

        self.summary_label.configure(wraplength=max(240, decision_width - 72))
        self.spotlight_detail_label.configure(wraplength=max(240, decision_width - 72))
        self.pomodoro_note_label.configure(wraplength=max(240, decision_width - 72))
        self.preview_footer.configure(wraplength=max(260, preview_width - 40))
        self.bottom_meta_label.configure(wraplength=max(240, bottom_width - 40))
        self.engagement_summary_note_label.configure(wraplength=max(260, bottom_width - 40))
        self.feedback_info_label.configure(wraplength=max(260, bottom_width - 40))
        self.feedback_status_label.configure(wraplength=max(260, bottom_width - 40))
        self.preview_stage.configure(height=PREVIEW_STAGE_COMPACT_HEIGHT if main_stacked else PREVIEW_STAGE_HEIGHT)
        self._on_preview_configure(None)

    def _create_card(self, parent: tk.Misc, bg: str | None = None, border: str | None = None) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=bg or COLORS["panel"],
            highlightthickness=1,
            highlightbackground=border or COLORS["border"],
            bd=0,
        )

    def _create_chip(self, parent: tk.Misc, text: str, bg: str, fg: str | None = None) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg or COLORS["text"],
            padx=12,
            pady=6,
            font=("Segoe UI Semibold", 9),
        )

    def _create_button(self, parent: tk.Misc, text: str, command, bg: str, fg: str = COLORS["text"]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=mix_color(bg, "#ffffff", 0.12),
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )

    def _effective_thresholds(self) -> tuple[float, float]:
        thresholds = self.feedback_manager.effective_thresholds()
        return float(thresholds["primary"]), float(thresholds["spotlight"])

    def _refresh_feedback_insight(self) -> None:
        insight = self.feedback_manager.current_session_insight()
        self.feedback_insight_var.set(
            "Check-ins: "
            f"{insight['review_count']} logged | "
            f"{insight['trusted_count']} trusted | "
            f"Primary {insight['primary_threshold'] * 100:.0f}% | "
            f"Spotlight {insight['spotlight_threshold'] * 100:.0f}%"
        )
        self.feedback_status_var.set(f"Latest: {insight['last_feedback_status']}")

    def _create_stat_block(self, parent: tk.Misc, title: str) -> dict[str, Any]:
        frame = self._create_card(parent, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        frame.configure(height=STAT_BLOCK_HEIGHT)
        frame.pack_propagate(False)
        title_label = tk.Label(frame, text=title, bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 9))
        value_label = tk.Label(frame, text="50%", bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18))
        detail_label = tk.Label(
            frame,
            text="Band share",
            bg=COLORS["panel_alt"],
            fg=COLORS["text_muted"],
            justify="left",
            anchor="w",
            font=("Segoe UI", 9),
        )
        title_label.pack(anchor="w", padx=14, pady=(10, 2))
        value_label.pack(anchor="w", padx=14, pady=(0, 1))
        detail_label.pack(anchor="w", padx=14, pady=(0, 9))
        return {"frame": frame, "value": value_label, "detail": detail_label}

    def _build_ui(self) -> None:
        assert self.content_frame is not None
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        self.top_bar = self._create_card(self.content_frame, bg=COLORS["panel"], border=COLORS["border"])
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        self.top_bar.grid_columnconfigure(1, weight=1)

        self.title_frame = tk.Frame(self.top_bar, bg=COLORS["panel"])
        self.title_frame.grid(row=0, column=0, sticky="w", padx=20, pady=18)
        tk.Label(self.title_frame, text="Live Engagement Monitor", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 22)).pack(anchor="w")
        tk.Label(
            self.title_frame,
            text="Primary engagement decision with live contextual affect signals.",
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        self.chip_frame = tk.Frame(self.top_bar, bg=COLORS["panel"])
        self.chip_frame.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=18)
        self.status_chip = self._create_chip(self.chip_frame, self.status_var.get(), COLORS["panel_soft"])
        self.status_chip.pack(side="left", padx=(0, 8))
        self._create_chip(self.chip_frame, self.device_label, COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(self.chip_frame, self.model_variant.title(), COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(self.chip_frame, f"{self.head_count} head{'s' if self.head_count != 1 else ''}", COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(self.chip_frame, f"Seq {self.seq_len}", COLORS["panel_alt"]).pack(side="left")

        self.button_frame = tk.Frame(self.top_bar, bg=COLORS["panel"])
        self.button_frame.grid(row=0, column=2, sticky="e", padx=20, pady=18)
        self.start_button = self._create_button(self.button_frame, "Start Camera", self.start, COLORS["green"])
        self.stop_button = self._create_button(self.button_frame, "Stop Camera", self.stop, COLORS["panel_alt"])
        self.start_pomodoro_button = self._create_button(self.button_frame, "Start Pomodoro", self.start_pomodoro, COLORS["blue_soft"], fg=COLORS["blue"])
        self.stop_pomodoro_button = self._create_button(self.button_frame, "Stop Pomodoro", self.stop_pomodoro, COLORS["panel_alt"])
        self.start_button.pack(side="left", padx=(0, 10))
        self.stop_button.pack(side="left")
        self.start_pomodoro_button.pack(side="left", padx=(10, 0))
        self.stop_pomodoro_button.pack(side="left", padx=(10, 0))
        self.stop_button.configure(state="disabled")
        self.start_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.stop_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

        self.main_area = tk.Frame(self.content_frame, bg=COLORS["bg"])
        self.main_area.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.main_area.grid_anchor("nw")
        self.main_area.grid_columnconfigure(0, weight=3, minsize=0)
        self.main_area.grid_columnconfigure(1, weight=2, minsize=0)
        self.main_area.grid_rowconfigure(0, weight=1)

        self.preview_card = self._create_card(self.main_area, bg=COLORS["panel"], border=COLORS["border"])
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        preview_header = tk.Frame(self.preview_card, bg=COLORS["panel"])
        preview_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
        preview_header.grid_columnconfigure(0, weight=1)
        tk.Label(preview_header, text="Camera Feed", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.preview_badge = self._create_chip(preview_header, self.preview_badge_var.get(), COLORS["panel_soft"])
        self.preview_badge.grid(row=0, column=1, sticky="e")

        self.preview_stage = tk.Frame(self.preview_card, bg=COLORS["preview"])
        self.preview_stage.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        self.preview_stage.configure(height=PREVIEW_STAGE_HEIGHT)
        self.preview_stage.grid_propagate(False)
        self.preview_stage.grid_rowconfigure(0, weight=1)
        self.preview_stage.grid_columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            self.preview_stage,
            text="Camera idle\nStart the camera to begin live monitoring.",
            bg=COLORS["preview"],
            fg=COLORS["text_soft"],
            justify="center",
            font=("Segoe UI", 16),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", self._on_preview_configure)

        self.preview_footer = tk.Label(
            self.preview_card,
            textvariable=self.footer_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            anchor="w",
            justify="left",
            font=("Segoe UI", 10),
        )
        self.preview_footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))

        self.decision_card = self._create_card(self.main_area, bg=COLORS["panel"], border=COLORS["border"])
        self.decision_card.grid(row=0, column=1, sticky="nsew")
        self.decision_card.grid_columnconfigure(0, weight=1)

        self.decision_body = tk.Frame(self.decision_card, bg=COLORS["panel"])
        self.decision_body.grid(row=0, column=0, sticky="nsew", padx=22, pady=22)
        self.decision_body.grid_columnconfigure(0, weight=1)
        self.decision_body.grid_rowconfigure(0, minsize=PRIMARY_DECISION_BOX_HEIGHT)
        self.decision_body.grid_rowconfigure(1, minsize=STAT_BLOCK_HEIGHT)
        self.decision_body.grid_rowconfigure(2, minsize=SPOTLIGHT_BOX_HEIGHT)
        self.decision_body.grid_rowconfigure(3, minsize=POMODORO_CARD_HEIGHT)

        self.primary_box = self._create_card(self.decision_body, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.primary_box.grid(row=0, column=0, sticky="ew")
        self.primary_box.configure(height=PRIMARY_DECISION_BOX_HEIGHT)
        self.primary_box.grid_propagate(False)
        self.primary_box.grid_columnconfigure(0, weight=1)
        self.primary_box.grid_rowconfigure(4, weight=1)

        tk.Label(self.primary_box, text="PRIMARY DECISION", bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))
        self.decision_badge = self._create_chip(self.primary_box, self.status_var.get(), COLORS["panel_soft"])
        self.decision_badge.grid(row=1, column=0, sticky="w", padx=18, pady=(10, 12))

        tk.Label(self.primary_box, textvariable=self.prediction_var, bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 30)).grid(row=2, column=0, sticky="w", padx=18)
        tk.Label(self.primary_box, textvariable=self.confidence_var, bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 13)).grid(row=3, column=0, sticky="w", padx=18, pady=(4, 6))
        self.summary_label = tk.Label(
            self.primary_box,
            textvariable=self.summary_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            wraplength=460,
            justify="left",
            anchor="nw",
            font=("Segoe UI", 10),
        )
        self.summary_label.grid(row=4, column=0, sticky="nsew", padx=18)

        self.primary_meter = tk.Canvas(self.primary_box, height=32, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.primary_meter.grid(row=5, column=0, sticky="ew", padx=18, pady=(10, 14))

        self.stat_row = tk.Frame(self.decision_body, bg=COLORS["panel"])
        self.stat_row.grid(row=1, column=0, sticky="ew", pady=(18, 0))
        self.stat_row.configure(height=STAT_BLOCK_HEIGHT)
        self.stat_row.grid_propagate(False)
        self.stat_row.grid_columnconfigure(0, weight=1)
        self.stat_row.grid_columnconfigure(1, weight=1)
        self.engaged_block = self._create_stat_block(self.stat_row, "Engaged")
        self.not_engaged_block = self._create_stat_block(self.stat_row, "Not Engaged")
        self.engaged_block["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.not_engaged_block["frame"].grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.spotlight_card = self._create_card(self.decision_body, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.spotlight_card.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        self.spotlight_card.configure(height=SPOTLIGHT_BOX_HEIGHT)
        self.spotlight_card.pack_propagate(False)
        tk.Label(self.spotlight_card, textvariable=self.spotlight_label_var, bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=16, pady=(12, 4))
        self.spotlight_value = tk.Label(self.spotlight_card, textvariable=self.spotlight_value_var, bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18))
        self.spotlight_value.pack(anchor="w", padx=16)
        self.spotlight_detail_label = tk.Label(
            self.spotlight_card,
            textvariable=self.spotlight_detail_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            wraplength=470,
            justify="left",
            font=("Segoe UI", 10),
        )
        self.spotlight_detail_label.pack(anchor="w", padx=16, pady=(4, 8))
        self.spotlight_meter = tk.Canvas(self.spotlight_card, height=20, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.spotlight_meter.pack(fill="x", padx=16, pady=(0, 12))

        self.pomodoro_card = self._create_card(self.decision_body, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.pomodoro_card.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        self.pomodoro_card.configure(height=POMODORO_CARD_HEIGHT)
        self.pomodoro_card.grid_propagate(False)
        self.pomodoro_card.grid_columnconfigure(0, weight=1)

        pomodoro_header = tk.Frame(self.pomodoro_card, bg=COLORS["panel_alt"])
        pomodoro_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 10))
        pomodoro_header.grid_columnconfigure(0, weight=1)
        tk.Label(pomodoro_header, text="FOCUS TIMER", bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w")
        self.pomodoro_chip = self._create_chip(pomodoro_header, "Idle", COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.pomodoro_chip.grid(row=0, column=1, sticky="e")

        tk.Label(self.pomodoro_card, textvariable=self.pomodoro_time_var, bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 28)).grid(row=1, column=0, sticky="w", padx=16)
        tk.Label(self.pomodoro_card, textvariable=self.pomodoro_block_var, bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 11)).grid(row=2, column=0, sticky="w", padx=16, pady=(2, 2))
        tk.Label(self.pomodoro_card, textvariable=self.pomodoro_next_var, bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI", 10)).grid(row=3, column=0, sticky="w", padx=16)
        self.pomodoro_note_label = tk.Label(
            self.pomodoro_card,
            textvariable=self.pomodoro_note_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            justify="left",
            anchor="w",
            wraplength=470,
            font=("Segoe UI", 9),
        )
        self.pomodoro_note_label.grid(row=4, column=0, sticky="ew", padx=16, pady=(8, 8))
        self.pomodoro_progress_canvas = tk.Canvas(self.pomodoro_card, height=22, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.pomodoro_progress_canvas.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 14))

        self.bottom_band = self._create_card(self.content_frame, bg=COLORS["panel"], border=COLORS["border"])
        self.bottom_band.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.bottom_band.grid_columnconfigure(0, weight=1)

        self.bottom_header = tk.Frame(self.bottom_band, bg=COLORS["panel"])
        self.bottom_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        self.bottom_header.grid_columnconfigure(0, weight=1)
        self.bottom_title_label = tk.Label(self.bottom_header, text="Signal Overview", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 13))
        self.bottom_title_label.grid(row=0, column=0, sticky="w")
        self.bottom_meta_label = tk.Label(self.bottom_header, textvariable=self.meta_var, bg=COLORS["panel"], fg=COLORS["text_muted"], font=("Segoe UI", 9), justify="left")
        self.bottom_meta_label.grid(row=0, column=1, sticky="e")

        self.summary_tiles_frame = tk.Frame(self.bottom_band, bg=COLORS["panel"])
        self.summary_tiles_frame.grid(row=1, column=0, sticky="ew", padx=14)
        for column in range(3):
            self.summary_tiles_frame.grid_columnconfigure(column, weight=1, uniform="summary_tile")

        self.summary_blocks["last_4"] = self._create_stat_block(self.summary_tiles_frame, "Last 4 Min")
        self.summary_blocks["last_8"] = self._create_stat_block(self.summary_tiles_frame, "Last 8 Min")
        self.summary_blocks["session"] = self._create_stat_block(self.summary_tiles_frame, "Session Avg")
        self.summary_blocks["last_4"]["frame"].grid(row=0, column=0, sticky="ew", padx=6)
        self.summary_blocks["last_8"]["frame"].grid(row=0, column=1, sticky="ew", padx=6)
        self.summary_blocks["session"]["frame"].grid(row=0, column=2, sticky="ew", padx=6)

        self.engagement_summary_note_label = tk.Label(
            self.bottom_band,
            textvariable=self.engagement_summary_note_var,
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            anchor="w",
            justify="left",
            font=("Segoe UI", 9),
        )
        self.engagement_summary_note_label.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 0))

        self.feedback_info_label = tk.Label(
            self.bottom_band,
            textvariable=self.feedback_insight_var,
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            anchor="w",
            justify="left",
            font=("Segoe UI", 9),
        )
        self.feedback_info_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 0))

        self.feedback_status_label = tk.Label(
            self.bottom_band,
            textvariable=self.feedback_status_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            anchor="w",
            justify="left",
            font=("Segoe UI", 9),
        )
        self.feedback_status_label.grid(row=4, column=0, sticky="ew", padx=20, pady=(4, 10))

        self.tiles_frame = tk.Frame(self.bottom_band, bg=COLORS["panel"])
        self.tiles_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        self._build_signal_tiles()

    def _build_signal_tiles(self) -> None:
        tile_specs = [
            {"key": "engaged", "label": "Engaged", "accent": HEAD_PALETTES["engagement"]["accent"], "surface": HEAD_PALETTES["engagement"]["surface"]},
            {"key": "not_engaged", "label": "Not Engaged", "accent": HEAD_PALETTES["not_engaged"]["accent"], "surface": HEAD_PALETTES["not_engaged"]["surface"]},
        ]
        for spec in self.secondary_specs:
            tile_specs.append(
                {
                    "key": f"head:{spec['index']}",
                    "label": spec["label"],
                    "accent": spec["accent"],
                    "surface": spec["surface"],
                    "head_index": spec["index"],
                }
            )

        for spec in tile_specs:
            tile = self._create_card(self.tiles_frame, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
            tk.Label(tile, text=spec["label"], bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=14, pady=(14, 4))
            value = tk.Label(tile, text="50%", bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 24))
            value.pack(anchor="w", padx=14)
            detail = tk.Label(tile, text="Standby", bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI", 9))
            detail.pack(anchor="w", padx=14, pady=(2, 10))
            meter = tk.Canvas(tile, height=16, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
            meter.pack(fill="x", padx=14, pady=(0, 14))
            self.signal_tile_order.append(spec["key"])
            self.signal_tiles[spec["key"]] = {
                "frame": tile,
                "value": value,
                "detail": detail,
                "meter": meter,
                "accent": spec["accent"],
                "surface": spec["surface"],
                "head_index": spec.get("head_index"),
            }

    def _draw_capsule(self, canvas: tk.Canvas, value: float, accent: str, track: str, text: str | None = None) -> None:
        canvas.delete("all")
        width = max(8, canvas.winfo_width() or 220)
        height = max(8, canvas.winfo_height() or 18)
        radius = height / 2
        canvas.create_rectangle(radius, 0, width - radius, height, fill=track, outline=track)
        canvas.create_oval(0, 0, radius * 2, height, fill=track, outline=track)
        canvas.create_oval(width - radius * 2, 0, width, height, fill=track, outline=track)

        fill_w = width * max(0.0, min(1.0, value))
        if fill_w > radius:
            canvas.create_rectangle(radius, 0, fill_w - radius, height, fill=accent, outline=accent)
            canvas.create_oval(0, 0, radius * 2, height, fill=accent, outline=accent)
            canvas.create_oval(fill_w - radius * 2, 0, fill_w, height, fill=accent, outline=accent)
        elif fill_w > 0:
            canvas.create_oval(0, 0, fill_w * 2, height, fill=accent, outline=accent)

        if text:
            canvas.create_text(width / 2, height / 2, text=text, fill=COLORS["text"], font=("Segoe UI Semibold", 9))

    def _draw_primary_meter(self, engaged_score: float, not_engaged_score: float) -> None:
        canvas = self.primary_meter
        canvas.delete("all")
        width = max(16, canvas.winfo_width() or 360)
        height = max(12, canvas.winfo_height() or 32)
        radius = height / 2
        track = COLORS["panel_alt"]

        canvas.create_rectangle(radius, 0, width - radius, height, fill=track, outline=track)
        canvas.create_oval(0, 0, radius * 2, height, fill=track, outline=track)
        canvas.create_oval(width - radius * 2, 0, width, height, fill=track, outline=track)

        left_width = width * max(0.0, min(1.0, engaged_score))
        if left_width > 0:
            fill = HEAD_PALETTES["engagement"]["accent"]
            canvas.create_rectangle(radius, 0, max(radius, left_width), height, fill=fill, outline=fill)
            canvas.create_oval(0, 0, radius * 2, height, fill=fill, outline=fill)
        if not_engaged_score > 0:
            fill = HEAD_PALETTES["not_engaged"]["accent"]
            right_start = width * max(0.0, min(1.0, engaged_score))
            canvas.create_rectangle(max(radius, right_start), 0, width - radius, height, fill=fill, outline=fill)
            canvas.create_oval(width - radius * 2, 0, width, height, fill=fill, outline=fill)
        canvas.create_text(
            width / 2,
            height / 2,
            text=f"Engaged {engaged_score * 100:.0f}%   |   Not Engaged {not_engaged_score * 100:.0f}%",
            fill=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        )

    def _engagement_scores(self, output: np.ndarray) -> tuple[float, float, int]:
        engagement_head = output[self.engagement_head_index]
        engaged = float(sum(engagement_head[idx] for idx in self.positive_class_indices if idx < len(engagement_head)))
        not_engaged = float(sum(engagement_head[idx] for idx in self.negative_class_indices if idx < len(engagement_head)))
        dominant_level = int(np.argmax(engagement_head))
        return engaged, not_engaged, dominant_level

    def _strongest_secondary_signal(self, output: np.ndarray) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for spec in self.secondary_specs:
            head = output[spec["index"]]
            elevated = float(sum(head[idx] for idx in self.positive_class_indices if idx < len(head)))
            dominant_level = int(np.argmax(head))
            candidate = {
                "spec": spec,
                "elevated": elevated,
                "dominant_level": dominant_level,
                "probabilities": head,
            }
            if best is None or elevated > best["elevated"]:
                best = candidate
        return best

    def _stable_choice(self, transition: dict[str, Any], candidate: str, patience: int) -> str:
        current = transition["current"]
        if current is None:
            transition["current"] = candidate
            transition["candidate"] = None
            transition["count"] = 0
            return candidate
        if candidate == current:
            transition["candidate"] = None
            transition["count"] = 0
            return current
        if transition["candidate"] == candidate:
            transition["count"] += 1
        else:
            transition["candidate"] = candidate
            transition["count"] = 1
        if transition["count"] >= patience:
            transition["current"] = candidate
            transition["candidate"] = None
            transition["count"] = 0
        return str(transition["current"])

    def _signal_for_key(self, output: np.ndarray, key: str) -> dict[str, Any] | None:
        if not key.startswith("head:"):
            return None
        try:
            head_index = int(key.split(":", 1)[1])
        except ValueError:
            return None
        spec = next((item for item in self.secondary_specs if item["index"] == head_index), None)
        if spec is None:
            return None
        head = output[head_index]
        elevated = float(sum(head[idx] for idx in self.positive_class_indices if idx < len(head)))
        return {
            "spec": spec,
            "elevated": elevated,
            "dominant_level": int(np.argmax(head)),
            "probabilities": head,
            "held": True,
        }

    def _primary_copy(self, state: str, active_signal: dict[str, Any] | None) -> tuple[str, str]:
        if state == "live_mixed":
            if active_signal is not None:
                return "Mixed Signals", f"Engagement is close while {active_signal['spec']['label'].lower()} is elevated."
            return "Mixed Signals", "Engagement is near the midpoint and the current window is still settling."
        if state == "live_engaged":
            if active_signal is not None:
                return "Engaged", f"High engagement leads while {active_signal['spec']['label'].lower()} is also elevated."
            return "Engaged", "High and very high engagement are leading the window."
        if state == "live_not_engaged":
            if active_signal is not None:
                return "Not Engaged", f"Low engagement leads with {active_signal['spec']['label'].lower()} elevated in the background."
            return "Not Engaged", "Low and very low engagement are leading the window."
        return "Camera Idle", self._idle_summary()

    def _apply_state_palette(self, state: str) -> None:
        style = STATE_STYLES[state]
        accent = style["accent"]
        surface = style["surface"]
        self.status_chip.configure(text=self.status_var.get(), bg=surface, fg=accent)
        self.preview_badge.configure(text=self.preview_badge_var.get(), bg=surface, fg=accent)
        self.decision_badge.configure(text=self.status_var.get(), bg=surface, fg=accent)
        self.decision_card.configure(highlightbackground=mix_color(accent, COLORS["border"], 0.25))

    def _set_state(
        self,
        state: str,
        headline: str,
        confidence: str,
        summary: str,
        footer: str,
        preview_badge: str | None = None,
    ) -> None:
        self.state = state
        self.status_var.set(STATE_STYLES[state]["badge"])
        self.preview_badge_var.set(preview_badge or STATE_STYLES[state]["badge"])
        self.prediction_var.set(headline)
        self.confidence_var.set(confidence)
        self.summary_var.set(summary)
        self.footer_var.set(footer)
        self._apply_state_palette(state)

    def _binary_detail(self, kind: str, score: float, dominant_level: int) -> str:
        level = AFFECT_LEVELS[min(dominant_level, len(AFFECT_LEVELS) - 1)]
        if kind == "engaged":
            return "High + Very High band" if score >= 0.55 else f"Lead level: {level}"
        return "Very Low + Low band" if score >= 0.55 else f"Lead level: {level}"

    def _spotlight_copy(self, signal: dict[str, Any] | None, spotlight_threshold: float | None = None) -> tuple[str, str, str, float, str]:
        if spotlight_threshold is None:
            _, spotlight_threshold = self._effective_thresholds()
        if not self.secondary_specs:
            return "Secondary Spotlight", "Engagement-only model active", "No secondary affect heads are available in this export.", 0.0, COLORS["text_soft"]
        signal_active = signal is not None and (signal.get("held") or signal["elevated"] >= spotlight_threshold)
        if not signal_active:
            return "Secondary Spotlight", "Secondary signals stable", "No non-engagement head is above the promotion threshold.", 0.0, COLORS["text_soft"]
        assert signal is not None
        spec = signal["spec"]
        level = AFFECT_LEVELS[min(signal["dominant_level"], len(AFFECT_LEVELS) - 1)]
        detail_prefix = "Holding trend" if signal.get("held") and signal["elevated"] < spotlight_threshold else "High band"
        detail = f"{detail_prefix} {signal['elevated'] * 100:.0f}% | Dominant level {level}"
        return "Secondary Spotlight", f"{spec['label']} rising", detail, float(signal["elevated"]), spec["accent"]

    def _update_spotlight(self, signal: dict[str, Any] | None, spotlight_threshold: float | None = None) -> None:
        label, value, detail, meter_value, accent = self._spotlight_copy(signal, spotlight_threshold)
        self.spotlight_label_var.set(label)
        self.spotlight_value_var.set(value)
        self.spotlight_detail_var.set(detail)
        self.spotlight_value.configure(fg=accent if meter_value > 0 else COLORS["text"])
        self.spotlight_card.configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
        self._draw_capsule(self.spotlight_meter, meter_value, accent, COLORS["panel_soft"])

    def _update_signal_tiles(self, output: np.ndarray, active_signal: dict[str, Any] | None, spotlight_threshold: float | None = None) -> None:
        if spotlight_threshold is None:
            _, spotlight_threshold = self._effective_thresholds()
        engaged, not_engaged, dominant_level = self._engagement_scores(output)
        engaged_tile = self.signal_tiles.get("engaged")
        not_engaged_tile = self.signal_tiles.get("not_engaged")
        for tile, score, detail in (
            (engaged_tile, engaged, self._binary_detail("engaged", engaged, dominant_level)),
            (not_engaged_tile, not_engaged, self._binary_detail("not_engaged", not_engaged, dominant_level)),
        ):
            if tile is None:
                continue
            accent = tile["accent"]
            tile["value"].configure(text=f"{score * 100:.0f}%")
            tile["detail"].configure(text=detail)
            tile["frame"].configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
            self._draw_capsule(tile["meter"], score, accent, tile["surface"])

        active_key = None
        if active_signal is not None and (active_signal.get("held") or active_signal["elevated"] >= spotlight_threshold):
            active_key = f"head:{active_signal['spec']['index']}"
        for spec in self.secondary_specs:
            tile = self.signal_tiles.get(f"head:{spec['index']}")
            if tile is None:
                continue
            head = output[spec["index"]]
            elevated = float(sum(head[idx] for idx in self.positive_class_indices if idx < len(head)))
            dominant = AFFECT_LEVELS[min(int(np.argmax(head)), len(AFFECT_LEVELS) - 1)]
            tile["value"].configure(text=f"{elevated * 100:.0f}%")
            tile["detail"].configure(text=f"{dominant} tendency")
            border = spec["accent"] if active_key == f"head:{spec['index']}" else mix_color(spec["accent"], COLORS["border_soft"], 0.35)
            tile["frame"].configure(highlightbackground=border)
            self._draw_capsule(tile["meter"], elevated, spec["accent"], spec["surface"])

    def _update_primary_view(self, output: np.ndarray) -> None:
        primary_threshold, spotlight_threshold = self._effective_thresholds()
        engaged, not_engaged, dominant_level = self._engagement_scores(output)
        self.engaged_block["value"].configure(text=f"{engaged * 100:.0f}%")
        self.engaged_block["detail"].configure(text=self._binary_detail("engaged", engaged, dominant_level))
        self.not_engaged_block["value"].configure(text=f"{not_engaged * 100:.0f}%")
        self.not_engaged_block["detail"].configure(text=self._binary_detail("not_engaged", not_engaged, dominant_level))
        self._draw_primary_meter(engaged, not_engaged)

        primary_confidence = max(engaged, not_engaged)
        raw_signal = self._strongest_secondary_signal(output)
        if not self.secondary_specs:
            spotlight_key = "engagement_only"
        elif raw_signal is not None and raw_signal["elevated"] >= spotlight_threshold:
            spotlight_key = f"head:{raw_signal['spec']['index']}"
        else:
            spotlight_key = "stable"
        stable_spotlight_key = self._stable_choice(self.spotlight_transition, spotlight_key, SPOTLIGHT_SWITCH_PATIENCE)
        active_signal = self._signal_for_key(output, stable_spotlight_key)
        spotlight_active = active_signal is not None

        if primary_confidence < primary_threshold and spotlight_active:
            raw_state = "live_mixed"
        elif engaged >= not_engaged:
            raw_state = "live_engaged"
        else:
            raw_state = "live_not_engaged"

        state = self._stable_choice(self.primary_transition, raw_state, STATE_SWITCH_PATIENCE)
        headline, summary = self._primary_copy(state, active_signal)

        confidence = (
            f"{primary_confidence * 100:.0f}% window confidence | "
            f"Primary threshold {primary_threshold * 100:.0f}% | "
            f"Dominant level {AFFECT_LEVELS[dominant_level]}"
        )
        footer = "Live monitoring active. Secondary tiles refresh from the current frame window."
        self._set_state(state, headline, confidence, summary, footer, preview_badge="Live")
        self._update_spotlight(active_signal, spotlight_threshold)
        self._update_signal_tiles(output, active_signal, spotlight_threshold)

    def _draw_preview_placeholder(self) -> None:
        self.preview_label.configure(
            image="",
            text="Camera idle\nStart the camera to begin live monitoring.",
            fg=COLORS["text_soft"],
        )
        self.preview_photo = None

    def _update_preview(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._draw_preview_placeholder()
            return

        display_frame = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
        width = max(320, self.preview_label.winfo_width())
        height = max(240, self.preview_label.winfo_height())
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), COLORS["preview"])
        x_pos = (width - image.width) // 2
        y_pos = (height - image.height) // 2
        canvas.paste(image, (x_pos, y_pos))
        photo = ImageTk.PhotoImage(canvas)
        self.preview_photo = photo
        self.preview_label.configure(image=photo, text="")

    def _on_preview_configure(self, _event) -> None:
        if self.last_frame is None:
            self._draw_preview_placeholder()
        else:
            self._update_preview(self.last_frame)

    def start(self) -> None:
        if self.running:
            return
        self._close_checkin_dialog()
        self.capture = open_camera(self.camera_index)
        if not self.capture or not self.capture.isOpened():
            self._set_error_view(f"Unable to open webcam on camera index {self.camera_index}.")
            return

        self.running = True
        self.session_token += 1
        self.frame_buffer.clear()
        self.last_frame = None
        self.frame_counter = 0
        self.capture_failures = 0
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self.last_prediction_time = 0.0
        self.last_output_time = 0.0
        self._reset_temporal_smoothing()
        self._reset_engagement_summary()
        if not self.pomodoro_active and self.pomodoro_phase not in {"idle", "unavailable"}:
            self._reset_pomodoro_state()
        with self.output_lock:
            self.pending_output = None
            self.pending_error = None
            self.inference_busy = False
            self.active_inference_id = None

        self._set_state(
            "warming_up",
            "Warming Up",
            "Collecting live frame context",
            f"Buffering {self.seq_len} frames before the first decision.",
            f"Frame buffer 0/{self.seq_len}",
            preview_badge="Warming",
        )
        self.engagement_summary_note_var.set("Collecting live predictions for the 4-minute and 8-minute engagement summary.")
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self._sync_control_states()
        self._schedule_capture()
        self._schedule_ui()

    def stop(self) -> None:
        if self.pomodoro_active or self.checkin_dialog is not None or self.pomodoro_prompt_pending:
            self._close_checkin_dialog()
            self._reset_pomodoro_state(phase="stopped", reason="Pomodoro halted because live monitoring stopped.")
        self.running = False
        self.session_token += 1
        if self.capture_after_id:
            self.root.after_cancel(self.capture_after_id)
            self.capture_after_id = None
        if self.ui_after_id:
            self.root.after_cancel(self.ui_after_id)
            self.ui_after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_buffer.clear()
        self.frame_counter = 0
        self.capture_failures = 0
        self.last_output_time = 0.0
        with self.output_lock:
            self.pending_output = None
            self.pending_error = None
            self.inference_busy = False
            self.active_inference_id = None
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self._reset_temporal_smoothing()
        self._set_state("idle", "Camera Idle", "Waiting for live input", self._idle_summary(), self._idle_footer(), preview_badge="Idle")
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        if self.session_engagement_count > 0:
            self.engagement_summary_note_var.set("Camera stopped. Summary blocks show the most recent live session.")
        else:
            self.engagement_summary_note_var.set(self._idle_engagement_summary_note())
        self.last_frame = None
        self._draw_preview_placeholder()
        self._sync_control_states()

    def _set_error_view(self, message: str) -> None:
        if self.pomodoro_phase == "stopped":
            self.pomodoro_status_reason = "Pomodoro halted because live monitoring hit a runtime error."
            self._refresh_pomodoro_ui()
        self._set_state("error", "Runtime Error", message, "Check camera access or the ONNX runtime configuration.", message, preview_badge="Error")
        self.preview_label.configure(image="", text="Runtime error\nSee the status panel for details.", fg=COLORS["red"])
        if self.session_engagement_count > 0:
            self.engagement_summary_note_var.set("Runtime error paused live updates. Summary blocks show the latest completed session data.")
        else:
            self.engagement_summary_note_var.set("Runtime error prevented the engagement summary from collecting live data.")
        self._sync_control_states()

    def _schedule_capture(self, delay_ms: int = 15) -> None:
        if not self.closing:
            self.capture_after_id = self.root.after(delay_ms, self._capture_loop)

    def _schedule_ui(self) -> None:
        if not self.closing:
            self.ui_after_id = self.root.after(50, self._update_loop)

    def _try_reopen_camera(self) -> bool:
        if self.capture is not None:
            self.capture.release()
        self.capture = open_camera(self.camera_index)
        return bool(self.capture and self.capture.isOpened())

    def _run_inference(self, frames: list[np.ndarray], inference_id: tuple[int, int]) -> None:
        token, frame_id = inference_id
        try:
            output = predict_output(
                session=self.session,
                input_name=self.input_name,
                output_name=self.output_name,
                frames=frames,
                img_size=self.img_size,
                class_count=self.class_count,
                head_count=self.head_count,
            )
            with self.output_lock:
                if token == self.session_token and self.active_inference_id == inference_id:
                    self.pending_output = {
                        "frame_id": frame_id,
                        "created_at": time.time(),
                        "output": output,
                    }
        except Exception as exc:
            with self.output_lock:
                if token == self.session_token and self.active_inference_id == inference_id:
                    self.pending_error = str(exc)
        finally:
            with self.output_lock:
                if self.active_inference_id == inference_id:
                    self.inference_busy = False

    def _capture_loop(self) -> None:
        if not self.running or self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            self.capture_failures += 1
            if self.capture_failures < CAMERA_READ_FAILURE_PATIENCE:
                self._schedule_capture(CAMERA_RECOVERY_DELAY_MS)
                return

            if self._try_reopen_camera():
                self.capture_failures = 0
                self.frame_buffer.clear()
                self.display_output = self._neutral_output()
                self.target_output = self._neutral_output()
                self.last_prediction_time = 0.0
                self.last_output_time = 0.0
                self._reset_temporal_smoothing()
                self._reset_engagement_summary()
                with self.output_lock:
                    self.pending_output = None
                    self.pending_error = None
                    self.inference_busy = False
                    self.active_inference_id = None
                self._update_spotlight(None)
                self._update_signal_tiles(self.display_output, None)
                self._set_state(
                    "warming_up",
                    "Warming Up",
                    "Collecting live frame context",
                    f"Buffering {self.seq_len} frames before the first decision.",
                    f"Frame buffer 0/{self.seq_len}",
                    preview_badge="Warming",
                )
                if self.pomodoro_active or self.checkin_dialog is not None or self.pomodoro_prompt_pending:
                    self._close_checkin_dialog()
                    self._reset_pomodoro_state(phase="stopped", reason="Pomodoro stopped after the camera recovered. Start a fresh 24-minute block.")
                self.engagement_summary_note_var.set("Camera recovered. Collecting a fresh engagement summary window.")
                self._schedule_capture(CAMERA_RECOVERY_DELAY_MS)
                return

            self.stop()
            self._set_error_view(f"The webcam on camera index {self.camera_index} stopped responding and could not be reopened.")
            return

        self.capture_failures = 0
        self.last_frame = frame
        self.frame_counter += 1
        self.frame_buffer.append(frame.copy())
        self._record_pomodoro_frame_sample(frame, time.time())
        self._update_preview(frame)

        buffered = len(self.frame_buffer)
        if buffered < self.seq_len:
            self._set_state(
                "warming_up",
                "Warming Up",
                "Collecting live frame context",
                f"Buffering {self.seq_len} frames before the first decision.",
                f"Frame buffer {buffered}/{self.seq_len}",
                preview_badge="Warming",
            )
        elif time.time() - self.last_prediction_time >= INFERENCE_INTERVAL_SEC:
            frames_for_inference: list[np.ndarray] | None = None
            inference_id: tuple[int, int] | None = None
            with self.output_lock:
                if not self.inference_busy:
                    inference_id = (self.session_token, self.frame_counter)
                    frames_for_inference = list(self.frame_buffer)
                    self.active_inference_id = inference_id
                    self.inference_busy = True
                    self.last_prediction_time = time.time()
            if frames_for_inference is not None and inference_id is not None:
                threading.Thread(target=self._run_inference, args=(frames_for_inference, inference_id), daemon=True).start()

        self._schedule_capture()

    def _update_loop(self) -> None:
        if not self.running:
            return

        pending_output: dict[str, Any] | None = None
        pending_error = None
        with self.output_lock:
            if self.pending_output is not None:
                pending_output = dict(self.pending_output)
                self.pending_output = None
            if self.pending_error is not None:
                pending_error = self.pending_error
                self.pending_error = None

        if pending_error:
            self.stop()
            self._set_error_view(pending_error)
            return

        if pending_output is not None:
            frame_lag = self.frame_counter - int(pending_output["frame_id"])
            if frame_lag <= max(6, min(self.seq_len, INFERENCE_STALE_FRAME_TOLERANCE)):
                output = np.asarray(pending_output["output"], dtype=np.float32)
                self.output_history.append(output)
                stacked = np.stack(list(self.output_history), axis=0)
                self.target_output = stacked.mean(axis=0).astype(np.float32)
                self.last_output_time = time.time()
                self._record_engagement_sample(self.target_output, self.last_output_time)
                self._record_pomodoro_output_sample(self.target_output, self.last_output_time)

        if self.target_output is not None and len(self.frame_buffer) >= self.seq_len:
            self.display_output = ((1.0 - DISPLAY_BLEND) * self.display_output + DISPLAY_BLEND * self.target_output).astype(np.float32)
            self._update_primary_view(self.display_output)

        self._update_pomodoro_timer()
        self._schedule_ui()

    def on_close(self) -> None:
        self.closing = True
        self._close_checkin_dialog()
        self.stop()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live engagement desktop app.")
    parser.add_argument("--variant", choices=["auto", "engagement", "multiaffect"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    runtime = load_model(args.variant)
    runtime["camera_index"] = int(args.camera_index)
    root = tk.Tk()
    app = EngagementApp(root, runtime)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
