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
from utils.affect import (
    AFFECT_COLUMNS,
    AFFECT_DISPLAY_NAMES,
    DISPLAY_AFFECT_LEVELS,
    display_level_index,
    infer_display_level,
)
from utils.guidance import (
    AffectProfile,
    MindfulnessPracticeSelection,
    PomodoroPracticeSelection,
    MINDFULNESS_CHECKIN_INTERVAL_SECONDS,
    MINDFULNESS_TOTAL_SECONDS,
    MINDFULNESS_STEERING_OPTIONS,
    POMODORO_BLOCK_MINUTES,
    POMODORO_BLOCK_SECONDS,
    POMODORO_STEERING_HISTORY_WINDOW,
    POMODORO_TOTAL_SECONDS,
    format_clock,
    mindfulness_checkin_boundaries,
    mindfulness_selection_from_practice_id,
    mindfulness_steering_key_for_profile,
    mindfulness_steering_option,
    mindfulness_timer_view,
    pomodoro_guidance_for_profile,
    pomodoro_selection_from_practice_id,
    pomodoro_timer_view,
    select_pomodoro_practice,
    select_mindfulness_practice,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEFAULT_SEQ_LEN = 30
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 4

WINDOW_MIN_WIDTH = 620
WINDOW_MIN_HEIGHT = 620
WINDOW_EDGE_MARGIN = 16
WINDOW_VERTICAL_MARGIN = 24
CAMERA_PANEL_WIDTH = 780
ENGAGEMENT_PANEL_WIDTH = 460
PREVIEW_STAGE_HEIGHT = 412
PREVIEW_STAGE_COMPACT_HEIGHT = 336
ENGAGEMENT_PANEL_HEIGHT = 532
PRIMARY_DECISION_BOX_HEIGHT = 260
STAT_BLOCK_HEIGHT = 96
SPOTLIGHT_BOX_HEIGHT = 182
POMODORO_CARD_HEIGHT = 0
MINDFULNESS_CARD_HEIGHT = 0
TOP_BAR_STACK_BREAKPOINT = 1240
MAIN_STACK_BREAKPOINT = 1180
TIMER_STACK_BREAKPOINT = 980
TILE_MIN_WIDTH = 220
SCROLL_NOTCH_PIXELS = 36.0
SCROLL_EASING = 0.32
SCROLL_FRAME_MS = 8
SCROLL_MIN_STEP_PIXELS = 0.5
LAYOUT_DEBOUNCE_MS = 60
PREVIEW_REFRESH_DEBOUNCE_MS = 110
PREVIEW_RESIZE_EPSILON = 6
LAYOUT_HYSTERESIS = 48
PREVIEW_RENDER_INTERVAL_SEC = 1.0 / 24.0
PANEL_RENDER_INTERVAL_SEC = 0.0
INTERACTION_COOLDOWN_SEC = 0.14
SIGNAL_TILE_HEIGHT = 168
UI_UPDATE_INTERVAL_MS = 50
PREVIEW_UPDATE_INTERVAL_SEC = 1.0 / 24.0
WARMING_STATUS_STEP = 5

SPOTLIGHT_THRESHOLD = 0.48
PRIMARY_CONFIDENCE_THRESHOLD = 0.58
INFERENCE_INTERVAL_SEC = 0.18
DISPLAY_BLEND = 0.28
OUTPUT_HISTORY_WINDOW = 8
STATE_SWITCH_PATIENCE = 4
SPOTLIGHT_SWITCH_PATIENCE = 4
DISPLAY_LEVEL_SWITCH_PATIENCE = 4
MEDIUM_DISPLAY_SWITCH_PATIENCE = 6
MIRROR_PREVIEW = True
CAMERA_READ_FAILURE_PATIENCE = 8
CAMERA_RECOVERY_DELAY_MS = 180
INFERENCE_STALE_FRAME_TOLERANCE = 18
SUMMARY_WINDOWS_MINUTES = (4, 8)
SUMMARY_MAX_WINDOW_SEC = SUMMARY_WINDOWS_MINUTES[-1] * 60
POMODORO_TOTAL_MINUTES = 24
MINDFULNESS_TOTAL_MINUTES = 8
POMODORO_FRAME_SAMPLE_INTERVAL_SEC = 4.0
POMODORO_FEEDBACK_SOURCE = "pomodoro_checkin"
POMODORO_FINAL_FEEDBACK_SOURCE = "pomodoro_final_experience"
MINDFULNESS_FINAL_FEEDBACK_SOURCE = "mindfulness_final_experience"

FINAL_EXPERIENCE_RATING_OPTIONS = (
    ("1", "Not helpful"),
    ("2", "Slightly helpful"),
    ("3", "Mixed"),
    ("4", "Helpful"),
    ("5", "Very helpful"),
)
MINDFULNESS_FINAL_OUTCOME_TAGS = (
    "Calmer",
    "Clearer",
    "More grounded",
    "Still unsettled",
    "No clear shift",
)
POMODORO_FINAL_OUTCOME_TAGS = (
    "More focused",
    "Clearer next step",
    "Less stuck",
    "Still stuck",
    "No clear shift",
)

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

DISPLAY_LEVEL_SHORT_LABELS = ("VL", "L", "M", "H", "VH")
INVERSE_DISPLAY_LEVEL_LABELS = {
    "Very Low": "Very High",
    "Low": "High",
    "Medium": "Medium",
    "High": "Low",
    "Very High": "Very Low",
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
        self.mindfulness_checkin_dialog: tk.Toplevel | None = None
        self.pomodoro_final_dialog: tk.Toplevel | None = None
        self.mindfulness_final_dialog: tk.Toplevel | None = None
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
        self.display_level_transitions: dict[str, dict[str, Any]] = {}
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
        self.timer_cards_stacked = False
        self.pomodoro_progress_canvas: tk.Canvas | None = None
        self.mindfulness_progress_canvas: tk.Canvas | None = None
        self.scroll_animation_after_id: str | None = None
        self.scroll_target_offset = 0.0
        self.scroll_pending_pixels = 0.0
        self.last_scroll_canvas_size = (0, 0)
        self.last_root_size = (0, 0)
        self.last_preview_update_monotonic = 0.0
        self.last_warming_status_buffered = -1
        self.top_bar_layout_compact: bool | None = None
        self.control_buttons_compact: bool | None = None
        self.main_area_layout_stacked: bool | None = None
        self.timer_row_layout_stacked: bool | None = None
        self.bottom_header_layout_compact: bool | None = None
        self.signal_tile_layout_columns: int | None = None
        self.top_chip_layout_columns: int | None = None
        self.preview_refresh_after_id: str | None = None
        self.preview_last_size = (0, 0)
        self.preview_requested_size = (0, 0)
        self.preview_resize_busy_until = 0.0
        self.last_preview_render_at = 0.0
        self.last_panel_render_at = 0.0
        self.interaction_busy_until = 0.0

        self.pomodoro_active = False
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_final_prompt_pending = False
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
        self.pomodoro_recent_affect_profiles: deque[AffectProfile] = deque(maxlen=POMODORO_STEERING_HISTORY_WINDOW)
        self.pomodoro_recent_block_profiles: deque[AffectProfile] = deque(maxlen=3)
        self.pomodoro_current_practice_id: str | None = None
        self.pomodoro_current_why_selected = ""
        self.pomodoro_current_stability_label = ""
        self.pomodoro_current_stability_reason = ""
        self.pomodoro_last_switch_monotonic: float | None = None
        self.mindfulness_active = False
        self.mindfulness_paused = False
        self.mindfulness_prompt_pending = False
        self.mindfulness_final_prompt_pending = False
        self.mindfulness_remaining_seconds = float(MINDFULNESS_TOTAL_SECONDS)
        self.mindfulness_elapsed_seconds = 0.0
        self.mindfulness_last_tick_monotonic: float | None = None
        self.mindfulness_session_start_epoch: float | None = None
        self.mindfulness_phase = "idle"
        self.mindfulness_status_reason = ""
        self.mindfulness_checkin_boundaries = mindfulness_checkin_boundaries()
        self.mindfulness_completed_checkins = 0
        self.mindfulness_next_checkin_elapsed_seconds = (
            float(self.mindfulness_checkin_boundaries[0]) if self.mindfulness_checkin_boundaries else None
        )
        self.mindfulness_recent_steering_keys: list[str] = []
        self.mindfulness_segment_practice_id: str | None = None
        self.mindfulness_segment_why_selected = ""
        self.mindfulness_segment_source = ""
        self.latest_affect_profile = AffectProfile(
            state="idle",
            engagement_label=None,
            boredom_label=None,
            confusion_label=None,
            frustration_label=None,
        )

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
        self.learning_suggestion_var = tk.StringVar(
            value=f"Learning suggestion: {pomodoro_guidance_for_profile(self.latest_affect_profile).body}"
        )
        self.pomodoro_status_var = tk.StringVar(value="")
        self.pomodoro_time_var = tk.StringVar(value="")
        self.pomodoro_block_var = tk.StringVar(value="")
        self.pomodoro_next_var = tk.StringVar(value="")
        self.pomodoro_note_var = tk.StringVar(value="")
        self.mindfulness_status_var = tk.StringVar(value="")
        self.mindfulness_time_var = tk.StringVar(value="")
        self.mindfulness_block_var = tk.StringVar(value="")
        self.mindfulness_next_var = tk.StringVar(value="")
        self.mindfulness_note_var = tk.StringVar(value="")

        self._set_initial_window()
        self._build_root_scaffold()
        self._build_ui()
        self._refresh_feedback_insight()
        self._reset_engagement_summary()
        self._reset_pomodoro_state()
        self._reset_mindfulness_state()
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self._apply_state_palette("idle")
        self._sync_control_states()
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.after_idle(self._apply_responsive_layout)
        self._schedule_ui()

    def _reset_temporal_smoothing(self) -> None:
        self.output_history.clear()
        self.primary_transition = {"current": None, "candidate": None, "count": 0}
        self.spotlight_transition = {"current": None, "candidate": None, "count": 0}
        self.display_level_transitions.clear()

    def _fallback_affect_profile(self, state: str | None = None) -> AffectProfile:
        return AffectProfile(
            state=state or self.state,
            engagement_label=None,
            boredom_label=None,
            confusion_label=None,
            frustration_label=None,
        )

    def _affect_profile_from_display_states(
        self,
        state: str,
        display_states: dict[int, dict[str, Any]] | None = None,
    ) -> AffectProfile:
        if not display_states:
            return self._fallback_affect_profile(state)

        labels: dict[str, str | None] = {
            "engagement": None,
            "boredom": None,
            "confusion": None,
            "frustration": None,
        }
        for spec in self.head_specs:
            key = _normalize_head_name(spec["name"])
            if key not in labels:
                continue
            display_state = display_states.get(spec["index"])
            labels[key] = None if display_state is None else str(display_state["label"])

        return AffectProfile(
            state=state,
            engagement_label=labels["engagement"],
            boredom_label=labels["boredom"],
            confusion_label=labels["confusion"],
            frustration_label=labels["frustration"],
        )

    def _refresh_learning_suggestion(self, profile: AffectProfile | None = None) -> None:
        resolved_profile = self.latest_affect_profile if profile is None else profile
        suggestion = pomodoro_guidance_for_profile(resolved_profile)
        self.learning_suggestion_var.set(f"Learning suggestion: {suggestion.body}")

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

    def _display_state_from_probabilities(self, probabilities: np.ndarray | list[float] | tuple[float, ...]) -> dict[str, Any]:
        inferred = infer_display_level([float(value) for value in probabilities])
        return {
            "label": str(inferred["label"]),
            "index": int(inferred["index"]),
            "score": float(inferred["score"]),
            "marker_position": float(inferred["marker_position"]),
        }

    def _inverse_display_state(self, display_state: dict[str, Any]) -> dict[str, Any]:
        label = str(display_state["label"])
        index = int(display_state["index"])
        score = float(display_state["score"])
        marker_position = float(display_state["marker_position"])
        return {
            "label": INVERSE_DISPLAY_LEVEL_LABELS.get(label, label),
            "index": max(0, min(len(DISPLAY_AFFECT_LEVELS) - 1, (len(DISPLAY_AFFECT_LEVELS) - 1) - index)),
            "score": max(0.0, min(4.0, 4.0 - score)),
            "marker_position": max(0.0, min(1.0, 1.0 - marker_position)),
        }

    def _stable_display_state(self, key: str, display_state: dict[str, Any]) -> dict[str, Any]:
        transition = self.display_level_transitions.setdefault(
            key,
            {"current": None, "candidate": None, "count": 0, "state": None},
        )
        candidate_label = str(display_state["label"])
        current_label = transition["current"]

        if current_label is None:
            transition["current"] = candidate_label
            transition["candidate"] = None
            transition["count"] = 0
            transition["state"] = dict(display_state)
            return dict(display_state)

        if candidate_label == current_label:
            transition["candidate"] = None
            transition["count"] = 0
            transition["state"] = dict(display_state)
            return dict(display_state)

        if transition["candidate"] == candidate_label:
            transition["count"] += 1
        else:
            transition["candidate"] = candidate_label
            transition["count"] = 1

        patience = MEDIUM_DISPLAY_SWITCH_PATIENCE if candidate_label == "Medium" else DISPLAY_LEVEL_SWITCH_PATIENCE
        if transition["count"] >= patience:
            transition["current"] = candidate_label
            transition["candidate"] = None
            transition["count"] = 0
            transition["state"] = dict(display_state)

        held_state = transition.get("state") or display_state
        return {
            "label": str(held_state["label"]),
            "index": int(held_state["index"]),
            "score": float(held_state["score"]),
            "marker_position": float(held_state["marker_position"]),
        }

    def _stable_display_states(self, output: np.ndarray) -> dict[int, dict[str, Any]]:
        return {
            head_index: self._stable_display_state(
                f"head:{head_index}",
                self._display_state_from_probabilities(np.asarray(output[head_index], dtype=np.float32)),
            )
            for head_index in range(min(self.head_count, int(np.asarray(output).shape[0])))
        }

    def _idle_pomodoro_note(self) -> str:
        if not self.pomodoro_supported:
            return "Pomodoro check-ins need the multi-affect model with engagement, boredom, confusion, and frustration heads."
        return "Start Pomodoro to begin a 24-minute focus block with self-checks every 8 minutes."

    def _format_clock(self, seconds: float) -> str:
        return format_clock(seconds)

    def _reset_pomodoro_block_capture(self, start_epoch: float | None = None) -> None:
        self.pomodoro_block_outputs.clear()
        self.pomodoro_block_frames.clear()
        self.pomodoro_block_start_epoch = start_epoch
        self.pomodoro_pending_window_end_epoch = None
        self.pomodoro_last_frame_sample_at = 0.0

    def _reset_pomodoro_state(self, *, phase: str | None = None, reason: str = "") -> None:
        self._close_pomodoro_final_dialog()
        self.pomodoro_active = False
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_final_prompt_pending = False
        self.pomodoro_completed_blocks = 0
        self.pomodoro_current_block_index = 0
        self.pomodoro_remaining_seconds = float(POMODORO_TOTAL_SECONDS)
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_last_tick_monotonic = None
        self.pomodoro_session_start_epoch = None
        self.pomodoro_status_reason = reason
        self._reset_pomodoro_block_capture()
        self.pomodoro_recent_affect_profiles.clear()
        self.pomodoro_recent_block_profiles.clear()
        self.pomodoro_current_practice_id = None
        self.pomodoro_current_why_selected = ""
        self.pomodoro_current_stability_label = ""
        self.pomodoro_current_stability_reason = ""
        self.pomodoro_last_switch_monotonic = None
        self.pomodoro_phase = phase or ("idle" if self.pomodoro_supported else "unavailable")
        self._refresh_pomodoro_ui()

    def _pomodoro_selection_for_guidance(self) -> PomodoroPracticeSelection:
        if self.pomodoro_current_practice_id is not None:
            return pomodoro_selection_from_practice_id(
                self.pomodoro_current_practice_id,
                why_selected=self.pomodoro_current_why_selected,
                stability_label=self.pomodoro_current_stability_label or "Monitoring",
                stability_reason=self.pomodoro_current_stability_reason,
            )
        return select_pomodoro_practice(
            self.latest_affect_profile,
            recent_profiles=tuple(self.pomodoro_recent_affect_profiles),
            recent_block_profiles=tuple(self.pomodoro_recent_block_profiles),
            block_elapsed_seconds=self.pomodoro_block_elapsed_seconds,
        )

    def _apply_pomodoro_selection(
        self,
        selection: PomodoroPracticeSelection,
        *,
        switched: bool,
        switched_at: float | None = None,
    ) -> None:
        self.pomodoro_current_practice_id = selection.practice_id
        self.pomodoro_current_why_selected = selection.why_selected
        self.pomodoro_current_stability_label = selection.stability_label
        self.pomodoro_current_stability_reason = selection.stability_reason
        if switched:
            self.pomodoro_last_switch_monotonic = time.monotonic() if switched_at is None else switched_at

    def _update_pomodoro_practice(self, *, force: bool = False) -> None:
        if not self.pomodoro_supported:
            return
        now = time.monotonic()
        seconds_since_switch = None
        if self.pomodoro_last_switch_monotonic is not None:
            seconds_since_switch = max(0.0, now - self.pomodoro_last_switch_monotonic)
        selection = select_pomodoro_practice(
            self.latest_affect_profile,
            recent_profiles=tuple(self.pomodoro_recent_affect_profiles),
            recent_block_profiles=tuple(self.pomodoro_recent_block_profiles),
            current_practice_id=self.pomodoro_current_practice_id,
            seconds_since_switch=seconds_since_switch,
            block_elapsed_seconds=self.pomodoro_block_elapsed_seconds,
        )
        previous_practice_id = self.pomodoro_current_practice_id
        switched = force or previous_practice_id != selection.practice_id
        if previous_practice_id is None:
            switched = True
        self._apply_pomodoro_selection(selection, switched=switched, switched_at=now)

    def _record_pomodoro_affect_profile(self, profile: AffectProfile) -> None:
        if not self.pomodoro_active or self.pomodoro_paused or not self.pomodoro_supported:
            return
        if not self.running or profile.state not in {"live_engaged", "live_not_engaged", "live_mixed"}:
            return
        self.pomodoro_recent_affect_profiles.append(profile)
        self._update_pomodoro_practice()

    def _reset_mindfulness_state(self, *, phase: str = "idle", reason: str = "") -> None:
        self._close_mindfulness_checkin_dialog()
        self._close_mindfulness_final_dialog()
        self.mindfulness_active = False
        self.mindfulness_paused = False
        self.mindfulness_prompt_pending = False
        self.mindfulness_final_prompt_pending = False
        self.mindfulness_remaining_seconds = float(MINDFULNESS_TOTAL_SECONDS)
        self.mindfulness_elapsed_seconds = 0.0
        self.mindfulness_last_tick_monotonic = None
        self.mindfulness_session_start_epoch = None
        self.mindfulness_phase = phase
        self.mindfulness_status_reason = reason
        self.mindfulness_completed_checkins = 0
        self.mindfulness_next_checkin_elapsed_seconds = (
            float(self.mindfulness_checkin_boundaries[0]) if self.mindfulness_checkin_boundaries else None
        )
        self.mindfulness_recent_steering_keys = []
        self.mindfulness_segment_practice_id = None
        self.mindfulness_segment_why_selected = ""
        self.mindfulness_segment_source = ""
        self._refresh_mindfulness_ui()

    def _mindfulness_selection_for_guidance(self) -> MindfulnessPracticeSelection:
        if self.mindfulness_segment_practice_id is not None:
            return mindfulness_selection_from_practice_id(
                self.mindfulness_segment_practice_id,
                elapsed_seconds=self.mindfulness_elapsed_seconds,
                why_selected=self.mindfulness_segment_why_selected,
                steering_source=self.mindfulness_segment_source or "checkin",
            )
        return select_mindfulness_practice(
            self.latest_affect_profile,
            elapsed_seconds=self.mindfulness_elapsed_seconds,
        )

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
        selection = self._pomodoro_selection_for_guidance()
        guidance = pomodoro_guidance_for_profile(self.latest_affect_profile)
        view = pomodoro_timer_view(
            supported=self.pomodoro_supported,
            phase=self.pomodoro_phase,
            remaining_seconds=self.pomodoro_remaining_seconds,
            block_elapsed_seconds=self.pomodoro_block_elapsed_seconds,
            completed_blocks=self.pomodoro_completed_blocks,
            current_block_index=self.pomodoro_current_block_index,
            status_reason=self.pomodoro_status_reason,
            selection=selection,
            guidance=guidance,
        )
        status = view.status
        self.pomodoro_status_var.set(view.status)
        self.pomodoro_time_var.set(view.time_text)
        self.pomodoro_block_var.set(view.block_text)
        self.pomodoro_next_var.set(view.next_text)
        self.pomodoro_note_var.set(view.note_text)

        if self.pomodoro_phase == "running":
            accent = COLORS["blue"]
            surface = COLORS["blue_soft"]
        elif self.pomodoro_phase == "paused":
            accent = COLORS["amber"]
            surface = COLORS["amber_soft"]
        elif self.pomodoro_phase == "reflect":
            accent = COLORS["blue"]
            surface = COLORS["blue_soft"]
        elif self.pomodoro_phase == "complete":
            accent = COLORS["green"]
            surface = COLORS["green_soft"]
        elif self.pomodoro_phase == "stopped":
            accent = COLORS["red"]
            surface = COLORS["red_soft"]
        else:
            accent = COLORS["text_soft"]
            surface = COLORS["panel_soft"]

        if hasattr(self, "pomodoro_chip"):
            self.pomodoro_chip.configure(text=status, bg=surface, fg=accent)
        if hasattr(self, "pomodoro_card"):
            self.pomodoro_card.configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
        self._draw_pomodoro_progress(view.completed_blocks, view.current_progress, accent)

    def _apply_mindfulness_card_style(
        self,
        *,
        card_bg: str,
        border: str,
        accent: str,
        surface: str,
        title_fg: str,
        time_fg: str,
        block_fg: str,
        next_fg: str,
        note_fg: str,
        progress_track: str,
        progress: float,
    ) -> None:
        if hasattr(self, "mindfulness_chip"):
            self.mindfulness_chip.configure(text=self.mindfulness_status_var.get(), bg=surface, fg=accent)
        if hasattr(self, "mindfulness_card"):
            self.mindfulness_card.configure(bg=card_bg, highlightbackground=border)
        if hasattr(self, "mindfulness_header"):
            self.mindfulness_header.configure(bg=card_bg)
        if hasattr(self, "mindfulness_title_label"):
            self.mindfulness_title_label.configure(bg=card_bg, fg=title_fg)
        if hasattr(self, "mindfulness_time_label"):
            self.mindfulness_time_label.configure(bg=card_bg, fg=time_fg)
        if hasattr(self, "mindfulness_block_label"):
            self.mindfulness_block_label.configure(bg=card_bg, fg=block_fg)
        if hasattr(self, "mindfulness_next_label"):
            self.mindfulness_next_label.configure(bg=card_bg, fg=next_fg)
        if hasattr(self, "mindfulness_note_label"):
            self.mindfulness_note_label.configure(bg=card_bg, fg=note_fg)
        if self.mindfulness_progress_canvas is not None:
            self.mindfulness_progress_canvas.configure(bg=card_bg)
            self._draw_capsule(self.mindfulness_progress_canvas, progress, accent, progress_track)

    def _refresh_mindfulness_ui(self) -> None:
        selection = self._mindfulness_selection_for_guidance()
        view = mindfulness_timer_view(
            phase=self.mindfulness_phase,
            remaining_seconds=self.mindfulness_remaining_seconds,
            elapsed_seconds=self.mindfulness_elapsed_seconds,
            status_reason=self.mindfulness_status_reason,
            selection=selection,
            next_checkin_seconds=(
                None
                if self.mindfulness_phase != "running" or self.mindfulness_next_checkin_elapsed_seconds is None
                else max(0.0, self.mindfulness_next_checkin_elapsed_seconds - self.mindfulness_elapsed_seconds)
            ),
        )
        self.mindfulness_status_var.set(view.status)
        self.mindfulness_time_var.set(view.time_text)
        self.mindfulness_block_var.set(view.block_text)
        self.mindfulness_next_var.set(view.next_text)
        self.mindfulness_note_var.set(view.note_text)

        camera_off_idle = (
            not self.running
            and not self.mindfulness_active
            and not self.mindfulness_paused
            and not self.mindfulness_prompt_pending
            and not self.mindfulness_final_prompt_pending
            and self.mindfulness_final_dialog is None
        )
        if self.mindfulness_phase == "running":
            accent = COLORS["green"]
            surface = COLORS["green_soft"]
        elif self.mindfulness_phase == "paused":
            accent = COLORS["amber"]
            surface = COLORS["amber_soft"]
        elif self.mindfulness_phase == "reflect":
            accent = COLORS["blue"]
            surface = COLORS["blue_soft"]
        elif self.mindfulness_phase == "complete":
            accent = COLORS["blue"]
            surface = COLORS["blue_soft"]
        elif self.mindfulness_phase == "stopped":
            accent = COLORS["red"]
            surface = COLORS["red_soft"]
        else:
            accent = COLORS["text_soft"]
            surface = COLORS["panel_soft"]

        card_bg = COLORS["panel_alt"]
        title_fg = COLORS["text_muted"]
        time_fg = COLORS["text"]
        block_fg = COLORS["text_soft"]
        next_fg = COLORS["text_muted"]
        note_fg = COLORS["text_soft"]
        progress_track = COLORS["panel_soft"]
        border = mix_color(accent, COLORS["border_soft"], 0.35)
        if camera_off_idle:
            card_bg = mix_color(COLORS["panel_alt"], COLORS["panel_soft"], 0.55)
            accent = COLORS["text_muted"]
            surface = COLORS["panel_soft"]
            title_fg = COLORS["text_muted"]
            time_fg = COLORS["text_soft"]
            block_fg = COLORS["text_muted"]
            next_fg = COLORS["text_soft"]
            note_fg = COLORS["text_muted"]
            progress_track = mix_color(COLORS["panel_soft"], COLORS["panel_alt"], 0.4)
            border = mix_color(COLORS["text_muted"], COLORS["border_soft"], 0.55)
        self._apply_mindfulness_card_style(
            card_bg=card_bg,
            border=border,
            accent=accent,
            surface=surface,
            title_fg=title_fg,
            time_fg=time_fg,
            block_fg=block_fg,
            next_fg=next_fg,
            note_fg=note_fg,
            progress_track=progress_track,
            progress=view.progress,
        )

    def _sync_control_states(self) -> None:
        camera_live = bool(self.running)
        if hasattr(self, "start_button"):
            if camera_live:
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
            else:
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled", bg=COLORS["panel_alt"], fg=COLORS["text"])

        if not hasattr(self, "start_pomodoro_button"):
            return

        pomodoro_open = (
            self.pomodoro_active
            or self.checkin_dialog is not None
            or self.pomodoro_final_dialog is not None
            or self.pomodoro_final_prompt_pending
        )
        pomodoro_stoppable = self.pomodoro_active or self.checkin_dialog is not None or self.pomodoro_prompt_pending
        if camera_live and self.pomodoro_supported and not pomodoro_open:
            self.start_pomodoro_button.configure(state="normal", bg=COLORS["blue_soft"], fg=COLORS["blue"])
        else:
            self.start_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

        if camera_live and pomodoro_stoppable:
            self.stop_pomodoro_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
        else:
            self.stop_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

        mindfulness_open = (
            self.mindfulness_active
            or self.mindfulness_checkin_dialog is not None
            or self.mindfulness_prompt_pending
            or self.mindfulness_final_dialog is not None
            or self.mindfulness_final_prompt_pending
        )
        mindfulness_stoppable = (
            self.mindfulness_active
            or self.mindfulness_checkin_dialog is not None
            or self.mindfulness_prompt_pending
        )
        if mindfulness_open:
            self.start_mindfulness_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
            if camera_live and mindfulness_stoppable:
                self.stop_mindfulness_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
            else:
                self.stop_mindfulness_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
        elif camera_live:
            self.start_mindfulness_button.configure(
                state="normal",
                bg=COLORS["green_soft"],
                fg=COLORS["green"],
            )
        else:
            self.start_mindfulness_button.configure(
                state="disabled",
                bg=COLORS["panel_soft"],
                fg=COLORS["text_soft"],
            )
            self.stop_mindfulness_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
            return
        if not mindfulness_open:
            self.stop_mindfulness_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

    def _modal_dialogs(self) -> tuple[tk.Toplevel | None, ...]:
        return (
            self.checkin_dialog,
            self.mindfulness_checkin_dialog,
            self.pomodoro_final_dialog,
            self.mindfulness_final_dialog,
        )

    def _open_pending_reflection_dialogs(self) -> None:
        active_modal = any(dialog is not None for dialog in self._modal_dialogs())
        if active_modal:
            return
        if self.pomodoro_final_prompt_pending and self.pomodoro_final_dialog is None:
            self._open_pomodoro_final_dialog()
            return
        if self.mindfulness_final_prompt_pending and self.mindfulness_final_dialog is None:
            self._open_mindfulness_final_dialog()

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

    def _derive_internal_rating(self, output: np.ndarray, corrected_display_levels: list[str | None], display_known_mask: list[bool]) -> int:
        predicted_display_indices = [
            int(self._display_state_from_probabilities(np.asarray(head, dtype=np.float32))["index"])
            for head in np.asarray(output, dtype=np.float32)
        ]
        head_distance = max(1, len(DISPLAY_AFFECT_LEVELS) - 1)
        agreement_scores: list[float] = []
        for index, predicted in enumerate(predicted_display_indices):
            if index >= len(display_known_mask) or not display_known_mask[index]:
                continue
            corrected = corrected_display_levels[index]
            if corrected is None:
                continue
            corrected_index = display_level_index(corrected)
            if corrected_index is None:
                continue
            distance = abs(predicted - int(corrected_index))
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

    def _pomodoro_profile_from_checkin_levels(
        self,
        corrected_display_levels: list[str | None],
        display_known_mask: list[bool],
    ) -> AffectProfile:
        labels: dict[str, str | None] = {
            "engagement": None,
            "boredom": None,
            "confusion": None,
            "frustration": None,
        }
        for spec in self.head_specs:
            key = _normalize_head_name(spec["name"])
            if key not in labels:
                continue
            if spec["index"] >= len(corrected_display_levels) or not display_known_mask[spec["index"]]:
                continue
            labels[key] = corrected_display_levels[spec["index"]]
        return AffectProfile(
            state=self.latest_affect_profile.state,
            engagement_label=labels["engagement"],
            boredom_label=labels["boredom"],
            confusion_label=labels["confusion"],
            frustration_label=labels["frustration"],
        )

    def _advance_pomodoro_after_checkin(
        self,
        status_reason: str,
        *,
        block_profile: AffectProfile | None = None,
    ) -> None:
        self.pomodoro_prompt_pending = False
        self.pomodoro_paused = False
        if block_profile is not None:
            self.pomodoro_recent_block_profiles.append(block_profile)
        self.pomodoro_completed_blocks += 1
        if self.pomodoro_completed_blocks >= 3:
            self._begin_pomodoro_final_reflection()
            return

        self.pomodoro_current_block_index = self.pomodoro_completed_blocks
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_last_tick_monotonic = time.monotonic()
        self.pomodoro_phase = "running"
        self.pomodoro_status_reason = status_reason
        self._reset_pomodoro_block_capture(time.time())
        self.pomodoro_recent_affect_profiles.clear()
        self._update_pomodoro_practice()
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

    def _close_pomodoro_final_dialog(self) -> None:
        if self.pomodoro_final_dialog is None:
            return
        try:
            self.pomodoro_final_dialog.grab_release()
        except tk.TclError:
            pass
        try:
            self.pomodoro_final_dialog.destroy()
        except tk.TclError:
            pass
        self.pomodoro_final_dialog = None
        self._sync_control_states()

    def _begin_pomodoro_final_reflection(self) -> None:
        self.pomodoro_active = False
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_final_prompt_pending = True
        self.pomodoro_phase = "reflect"
        self.pomodoro_status_reason = "The 24-minute Pomodoro just ended. Add one quick overall reflection or skip it."
        self.pomodoro_remaining_seconds = 0.0
        self.pomodoro_block_elapsed_seconds = float(POMODORO_BLOCK_SECONDS)
        self.pomodoro_current_block_index = 2
        self.pomodoro_last_tick_monotonic = None
        self._reset_pomodoro_block_capture()
        self._refresh_pomodoro_ui()
        self._sync_control_states()
        self._open_pending_reflection_dialogs()

    def _finish_pomodoro_final_reflection(self, status_reason: str) -> None:
        self.pomodoro_final_prompt_pending = False
        self.pomodoro_phase = "complete"
        self.pomodoro_status_reason = status_reason
        self.pomodoro_remaining_seconds = 0.0
        self.pomodoro_block_elapsed_seconds = float(POMODORO_BLOCK_SECONDS)
        self._refresh_pomodoro_ui()
        self._sync_control_states()

    def _skip_pomodoro_final_reflection(self) -> None:
        self._close_pomodoro_final_dialog()
        self.feedback_status_var.set("Latest: Pomodoro final reflection skipped.")
        self._finish_pomodoro_final_reflection("Pomodoro complete. Final reflection skipped.")

    def _submit_pomodoro_final_reflection(self, rating: int, outcome_tag: str) -> None:
        resolved_rating = int(rating)
        resolved_outcome = str(outcome_tag).strip()
        summary = (
            f"Overall Pomodoro reflection after {POMODORO_TOTAL_MINUTES} minutes: "
            f"{resolved_outcome} (rating {resolved_rating}/5)."
        )
        self.feedback_manager.log_session_experience(
            mode="pomodoro",
            feedback_source=POMODORO_FINAL_FEEDBACK_SOURCE,
            rating=resolved_rating,
            outcome_tag=resolved_outcome,
            summary=summary,
            window_start_epoch=self.pomodoro_session_start_epoch,
            window_end_epoch=time.time(),
            practice_id=self.pomodoro_current_practice_id,
            completed_blocks=self.pomodoro_completed_blocks,
        )
        self._close_pomodoro_final_dialog()
        self.feedback_status_var.set("Latest: Pomodoro final reflection saved.")
        self._finish_pomodoro_final_reflection(
            f"Pomodoro complete. Final reflection saved: {resolved_outcome}."
        )

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
        corrected_display_levels: list[str | None] = [None] * self.head_count
        display_known_mask: list[bool] = [False] * self.head_count
        for spec in self.pomodoro_specs:
            choice = selections[spec["index"]].get()
            if choice == "Not sure":
                corrected_display_levels[spec["index"]] = None
                display_known_mask[spec["index"]] = False
            else:
                corrected_display_levels[spec["index"]] = choice
                display_known_mask[spec["index"]] = True
        block_profile = self._pomodoro_profile_from_checkin_levels(
            corrected_display_levels,
            display_known_mask,
        )

        if not self.pomodoro_block_outputs:
            self._close_checkin_dialog()
            self.feedback_status_var.set("Latest: Not enough live data was captured for that self-check window.")
            self._advance_pomodoro_after_checkin(
                f"Block {block_number} had insufficient live data.",
                block_profile=block_profile,
            )
            return

        aggregated_output = np.stack([sample for _, sample in self.pomodoro_block_outputs], axis=0).mean(axis=0).astype(np.float32)
        derived_rating = self._derive_internal_rating(aggregated_output, corrected_display_levels, display_known_mask)
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
            self._advance_pomodoro_after_checkin(
                f"Block {block_number} could not be saved.",
                block_profile=block_profile,
            )
            return

        self.feedback_manager.submit_feedback(
            snapshot,
            rating=derived_rating,
            corrected_display_levels=corrected_display_levels,
            display_known_mask=display_known_mask,
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
        self._advance_pomodoro_after_checkin(status_reason, block_profile=block_profile)

    def _open_pomodoro_checkin(self) -> None:
        if self.checkin_dialog is not None:
            self.checkin_dialog.lift()
            self.checkin_dialog.focus_force()
            return
        if (
            self.mindfulness_checkin_dialog is not None
            or self.pomodoro_final_dialog is not None
            or self.mindfulness_final_dialog is not None
        ):
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

        choice_values = [*DISPLAY_AFFECT_LEVELS, "Not sure"]
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

    def _open_pomodoro_final_dialog(self) -> None:
        if self.pomodoro_final_dialog is not None:
            self.pomodoro_final_dialog.lift()
            self.pomodoro_final_dialog.focus_force()
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Pomodoro Reflection")
        dialog.configure(bg=COLORS["panel"])
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        self.pomodoro_final_dialog = dialog
        self._sync_control_states()

        container = tk.Frame(dialog, bg=COLORS["panel"])
        container.pack(fill="both", expand=True, padx=22, pady=22)

        tk.Label(container, text="Pomodoro Wrap-Up", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18)).pack(anchor="w")
        tk.Label(
            container,
            text="How was the full 24-minute Pomodoro overall? This stays analytics-only and does not enter model training.",
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            justify="left",
            wraplength=560,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 16))

        rating_var = tk.StringVar(value="3")
        outcome_var = tk.StringVar(value="No clear shift")

        rating_row = self._create_card(container, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        rating_row.pack(fill="x", pady=(0, 10))
        tk.Label(
            rating_row,
            text="Overall rating",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 8))
        rating_buttons = tk.Frame(rating_row, bg=COLORS["panel_alt"])
        rating_buttons.pack(fill="x", padx=12, pady=(0, 14))
        for column, (value, label) in enumerate(FINAL_EXPERIENCE_RATING_OPTIONS):
            rating_buttons.grid_columnconfigure(column, weight=1)
            tk.Radiobutton(
                rating_buttons,
                text=f"{value}\n{label}",
                variable=rating_var,
                value=value,
                indicatoron=False,
                bg=COLORS["panel_soft"],
                fg=COLORS["text"],
                activebackground=mix_color(COLORS["blue_soft"], "#ffffff", 0.06),
                activeforeground=COLORS["text"],
                selectcolor=COLORS["blue_soft"],
                relief="flat",
                bd=0,
                padx=10,
                pady=10,
                font=("Segoe UI Semibold", 9),
                highlightthickness=0,
                cursor="hand2",
                justify="center",
            ).grid(row=0, column=column, sticky="ew", padx=4)

        outcome_row = self._create_card(container, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        outcome_row.pack(fill="x")
        tk.Label(
            outcome_row,
            text="What shifted most?",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 8))
        outcome_buttons = tk.Frame(outcome_row, bg=COLORS["panel_alt"])
        outcome_buttons.pack(fill="x", padx=12, pady=(0, 14))
        for column, label in enumerate(POMODORO_FINAL_OUTCOME_TAGS):
            outcome_buttons.grid_columnconfigure(column, weight=1)
            tk.Radiobutton(
                outcome_buttons,
                text=label,
                variable=outcome_var,
                value=label,
                indicatoron=False,
                bg=COLORS["panel_soft"],
                fg=COLORS["text"],
                activebackground=mix_color(COLORS["blue_soft"], "#ffffff", 0.06),
                activeforeground=COLORS["text"],
                selectcolor=COLORS["blue_soft"],
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
            command=lambda: self._submit_pomodoro_final_reflection(int(rating_var.get()), outcome_var.get()),
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
            command=self._skip_pomodoro_final_reflection,
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

        dialog.protocol("WM_DELETE_WINDOW", self._skip_pomodoro_final_reflection)

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

    def _close_mindfulness_checkin_dialog(self) -> None:
        if self.mindfulness_checkin_dialog is None:
            return
        try:
            self.mindfulness_checkin_dialog.grab_release()
        except tk.TclError:
            pass
        try:
            self.mindfulness_checkin_dialog.destroy()
        except tk.TclError:
            pass
        self.mindfulness_checkin_dialog = None
        self._sync_control_states()

    def _close_mindfulness_final_dialog(self) -> None:
        if self.mindfulness_final_dialog is None:
            return
        try:
            self.mindfulness_final_dialog.grab_release()
        except tk.TclError:
            pass
        try:
            self.mindfulness_final_dialog.destroy()
        except tk.TclError:
            pass
        self.mindfulness_final_dialog = None
        self._sync_control_states()

    def _advance_mindfulness_after_checkin(
        self,
        status_reason: str,
        selection: MindfulnessPracticeSelection | None = None,
    ) -> None:
        if selection is None:
            selection = select_mindfulness_practice(
                self.latest_affect_profile,
                elapsed_seconds=self.mindfulness_elapsed_seconds,
            )
        self.mindfulness_segment_practice_id = selection.practice_id
        self.mindfulness_segment_why_selected = selection.why_selected
        self.mindfulness_segment_source = selection.steering_source
        self.mindfulness_prompt_pending = False
        self.mindfulness_paused = False
        self.mindfulness_completed_checkins += 1
        if self.mindfulness_completed_checkins < len(self.mindfulness_checkin_boundaries):
            self.mindfulness_next_checkin_elapsed_seconds = float(
                self.mindfulness_checkin_boundaries[self.mindfulness_completed_checkins]
            )
        else:
            self.mindfulness_next_checkin_elapsed_seconds = None
        self.mindfulness_phase = "running"
        self.mindfulness_status_reason = status_reason
        self.mindfulness_last_tick_monotonic = time.monotonic() if self.mindfulness_active else None
        self._refresh_mindfulness_ui()
        self._sync_control_states()

    def _skip_mindfulness_checkin(self) -> None:
        self._close_mindfulness_checkin_dialog()
        self.feedback_status_var.set("Latest: Mindfulness steer-in skipped.")
        self._advance_mindfulness_after_checkin(
            "Mindfulness steer-in skipped. The next segment will follow the live practice selector."
        )

    def _submit_mindfulness_checkin(self, choice_var: tk.StringVar) -> None:
        steering_key = choice_var.get().strip()
        if not steering_key:
            return
        selection = select_mindfulness_practice(
            self.latest_affect_profile,
            elapsed_seconds=self.mindfulness_elapsed_seconds,
            steering_key=steering_key,
            recent_steering_keys=tuple(self.mindfulness_recent_steering_keys),
        )
        self.mindfulness_recent_steering_keys.append(steering_key)
        self._close_mindfulness_checkin_dialog()
        self.feedback_status_var.set(f"Latest: Mindfulness switched to {selection.practice_label.lower()}.")
        self._advance_mindfulness_after_checkin(
            f"Mindfulness switched to {selection.practice_label.lower()} for the next segment.",
            selection=selection,
        )

    def _begin_mindfulness_final_reflection(self) -> None:
        self.mindfulness_active = False
        self.mindfulness_paused = False
        self.mindfulness_prompt_pending = False
        self.mindfulness_final_prompt_pending = True
        self.mindfulness_phase = "reflect"
        self.mindfulness_remaining_seconds = 0.0
        self.mindfulness_elapsed_seconds = float(MINDFULNESS_TOTAL_SECONDS)
        self.mindfulness_last_tick_monotonic = None
        self.mindfulness_status_reason = "The 8-minute mindfulness reset just ended. Add one quick overall reflection or skip it."
        self._refresh_mindfulness_ui()
        self._sync_control_states()
        self._open_pending_reflection_dialogs()

    def _finish_mindfulness_final_reflection(self, status_reason: str) -> None:
        self.mindfulness_final_prompt_pending = False
        self.mindfulness_phase = "complete"
        self.mindfulness_status_reason = status_reason
        self.mindfulness_remaining_seconds = 0.0
        self.mindfulness_elapsed_seconds = float(MINDFULNESS_TOTAL_SECONDS)
        self._refresh_mindfulness_ui()
        self._sync_control_states()

    def _skip_mindfulness_final_reflection(self) -> None:
        self._close_mindfulness_final_dialog()
        self.feedback_status_var.set("Latest: Mindfulness final reflection skipped.")
        self._finish_mindfulness_final_reflection("Mindfulness reset complete. Final reflection skipped.")

    def _submit_mindfulness_final_reflection(self, rating: int, outcome_tag: str) -> None:
        resolved_rating = int(rating)
        resolved_outcome = str(outcome_tag).strip()
        summary = (
            f"Overall mindfulness reflection after {MINDFULNESS_TOTAL_MINUTES} minutes: "
            f"{resolved_outcome} (rating {resolved_rating}/5)."
        )
        self.feedback_manager.log_session_experience(
            mode="mindfulness",
            feedback_source=MINDFULNESS_FINAL_FEEDBACK_SOURCE,
            rating=resolved_rating,
            outcome_tag=resolved_outcome,
            summary=summary,
            window_start_epoch=self.mindfulness_session_start_epoch,
            window_end_epoch=time.time(),
            practice_id=self.mindfulness_segment_practice_id,
            completed_checkins=self.mindfulness_completed_checkins,
        )
        self._close_mindfulness_final_dialog()
        self.feedback_status_var.set("Latest: Mindfulness final reflection saved.")
        self._finish_mindfulness_final_reflection(
            f"Mindfulness reset complete. Final reflection saved: {resolved_outcome}."
        )

    def _open_mindfulness_checkin(self) -> None:
        if self.mindfulness_checkin_dialog is not None:
            self.mindfulness_checkin_dialog.lift()
            self.mindfulness_checkin_dialog.focus_force()
            return
        if (
            self.checkin_dialog is not None
            or self.pomodoro_final_dialog is not None
            or self.mindfulness_final_dialog is not None
        ):
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Mindfulness Steer-In")
        dialog.configure(bg=COLORS["panel"])
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        self.mindfulness_checkin_dialog = dialog
        self._sync_control_states()

        container = tk.Frame(dialog, bg=COLORS["panel"])
        container.pack(fill="both", expand=True, padx=22, pady=22)

        tk.Label(container, text="1.8-Minute Steer-In", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18)).pack(anchor="w")
        tk.Label(
            container,
            text=(
                "How do you feel right now? "
                f"Your answer steers the next {format_clock(MINDFULNESS_CHECKIN_INTERVAL_SECONDS)} of practice."
            ),
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            justify="left",
            wraplength=560,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 16))

        default_key = mindfulness_steering_key_for_profile(self.latest_affect_profile)
        choice_var = tk.StringVar(value=default_key)
        description_var = tk.StringVar(value=mindfulness_steering_option(default_key).description)

        def _update_description(*_args) -> None:
            description_var.set(mindfulness_steering_option(choice_var.get()).description)

        choice_var.trace_add("write", _update_description)

        button_row = tk.Frame(container, bg=COLORS["panel"])
        button_row.pack(fill="x", pady=(0, 10))
        for column, option in enumerate(MINDFULNESS_STEERING_OPTIONS):
            button_row.grid_columnconfigure(column, weight=1, uniform="mindfulness_option")
            tk.Radiobutton(
                button_row,
                text=option.label,
                variable=choice_var,
                value=option.key,
                indicatoron=False,
                bg=COLORS["panel_soft"],
                fg=COLORS["text"],
                activebackground=mix_color(COLORS["green_soft"], "#ffffff", 0.08),
                activeforeground=COLORS["text"],
                selectcolor=COLORS["green_soft"],
                relief="flat",
                bd=0,
                padx=12,
                pady=12,
                font=("Segoe UI Semibold", 9),
                highlightthickness=0,
                cursor="hand2",
            ).grid(row=0, column=column, sticky="ew", padx=4)

        tk.Label(
            container,
            textvariable=description_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            justify="left",
            wraplength=560,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 16))

        action_row = tk.Frame(container, bg=COLORS["panel"])
        action_row.pack(fill="x")
        tk.Button(
            action_row,
            text="Steer Practice",
            command=lambda: self._submit_mindfulness_checkin(choice_var),
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
            command=self._skip_mindfulness_checkin,
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
            text="Stop Mindfulness",
            command=self.stop_mindfulness,
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

        dialog.protocol("WM_DELETE_WINDOW", self._skip_mindfulness_checkin)

    def _open_mindfulness_final_dialog(self) -> None:
        if self.mindfulness_final_dialog is not None:
            self.mindfulness_final_dialog.lift()
            self.mindfulness_final_dialog.focus_force()
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Mindfulness Reflection")
        dialog.configure(bg=COLORS["panel"])
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        self.mindfulness_final_dialog = dialog
        self._sync_control_states()

        container = tk.Frame(dialog, bg=COLORS["panel"])
        container.pack(fill="both", expand=True, padx=22, pady=22)

        tk.Label(container, text="Mindfulness Wrap-Up", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18)).pack(anchor="w")
        tk.Label(
            container,
            text="How did the full 8-minute reset feel overall? This stays analytics-only and does not enter model training.",
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            justify="left",
            wraplength=560,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 16))

        rating_var = tk.StringVar(value="3")
        outcome_var = tk.StringVar(value="No clear shift")

        rating_row = self._create_card(container, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        rating_row.pack(fill="x", pady=(0, 10))
        tk.Label(
            rating_row,
            text="Overall rating",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 8))
        rating_buttons = tk.Frame(rating_row, bg=COLORS["panel_alt"])
        rating_buttons.pack(fill="x", padx=12, pady=(0, 14))
        for column, (value, label) in enumerate(FINAL_EXPERIENCE_RATING_OPTIONS):
            rating_buttons.grid_columnconfigure(column, weight=1)
            tk.Radiobutton(
                rating_buttons,
                text=f"{value}\n{label}",
                variable=rating_var,
                value=value,
                indicatoron=False,
                bg=COLORS["panel_soft"],
                fg=COLORS["text"],
                activebackground=mix_color(COLORS["green_soft"], "#ffffff", 0.06),
                activeforeground=COLORS["text"],
                selectcolor=COLORS["green_soft"],
                relief="flat",
                bd=0,
                padx=10,
                pady=10,
                font=("Segoe UI Semibold", 9),
                highlightthickness=0,
                cursor="hand2",
                justify="center",
            ).grid(row=0, column=column, sticky="ew", padx=4)

        outcome_row = self._create_card(container, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        outcome_row.pack(fill="x")
        tk.Label(
            outcome_row,
            text="What shifted most?",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 8))
        outcome_buttons = tk.Frame(outcome_row, bg=COLORS["panel_alt"])
        outcome_buttons.pack(fill="x", padx=12, pady=(0, 14))
        for column, label in enumerate(MINDFULNESS_FINAL_OUTCOME_TAGS):
            outcome_buttons.grid_columnconfigure(column, weight=1)
            tk.Radiobutton(
                outcome_buttons,
                text=label,
                variable=outcome_var,
                value=label,
                indicatoron=False,
                bg=COLORS["panel_soft"],
                fg=COLORS["text"],
                activebackground=mix_color(COLORS["green_soft"], "#ffffff", 0.06),
                activeforeground=COLORS["text"],
                selectcolor=COLORS["green_soft"],
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
            command=lambda: self._submit_mindfulness_final_reflection(int(rating_var.get()), outcome_var.get()),
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
            command=self._skip_mindfulness_final_reflection,
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

        dialog.protocol("WM_DELETE_WINDOW", self._skip_mindfulness_final_reflection)

    def _update_mindfulness_timer(self) -> None:
        if not self.mindfulness_active:
            return
        if self.mindfulness_paused:
            self.mindfulness_phase = "paused"
            if self.mindfulness_prompt_pending and self.mindfulness_checkin_dialog is None:
                self._open_mindfulness_checkin()
            self._refresh_mindfulness_ui()
            return

        now = time.monotonic()
        if self.mindfulness_last_tick_monotonic is None:
            self.mindfulness_last_tick_monotonic = now
        delta = max(0.0, now - self.mindfulness_last_tick_monotonic)
        self.mindfulness_last_tick_monotonic = now
        if delta <= 0.0:
            self._refresh_mindfulness_ui()
            return

        self.mindfulness_remaining_seconds = max(0.0, self.mindfulness_remaining_seconds - delta)
        self.mindfulness_elapsed_seconds = min(float(MINDFULNESS_TOTAL_SECONDS), self.mindfulness_elapsed_seconds + delta)
        self.mindfulness_phase = "running"
        if self.mindfulness_elapsed_seconds >= float(MINDFULNESS_TOTAL_SECONDS):
            self._begin_mindfulness_final_reflection()
            return
        elif (
            self.mindfulness_next_checkin_elapsed_seconds is not None
            and self.mindfulness_elapsed_seconds >= self.mindfulness_next_checkin_elapsed_seconds
            and not self.mindfulness_prompt_pending
        ):
            self.mindfulness_paused = True
            self.mindfulness_prompt_pending = True
            self.mindfulness_phase = "paused"
            self.mindfulness_last_tick_monotonic = None
            self.mindfulness_status_reason = "Mindfulness steer-in due. Choose how you feel to guide the next segment."
            self._refresh_mindfulness_ui()
            self._open_mindfulness_checkin()
            return
        self._refresh_mindfulness_ui()

    def start_pomodoro(self) -> None:
        if not self.pomodoro_supported:
            self.feedback_status_var.set("Latest: Pomodoro check-ins need the multi-affect model.")
            self._refresh_pomodoro_ui()
            return
        if self.pomodoro_active or self.checkin_dialog is not None or self.pomodoro_final_prompt_pending or self.pomodoro_final_dialog is not None:
            return
        if not self.running:
            self.feedback_status_var.set("Latest: Start the camera before starting Pomodoro.")
            self._sync_control_states()
            return

        self.pomodoro_active = True
        self.pomodoro_paused = False
        self.pomodoro_prompt_pending = False
        self.pomodoro_final_prompt_pending = False
        self.pomodoro_phase = "running"
        self.pomodoro_status_reason = ""
        self.pomodoro_completed_blocks = 0
        self.pomodoro_current_block_index = 0
        self.pomodoro_remaining_seconds = float(POMODORO_TOTAL_SECONDS)
        self.pomodoro_block_elapsed_seconds = 0.0
        self.pomodoro_session_start_epoch = time.time()
        self.pomodoro_last_tick_monotonic = time.monotonic()
        self.pomodoro_recent_affect_profiles.clear()
        self.pomodoro_recent_block_profiles.clear()
        self.pomodoro_current_practice_id = None
        self.pomodoro_current_why_selected = ""
        self.pomodoro_current_stability_label = ""
        self.pomodoro_current_stability_reason = ""
        self.pomodoro_last_switch_monotonic = None
        if self.latest_affect_profile.state in {"live_engaged", "live_not_engaged", "live_mixed"}:
            self.pomodoro_recent_affect_profiles.append(self.latest_affect_profile)
        self._update_pomodoro_practice(force=True)
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
        self._sync_control_states()

    def start_mindfulness(self) -> None:
        if (
            self.mindfulness_active
            or self.mindfulness_checkin_dialog is not None
            or self.mindfulness_prompt_pending
            or self.mindfulness_final_prompt_pending
            or self.mindfulness_final_dialog is not None
        ):
            return
        if not self.running:
            self.feedback_status_var.set("Latest: Start the camera before starting mindfulness.")
            self._sync_control_states()
            return
        self.mindfulness_active = True
        self.mindfulness_paused = False
        self.mindfulness_prompt_pending = False
        self.mindfulness_final_prompt_pending = False
        self.mindfulness_phase = "running"
        self.mindfulness_remaining_seconds = float(MINDFULNESS_TOTAL_SECONDS)
        self.mindfulness_elapsed_seconds = 0.0
        self.mindfulness_last_tick_monotonic = time.monotonic()
        self.mindfulness_session_start_epoch = time.time()
        self.mindfulness_status_reason = ""
        self.mindfulness_completed_checkins = 0
        self.mindfulness_next_checkin_elapsed_seconds = (
            float(self.mindfulness_checkin_boundaries[0]) if self.mindfulness_checkin_boundaries else None
        )
        self.mindfulness_recent_steering_keys = []
        initial_selection = select_mindfulness_practice(
            self.latest_affect_profile,
            elapsed_seconds=self.mindfulness_elapsed_seconds,
        )
        self.mindfulness_segment_practice_id = initial_selection.practice_id
        self.mindfulness_segment_why_selected = initial_selection.why_selected
        self.mindfulness_segment_source = initial_selection.steering_source
        self.feedback_status_var.set(
            f"Latest: Mindfulness started. The first steer-in opens in {format_clock(MINDFULNESS_CHECKIN_INTERVAL_SECONDS)}."
        )
        self._refresh_mindfulness_ui()
        self._sync_control_states()

    def stop_mindfulness(self) -> None:
        if not (self.mindfulness_active or self.mindfulness_checkin_dialog is not None or self.mindfulness_prompt_pending):
            return
        self._close_mindfulness_checkin_dialog()
        self.mindfulness_active = False
        self.mindfulness_paused = False
        self.mindfulness_prompt_pending = False
        self.mindfulness_phase = "stopped"
        self.mindfulness_last_tick_monotonic = None
        self.mindfulness_status_reason = "Mindfulness timer stopped before the full 8-minute reset."
        self.feedback_status_var.set("Latest: Mindfulness stopped before completion.")
        self._refresh_mindfulness_ui()
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
        width = min(usable_w, max(1360, int(usable_w * 0.78)))
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
        size = (int(event.width), int(event.height))
        if size == self.last_scroll_canvas_size:
            return
        self.last_scroll_canvas_size = size
        self.scroll_canvas.itemconfigure(self.scroll_canvas_window, width=event.width)
        self._schedule_responsive_layout()

    def _on_content_frame_configure(self, _event) -> None:
        if self.scroll_canvas is None:
            return
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if self.scroll_canvas is None:
            return
        if any(dialog is not None for dialog in self._modal_dialogs()):
            widget = getattr(event, "widget", None)
            if widget is not None:
                try:
                    dialog = widget.winfo_toplevel()
                    if dialog in self._modal_dialogs():
                        return
                except tk.TclError:
                    return

        scroll_region = self.scroll_canvas.bbox("all")
        if not scroll_region:
            return
        if (scroll_region[3] - scroll_region[1]) <= self.scroll_canvas.winfo_height():
            return

        delta_pixels = 0.0
        delta = getattr(event, "delta", 0)
        if delta:
            delta_pixels = -(float(delta) / 120.0) * SCROLL_NOTCH_PIXELS
        else:
            event_num = getattr(event, "num", None)
            if event_num == 4:
                delta_pixels = -SCROLL_NOTCH_PIXELS
            elif event_num == 5:
                delta_pixels = SCROLL_NOTCH_PIXELS

        if abs(delta_pixels) < 0.1:
            return
        self._queue_smooth_scroll(delta_pixels)
        return "break"

    def _queue_smooth_scroll(self, delta_pixels: float) -> None:
        if self.scroll_canvas is None:
            return
        current_offset, max_offset = self._scroll_offsets()
        if current_offset is None or max_offset <= 0.0:
            return
        if self.scroll_animation_after_id is None:
            self.scroll_target_offset = current_offset
        self.scroll_target_offset = max(0.0, min(max_offset, self.scroll_target_offset + float(delta_pixels)))
        self._mark_ui_interaction()
        if self.scroll_animation_after_id is None:
            self.scroll_animation_after_id = self.root.after(SCROLL_FRAME_MS, self._animate_scroll)

    def _scroll_offsets(self) -> tuple[float | None, float]:
        if self.scroll_canvas is None:
            return None, 0.0
        scroll_region = self.scroll_canvas.bbox("all")
        if not scroll_region:
            return None, 0.0

        viewport_height = max(1.0, float(self.scroll_canvas.winfo_height()))
        content_height = max(viewport_height, float(scroll_region[3] - scroll_region[1]))
        max_offset = max(0.0, content_height - viewport_height)
        if max_offset <= 0.0:
            return 0.0, 0.0

        top_fraction = float(self.scroll_canvas.yview()[0])
        current_offset = top_fraction * max_offset
        return current_offset, max_offset

    def _scroll_canvas_by_pixels(self, delta_pixels: float) -> bool:
        current_offset, max_offset = self._scroll_offsets()
        if current_offset is None or max_offset <= 0.0:
            return False
        next_offset = max(0.0, min(max_offset, current_offset + float(delta_pixels)))
        self.scroll_canvas.yview_moveto(next_offset / max_offset)
        return True

    def _animate_scroll(self) -> None:
        self.scroll_animation_after_id = None
        if self.scroll_canvas is None:
            self.scroll_pending_pixels = 0.0
            self.scroll_target_offset = 0.0
            return

        current_offset, max_offset = self._scroll_offsets()
        if current_offset is None or max_offset <= 0.0:
            self.scroll_pending_pixels = 0.0
            self.scroll_target_offset = 0.0
            return

        self.scroll_target_offset = max(0.0, min(max_offset, self.scroll_target_offset))
        remaining = self.scroll_target_offset - current_offset
        if abs(remaining) < 0.6:
            if max_offset > 0.0:
                self.scroll_canvas.yview_moveto(self.scroll_target_offset / max_offset)
            self.scroll_pending_pixels = 0.0
            self.scroll_target_offset = current_offset if max_offset <= 0.0 else self.scroll_target_offset
            return

        step = remaining * SCROLL_EASING
        step = max(-SCROLL_NOTCH_PIXELS, min(SCROLL_NOTCH_PIXELS, step))
        if abs(step) < SCROLL_MIN_STEP_PIXELS:
            step = SCROLL_MIN_STEP_PIXELS if remaining > 0 else -SCROLL_MIN_STEP_PIXELS

        before = self.scroll_canvas.yview()
        moved = self._scroll_canvas_by_pixels(step)
        after = self.scroll_canvas.yview()
        if not moved or before == after:
            self.scroll_pending_pixels = 0.0
            self.scroll_target_offset = current_offset
            return
        self.scroll_pending_pixels = self.scroll_target_offset - (self._scroll_offsets()[0] or 0.0)
        self.scroll_animation_after_id = self.root.after(SCROLL_FRAME_MS, self._animate_scroll)

    def _on_root_configure(self, event) -> None:
        if event.widget is self.root:
            size = (int(event.width), int(event.height))
            if size == self.last_root_size:
                return
            self.last_root_size = size
            self._schedule_responsive_layout()

    def _schedule_responsive_layout(self) -> None:
        if self.layout_after_id is not None:
            self.root.after_cancel(self.layout_after_id)
        self.layout_after_id = self.root.after(LAYOUT_DEBOUNCE_MS, self._apply_responsive_layout)

    def _mark_ui_interaction(self, duration: float = INTERACTION_COOLDOWN_SEC) -> None:
        self.interaction_busy_until = max(self.interaction_busy_until, time.monotonic() + max(0.0, duration))

    def _ui_interaction_active(self) -> bool:
        return time.monotonic() < self.interaction_busy_until

    def _responsive_flag(self, current: bool | None, width: int, breakpoint: int) -> bool:
        if current is None:
            return width < breakpoint
        lower = breakpoint - LAYOUT_HYSTERESIS
        upper = breakpoint + LAYOUT_HYSTERESIS
        if current:
            return width < upper
        return width < lower

    def _apply_responsive_layout(self) -> None:
        self.layout_after_id = None
        if self.content_frame is None:
            return

        viewport_width = 0
        if self.scroll_canvas is not None:
            viewport_width = int(self.scroll_canvas.winfo_width() or 0)
        content_width = max(1, viewport_width or self.root.winfo_width())
        main_stacked = self._responsive_flag(self.main_area_layout_stacked, content_width, MAIN_STACK_BREAKPOINT)
        timer_stacked = self._responsive_flag(self.timer_row_layout_stacked, content_width, TIMER_STACK_BREAKPOINT)
        tile_columns = max(1, min(len(self.signal_tile_order), content_width // TILE_MIN_WIDTH))

        self._layout_top_bar(False)
        self._layout_top_bar_chips(3)
        self._layout_main_area(main_stacked)
        self._layout_timer_cards(timer_stacked)
        self._layout_bottom_header(True)
        self._layout_signal_tiles(tile_columns)
        self._update_wraplengths(main_stacked)

    def _layout_top_bar(self, compact: bool) -> None:
        if self.top_bar_layout_compact is not None:
            self._layout_control_buttons(False)
            return
        self.top_bar_layout_compact = False
        self.title_frame.grid_forget()
        self.chip_frame.grid_forget()
        self.button_frame.grid_forget()
        self.top_bar.grid_columnconfigure(0, weight=1)
        self.top_bar.grid_columnconfigure(1, weight=0)
        self.top_bar.grid_columnconfigure(2, weight=0)
        self.title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        self.chip_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20, pady=(0, 8))
        self.button_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(0, 18))
        self._layout_control_buttons(False)

    def _layout_top_bar_chips(self, columns: int) -> None:
        if not hasattr(self, "top_bar_chips") or not self.top_bar_chips:
            return
        columns = max(1, min(3, len(self.top_bar_chips)))
        if self.top_chip_layout_columns is not None:
            return
        self.top_chip_layout_columns = columns

        for index in range(len(self.top_bar_chips)):
            self.chip_frame.grid_columnconfigure(index, weight=0, uniform="")
            self.chip_frame.grid_rowconfigure(index, weight=0)

        for chip in self.top_bar_chips:
            chip.grid_forget()

        for column in range(columns):
            self.chip_frame.grid_columnconfigure(column, weight=1)

        for index, chip in enumerate(self.top_bar_chips):
            row = index // columns
            column = index % columns
            chip.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0, 8 if column < columns - 1 else 0),
                pady=(0 if row == 0 else 6, 0),
            )

    def _layout_control_buttons(self, compact: bool) -> None:
        if self.control_buttons_compact is not None:
            return
        self.control_buttons_compact = False
        buttons = [
            self.start_button,
            self.stop_button,
            self.start_pomodoro_button,
            self.stop_pomodoro_button,
            self.start_mindfulness_button,
            self.stop_mindfulness_button,
        ]
        for button in buttons:
            button.grid_forget()

        for index in range(6):
            self.button_frame.grid_columnconfigure(index, weight=0, uniform="")
        for column in range(3):
            self.button_frame.grid_columnconfigure(column, weight=1, uniform="control_button", minsize=168)
        positions = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
        for button, (row, column) in zip(buttons, positions):
            button.grid(row=row, column=column, sticky="ew", padx=5, pady=5)

    def _layout_main_area(self, stacked: bool) -> None:
        if self.main_area_layout_stacked == stacked:
            return
        self.main_area_layout_stacked = stacked
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

        self.main_area.grid_columnconfigure(0, weight=7, minsize=CAMERA_PANEL_WIDTH)
        self.main_area.grid_columnconfigure(1, weight=3, minsize=ENGAGEMENT_PANEL_WIDTH)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(1, weight=0)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.decision_card.grid(row=0, column=1, sticky="nsew")

    def _layout_timer_cards(self, stacked: bool) -> None:
        if self.timer_row_layout_stacked == stacked:
            return
        self.timer_row_layout_stacked = stacked
        self.timer_cards_stacked = stacked
        self.pomodoro_card.grid_forget()
        self.mindfulness_card.grid_forget()
        self.timer_row.grid_columnconfigure(0, weight=0, uniform="")
        self.timer_row.grid_columnconfigure(1, weight=0, uniform="")
        self.timer_row.grid_rowconfigure(0, weight=0)
        self.timer_row.grid_rowconfigure(1, weight=0)
        if stacked:
            self.timer_row.grid_columnconfigure(0, weight=1, uniform="")
            self.pomodoro_card.grid(row=0, column=0, sticky="ew")
            self.mindfulness_card.grid(row=1, column=0, sticky="ew", pady=(12, 0))
            self.decision_body.grid_rowconfigure(3, minsize=0)
            return

        self.timer_row.grid_columnconfigure(0, weight=1, uniform="timer_card")
        self.timer_row.grid_columnconfigure(1, weight=1, uniform="timer_card")
        self.pomodoro_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.mindfulness_card.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.decision_body.grid_rowconfigure(3, minsize=0)

    def _layout_bottom_header(self, compact: bool) -> None:
        if self.bottom_header_layout_compact is not None:
            return
        self.bottom_header_layout_compact = True
        self.bottom_title_label.grid_forget()
        self.bottom_meta_label.grid_forget()
        self.bottom_header.grid_columnconfigure(0, weight=1)
        self.bottom_title_label.grid(row=0, column=0, sticky="w")
        self.bottom_meta_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _layout_signal_tiles(self, columns: int) -> None:
        if not self.signal_tile_order:
            return
        if self.signal_tile_layout_columns == columns:
            return
        self.signal_tile_layout_columns = columns

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
        tile_width = max(220, int((self.tiles_frame.winfo_width() or bottom_width) / max(1, self.signal_tile_layout_columns or len(self.signal_tile_order) or 1)))
        timer_width = max(220, decision_width - 72 if self.timer_cards_stacked else int((decision_width - 96) / 2))
        pomodoro_width = max(220, (self.pomodoro_card.winfo_width() or timer_width) - 32)
        mindfulness_width = max(220, (self.mindfulness_card.winfo_width() or timer_width) - 32)

        headline_wrap = max(260, decision_width - 72)
        headline_size = 30 if decision_width >= 560 else 28 if decision_width >= 500 else 26 if decision_width >= 440 else 24
        confidence_size = 13 if decision_width >= 520 else 12 if decision_width >= 420 else 11

        self.primary_headline_label.configure(wraplength=headline_wrap, font=("Bahnschrift SemiBold", headline_size))
        self.primary_confidence_label.configure(wraplength=headline_wrap, font=("Segoe UI Semibold", confidence_size))
        self.summary_label.configure(wraplength=max(240, decision_width - 72))
        self.spotlight_detail_label.configure(wraplength=max(240, decision_width - 72))
        self.spotlight_value.configure(font=("Bahnschrift SemiBold", 18 if decision_width >= 440 else 16))
        self.pomodoro_next_label.configure(wraplength=max(180, pomodoro_width - 12))
        self.pomodoro_note_label.configure(wraplength=max(180, pomodoro_width - 12))
        self.mindfulness_next_label.configure(wraplength=max(180, mindfulness_width - 12))
        self.mindfulness_note_label.configure(wraplength=max(180, mindfulness_width - 12))
        self.preview_footer.configure(wraplength=max(260, preview_width - 40))
        self.bottom_meta_label.configure(wraplength=max(240, bottom_width - 40))
        self.engagement_summary_note_label.configure(wraplength=max(260, bottom_width - 40))
        self.learning_suggestion_label.configure(wraplength=max(260, bottom_width - 40))
        self.feedback_info_label.configure(wraplength=max(260, bottom_width - 40))
        self.feedback_status_label.configure(wraplength=max(260, bottom_width - 40))
        for tile in self.signal_tiles.values():
            tile["detail"].configure(wraplength=max(180, tile_width - 36))
        target_height = PREVIEW_STAGE_COMPACT_HEIGHT if main_stacked else PREVIEW_STAGE_HEIGHT
        if int(self.preview_stage.cget("height")) != target_height:
            self.preview_stage.configure(height=target_height)

    def _preview_stage_target_height(self, main_stacked: bool) -> int:
        return PREVIEW_STAGE_COMPACT_HEIGHT if main_stacked else PREVIEW_STAGE_HEIGHT

    def _current_preview_size(self) -> tuple[int, int]:
        stage_width = max(320, self.preview_stage.winfo_width() if hasattr(self, "preview_stage") else 0)
        stage_height = max(240, self.preview_stage.winfo_height() if hasattr(self, "preview_stage") else 0)
        return int(stage_width), int(stage_height)

    def _schedule_preview_refresh(self, delay_ms: int | None = None) -> None:
        if self.closing or not hasattr(self, "preview_stage"):
            return
        if self.preview_refresh_after_id is not None:
            self.root.after_cancel(self.preview_refresh_after_id)
        wait_ms = PREVIEW_REFRESH_DEBOUNCE_MS if delay_ms is None else max(0, int(delay_ms))
        self.preview_refresh_after_id = self.root.after(wait_ms, self._refresh_preview_after_resize)

    def _refresh_preview_after_resize(self) -> None:
        self.preview_refresh_after_id = None
        self.preview_resize_busy_until = 0.0
        requested_size = self.preview_requested_size or self._current_preview_size()
        self.preview_requested_size = requested_size
        if self.last_frame is None:
            self.preview_last_size = requested_size
            self._draw_preview_placeholder()
            return
        self.preview_last_size = requested_size
        self.last_preview_update_monotonic = 0.0

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
            padx=10,
            pady=6,
            font=("Segoe UI Semibold", 9),
        )

    def _invoke_button_command(self, command) -> None:
        self._mark_ui_interaction()
        if self.closing:
            return
        self.root.after_idle(command)

    def _create_button(self, parent: tk.Misc, text: str, command, bg: str, fg: str = COLORS["text"]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=lambda cmd=command: self._invoke_button_command(cmd),
            bg=bg,
            fg=fg,
            activebackground=mix_color(bg, "#ffffff", 0.12),
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            width=16,
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
            f"{insight['review_count']} logged, "
            f"{insight['trusted_count']} trusted, "
            f"{insight['analytics_only_count']} analytics-only\n"
            f"Medium: {insight['inferred_medium_heads']} inferred ({insight['inferred_medium_rate'] * 100:.0f}%), "
            f"{insight['selected_medium_heads']} selected ({insight['selected_medium_rate'] * 100:.0f}%), "
            f"{insight['trainable_heads']} trainable ({insight['trainable_head_rate'] * 100:.0f}%), "
            f"{insight['steering_only_heads']} steering-only ({insight['steering_only_head_rate'] * 100:.0f}%), "
            f"Primary {insight['primary_threshold'] * 100:.0f}%, "
            f"Spotlight {insight['spotlight_threshold'] * 100:.0f}%"
        )
        self.feedback_status_var.set(f"Latest: {insight['last_feedback_status']}")

    def _create_stat_block(self, parent: tk.Misc, title: str) -> dict[str, Any]:
        frame = self._create_card(parent, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        frame.configure(height=STAT_BLOCK_HEIGHT)
        frame.grid_propagate(False)
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
            height=2,
            wraplength=220,
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
        self.chip_frame.grid(row=0, column=1, sticky="ew", padx=(0, 20), pady=18)
        self.status_chip = self._create_chip(self.chip_frame, self.status_var.get(), COLORS["panel_soft"])
        self.device_chip = self._create_chip(self.chip_frame, self.device_label, COLORS["panel_alt"])
        self.variant_chip = self._create_chip(self.chip_frame, self.model_variant.title(), COLORS["panel_alt"])
        self.head_count_chip = self._create_chip(self.chip_frame, f"{self.head_count} Head{'s' if self.head_count != 1 else ''}", COLORS["panel_alt"])
        self.sequence_chip = self._create_chip(self.chip_frame, f"Seq {self.seq_len}", COLORS["panel_alt"])
        self.top_bar_chips = [
            self.status_chip,
            self.device_chip,
            self.variant_chip,
            self.head_count_chip,
            self.sequence_chip,
        ]
        self._layout_top_bar_chips(3)

        self.button_frame = tk.Frame(self.top_bar, bg=COLORS["panel"])
        self.button_frame.grid(row=0, column=2, sticky="e", padx=20, pady=18)
        self.start_button = self._create_button(self.button_frame, "Start Camera", self.start, COLORS["green"])
        self.stop_button = self._create_button(self.button_frame, "Stop Camera", self.stop, COLORS["panel_alt"])
        self.start_pomodoro_button = self._create_button(self.button_frame, "Start Pomodoro", self.start_pomodoro, COLORS["blue_soft"], fg=COLORS["blue"])
        self.stop_pomodoro_button = self._create_button(self.button_frame, "Stop Pomodoro", self.stop_pomodoro, COLORS["panel_alt"])
        self.start_mindfulness_button = self._create_button(self.button_frame, "Start Mindfulness", self.start_mindfulness, COLORS["green_soft"], fg=COLORS["green"])
        self.stop_mindfulness_button = self._create_button(self.button_frame, "Stop Mindfulness", self.stop_mindfulness, COLORS["panel_alt"])
        self._layout_control_buttons(compact=False)
        self.stop_button.configure(state="disabled")
        self.start_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.stop_pomodoro_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.stop_mindfulness_button.configure(state="disabled", bg=COLORS["panel_soft"], fg=COLORS["text_soft"])

        self.main_area = tk.Frame(self.content_frame, bg=COLORS["bg"])
        self.main_area.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.main_area.grid_anchor("nw")
        self.main_area.grid_columnconfigure(0, weight=7, minsize=CAMERA_PANEL_WIDTH)
        self.main_area.grid_columnconfigure(1, weight=3, minsize=ENGAGEMENT_PANEL_WIDTH)
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
        self.preview_stage.bind("<Configure>", self._on_preview_configure)

        self.preview_footer = tk.Label(
            self.preview_card,
            textvariable=self.footer_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            anchor="w",
            height=2,
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

        self.primary_headline_label = tk.Label(
            self.primary_box,
            textvariable=self.prediction_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            justify="left",
            anchor="w",
            wraplength=460,
            font=("Bahnschrift SemiBold", 30),
        )
        self.primary_headline_label.grid(row=2, column=0, sticky="ew", padx=18)
        self.primary_confidence_label = tk.Label(
            self.primary_box,
            textvariable=self.confidence_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            justify="left",
            anchor="w",
            height=2,
            wraplength=460,
            font=("Segoe UI Semibold", 13),
        )
        self.primary_confidence_label.grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 6))
        self.summary_label = tk.Label(
            self.primary_box,
            textvariable=self.summary_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            height=3,
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
            height=2,
            wraplength=470,
            justify="left",
            font=("Segoe UI", 10),
        )
        self.spotlight_detail_label.pack(anchor="w", padx=16, pady=(4, 8))
        self.spotlight_meter = tk.Canvas(self.spotlight_card, height=16, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.spotlight_meter.pack(fill="x", padx=16, pady=(0, 6))
        self.spotlight_band = tk.Canvas(self.spotlight_card, height=42, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.spotlight_band.pack(fill="x", padx=16, pady=(0, 12))

        self.timer_row = tk.Frame(self.decision_body, bg=COLORS["panel"])
        self.timer_row.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        self.timer_row.grid_columnconfigure(0, weight=1)
        self.timer_row.grid_columnconfigure(1, weight=1)

        self.pomodoro_card = self._create_card(self.timer_row, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.pomodoro_card.grid_columnconfigure(0, weight=1)

        pomodoro_header = tk.Frame(self.pomodoro_card, bg=COLORS["panel_alt"])
        pomodoro_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 10))
        pomodoro_header.grid_columnconfigure(0, weight=1)
        self.pomodoro_title_label = tk.Label(
            pomodoro_header,
            text="FOCUS TIMER",
            bg=COLORS["panel_alt"],
            fg=COLORS["text_muted"],
            font=("Segoe UI Semibold", 9),
        )
        self.pomodoro_title_label.grid(row=0, column=0, sticky="w")
        self.pomodoro_chip = self._create_chip(pomodoro_header, "Idle", COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.pomodoro_chip.grid(row=0, column=1, sticky="e")

        self.pomodoro_time_label = tk.Label(
            self.pomodoro_card,
            textvariable=self.pomodoro_time_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Bahnschrift SemiBold", 28),
        )
        self.pomodoro_time_label.grid(row=1, column=0, sticky="w", padx=16)
        self.pomodoro_block_label = tk.Label(
            self.pomodoro_card,
            textvariable=self.pomodoro_block_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            font=("Segoe UI Semibold", 11),
        )
        self.pomodoro_block_label.grid(row=2, column=0, sticky="w", padx=16, pady=(2, 2))
        self.pomodoro_next_label = tk.Label(
            self.pomodoro_card,
            textvariable=self.pomodoro_next_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_muted"],
            justify="left",
            anchor="w",
            wraplength=470,
            font=("Segoe UI", 10),
        )
        self.pomodoro_next_label.grid(row=3, column=0, sticky="ew", padx=16)
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

        self.mindfulness_card = self._create_card(self.timer_row, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.mindfulness_card.grid_columnconfigure(0, weight=1)

        self.mindfulness_header = tk.Frame(self.mindfulness_card, bg=COLORS["panel_alt"])
        self.mindfulness_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 10))
        self.mindfulness_header.grid_columnconfigure(0, weight=1)
        self.mindfulness_title_label = tk.Label(
            self.mindfulness_header,
            text="MINDFULNESS TIMER",
            bg=COLORS["panel_alt"],
            fg=COLORS["text_muted"],
            font=("Segoe UI Semibold", 9),
        )
        self.mindfulness_title_label.grid(row=0, column=0, sticky="w")
        self.mindfulness_chip = self._create_chip(self.mindfulness_header, "Idle", COLORS["panel_soft"], fg=COLORS["text_soft"])
        self.mindfulness_chip.grid(row=0, column=1, sticky="e")

        self.mindfulness_time_label = tk.Label(
            self.mindfulness_card,
            textvariable=self.mindfulness_time_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=("Bahnschrift SemiBold", 28),
        )
        self.mindfulness_time_label.grid(row=1, column=0, sticky="w", padx=16)
        self.mindfulness_block_label = tk.Label(
            self.mindfulness_card,
            textvariable=self.mindfulness_block_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            font=("Segoe UI Semibold", 11),
        )
        self.mindfulness_block_label.grid(row=2, column=0, sticky="w", padx=16, pady=(2, 2))
        self.mindfulness_next_label = tk.Label(
            self.mindfulness_card,
            textvariable=self.mindfulness_next_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_muted"],
            justify="left",
            anchor="w",
            wraplength=470,
            font=("Segoe UI", 10),
        )
        self.mindfulness_next_label.grid(row=3, column=0, sticky="ew", padx=16)
        self.mindfulness_note_label = tk.Label(
            self.mindfulness_card,
            textvariable=self.mindfulness_note_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            justify="left",
            anchor="w",
            wraplength=470,
            font=("Segoe UI", 9),
        )
        self.mindfulness_note_label.grid(row=4, column=0, sticky="ew", padx=16, pady=(8, 8))
        self.mindfulness_progress_canvas = tk.Canvas(self.mindfulness_card, height=22, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.mindfulness_progress_canvas.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 14))
        self._layout_timer_cards(stacked=False)

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
            height=2,
            justify="left",
            font=("Segoe UI", 9),
        )
        self.engagement_summary_note_label.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 0))

        self.learning_suggestion_label = tk.Label(
            self.bottom_band,
            textvariable=self.learning_suggestion_var,
            bg=COLORS["panel"],
            fg=COLORS["green"],
            anchor="w",
            height=2,
            justify="left",
            font=("Segoe UI Semibold", 9),
        )
        self.learning_suggestion_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(8, 0))

        self.feedback_info_label = tk.Label(
            self.bottom_band,
            textvariable=self.feedback_insight_var,
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            anchor="w",
            height=3,
            justify="left",
            font=("Segoe UI", 9),
        )
        self.feedback_info_label.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 0))

        self.feedback_status_label = tk.Label(
            self.bottom_band,
            textvariable=self.feedback_status_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            anchor="w",
            height=1,
            justify="left",
            font=("Segoe UI", 9),
        )
        self.feedback_status_label.grid(row=5, column=0, sticky="ew", padx=20, pady=(4, 10))

        self.tiles_frame = tk.Frame(self.bottom_band, bg=COLORS["panel"])
        self.tiles_frame.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 14))
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
            tile.configure(height=SIGNAL_TILE_HEIGHT)
            tile.pack_propagate(False)
            tk.Label(tile, text=spec["label"], bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=14, pady=(14, 4))
            value = tk.Label(tile, text="50%", bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 24))
            value.pack(anchor="w", padx=14)
            detail = tk.Label(
                tile,
                text="Standby",
                bg=COLORS["panel_alt"],
                fg=COLORS["text_muted"],
                justify="left",
                anchor="w",
                height=2,
                wraplength=220,
                font=("Segoe UI", 9),
            )
            detail.pack(anchor="w", padx=14, pady=(2, 10))
            meter = tk.Canvas(tile, height=16, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
            meter.pack(fill="x", padx=14, pady=(0, 6))
            band = tk.Canvas(tile, height=40, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
            band.pack(fill="x", padx=14, pady=(0, 14))
            self.signal_tile_order.append(spec["key"])
            self.signal_tiles[spec["key"]] = {
                "frame": tile,
                "value": value,
                "detail": detail,
                "meter": meter,
                "band": band,
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

    def _draw_display_band(
        self,
        canvas: tk.Canvas,
        display_state: dict[str, Any] | None,
        accent: str,
        track: str,
    ) -> None:
        canvas.delete("all")
        width = max(120, canvas.winfo_width() or 240)
        height = max(40, canvas.winfo_height() or 40)
        label_y = 4
        band_top = 20
        band_bottom = max(band_top + 8, height - 10)
        segment_width = width / len(DISPLAY_AFFECT_LEVELS)
        muted_label = COLORS["text_muted"]
        active_index = None if display_state is None else int(display_state["index"])

        for index, label in enumerate(DISPLAY_LEVEL_SHORT_LABELS):
            x0 = index * segment_width
            x1 = x0 + segment_width
            fill = mix_color(track, accent, 0.08 if index == active_index else 0.03)
            canvas.create_rectangle(x0, band_top, x1, band_bottom, fill=fill, outline=fill)
            if index > 0:
                canvas.create_line(x0, band_top, x0, band_bottom, fill=COLORS["border_soft"], width=1)
            canvas.create_text(
                (x0 + x1) / 2,
                label_y,
                text=label,
                fill=accent if index == active_index else muted_label,
                font=("Segoe UI Semibold", 8),
                anchor="n",
            )

        canvas.create_rectangle(0, band_top, width, band_bottom, outline=COLORS["border_soft"], width=1)
        if display_state is None:
            return

        marker_x = max(4.0, min(width - 4.0, float(display_state["marker_position"]) * width))
        canvas.create_line(marker_x, band_top - 2, marker_x, band_bottom + 5, fill=accent, width=2)
        canvas.create_oval(
            marker_x - 4,
            band_bottom + 1,
            marker_x + 4,
            band_bottom + 9,
            fill=accent,
            outline=accent,
        )

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
            text=f"Engaged {engaged_score * 100:.0f}%   Not Engaged {not_engaged_score * 100:.0f}%",
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

    def _signal_for_key(
        self,
        output: np.ndarray,
        key: str,
        display_states: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
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
            "display_state": None if display_states is None else display_states.get(head_index),
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
        profile: AffectProfile | None = None,
    ) -> None:
        self.state = state
        self.status_var.set(STATE_STYLES[state]["badge"])
        self.preview_badge_var.set(preview_badge or STATE_STYLES[state]["badge"])
        self.prediction_var.set(headline)
        self.confidence_var.set(confidence)
        self.summary_var.set(summary)
        self.footer_var.set(footer)
        self._apply_state_palette(state)
        self.latest_affect_profile = self._fallback_affect_profile(state) if profile is None else profile
        self._refresh_learning_suggestion(self.latest_affect_profile)
        self._record_pomodoro_affect_profile(self.latest_affect_profile)
        if hasattr(self, "pomodoro_note_var"):
            self._refresh_pomodoro_ui()
        if hasattr(self, "mindfulness_note_var"):
            self._refresh_mindfulness_ui()

    def _binary_detail(self, display_state: dict[str, Any]) -> str:
        return f"Lead level: {display_state['label']}"

    def _spotlight_copy(
        self,
        signal: dict[str, Any] | None,
        spotlight_threshold: float | None = None,
    ) -> tuple[str, str, str, float, str, dict[str, Any] | None]:
        if spotlight_threshold is None:
            _, spotlight_threshold = self._effective_thresholds()
        if not self.secondary_specs:
            return (
                "Secondary Spotlight",
                "Engagement-only model active",
                "No secondary affect heads are available in this export.",
                0.0,
                COLORS["text_soft"],
                None,
            )
        signal_active = signal is not None and (signal.get("held") or signal["elevated"] >= spotlight_threshold)
        if not signal_active:
            return (
                "Secondary Spotlight",
                "Secondary signals stable",
                "No non-engagement head is above the promotion threshold.",
                0.0,
                COLORS["text_soft"],
                None,
            )
        assert signal is not None
        spec = signal["spec"]
        display_state = signal.get("display_state")
        level = "Stable" if display_state is None else str(display_state["label"])
        detail_prefix = "Holding trend" if signal.get("held") and signal["elevated"] < spotlight_threshold else "High band"
        detail = f"{detail_prefix} {signal['elevated'] * 100:.0f}%, Dominant level {level}"
        return "Secondary Spotlight", f"{spec['label']} rising", detail, float(signal["elevated"]), spec["accent"], display_state

    def _update_spotlight(self, signal: dict[str, Any] | None, spotlight_threshold: float | None = None) -> None:
        label, value, detail, meter_value, accent, display_state = self._spotlight_copy(signal, spotlight_threshold)
        self.spotlight_label_var.set(label)
        self.spotlight_value_var.set(value)
        self.spotlight_detail_var.set(detail)
        self.spotlight_value.configure(fg=accent if meter_value > 0 else COLORS["text"])
        self.spotlight_card.configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
        self._draw_capsule(self.spotlight_meter, meter_value, accent, COLORS["panel_soft"])
        self._draw_display_band(self.spotlight_band, display_state, accent, COLORS["panel_soft"])

    def _update_signal_tiles(
        self,
        output: np.ndarray,
        active_signal: dict[str, Any] | None,
        display_states: dict[int, dict[str, Any]] | None = None,
        spotlight_threshold: float | None = None,
    ) -> None:
        if spotlight_threshold is None:
            _, spotlight_threshold = self._effective_thresholds()
        if display_states is None:
            for tile in self.signal_tiles.values():
                tile["value"].configure(text="--")
                tile["detail"].configure(text="Waiting for live predictions")
                tile["frame"].configure(highlightbackground=COLORS["border_soft"])
                self._draw_capsule(tile["meter"], 0.0, tile["accent"], tile["surface"])
                self._draw_display_band(tile["band"], None, tile["accent"], tile["surface"])
            return

        engaged, not_engaged, dominant_level = self._engagement_scores(output)
        engagement_display_state = display_states.get(self.engagement_head_index)
        if engagement_display_state is None:
            engagement_display_state = self._display_state_from_probabilities(
                np.asarray(output[self.engagement_head_index], dtype=np.float32)
            )
        not_engagement_display_state = self._inverse_display_state(engagement_display_state)
        engaged_tile = self.signal_tiles.get("engaged")
        not_engaged_tile = self.signal_tiles.get("not_engaged")
        for tile, score, detail, display_state in (
            (engaged_tile, engaged, self._binary_detail(engagement_display_state), engagement_display_state),
            (not_engaged_tile, not_engaged, self._binary_detail(not_engagement_display_state), not_engagement_display_state),
        ):
            if tile is None:
                continue
            accent = tile["accent"]
            tile["value"].configure(text=f"{score * 100:.0f}%")
            tile["detail"].configure(text=detail)
            tile["frame"].configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
            self._draw_capsule(tile["meter"], score, accent, tile["surface"])
            self._draw_display_band(tile["band"], display_state, accent, tile["surface"])

        active_key = None
        if active_signal is not None and (active_signal.get("held") or active_signal["elevated"] >= spotlight_threshold):
            active_key = f"head:{active_signal['spec']['index']}"
        for spec in self.secondary_specs:
            tile = self.signal_tiles.get(f"head:{spec['index']}")
            if tile is None:
                continue
            head = output[spec["index"]]
            elevated = float(sum(head[idx] for idx in self.positive_class_indices if idx < len(head)))
            display_state = display_states.get(spec["index"])
            if display_state is None:
                display_state = self._display_state_from_probabilities(np.asarray(head, dtype=np.float32))
            tile["value"].configure(text=f"{elevated * 100:.0f}%")
            tile["detail"].configure(text=f"{display_state['label']} tendency")
            border = spec["accent"] if active_key == f"head:{spec['index']}" else mix_color(spec["accent"], COLORS["border_soft"], 0.35)
            tile["frame"].configure(highlightbackground=border)
            self._draw_capsule(tile["meter"], elevated, spec["accent"], spec["surface"])
            self._draw_display_band(tile["band"], display_state, spec["accent"], spec["surface"])

    def _update_primary_view(self, output: np.ndarray) -> None:
        primary_threshold, spotlight_threshold = self._effective_thresholds()
        engaged, not_engaged, dominant_level = self._engagement_scores(output)
        display_states = self._stable_display_states(output)
        engagement_display_state = display_states.get(self.engagement_head_index)
        if engagement_display_state is None:
            engagement_display_state = self._display_state_from_probabilities(
                np.asarray(output[self.engagement_head_index], dtype=np.float32)
            )
        self.engaged_block["value"].configure(text=f"{engaged * 100:.0f}%")
        self.engaged_block["detail"].configure(text=self._binary_detail(engagement_display_state))
        self.not_engaged_block["value"].configure(text=f"{not_engaged * 100:.0f}%")
        self.not_engaged_block["detail"].configure(text=self._binary_detail(self._inverse_display_state(engagement_display_state)))
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
        active_signal = self._signal_for_key(output, stable_spotlight_key, display_states)
        spotlight_active = active_signal is not None

        if primary_confidence < primary_threshold and spotlight_active:
            raw_state = "live_mixed"
        elif engaged >= not_engaged:
            raw_state = "live_engaged"
        else:
            raw_state = "live_not_engaged"

        state = self._stable_choice(self.primary_transition, raw_state, STATE_SWITCH_PATIENCE)
        headline, summary = self._primary_copy(state, active_signal)
        profile = self._affect_profile_from_display_states(state, display_states)

        confidence = (
            f"{primary_confidence * 100:.0f}% window confidence, "
            f"Primary threshold {primary_threshold * 100:.0f}%, "
            f"Dominant level {engagement_display_state['label']}"
        )
        footer = "Live monitoring active. Secondary tiles refresh from the current frame window."
        self._set_state(state, headline, confidence, summary, footer, preview_badge="Live", profile=profile)
        self._update_spotlight(active_signal, spotlight_threshold)
        self._update_signal_tiles(output, active_signal, display_states, spotlight_threshold)

    def _draw_preview_placeholder(self) -> None:
        self.preview_label.configure(
            image="",
            text="Camera idle\nStart the camera to begin live monitoring.",
            fg=COLORS["text_soft"],
        )
        self.preview_photo = None

    def _update_preview(self, frame: np.ndarray | None, *, force: bool = False) -> None:
        if frame is None:
            self._draw_preview_placeholder()
            return
        now = time.monotonic()
        if not force and (now - self.last_preview_update_monotonic) < PREVIEW_UPDATE_INTERVAL_SEC:
            return

        display_frame = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
        if not force and now < self.preview_resize_busy_until and all(self.preview_requested_size):
            width, height = self.preview_requested_size
        else:
            width = max(320, self.preview_stage.winfo_width() if hasattr(self, "preview_stage") else 0)
            height = max(240, self.preview_stage.winfo_height() if hasattr(self, "preview_stage") else 0)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), COLORS["preview"])
        x_pos = (width - image.width) // 2
        y_pos = (height - image.height) // 2
        canvas.paste(image, (x_pos, y_pos))
        photo = ImageTk.PhotoImage(canvas)
        self.preview_photo = photo
        self.preview_label.configure(image=photo, text="")
        self.last_preview_update_monotonic = now
        self.preview_last_size = (width, height)

    def _on_preview_configure(self, _event) -> None:
        size = self._current_preview_size()
        self.preview_requested_size = size
        if self.last_frame is None:
            self._draw_preview_placeholder()
            self.preview_last_size = size
            return
        width_delta = abs(size[0] - self.preview_last_size[0])
        height_delta = abs(size[1] - self.preview_last_size[1])
        if width_delta < PREVIEW_RESIZE_EPSILON and height_delta < PREVIEW_RESIZE_EPSILON:
            return
        self.preview_resize_busy_until = time.monotonic() + (PREVIEW_REFRESH_DEBOUNCE_MS / 1000.0)
        self._schedule_preview_refresh()

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
        self.last_preview_update_monotonic = 0.0
        self.last_warming_status_buffered = -1
        self.preview_resize_busy_until = 0.0
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
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_buffer.clear()
        self.last_preview_update_monotonic = 0.0
        self.last_warming_status_buffered = -1
        self.preview_resize_busy_until = 0.0
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
        if not self.closing and self.ui_after_id is None:
            self.ui_after_id = self.root.after(UI_UPDATE_INTERVAL_MS, self._update_loop)

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
            should_refresh_warming = (
                buffered != self.last_warming_status_buffered
                and (
                    buffered <= 1
                    or buffered >= (self.seq_len - 1)
                    or (buffered % WARMING_STATUS_STEP) == 0
                )
            )
            if should_refresh_warming:
                self.last_warming_status_buffered = buffered
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
        self.ui_after_id = None
        if self.closing:
            return

        if not self.running:
            self._update_pomodoro_timer()
            self._update_mindfulness_timer()
            self._open_pending_reflection_dialogs()
            self._schedule_ui()
            return

        pending_output: dict[str, Any] | None = None
        pending_error = None
        output_updated = False
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
            self._schedule_ui()
            return

        if pending_output is not None:
            frame_lag = self.frame_counter - int(pending_output["frame_id"])
            if frame_lag <= max(6, min(self.seq_len, INFERENCE_STALE_FRAME_TOLERANCE)):
                output = np.asarray(pending_output["output"], dtype=np.float32)
                self.output_history.append(output)
                stacked = np.stack(list(self.output_history), axis=0)
                self.target_output = stacked.mean(axis=0).astype(np.float32)
                output_updated = True
                self.last_output_time = time.time()
                self._record_engagement_sample(self.target_output, self.last_output_time)
                self._record_pomodoro_output_sample(self.target_output, self.last_output_time)

        if output_updated and self.target_output is not None and len(self.frame_buffer) >= self.seq_len and not self._ui_interaction_active():
            self.display_output = ((1.0 - DISPLAY_BLEND) * self.display_output + DISPLAY_BLEND * self.target_output).astype(np.float32)
            self._update_primary_view(self.display_output)

        self._update_pomodoro_timer()
        self._update_mindfulness_timer()
        self._open_pending_reflection_dialogs()
        self._schedule_ui()

    def on_close(self) -> None:
        self.closing = True
        if self.scroll_animation_after_id is not None:
            self.root.after_cancel(self.scroll_animation_after_id)
            self.scroll_animation_after_id = None
        self.scroll_pending_pixels = 0.0
        if self.preview_refresh_after_id is not None:
            self.root.after_cancel(self.preview_refresh_after_id)
            self.preview_refresh_after_id = None
        if self.ui_after_id is not None:
            self.root.after_cancel(self.ui_after_id)
            self.ui_after_id = None
        self._close_checkin_dialog()
        self._close_mindfulness_checkin_dialog()
        self._close_pomodoro_final_dialog()
        self._close_mindfulness_final_dialog()
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
