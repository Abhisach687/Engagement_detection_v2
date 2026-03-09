from __future__ import annotations

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

from utils.affect import AFFECT_COLUMNS, AFFECT_DISPLAY_NAMES, AFFECT_LEVELS


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEFAULT_SEQ_LEN = 30
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 4

WINDOW_MIN_WIDTH = 1366
WINDOW_MIN_HEIGHT = 768
CAMERA_PANEL_WIDTH = 760
ENGAGEMENT_PANEL_WIDTH = 530
PREVIEW_STAGE_HEIGHT = 412
ENGAGEMENT_PANEL_HEIGHT = 532
PRIMARY_DECISION_BOX_HEIGHT = 238
STAT_BLOCK_HEIGHT = 96
SPOTLIGHT_BOX_HEIGHT = 108

SPOTLIGHT_THRESHOLD = 0.55
PRIMARY_CONFIDENCE_THRESHOLD = 0.60
INFERENCE_INTERVAL_SEC = 0.12
DISPLAY_BLEND = 0.28
OUTPUT_HISTORY_WINDOW = 6
STATE_SWITCH_PATIENCE = 3
SPOTLIGHT_SWITCH_PATIENCE = 4
MIRROR_PREVIEW = True

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
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not capture or not capture.isOpened():
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
        self.inference_busy = False
        self.session_token = 0

        self.output_lock = threading.Lock()
        self.pending_output: np.ndarray | None = None
        self.pending_error: str | None = None
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self.output_history: deque[np.ndarray] = deque(maxlen=OUTPUT_HISTORY_WINDOW)
        self.primary_transition = {"current": None, "candidate": None, "count": 0}
        self.spotlight_transition = {"current": None, "candidate": None, "count": 0}

        self.head_specs = self._build_head_specs()
        self.secondary_specs = [spec for spec in self.head_specs if spec["index"] != self.engagement_head_index]
        self.signal_tiles: dict[str, dict[str, Any]] = {}
        self.state = "idle"

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

        self._set_initial_window()
        self._build_ui()
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self._apply_state_palette("idle")

    def _reset_temporal_smoothing(self) -> None:
        self.output_history.clear()
        self.primary_transition = {"current": None, "candidate": None, "count": 0}
        self.spotlight_transition = {"current": None, "candidate": None, "count": 0}

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

    def _set_initial_window(self) -> None:
        self.root.title("Live Engagement Monitor")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.root.option_add("*Font", "{Segoe UI} 10")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(max(WINDOW_MIN_WIDTH, screen_w - 96), 1600)
        height = min(max(WINDOW_MIN_HEIGHT, screen_h - 96), 920)
        x_pos = max(24, int((screen_w - width) / 2))
        y_pos = max(24, int((screen_h - height) / 2))
        self.root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

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
        top_bar = self._create_card(self.root, bg=COLORS["panel"], border=COLORS["border"])
        top_bar.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        top_bar.grid_columnconfigure(1, weight=1)

        title_frame = tk.Frame(top_bar, bg=COLORS["panel"])
        title_frame.grid(row=0, column=0, sticky="w", padx=20, pady=18)
        tk.Label(title_frame, text="Live Engagement Monitor", bg=COLORS["panel"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 22)).pack(anchor="w")
        tk.Label(
            title_frame,
            text="Primary engagement decision with live contextual affect signals.",
            bg=COLORS["panel"],
            fg=COLORS["text_soft"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        chip_frame = tk.Frame(top_bar, bg=COLORS["panel"])
        chip_frame.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=18)
        self.status_chip = self._create_chip(chip_frame, self.status_var.get(), COLORS["panel_soft"])
        self.status_chip.pack(side="left", padx=(0, 8))
        self._create_chip(chip_frame, self.device_label, COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(chip_frame, self.model_variant.title(), COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(chip_frame, f"{self.head_count} head{'s' if self.head_count != 1 else ''}", COLORS["panel_alt"]).pack(side="left", padx=(0, 8))
        self._create_chip(chip_frame, f"Seq {self.seq_len}", COLORS["panel_alt"]).pack(side="left")

        button_frame = tk.Frame(top_bar, bg=COLORS["panel"])
        button_frame.grid(row=0, column=2, sticky="e", padx=20, pady=18)
        self.start_button = self._create_button(button_frame, "Start Camera", self.start, COLORS["green"])
        self.stop_button = self._create_button(button_frame, "Stop", self.stop, COLORS["panel_alt"])
        self.start_button.pack(side="left", padx=(0, 10))
        self.stop_button.pack(side="left")
        self.stop_button.configure(state="disabled")

        main_area = tk.Frame(self.root, bg=COLORS["bg"])
        main_area.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        main_area.grid_anchor("nw")
        main_area.grid_columnconfigure(0, weight=0, minsize=CAMERA_PANEL_WIDTH)
        main_area.grid_columnconfigure(1, weight=0, minsize=ENGAGEMENT_PANEL_WIDTH)
        main_area.grid_rowconfigure(0, weight=1)

        preview_card = self._create_card(main_area, bg=COLORS["panel"], border=COLORS["border"])
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_card.configure(width=CAMERA_PANEL_WIDTH)
        preview_card.grid_rowconfigure(1, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)

        preview_header = tk.Frame(preview_card, bg=COLORS["panel"])
        preview_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
        preview_header.grid_columnconfigure(0, weight=1)
        tk.Label(preview_header, text="Camera Feed", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.preview_badge = self._create_chip(preview_header, self.preview_badge_var.get(), COLORS["panel_soft"])
        self.preview_badge.grid(row=0, column=1, sticky="e")

        preview_stage = tk.Frame(preview_card, bg=COLORS["preview"])
        preview_stage.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        preview_stage.configure(height=PREVIEW_STAGE_HEIGHT)
        preview_stage.grid_propagate(False)
        preview_stage.grid_rowconfigure(0, weight=1)
        preview_stage.grid_columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_stage,
            text="Camera idle\nStart the camera to begin live monitoring.",
            bg=COLORS["preview"],
            fg=COLORS["text_soft"],
            justify="center",
            font=("Segoe UI", 16),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", self._on_preview_configure)

        self.preview_footer = tk.Label(
            preview_card,
            textvariable=self.footer_var,
            bg=COLORS["panel"],
            fg=COLORS["text_muted"],
            anchor="w",
            justify="left",
            font=("Segoe UI", 10),
        )
        self.preview_footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))

        self.decision_card = self._create_card(main_area, bg=COLORS["panel"], border=COLORS["border"])
        self.decision_card.grid(row=0, column=1, sticky="nsew")
        self.decision_card.configure(width=ENGAGEMENT_PANEL_WIDTH, height=ENGAGEMENT_PANEL_HEIGHT)
        self.decision_card.grid_propagate(False)
        self.decision_card.grid_columnconfigure(0, weight=1)

        decision_body = tk.Frame(self.decision_card, bg=COLORS["panel"])
        decision_body.grid(row=0, column=0, sticky="nsew", padx=22, pady=22)
        decision_body.grid_columnconfigure(0, weight=1)
        decision_body.grid_rowconfigure(0, minsize=PRIMARY_DECISION_BOX_HEIGHT)
        decision_body.grid_rowconfigure(1, minsize=STAT_BLOCK_HEIGHT)
        decision_body.grid_rowconfigure(2, minsize=SPOTLIGHT_BOX_HEIGHT)

        primary_box = self._create_card(decision_body, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        primary_box.grid(row=0, column=0, sticky="ew")
        primary_box.configure(height=PRIMARY_DECISION_BOX_HEIGHT)
        primary_box.grid_propagate(False)
        primary_box.grid_columnconfigure(0, weight=1)
        primary_box.grid_rowconfigure(4, weight=1)

        tk.Label(primary_box, text="PRIMARY DECISION", bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))
        self.decision_badge = self._create_chip(primary_box, self.status_var.get(), COLORS["panel_soft"])
        self.decision_badge.grid(row=1, column=0, sticky="w", padx=18, pady=(10, 12))

        tk.Label(primary_box, textvariable=self.prediction_var, bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 30)).grid(row=2, column=0, sticky="w", padx=18)
        tk.Label(primary_box, textvariable=self.confidence_var, bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 13)).grid(row=3, column=0, sticky="w", padx=18, pady=(4, 6))
        tk.Label(
            primary_box,
            textvariable=self.summary_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            wraplength=460,
            justify="left",
            anchor="nw",
            height=3,
            font=("Segoe UI", 10),
        ).grid(row=4, column=0, sticky="nsew", padx=18)

        self.primary_meter = tk.Canvas(primary_box, height=32, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.primary_meter.grid(row=5, column=0, sticky="ew", padx=18, pady=(10, 14))

        stat_row = tk.Frame(decision_body, bg=COLORS["panel"])
        stat_row.grid(row=1, column=0, sticky="ew", pady=(18, 0))
        stat_row.configure(height=STAT_BLOCK_HEIGHT)
        stat_row.grid_propagate(False)
        stat_row.grid_columnconfigure(0, weight=1)
        stat_row.grid_columnconfigure(1, weight=1)
        self.engaged_block = self._create_stat_block(stat_row, "Engaged")
        self.not_engaged_block = self._create_stat_block(stat_row, "Not Engaged")
        self.engaged_block["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.not_engaged_block["frame"].grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.spotlight_card = self._create_card(decision_body, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
        self.spotlight_card.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        self.spotlight_card.configure(height=SPOTLIGHT_BOX_HEIGHT)
        self.spotlight_card.pack_propagate(False)
        tk.Label(self.spotlight_card, textvariable=self.spotlight_label_var, bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=16, pady=(12, 4))
        self.spotlight_value = tk.Label(self.spotlight_card, textvariable=self.spotlight_value_var, bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 18))
        self.spotlight_value.pack(anchor="w", padx=16)
        tk.Label(
            self.spotlight_card,
            textvariable=self.spotlight_detail_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text_soft"],
            wraplength=470,
            justify="left",
            height=2,
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(4, 8))
        self.spotlight_meter = tk.Canvas(self.spotlight_card, height=20, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
        self.spotlight_meter.pack(fill="x", padx=16, pady=(0, 12))

        bottom_band = self._create_card(self.root, bg=COLORS["panel"], border=COLORS["border"])
        bottom_band.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        bottom_band.grid_columnconfigure(0, weight=1)

        bottom_header = tk.Frame(bottom_band, bg=COLORS["panel"])
        bottom_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        bottom_header.grid_columnconfigure(0, weight=1)
        tk.Label(bottom_header, text="Signal Overview", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 13)).grid(row=0, column=0, sticky="w")
        tk.Label(bottom_header, textvariable=self.meta_var, bg=COLORS["panel"], fg=COLORS["text_muted"], font=("Segoe UI", 9)).grid(row=0, column=1, sticky="e")

        self.tiles_frame = tk.Frame(bottom_band, bg=COLORS["panel"])
        self.tiles_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
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

        for column, spec in enumerate(tile_specs):
            self.tiles_frame.grid_columnconfigure(column, weight=1)
            tile = self._create_card(self.tiles_frame, bg=COLORS["panel_alt"], border=COLORS["border_soft"])
            tile.grid(row=0, column=column, sticky="ew", padx=6)
            tk.Label(tile, text=spec["label"], bg=COLORS["panel_alt"], fg=COLORS["text_soft"], font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=14, pady=(14, 4))
            value = tk.Label(tile, text="50%", bg=COLORS["panel_alt"], fg=COLORS["text"], font=("Bahnschrift SemiBold", 24))
            value.pack(anchor="w", padx=14)
            detail = tk.Label(tile, text="Standby", bg=COLORS["panel_alt"], fg=COLORS["text_muted"], font=("Segoe UI", 9))
            detail.pack(anchor="w", padx=14, pady=(2, 10))
            meter = tk.Canvas(tile, height=16, bg=COLORS["panel_alt"], highlightthickness=0, bd=0)
            meter.pack(fill="x", padx=14, pady=(0, 14))
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

    def _spotlight_copy(self, signal: dict[str, Any] | None) -> tuple[str, str, str, float, str]:
        if not self.secondary_specs:
            return "Secondary Spotlight", "Engagement-only model active", "No secondary affect heads are available in this export.", 0.0, COLORS["text_soft"]
        signal_active = signal is not None and (signal.get("held") or signal["elevated"] >= SPOTLIGHT_THRESHOLD)
        if not signal_active:
            return "Secondary Spotlight", "Secondary signals stable", "No non-engagement head is above the promotion threshold.", 0.0, COLORS["text_soft"]
        assert signal is not None
        spec = signal["spec"]
        level = AFFECT_LEVELS[min(signal["dominant_level"], len(AFFECT_LEVELS) - 1)]
        detail_prefix = "Holding trend" if signal.get("held") and signal["elevated"] < SPOTLIGHT_THRESHOLD else "High band"
        detail = f"{detail_prefix} {signal['elevated'] * 100:.0f}% | Dominant level {level}"
        return "Secondary Spotlight", f"{spec['label']} rising", detail, float(signal["elevated"]), spec["accent"]

    def _update_spotlight(self, signal: dict[str, Any] | None) -> None:
        label, value, detail, meter_value, accent = self._spotlight_copy(signal)
        self.spotlight_label_var.set(label)
        self.spotlight_value_var.set(value)
        self.spotlight_detail_var.set(detail)
        self.spotlight_value.configure(fg=accent if meter_value > 0 else COLORS["text"])
        self.spotlight_card.configure(highlightbackground=mix_color(accent, COLORS["border_soft"], 0.35))
        self._draw_capsule(self.spotlight_meter, meter_value, accent, COLORS["panel_soft"])

    def _update_signal_tiles(self, output: np.ndarray, active_signal: dict[str, Any] | None) -> None:
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
        if active_signal is not None and (active_signal.get("held") or active_signal["elevated"] >= SPOTLIGHT_THRESHOLD):
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
        elif raw_signal is not None and raw_signal["elevated"] >= SPOTLIGHT_THRESHOLD:
            spotlight_key = f"head:{raw_signal['spec']['index']}"
        else:
            spotlight_key = "stable"
        stable_spotlight_key = self._stable_choice(self.spotlight_transition, spotlight_key, SPOTLIGHT_SWITCH_PATIENCE)
        active_signal = self._signal_for_key(output, stable_spotlight_key)
        spotlight_active = active_signal is not None

        if primary_confidence < PRIMARY_CONFIDENCE_THRESHOLD and spotlight_active:
            raw_state = "live_mixed"
        elif engaged >= not_engaged:
            raw_state = "live_engaged"
        else:
            raw_state = "live_not_engaged"

        state = self._stable_choice(self.primary_transition, raw_state, STATE_SWITCH_PATIENCE)
        headline, summary = self._primary_copy(state, active_signal)

        confidence = f"{primary_confidence * 100:.0f}% window confidence | Dominant level {AFFECT_LEVELS[dominant_level]}"
        footer = "Live monitoring active. Secondary tiles refresh from the current frame window."
        self._set_state(state, headline, confidence, summary, footer, preview_badge="Live")
        self._update_spotlight(active_signal)
        self._update_signal_tiles(output, active_signal)

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
        self.capture = open_camera(0)
        if not self.capture or not self.capture.isOpened():
            self._set_error_view("Unable to open the webcam.")
            return

        self.running = True
        self.session_token += 1
        self.frame_buffer.clear()
        self.last_frame = None
        self.pending_output = None
        self.pending_error = None
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self.last_prediction_time = 0.0
        self._reset_temporal_smoothing()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal", bg=COLORS["red_soft"], fg=COLORS["red"])
        self._set_state(
            "warming_up",
            "Warming Up",
            "Collecting live frame context",
            f"Buffering {self.seq_len} frames before the first decision.",
            f"Frame buffer 0/{self.seq_len}",
            preview_badge="Warming",
        )
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self._schedule_capture()
        self._schedule_ui()

    def stop(self) -> None:
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
        self.inference_busy = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled", bg=COLORS["panel_alt"], fg=COLORS["text"])
        self.display_output = self._neutral_output()
        self.target_output = self._neutral_output()
        self._reset_temporal_smoothing()
        self._set_state("idle", "Camera Idle", "Waiting for live input", self._idle_summary(), self._idle_footer(), preview_badge="Idle")
        self._update_spotlight(None)
        self._update_signal_tiles(self.display_output, None)
        self.last_frame = None
        self._draw_preview_placeholder()

    def _set_error_view(self, message: str) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled", bg=COLORS["panel_alt"], fg=COLORS["text"])
        self._set_state("error", "Runtime Error", message, "Check camera access or the ONNX runtime configuration.", message, preview_badge="Error")
        self.preview_label.configure(image="", text="Runtime error\nSee the status panel for details.", fg=COLORS["red"])

    def _schedule_capture(self) -> None:
        if not self.closing:
            self.capture_after_id = self.root.after(15, self._capture_loop)

    def _schedule_ui(self) -> None:
        if not self.closing:
            self.ui_after_id = self.root.after(50, self._update_loop)

    def _run_inference(self, frames: list[np.ndarray], token: int) -> None:
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
                if token == self.session_token:
                    self.pending_output = output
        except Exception as exc:
            with self.output_lock:
                if token == self.session_token:
                    self.pending_error = str(exc)
        finally:
            self.inference_busy = False

    def _capture_loop(self) -> None:
        if not self.running or self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            self._set_error_view("The webcam stopped responding.")
            return

        self.last_frame = frame
        self.frame_buffer.append(frame.copy())
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
        elif not self.inference_busy and time.time() - self.last_prediction_time >= INFERENCE_INTERVAL_SEC:
            self.inference_busy = True
            self.last_prediction_time = time.time()
            threading.Thread(target=self._run_inference, args=(list(self.frame_buffer), self.session_token), daemon=True).start()

        self._schedule_capture()

    def _update_loop(self) -> None:
        if not self.running:
            return

        pending_output = None
        pending_error = None
        with self.output_lock:
            if self.pending_output is not None:
                pending_output = self.pending_output.copy()
                self.pending_output = None
            if self.pending_error is not None:
                pending_error = self.pending_error
                self.pending_error = None

        if pending_error:
            self.stop()
            self._set_error_view(pending_error)
            return

        if pending_output is not None:
            self.output_history.append(pending_output.astype(np.float32))
            stacked = np.stack(list(self.output_history), axis=0)
            self.target_output = stacked.mean(axis=0).astype(np.float32)

        if self.target_output is not None and len(self.frame_buffer) >= self.seq_len:
            self.display_output = ((1.0 - DISPLAY_BLEND) * self.display_output + DISPLAY_BLEND * self.target_output).astype(np.float32)
            self._update_primary_view(self.display_output)

        self._schedule_ui()

    def on_close(self) -> None:
        self.closing = True
        self.stop()
        self.root.destroy()


def main() -> None:
    runtime = load_model()
    root = tk.Tk()
    app = EngagementApp(root, runtime)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
