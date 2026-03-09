import json
import os
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import onnxruntime as ort
except ImportError:
    ort = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SEQ_LEN = 30
IMG_SIZE = 224
NUM_CLASSES = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ENGAGEMENT_LEVELS = ["Very Low", "Low", "High", "Very High"]
MODEL_VARIANTS = {
    "engagement": {"stem": "mobilenetv2_tcn_distilled", "label": "Engagement"},
    "multiaffect": {"stem": "mobilenetv2_tcn_multiaffect_distilled", "label": "Multi-Affect"},
}
AFFECT_CARD_SPECS = (
    ("bored", 1, "Bored", "Boredom signal", "High + Very High boredom confidence"),
    ("confused", 2, "Confused", "Confusion signal", "High + Very High confusion confidence"),
    ("frustrated", 3, "Frustrated", "Frustration signal", "High + Very High frustration confidence"),
)
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"}
PREVIEW_WIDTH = 768
PREVIEW_HEIGHT = 432
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
UI_REFRESH_MS = 12
INFERENCE_INTERVAL_MS = 90
FRAME_SKIP = 2
BAR_SMOOTHING = 0.12
EMA_DECAY = 0.90
APP_BG = "#FFF9EF"
SURFACE_BG = "#FFFDF8"
SURFACE_BORDER = "#F0E1C7"
SURFACE_SHADOW = "#E8D7BB"
HERO_BG = "#FFF2B8"
HERO_BORDER = "#F0D067"
HERO_SHADOW = "#E3C978"
PREVIEW_PANEL_BG = "#1E2740"
PREVIEW_BORDER = "#6175E7"
PREVIEW_SHADOW = "#B8C1E7"
PREVIEW_TEXT = "#F6F9FF"
PREVIEW_MUTED = "#B7C3DD"
ENGAGED_BG = "#E8F6FF"
ENGAGED_ACCENT = "#62B2F7"
ENGAGED_SHADOW = "#C9DFF0"
NOT_ENGAGED_BG = "#FFE7EF"
NOT_ENGAGED_ACCENT = "#F2A3BE"
NOT_ENGAGED_SHADOW = "#ECC5D2"
BORED_BG = "#FFF3D8"
BORED_ACCENT = "#E0A33B"
BORED_SHADOW = "#E7D2A4"
CONFUSED_BG = "#E5F7F5"
CONFUSED_ACCENT = "#4FB1A7"
CONFUSED_SHADOW = "#C8E4E0"
FRUSTRATED_BG = "#FFEADF"
FRUSTRATED_ACCENT = "#E38864"
FRUSTRATED_SHADOW = "#EDC8B9"
TRACK_COLOR = "#FFF8EB"
TRACK_SHADOW = "#E9DDC9"
TEXT_PRIMARY = "#23324A"
TEXT_MUTED = "#6B7790"
BUTTON_TEXT = "#1E2E48"
BUTTON_YELLOW = "#FFD766"
BUTTON_YELLOW_ACTIVE = "#FFE083"
BUTTON_DARK = "#2A3450"
BUTTON_DARK_ACTIVE = "#3A4769"
BUTTON_DARK_TEXT = "#F8FBFF"
PREVIEW_ASPECT_RATIO = 16 / 9
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 720


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=axis, keepdims=True)
    return probs


def _format_meta_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mix_color(color_a: str, color_b: str, weight: float) -> str:
    weight = max(0.0, min(1.0, weight))
    rgb_a = tuple(int(color_a[idx : idx + 2], 16) for idx in (1, 3, 5))
    rgb_b = tuple(int(color_b[idx : idx + 2], 16) for idx in (1, 3, 5))
    mixed = tuple(int(round((1.0 - weight) * a + weight * b)) for a, b in zip(rgb_a, rgb_b))
    return "#" + "".join(f"{value:02X}" for value in mixed)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _select_providers():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install it with `pip install onnxruntime-gpu` "
            "or `pip install onnxruntime`."
        )

    available = ort.get_available_providers()
    if REQUIRE_CUDA and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_model_variant() -> str:
    multiaffect_stem = MODEL_VARIANTS["multiaffect"]["stem"]
    for suffix in (".onnx", ".ts", ".pt"):
        if (MODEL_DIR / f"{multiaffect_stem}{suffix}").exists():
            return "multiaffect"
    return "engagement"


def _onnx_meta_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(f"{onnx_path.stem}_onnx_meta.json")


def _load_onnx_export_meta(onnx_path: Path) -> dict:
    meta_path = _onnx_meta_path(onnx_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r") as handle:
        return json.load(handle)


def ensure_onnx_model(onnx_path: Path, variant: str):
    if onnx_path.exists():
        return

    try:
        from onxx_port import export_onnx
    except Exception as exc:
        raise RuntimeError(
            f"Missing ONNX model at {onnx_path}. Automatic export setup failed: {exc}"
        ) from exc

    print(f"[app] missing ONNX model, exporting {onnx_path.name} from distilled weights...")
    try:
        export_onnx(onnx_path, variant=variant)
    except Exception as exc:
        raise RuntimeError(
            f"Missing ONNX model at {onnx_path}. Automatic export failed: {exc}"
        ) from exc


def load_model():
    variant = _resolve_model_variant()
    stem = MODEL_VARIANTS[variant]["stem"]
    onnx_path = MODEL_DIR / f"{stem}.onnx"
    ensure_onnx_model(onnx_path, variant)

    providers = _select_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    device_label = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    meta_path = MODEL_DIR / f"{stem}_metrics.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as handle:
            meta = json.load(handle)
    onnx_meta = _load_onnx_export_meta(onnx_path)
    runtime_variant = onnx_meta.get("variant", variant)
    if runtime_variant not in MODEL_VARIANTS:
        runtime_variant = variant
    return session, meta, onnx_meta, device_label, runtime_variant


SESSION, MODEL_META, ONNX_EXPORT_META, DEVICE_LABEL, MODEL_VARIANT = load_model()
MODEL_HEAD_COUNT = int(ONNX_EXPORT_META.get("num_heads", len(AFFECT_CARD_SPECS) + 1 if MODEL_VARIANT == "multiaffect" else 1))
print(f"[app] using backend: onnx | device: {DEVICE_LABEL} | variant: {MODEL_VARIANT} | heads: {MODEL_HEAD_COUNT}")
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def predict_output(batch: np.ndarray) -> np.ndarray:
    logits = SESSION.run(None, {INPUT_NAME: batch.astype(np.float32, copy=False)})[0]
    probs = _softmax(np.asarray(logits, dtype=np.float32), axis=-1)
    output = probs[0]
    if MODEL_VARIANT == "multiaffect" and output.ndim == 1 and output.size == MODEL_HEAD_COUNT * NUM_CLASSES:
        output = output.reshape(MODEL_HEAD_COUNT, NUM_CLASSES)
    return output


def open_camera():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for backend in backends:
        cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            return cap
        cap.release()
    return None


class EngagementApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Engagement Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(True, True)
        self.root.configure(bg=APP_BG)
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.is_multiaffect = MODEL_VARIANT == "multiaffect"

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        layout = self._calculate_layout_metrics(self.screen_width, self.screen_height)
        self.preview_width = layout["preview_width"]
        self.preview_height = layout["preview_height"]
        self.slider_width = layout["slider_width"]
        self.hero_wrap = layout["hero_wrap"]
        self.metric_wrap = layout["metric_wrap"]
        self.meta_wrap = layout["meta_wrap"]
        self.container_pad_x = layout["pad_x"]
        self.container_pad_y = layout["pad_y"]
        self.scroll_canvas = None
        self.scroll_window = None
        self._maximize_window()

        self.cap = None
        self.running = False
        self.window = deque(maxlen=SEQ_LEN)
        self.ema_output = self._neutral_output()
        self.display_output = self.ema_output.copy()
        self.preview_image = None
        self.capture_thread = None
        self.state_lock = threading.Lock()
        self.latest_frame_bgr = None
        self.last_render_tick = time.perf_counter()
        self.last_inference_tick = 0.0
        self.preview_item = None
        self.capture_count = 0

        self.status_var = tk.StringVar(value="Idle")
        self.prediction_var = tk.StringVar(value="Awaiting webcam feed")
        self.summary_var = tk.StringVar(value="Start the camera to light up the dashboard.")
        self.fps_var = tk.StringVar(value="Preview 0.0 FPS")
        self.preview_badge_var = tk.StringVar(value="Offline")
        self.preview_footer_var = tk.StringVar(value=self._idle_footer_text())
        self.metric_widgets = {}
        self.status_chip = None
        self.preview_badge = None

        self._build_ui()
        self._set_binary_values(0.0, 0.0)
        self._set_affect_values(None)
        self._apply_state_palette("idle")

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        viewport = tk.Frame(self.root, bg=APP_BG)
        viewport.grid(row=0, column=0, sticky="nsew")
        viewport.grid_rowconfigure(0, weight=1)
        viewport.grid_columnconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(viewport, bg=APP_BG, highlightthickness=0, bd=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        container = tk.Frame(
            self.scroll_canvas,
            bg=APP_BG,
            padx=self.container_pad_x,
            pady=self.container_pad_y,
        )
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=container, anchor="nw")
        self.scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        container.bind("<Configure>", self._on_container_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        container.grid_columnconfigure(0, weight=1)

        hero_card = self._create_panel(
            container,
            row=0,
            background=HERO_BG,
            border=HERO_BORDER,
            shadow=HERO_SHADOW,
            pady=(0, 18),
        )
        hero_card.grid_columnconfigure(0, weight=1)
        hero_card.grid_columnconfigure(1, weight=0)

        hero_left = tk.Frame(hero_card, bg=HERO_BG)
        hero_left.grid(row=0, column=0, sticky="nw")
        tk.Label(
            hero_left,
            text="Focus Pulse",
            font=("Segoe UI", 24, "bold"),
            bg=HERO_BG,
            fg=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            hero_left,
            text=self._hero_subtitle(),
            font=("Segoe UI", 11),
            bg=HERO_BG,
            fg=TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        chip_row = tk.Frame(hero_left, bg=HERO_BG)
        chip_row.grid(row=2, column=0, sticky="w", pady=(16, 0))
        self.status_chip = tk.Label(
            chip_row,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            bg="#FFFDF5",
            fg=TEXT_PRIMARY,
            padx=14,
            pady=6,
        )
        self.status_chip.grid(row=0, column=0, padx=(0, 8))
        tk.Label(
            chip_row,
            text=f"{DEVICE_LABEL.upper()} | ONNX | {MODEL_VARIANTS[MODEL_VARIANT]['label'].upper()}",
            font=("Segoe UI", 10, "bold"),
            bg="#FFF8D6",
            fg=TEXT_PRIMARY,
            padx=14,
            pady=6,
        ).grid(row=0, column=1, padx=(0, 8))
        tk.Label(
            chip_row,
            textvariable=self.fps_var,
            font=("Segoe UI", 10, "bold"),
            bg="#FFF2EE",
            fg=TEXT_PRIMARY,
            padx=14,
            pady=6,
        ).grid(row=0, column=2)

        hero_right = tk.Frame(hero_card, bg=HERO_BG)
        hero_right.grid(row=0, column=1, sticky="ne", padx=(28, 0))
        tk.Label(
            hero_right,
            text="Live Decision",
            font=("Segoe UI", 10, "bold"),
            bg=HERO_BG,
            fg=TEXT_MUTED,
        ).grid(row=0, column=0, sticky="e")
        tk.Label(
            hero_right,
            textvariable=self.prediction_var,
            font=("Segoe UI", 20, "bold"),
            bg=HERO_BG,
            fg=TEXT_PRIMARY,
            justify="right",
            wraplength=self.hero_wrap,
        ).grid(row=1, column=0, sticky="e", pady=(6, 0))
        tk.Label(
            hero_right,
            textvariable=self.summary_var,
            font=("Segoe UI", 10),
            bg=HERO_BG,
            fg=TEXT_MUTED,
            justify="right",
            wraplength=self.hero_wrap,
        ).grid(row=2, column=0, sticky="e", pady=(8, 14))

        button_row = tk.Frame(hero_right, bg=HERO_BG)
        button_row.grid(row=3, column=0, sticky="e")
        tk.Button(
            button_row,
            text="Start Webcam",
            command=self.start,
            bg=BUTTON_YELLOW,
            fg=BUTTON_TEXT,
            activebackground=BUTTON_YELLOW_ACTIVE,
            activeforeground=BUTTON_TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=10,
        ).grid(row=0, column=0, padx=(0, 10))
        tk.Button(
            button_row,
            text="Stop Webcam",
            command=self.stop,
            bg=BUTTON_DARK,
            fg=BUTTON_DARK_TEXT,
            activebackground=BUTTON_DARK_ACTIVE,
            activeforeground=BUTTON_DARK_TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=10,
        ).grid(row=0, column=1)

        self.metric_widgets["engaged"] = self._create_metric_card(
            container,
            row=1,
            title="Engaged",
            eyebrow="Attention signal",
            subtitle="High + Very High engagement confidence",
            background=ENGAGED_BG,
            accent=ENGAGED_ACCENT,
            shadow=ENGAGED_SHADOW,
        )

        preview_card = self._create_panel(
            container,
            row=2,
            background=PREVIEW_PANEL_BG,
            border=PREVIEW_BORDER,
            shadow=PREVIEW_SHADOW,
            pady=(0, 18),
        )

        preview_header = tk.Frame(preview_card, bg=PREVIEW_PANEL_BG)
        preview_header.grid(row=0, column=0, sticky="ew")
        preview_header.grid_columnconfigure(0, weight=1)
        tk.Label(
            preview_header,
            text="Live Webcam",
            font=("Segoe UI", 18, "bold"),
            bg=PREVIEW_PANEL_BG,
            fg=PREVIEW_TEXT,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            preview_header,
            text=self._preview_subtitle(),
            font=("Segoe UI", 10),
            bg=PREVIEW_PANEL_BG,
            fg=PREVIEW_MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.preview_badge = tk.Label(
            preview_header,
            textvariable=self.preview_badge_var,
            font=("Segoe UI", 10, "bold"),
            bg="#FAE393",
            fg=TEXT_PRIMARY,
            padx=14,
            pady=6,
        )
        self.preview_badge.grid(row=0, column=1, rowspan=2, sticky="e")

        preview_shell = tk.Frame(
            preview_card,
            bg=mix_color(PREVIEW_PANEL_BG, "#000000", 0.16),
            highlightthickness=1,
            highlightbackground=mix_color(PREVIEW_BORDER, "#FFFFFF", 0.25),
            padx=10,
            pady=10,
        )
        preview_shell.grid(row=1, column=0, sticky="ew", pady=(16, 12))
        self.preview_canvas = tk.Canvas(
            preview_shell,
            width=self.preview_width,
            height=self.preview_height,
            bg="#0E1528",
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self._set_preview_placeholder()

        tk.Label(
            preview_card,
            textvariable=self.preview_footer_var,
            justify="left",
            bg=PREVIEW_PANEL_BG,
            fg=PREVIEW_MUTED,
            font=("Segoe UI", 10),
        ).grid(row=2, column=0, sticky="w")

        self.metric_widgets["not_engaged"] = self._create_metric_card(
            container,
            row=3,
            title="Not Engaged",
            eyebrow="Attention drift",
            subtitle="Very Low + Low engagement confidence",
            background=NOT_ENGAGED_BG,
            accent=NOT_ENGAGED_ACCENT,
            shadow=NOT_ENGAGED_SHADOW,
        )

        next_row = 4
        if self.is_multiaffect:
            card_colors = {
                "bored": (BORED_BG, BORED_ACCENT, BORED_SHADOW),
                "confused": (CONFUSED_BG, CONFUSED_ACCENT, CONFUSED_SHADOW),
                "frustrated": (FRUSTRATED_BG, FRUSTRATED_ACCENT, FRUSTRATED_SHADOW),
            }
            for key, _head_idx, title, eyebrow, subtitle in AFFECT_CARD_SPECS:
                background, accent, shadow = card_colors[key]
                self.metric_widgets[key] = self._create_metric_card(
                    container,
                    row=next_row,
                    title=title,
                    eyebrow=eyebrow,
                    subtitle=subtitle,
                    background=background,
                    accent=accent,
                    shadow=shadow,
                )
                next_row += 1

        meta_card = self._create_panel(
            container,
            row=next_row,
            background=SURFACE_BG,
            border=SURFACE_BORDER,
            shadow=SURFACE_SHADOW,
        )
        tk.Label(
            meta_card,
            text=self._meta_text(),
            justify="left",
            bg=SURFACE_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            wraplength=self.meta_wrap,
        ).grid(row=0, column=0, sticky="w")

    def _hero_subtitle(self) -> str:
        if self.is_multiaffect:
            return "Multi-affect dashboard driven by the distilled four-head MobileNetV2 + TCN student"
        return "Binary engagement dashboard driven by the distilled MobileNetV2 + TCN model"

    def _preview_subtitle(self) -> str:
        if self.is_multiaffect:
            return "Realtime preview plus smoothed engagement, boredom, confusion, and frustration inference"
        return "Realtime preview plus smoothed binary inference"

    def _idle_footer_text(self) -> str:
        if self.is_multiaffect:
            return (
                f"Multi-affect view | engagement + boredom + confusion + frustration | "
                f"frame skip {FRAME_SKIP} | infer every {INFERENCE_INTERVAL_MS} ms"
            )
        return f"Binary engagement view | frame skip {FRAME_SKIP} | infer every {INFERENCE_INTERVAL_MS} ms"

    def _live_footer_text(self) -> str:
        if self.is_multiaffect:
            return (
                f"Multi-affect trend from four emotion heads | frame skip {FRAME_SKIP} | "
                f"infer every {INFERENCE_INTERVAL_MS} ms"
            )
        return f"Binary trend from four raw classes | frame skip {FRAME_SKIP} | infer every {INFERENCE_INTERVAL_MS} ms"

    def _neutral_output(self) -> np.ndarray:
        shape = (len(AFFECT_CARD_SPECS) + 1, NUM_CLASSES) if self.is_multiaffect else (NUM_CLASSES,)
        return np.full(shape, 1.0 / NUM_CLASSES, dtype=np.float32)

    def _meta_text(self) -> str:
        parts = [
            "Backend: onnxruntime",
            f"Device: {DEVICE_LABEL}",
            f"Variant: {MODEL_VARIANTS[MODEL_VARIANT]['label']}",
            f"Heads: {MODEL_HEAD_COUNT}",
        ]
        if "alpha" in MODEL_META:
            parts.append(f"alpha={_format_meta_value(MODEL_META['alpha'])}")
        if "temperature" in MODEL_META:
            parts.append(f"T={_format_meta_value(MODEL_META['temperature'])}")
        if self.is_multiaffect:
            if "best_mean_accuracy" in MODEL_META:
                parts.append(f"mean_acc={_format_meta_value(MODEL_META['best_mean_accuracy'])}")
            if "best_exact_match" in MODEL_META:
                parts.append(f"exact_match={_format_meta_value(MODEL_META['best_exact_match'])}")
            if MODEL_META.get("initialized_from"):
                parts.append(f"warm_start={MODEL_META['initialized_from'].get('checkpoint', '?')}")
        elif "val_accuracy" in MODEL_META:
            parts.append(f"val_acc={_format_meta_value(MODEL_META['val_accuracy'])}")
        source_path = ONNX_EXPORT_META.get("source_path")
        if source_path:
            parts.append(f"onnx_source={Path(source_path).name}")
        return " | ".join(parts)

    def _calculate_layout_metrics(self, screen_width: int, screen_height: int):
        pad_x = clamp(int(screen_width * 0.018), 16, 28)
        pad_y = clamp(int(screen_height * 0.018), 14, 24)
        preview_height = clamp(int(screen_height * 0.30), 230, 420)
        preview_width = int(preview_height * PREVIEW_ASPECT_RATIO)
        preview_width = min(preview_width, clamp(int(screen_width * 0.78), 460, 1100))
        preview_height = int(preview_width / PREVIEW_ASPECT_RATIO)
        slider_width = clamp(int(screen_width * 0.64), 460, 920)
        hero_wrap = clamp(int(screen_width * 0.24), 250, 360)
        metric_wrap = clamp(slider_width - 120, 320, 720)
        meta_wrap = clamp(int(screen_width * 0.78), 520, 1100)
        return {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "slider_width": slider_width,
            "hero_wrap": hero_wrap,
            "metric_wrap": metric_wrap,
            "meta_wrap": meta_wrap,
        }

    def _maximize_window(self):
        try:
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass

        try:
            self.root.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")

    def _on_canvas_configure(self, event):
        if self.scroll_canvas is not None and self.scroll_window is not None:
            self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def _on_container_configure(self, _event):
        if self.scroll_canvas is not None:
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if self.scroll_canvas is None:
            return
        self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_panel(self, parent, row: int, background: str, border: str, shadow: str, pady=(0, 0)):
        shell = tk.Frame(parent, bg=shadow, bd=0, highlightthickness=0)
        shell.grid(row=row, column=0, sticky="ew", pady=pady)
        shell.grid_columnconfigure(0, weight=1)
        card = tk.Frame(
            shell,
            bg=background,
            highlightthickness=1,
            highlightbackground=border,
            padx=20,
            pady=18,
        )
        card.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)
        return card

    def _create_metric_card(
        self,
        parent,
        row: int,
        title: str,
        eyebrow: str,
        subtitle: str,
        background: str,
        accent: str,
        shadow: str,
    ):
        card = self._create_panel(parent, row=row, background=background, border=accent, shadow=shadow, pady=(0, 18))
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=0)

        tk.Label(
            card,
            text=eyebrow.upper(),
            font=("Segoe UI", 9, "bold"),
            bg=background,
            fg=accent,
            padx=10,
            pady=4,
        ).grid(row=0, column=0, sticky="w")
        value_var = tk.StringVar(value="0%")
        detail_var = tk.StringVar(value=subtitle)
        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 18, "bold"),
            bg=background,
            fg=TEXT_PRIMARY,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))
        tk.Label(
            card,
            textvariable=value_var,
            font=("Segoe UI", 30, "bold"),
            bg=background,
            fg=TEXT_PRIMARY,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
        tk.Label(
            card,
            textvariable=detail_var,
            font=("Segoe UI", 10),
            bg=background,
            fg=TEXT_MUTED,
            wraplength=self.metric_wrap,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 12))

        slider = tk.Canvas(card, width=self.slider_width, height=80, bg=background, highlightthickness=0)
        slider.grid(row=3, column=0, columnspan=2, sticky="ew")
        return {
            "value_var": value_var,
            "detail_var": detail_var,
            "slider": slider,
            "background": background,
            "accent": accent,
        }

    def start(self):
        if self.running:
            return

        cap = open_camera()
        if cap is None:
            messagebox.showerror("Webcam Error", "Could not open the webcam.")
            return

        self.cap = cap
        self.running = True
        self.window.clear()
        self.ema_output = self._neutral_output()
        self.display_output = self.ema_output.copy()
        self.last_render_tick = time.perf_counter()
        self.last_inference_tick = 0.0
        self.latest_frame_bgr = None
        self.capture_count = 0
        self.status_var.set("Warming Up")
        self.prediction_var.set("Reading the room")
        self.summary_var.set(f"Collecting {SEQ_LEN} frames before the first decision.")
        self.fps_var.set("Preview 0.0 FPS")
        self.preview_badge_var.set("Connecting")
        self.preview_footer_var.set(
            f"Camera opened. Building a {SEQ_LEN}-frame temporal window for smoother inference."
        )
        self._set_binary_values(0.5, 0.5)
        self._set_affect_values(None)
        self._apply_state_palette("warmup")
        self.capture_thread = threading.Thread(target=self._capture_loop, name="webcam-capture", daemon=True)
        self.capture_thread.start()
        self._update_loop()

    def stop(self):
        self.running = False
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.5)
        self.capture_thread = None
        with self.state_lock:
            self.window.clear()
            self.latest_frame_bgr = None
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.status_var.set("Idle")
        self.prediction_var.set("Awaiting webcam feed")
        self.summary_var.set("Start the camera to light up the dashboard.")
        self.fps_var.set("Preview 0.0 FPS")
        self.preview_badge_var.set("Offline")
        self.preview_footer_var.set(self._idle_footer_text())
        self.ema_output = self._neutral_output()
        self.display_output = self._neutral_output()
        self._set_binary_values(0.0, 0.0)
        self._set_affect_values(None)
        self._apply_state_palette("idle")
        self.preview_canvas.delete("all")
        self.preview_item = None
        self._set_preview_placeholder()

    def on_close(self):
        self.stop()
        self.root.destroy()

    def _engagement_probs(self, model_output: np.ndarray) -> np.ndarray:
        if model_output.ndim == 2:
            return model_output[0]
        return model_output

    def _binary_scores(self, model_output: np.ndarray):
        probs = self._engagement_probs(model_output)
        engaged = float(probs[2] + probs[3])
        not_engaged = float(probs[0] + probs[1])
        total = max(engaged + not_engaged, 1e-6)
        return engaged / total, not_engaged / total

    def _confidence_copy(self, key: str, value: float) -> str:
        if key == "engaged":
            if value >= 0.80:
                return "The latest sequence shows a strong, stable focus signal."
            if value >= 0.60:
                return "Attention is leading and holding above the midpoint."
            if value >= 0.40:
                return "Focus is present, but it is still fluctuating."
            return "Low focus signal right now across the recent window."
        if value >= 0.80:
            return "Attention drift is dominating the recent sequence."
        if value >= 0.60:
            return "The model sees clear signs of low engagement."
        if value >= 0.40:
            return "Attention drift is rising, but not fully dominant."
        return "Low disengagement signal right now across the recent window."

    def _dominant_affect(self, model_output: np.ndarray) -> tuple[str, float] | None:
        if not self.is_multiaffect:
            return None
        strongest = None
        for _key, head_idx, title, _eyebrow, _subtitle in AFFECT_CARD_SPECS:
            head_probs = model_output[head_idx]
            elevated = float(head_probs[2] + head_probs[3])
            if strongest is None or elevated > strongest[1]:
                strongest = (title, elevated)
        return strongest

    def _affect_copy(self, title: str, elevated: float, dominant_level: str) -> str:
        if elevated >= 0.80:
            return f"{title} is running high right now. Dominant level: {dominant_level}."
        if elevated >= 0.55:
            return f"{title} is noticeably elevated. Dominant level: {dominant_level}."
        if elevated >= 0.35:
            return f"{title} is present but still moderate. Dominant level: {dominant_level}."
        return f"{title} remains relatively low in the latest window. Dominant level: {dominant_level}."

    def _live_summary(self, pred_label: str, pred_conf: float, model_output: np.ndarray) -> str:
        if pred_conf >= 0.85:
            base = f"{pred_label} is leading decisively in the current temporal window."
        elif pred_conf >= 0.65:
            base = f"{pred_label} is ahead, with moderate stability across recent frames."
        else:
            base = "The decision is still soft, so read it as a trend instead of a hard state."

        dominant_affect = self._dominant_affect(model_output)
        if dominant_affect is None:
            return base
        label, elevated = dominant_affect
        if elevated < 0.35:
            return f"{base} Secondary affect signals stay fairly contained."
        return f"{base} Strongest secondary signal: {label.lower()} at {elevated * 100:.0f}%."

    def _apply_state_palette(self, state: str):
        if self.status_chip is None or self.preview_badge is None:
            return

        if state == "engaged":
            chip_bg = mix_color(ENGAGED_ACCENT, "#FFFFFF", 0.60)
            badge_bg = ENGAGED_ACCENT
        elif state == "not_engaged":
            chip_bg = mix_color(NOT_ENGAGED_ACCENT, "#FFFFFF", 0.52)
            badge_bg = NOT_ENGAGED_ACCENT
        elif state == "warmup":
            chip_bg = mix_color(BUTTON_YELLOW, "#FFFFFF", 0.45)
            badge_bg = "#F7D46A"
        else:
            chip_bg = "#FFFDF5"
            badge_bg = "#FAE393"

        self.status_chip.configure(bg=chip_bg, fg=TEXT_PRIMARY)
        self.preview_badge.configure(bg=badge_bg, fg=TEXT_PRIMARY)

    def _draw_capsule(self, canvas: tk.Canvas, x0: int, y0: int, x1: int, y1: int, fill: str, outline: str):
        radius = max(1, min((y1 - y0) // 2, (x1 - x0) // 2))
        canvas.create_rectangle(x0 + radius, y0, x1 - radius, y1, fill=fill, outline=outline, width=1)
        canvas.create_oval(x0, y0, x0 + 2 * radius, y1, fill=fill, outline=outline, width=1)
        canvas.create_oval(x1 - 2 * radius, y0, x1, y1, fill=fill, outline=outline, width=1)

    def _set_preview_placeholder(self):
        center_x = self.preview_width // 2
        center_y = self.preview_height // 2
        self.preview_canvas.create_text(
            center_x,
            center_y,
            text="Webcam preview",
            fill="#E7EEFF",
            font=("Segoe UI", 18, "bold"),
            tags="placeholder",
        )
        self.preview_canvas.create_text(
            center_x,
            center_y + 32,
            text="Press Start Webcam to begin realtime inference",
            fill="#9CA9C8",
            font=("Segoe UI", 10),
            tags="placeholder_hint",
        )

    def _draw_slider(self, canvas: tk.Canvas, value: float, accent: str, background: str):
        canvas.delete("all")
        width = int(canvas.cget("width"))
        height = int(canvas.cget("height"))
        pad_x = 28
        track_height = 20
        radius = track_height // 2
        y0 = height - 32
        y1 = y0 + track_height
        x0 = pad_x
        x1 = width - pad_x
        track_width = x1 - x0
        value = max(0.0, min(1.0, value))
        fill_x = x0 + int(track_width * value)
        accent_glow = mix_color(accent, "#FFFFFF", 0.35)

        self._draw_capsule(canvas, x0, y0 + 3, x1, y1 + 3, TRACK_SHADOW, TRACK_SHADOW)
        self._draw_capsule(canvas, x0, y0, x1, y1, TRACK_COLOR, TRACK_COLOR)

        if fill_x > x0:
            fill_right = max(fill_x, x0 + 2 * radius)
            self._draw_capsule(canvas, x0, y0, fill_right, y1, accent, accent)
            highlight_y0 = y0 + 3
            highlight_y1 = highlight_y0 + 6
            if highlight_y1 < y1:
                self._draw_capsule(
                    canvas,
                    x0 + 4,
                    highlight_y0,
                    max(x0 + 10, fill_right - 4),
                    highlight_y1,
                    accent_glow,
                    accent_glow,
                )

        knob_x = x0 + int(track_width * value)
        knob_radius = 14
        canvas.create_oval(
            knob_x - knob_radius,
            (y0 + y1) // 2 - knob_radius,
            knob_x + knob_radius,
            (y0 + y1) // 2 + knob_radius,
            fill=background,
            outline=accent,
            width=4,
        )
        canvas.create_oval(
            knob_x - 5,
            (y0 + y1) // 2 - 5,
            knob_x + 5,
            (y0 + y1) // 2 + 5,
            fill=accent,
            outline=accent,
        )

        bubble_text = f"{int(round(value * 100.0))}%"
        bubble_width = 62
        bubble_height = 30
        bubble_x0 = max(x0, min(knob_x - bubble_width // 2, x1 - bubble_width))
        bubble_y0 = 10
        self._draw_capsule(
            canvas,
            bubble_x0,
            bubble_y0,
            bubble_x0 + bubble_width,
            bubble_y0 + bubble_height,
            accent,
            accent,
        )
        canvas.create_text(
            bubble_x0 + bubble_width // 2,
            bubble_y0 + bubble_height // 2,
            text=bubble_text,
            fill=TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
        )

    def _set_binary_values(self, engaged: float, not_engaged: float):
        values = {
            "engaged": engaged,
            "not_engaged": not_engaged,
        }
        for key, value in values.items():
            widget = self.metric_widgets[key]
            percent = int(round(value * 100.0))
            widget["value_var"].set(f"{percent}%")
            widget["detail_var"].set(self._confidence_copy(key, value))
            self._draw_slider(widget["slider"], value, widget["accent"], widget["background"])

    def _set_affect_values(self, model_output: np.ndarray | None):
        if not self.is_multiaffect:
            return
        for key, head_idx, title, _eyebrow, subtitle in AFFECT_CARD_SPECS:
            widget = self.metric_widgets[key]
            if model_output is None:
                widget["value_var"].set("0%")
                widget["detail_var"].set(subtitle)
                self._draw_slider(widget["slider"], 0.0, widget["accent"], widget["background"])
                continue
            head_probs = model_output[head_idx]
            elevated = float(head_probs[2] + head_probs[3])
            dominant_level = ENGAGEMENT_LEVELS[int(np.argmax(head_probs))]
            widget["value_var"].set(f"{int(round(elevated * 100.0))}%")
            widget["detail_var"].set(self._affect_copy(title, elevated, dominant_level))
            self._draw_slider(widget["slider"], elevated, widget["accent"], widget["background"])

    def _update_preview(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.preview_width, self.preview_height), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(resized)
        self.preview_image = ImageTk.PhotoImage(image=image)
        if self.preview_item is None:
            self.preview_canvas.delete("all")
            self.preview_item = self.preview_canvas.create_image(0, 0, image=self.preview_image, anchor="nw")
        else:
            self.preview_canvas.itemconfig(self.preview_item, image=self.preview_image)

    def _capture_loop(self):
        local_window = deque(maxlen=SEQ_LEN)
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            with self.state_lock:
                self.latest_frame_bgr = frame.copy()
                self.capture_count += 1
                should_sample = self.capture_count % FRAME_SKIP == 0

            if not should_sample:
                continue

            local_window.append(preprocess(frame))
            with self.state_lock:
                self.window = deque(local_window, maxlen=SEQ_LEN)
                warming = len(local_window) < SEQ_LEN

            if warming:
                continue

            if (now - self.last_inference_tick) * 1000.0 < INFERENCE_INTERVAL_MS:
                continue

            batch = np.expand_dims(np.stack(list(local_window), axis=0), axis=0)
            model_output = predict_output(batch)
            with self.state_lock:
                self.ema_output = EMA_DECAY * self.ema_output + (1.0 - EMA_DECAY) * model_output
                self.last_inference_tick = now

    def _update_loop(self):
        if not self.running or self.cap is None:
            return

        with self.state_lock:
            frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()
            frame_count = len(self.window)
            target_output = self.ema_output.copy()

        if frame is not None:
            self._update_preview(frame)

        if frame_count < SEQ_LEN:
            self.status_var.set("Warming Up")
            self.prediction_var.set("Reading the room")
            self.summary_var.set(f"Buffering temporal context: {frame_count}/{SEQ_LEN} frames collected.")
            self.preview_badge_var.set(f"{frame_count}/{SEQ_LEN} Frames")
            self.preview_footer_var.set(
                f"Building the temporal window for a steadier result. Sampling every {FRAME_SKIP} frames."
            )
            self._apply_state_palette("warmup")
        else:
            self.status_var.set("Live")
            self.display_output = (1.0 - BAR_SMOOTHING) * self.display_output + BAR_SMOOTHING * target_output
            engaged, not_engaged = self._binary_scores(self.display_output)
            pred_label = "Engaged" if engaged >= not_engaged else "Not Engaged"
            pred_conf = max(engaged, not_engaged)
            self.prediction_var.set(f"{pred_label} at {pred_conf * 100:.0f}%")
            self.summary_var.set(self._live_summary(pred_label, pred_conf, self.display_output))
            self.preview_badge_var.set(pred_label.upper())
            self.preview_footer_var.set(self._live_footer_text())
            self._set_binary_values(engaged, not_engaged)
            self._set_affect_values(self.display_output)
            self._apply_state_palette("engaged" if pred_label == "Engaged" else "not_engaged")

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_render_tick, 1e-6)
        self.last_render_tick = now
        self.fps_var.set(f"Preview {fps:.1f} FPS")

        self.root.after(UI_REFRESH_MS, self._update_loop)


def main():
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    EngagementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
