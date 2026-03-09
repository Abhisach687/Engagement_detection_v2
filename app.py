import asyncio
import base64
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from nicegui import ui

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
LABELS = ["Very Low", "Low", "High", "Very High"]
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1").lower() in {"1", "true", "yes"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def _select_providers():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install it with `pip install onnxruntime-gpu` "
            "or `pip install onnxruntime`."
        )

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDAExecutionProvider is not available in onnxruntime. "
            "Install `onnxruntime-gpu` or set REQUIRE_CUDA=0 to allow CPU inference."
        )
    return ["CPUExecutionProvider"], "cpu"


def load_model():
    onnx_path = MODEL_DIR / "mobilenetv2_tcn_distilled.onnx"
    if not onnx_path.exists():
        raise RuntimeError(
            f"Missing ONNX model at {onnx_path}. Run `python onxx_port.py` after installing `onnx`."
        )

    providers, device_label = _select_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    meta_path = MODEL_DIR / "mobilenetv2_tcn_distilled_metrics.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return session, meta, device_label


SESSION, MODEL_META, DEVICE_LABEL = load_model()
print(f"[app] using backend: onnx | device: {DEVICE_LABEL}")
WINDOW = deque(maxlen=SEQ_LEN)
EMA_PROBS = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
CAP = None
RUNNING = False
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def predict_probs(batch: np.ndarray) -> np.ndarray:
    logits = SESSION.run(None, {INPUT_NAME: batch.astype(np.float32, copy=False)})[0]
    probs = _softmax(np.asarray(logits, dtype=np.float32))
    return probs[0]


def to_base64(frame_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


async def infer_loop(prob_bars, label_text, fps_text, trend_chart):
    global EMA_PROBS
    last_time = cv2.getTickCount()
    step = 0

    while RUNNING:
        if CAP is None:
            await asyncio.sleep(0.05)
            continue

        ret, frame = CAP.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        WINDOW.append(preprocess(frame))
        ui.run_javascript(f"document.getElementById('live_frame').src='{to_base64(frame)}';")

        if len(WINDOW) == SEQ_LEN:
            batch = np.expand_dims(np.stack(list(WINDOW), axis=0), axis=0)
            probs = predict_probs(batch)
            EMA_PROBS = 0.8 * EMA_PROBS + 0.2 * probs
            pred_idx = int(np.argmax(EMA_PROBS))
            max_prob = float(EMA_PROBS[pred_idx])

            for i, p in enumerate(EMA_PROBS):
                prob_bars[i].set_value(float(p))

            if max_prob < 0.45:
                label_text.set_text("Prediction: Uncertain")
            else:
                label_text.set_text(f"Prediction: {LABELS[pred_idx]} ({max_prob:.2f})")

            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(now - last_time, 1)
            last_time = now
            fps_text.set_text(f"FPS: {fps:.1f}")

            trend_chart.push(step, max_prob)
            step += 1

        await asyncio.sleep(0.03)


@ui.page("/")
async def main_page():
    ui.label("Engagement Detection (ONNX Runtime)").style("font-size: 24px; font-weight: 600;")
    with ui.row():
        ui.image().props('id="live_frame"').style("width: 480px; height: 320px; object-fit: cover;")
        with ui.column():
            label_text = ui.label("Prediction: --")
            fps_text = ui.label("FPS: --")
            prob_bars = []
            for name in LABELS:
                with ui.row():
                    ui.label(name).style("width: 90px;")
                    prob_bars.append(ui.linear_progress(value=0.0, show_value=True, color="green"))
            trend_chart = ui.chart(
                {
                    "type": "line",
                    "data": {"labels": [], "datasets": [{"label": "Engagement Score", "data": []}]},
                    "options": {"animation": False, "scales": {"y": {"min": 0, "max": 1}}},
                }
            )

    ui.button("Start", on_click=lambda: start_stream(prob_bars, label_text, fps_text, trend_chart))
    ui.button("Stop", on_click=stop_stream)
    meta_str = (
        f"alpha={MODEL_META.get('alpha', '?')}, T={MODEL_META.get('temperature', '?')}, "
        f"val_acc={MODEL_META.get('val_accuracy', '?')}"
    )
    ui.label(f"Backend: onnxruntime")
    ui.label(f"Device: {DEVICE_LABEL}")
    ui.label(f"Distilled model meta: {meta_str}")


def start_stream(prob_bars, label_text, fps_text, trend_chart):
    global CAP, RUNNING
    if RUNNING:
        return

    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        ui.notify("Cannot open webcam", color="red")
        return

    CAP = cap_local
    RUNNING = True
    ui.run_async(infer_loop(prob_bars, label_text, fps_text, trend_chart))


def stop_stream():
    global CAP, RUNNING
    RUNNING = False
    if CAP is not None:
        CAP.release()
        CAP = None


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title="Engagement Detection")
