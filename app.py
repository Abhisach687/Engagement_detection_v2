import asyncio
import base64
import json
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from nicegui import ui

from config import MODEL_DIR, SEQ_LEN, IMG_SIZE, NUM_CLASSES
from features.frame_dataset import IMAGENET_MEAN, IMAGENET_STD
from models.mobilenetv2_tcn import MobileNetV2_TCN


from config import DEVICE
print(f"[app] using device: {DEVICE}")
LABELS = ["Very Low", "Low", "High", "Very High"]

_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_model():
    ts_path = MODEL_DIR / "mobilenetv2_tcn_distilled.ts"
    pt_path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt"
    # ensure DEVICE imported from config triggers cuda check
    from config import DEVICE as _DEVICE
    meta_path = MODEL_DIR / "mobilenetv2_tcn_distilled_metrics.json"

    meta = {}
    if ts_path.exists():
        model = torch.jit.load(ts_path, map_location=DEVICE)
    else:
        model = MobileNetV2_TCN(num_classes=NUM_CLASSES)
        if pt_path.exists():
            state = torch.load(pt_path, map_location=DEVICE)
            model.load_state_dict(state)
        model.to(DEVICE).eval()

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return model.eval(), meta


model, model_meta = load_model()
window = deque(maxlen=SEQ_LEN)
ema_probs = np.ones(NUM_CLASSES) / NUM_CLASSES

cap = None
running = False


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb)
    return tensor


def to_base64(frame_bgr):
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


async def infer_loop(prob_bars, label_text, fps_text, trend_chart):
    global ema_probs
    last_time = cv2.getTickCount()
    step = 0
    while running:
        if cap is None:
            await asyncio.sleep(0.05)
            continue
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        window.append(preprocess(frame))
        ui.run_javascript(f"document.getElementById('live_frame').src='{to_base64(frame)}';")

        if len(window) == SEQ_LEN:
            batch = torch.stack(list(window)).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            ema_probs = 0.8 * ema_probs + 0.2 * probs
            pred_idx = int(np.argmax(ema_probs))
            max_prob = float(ema_probs[pred_idx])

            for i, p in enumerate(ema_probs):
                prob_bars[i].set_value(float(p))
            if max_prob < 0.45:
                label_text.set_text("Prediction: Uncertain")
            else:
                label_text.set_text(f"Prediction: {LABELS[pred_idx]} ({max_prob:.2f})")

            # FPS calculation
            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (now - last_time)
            last_time = now
            fps_text.set_text(f"FPS: {fps:.1f}")

            trend_chart.push(step, max_prob)
            step += 1

        await asyncio.sleep(0.03)


@ui.page("/")
async def main_page():
    global running, cap
    ui.label("Engagement Detection (MobileNetV2-TCN)").style("font-size: 24px; font-weight: 600;")
    with ui.row():
        ui.image().props('id="live_frame"').style("width: 480px; height: 320px; object-fit: cover;")
        with ui.column():
            label_text = ui.label("Prediction: --")
            fps_text = ui.label("FPS: --")
            prob_bars = []
            for i, name in enumerate(LABELS):
                with ui.row():
                    ui.label(name).style("width: 90px;")
                    bar = ui.linear_progress(value=0.0, show_value=True, color="green")
                    prob_bars.append(bar)
            trend_chart = ui.chart(
                {
                    "type": "line",
                    "data": {"labels": [], "datasets": [{"label": "Engagement Score", "data": []}]},
                    "options": {"animation": False, "scales": {"y": {"min": 0, "max": 1}}},
                }
            )

    start_btn = ui.button("Start", on_click=lambda: start_stream(prob_bars, label_text, fps_text, trend_chart))
    ui.button("Stop", on_click=stop_stream)
    meta_str = (
        f"alpha={model_meta.get('alpha','?')}, T={model_meta.get('temperature','?')}, "
        f"val_acc={model_meta.get('val_accuracy','?')}"
    )
    ui.label(f"Device: {DEVICE}")
    ui.label(f"Distilled model meta: {meta_str}")


def start_stream(prob_bars, label_text, fps_text, trend_chart):
    global running, cap
    if running:
        return
    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        ui.notify("Cannot open webcam", color="red")
        return
    cap = cap_local
    running = True
    ui.run_async(infer_loop(prob_bars, label_text, fps_text, trend_chart))


def stop_stream():
    global running, cap
    running = False
    if cap is not None:
        cap.release()
        cap = None


if __name__ == "__main__":
    ui.run(native=True, title="Engagement Detection")
