import json
import os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
import joblib
import shutil
from tqdm import tqdm

from config import (
    FRAMES_DIR,
    LABELS_DIR,
    MODEL_DIR,
    SEQ_LEN,
    IMG_SIZE,
    NUM_CLASSES,
    EPOCHS,
    LMDB_CACHE_PATH,
)
from data.dataset_loader import load_labels
from data.cache import load_from_cache
from features.frame_dataset import VideoFrameDataset
from features.feature_pipeline import process_video
from models.mobilenetv2_lstm import MobileNetV2_LSTM
from models.mobilenetv2_bilstm import MobileNetV2_BiLSTM
from models.mobilenetv2_tcn import MobileNetV2_TCN


def _autocast(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

XGB_GPU_PARAMS = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "device": "cuda",
}


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _fallback_to_cpu(model, optimizer, tag: str):
    device = torch.device("cpu")
    model.to(device)
    _move_optimizer_state(optimizer, device)
    torch.cuda.empty_cache()
    print(f"[{tag}] CUDA OOM -> falling back to CPU for remaining training.")
    return device


def _load_xgb_teacher():
    model_path = MODEL_DIR / "xgb_hog.json"
    scaler_path = MODEL_DIR / "xgb_hog_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    try:
        model.set_params(**XGB_GPU_PARAMS)
    except ValueError:
        # Older xgboost versions might not accept device param; ignore silently
        model.set_params(tree_method="gpu_hist", predictor="gpu_predictor")
    scaler = joblib.load(scaler_path)
    return model, scaler


def _load_hog_feature(clip: str, cache_mode: str = "prefer"):
    cached = None
    if cache_mode != "off":
        cached = load_from_cache(LMDB_CACHE_PATH, clip, split="Train")
    if cached and cached[2] is not None:
        return cached[2]
    if cache_mode == "force":
        raise RuntimeError(f"Cache miss for clip {clip} in split Train")
    folder = FRAMES_DIR / "Train" / clip
    from config import FEATURE_CACHE_DIR

    feat = process_video(folder, FEATURE_CACHE_DIR, num_frames=30)
    return feat


def _build_teachers(device):
    teachers = []

    lstm_path = MODEL_DIR / "mobilenetv2_lstm.pt"
    bilstm_path = MODEL_DIR / "mobilenetv2_bilstm.pt"

    if lstm_path.exists():
        lstm = MobileNetV2_LSTM(num_classes=NUM_CLASSES)
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm.to(device).eval()
        teachers.append(lstm)

    if bilstm_path.exists():
        bilstm = MobileNetV2_BiLSTM(num_classes=NUM_CLASSES)
        bilstm.load_state_dict(torch.load(bilstm_path, map_location=device))
        bilstm.to(device).eval()
        teachers.append(bilstm)

    return teachers

def _make_loaders(batch_size, cache_path, device: torch.device, cache_mode: str = "prefer"):
    cache = cache_path if cache_path and Path(cache_path).exists() else None
    force_cache = cache_mode == "force"
    no_cache = cache_mode == "off"
    df_train = load_labels(LABELS_DIR / "TrainLabels.csv")
    df_val = load_labels(LABELS_DIR / "ValidationLabels.csv")
    worker_count = min(8, os.cpu_count() or 4)
    pin_mem = device.type == "cuda"

    train_ds = VideoFrameDataset(
        df_train,
        FRAMES_DIR,
        seq_len=SEQ_LEN,
        img_size=IMG_SIZE,
        augment=True,
        return_clip_id=True,
        split="Train",
        cache_path=cache,
        force_cache=force_cache,
        no_cache=no_cache,
    )
    val_ds = VideoFrameDataset(
        df_val,
        FRAMES_DIR,
        seq_len=SEQ_LEN,
        img_size=IMG_SIZE,
        augment=False,
        return_clip_id=False,
        split="Validation",
        cache_path=cache,
        force_cache=force_cache,
        no_cache=no_cache,
    )

    class_counts = df_train["Engagement"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.DoubleTensor([weights[int(y)] for y in df_train["Engagement"]])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    common_kwargs = {
        "num_workers": worker_count,
        "pin_memory": pin_mem,
    }
    if worker_count > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, train_ds, val_ds


def _eval(model, loader, device, progress_desc: str = None):
    model.eval()
    correct, total = 0, 0
    iterator = tqdm(loader, desc=progress_desc, leave=False, position=2, dynamic_ncols=True) if progress_desc else loader
    with torch.no_grad():
        for frames, labels in iterator:
            frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(frames)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def _distill_single(
    alpha,
    temperature,
    batch_size,
    lr,
    epochs,
    cache_path,
    output_path: Path,
    cache_mode: str = "prefer",
):
    from config import DEVICE
    device = DEVICE
    current_batch = batch_size
    print(f"[distill_tcn] using device: {device} | start batch_size={current_batch}")
    train_loader, val_loader, train_ds, _ = _make_loaders(
        current_batch, cache_path, device, cache_mode=cache_mode
    )

    teachers = _build_teachers(device)
    xgb_model, xgb_scaler = _load_xgb_teacher()

    student = MobileNetV2_TCN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    def teacher_probs(teacher_logits):
        probs = []
        for logits in teacher_logits:
            probs.append(F.softmax(logits / temperature, dim=1))
        if probs:
            return torch.stack(probs).mean(dim=0)
        return None

    epoch_bar = tqdm(range(epochs), desc=f"[distill] a={alpha} T={temperature} epochs", position=0)
    best_acc = -1.0
    bad_epochs = 0
    patience = 3
    best_state = None
    for epoch in epoch_bar:
        rerun_epoch = True
        while rerun_epoch:
            try:
                student.train()
                total_loss = 0.0
                train_iter = tqdm(
                    train_loader,
                    desc=f"[distill] a={alpha} T={temperature} epoch {epoch+1}/{epochs} train",
                    leave=False,
                    position=1,
                    dynamic_ncols=True,
                )
                for frames, labels, clip_ids in train_iter:
                    frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    with torch.no_grad():
                        neural_logits = [t(frames) for t in teachers]

                        xgb_logits = None
                        if xgb_model is not None:
                            feats = np.stack([_load_hog_feature(c, cache_mode=cache_mode) for c in clip_ids])
                            feats_scaled = xgb_scaler.transform(feats)
                            xgb_probs = xgb_model.predict_proba(feats_scaled)
                            xgb_logits = torch.from_numpy(np.log(xgb_probs + 1e-8)).to(device)

                        combined_probs = teacher_probs(neural_logits)
                        if combined_probs is None and xgb_logits is not None:
                            combined_probs = F.softmax(xgb_logits / temperature, dim=1)
                        elif combined_probs is not None and xgb_logits is not None:
                            xgb_soft = F.softmax(xgb_logits / temperature, dim=1)
                            combined_probs = 0.5 * combined_probs + 0.5 * xgb_soft

                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device):
                        student_logits = student(frames)
                        ce_loss = F.cross_entropy(student_logits, labels)
                        if combined_probs is not None:
                            kd_loss = F.kl_div(
                                F.log_softmax(student_logits / temperature, dim=1),
                                combined_probs,
                                reduction="batchmean",
                            ) * (temperature**2)
                            loss = alpha * kd_loss + (1 - alpha) * ce_loss
                        else:
                            loss = ce_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * labels.size(0)
                    train_iter.set_postfix(loss=f"{loss.item():.4f}", bs=current_batch)
                rerun_epoch = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if device.type == "cuda" and current_batch > 1:
                        new_bs = max(1, current_batch // 2)
                        print(f"[distill] OOM at batch_size={current_batch}; retrying with {new_bs}")
                        current_batch = new_bs
                        train_loader, val_loader, train_ds, _ = _make_loaders(current_batch, cache_path, device)
                        continue
                    if device.type == "cuda":
                        device = _fallback_to_cpu(student, optimizer, tag="distill")
                        scaler = amp.GradScaler("cuda", enabled=False)
                        train_loader, val_loader, train_ds, _ = _make_loaders(current_batch, cache_path, device)
                        continue
                raise

        avg_loss = total_loss / len(train_ds)
        val_acc = _eval(
            student, val_loader, device, progress_desc=f"[distill] val a={alpha} T={temperature} epoch {epoch+1}"
        )
        print(f"[alpha={alpha},T={temperature}] epoch={epoch+1}/{epochs} distill_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            bad_epochs = 0
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[distill] Early stopping after {epoch+1} epochs (no improvement for {patience} epochs).")
                break

    # Validation accuracy for model selection
    if best_state is not None:
        student.load_state_dict(best_state)
    val_acc = _eval(student, val_loader, device, progress_desc="[distill] final val")

    torch.save(student.state_dict(), output_path)
    ts_path = output_path.with_suffix(".ts")
    scripted = torch.jit.script(student.cpu())
    scripted.save(ts_path)

    metrics = {
        "alpha": alpha,
        "temperature": temperature,
        "epochs": epochs,
        "val_accuracy": val_acc,
        "batch_size": batch_size,
        "lr": lr,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved student weights to {output_path} (val_acc={val_acc:.4f})")
    return val_acc, output_path, ts_path, metrics


def distill(
    alpha=0.5,
    temperature=4.0,
    batch_size=8,
    lr=1e-4,
    output_path: Path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt",
    cache_path: Path = LMDB_CACHE_PATH,
    search: bool = False,
    cache_mode: str = "prefer",
):
    # Cap distillation to 8 epochs to keep runs consistent with other models.
    epochs = min(EPOCHS, 8)
    candidates = [(alpha, temperature)]
    if search:
        candidates = [(a, t) for a in (0.3, 0.5, 0.7) for t in (2.0, 4.0, 6.0)]

    best = {"val_acc": -1, "pt": None, "ts": None, "metrics": None}
    for a, t in candidates:
        trial_path = output_path.with_name(f"mobilenetv2_tcn_distilled_a{a}_t{t}.pt")
        val_acc, pt_path, ts_path, metrics = _distill_single(
            alpha=a,
            temperature=t,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            cache_path=cache_path,
            output_path=trial_path,
            cache_mode=cache_mode,
        )
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "pt": pt_path, "ts": ts_path, "metrics": metrics}

    # Copy best to canonical paths
    if best["pt"]:
        shutil.copy(best["pt"], output_path)
        if best["ts"]:
            shutil.copy(best["ts"], output_path.with_suffix(".ts"))
        with open(MODEL_DIR / "mobilenetv2_tcn_distilled_metrics.json", "w") as f:
            json.dump(best["metrics"], f, indent=2)
        print(f"Selected best distillation (val_acc={best['val_acc']:.4f}) -> {output_path}")
    else:
        raise RuntimeError("Distillation failed: no candidate models produced.")


if __name__ == "__main__":
    distill()
