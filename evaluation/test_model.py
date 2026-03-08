import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    FRAMES_DIR,
    LABELS_DIR,
    FEATURE_CACHE_DIR,
    MODEL_DIR,
    SEQ_LEN,
    IMG_SIZE,
    NUM_CLASSES,
    LMDB_CACHE_PATH,
    DEVICE,
)
from data.dataset_loader import load_labels
from data.cache import load_from_cache
from features.frame_dataset import VideoFrameDataset
from features.feature_pipeline import process_video
from models.mobilenetv2_lstm import MobileNetV2_LSTM
from models.mobilenetv2_bilstm import MobileNetV2_BiLSTM
from models.mobilenetv2_tcn import MobileNetV2_TCN

XGB_FORCE_CPU = os.getenv("XGB_FORCE_CPU", "0").lower() in {"1", "true", "yes"}


def _xgb_has_cuda(xgb):
    return hasattr(xgb.core._LIB, "XGDeviceQuantileDMatrixCreate")


def _xgb_device_params(xgb):
    if XGB_FORCE_CPU or not _xgb_has_cuda(xgb):
        return {"tree_method": "hist", "predictor": "auto"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "device": "cuda"}


def _autocast(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _metrics(y_true, y_pred) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_xgb(split: str = "Validation"):
    import xgboost as xgb

    df = load_labels(LABELS_DIR / f"{split}Labels.csv")
    model_path = MODEL_DIR / "xgb_hog.json"
    scaler_path = MODEL_DIR / "xgb_hog_scaler.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained XGBoost model not found.")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(**_xgb_device_params(xgb))
    scaler = joblib.load(scaler_path)

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[xgb] {split} feature build", leave=False):
        clip = row["ClipID"]
        folder = FRAMES_DIR / split / clip
        if not folder.exists():
            continue
        cached = load_from_cache(LMDB_CACHE_PATH, clip, split=split)
        if cached and cached[2] is not None:
            feat = cached[2]
        else:
            feat = process_video(folder, FEATURE_CACHE_DIR, num_frames=30)
        X.append(feat)
        y.append(int(row["Engagement"]))

    X = scaler.transform(np.array(X))
    preds = model.predict(X)
    metrics = _metrics(y, preds)

    out_path = MODEL_DIR / f"xgb_{split.lower()}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[xgb] {split} accuracy: {metrics['accuracy']:.4f}")
    return metrics


def _load_seq_model(model_type: str, device):
    if model_type == "lstm":
        model = MobileNetV2_LSTM(num_classes=NUM_CLASSES)
        weight_path = MODEL_DIR / "mobilenetv2_lstm.pt"
    elif model_type == "bilstm":
        model = MobileNetV2_BiLSTM(num_classes=NUM_CLASSES)
        weight_path = MODEL_DIR / "mobilenetv2_bilstm.pt"
    elif model_type == "tcn":
        model = MobileNetV2_TCN(num_classes=NUM_CLASSES)
        weight_path = MODEL_DIR / "mobilenetv2_tcn.pt"
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    if not weight_path.exists():
        raise FileNotFoundError(f"Missing weights: {weight_path}")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _make_loader(df, split, batch_size, device: torch.device):
    cache = LMDB_CACHE_PATH if LMDB_CACHE_PATH.exists() else None
    dataset = VideoFrameDataset(
        df,
        FRAMES_DIR,
        seq_len=SEQ_LEN,
        img_size=IMG_SIZE,
        augment=False,
        split=split,
        cache_path=cache,
    )
    worker_count = min(8, os.cpu_count() or 4)
    pin_mem = device.type == "cuda"
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": pin_mem,
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs), dataset


def evaluate_seq(model_type: str, split: str = "Validation", batch_size: int = 4):
    df = load_labels(LABELS_DIR / f"{split}Labels.csv")
    current_device = DEVICE
    current_bs = batch_size
    loader, _ = _make_loader(df, split, current_bs, current_device)

    model = _load_seq_model(model_type, current_device)
    while True:
        all_preds, all_labels = [], []
        with torch.no_grad():
            try:
                for x, y in tqdm(loader, desc=f"[{model_type}] {split} eval", leave=False, dynamic_ncols=True):
                    x = x.to(current_device, non_blocking=True)
                    with _autocast(current_device):
                        logits = model(x)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(y.numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if current_device.type == "cuda" and current_bs > 1:
                        new_bs = max(1, current_bs // 2)
                        print(f"[{model_type} eval] CUDA OOM at bs={current_bs}; retrying with bs={new_bs}")
                        current_bs = new_bs
                        loader, _ = _make_loader(df, split, current_bs, current_device)
                        continue
                    if current_device.type == "cuda":
                        print(f"[{model_type} eval] CUDA OOM -> falling back to CPU.")
                        current_device = torch.device("cpu")
                        model = model.to(current_device)
                        loader, _ = _make_loader(df, split, current_bs, current_device)
                        continue
                raise

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        metrics = _metrics(y_true, y_pred)

        out_path = MODEL_DIR / f"{model_type}_{split.lower()}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{model_type}] {split} accuracy: {metrics['accuracy']:.4f}")
        return metrics


def evaluate_distilled(split: str = "Validation", batch_size: int = 4):
    df = load_labels(LABELS_DIR / f"{split}Labels.csv")
    current_device = DEVICE
    current_bs = batch_size
    loader, _ = _make_loader(df, split, current_bs, current_device)

    ts_path = MODEL_DIR / "mobilenetv2_tcn_distilled.ts"
    pt_path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt"
    if ts_path.exists():
        model = torch.jit.load(ts_path, map_location=current_device)
    elif pt_path.exists():
        model = MobileNetV2_TCN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(pt_path, map_location=current_device))
    else:
        raise FileNotFoundError("Distilled TCN weights not found.")
    model = model.to(current_device).eval()

    while True:
        all_preds, all_labels = [], []
        with torch.no_grad():
            try:
                for x, y in tqdm(loader, desc=f"[tcn_distilled] {split} eval", leave=False, dynamic_ncols=True):
                    x = x.to(current_device, non_blocking=True)
                    with _autocast(current_device):
                        logits = model(x)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(y.numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if current_device.type == "cuda" and current_bs > 1:
                        new_bs = max(1, current_bs // 2)
                        print(f"[tcn_distilled eval] CUDA OOM at bs={current_bs}; retrying with bs={new_bs}")
                        current_bs = new_bs
                        loader, _ = _make_loader(df, split, current_bs, current_device)
                        continue
                    if current_device.type == "cuda":
                        print("[tcn_distilled eval] CUDA OOM -> falling back to CPU.")
                        current_device = torch.device("cpu")
                        model = model.to(current_device)
                        loader, _ = _make_loader(df, split, current_bs, current_device)
                        continue
                raise

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        metrics = _metrics(y_true, y_pred)
        out_path = MODEL_DIR / f"tcn_distilled_{split.lower()}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[tcn_distilled] {split} accuracy: {metrics['accuracy']:.4f}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on a split.")
    parser.add_argument(
        "--model",
        choices=["xgb", "lstm", "bilstm", "tcn", "tcn_distilled", "all"],
        default="all",
        help="Which model to evaluate",
    )
    parser.add_argument("--split", choices=["Validation", "Test"], default="Validation")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    results = {}
    targets = ["xgb", "lstm", "bilstm", "tcn", "tcn_distilled"] if args.model == "all" else [args.model]
    for m in targets:
        if m == "xgb":
            results[m] = evaluate_xgb(args.split)
        elif m in {"lstm", "bilstm", "tcn"}:
            results[m] = evaluate_seq(m, split=args.split, batch_size=args.batch_size)
        else:
            results[m] = evaluate_distilled(split=args.split, batch_size=args.batch_size)

    summary_path = MODEL_DIR / f"evaluation_{args.split.lower()}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
