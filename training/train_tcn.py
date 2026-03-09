import json
import os
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    FRAMES_DIR,
    LABELS_DIR,
    MODEL_DIR,
    LOG_DIR,
    SEQ_LEN,
    IMG_SIZE,
    NUM_CLASSES,
    EPOCHS,
    RANDOM_STATE,
    LMDB_CACHE_PATH,
    LR,
    TCN_BATCH_SIZE,
    TCN_NUM_WORKERS,
    TCN_EPOCHS,
    TCN_MAX_EPOCHS,
)
from data.dataset_loader import load_labels
from features.frame_dataset import VideoFrameDataset
from models.mobilenetv2_tcn import MobileNetV2_TCN


def _autocast(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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


def _cleanup_after_oom(optimizer):
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_labels():
    df = load_labels(LABELS_DIR / "TrainLabels.csv")
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Engagement"])


def _make_loaders(df_train, df_val, batch_size, seq_len, cache_path, device: torch.device, cache_mode: str = "prefer"):
    cache = cache_path if cache_path and Path(cache_path).exists() else None
    force_cache = cache_mode == "force"
    no_cache = cache_mode == "off"
    worker_count = min(TCN_NUM_WORKERS, os.cpu_count() or TCN_NUM_WORKERS)
    pin_mem = device.type == "cuda"
    train_ds = VideoFrameDataset(
        df_train,
        FRAMES_DIR,
        seq_len=seq_len,
        img_size=IMG_SIZE,
        augment=True,
        split="Train",
        cache_path=cache,
        force_cache=force_cache,
        no_cache=no_cache,
    )
    val_ds = VideoFrameDataset(
        df_val,
        FRAMES_DIR,
        seq_len=seq_len,
        img_size=IMG_SIZE,
        augment=False,
        split="Train",
        cache_path=cache,
        force_cache=force_cache,
        no_cache=no_cache,
    )

    train_df_filtered = train_ds.df  # may be pruned if frames/cache missing
    class_counts = train_df_filtered["Engagement"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.DoubleTensor([weights[int(y)] for y in train_df_filtered["Engagement"]])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    common_kwargs = {
        "num_workers": worker_count,
        "pin_memory": pin_mem,
    }
    if worker_count > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, train_ds, val_ds


def _train_step(model, optimizer, scaler, criterion, x, y, device):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device):
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), y.size(0)


def _eval(model, loader, device, progress_desc: str = None, position: int = 2):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    iterator = (
        tqdm(loader, desc=progress_desc, leave=False, position=position, dynamic_ncols=True)
        if progress_desc
        else loader
    )
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total


def train_tcn(
    epochs: Optional[int] = None,
    lr: float = LR,
    weight_decay: float = 1e-4,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    cache_path: Path = LMDB_CACHE_PATH,
    cache_mode: str = "prefer",
):
    if epochs is None:
        epochs = TCN_EPOCHS
    epochs = min(max(1, int(epochs)), TCN_MAX_EPOCHS)
    if batch_size is None:
        batch_size = TCN_BATCH_SIZE
    if seq_len is None:
        seq_len = SEQ_LEN

    df_train, df_val = _split_labels()
    from config import DEVICE

    device = DEVICE
    current_batch = batch_size
    train_loader, val_loader, train_ds, _ = _make_loaders(
        df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
    )

    model = MobileNetV2_TCN(num_classes=NUM_CLASSES).to(device)
    print(
        "[train_tcn] using device: "
        f"{device} | epochs={epochs} | start batch_size={current_batch} | seq_len={seq_len} "
        f"| workers={train_loader.num_workers} | frame_chunk={model.backbone.frame_chunk_size or 'full'}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    bad_epochs = 0
    patience = 3
    best_path = MODEL_DIR / "mobilenetv2_tcn.pt"

    epoch_bar = tqdm(range(epochs), desc="[tcn] epochs", position=0)
    for epoch in epoch_bar:
        rerun_epoch = True
        while rerun_epoch:
            try:
                model.train()
                total_loss = 0.0
                train_iter = tqdm(
                    train_loader,
                    desc=f"[tcn] epoch {epoch+1}/{epochs} train",
                    leave=False,
                    position=1,
                    dynamic_ncols=True,
                )
                for x, y in train_iter:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    loss_value, batch_items = _train_step(model, optimizer, scaler, criterion, x, y, device)
                    total_loss += loss_value * batch_items
                    train_iter.set_postfix(loss=f"{loss_value:.4f}", bs=current_batch)
                rerun_epoch = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _cleanup_after_oom(optimizer)
                    if device.type == "cuda" and current_batch > 1:
                        new_bs = max(1, current_batch // 2)
                        print(f"[tcn] OOM at batch_size={current_batch}; retrying with {new_bs}")
                        current_batch = new_bs
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
                        )
                        continue
                    if device.type == "cuda":
                        device = _fallback_to_cpu(model, optimizer, tag="tcn")
                        scaler = amp.GradScaler("cuda", enabled=False)
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
                        )
                        continue
                raise

        val_acc, val_loss = _eval(
            model, val_loader, device, progress_desc=f"[tcn] val epoch {epoch+1}/{epochs}", position=2
        )
        train_loss = total_loss / len(train_ds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}", best=f"{best_acc:.4f}")
        print(f"[tcn] epoch={epoch+1}/{epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"[tcn] Epoch {epoch+1}: new best acc {best_acc:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[tcn] Early stopping after {epoch+1} epochs (no improvement for {patience} epochs).")
                break

    metrics_out = MODEL_DIR / "mobilenetv2_tcn_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({"history": history, "best_acc": best_acc}, f, indent=2)

    # Plot curves
    fig, ax1 = plt.subplots()
    ax1.plot(history["train_loss"], label="train_loss")
    ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(history["val_acc"], color="green", label="val_acc")
    ax2.set_ylabel("val_acc")
    fig.tight_layout()
    fig.savefig(LOG_DIR / "tcn_training_curve.png")
    plt.close(fig)

    print(f"Saved best TCN model to {best_path}")
    return best_path


if __name__ == "__main__":
    train_tcn()
