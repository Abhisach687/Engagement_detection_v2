import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
)
from data.dataset_loader import load_labels
from features.frame_dataset import VideoFrameDataset
from models.mobilenetv2_lstm import MobileNetV2_LSTM
from models.mobilenetv2_bilstm import MobileNetV2_BiLSTM


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


def _split_labels():
    df = load_labels(LABELS_DIR / "TrainLabels.csv")
    return train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Engagement"]
    )


def _plot_history(history: dict, out_path: Path):
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
    fig.savefig(out_path)
    plt.close(fig)


def _build_model(model_type: str, hidden_size: int, num_layers: int, dropout: float):
    if model_type == "lstm":
        return MobileNetV2_LSTM(
            num_classes=NUM_CLASSES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    return MobileNetV2_BiLSTM(
        num_classes=NUM_CLASSES,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


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


def _make_loaders(df_train, df_val, batch_size, seq_len, cache_path, device: torch.device, cache_mode: str = "prefer"):
    cache = cache_path if cache_path and Path(cache_path).exists() else None
    force_cache = cache_mode == "force"
    no_cache = cache_mode == "off"
    worker_count = min(8, os.cpu_count() or 4)
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
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, train_ds, val_ds


def objective(trial: optuna.Trial, model_type: str, cache_path=LMDB_CACHE_PATH, cache_mode: str = "prefer"):
    try:
        # Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])
        seq_len = trial.suggest_int("seq_len", 16, 40, step=4)

        df_train, df_val = _split_labels()
        from config import DEVICE

        device = DEVICE
        current_batch = batch_size
        train_loader, val_loader, train_ds, _ = _make_loaders(
            df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
        )

        print(f"[train_lstm.objective] using device: {device} | start batch_size={current_batch}")
        model = _build_model(model_type, hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

        best_acc = 0.0
        patience, bad_epochs = 3, 0
        max_epochs = min(EPOCHS, 8)

        epoch_bar = tqdm(range(max_epochs), desc=f"[{model_type}] trial {trial.number} epochs", position=0)
        for epoch in epoch_bar:
            rerun_epoch = True
            while rerun_epoch:
                try:
                    model.train()
                    total_loss = 0.0
                    train_iter = tqdm(
                        train_loader,
                        desc=f"[{model_type}] trial {trial.number} epoch {epoch+1}/{max_epochs} train",
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                    )
                    for x, y in train_iter:
                        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        with _autocast(device):
                            logits = model(x)
                            loss = criterion(logits, y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        scaler.step(optimizer)
                        scaler.update()
                        total_loss += loss.item() * y.size(0)
                        train_iter.set_postfix(loss=f"{loss.item():.4f}", bs=current_batch)
                    rerun_epoch = False
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        if device.type == "cuda" and current_batch > 1:
                            new_bs = max(1, current_batch // 2)
                            print(f"[{model_type}] OOM at batch_size={current_batch}; retrying with {new_bs}")
                            current_batch = new_bs
                            train_loader, val_loader, train_ds, _ = _make_loaders(
                                df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
                            )
                            continue
                        if device.type == "cuda":
                            device = _fallback_to_cpu(model, optimizer, tag=model_type)
                            scaler = amp.GradScaler("cuda", enabled=False)
                            train_loader, val_loader, train_ds, _ = _make_loaders(
                                df_train, df_val, current_batch, seq_len, cache_path, device, cache_mode=cache_mode
                            )
                            continue
                    raise

            val_acc, val_loss = _eval(
                model,
                val_loader,
                device,
                progress_desc=f"[{model_type}] trial {trial.number} val epoch {epoch+1}",
                position=2,
            )
            train_loss = total_loss / len(train_ds)
            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}")
            print(
                f"[{model_type}] trial={trial.number} epoch={epoch+1}/{max_epochs} "
                f"train_loss={train_loss:.4f} val_acc={val_acc:.4f}"
            )
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        return best_acc

    except Exception as e:
        trial.set_user_attr("failure", str(e))
        print(f"[{model_type}][trial {trial.number}] failed: {e}")
        raise optuna.exceptions.TrialPruned(f"failed: {e}")


def run_study(
    model_type: str,
    n_trials: int = 8,
    study_path: Path = None,
    resume: bool = True,
    cache_mode: str = "prefer",
):
    name = f"{model_type}_study"
    storage = None
    if study_path:
        if study_path.exists() and not resume:
            study_path.unlink()
        storage = f"sqlite:///{study_path}"
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, load_if_exists=bool(storage))
    study.optimize(
        lambda t: objective(t, model_type, cache_path=LMDB_CACHE_PATH, cache_mode=cache_mode),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def train_best_from_study(model_type: str, study: optuna.Study, cache_mode: str = "prefer"):
    params = study.best_params
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    batch_size = min(params["batch_size"], 8)  # clamp legacy studies to our max batch size
    seq_len = params["seq_len"]

    df_train, df_val = _split_labels()
    from config import DEVICE

    device = DEVICE
    current_batch = batch_size
    train_loader, val_loader, train_ds, val_ds = _make_loaders(
        df_train,
        df_val,
        current_batch,
        seq_len,
        cache_path=LMDB_CACHE_PATH,
        device=device,
        cache_mode=cache_mode,
    )

    print(f"[train_lstm.train_best_from_study] using device: {device} | start batch_size={current_batch}")
    model = _build_model(model_type, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_path = MODEL_DIR / f"mobilenetv2_{model_type}.pt"

    epoch_bar = tqdm(range(EPOCHS), desc=f"[{model_type}] finetune epochs", position=0)
    for epoch in epoch_bar:
        rerun_epoch = True
        while rerun_epoch:
            try:
                model.train()
                total_loss = 0.0
                # Keep final training quieter: no inner progress bar by default.
                for x, y in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * y.size(0)
                rerun_epoch = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if device.type == "cuda" and current_batch > 1:
                        new_bs = max(1, current_batch // 2)
                        print(f"[{model_type}] OOM at batch_size={current_batch}; retrying with {new_bs}")
                        current_batch = new_bs
                        train_loader, val_loader, train_ds, val_ds = _make_loaders(
                            df_train,
                            df_val,
                            current_batch,
                            seq_len,
                            cache_path=LMDB_CACHE_PATH,
                            device=device,
                            cache_mode=cache_mode,
                        )
                        continue
                    if device.type == "cuda":
                        device = _fallback_to_cpu(model, optimizer, tag=model_type)
                        scaler = amp.GradScaler("cuda", enabled=False)
                        train_loader, val_loader, train_ds, val_ds = _make_loaders(
                            df_train,
                            df_val,
                            current_batch,
                            seq_len,
                            cache_path=LMDB_CACHE_PATH,
                            device=device,
                            cache_mode=cache_mode,
                        )
                        continue
                raise

        val_acc, val_loss = _eval(
            model, val_loader, device, progress_desc=None
        )
        train_loss = total_loss / len(train_ds)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}", best=f"{best_acc:.4f}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[{model_type}] epoch={epoch+1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[{model_type}] Epoch {epoch}: new best acc {best_acc:.4f}")

    metrics_out = MODEL_DIR / f"mobilenetv2_{model_type}_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({"history": history, "best_acc": best_acc, "best_params": study.best_params}, f, indent=2)
    _plot_history(history, LOG_DIR / f"{model_type}_training_curve.png")
    print(f"Saved best {model_type} model to {best_path}")
    return best_path


if __name__ == "__main__":
    # Train both models by default
    for model_type in ["lstm", "bilstm"]:
        study = run_study(model_type=model_type, n_trials=20, study_path=MODEL_DIR / f"{model_type}_study.db")
        train_best_from_study(model_type, study)
