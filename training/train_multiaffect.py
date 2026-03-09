import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEVICE,
    DISTILL_BATCH_SIZE,
    DISTILL_EPOCHS,
    DISTILL_MAX_EPOCHS,
    DISTILL_NUM_WORKERS,
    FRAMES_DIR,
    IMG_SIZE,
    LABELS_DIR,
    LMDB_CACHE_PATH,
    MODEL_DIR,
    NUM_CLASSES,
    SEQ_LEN,
)
from data.dataset_loader import load_labels
from features.frame_dataset import VideoFrameDataset
from models.mobilenetv2_bilstm import MobileNetV2_BiLSTM
from models.mobilenetv2_lstm import MobileNetV2_LSTM
from models.mobilenetv2_tcn import MobileNetV2_TCN
from utils.affect import (
    AFFECT_COLUMNS,
    NUM_AFFECTS,
    estimate_distill_hparams,
    multitask_accuracy,
    multitask_cross_entropy,
    multitask_kl_div,
)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _cleanup_after_oom(optimizer):
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fallback_to_cpu(model, optimizer, tag: str):
    device = torch.device("cpu")
    model.to(device)
    _move_optimizer_state(optimizer, device)
    torch.cuda.empty_cache()
    print(f"[{tag}] CUDA OOM -> falling back to CPU for remaining training.")
    return device


def _load_best_params(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r") as handle:
        metrics = json.load(handle)
    return metrics.get("best_params", {}) or metrics.get("train_params", {}) or {}


def _load_distilled_student_defaults() -> dict:
    metrics_path = MODEL_DIR / "mobilenetv2_tcn_distilled_metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r") as handle:
        metrics = json.load(handle)

    params = dict(metrics.get("train_params", {}) or {})
    for key in ("alpha", "temperature", "epochs", "batch_size", "lr", "weight_decay", "seq_len"):
        value = metrics.get(key)
        if value is not None:
            params[key] = value
    return params


def _resolve_distill_targets(alpha: float | None, temperature: float | None) -> tuple[float, float]:
    student_defaults = _load_distilled_student_defaults()
    resolved_alpha = student_defaults.get("alpha", 0.5) if alpha is None else alpha
    resolved_temperature = student_defaults.get("temperature", 4.0) if temperature is None else temperature
    return float(resolved_alpha), float(resolved_temperature)


def _estimated_teacher_params(model_type: str) -> dict:
    defaults = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 4,
        "seq_len": SEQ_LEN,
        "epochs": 8,
    }
    metrics_path = MODEL_DIR / f"mobilenetv2_{model_type}_metrics.json"
    params = defaults.copy()
    params.update(_load_best_params(metrics_path))
    params["batch_size"] = max(1, int(params.get("batch_size", defaults["batch_size"])))
    params["seq_len"] = max(4, int(params.get("seq_len", defaults["seq_len"])))
    params["num_layers"] = max(1, int(params.get("num_layers", defaults["num_layers"])))
    params["hidden_size"] = int(params.get("hidden_size", defaults["hidden_size"]))
    params["epochs"] = int(params.get("epochs", defaults["epochs"]))
    return params


def _estimated_distill_params() -> dict:
    lstm_params = _estimated_teacher_params("lstm")
    bilstm_params = _estimated_teacher_params("bilstm")
    estimated = estimate_distill_hparams(lstm_params, bilstm_params)
    estimated.update(_load_distilled_student_defaults())
    estimated["lr"] = float(estimated.get("lr", 1e-4))
    estimated["weight_decay"] = float(estimated.get("weight_decay", 1e-4))
    estimated["epochs"] = min(max(1, int(estimated.get("epochs", DISTILL_EPOCHS))), DISTILL_MAX_EPOCHS)
    estimated["batch_size"] = max(1, min(int(estimated["batch_size"]), DISTILL_BATCH_SIZE))
    estimated["seq_len"] = max(4, int(estimated["seq_len"]))
    return estimated


def _warm_start_multiaffect_student(student: MobileNetV2_TCN) -> dict | None:
    ckpt_path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt"
    if not ckpt_path.exists():
        return None

    state_dict = torch.load(ckpt_path, map_location="cpu")
    student_state = student.state_dict()
    reused = []

    for key, value in state_dict.items():
        if key.startswith("fc."):
            continue
        target = student_state.get(key)
        if target is None or target.shape != value.shape:
            continue
        student_state[key] = value.to(target.device)
        reused.append(key)

    fc_weight = state_dict.get("fc.weight")
    fc_bias = state_dict.get("fc.bias")
    if fc_weight is not None and student_state["fc.weight"].shape[1] == fc_weight.shape[1] and fc_weight.shape[0] == NUM_CLASSES:
        student_state["fc.weight"][:NUM_CLASSES].copy_(fc_weight.to(student_state["fc.weight"].device))
        reused.append("fc.weight[:engagement]")
    if fc_bias is not None and student_state["fc.bias"].shape[0] >= NUM_CLASSES and fc_bias.shape[0] == NUM_CLASSES:
        student_state["fc.bias"][:NUM_CLASSES].copy_(fc_bias.to(student_state["fc.bias"].device))
        reused.append("fc.bias[:engagement]")

    student.load_state_dict(student_state)
    return {
        "checkpoint": ckpt_path.name,
        "reused_tensors": len(reused),
        "engagement_head_initialized": fc_weight is not None and fc_bias is not None,
    }


def _make_loaders(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    return_clip_id: bool = False,
    cache_mode: str = "prefer",
):
    cache = LMDB_CACHE_PATH if LMDB_CACHE_PATH.exists() else None
    force_cache = cache_mode == "force"
    no_cache = cache_mode == "off"
    df_train = load_labels(LABELS_DIR / "TrainLabels.csv")
    df_val = load_labels(LABELS_DIR / "ValidationLabels.csv")
    worker_count = min(DISTILL_NUM_WORKERS, os.cpu_count() or DISTILL_NUM_WORKERS)
    pin_mem = device.type == "cuda"

    train_ds = VideoFrameDataset(
        df_train,
        FRAMES_DIR,
        seq_len=seq_len,
        img_size=IMG_SIZE,
        augment=True,
        return_clip_id=return_clip_id,
        split="Train",
        cache_path=cache,
        force_cache=force_cache,
        no_cache=no_cache,
        label_mode="multiaffect",
    )
    val_ds = VideoFrameDataset(
        df_val,
        FRAMES_DIR,
        seq_len=seq_len,
        img_size=IMG_SIZE,
        augment=False,
        return_clip_id=False,
        split="Validation",
        cache_path=cache,
        force_cache=False,
        no_cache=no_cache,
        label_mode="multiaffect",
    )

    train_df_filtered = train_ds.df
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


def _build_recurrent_model(model_type: str, params: dict):
    common_kwargs = {
        "num_classes": NUM_CLASSES,
        "hidden_size": int(params["hidden_size"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "num_heads": NUM_AFFECTS,
    }
    if model_type == "lstm":
        return MobileNetV2_LSTM(**common_kwargs)
    if model_type == "bilstm":
        return MobileNetV2_BiLSTM(**common_kwargs)
    raise ValueError(f"Unknown recurrent model type: {model_type}")


def _eval_multiaffect(model, loader, device, progress_desc: str | None = None):
    model.eval()
    total_items = 0
    total_loss = 0.0
    head_correct = {column: 0 for column in AFFECT_COLUMNS}
    exact_correct = 0
    iterator = tqdm(loader, desc=progress_desc, leave=False, position=2, dynamic_ncols=True) if progress_desc else loader
    with torch.no_grad():
        for batch in iterator:
            frames, labels = batch[:2]
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(frames)
                loss = multitask_cross_entropy(logits, labels)
            mean_acc, exact_acc, per_head = multitask_accuracy(logits, labels)
            batch_size = labels.size(0)
            total_items += batch_size
            total_loss += loss.item() * batch_size
            exact_correct += exact_acc * batch_size
            for column, score in per_head.items():
                head_correct[column] += score * batch_size

    if total_items == 0:
        per_head = {column: 0.0 for column in AFFECT_COLUMNS}
        return 0.0, 0.0, per_head, 0.0

    per_head = {column: head_correct[column] / total_items for column in AFFECT_COLUMNS}
    mean_acc = sum(per_head.values()) / len(per_head)
    exact_acc = exact_correct / total_items
    avg_loss = total_loss / total_items
    return mean_acc, exact_acc, per_head, avg_loss


def train_multiaffect_teacher(model_type: str, epochs: int | None = None, cache_mode: str = "prefer"):
    params = _estimated_teacher_params(model_type)
    if epochs is not None:
        params["epochs"] = int(epochs)

    current_batch = int(params["batch_size"])
    device = DEVICE
    train_loader, val_loader, train_ds, _ = _make_loaders(
        current_batch,
        int(params["seq_len"]),
        device,
        return_clip_id=False,
        cache_mode=cache_mode,
    )
    model = _build_recurrent_model(model_type, params).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_state = None
    best_metrics = None
    patience = 3
    bad_epochs = 0
    epoch_bar = tqdm(range(int(params["epochs"])), desc=f"[{model_type}_multiaffect] epochs", position=0)

    for epoch in epoch_bar:
        rerun_epoch = True
        while rerun_epoch:
            try:
                model.train()
                total_loss = 0.0
                train_iter = tqdm(
                    train_loader,
                    desc=f"[{model_type}_multiaffect] epoch {epoch+1}/{int(params['epochs'])} train",
                    leave=False,
                    position=1,
                    dynamic_ncols=True,
                )
                for frames, labels in train_iter:
                    frames = frames.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device):
                        logits = model(frames)
                        loss = multitask_cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * labels.size(0)
                    train_iter.set_postfix(loss=f"{loss.item():.4f}", bs=current_batch)
                rerun_epoch = False
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _cleanup_after_oom(optimizer)
                    if device.type == "cuda" and current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            current_batch,
                            int(params["seq_len"]),
                            device,
                            return_clip_id=False,
                            cache_mode=cache_mode,
                        )
                        continue
                    if device.type == "cuda":
                        device = _fallback_to_cpu(model, optimizer, tag=f"{model_type}_multiaffect")
                        scaler = amp.GradScaler("cuda", enabled=False)
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            current_batch,
                            int(params["seq_len"]),
                            device,
                            return_clip_id=False,
                            cache_mode=cache_mode,
                        )
                        continue
                raise

        train_loss = total_loss / max(1, len(train_ds))
        mean_acc, exact_acc, per_head, val_loss = _eval_multiaffect(
            model,
            val_loader,
            device,
            progress_desc=f"[{model_type}_multiaffect] val epoch {epoch+1}",
        )
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_mean=f"{mean_acc:.4f}", exact=f"{exact_acc:.4f}")
        print(
            f"[{model_type}_multiaffect] epoch={epoch+1}/{int(params['epochs'])} "
            f"train_loss={train_loss:.4f} val_mean_acc={mean_acc:.4f} val_exact={exact_acc:.4f}"
        )

        if mean_acc > best_score + 1e-4:
            best_score = mean_acc
            bad_epochs = 0
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_metrics = {
                "val_mean_accuracy": mean_acc,
                "val_exact_match": exact_acc,
                "val_loss": val_loss,
                "per_head_accuracy": per_head,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"Failed to train {model_type} multi-affect teacher.")

    model.load_state_dict(best_state)
    output_path = MODEL_DIR / f"mobilenetv2_{model_type}_multiaffect.pt"
    torch.save(model.state_dict(), output_path)
    scripted = torch.jit.script(model.cpu())
    scripted.save(output_path.with_suffix(".ts"))

    metrics = {
        "train_params": params,
        "best_mean_accuracy": best_metrics["val_mean_accuracy"],
        "best_exact_match": best_metrics["val_exact_match"],
        "best_val_loss": best_metrics["val_loss"],
        "best_per_head_accuracy": best_metrics["per_head_accuracy"],
        "estimated_from": f"mobilenetv2_{model_type}_metrics.json",
    }
    with open(MODEL_DIR / f"mobilenetv2_{model_type}_multiaffect_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[{model_type}_multiaffect] saved teacher to {output_path}")
    return output_path


def _load_multiaffect_teacher(model_type: str, device: torch.device):
    ckpt_path = MODEL_DIR / f"mobilenetv2_{model_type}_multiaffect.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing multi-affect teacher weights: {ckpt_path}")
    params = _estimated_teacher_params(model_type)
    params.update(_load_best_params(MODEL_DIR / f"mobilenetv2_{model_type}_multiaffect_metrics.json"))
    model = _build_recurrent_model(model_type, params)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def distill_multiaffect(
    alpha: float | None = None,
    temperature: float | None = None,
    epochs: int | None = None,
    cache_mode: str = "prefer",
):
    alpha, temperature = _resolve_distill_targets(alpha, temperature)
    params = _estimated_distill_params()
    if epochs is not None:
        params["epochs"] = int(epochs)

    current_batch = int(params["batch_size"])
    device = DEVICE
    train_loader, val_loader, train_ds, _ = _make_loaders(
        current_batch,
        int(params["seq_len"]),
        device,
        return_clip_id=False,
        cache_mode=cache_mode,
    )
    teachers = [_load_multiaffect_teacher("lstm", device), _load_multiaffect_teacher("bilstm", device)]
    student = MobileNetV2_TCN(num_classes=NUM_CLASSES, num_heads=NUM_AFFECTS).to(device)
    warm_start = _warm_start_multiaffect_student(student)
    if warm_start is not None:
        print(
            "[tcn_multiaffect_distill] warm start from "
            f"{warm_start['checkpoint']} | reused_tensors={warm_start['reused_tensors']}"
        )
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_state = None
    best_metrics = None
    patience = 3
    bad_epochs = 0
    epoch_bar = tqdm(range(int(params["epochs"])), desc="[tcn_multiaffect_distill] epochs", position=0)

    for epoch in epoch_bar:
        rerun_epoch = True
        while rerun_epoch:
            try:
                student.train()
                total_loss = 0.0
                train_iter = tqdm(
                    train_loader,
                    desc=f"[tcn_multiaffect_distill] epoch {epoch+1}/{int(params['epochs'])} train",
                    leave=False,
                    position=1,
                    dynamic_ncols=True,
                )
                for frames, labels in train_iter:
                    frames = frames.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.no_grad():
                        teacher_probs = torch.stack(
                            [F.softmax(teacher(frames) / temperature, dim=-1) for teacher in teachers]
                        ).mean(dim=0)

                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device):
                        logits = student(frames)
                        ce_loss = multitask_cross_entropy(logits, labels)
                        kd_loss = multitask_kl_div(logits, teacher_probs, temperature)
                        loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * labels.size(0)
                    train_iter.set_postfix(loss=f"{loss.item():.4f}", bs=current_batch)
                rerun_epoch = False
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _cleanup_after_oom(optimizer)
                    if device.type == "cuda" and current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            current_batch,
                            int(params["seq_len"]),
                            device,
                            return_clip_id=False,
                            cache_mode=cache_mode,
                        )
                        continue
                    if device.type == "cuda":
                        device = _fallback_to_cpu(student, optimizer, tag="tcn_multiaffect_distill")
                        teachers = [teacher.to(device).eval() for teacher in teachers]
                        scaler = amp.GradScaler("cuda", enabled=False)
                        train_loader, val_loader, train_ds, _ = _make_loaders(
                            current_batch,
                            int(params["seq_len"]),
                            device,
                            return_clip_id=False,
                            cache_mode=cache_mode,
                        )
                        continue
                raise

        train_loss = total_loss / max(1, len(train_ds))
        mean_acc, exact_acc, per_head, val_loss = _eval_multiaffect(
            student,
            val_loader,
            device,
            progress_desc=f"[tcn_multiaffect_distill] val epoch {epoch+1}",
        )
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_mean=f"{mean_acc:.4f}", exact=f"{exact_acc:.4f}")
        print(
            f"[tcn_multiaffect_distill] epoch={epoch+1}/{int(params['epochs'])} "
            f"train_loss={train_loss:.4f} val_mean_acc={mean_acc:.4f} val_exact={exact_acc:.4f}"
        )

        if mean_acc > best_score + 1e-4:
            best_score = mean_acc
            bad_epochs = 0
            best_state = {key: value.cpu() for key, value in student.state_dict().items()}
            best_metrics = {
                "val_mean_accuracy": mean_acc,
                "val_exact_match": exact_acc,
                "val_loss": val_loss,
                "per_head_accuracy": per_head,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Failed to distill multi-affect TCN student.")

    student.load_state_dict(best_state)
    output_path = MODEL_DIR / "mobilenetv2_tcn_multiaffect_distilled.pt"
    torch.save(student.state_dict(), output_path)
    scripted = torch.jit.script(student.cpu())
    scripted.save(output_path.with_suffix(".ts"))

    metrics = {
        "alpha": alpha,
        "temperature": temperature,
        "train_params": params,
        "initialized_from": warm_start,
        "best_mean_accuracy": best_metrics["val_mean_accuracy"],
        "best_exact_match": best_metrics["val_exact_match"],
        "best_val_loss": best_metrics["val_loss"],
        "best_per_head_accuracy": best_metrics["per_head_accuracy"],
        "estimated_from": {
            "lstm": "mobilenetv2_lstm_metrics.json",
            "bilstm": "mobilenetv2_bilstm_metrics.json",
        },
    }
    with open(MODEL_DIR / "mobilenetv2_tcn_multiaffect_distilled_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[tcn_multiaffect_distill] saved student to {output_path}")
    return output_path


def run_all(alpha: float | None = None, temperature: float | None = None, cache_mode: str = "prefer"):
    train_multiaffect_teacher("lstm", cache_mode=cache_mode)
    train_multiaffect_teacher("bilstm", cache_mode=cache_mode)
    distill_multiaffect(alpha=alpha, temperature=temperature, cache_mode=cache_mode)


def main():
    parser = argparse.ArgumentParser(description="Train multi-affect DAiSEE teachers and distilled TCN student.")
    parser.add_argument("task", choices=["lstm", "bilstm", "teachers", "distill", "all"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cache_mode", choices=["prefer", "force", "off"], default="prefer")
    args = parser.parse_args()

    if args.task == "lstm":
        train_multiaffect_teacher("lstm", epochs=args.epochs, cache_mode=args.cache_mode)
    elif args.task == "bilstm":
        train_multiaffect_teacher("bilstm", epochs=args.epochs, cache_mode=args.cache_mode)
    elif args.task == "teachers":
        train_multiaffect_teacher("lstm", epochs=args.epochs, cache_mode=args.cache_mode)
        train_multiaffect_teacher("bilstm", epochs=args.epochs, cache_mode=args.cache_mode)
    elif args.task == "distill":
        distill_multiaffect(
            alpha=args.alpha,
            temperature=args.temperature,
            epochs=args.epochs,
            cache_mode=args.cache_mode,
        )
    else:
        run_all(alpha=args.alpha, temperature=args.temperature, cache_mode=args.cache_mode)


if __name__ == "__main__":
    main()
