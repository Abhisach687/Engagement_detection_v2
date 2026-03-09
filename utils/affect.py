from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


AFFECT_COLUMNS = ("Engagement", "Boredom", "Confusion", "Frustration")
AFFECT_DISPLAY_NAMES = {
    "Engagement": "Engaged",
    "Boredom": "Bored",
    "Confusion": "Confused",
    "Frustration": "Frustrated",
}
AFFECT_LEVELS = ("Very Low", "Low", "High", "Very High")
NUM_AFFECTS = len(AFFECT_COLUMNS)


def row_to_affect_labels(row) -> list[int]:
    return [int(row[column]) for column in AFFECT_COLUMNS]


def multitask_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = [F.cross_entropy(logits[:, head_idx, :], labels[:, head_idx]) for head_idx in range(logits.size(1))]
    return torch.stack(losses).mean()


def multitask_kl_div(student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    losses = []
    for head_idx in range(student_logits.size(1)):
        losses.append(
            F.kl_div(
                F.log_softmax(student_logits[:, head_idx, :] / temperature, dim=1),
                teacher_probs[:, head_idx, :],
                reduction="batchmean",
            )
        )
    return torch.stack(losses).mean() * (temperature**2)


def multitask_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def multitask_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, dict[str, float]]:
    preds = multitask_predictions(logits)
    return multitask_accuracy_from_preds(preds, labels)


def multitask_accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, dict[str, float]]:
    correct = preds.eq(labels)
    per_head = {
        column: float(correct[:, idx].float().mean().item()) for idx, column in enumerate(AFFECT_COLUMNS)
    }
    mean_accuracy = float(sum(per_head.values()) / len(per_head))
    exact_match = float(correct.all(dim=1).float().mean().item())
    return mean_accuracy, exact_match, per_head


def reshape_affect_output(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if logits.ndim == 3:
        return logits
    if logits.ndim != 2:
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
    if logits.size(1) == num_classes:
        return logits.unsqueeze(1)
    if logits.size(1) == NUM_AFFECTS * num_classes:
        return logits.view(logits.size(0), NUM_AFFECTS, num_classes)
    raise ValueError(f"Cannot reshape logits of shape {tuple(logits.shape)} into affect heads.")


def geometric_mean(values: list[float], default: float) -> float:
    valid = [float(v) for v in values if float(v) > 0.0]
    if not valid:
        return float(default)
    return math.exp(sum(math.log(v) for v in valid) / len(valid))


def rounded_even(value: float, step: int = 4, minimum: int = 4) -> int:
    return max(minimum, int(round(float(value) / step) * step))


def estimate_distill_hparams(
    lstm_params: Mapping[str, float] | None,
    bilstm_params: Mapping[str, float] | None,
) -> dict[str, float]:
    lstm_params = dict(lstm_params or {})
    bilstm_params = dict(bilstm_params or {})
    seq_len = rounded_even((float(lstm_params.get("seq_len", 32)) + float(bilstm_params.get("seq_len", 32))) / 2.0)
    batch_size = int(min(float(lstm_params.get("batch_size", 4)), float(bilstm_params.get("batch_size", 4))))
    return {
        "lr": geometric_mean(
            [float(lstm_params.get("lr", 0.0)), float(bilstm_params.get("lr", 0.0))],
            default=1e-4,
        ),
        "weight_decay": geometric_mean(
            [float(lstm_params.get("weight_decay", 0.0)), float(bilstm_params.get("weight_decay", 0.0))],
            default=1e-4,
        ),
        "seq_len": seq_len,
        "batch_size": max(1, batch_size),
    }
