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
RAW_AFFECT_LEVELS = ("Very Low", "Low", "High", "Very High")
DISPLAY_AFFECT_LEVELS = ("Very Low", "Low", "Medium", "High", "Very High")
AFFECT_LEVELS = RAW_AFFECT_LEVELS
DISPLAY_LEVEL_ANCHORS = (0.0, 1.0, 3.0, 4.0)
MEDIUM_DISPLAY_RANGE = (1.6, 2.4)
MEDIUM_CENTER_MIN_MASS = 0.55
MEDIUM_BALANCE_MAX_GAP = 0.20
MEDIUM_EXTREME_MAX_MASS = 0.20
DISPLAY_LEVEL_INDEX = {label: index for index, label in enumerate(DISPLAY_AFFECT_LEVELS)}
NUM_AFFECTS = len(AFFECT_COLUMNS)


def row_to_affect_labels(row) -> list[int]:
    return [int(row[column]) for column in AFFECT_COLUMNS]


def _normalized_probabilities(probabilities: list[float] | tuple[float, ...]) -> list[float]:
    values = [max(0.0, float(value)) for value in probabilities]
    if len(values) != len(RAW_AFFECT_LEVELS):
        raise ValueError(f"Expected {len(RAW_AFFECT_LEVELS)} raw affect probabilities, got {len(values)}.")
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def display_score_from_raw_probabilities(probabilities: list[float] | tuple[float, ...]) -> float:
    normalized = _normalized_probabilities(probabilities)
    return float(sum(probability * anchor for probability, anchor in zip(normalized, DISPLAY_LEVEL_ANCHORS)))


def display_marker_position(score: float) -> float:
    return max(0.0, min(1.0, float(score) / 4.0))


def display_level_from_raw_index(raw_index: int) -> str:
    index = int(raw_index)
    if index < 0 or index >= len(RAW_AFFECT_LEVELS):
        raise ValueError(f"Raw affect index out of range: {raw_index}")
    return DISPLAY_AFFECT_LEVELS[index if index < 2 else index + 1]


def display_level_index(label: str | None) -> int | None:
    if label is None:
        return None
    return DISPLAY_LEVEL_INDEX[str(label)]


def display_label_to_raw_label(label: str | None) -> int | None:
    if label is None:
        return None
    normalized = str(label).strip()
    if normalized == "Medium":
        return None
    if normalized == "Very Low":
        return 0
    if normalized == "Low":
        return 1
    if normalized == "High":
        return 2
    if normalized == "Very High":
        return 3
    raise ValueError(f"Unsupported display affect label: {label}")


def infer_display_level(probabilities: list[float] | tuple[float, ...]) -> dict[str, float | int | str]:
    normalized = _normalized_probabilities(probabilities)
    score = display_score_from_raw_probabilities(normalized)
    very_low, low, high, very_high = normalized
    center_mass = low + high
    centered_gap = abs(low - high)
    extreme_mass = max(very_low, very_high)

    if (
        MEDIUM_DISPLAY_RANGE[0] <= score <= MEDIUM_DISPLAY_RANGE[1]
        and center_mass >= MEDIUM_CENTER_MIN_MASS
        and centered_gap <= MEDIUM_BALANCE_MAX_GAP
        and extreme_mass <= MEDIUM_EXTREME_MAX_MASS
    ):
        label = "Medium"
    elif score < 0.5:
        label = "Very Low"
    elif score < 2.0:
        label = "Low"
    elif score < 3.5:
        label = "High"
    else:
        label = "Very High"
    index = DISPLAY_LEVEL_INDEX[label]
    return {
        "label": label,
        "index": index,
        "score": score,
        "marker_position": display_marker_position(score),
    }


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
