from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"
FEEDBACK_ROOT = CACHE_DIR / "user_feedback"
CLIPS_DIR = FEEDBACK_ROOT / "clips"
EXPORTS_DIR = FEEDBACK_ROOT / "exports"
CANDIDATES_DIR = FEEDBACK_ROOT / "candidates"
LOG_PATH = FEEDBACK_ROOT / "feedback_log.jsonl"
STATE_PATH = FEEDBACK_ROOT / "session_state.json"

DEFAULT_SEQ_LEN = 30
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 4
DEFAULT_PRIMARY_THRESHOLD = 0.58
DEFAULT_SPOTLIGHT_THRESHOLD = 0.48
PRIMARY_THRESHOLD_RANGE = (0.45, 0.75)
SPOTLIGHT_THRESHOLD_RANGE = (0.35, 0.70)
ADAPT_AFTER_REVIEWS = 5
EMA_ALPHA = 0.20
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AFFECT_COLUMNS = ("Engagement", "Boredom", "Confusion", "Frustration")
AFFECT_LEVELS = ("Very Low", "Low", "High", "Very High")

MODEL_VARIANTS = {
    "engagement": {
        "stem": "mobilenetv2_tcn_distilled",
        "head_names": ["Engagement"],
        "num_heads": 1,
    },
    "multiaffect": {
        "stem": "mobilenetv2_tcn_multiaffect_distilled",
        "head_names": list(AFFECT_COLUMNS),
        "num_heads": len(AFFECT_COLUMNS),
    },
}


@dataclass
class FeedbackRecord:
    feedback_id: str
    session_id: str
    created_at: str
    created_at_epoch: float
    model_variant: str
    head_names: list[str]
    head_count: int
    class_count: int
    seq_len: int
    img_size: int
    state: str
    headline: str
    confidence_text: str
    summary: str
    primary_confidence: float
    spotlight_key: str | None
    spotlight_confidence: float
    primary_threshold_at_review: float
    spotlight_threshold_at_review: float
    rating: int
    predicted_labels: list[int]
    predicted_probabilities: list[list[float]]
    corrected_labels: list[int | None]
    known_mask: list[bool]
    explicit_corrections: list[bool]
    trusted_for_training: bool
    trust_level: str
    clip_path: str


@dataclass
class ThresholdState:
    session_id: str
    review_count: int = 0
    trusted_count: int = 0
    analytics_only_count: int = 0
    average_rating: float = 0.0
    primary_agreement_ema: float = 0.5
    spotlight_agreement_ema: float = 0.5
    effective_primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD
    effective_spotlight_threshold: float = DEFAULT_SPOTLIGHT_THRESHOLD
    last_feedback_status: str = "No feedback collected yet."
    last_feedback_id: str | None = None
    last_online_trained_at: float = 0.0
    last_offline_trained_at: float = 0.0
    last_online_candidate: str | None = None
    last_offline_candidate: str | None = None


def _utcnow() -> tuple[str, float]:
    now = datetime.now(timezone.utc)
    return now.isoformat(), now.timestamp()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_variant(variant: str) -> str:
    if variant != "auto":
        return variant
    stem = MODEL_VARIANTS["multiaffect"]["stem"]
    if (MODEL_DIR / f"{stem}.pt").exists() or (MODEL_DIR / f"{stem}.ts").exists():
        return "multiaffect"
    return "engagement"


def _state_from_payload(payload: dict[str, Any] | None, *, session_id: str) -> ThresholdState:
    payload = dict(payload or {})
    merged = asdict(ThresholdState(session_id=session_id))
    merged.update(payload)
    merged["session_id"] = str(payload.get("session_id") or session_id)
    return ThresholdState(**merged)


class FeedbackManager:
    def __init__(
        self,
        *,
        base_primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
        base_spotlight_threshold: float = DEFAULT_SPOTLIGHT_THRESHOLD,
        feedback_root: Path | None = None,
        session_id: str | None = None,
        start_new_session: bool = True,
    ) -> None:
        self.feedback_root = Path(feedback_root) if feedback_root else FEEDBACK_ROOT
        self.clips_dir = self.feedback_root / "clips"
        self.exports_dir = self.feedback_root / "exports"
        self.candidates_dir = self.feedback_root / "candidates"
        self.log_path = self.feedback_root / "feedback_log.jsonl"
        self.state_path = self.feedback_root / "session_state.json"
        for folder in (self.feedback_root, self.clips_dir, self.exports_dir, self.candidates_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self.base_primary_threshold = float(base_primary_threshold)
        self.base_spotlight_threshold = float(base_spotlight_threshold)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.state = self._load_state(start_new_session=start_new_session)
        self._persist_state()

    def _load_state(self, *, start_new_session: bool) -> ThresholdState:
        existing = _read_json(self.state_path)
        if not existing:
            return _state_from_payload({}, session_id=self.session_id)
        if not start_new_session:
            return _state_from_payload(existing, session_id=str(existing.get("session_id") or self.session_id))
        preserved = {
            "last_online_trained_at": float(existing.get("last_online_trained_at", 0.0) or 0.0),
            "last_offline_trained_at": float(existing.get("last_offline_trained_at", 0.0) or 0.0),
            "last_online_candidate": existing.get("last_online_candidate"),
            "last_offline_candidate": existing.get("last_offline_candidate"),
        }
        return _state_from_payload(preserved, session_id=self.session_id)

    def _persist_state(self) -> None:
        _write_json(self.state_path, asdict(self.state))

    def effective_thresholds(self) -> dict[str, float]:
        return {
            "primary": float(self.state.effective_primary_threshold),
            "spotlight": float(self.state.effective_spotlight_threshold),
        }

    def current_session_insight(self) -> dict[str, Any]:
        return {
            "session_id": self.state.session_id,
            "review_count": self.state.review_count,
            "trusted_count": self.state.trusted_count,
            "analytics_only_count": self.state.analytics_only_count,
            "average_rating": self.state.average_rating,
            "primary_threshold": self.state.effective_primary_threshold,
            "spotlight_threshold": self.state.effective_spotlight_threshold,
            "last_feedback_status": self.state.last_feedback_status,
            "last_feedback_id": self.state.last_feedback_id,
        }

    def build_review_snapshot(
        self,
        *,
        frames: list[np.ndarray],
        output: np.ndarray,
        model_variant: str,
        head_names: list[str],
        class_count: int,
        seq_len: int,
        img_size: int,
        state: str,
        headline: str,
        confidence_text: str,
        summary: str,
        primary_confidence: float,
        spotlight_key: str | None,
        spotlight_confidence: float,
        primary_threshold: float,
        spotlight_threshold: float,
    ) -> dict[str, Any]:
        return {
            "frames": [frame.copy() for frame in frames],
            "output": np.asarray(output, dtype=np.float32).copy(),
            "model_variant": str(model_variant),
            "head_names": [str(name) for name in head_names],
            "class_count": int(class_count),
            "seq_len": int(seq_len),
            "img_size": int(img_size),
            "state": str(state),
            "headline": str(headline),
            "confidence_text": str(confidence_text),
            "summary": str(summary),
            "primary_confidence": float(primary_confidence),
            "spotlight_key": spotlight_key,
            "spotlight_confidence": float(spotlight_confidence),
            "primary_threshold": float(primary_threshold),
            "spotlight_threshold": float(spotlight_threshold),
        }

    def submit_feedback(
        self,
        snapshot: dict[str, Any],
        *,
        rating: int,
        corrected_labels: list[int | None],
        known_mask: list[bool],
    ) -> FeedbackRecord:
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5.")

        output = np.asarray(snapshot["output"], dtype=np.float32)
        predicted_labels = [int(np.argmax(head)) for head in output]
        head_count = int(output.shape[0])
        if len(corrected_labels) != head_count or len(known_mask) != head_count:
            raise ValueError("Feedback dimensions do not match the active heads.")

        explicit_corrections = []
        for index, predicted in enumerate(predicted_labels):
            corrected = corrected_labels[index]
            explicit_corrections.append(bool(known_mask[index] and corrected is not None and int(corrected) != predicted))

        trust_level, trusted = self._classify_feedback(
            rating=rating,
            known_mask=known_mask,
            explicit_corrections=explicit_corrections,
        )
        created_at, created_at_epoch = _utcnow()
        feedback_id = f"{int(created_at_epoch * 1000)}_{uuid.uuid4().hex[:8]}"
        clip_path = self._save_clip(feedback_id, snapshot["frames"])

        record = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=self.state.session_id,
            created_at=created_at,
            created_at_epoch=created_at_epoch,
            model_variant=str(snapshot["model_variant"]),
            head_names=[str(name) for name in snapshot["head_names"]],
            head_count=head_count,
            class_count=int(snapshot["class_count"]),
            seq_len=int(snapshot["seq_len"]),
            img_size=int(snapshot["img_size"]),
            state=str(snapshot["state"]),
            headline=str(snapshot["headline"]),
            confidence_text=str(snapshot["confidence_text"]),
            summary=str(snapshot["summary"]),
            primary_confidence=float(snapshot["primary_confidence"]),
            spotlight_key=str(snapshot["spotlight_key"]) if snapshot["spotlight_key"] else None,
            spotlight_confidence=float(snapshot["spotlight_confidence"]),
            primary_threshold_at_review=float(snapshot["primary_threshold"]),
            spotlight_threshold_at_review=float(snapshot["spotlight_threshold"]),
            rating=rating,
            predicted_labels=predicted_labels,
            predicted_probabilities=output.tolist(),
            corrected_labels=[None if value is None else int(value) for value in corrected_labels],
            known_mask=[bool(value) for value in known_mask],
            explicit_corrections=explicit_corrections,
            trusted_for_training=trusted,
            trust_level=trust_level,
            clip_path=str(clip_path),
        )

        _append_jsonl(self.log_path, asdict(record))
        self._update_state(record)
        self._persist_state()
        return record

    def _save_clip(self, feedback_id: str, frames: list[np.ndarray]) -> Path:
        rgb_frames = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames]
        clip = np.stack(rgb_frames, axis=0).astype(np.uint8)
        clip_path = self.clips_dir / f"{feedback_id}.npz"
        np.savez_compressed(clip_path, frames=clip)
        return clip_path

    def _classify_feedback(
        self,
        *,
        rating: int,
        known_mask: list[bool],
        explicit_corrections: list[bool],
    ) -> tuple[str, bool]:
        any_known = any(known_mask)
        all_known = all(known_mask) if known_mask else False
        any_correction = any(explicit_corrections)
        trusted = bool(any_correction or (rating >= 4 and any_known and all_known))
        if not any_known:
            return "analytics_only_all_unknown", False
        if trusted:
            return "trusted_for_training", True
        return "analytics_only", False

    def _update_state(self, record: FeedbackRecord) -> None:
        old_primary = self.state.effective_primary_threshold
        old_spotlight = self.state.effective_spotlight_threshold
        self.state.review_count += 1
        if record.trusted_for_training:
            self.state.trusted_count += 1
        else:
            self.state.analytics_only_count += 1
        self.state.average_rating = (
            ((self.state.review_count - 1) * self.state.average_rating) + float(record.rating)
        ) / max(1, self.state.review_count)

        rating_score = float(record.rating - 1) / 4.0
        primary_score, primary_weight = self._primary_signal(record, rating_score)
        spotlight_score, spotlight_weight = self._spotlight_signal(record, rating_score)
        if primary_weight > 0:
            alpha = EMA_ALPHA * primary_weight
            self.state.primary_agreement_ema = ((1.0 - alpha) * self.state.primary_agreement_ema) + (alpha * primary_score)
        if spotlight_weight > 0:
            alpha = EMA_ALPHA * spotlight_weight
            self.state.spotlight_agreement_ema = ((1.0 - alpha) * self.state.spotlight_agreement_ema) + (alpha * spotlight_score)

        if self.state.review_count >= ADAPT_AFTER_REVIEWS:
            self.state.effective_primary_threshold = _clamp(
                self.base_primary_threshold + ((0.5 - self.state.primary_agreement_ema) * 0.20),
                PRIMARY_THRESHOLD_RANGE[0],
                PRIMARY_THRESHOLD_RANGE[1],
            )
            self.state.effective_spotlight_threshold = _clamp(
                self.base_spotlight_threshold + ((0.5 - self.state.spotlight_agreement_ema) * 0.18),
                SPOTLIGHT_THRESHOLD_RANGE[0],
                SPOTLIGHT_THRESHOLD_RANGE[1],
            )
        else:
            self.state.effective_primary_threshold = self.base_primary_threshold
            self.state.effective_spotlight_threshold = self.base_spotlight_threshold

        self.state.last_feedback_id = record.feedback_id
        self.state.last_feedback_status = self._feedback_status_message(record, old_primary=old_primary, old_spotlight=old_spotlight)

    def _primary_signal(self, record: FeedbackRecord, rating_score: float) -> tuple[float, float]:
        if not record.known_mask:
            return 0.5, 0.0
        if not record.known_mask[0]:
            return 0.5, 0.05
        if record.explicit_corrections[0]:
            return 0.0, 1.0
        return rating_score, (1.0 if record.trusted_for_training else 0.35)

    def _spotlight_signal(self, record: FeedbackRecord, rating_score: float) -> tuple[float, float]:
        if not record.spotlight_key or not record.spotlight_key.startswith("head:"):
            return 0.5, 0.0
        try:
            index = int(record.spotlight_key.split(":", 1)[1])
        except ValueError:
            return 0.5, 0.0
        if index < 0 or index >= len(record.known_mask):
            return 0.5, 0.0
        if not record.known_mask[index]:
            return 0.5, 0.05
        if record.explicit_corrections[index]:
            return 0.0, 1.0
        return rating_score, (1.0 if record.trusted_for_training else 0.35)

    def _feedback_status_message(self, record: FeedbackRecord, *, old_primary: float, old_spotlight: float) -> str:
        prefix = "Trusted feedback saved." if record.trusted_for_training else "Feedback saved for analytics."
        if self.state.review_count < ADAPT_AFTER_REVIEWS:
            return f"{prefix} Thresholds stay at base values until {ADAPT_AFTER_REVIEWS} reviews."
        primary_shift = self.state.effective_primary_threshold - old_primary
        spotlight_shift = self.state.effective_spotlight_threshold - old_spotlight
        changes = []
        if abs(primary_shift) >= 0.0025:
            changes.append(f"primary {'up' if primary_shift > 0 else 'down'} {abs(primary_shift) * 100:.1f} pts")
        if abs(spotlight_shift) >= 0.0025:
            changes.append(f"spotlight {'up' if spotlight_shift > 0 else 'down'} {abs(spotlight_shift) * 100:.1f} pts")
        if not changes:
            return f"{prefix} Thresholds held steady."
        return f"{prefix} Adjusted {' and '.join(changes)}."

    def all_feedback(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.log_path)

    def summarize_feedback(self) -> dict[str, Any]:
        records = self.all_feedback()
        rating_distribution = {str(score): 0 for score in range(1, 6)}
        trusted = 0
        total_unknown_heads = 0
        total_heads = 0
        variant_counts = {"engagement": 0, "multiaffect": 0}
        for record in records:
            rating_distribution[str(int(record["rating"]))] += 1
            if record.get("trusted_for_training"):
                trusted += 1
            known_mask = [bool(value) for value in record.get("known_mask", [])]
            total_unknown_heads += sum(1 for value in known_mask if not value)
            total_heads += len(known_mask)
            variant = str(record.get("model_variant", ""))
            if variant in variant_counts:
                variant_counts[variant] += 1
        return {
            "total_reviews": len(records),
            "trusted_reviews": trusted,
            "analytics_only_reviews": len(records) - trusted,
            "unknown_head_rate": (total_unknown_heads / total_heads) if total_heads else 0.0,
            "rating_distribution": rating_distribution,
            "variant_counts": variant_counts,
            "current_thresholds": self.effective_thresholds(),
            "current_session": self.current_session_insight(),
        }

    def export_manifest(self, *, variant: str = "auto", since_epoch: float | None = None) -> tuple[Path, dict[str, Any]]:
        resolved_variant = _resolve_variant(variant)
        records = self.all_feedback()
        num_heads = int(MODEL_VARIANTS[resolved_variant]["num_heads"])
        samples = []
        latest_epoch = 0.0
        for record in records:
            if str(record.get("model_variant")) != resolved_variant:
                continue
            if not bool(record.get("trusted_for_training")):
                continue
            created_at_epoch = float(record.get("created_at_epoch", 0.0) or 0.0)
            if since_epoch is not None and created_at_epoch <= since_epoch:
                continue
            known_mask = [bool(value) for value in record.get("known_mask", [])][:num_heads]
            if not any(known_mask):
                continue
            predicted = [int(value) for value in record.get("predicted_labels", [])][:num_heads]
            corrected = record.get("corrected_labels", [])[:num_heads]
            labels = []
            for index in range(num_heads):
                if not known_mask[index]:
                    labels.append(-1)
                else:
                    corrected_value = corrected[index]
                    labels.append(predicted[index] if corrected_value is None else int(corrected_value))
            samples.append(
                {
                    "feedback_id": str(record["feedback_id"]),
                    "created_at": str(record["created_at"]),
                    "created_at_epoch": created_at_epoch,
                    "clip_path": str(record["clip_path"]),
                    "labels": labels,
                    "known_mask": known_mask,
                    "rating": int(record["rating"]),
                    "trust_level": str(record["trust_level"]),
                    "head_names": list(record["head_names"])[:num_heads],
                    "class_count": int(record["class_count"]),
                    "seq_len": int(record["seq_len"]),
                    "img_size": int(record["img_size"]),
                }
            )
            latest_epoch = max(latest_epoch, created_at_epoch)

        created_at, created_at_epoch = _utcnow()
        export_mode = "incremental" if since_epoch is not None else "full"
        manifest = {
            "variant": resolved_variant,
            "created_at": created_at,
            "created_at_epoch": created_at_epoch,
            "export_mode": export_mode,
            "since_epoch": since_epoch,
            "sample_count": len(samples),
            "num_heads": num_heads,
            "class_count": DEFAULT_NUM_CLASSES,
            "latest_feedback_epoch": latest_epoch,
            "samples": samples,
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        manifest_path = self.exports_dir / f"{resolved_variant}_{export_mode}_{stamp}.json"
        _write_json(manifest_path, manifest)
        return manifest_path, manifest

    def mark_training_complete(self, *, mode: str, latest_feedback_epoch: float, candidate_path: Path) -> None:
        if mode == "online":
            self.state.last_online_trained_at = float(latest_feedback_epoch)
            self.state.last_online_candidate = str(candidate_path)
        else:
            self.state.last_offline_trained_at = float(latest_feedback_epoch)
            self.state.last_offline_candidate = str(candidate_path)
        self._persist_state()


class FeedbackClipDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], *, img_size: int, seq_len: int) -> None:
        self.samples = list(samples)
        self.img_size = int(img_size)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with np.load(sample["clip_path"]) as payload:
            frames = np.asarray(payload["frames"], dtype=np.uint8)
        if len(frames) >= self.seq_len:
            frames = frames[: self.seq_len]
        elif len(frames) > 0:
            last = frames[-1:]
            frames = np.concatenate([frames, np.repeat(last, self.seq_len - len(frames), axis=0)], axis=0)
        else:
            raise ValueError(f"Feedback clip has no frames: {sample['clip_path']}")

        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            processed.append(np.transpose(normalized, (2, 0, 1)))
        frames_tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        known_mask = torch.tensor(sample["known_mask"], dtype=torch.bool)
        return frames_tensor, labels, known_mask


def _autocast(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resolve_device() -> torch.device:
    require_cuda = os.getenv("REQUIRE_CUDA", "1").lower() in {"1", "true", "yes"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and require_cuda:
        raise RuntimeError(
            "CUDA device not available; feedback fine-tuning requires a GPU by default. "
            "Set REQUIRE_CUDA=0 to allow CPU fine-tuning."
        )
    return device


def _masked_head_loss(logits: torch.Tensor, labels: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor | None:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if known_mask.ndim == 1:
        known_mask = known_mask.unsqueeze(1)
    total_loss = None
    total_items = 0
    for head_index in range(logits.size(1)):
        head_mask = known_mask[:, head_index].bool()
        if not head_mask.any():
            continue
        head_logits = logits[head_mask, head_index, :]
        head_labels = labels[head_mask, head_index]
        head_loss = F.cross_entropy(head_logits, head_labels, reduction="mean")
        head_items = int(head_mask.sum().item())
        total_loss = head_loss * head_items if total_loss is None else total_loss + (head_loss * head_items)
        total_items += head_items
    if total_loss is None or total_items == 0:
        return None
    return total_loss / total_items


def _masked_head_accuracy(logits: torch.Tensor, labels: torch.Tensor, known_mask: torch.Tensor) -> float:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if known_mask.ndim == 1:
        known_mask = known_mask.unsqueeze(1)
    preds = torch.argmax(logits, dim=-1)
    correct = preds.eq(labels)
    valid = known_mask.bool()
    total = int(valid.sum().item())
    if total == 0:
        return 0.0
    return float(correct[valid].float().mean().item())


def _load_trainable_model(variant: str):
    from models.mobilenetv2_tcn import MobileNetV2_TCN

    resolved_variant = _resolve_variant(variant)
    model_info = MODEL_VARIANTS[resolved_variant]
    model = MobileNetV2_TCN(num_classes=DEFAULT_NUM_CLASSES, num_heads=int(model_info["num_heads"]))
    stem = model_info["stem"]
    pt_path = MODEL_DIR / f"{stem}.pt"
    ts_path = MODEL_DIR / f"{stem}.ts"
    if pt_path.exists():
        state_dict = torch.load(pt_path, map_location="cpu")
    elif ts_path.exists():
        scripted = torch.jit.load(ts_path, map_location="cpu")
        state_dict = scripted.state_dict()
    else:
        raise FileNotFoundError(f"Missing distilled checkpoint for variant '{resolved_variant}'.")
    model.load_state_dict(state_dict)
    return model, resolved_variant


def _split_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(samples) < 5:
        return samples, []
    rng = random.Random(42)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.2)))
    return shuffled[val_count:], shuffled[:val_count]


def _make_loader(samples: list[dict[str, Any]], *, img_size: int, seq_len: int, batch_size: int, shuffle: bool):
    dataset = FeedbackClipDataset(samples, img_size=img_size, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate_feedback_model(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_accuracy = 0.0
    with torch.no_grad():
        for frames, labels, known_mask in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            known_mask = known_mask.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(frames)
                loss = _masked_head_loss(logits, labels, known_mask)
            if loss is None:
                continue
            total_loss += float(loss.item())
            total_accuracy += _masked_head_accuracy(logits, labels, known_mask)
            total_batches += 1
    if total_batches == 0:
        return 0.0, 0.0
    return total_loss / total_batches, total_accuracy / total_batches


def _train_candidate(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[Path, Path, dict[str, Any]]:
    samples = list(manifest.get("samples", []))
    if not samples:
        raise RuntimeError("No trusted feedback samples available for training.")

    model, resolved_variant = _load_trainable_model(manifest.get("variant", "auto"))
    device = _resolve_device()
    model = model.to(device)

    seq_len = int(samples[0].get("seq_len", DEFAULT_SEQ_LEN))
    img_size = int(samples[0].get("img_size", DEFAULT_IMG_SIZE))
    train_samples, val_samples = _split_samples(samples) if mode == "offline" else (samples, [])
    train_loader = _make_loader(train_samples, img_size=img_size, seq_len=seq_len, batch_size=batch_size, shuffle=True)
    val_loader = (
        _make_loader(val_samples, img_size=img_size, seq_len=seq_len, batch_size=batch_size, shuffle=False)
        if val_samples
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_metric = -1.0
    best_state = None

    for _ in range(max(1, int(epochs))):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0
        for frames, labels, known_mask in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            known_mask = known_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(frames)
                loss = _masked_head_loss(logits, labels, known_mask)
            if loss is None:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_accuracy += _masked_head_accuracy(logits.detach(), labels, known_mask)
            total_batches += 1

        train_loss = total_loss / max(1, total_batches)
        train_acc = total_accuracy / max(1, total_batches)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = _evaluate_feedback_model(model, val_loader, device)
        else:
            val_loss, val_acc = 0.0, train_acc
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        metric = val_acc if val_loader is not None else train_acc
        if metric >= best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Feedback fine-tuning did not produce a trainable checkpoint.")
    model.load_state_dict(best_state)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    variant_dir = CANDIDATES_DIR / resolved_variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{resolved_variant}_{mode}_{stamp}"
    checkpoint_path = variant_dir / f"{stem}.pt"
    metrics_path = variant_dir / f"{stem}_metrics.json"
    manifest_copy_path = variant_dir / f"{stem}_manifest.json"
    torch.save(model.state_dict(), checkpoint_path)
    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save(variant_dir / f"{stem}.ts")
        model.to(device)
    except Exception:
        pass

    metrics = {
        "mode": mode,
        "variant": resolved_variant,
        "source_manifest": str(manifest_path),
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "history": history,
        "best_metric": best_metric,
        "latest_feedback_epoch": float(manifest.get("latest_feedback_epoch", 0.0) or 0.0),
    }
    _write_json(metrics_path, metrics)
    _write_json(manifest_copy_path, manifest)
    return checkpoint_path, metrics_path, metrics


def _summarize_command(args) -> int:
    manager = FeedbackManager(start_new_session=False)
    print(json.dumps(manager.summarize_feedback(), indent=2))
    return 0


def _export_command(args) -> int:
    manager = FeedbackManager(start_new_session=False)
    manifest_path, manifest = manager.export_manifest(variant=args.variant)
    print(f"Exported {manifest['sample_count']} trusted samples to {manifest_path}")
    return 0


def _train_command(args, *, mode: str) -> int:
    manager = FeedbackManager(start_new_session=False)
    since_epoch = None
    if mode == "online" and args.since_last:
        since_epoch = float(manager.state.last_online_trained_at or 0.0)
    manifest_path, manifest = manager.export_manifest(variant=args.variant, since_epoch=since_epoch)
    if int(manifest["sample_count"]) == 0:
        print("No trusted feedback samples matched the requested training set.")
        return 0

    checkpoint_path, metrics_path, metrics = _train_candidate(
        manifest=manifest,
        manifest_path=manifest_path,
        mode=mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    manager.mark_training_complete(
        mode=mode,
        latest_feedback_epoch=float(metrics.get("latest_feedback_epoch", 0.0) or 0.0),
        candidate_path=checkpoint_path,
    )
    print(f"Saved {mode} candidate checkpoint to {checkpoint_path}")
    print(f"Saved {mode} candidate metrics to {metrics_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="User-in-the-loop feedback capture and fine-tuning utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize collected feedback and active thresholds.")
    summarize_parser.set_defaults(func=_summarize_command)

    export_parser = subparsers.add_parser("export", help="Export trusted feedback samples into a training manifest.")
    export_parser.add_argument("--variant", choices=["engagement", "multiaffect", "auto"], default="auto")
    export_parser.set_defaults(func=_export_command)

    offline_parser = subparsers.add_parser("train_offline", help="Run offline fine-tuning on trusted feedback.")
    offline_parser.add_argument("--variant", choices=["engagement", "multiaffect", "auto"], default="auto")
    offline_parser.add_argument("--epochs", type=int, default=4)
    offline_parser.add_argument("--batch_size", type=int, default=2)
    offline_parser.add_argument("--lr", type=float, default=1e-5)
    offline_parser.set_defaults(func=lambda args: _train_command(args, mode="offline"))

    online_parser = subparsers.add_parser("train_online", help="Run short incremental fine-tuning on trusted feedback.")
    online_parser.add_argument("--variant", choices=["engagement", "multiaffect", "auto"], default="auto")
    online_parser.add_argument("--epochs", type=int, default=2)
    online_parser.add_argument("--batch_size", type=int, default=2)
    online_parser.add_argument("--lr", type=float, default=1e-5)
    online_parser.add_argument("--since-last", action="store_true", help="Use only trusted samples newer than the last online job.")
    online_parser.set_defaults(func=lambda args: _train_command(args, mode="online"))

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
