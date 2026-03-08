import io
from pathlib import Path
from typing import Optional, Tuple

import lmdb
import numpy as np
from tqdm import tqdm

from config import SEQ_LEN, IMG_SIZE, LMDB_MAP_SIZE, FEATURE_CACHE_DIR
from features.feature_pipeline import process_video
from features.frame_dataset import build_transform, load_frame_tensor


def _serialize(record: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **record)
    return buf.getvalue()


def _deserialize(blob: bytes) -> dict:
    buf = io.BytesIO(blob)
    return dict(np.load(buf, allow_pickle=False))


def build_frame_cache(
    df,
    frames_dir: Path,
    cache_path: Path,
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
    map_size: int = LMDB_MAP_SIZE,
    store_hog: bool = True,
    commit_interval: int = 200,
    split: str = "Train",
    use_cuda: bool = False,
) -> None:
    """
    Precompute and store normalized frame tensors (and optional HOG features)
    into an LMDB at `cache_path`. Each key is the clip id.
    """
    from config import DEVICE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build_frame_cache] using device: {DEVICE} (use_cuda={use_cuda})")
    env = lmdb.open(
        str(cache_path),
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
    )

    transform = build_transform(img_size, augment=False)
    txn = env.begin(write=True)

    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Caching clips")):
        clip = row.ClipID
        label = int(row.Engagement)
        folder = Path(frames_dir) / split / clip
        frame_paths = sorted(folder.glob("frame_*.jpg"))
        if len(frame_paths) == 0:
            continue

        frames = load_frame_tensor(frame_paths, transform, seq_len)
        # optional torch CUDA round-trip; most of the work is still CPU/I/O
        if use_cuda and DEVICE.type == "cuda":
            frames = frames.to(DEVICE)
            # any GPU-side ops could go here (none currently)
            frames = frames.cpu()
        frames = frames.numpy().astype(np.float16)
        record = {"frames": frames, "label": np.int64(label)}

        if store_hog:
            # Reuse the existing HOG pipeline; this will also drop .npy files to FEATURE_CACHE_DIR
            hog_feat = process_video(folder, FEATURE_CACHE_DIR, num_frames=min(seq_len, 30))
            record["hog"] = hog_feat.astype(np.float32)

        key = f"{split}:{clip}".encode()
        txn.put(key, _serialize(record))

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.sync()
    env.close()


def load_from_cache(
    cache_path: Path, clip_id: str, split: str = None
) -> Optional[Tuple[np.ndarray, int, Optional[np.ndarray]]]:
    """
    Convenience loader for ad-hoc reads outside of Dataset classes.
    Returns (frames, label, hog_feature or None).
    """
    if not cache_path.exists():
        return None
    env = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        key = f"{split}:{clip_id}" if split else clip_id
        raw = txn.get(key.encode())
        if raw is None and split:
            # backward compatibility: try legacy key without split
            raw = txn.get(clip_id.encode())
    env.close()
    if raw is None:
        return None
    record = _deserialize(raw)
    frames = record["frames"]
    label = int(record["label"])
    hog = record.get("hog")
    return frames, label, hog
