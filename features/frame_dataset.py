import io
from pathlib import Path
from typing import List, Optional

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(img_size: int, augment: bool = False):
    tfs = [transforms.Resize((img_size, img_size))]
    if augment:
        tfs.extend(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(),
            ]
        )
    tfs.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(tfs)


def pad_or_trim(frames: List[Path], seq_len: int) -> List[Path]:
    if len(frames) >= seq_len:
        return frames[:seq_len]
    if len(frames) == 0:
        raise ValueError("No frames found for sample.")
    last = frames[-1]
    return frames + [last] * (seq_len - len(frames))


def load_frame_tensor(frame_paths: List[Path], transform, seq_len: int):
    frame_paths = pad_or_trim(frame_paths, seq_len)
    imgs = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs)  # (T, C, H, W)


class VideoFrameDataset(Dataset):
    """
    Loads pre-extracted frames for a clip and returns a fixed-length tensor
    of shape (T, 3, H, W). If `cache_path` is provided and contains an LMDB
    built by `generatecache.py`, it will read preprocessed tensors from there.
    """

    def __init__(
        self,
        df,
        frames_dir: Path,
        seq_len: int = 30,
        img_size: int = 224,
        augment: bool = False,
        return_clip_id: bool = False,
        split: str = "Train",
        cache_path: Optional[Path] = None,
        force_cache: bool = False,
        no_cache: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.seq_len = seq_len
        self.return_clip_id = return_clip_id
        self.split = split
        self.force_cache = force_cache
        self.no_cache = no_cache

        # When reading from cache we skip stochastic augmentations
        use_augment = augment and (cache_path is None or no_cache)
        self.transform = build_transform(img_size, augment=use_augment)

        self.cache_env = None
        self.cache_path = Path(cache_path) if cache_path and Path(cache_path).exists() else None
        self._prune_missing_clips()

    def _prune_missing_clips(self):
        # Drop rows where neither cache nor frame folder exists to avoid runtime crashes
        valid_idx = []
        missing = []
        for idx, clip in enumerate(self.df["ClipID"]):
            if self._has_frames_or_cache(clip):
                valid_idx.append(idx)
            else:
                missing.append(clip)
        if missing:
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            print(f"[VideoFrameDataset] skipped {len(missing)} clips with no frames/cache in split={self.split}")

    def _has_frames_or_cache(self, clip: str) -> bool:
        if self.force_cache:
            env = self._ensure_cache_env()
            if env is None:
                return False
            key = f"{self.split}:{clip}".encode()
            with env.begin(write=False) as txn:
                if txn.get(key) is not None:
                    return True
                # legacy fallback without split
                return txn.get(clip.encode()) is not None

        folder = self.frames_dir / self.split / clip
        if any(folder.glob("frame_*.jpg")):
            return True
        env = self._ensure_cache_env()
        if env is None:
            return False
        key = f"{self.split}:{clip}".encode()
        with env.begin(write=False) as txn:
            if txn.get(key) is not None:
                return True
            # legacy fallback without split
            return txn.get(clip.encode()) is not None

    def _ensure_cache_env(self):
        # Lazily open LMDB inside worker process to avoid pickling lmdb.Environment
        if self.cache_env is None and self.cache_path is not None:
            self.cache_env = lmdb.open(
                str(self.cache_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self.cache_env

    def __getstate__(self):
        state = self.__dict__.copy()
        # lmdb.Environment objects are not picklable; reopen per worker
        state["cache_env"] = None
        return state

    def __len__(self):
        return len(self.df)

    def _load_from_cache(self, clip: str):
        env = self._ensure_cache_env()
        if env is None:
            return None
        key = f"{self.split}:{clip}".encode()
        with env.begin(write=False) as txn:
            raw = txn.get(key)
            if raw is None:
                raw = txn.get(clip.encode())  # legacy
        if raw is None:
            return None
        buf = io.BytesIO(raw)
        data = np.load(buf)
        frames = torch.from_numpy(data["frames"]).float()
        label = int(data["label"])
        return frames, label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = row["ClipID"]

        cached = None if self.no_cache else self._load_from_cache(clip)
        if cached is not None:
            frames, label = cached
        else:
            folder = self.frames_dir / self.split / clip
            frame_paths = sorted(folder.glob("frame_*.jpg"))
            frames = load_frame_tensor(frame_paths, self.transform, self.seq_len)
            label = int(row["Engagement"])
            if self.force_cache:
                raise RuntimeError(f"Cache miss for clip {clip} in split {self.split}")

        if self.return_clip_id:
            return frames, torch.tensor(label, dtype=torch.long), clip
        return frames, torch.tensor(label, dtype=torch.long)
