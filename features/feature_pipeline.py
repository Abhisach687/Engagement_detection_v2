import numpy as np
from pathlib import Path
from .frame_selector import select_top_frames
from .hog_extractor import extract_hog, DEFAULT_HOG_SIZE, _expected_hog_length


def process_video(video_folder: Path, cache_dir: Path, num_frames: int = 30, target_size: int = DEFAULT_HOG_SIZE):
    """
    Extract HOG features for a video (mean of top frames).
    Cached to .npy for speed. If an existing cache was created with a different
    resize setting (inferred via feature length), it will be refreshed.
    """
    cache_file = cache_dir / f"{video_folder.name}.npy"
    if cache_file.exists():
        cached = np.load(cache_file)
        if target_size is None:
            return cached
        expected_len = _expected_hog_length(target_size)
        if cached.shape[-1] == expected_len:
            return cached
        # Stale cache with mismatched HOG length — rebuild below.

    frames = select_top_frames(video_folder, num_frames)
    feats = [extract_hog(f, target_size=target_size) for f in frames]
    if len(feats) == 0:
        raise ValueError(f"No frames found in {video_folder}")

    video_feature = np.mean(np.stack(feats), axis=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, video_feature)
    return video_feature
