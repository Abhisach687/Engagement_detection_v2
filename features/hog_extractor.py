import cv2
import numpy as np
from skimage.feature import hog


# Default square side (pixels) to downscale frames before HOG extraction.
# Keeping this small dramatically reduces feature dimensionality and training time.
DEFAULT_HOG_SIZE = 224


def _expected_hog_length(side: int) -> int:
    """
    Compute the flattened HOG length for a square image of dimension `side`
    using the same HOG parameters below. Used to detect stale cache entries.
    """
    cells = side // 8
    blocks = max(cells - 1, 0)
    return blocks * blocks * 4 * 9  # (2x2 cells per block) * orientations


def extract_hog(image_path, target_size: int = DEFAULT_HOG_SIZE):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    )
    return features.astype(np.float32)
