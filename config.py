from pathlib import Path
import os

# Root paths (repo-relative)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"

MODEL_DIR = BASE_DIR / "models"
FEATURE_CACHE_DIR = MODEL_DIR / "hog_features"  # HOG npy cache for XGB
CACHE_DIR = BASE_DIR / "cache"
LMDB_CACHE_PATH = CACHE_DIR / "frames.lmdb"  # LMDB for preprocessed frame tensors
LOG_DIR = BASE_DIR / "logs"

# Create folders that training/app expect
for _p in [MODEL_DIR, FEATURE_CACHE_DIR, CACHE_DIR, LOG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

import torch

# Hard require CUDA by default; opt out with REQUIRE_CUDA=0
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1").lower() in {"1", "true", "yes"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA device not available; training/inference requires a GPU. "
            "If you intentionally want CPU, set REQUIRE_CUDA=0."
        )
    else:
        print("[config] CUDA not available; falling back to CPU by user override.")
else:
    # Enable common performance flags when a GPU is present
    torch.backends.cudnn.benchmark = True

# Data / model hyperparams
SEQ_LEN = 30
IMG_SIZE = 224
NUM_CLASSES = 4

# Training defaults (overridden by tuners)
BATCH_SIZE = 8
TCN_BATCH_SIZE = max(1, int(os.getenv("TCN_BATCH_SIZE", str(min(BATCH_SIZE, 4)))))
DISTILL_BATCH_SIZE = max(1, int(os.getenv("DISTILL_BATCH_SIZE", str(TCN_BATCH_SIZE))))
TCN_NUM_WORKERS = max(0, int(os.getenv("TCN_NUM_WORKERS", "2")))
DISTILL_NUM_WORKERS = max(0, int(os.getenv("DISTILL_NUM_WORKERS", str(TCN_NUM_WORKERS))))
BACKBONE_FRAME_CHUNK = max(0, int(os.getenv("BACKBONE_FRAME_CHUNK", "32")))
TCN_MAX_EPOCHS = 8
TCN_EPOCHS = min(max(1, int(os.getenv("TCN_EPOCHS", str(TCN_MAX_EPOCHS)))), TCN_MAX_EPOCHS)
DISTILL_MAX_EPOCHS = 8
DISTILL_EPOCHS = min(max(1, int(os.getenv("DISTILL_EPOCHS", str(DISTILL_MAX_EPOCHS)))), DISTILL_MAX_EPOCHS)
DISTILL_MAX_TRIALS = 8
DISTILL_SEARCH_TRIALS = min(
    max(1, int(os.getenv("DISTILL_SEARCH_TRIALS", str(DISTILL_MAX_TRIALS)))),
    DISTILL_MAX_TRIALS,
)
EPOCHS = 25
LR = 1e-4

# Caching
LMDB_MAP_SIZE = 32 * 1024**3  # 32GB default map size; adjust via CLI if needed

RANDOM_STATE = 42
