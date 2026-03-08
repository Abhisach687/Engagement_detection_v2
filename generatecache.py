import argparse
from pathlib import Path

from config import FRAMES_DIR, LABELS_DIR, LMDB_CACHE_PATH, LMDB_MAP_SIZE, SEQ_LEN, IMG_SIZE
from data.dataset_loader import load_labels
from data.cache import build_frame_cache


def main():
    parser = argparse.ArgumentParser(description="Build LMDB cache for frame sequences (and HOG features).")
    parser.add_argument("--split", choices=["Train", "Validation", "Test"], default="Train", help="Dataset split to cache")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Number of frames per clip to store")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Square image size for resized frames")
    parser.add_argument("--map_size", type=int, default=LMDB_MAP_SIZE, help="LMDB map size in bytes")
    parser.add_argument("--cache_path", type=str, default=str(LMDB_CACHE_PATH), help="Output LMDB path")
    parser.add_argument("--no_hog", action="store_true", help="Skip storing HOG features in the cache")
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Send tensors through CUDA during preprocessing (no impact on LMDB contents, may slightly speed transforms).",
    )
    args = parser.parse_args()

    df = load_labels(LABELS_DIR / f"{args.split}Labels.csv")
    cache_path = Path(args.cache_path)
    build_frame_cache(
        df=df,
        frames_dir=FRAMES_DIR,
        cache_path=cache_path,
        seq_len=args.seq_len,
        img_size=args.img_size,
        map_size=args.map_size,
        store_hog=not args.no_hog,
        split=args.split,
        use_cuda=args.use_cuda,
    )
    print(f"LMDB cache written to {cache_path}")


if __name__ == "__main__":
    main()
