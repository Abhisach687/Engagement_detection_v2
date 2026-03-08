import argparse

from main_train import run_xgb, run_lstm, run_tcn, run_distill


def main():
    parser = argparse.ArgumentParser(description="Train all engagement models sequentially.")
    parser.add_argument("--skip_xgb", action="store_true", help="Skip HOG+XGBoost")
    parser.add_argument("--skip_seq", action="store_true", help="Skip LSTM/BiLSTM")
    parser.add_argument("--skip_tcn", action="store_true", help="Skip vanilla TCN training")
    parser.add_argument("--skip_distill", action="store_true", help="Skip knowledge distillation step")
    parser.add_argument(
        "--fresh_studies",
        action="store_true",
        help="Delete any existing Optuna study DBs and start fresh (disables resume)",
    )
    parser.add_argument(
        "--retrain_xgb",
        action="store_true",
        help="Force re-training XGB even if artifacts already exist",
    )
    parser.add_argument(
        "--retrain_lstm",
        action="store_true",
        help="Force re-training LSTM even if artifacts already exist",
    )
    parser.add_argument(
        "--retrain_bilstm",
        action="store_true",
        help="Force re-training BiLSTM even if artifacts already exist",
    )
    parser.add_argument(
        "--retrain_distill",
        action="store_true",
        help="Force re-training distilled TCN even if artifacts already exist",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--force_cache", action="store_true", help="Fail if LMDB cache entry missing")
    cache_group.add_argument("--no_cache", action="store_true", help="Bypass LMDB cache (debug)")
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature")
    args = parser.parse_args()

    cache_mode = "prefer"
    if args.force_cache:
        cache_mode = "force"
    elif args.no_cache:
        cache_mode = "off"

    if not args.skip_xgb:
        run_xgb(resume=not args.fresh_studies, cache_mode=cache_mode, force=args.retrain_xgb)
    if not args.skip_seq:
        run_lstm(
            "lstm",
            resume=not args.fresh_studies,
            cache_mode=cache_mode,
            force=args.retrain_lstm,
        )
        run_lstm(
            "bilstm",
            resume=not args.fresh_studies,
            cache_mode=cache_mode,
            force=args.retrain_bilstm,
        )
    if not args.skip_tcn:
        run_tcn(cache_mode=cache_mode)
    if not args.skip_distill:
        run_distill(
            alpha=args.alpha,
            temperature=args.temperature,
            cache_mode=cache_mode,
            force=args.retrain_distill,
        )


if __name__ == "__main__":
    main()
