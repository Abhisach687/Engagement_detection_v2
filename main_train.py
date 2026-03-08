import argparse
from config import MODEL_DIR


def run_xgb(resume: bool = True, cache_mode: str = "prefer", force: bool = False):
    from training.train_xgb import run_study, train_best_from_study
    from pathlib import Path

    model_path = MODEL_DIR / "xgb_hog.json"
    if model_path.exists() and not force:
        print(f"[xgb] Found existing model at {model_path}; skipping XGB training (use --retrain_xgb to force).")
        return

    study = run_study(resume=resume, cache_mode=cache_mode)
    train_best_from_study(study, cache_mode=cache_mode)


def run_lstm(model_type: str, resume: bool = True, cache_mode: str = "prefer", force: bool = False):
    from training.train_lstm import run_study, train_best_from_study
    from pathlib import Path

    weight_path = MODEL_DIR / f"mobilenetv2_{model_type}.pt"
    if weight_path.exists() and not force:
        print(f"[{model_type}] Found existing model at {weight_path}; skipping training (use --retrain_{model_type} to force).")
        return

    study = run_study(
        model_type=model_type,
        n_trials=20,
        study_path=MODEL_DIR / f"{model_type}_study.db",
        resume=resume,
        cache_mode=cache_mode,
    )
    train_best_from_study(model_type, study, cache_mode=cache_mode)


def run_tcn(cache_mode: str = "prefer"):
    from training.train_tcn import train_tcn

    train_tcn(cache_mode=cache_mode)


def run_distill(alpha: float, temperature: float, cache_mode: str = "prefer", force: bool = False):
    from training.distill_tcn import distill
    from pathlib import Path

    distilled_path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt"
    if distilled_path.exists() and not force:
        print(f"[distill] Found existing distilled model at {distilled_path}; skipping (use --retrain_distill to force).")
        return

    distill(alpha=alpha, temperature=temperature, cache_mode=cache_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train engagement models")
    parser.add_argument(
        "task",
        choices=["xgb", "xgboost", "lstm", "bilstm", "tcn", "distill", "all"],
        help="Which training pipeline to run (alias: xgboost -> xgb)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight (distill only)")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature (distill only)")
    args = parser.parse_args()

    if args.task in {"xgb", "xgboost"}:
        run_xgb()
    elif args.task in {"lstm", "bilstm"}:
        run_lstm(args.task)
    elif args.task == "tcn":
        run_tcn()
    elif args.task == "all":
        run_xgb()
        run_lstm("lstm")
        run_lstm("bilstm")
        run_tcn()
        run_distill(alpha=args.alpha, temperature=args.temperature)
    else:
        run_distill(alpha=args.alpha, temperature=args.temperature)
