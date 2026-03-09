import json
import os
import inspect
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

from config import (
    FRAMES_DIR,
    LABELS_DIR,
    FEATURE_CACHE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    LMDB_CACHE_PATH,
)
from data.dataset_loader import load_labels
from data.cache import load_from_cache
from features.feature_pipeline import process_video
from features.hog_extractor import DEFAULT_HOG_SIZE, _expected_hog_length


XGB_FORCE_CPU = os.getenv("XGB_FORCE_CPU", "0").lower() in {"1", "true", "yes"}
EXPECTED_HOG_LEN = _expected_hog_length(DEFAULT_HOG_SIZE)

# tqdm bar positions to keep nesting stable across prints
TRIAL_BAR_POS = 0
BOOST_BAR_POS = 1
HOG_BAR_POS = 2


def _xgb_has_cuda():
    # GPU builds export this symbol; CPU builds typically do not.
    return hasattr(xgb.core._LIB, "XGDeviceQuantileDMatrixCreate")


def _xgb_device_params():
    if XGB_FORCE_CPU or not _xgb_has_cuda():
        return {"tree_method": "hist"}
    return {"tree_method": "hist", "device": "cuda"}


XGB_GPU_PARAMS = _xgb_device_params()
HAS_FIT_CALLBACKS = "callbacks" in inspect.signature(xgb.XGBClassifier().fit).parameters


class TQDMBoosterCallback(xgb.callback.TrainingCallback):
    """Progress bar over boosting iterations for a single trial."""

    def __init__(self, total_rounds: int, desc: str):
        self.total_rounds = total_rounds
        self.desc = desc
        self._pbar = None

    def before_training(self, model):
        self._pbar = tqdm(
            total=self.total_rounds,
            desc=self.desc,
            leave=True,  # keep visible alongside trial bar
            dynamic_ncols=True,
            position=BOOST_BAR_POS,
            mininterval=0.1,
        )
        return None

    def after_iteration(self, model, epoch: int, evals_log):
        if self._pbar:
            self._pbar.update(1)
        return False

    def after_training(self, model):
        if self._pbar:
            self._pbar.close()
        return None


def _fit_with_progress(
    model: xgb.XGBClassifier,
    X_train,
    y_train,
    cb: TQDMBoosterCallback,
    eval_set=None,
    early_stopping_rounds: int = None,
):
    """Fit model with callbacks and optional early stopping; degrade gracefully otherwise."""
    fit_kwargs = {"verbose": False}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["eval_metric"] = "mlogloss"
        if early_stopping_rounds:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    if HAS_FIT_CALLBACKS:
        try:
            model.fit(X_train, y_train, callbacks=[cb], **fit_kwargs)
            return
        except TypeError:
            # Older xgboost despite signature mismatch; fall back
            pass
    # Fallback: no per-round progress, but ensure training runs
    if cb:
        cb.before_training(model)
    model.fit(X_train, y_train, **fit_kwargs)
    if cb:
        cb.after_training(model)


def _fit_with_gpu_fallback(
    params: dict,
    X_train,
    y_train,
    desc: str,
    eval_set=None,
    early_stopping_rounds: int = None,
):
    """
    Try GPU params first; if unsupported,
    fall back to CPU-friendly params. If GPU is known to be unavailable (env
    XGB_FORCE_CPU or CPU-only build), skip straight to CPU to avoid noisy logs.
    """
    params_gpu = params.copy()
    wants_gpu = params_gpu.get("device") == "cuda"
    if not wants_gpu:
        # Already CPU params
        model = xgb.XGBClassifier(**params_gpu)
        cb = TQDMBoosterCallback(total_rounds=params_gpu["n_estimators"], desc=desc + " [cpu]")
        _fit_with_progress(
            model,
            X_train,
            y_train,
            cb,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, params_gpu, "cpu"

    model = xgb.XGBClassifier(**params_gpu)
    cb = TQDMBoosterCallback(total_rounds=params_gpu["n_estimators"], desc=desc)
    try:
        _fit_with_progress(
            model,
            X_train,
            y_train,
            cb,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, params_gpu, "gpu"
    except (ValueError, xgb.core.XGBoostError) as e:
        err = str(e)
        if not any(token in err for token in ("gpu_hist", "device", "cuda")):
            raise
        params_cpu = {"tree_method": "hist"}
        params_cpu.update({k: v for k, v in params_gpu.items() if k not in {"tree_method", "device"}})
        print(f"[xgb] GPU unsupported in this xgboost build; falling back to CPU hist. ({e})")
        model = xgb.XGBClassifier(**params_cpu)
        cb = TQDMBoosterCallback(total_rounds=params_cpu["n_estimators"], desc=desc + " [cpu]")
        _fit_with_progress(
            model,
            X_train,
            y_train,
            cb,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, params_cpu, "cpu"


def _load_hog_features(num_frames: int = 30, cache_mode: str = "prefer") -> Tuple[np.ndarray, np.ndarray]:
    df = load_labels(LABELS_DIR / "TrainLabels.csv")
    X, y = [], []
    missing_frames = 0
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="[xgb] build HOG",
        leave=False,
        position=HOG_BAR_POS,
        dynamic_ncols=True,
    ):
        clip = row["ClipID"]
        folder = FRAMES_DIR / "Train" / clip
        if not folder.exists():
            missing_frames += 1
            continue
        feat = None
        if cache_mode != "off":
            cached = load_from_cache(LMDB_CACHE_PATH, clip, split="Train")
            if cached and cached[2] is not None and cached[2].shape[-1] == EXPECTED_HOG_LEN:
                feat = cached[2]
        if feat is None:
            if cache_mode == "force":
                raise RuntimeError(f"Cache miss for clip {clip} in split Train")
            feat = process_video(
                folder,
                FEATURE_CACHE_DIR,
                num_frames=num_frames,
                target_size=DEFAULT_HOG_SIZE,
            )
        X.append(feat)
        y.append(int(row["Engagement"]))
    if missing_frames:
        print(f"[xgb] skipped {missing_frames} clips with missing frames/cache")
    return np.array(X), np.array(y)


def _prepare_data(num_frames: int = 30, cache_mode: str = "prefer") -> Dict[str, np.ndarray]:
    """Load + preprocess once and reuse across Optuna trials to avoid 40x I/O."""
    X, y = _load_hog_features(num_frames, cache_mode=cache_mode)
    if len(X) == 0:
        raise RuntimeError("No training clips found (missing frames/cache).")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    # Cast to float32 to halve memory and speed up CPU-bound training
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    smote_warning = None
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        smote_warning = str(e)
        print(f"[xgb] SMOTE skipped for all trials: {e}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "smote_warning": smote_warning,
    }


def _plot_feature_importance(model: xgb.XGBClassifier, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=20, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion_matrix(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def objective(trial: optuna.Trial, prepared: Dict[str, np.ndarray] = None, total_trials: int = None):
    try:
        data = prepared or _prepare_data()
        X_train = data["X_train"]
        X_val = data["X_val"]
        y_train = data["y_train"]
        y_val = data["y_val"]

        params = {
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            # Keep boosting short for faster trials/debugging
            "n_estimators": trial.suggest_int("n_estimators", 8, 8),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        params.update(XGB_GPU_PARAMS)

        model, used_params, device_used = _fit_with_gpu_fallback(
            params,
            X_train,
            y_train,
            desc=f"[xgb] trial {trial.number + 1}/{total_trials or '?'}",
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
        )
        trial.set_user_attr("xgb_device", device_used)

        smote_warning = data.get("smote_warning")
        if smote_warning:
            trial.set_user_attr("smote_warning", smote_warning)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        trial.set_user_attr("acc", acc)
        return acc
    except Exception as e:
        # Mark trial as failed but keep study running; message shows in Optuna logs
        trial.set_user_attr("failure", str(e))
        print(f"[xgb][trial {trial.number}] failed: {e}")
        raise optuna.TrialPruned(f"failed: {e}")


def run_study(
    n_trials: int = 8,
    study_path: Path = MODEL_DIR / "xgb_hog_study.db",
    resume: bool = True,
    cache_mode: str = "prefer",
):
    if study_path and study_path.exists() and not resume:
        study_path.unlink()
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_hog",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
    )
    prepared = _prepare_data(cache_mode=cache_mode)
    progress = tqdm(total=n_trials, desc="[xgb] trials", leave=True, position=TRIAL_BAR_POS, dynamic_ncols=True)

    def _on_trial_end(study, trial):
        progress.update(1)

    def _objective(trial: optuna.Trial):
        return objective(trial, prepared=prepared, total_trials=n_trials)

    try:
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_on_trial_end])
    finally:
        progress.close()
    return study


def train_best_from_study(study: optuna.Study, cache_mode: str = "prefer"):
    X, y = _load_hog_features(cache_mode=cache_mode)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    best_params = {**study.best_params}
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_estimators": 8,
        }
    )
    best_params.update(XGB_GPU_PARAMS)
    model, best_params, device_used = _fit_with_gpu_fallback(
        best_params,
        X_train,
        y_train,
        desc="[xgb] final train",
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=5,
    )

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds)

    # Persist artifacts
    model_out = MODEL_DIR / "xgb_hog.json"
    scaler_out = MODEL_DIR / "xgb_hog_scaler.joblib"
    importance_out = MODEL_DIR / "xgb_hog_feature_importance.png"
    metrics_out = MODEL_DIR / "xgb_hog_metrics.json"
    cm_out = MODEL_DIR / "xgb_hog_confusion.png"

    model.save_model(model_out)
    joblib.dump(scaler, scaler_out)
    _plot_feature_importance(model, importance_out)
    _plot_confusion_matrix(cm, cm_out)

    with open(metrics_out, "w") as f:
        json.dump(
            {"val_accuracy": acc, "classification_report": report, "confusion_matrix": cm.tolist()},
            f,
            indent=2,
        )

    print(f"Saved model to {model_out}")
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    study = run_study()
    train_best_from_study(study)
