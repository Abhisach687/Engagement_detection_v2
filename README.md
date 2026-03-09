# Engagement Detection Project

Complete DAiSEE engagement pipeline: HOG + XGBoost baseline, MobileNetV2 + LSTM/BiLSTM/TCN, single-task and multi-affect distillation, LMDB caching, and a desktop webcam app. Project is AI-assisted (ChatGPT/Codex).

## Quick Start
1. Create and activate a virtual environment (Python 3.11):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   $env:REQUIRE_CUDA = "1"   # require CUDA for training/inference
   ```
2. Place DAiSEE data:
   ```text
   DAiSEE/ExtractedFrames/<Split>/<ClipID>/frame_XXXX.jpg
   DAiSEE/Labels/<Split>Labels.csv   # Train / Validation / Test
   ```
3. Build LMDB caches:
   ```powershell
   python generatecache.py --split Train
   python generatecache.py --split Validation
   python generatecache.py --split Test

   # optional GPU preprocessing
   python generatecache.py --split Train --use_cuda
   ```
   `Validation` cache is strongly recommended if you plan to run distillation or validation with `--force_cache`.

## Train
Default behavior resumes existing Optuna studies and prefers LMDB cache when available.

```powershell
python trainmodels.py
python trainmodels.py --fresh_studies
python trainmodels.py --force_cache
python trainmodels.py --no_cache
python trainmodels.py --retrain_xgb
python trainmodels.py --retrain_lstm
python trainmodels.py --retrain_bilstm
python trainmodels.py --retrain_distill

# selective runs
python trainmodels.py --fresh_studies --force_cache --skip_seq --skip_tcn --skip_distill   # XGB only
python trainmodels.py --skip_xgb --skip_tcn --skip_distill                                  # LSTM + BiLSTM only
python trainmodels.py --skip_xgb --skip_seq --skip_distill                                  # TCN only
python trainmodels.py --skip_xgb --skip_seq --skip_tcn                                      # Distill only
```

Notes:
- `--force_cache` is strict for training splits.
- Distillation validation uses cache when present, but falls back to frame folders if Validation LMDB entries are missing.
- Progress bars are nested as `trial -> boost/epoch -> batch -> val`.

## Direct Training Commands
Use `main_train.py` when you want a specific pipeline stage instead of the broader `trainmodels.py` wrapper.

```powershell
# full single-task pipeline
.\.venv\Scripts\python.exe main_train.py all

# full multi-affect pipeline
.\.venv\Scripts\python.exe main_train.py all_multi

# full multi-affect pipeline with explicit KD settings
.\.venv\Scripts\python.exe main_train.py all_multi --alpha 0.48 --temperature 4.48

# only distill the 4-head TCN student after the multi-affect teachers exist
.\.venv\Scripts\python.exe main_train.py distill_multi
```

Multi-affect notes:
- `all_multi` trains the LSTM multi-affect teacher, the BiLSTM multi-affect teacher, then the 4-head TCN student.
- `distill_multi` now warm-starts from `models/mobilenetv2_tcn_distilled.pt` when that single-task student exists, so the shared TCN trunk does not start from scratch.
- If you omit `--alpha` and `--temperature`, the multi-affect distillation stage reuses the current saved distilled student settings from `models/mobilenetv2_tcn_distilled_metrics.json` when available.

## Evaluate
```powershell
python testmodel.py --split Validation
python testmodel.py --split Test
python evaluatemodels.py
```

## Inference App
```powershell
.\.venv\Scripts\python.exe app.py
```

App behavior:
- The app auto-loads the multi-affect distilled student if `models/mobilenetv2_tcn_multiaffect_distilled.*` exists.
- Otherwise it falls back to the original engagement-only distilled student.
- If the required ONNX export is missing, the app tries to export it automatically from the saved PyTorch or TorchScript checkpoint.
- The GUI shows the active device (`cuda` or `cpu`) and the active model variant.
- The desktop UI is now a fixed single-screen dashboard sized for laptop monitoring, with a large live preview, a primary engagement decision panel, and a compact bottom signal band.
- When the multi-affect ONNX sidecar metadata is present, the app derives the secondary affect tiles from `head_names` and promotes the strongest non-engagement signal only when its `High + Very High` band reaches at least `0.55`.

## Outputs
- Models and metrics: `models/`
- Training curves and logs: `logs/`
- HOG cache: `models/hog_features/`
- Frame LMDB: `cache/frames.lmdb`
  Keys are split-namespaced: `Train:clip`, `Validation:clip`, `Test:clip`
- Study databases: `models/*.db`
- Multi-affect distilled student: `models/mobilenetv2_tcn_multiaffect_distilled.pt`
- Multi-affect distilled metrics: `models/mobilenetv2_tcn_multiaffect_distilled_metrics.json`

## Current Training Defaults
- XGB uses version-safe GPU settings when available and falls back to CPU `hist` if unsupported.
- TCN is capped at 8 epochs.
- Distillation is capped at 8 epochs.
- Distillation search is capped at 8 candidate trials.
- TCN defaults: batch size 4, 2 workers, backbone frame chunk size 32.
- Distillation defaults: batch size 4, 2 workers.
- Distillation memoizes cached HOG teacher features in memory across epochs.
- TCN and distillation use `torch.amp` autocast and `GradScaler`, auto-shrink batch size on CUDA OOM, then fall back to CPU if needed.
- LSTM/BiLSTM Optuna batch search is capped at 8 trials with batch choices `4/6/8`.
- Multi-affect distillation reuses the current single-task distilled student hyperparameters when available and warm-starts compatible student weights from the single-task TCN checkpoint.

## Tips
- Set `XGB_FORCE_CPU=1` to skip the GPU attempt for XGBoost.
- Set `CUDA_VISIBLE_DEVICES` to pick a specific GPU.
- If distillation validation is slow, build the Validation LMDB once:
  ```powershell
  python generatecache.py --split Validation --use_cuda
  ```

## License
- GPL v2 (see `LICENSE`)
