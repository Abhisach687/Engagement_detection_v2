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
Fastest run path:

1. Activate the project virtual environment.
2. Make sure the distilled app model exists under `models/`.
3. Start the desktop app:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
4. Click `Start Camera`.
5. Click `Start Pomodoro` to begin the 24-minute focus session and answer the timed self-checks every 8 minutes.

If you want the feedback fine-tuning commands to work on CPU when CUDA is unavailable, set:

```powershell
$env:REQUIRE_CUDA = "0"
```

```powershell
.\.venv\Scripts\python.exe app.py
```

App behavior:
- The app auto-loads the multi-affect distilled student if `models/mobilenetv2_tcn_multiaffect_distilled.*` exists.
- Otherwise it falls back to the original engagement-only distilled student.
- If the required ONNX export is missing, the app tries to export it automatically from the saved PyTorch or TorchScript checkpoint.
- The GUI shows the active device (`cuda` or `cpu`) and the active model variant.
- The desktop UI opens in a split-screen-friendly layout, supports mouse-wheel and trackpad scrolling, and includes a live preview, engagement decision panel, rolling summaries, and a Pomodoro timer card.
- When the multi-affect ONNX sidecar metadata is present, the app derives the secondary affect tiles from `head_names` and uses the feedback-adjusted spotlight threshold shown in the UI.
- After enough check-ins are collected, the app adjusts its live primary and spotlight thresholds from user feedback without changing model weights in-session.

### User-in-the-Loop Feedback
Run the live app first:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then use the feedback loop like this:

1. Click `Start Camera`.
2. Click `Start Pomodoro`.
3. Work through the current 8-minute focus block while the app tracks live model outputs.
4. When the self-check opens, answer how engaged, bored, confused, and frustrated you felt in the last 8 minutes.
5. Submit the check-in, skip it, or stop the Pomodoro.
6. Repeat for all three 8-minute blocks in the 24-minute session.

What happens after submission:
- The corresponding Pomodoro frame window is saved under `cache/user_feedback/clips/`.
- The check-in metadata is appended to `cache/user_feedback/feedback_log.jsonl`.
- The app updates the feedback insight line in the UI.
- After `5` check-ins, the live primary and spotlight thresholds begin adapting conservatively from the collected feedback.

Trust rules:
- Explicit corrections are trusted for training.
- The app derives an internal rating from how closely the user self-check matches the model's 8-minute aggregate.
- Derived ratings `4` or `5` are trusted for training only when all visible heads are known.
- `Don't know` excludes that head from supervised training.
- Check-ins without strong trust stay available for analytics and live threshold steering, but are not exported as supervised training samples by default.

### Feedback Commands
Use these commands after you have collected some check-ins:

```powershell
# inspect check-in counts, trust counts, rating distribution, and current thresholds
.\.venv\Scripts\python.exe user_in_the_loop_training.py summarize

# export the currently trusted feedback records into a training manifest
.\.venv\Scripts\python.exe user_in_the_loop_training.py export --variant auto

# short incremental fine-tune on newly trusted records since the last online run
.\.venv\Scripts\python.exe user_in_the_loop_training.py train_online --variant auto --since-last

# full batch fine-tune on the entire trusted feedback pool
.\.venv\Scripts\python.exe user_in_the_loop_training.py train_offline --variant auto
```

Command behavior:
- `summarize` reports current feedback statistics and the effective live thresholds.
- `export` writes a trusted manifest under `cache/user_feedback/exports/`.
- `train_online` creates a short candidate fine-tune from only newly trusted user-labeled data.
- `train_offline` creates a fuller candidate fine-tune from the complete trusted user-labeled pool.
- Candidate checkpoints are written under `cache/user_feedback/candidates/` and are not auto-promoted into the live app.

### Project Explainer
The explainer workflow now lives entirely under `cache/explainer/`.

- Markdown explainer: `cache/explainer/project_explained.md`
- PDF explainer: `cache/explainer/project_explained.pdf`

Regenerate the explainer from the repo root with:

```powershell
.\.venv\Scripts\python.exe cache\explainer\generate_explainer.py
```

## Outputs
- Models and metrics: `models/`
- Training curves and logs: `logs/`
- HOG cache: `models/hog_features/`
- Frame LMDB: `cache/frames.lmdb`
  Keys are split-namespaced: `Train:clip`, `Validation:clip`, `Test:clip`
- Study databases: `models/*.db`
- Multi-affect distilled student: `models/mobilenetv2_tcn_multiaffect_distilled.pt`
- Multi-affect distilled metrics: `models/mobilenetv2_tcn_multiaffect_distilled_metrics.json`
- User feedback clips: `cache/user_feedback/clips/*.npz`
- User feedback log: `cache/user_feedback/feedback_log.jsonl`
- User feedback exports: `cache/user_feedback/exports/`
- User feedback candidate checkpoints: `cache/user_feedback/candidates/`
- Project explainer markdown: `cache/explainer/project_explained.md`
- Project explainer PDF: `cache/explainer/project_explained.pdf`

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
- Set `REQUIRE_CUDA=0` if you want the feedback fine-tuning commands to allow CPU fallback.
- If distillation validation is slow, build the Validation LMDB once:
  ```powershell
  python generatecache.py --split Validation --use_cuda
  ```

## License
- GPL v2 (see `LICENSE`)
