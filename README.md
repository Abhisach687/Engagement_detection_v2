# Engagement Detection Project

Complete DAiSEE engagement pipeline: HOG+XGBoost baseline, MobileNetV2 + LSTM/BiLSTM/TCN, knowledge distillation, LMDB caching, and a NiceGUI webcam app. Project is AI‑assisted (ChatGPT/Codex).

## Quick Start (fastest path)
1. Create & activate venv (Python 3.10+):
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Place DAiSEE data:
   ```
   DAiSEE/ExtractedFrames/<Split>/<ClipID>/frame_XXXX.jpg
   DAiSEE/Labels/<Split>Labels.csv   # Train/Validation/Test
   ```
3. Build LMDB caches (split‑namespaced keys; greatly speeds train/val/test):
   ```powershell
   python generatecache.py --split Train
   python generatecache.py --split Validation
   python generatecache.py --split Test
   # optional GPU preprocessing
   python generatecache.py --split Train --use_cuda
   ```

## Train
Default resumes Optuna studies and prefers cache.
```powershell
python trainmodels.py                                     # full pipeline
python trainmodels.py --fresh_studies                     # delete study DBs, start fresh
python trainmodels.py --force_cache                       # fail on cache miss
python trainmodels.py --no_cache                          # bypass cache (debug)
python trainmodels.py --retrain_xgb                       # force XGB retrain even if model exists
python trainmodels.py --retrain_lstm                      # force LSTM retrain
python trainmodels.py --retrain_bilstm                    # force BiLSTM retrain
python trainmodels.py --retrain_distill                   # force distilled TCN retrain
python trainmodels.py --fresh_studies --force_cache --skip_seq --skip_tcn --skip_distill   # XGB only
python trainmodels.py --skip_xgb --skip_tcn --skip_distill                               # LSTM + BiLSTM only
python trainmodels.py --skip_xgb --skip_seq --skip_distill                               # TCN only
python trainmodels.py --skip_xgb --skip_seq --skip_tcn                                   # Distill only
# Keep existing XGB, train LSTM/BiLSTM + distill with nested progress bars:
python trainmodels.py --skip_xgb --skip_tcn --force_cache --fresh_studies
```
Progress bars: trial → boost/epoch → batch → val, visible from trial 0; XGB booster bar stays on screen.

## Evaluate
```powershell
python testmodel.py --split Validation   # or --split Test
python evaluatemodels.py
```

## Inference App
```powershell
python app.py
```
Shows device (`cuda`/`cpu`) in GUI.

## Outputs & Logs
- Models/metrics: `models/`
- Training curves/logs: `logs/`
- HOG cache: `models/hog_features/`
- Frame LMDB: `cache/frames.lmdb` (keys: `Train:clip`, `Validation:clip`, `Test:clip`; legacy fallback supported)
- Study DBs: `models/*.db` (resume by default; use `--fresh_studies` to delete)

## Tips
- XGB tries GPU (`gpu_hist`); falls back to CPU hist if unsupported. Set `XGB_FORCE_CPU=1` to skip GPU attempt.
- DataLoaders use up to 8 workers, pin-memory on CUDA.
- LSTM/BiLSTM/TCN use `torch.amp` autocast/GradScaler; auto-shrink batch on CUDA OOM, then fall back to CPU.
- Optuna batch search capped at 8 (choices 4/6/8).
- Set `CUDA_VISIBLE_DEVICES` to pick GPU.

## License
- GPL v2 (see `LICENSE`). Project is AI-assisted (ChatGPT/Codex).
