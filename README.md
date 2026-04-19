# Mnemosky

**A satellite and airplane trail detector for long-exposure night-sky video and RAW image sequences.**

Mnemosky finds dim celestial trails in MP4 videos or folders of RAW images, classifies them as satellites or airplanes, optionally freezes the trail on-screen, and can export detections as ML-ready datasets or review them interactively. It ships three swappable detection algorithms (classical Canny/Hough, a Radon-transform pipeline, and a neural-network backend), transparent rejection accounting inspired by Science & Technology Studies, and an end-to-end human-in-the-loop learning system.

Current version: **v0.2.0-sts** · v0.3.0 learning foundation landed on `main` (see [What's new](#whats-new-v030-foundation)).

---

## Gallery

**Preprocessing preview** — interactive CLAHE / blur / Canny tuning with user-marked trail examples:

<img width="1861" height="931" alt="Preprocessing preview" src="https://github.com/user-attachments/assets/053cb18d-e020-44a2-b73f-3d9374eef083" />

**Debug visualization** — side-by-side view of preprocessing, detection masks, and classifications:

<img width="1366" height="1024" alt="Debug view" src="https://github.com/user-attachments/assets/6911a2f1-840d-4e76-b08f-5d59583c456c" />

**Detection output** — satellite (gold), airplane (orange), anomalous (magenta):

<img width="1366" height="1024" alt="Output" src="https://github.com/user-attachments/assets/07d3a4d7-51ed-44ce-adb6-fde3623cd305" />

---

## Features

### Detection

- **Three algorithms**, selected with `--algorithm`:
  - `default` — adaptive Canny + multi-scale HoughLinesP + matched-filter bank, with PSF-aware kernels and star-eccentricity rejection
  - `radon` — a-contrario LSD + Radon transform + perpendicular cross-filter, with ground-truth calibration and multi-frame residual accumulation for √N SNR boost
  - `nn` — YOLOv8/YOLOv11 / ONNX Runtime / cv2.dnn inference, with optional hybrid mode that merges NN and classical results
- **Interactive preview GUIs** (`--preview`) for every algorithm — live sliders for thresholds, kernel sizes, and SNR cuts, all rendered in the same dark-theme single-window layout
- **RAW image folder input** — point at a directory of Sony ARW / Canon CR2 / Nikon NEF / DNG files; Mnemosky decodes with `rawpy`, applies 16-bit CLAHE + percentile stretch in LAB space, and builds the MP4 on the fly
- **Trail classification** — smoothness, monochromaticity, PSF, SNR, and bright-spot-cluster features separate satellites from airplanes from anomalies

### Performance

- **Frame-level parallelism** via `multiprocessing.Pool` (`--workers N`, default auto-detect, cap 8); detection is stateless per frame so throughput scales nearly linearly
- **Optional CUDA acceleration** for `filter2D` (matched filter) and `warpAffine` (Radon), with per-operation fallback — if one GPU path fails, the others keep running
- **Radon pipeline**: 2.2× sequential, 3.96× parallel speedup versus the original implementation (aggressive downsampling, float32 pipeline, LSD capping, GT-calibration reuse across workers)

### Human-in-the-loop learning

- **Interactive review UI** (`--review` / `--review-only`) — dark-theme OpenCV window, 280 px sidebar, 56 px status bar, keyboard shortcuts (A/R/S/P/M/Z/L/Tab/Space/X/F/H/Q)
- **Two-tier parameter learning** — Tier 1 EMA per-correction + Tier 2 batch optimizer over 13 parameters, with named loss profiles (`discovery` / `precision` / `balanced` / `catalog`) and hard safety bounds
- **COCO-compatible annotation database** with correction history, session tracking, atomic writes, and corruption recovery
- **Learned-parameter profiles** persist to `~/.mnemosky/learned_params.json` and reload on the next run with `--hitl-profile <name>`

### Dataset export

- **Four ML-dataset formats** (`--dataset --dataset-format ...`): `aabb` (standard YOLO), `obb` (YOLO oriented bboxes, ideal for thin trails), `segment` (YOLO instance segmentation), `coco` (Detectron2 / MMDetection)
- **Temporal-episode train/val/test splitting** avoids leakage across frame neighbors
- **Perceptual-hash dedup**, configurable frame skip, negative-sample mining
- **HITL-verified export** (`--dataset-from-annotations`) — train only on confirmed detections

### Science & Technology Studies instrumentation

- **Translation Ledger** (`--ledger`) — Callon-inspired rejection audit, 14 named rejection types per pipeline stage, printed summary at end of run, JSONL sidecar for programmatic analysis
- **Situated loss profiles** — the four named profiles trade FP / FN / misclassification weights for different observer goals (catalog completeness vs. precision vs. discovery)
- **Observer context** (`--observer-lat/-lon/-elevation/-bortle/-fov/-notes`) — Haraway's "situated knowledges" made functional; observer metadata is embedded in every annotation

### What's new (v0.3.0 foundation)

A 12-initiative learning system has landed on `main` as the foundation for the upcoming v0.3.0 release. All 12 are wired end-to-end from CLI flag to consumer:

| # | Name | Flag | Purpose |
|---|---|---|---|
| I1 | RescueClassifier | `--train-rescue` | MLP consulted on near-boundary rejections; trained offline from annotations + ledger |
| I2 | Residual stack | (automatic) | `TemporalFrameBuffer` passes a rolling diff stack to Radon for √N SNR gain in parallel mode |
| I3 | ThresholdHyperNet | `--hypernet` | Per-frame adaptive thresholds driven by rolling detection rate |
| I4 | IMM Kalman tracker | `--tracker imm` | Motion-model tracker with LEO / MEO / GEO priors; handles crossing trails + 1-3 frame gaps |
| I5 | LongBackgroundModel | `--long-bg` | P²-quantile per-pixel long-horizon median for geosynchronous / slow-object rescue |
| I6a | TPE optimizer | `--tier2-optimizer tpe` | Tree-structured Parzen Estimator replacement for coordinate-wise Tier-2 search |
| I6b | TrustRegionAdapter | `--use-trust-region` | Batched Tier-1 corrections with loss-regression rejection; robust to mis-clicks |
| I6c | Platt calibrator | `--train-rescue` | Calibrated confidence scores in `ParameterAdapter.compute_confidence` |
| I6d | BALDQueue | (automatic in review) | Entropy + rescue-classifier disagreement ranking for active learning |
| I7a | TrackletPseudoLabeler | `--pseudo-label` | Emits 3+ frame tracklets as auto-confirmed annotations (zero-effort training corpus) |
| I7b | TrackletSequenceHead | (automatic) | Temporal classifier that refines tracklet class once 3+ members accumulate |
| I7c | AlgorithmFusionHead | `--fusion` | Logistic-regression fusion of (default, radon, nn) scores + cross-IoU; attaches `fusion_score` |

---

## Installation

Mnemosky is a single-file Python application. Clone the repo and install the dependencies you need:

```bash
git clone https://github.com/maxabernathy/Mnemosky.git
cd Mnemosky

# Core (required)
pip install opencv-python numpy

# Recommended
pip install scipy           # faster Radon NMS; falls back to cv2.dilate otherwise
pip install rawpy           # ARW / CR2 / NEF / DNG decoding
pip install exifread        # EXIF metadata extraction

# Neural-network backend (pick one or more)
pip install ultralytics     # YOLOv8 / v11; auto-installed on first --algorithm nn run
pip install onnxruntime     # ONNX Runtime backend
# cv2.dnn ships with opencv-python — no extra install
```

Python 3.9 or newer recommended. GPU acceleration requires a CUDA-enabled build of OpenCV (see `build_exe.sh` for a portable Windows build example).

---

## Quick start

### Detect trails in a video

```bash
python satellite_trail_detector.py input.mp4 output.mp4
```

### Tune before you run

```bash
python satellite_trail_detector.py input.mp4 output.mp4 --preview
```

Opens the interactive preview. Drag sliders to dial in CLAHE clip, blur kernel, Canny thresholds; click to mark example trails (the signal envelope is extracted and fed to the detector).

### Process a folder of RAW images

```bash
python satellite_trail_detector.py /path/to/raws/ stars.mp4 --target-height 2160
```

### Export a YOLO dataset

```bash
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-format obb
```

Emits `output_dataset/` with `train/val/test` splits, `images/`, `labels/`, and a `data.yaml`.

### Review and learn

```bash
# Process + open review UI
python satellite_trail_detector.py input.mp4 output.mp4 --review --hitl-profile my_camera

# Review existing annotations later
python satellite_trail_detector.py input.mp4 output.mp4 --review-only \
    --annotations output.json --hitl-profile my_camera
```

### Full v0.3.0 learning stack

```bash
python satellite_trail_detector.py input.mp4 output.mp4 \
    --algorithm nn --model trail.pt --nn-hybrid \
    --tracker imm --long-bg --hypernet --fusion \
    --pseudo-label --train-rescue
```

Runs NN + classical in hybrid mode, uses the IMM Kalman tracker with motion priors, enables the long-horizon background model and per-frame adaptive thresholds, attaches fused scores, emits auto-confirmed tracklets as pseudo-labels, and (after processing) trains the rescue classifier, Platt calibrator, and fusion head from everything labeled next to the output.

---

## CLI reference

Run `python satellite_trail_detector.py --help` for the full list. Grouped highlights:

**Detection & output**
`--sensitivity {low,medium,high}` · `--algorithm {default,radon,nn}` · `--detect-type {both,all,satellites,airplanes,anomalous}` · `--freeze-duration SECS` · `--max-duration SECS` · `--no-labels`

**Interactive**
`--preview` · `--debug` · `--debug-only` · `--show-processing`

**Parallelism & GPU**
`--workers N` · `--no-gpu`

**Neural network**
`--model PATH` · `--nn-backend {ultralytics,cv2dnn,onnxruntime}` · `--confidence FLOAT` · `--nms-iou FLOAT` · `--nn-hybrid` · `--nn-class-map JSON`

**RAW images**
`--half-size-raw` · `--no-raw-enhance` · `--target-height PX` · `--keep-video`

**Dataset export**
`--dataset` · `--dataset-format {aabb,obb,segment,coco}` · `--dataset-split A B C` · `--dataset-skip N` · `--dataset-dedup PX` · `--dataset-negatives RATIO` · `--dataset-from-annotations JSON`

**Human-in-the-loop**
`--review` · `--review-only` · `--annotations PATH` · `--hitl-profile NAME` · `--auto-accept FLOAT` · `--no-learn` · `--loss-profile {discovery,precision,balanced,catalog}`

**STS instrumentation**
`--ledger` · `--observer-lat FLOAT` · `--observer-lon FLOAT` · `--observer-elevation M` · `--observer-bortle 1-9` · `--observer-fov DEG` · `--observer-notes TEXT`

**v0.3.0 learning**
`--tracker {default,imm}` · `--long-bg` · `--hypernet` · `--pseudo-label` · `--train-rescue` · `--tier2-optimizer {golden,tpe}` · `--use-trust-region` · `--fusion`

---

## Architecture

Mnemosky is a single file — `satellite_trail_detector.py`, ~15,700 lines. The deep-dive map lives in [CLAUDE.md](CLAUDE.md), which documents:

- Class hierarchy (`SatelliteTrailDetector` base + `RadonStreakDetector` / `NeuralNetDetector` overrides)
- Module-level constants (`LOSS_PROFILES`, `PARAMETER_SAFETY_BOUNDS`, `CORRECTION_RULES`, `_HAS_CUDA`, ...)
- Detection data format and seven satellite-detection paths
- Parallelism model (main process owns the temporal buffer; workers are stateless)
- File-location quick reference (line numbers for every major component)

The HITL reinforcement-learning system is documented separately in [hitl_architecture.md](hitl_architecture.md).

---

## Configuration

Per-user configuration lives in `~/.mnemosky/`:

- `config.json` — default sensitivity, NN model path, backend preferences
- `learned_params.json` — per-profile HITL-learned parameters
- `rescue.json` — trained RescueClassifier weights (populated by `--train-rescue`)
- `platt.json` — confidence-calibration coefficients
- `fusion_head.json` — AlgorithmFusionHead weights
- `hypernet.json` — ThresholdHyperNet weights

Use `--save-config` to persist the current run's detection parameters as the new defaults.

---

## Project layout

```
Mnemosky/
├── satellite_trail_detector.py   # Everything: detection, HITL, UI, export, CLI
├── CLAUDE.md                     # Deep-dive architecture & conventions (AI-reader friendly)
├── hitl_architecture.md          # HITL RL design document
├── build_exe.py / build_exe.sh   # Portable Windows executable build
└── README.md                     # This file
```
