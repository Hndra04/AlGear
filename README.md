# AIGear

**Real-Time AI-Powered PPE Compliance Monitoring System**

AIGear is a real-time PPE (Personal Protective Equipment) compliance monitoring system built on YOLOv8 computer vision. It leverages existing CCTV infrastructure to detect workers, verify helmet and safety-vest usage, count people in monitored zones, and issue alerts — all without additional hardware.

## Problem

Workplace accidents in industrial environments (construction, factories, warehouses) remain a serious challenge. Manual PPE inspections are inefficient and cannot be sustained continuously across large areas. AIGear automates this process using AI.

## Target Metrics

| Metric | Description | Target |
|---|---|---|
| mAP@50 (PPE Detection) | Object detection accuracy at IoU = 0.50 | ≥ 80% |
| PPE Compliance Accuracy | Safe/violation classification per detected worker | ≥ 85% |
| False Negative Rate (Violations) | Missed PPE violation detections | ≤ 10% |
| People Counting Accuracy | MAE between actual vs. predicted worker count | ≤ 2 persons |
| Inference Speed | Frames per second on target hardware | ≥ 15 FPS |

> **Note:** FNR is prioritized — in safety contexts, missing a PPE violation is more dangerous than a false alarm.

## Domain

**Computer Vision — Human Safety Monitoring.** This domain was chosen because:

1. **Leverages existing infrastructure** — most industrial facilities already have CCTV.
2. **24/7 consistent monitoring** — unlike manual inspections, AI never gets tired.
3. **K3 (Occupational Safety) regulatory relevance** — automated systems provide real-time archived compliance evidence.

## Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | YOLOv8s (Ultralytics) |
| Multi-Object Tracking | ByteTrack |
| API Framework | FastAPI + Uvicorn |
| Visualization | OpenCV |
| Dataset | Roboflow Universe (Construction Site Safety) |
| Oversampling | Albumentations |
| CLI | Typer |
| Deployment | Docker, Railway |

## Dataset

**Construction Site Safety Dataset** (Roboflow Universe) — **1,206 labeled images** with YOLOv8-ready format.

- **Classes:** `person`, `helmet`, `no-helmet`, `safety-vest`, `no-vest`
- **Split:** 70% train (834) / 15% validation (174) / 15% test (198)
- **Augmentation:** Horizontal flip, HSV jitter, mosaic crop
- **Class imbalance mitigation:** Weighted loss + oversampling for minority classes (`no-helmet` 3×, `no-vest` 2×)

### Class Distribution (Train Split)

| Class | Instances | Percentage |
|---|---:|---:|
| person | 1,937 | 38.5% |
| helmet | 1,736 | 34.5% |
| vest | 917 | 18.2% |
| no-vest | 623 | 12.4% |
| no-helmet | 80 | 1.6% |

> **Imbalance ratio (helmet : no-helmet):** 21.7×

## System Architecture

A single unified **YOLOv8s** model detects all five classes simultaneously (replacing a two-model approach for efficiency).

### Pipeline (per video frame)

```
Video/Image Input
       │
       ▼
[YOLOv8 Detection]        → person, helmet, no-helmet, vest, no-vest
       │
       ▼
[ByteTrack]               → persistent worker IDs across frames
       │
       ▼
[PPE-Person Association]  → IoU-based matching of PPE boxes to person boxes
       │
       ▼
[Compliance Classification] → Safe = helmet AND vest (no violations detected)
       │                       Violation = otherwise
       ▼
[Zone Counting]           → point-in-polygon test on person bounding-box centers
       │
       ▼
[Visualization]           → zones, bounding boxes, compliance colors, track IDs, HUD
       │
       ▼
[API / CLI / Video Output]
```

### Key Stages

1. **Person Detection & People Counting** — `person` class detections are counted per frame (zone-based counting).
2. **Multi-Object Tracking** — ByteTrack assigns persistent IDs to workers across frames for consistent monitoring.
3. **PPE Detection** — `helmet`, `no-helmet`, `safety-vest`, `no-vest` detections are associated with the nearest worker via IoU overlap on relevant body regions (head for helmet, torso for vest).
4. **Compliance Decision Engine** — rule-based association determines status per individual:
   - **Safe** — both helmet and vest detected for that worker (no `no-helmet` or `no-vest` detections).
   - **Violation** — one or more PPE items missing or detected as `no-helmet`/`no-vest`.
5. **Zone Counting** — configurable polygonal zones with point-in-polygon person counting.

## Model Variants

Four model variants were trained to evaluate different imbalance-handling strategies:

| Variant | Strategy | mAP@50 | FNR (no-helmet) | Notes |
|---|---|---:|---:|---|
| `baseline` | No imbalance handling | 77.8% | 34.8% | Reference baseline |
| `oversample-10x` | 10× oversampling (aggressive) | 76.1% | — | Degraded overall |
| `oversample-conservative` | 3× no-helmet, 2× no-vest | 76.2% | — | Balanced approach |
| **`resplit-oversample-conservative`** | **Stratified resplit + conservative oversampling** | **81.0%** | **36.2%** | **Production model** |

> **Production model:** `models/resplit-oversample-conservative/weights/best.pt`

### Production Model Metrics (Test Set)

| Class | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| helmet | 0.777 | 0.930 | 0.932 | 0.510 |
| no-helmet | 0.752 | 0.638 | 0.660 | 0.257 |
| no-vest | 0.544 | 0.819 | 0.663 | 0.340 |
| person | 0.683 | 0.957 | 0.928 | 0.599 |
| vest | 0.644 | 0.893 | 0.865 | 0.466 |
| **Overall** | **0.680** | **0.847** | **0.810** | **0.434** |

## Evaluation Plan

### Ground Truth Sources
- **Test set annotations** — existing bounding box labels from the dataset.
- **Controlled test videos** — 2–3 short clips (~30s) with manually recorded ground truth counts.

### Evaluation Metrics

| Component | Metrics | Notes |
|---|---|---|
| PPE & Person Detection | mAP@50, Precision, Recall, F1 | Per-class and overall |
| People Counting | MAE | Average difference in `person` count per image |
| Compliance Classification | Accuracy, FNR, 2×2 Confusion Matrix | FNR prioritized |
| Speed | FPS | Measured during test video inference |

## Running Evaluation

The project includes a reproducible evaluation script (`algear/evaluate.py`) that measures both detection-level and pipeline-level metrics.

**No data? No problem.** The script automatically downloads the dataset from Roboflow and prepares it (stratified resplit) if not found locally. Just set your API key:

```bash
# 1. Set your Roboflow API key
cp .env.example .env
# Edit .env and add your ROBOFLOW_API_KEY (get it at https://app.roboflow.com/settings/api)

# 2. Run evaluation (auto-downloads + resplits if needed)
python -m algear.evaluate
```

### Basic Usage

```bash
# Default: evaluate production model on test set (seed=42)
python -m algear.evaluate

# Or via Makefile
make evaluate
```

### Full Workflow

When running for the first time, the script executes:

1. **Download** — fetches dataset from Roboflow Universe (~1,206 images)
2. **Resplit** — stratified 70:15:15 split (train:val:test)
3. **Evaluate** — runs detection + pipeline metrics
4. **Report** — saves JSON report to `models/evaluation/`

### Options

| Flag | Default | Description |
|---|---|---|
| `--model, -m` | `models/resplit-oversample-conservative/weights/best.pt` | Path to `.pt` model file |
| `--data, -d` | auto-detect | Path to dataset `data.yaml` (auto-resolved from resplit dir) |
| `--split, -s` | `test` | Dataset split: `train` / `val` / `test` |
| `--conf, -c` | `0.25` | Confidence threshold for inference |
| `--device` | `cpu` | Device: `cpu` / `cuda` / `0` |
| `--output, -o` | `models/evaluation` | Directory for JSON report output |
| `--seed` | `42` | Random seed for reproducibility |
| `--skip-prepare` | `false` | Skip auto data download/resplit |

### Examples

```bash
# Evaluate baseline model
python -m algear.evaluate --model models/baseline/weights/best.pt

# Evaluate on validation set with higher confidence threshold
python -m algear.evaluate --split val --conf 0.3

# Use GPU and custom output directory
python -m algear.evaluate --device cuda --output results/eval-run1

# Compare models with different seeds
python -m algear.evaluate --model models/baseline/weights/best.pt --seed 123
python -m algear.evaluate --model models/resplit-oversample-conservative/weights/best.pt --seed 123

# Skip data preparation (if data already exists)
python -m algear.evaluate --skip-prepare
```

### Output

The script generates a JSON report at `models/evaluation/eval_<timestamp>.json` containing:

- **Configuration** — model path, data yaml, split, confidence, device
- **Environment** — Python, torch, numpy, ultralytics versions, CUDA availability
- **Detection metrics** — mAP@50, mAP@50:95, Precision, Recall (per-class and overall)
- **Pipeline metrics** — PPE Compliance Accuracy, People Counting MAE
- **Target compliance** — pass/fail for each target metric

Example output summary:

```
============================================================
RESULTS SUMMARY
============================================================
  mAP@50:              0.8100
  mAP@50:95:           0.4340
  Precision:           0.6800
  Recall:              0.8470
  Per-class:
    helmet           mAP50=0.932  P=0.777  R=0.930
    no-helmet        mAP50=0.660  P=0.752  R=0.638
    ...
  PPE Compliance Acc:  0.8723
  Counting MAE:        1.4500

  Target compliance:
    [PASS] mAP50 >= 0.80
    [PASS] PPE_compliance >= 0.85
    [PASS] counting_MAE <= 2
============================================================
```

### Reproducibility

The script sets fixed random seeds (`random`, `numpy`, `torch`) and captures the full environment snapshot in the JSON report. Use `--seed` to reproduce results across runs.

## REST API

AIGear exposes a FastAPI inference API with the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check with model status |
| `/model/info` | GET | Model metadata and class definitions |
| `/infer/image` | POST | Single image inference (multipart upload) — returns JSON + annotated image as base64 |
| `/infer/video` | POST | Video inference (multipart upload) — returns per-frame results + summary + annotated video as base64 |

### Running the API

```bash
# Local development
make serve

# Or directly
python -m algear.api
```

### API Usage Example

```bash
# Health check
curl http://localhost:8000/health

# Image inference
curl -X POST http://localhost:8000/infer/image \
  -F "file=@test_image.jpg"
```

## Deployment

### Docker

```bash
# Build and start
make docker-up

# Stop
make docker-down
```

The Docker container exposes port 8000 and mounts `./models` as read-only.

### Railway

The project includes Railway deployment configuration (`railway.json`) with CI/CD via GitHub Actions (`.github/workflows/railway-deploy.yml`).

## Configuration

Pipeline configuration is loaded from YAML files via `algear/pipeline_config.py`. Example:

```yaml
model: models/resplit-oversample-conservative/weights/best.pt
conf: 0.25
iou: 0.45
imgsz: 640
device: cpu
zones:
  - name: "Zone A"
    polygon: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
```

Zone coordinates support normalized (0–1) or absolute pixel values.

## Roadmap — Phase 2

| Stage | Focus | Output | Status |
|---|---|---|---|
| 1 | Data & Model Preparation | Dataset validation, unified YOLOv8s pipeline, initial fine-tuning, per-class mAP & Recall | ✅ Complete |
| 2 | Core Pipeline Development | Zone-based people counting, IoU-based PPE–worker association, configurable compliance decision engine | ✅ Complete |
| 3 | Real-Time Monitoring UI | OpenCV visualization (color-coded bounding boxes, active worker counter, violation alerts), ByteTrack integration | ✅ Complete |
| 4 | Testing & Optimization | Full metric evaluation, detection threshold tuning to minimize FNR, API deployment, demo preparation | 🔄 In Progress |

## Known Limitations

1. **Minority class deficit** — `no-helmet` has only 80 training instances (1.6%), resulting in 36.2% FNR.
2. **PPE Compliance Accuracy and People Counting MAE** — measured via `python -m algear.evaluate`, but results vary per model variant.
3. **No real-world evaluation** — tested only on dataset images, not actual CCTV footage.
4. **Distribution shift** — validation metrics significantly higher than test metrics, suggesting overfitting.
5. **No confidence threshold tuning** — default thresholds not optimized for per-class recall.
6. **Empty test suite** — `tests/` directory exists but contains no implementations.
7. **Placeholder modules** — `features.py` and `plots.py` are stubs.

## Future Enhancements

- Virtual line crossing for cumulative in/out counting
- Restricted zone detection
- ONNX model export for cross-platform deployment
- Worker compliance trend analytics
- Confidence threshold tuning per class
- Real-world CCTV evaluation dataset
- Regression test suite

## Notebooks

| Notebook | Purpose |
|---|---|
| `1.0-alg-data-loading-and-exploration.ipynb` | Data loading and EDA |
| `2.0-alg-baseline-training.ipynb` | Baseline YOLOv8s training |
| `2.1-alg-oversampling.ipynb` | 10× oversampling experiment |
| `2.2-alg-oversampling-conservative.ipynb` | Conservative oversampling (3×/2×) |
| `3.0-alg-dataset-resplit.ipynb` | Stratified resplit (70:15:15) |
| `3.1-alg-resplit-oversampling.ipynb` | Resplit + conservative oversampling |
| `4.0-alg-pipeline-metrics.ipynb` | PPE Compliance Accuracy & People Counting MAE |
| `4.1-alg-core-pipeline-demo.ipynb` | Pipeline demo on images and video |

## Project Organization

```
├── LICENSE                     <- MIT License
├── Makefile                    <- Convenience commands (make serve, make lint, etc.)
├── README.md                   <- The top-level README for developers
├── Dockerfile                  <- Container image definition
├── docker-compose.yml          <- Docker Compose service config
├── railway.json                <- Railway.app deployment config
├── pyproject.toml              <- Package metadata and Ruff config
├── requirements.txt            <- Python dependencies
├── setup.cfg                   <- Flake8 configuration (legacy)
├── .env.example                <- Environment variable template
│
├── data/
│   ├── raw/                    <- Original dataset from Roboflow
│   ├── processed/              <- Resplit and oversampled datasets
│   ├── interim/                <- Intermediate transforms
│   └── external/               <- Third-party data
│
├── models/
│   ├── baseline/               <- Baseline (no imbalance handling)
│   ├── oversample-10x/         <- 10× oversampled
│   ├── oversample-conservative/ <- Conservative oversampling
│   └── resplit-oversample-conservative/  <- Production model
│       └── weights/best.pt
│
├── notebooks/                  <- Jupyter notebooks (experiments 1.0–4.1)
│
├── algear/                     <- Main Python package
│   ├── __init__.py
│   ├── config.py               <- Project paths and Roboflow constants
│   ├── core.py                 <- CompliancePipeline orchestrator
│   ├── tracker.py              <- ByteTrack multi-object tracker
│   ├── zones.py                <- Zone definitions and point-in-polygon counting
│   ├── visualize.py            <- OpenCV drawing (boxes, zones, HUD)
│   ├── metrics.py              <- PPE Compliance Accuracy & People Counting MAE
│   ├── evaluate.py             <- Reproducible evaluation script (CLI + JSON report)
│   ├── pipeline_config.py      <- YAML config loader
│   ├── dataset.py              <- Dataset CLI (download, prepare, oversample, resplit)
│   ├── features.py             <- Feature generation (stub)
│   ├── plots.py                <- Plot generation (stub)
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── pipeline.py         <- IoU-based PPE–worker association + compliance logic
│   │   ├── predict.py          <- YOLOv8 inference wrapper
│   │   ├── train.py            <- Train & evaluate models
│   │   └── oversample.py       <- Augmentation-based oversampling
│   │
│   └── api/
│       ├── __init__.py
│       ├── __main__.py         <- CLI entry point (python -m algear.api)
│       ├── app.py              <- FastAPI application and endpoints
│       ├── schemas.py          <- Pydantic request/response models
│       ├── inference.py        <- Image/video inference service
│       ├── dependencies.py     <- Pipeline singleton for DI
│       └── utils.py            <- Image/video encoding and validation
│
├── docs/                       <- Training reports and pipeline documentation
├── reports/                    <- Generated technical report (PDF)
├── references/                 <- Data dictionaries and manuals
├── tests/                      <- Test suite (empty — TODO)
└── .github/workflows/          <- CI/CD (Railway deployment)
```

## Quick Start

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Set your Roboflow API key (for auto-download)
cp .env.example .env
# Edit .env and add your ROBOFLOW_API_KEY

# 3. Run evaluation (auto-downloads data, resplits, and evaluates)
python -m algear.evaluate

# 4. Start API server
make serve
```

## License

MIT License. See [LICENSE](LICENSE) for details.
