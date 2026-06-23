# Core Pipeline Documentation

## Overview

The PPE compliance pipeline processes video frames (or images) through 5 stages:

```
Input Frame → YOLO Detection → ByteTrack → PPE Association → Compliance Classification → Zone Counting → Annotated Output
```

## Architecture

```
algear/
├── core.py              # CompliancePipeline — main orchestrator
├── tracker.py           # ByteTrack multi-object tracking wrapper
├── modeling/
│   ├── pipeline.py      # IoU-based PPE→person association, compliance rules
│   └── predict.py       # YOLOv8 inference wrapper
├── zones.py             # Zone definitions, point-in-polygon counting
├── visualize.py         # OpenCV drawing (boxes, zones, HUD)
└── pipeline_config.py   # YAML config loader
```

## Pipeline Stages

### Stage 1: Object Detection (`core.py:detect`)

- YOLOv8s model detects 5 classes: `helmet`, `no-helmet`, `no-vest`, `person`, `vest`
- Returns normalised bounding boxes `[class_id, cx, cy, w, h]` + confidence scores
- Configurable: `conf` (confidence threshold), `iou` (NMS IoU), `imgsz`, `device`

### Stage 2: ByteTrack (`core.py:track` → `tracker.py`)

- **ByteTrack** assigns persistent IDs to detected objects across frames
- Benefits over per-frame detection:
  - Re-identifies occluded/missed detections
  - Smooths compliance state (no flickering)
  - Proper zone entry/exit counting
  - Trajectory analysis possible
- Uses Ultralytics' built-in ByteTrack integration via `model.track()`
- Config: `use_tracker=True`, `tracker_cfg="bytetrack.yaml"`

### Stage 3: PPE–Worker Association (`pipeline.py:associate_ppe_to_persons`)

- For each detected person, finds the **nearest** helmet/no-helmet and vest/no-vest by IoU
- IoU thresholds: `IOU_HEAD_THRESHOLD=0.01`, `IOU_BODY_THRESHOLD=0.01` (low because PPE boxes are smaller than person boxes)
- Each person gets associated head PPE and body PPE labels

### Stage 4: Compliance Classification (`pipeline.py:classify_frame`)

A worker is **compliant** if ALL conditions are met:
- ✅ Has `helmet` detection AND no `no-helmet` detection
- ✅ Has `vest` detection AND no `no-vest` detection

Otherwise → **violation**.

### Stage 5: Zone-Based People Counting (`zones.py`)

- Zones are named polygonal regions (or full-frame by default)
- Point-in-polygon test on each person's bounding box centre
- Returns per-zone person counts

## Usage

### Single Image

```python
from algear.core import CompliancePipeline, PipelineConfig

pipeline = CompliancePipeline(
    model_path="models/resplit-oversample-conservative/weights/best.pt",
    config=PipelineConfig(conf=0.25, device="0", use_tracker=True),
)

frame = cv2.imread("site.jpg")
result = pipeline.process_frame(frame, frame_idx=0)

print(f"Persons: {result.person_count}")
print(f"Violations: {result.violation_count}")
print(f"Compliance rate: {result.compliance_rate:.1%}")
print(f"Track IDs: {result.track_ids}")

annotated = pipeline.render(frame, result)
```

### Video Processing (with ByteTrack)

```python
# ByteTrack assigns persistent IDs across frames
results = pipeline.process_video(
    source="site.mp4",
    output_path="output_annotated.mp4",
    show=False,
    use_tracker=True,
)

# Get unique persons seen across entire video
unique_persons = pipeline._tracker.get_unique_person_ids()
print(f"Unique persons tracked: {len(unique_persons)}")
```

### Video Processing (without tracking)

```python
# Per-frame detection only (no persistent IDs)
results = pipeline.process_video(
    source="site.mp4",
    output_path="output_annotated.mp4",
    show=False,
    use_tracker=False,
)
```

### Custom Zones

```python
config = PipelineConfig(
    zone_configs=[
        {"name": "zone_a", "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]},
        {"name": "zone_b", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]},
    ]
)
```

### YAML Config

```yaml
# config.yaml
model: models/resplit-oversample-conservative/weights/best.pt
conf: 0.25
iou: 0.45
imgsz: 640
device: cpu
use_tracker: true
tracker_cfg: bytetrack.yaml
tracker_persist: true
zones:
  - name: left
    polygon: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
  - name: right
    polygon: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
```

## Output Structure

```python
@dataclass
class FrameResult:
    detections: np.ndarray       # (N, 5) raw YOLO output
    scores: np.ndarray           # (N,) confidence per detection
    workers: list[WorkerCompliance]  # per-person compliance info
    person_count: int
    compliant_count: int
    violation_count: int
    compliance_rate: float
    zone_counts: dict[str, int]  # per-zone person count
    img_w: int
    img_h: int
    inference_ms: float
    # Tracking fields
    tracking_result: TrackingResult | None  # ByteTrack output
    track_ids: list[int]          # persistent track ID per worker
```

## Visualisation

The `render()` method draws:
- **Zone polygons** with semi-transparent fill
- **Person boxes** — green (compliant) / red (violation)
- **PPE labels** — H (helmet), !H (no-helmet), V (vest), !V (no-vest)
- **HUD bar** — people count, compliant count, violations, FPS

## Notebooks

| Notebook | Description |
|---|---|
| `4.0-alg-pipeline-metrics.ipynb` | Measures PPE Compliance Accuracy & People Counting MAE on test set |
| `4.1-alg-core-pipeline-demo.ipynb` | Demo pipeline on single images, batch, and video |
