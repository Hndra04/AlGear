# AIGear

**Sistem Monitoring Kepatuhan APD Berbasis AI Secara Real-Time**

AIGear is a real-time PPE (Personal Protective Equipment) compliance monitoring system powered by computer vision. It leverages existing CCTV infrastructure to detect workers, verify helmet and safety-vest usage, count people in monitored zones, and issue alerts — all without additional hardware.

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

## Dataset

**Construction Site Safety Dataset** (Roboflow Universe) — ~3,000 labeled images with YOLOv8-ready format.

- **Classes:** `person`, `helmet`, `no-helmet`, `safety-vest`, `no-vest`
- **Split:** 70% train / 15% validation / 15% test
- **Augmentation:** Horizontal flip, HSV jitter, mosaic crop
- **Class imbalance mitigation:** Weighted loss + oversampling for minority classes (`no-helmet`, `no-vest`)

## System Architecture

A single unified **YOLOv8s** model detects all five classes simultaneously (replacing a two-model approach for efficiency).

### Pipeline (per video frame)

1. **Person Detection & People Counting** — `person` class detections are counted per frame (zone-based counting).
2. **PPE Detection** — `helmet`, `no-helmet`, `safety-vest`, `no-vest` detections are associated with the nearest worker via IoU overlap on relevant body regions (head for helmet, torso for vest).
3. **Compliance Decision Engine** — rule-based association determines status per individual:
   - **Safe** — both helmet and vest detected for that worker.
   - **Warning** — one or more PPE items missing or detected as `no-helmet`/`no-vest`.

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

## Roadmap — Phase 2

| Stage | Focus | Output |
|---|---|---|
| 1 | Data & Model Preparation | Dataset validation, unified YOLOv8s pipeline, initial fine-tuning, per-class mAP & Recall |
| 2 | Core Pipeline Development | Zone-based people counting, IoU-based PPE–worker association, configurable compliance decision engine |
| 3 | Real-Time Monitoring UI | OpenCV visualization (color-coded bounding boxes, active worker counter, violation alerts) |
| 4 | Testing & Optimization | Full metric evaluation, detection threshold tuning to minimize FNR, demo preparation |

## Future Enhancements

- Virtual line crossing for cumulative in/out counting
- Restricted zone detection
- ONNX model export for cross-platform deployment
- Worker compliance trend analytics
