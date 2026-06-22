# Baseline Training Report

## Objective

Train a YOLOv8s model on the raw Construction Site Safety dataset **without any class imbalance handling** to establish a baseline performance reference.

## Configuration

| Parameter   | Value      |
|-------------|------------|
| Model       | YOLOv8s    |
| Pretrained  | `yolov8s.pt` |
| Epochs      | 50 (completed) |
| Batch size  | 16         |
| Image size  | 640×640    |
| Optimizer   | AdamW (lr=0.001111, momentum=0.9) |
| Patience    | 20         |
| Device      | GPU (Tesla T4, Colab) |
| AMP         | enabled    |

## Dataset

**Source:** Construction Site Safety Dataset (Roboflow Universe)  
**Classes:** `helmet`, `no-helmet`, `no-vest`, `person`, `vest`  
**Splits:** 997 train / 119 validation / 90 test images

### Class Distribution (Train)

| Class      | Instances | Percentage |
|------------|----------:|-----------:|
| person     | 2362      | 37.0%      |
| helmet     | 2116      | 33.1%      |
| vest       | 1073      | 16.8%      |
| no-vest    | 741       | 11.6%      |
| no-helmet  | 94        | 1.5%       |

**Imbalance ratio (helmet : no-helmet):** 22.5×  
**Imbalance ratio (vest : no-vest):** 1.4×

## Baseline Results

### Test Set Evaluation (best.pt)

| Class      | Precision | Recall | mAP@50 | mAP@50:95 |
|------------|----------:|-------:|-------:|----------:|
| helmet     | 0.825     | 0.933  | 0.929  | 0.512     |
| no-helmet  | 0.796     | 0.652  | 0.592  | 0.192     |
| no-vest    | 0.568     | 0.705  | 0.630  | 0.279     |
| person     | 0.791     | 0.911  | 0.890  | 0.573     |
| vest       | 0.835     | 0.837  | 0.850  | 0.459     |
| **Overall** | **0.763** | **0.808** | **0.778** | **0.403** |

### Validation Set Evaluation (best.pt — best epoch)

| Class      | Precision | Recall | mAP@50 | mAP@50:95 |
|------------|----------:|-------:|-------:|----------:|
| helmet     | 0.949     | 0.914  | 0.952  | 0.537     |
| no-helmet  | 1.000     | 0.778  | 0.973  | 0.450     |
| no-vest    | 0.790     | 0.800  | 0.790  | 0.408     |
| person     | 0.904     | 0.934  | 0.946  | 0.642     |
| vest       | 0.898     | 0.809  | 0.889  | 0.507     |
| **Overall** | **0.908** | **0.847** | **0.910** | **0.509** |

### Training Summary

- **Best epoch:** epoch 50 (mAP@50=0.921 on val, P=0.914, R=0.884)
- **Total time:** 0.304 hours
- **Inference speed:** 10.2ms per image on Tesla T4 (test set)

## Analysis

### Target Metric Attainment

| Metric | Target | Achieved | Status |
|--------|-------:|---------:|:------:|
| mAP@50 (overall) | ≥ 80% | **77.8%** | ❌ 2.2pp short |
| mAP@50 (no-helmet) | ≥ 80% | **59.2%** | ❌ severely low |
| mAP@50 (no-vest) | ≥ 80% | **63.0%** | ❌ low |
| Recall (no-helmet) | ≥ 80% | **65.2%** | ❌ low |
| FNR (no-helmet) | ≤ 10% | **34.8%** | ❌ too high |
| Inference speed | ≥ 15 FPS | **~98 FPS** | ✅ (10.2ms/img) |

### Key Findings

1. **Overall performance (77.8% mAP@50)** is near the 80% target — the model learns general PPE detection well for majority classes.
2. **Severe minority class degradation:**
   - `no-helmet` mAP@50 is only **59.2%** — 20pp below target.
   - `no-vest` mAP@50 is **63.0%** — 17pp below target.
   - `no-helmet` recall is **65.2%** — meaning ~35% of missing-helmet violations go undetected.
3. **The model is biased toward safe conditions** (helmet/vest have much higher metrics), which is dangerous in a safety context — missed violations are worse than false alarms.
4. **Validation metrics are significantly higher than test metrics** (91% vs 77.8% mAP@50), suggesting slight overfitting or distribution shift between val/test splits.

## Next Steps

1. **Inverse-frequency class weights** — to force the model to pay more attention to minority classes.
2. **Focal loss** (`fl_gamma=1.5`) — further penalizes misclassified hard examples.
3. **Oversample minority images** — duplicate training samples containing `no-helmet`/`no-vest`.
4. **Compliance pipeline** — build the full decision engine (person→PPE association) to measure FNR on the compliance metric directly.
5. **Confidence threshold tuning** — lower thresholds for minority classes to boost recall at the cost of precision.

## References

- Notebook: `notebooks/2_0_alg_baseline_training.ipynb`
- Trained model: `models/baseline/weights/best.pt`
- Dataset: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb
- YOLOv8 docs: https://docs.ultralytics.com
