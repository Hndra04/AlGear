# Resplit Oversampling Report (Notebook 3.1)

## Objective

Apply **conservative oversampling** (3× no-helmet, 2× no-vest) on the **resplit dataset** (70:15:15 stratified split from notebook 3.0) and train YOLOv8s to evaluate whether a fresh data split improves minority class detection compared to the original split used in notebook 2.2.

**Key differences from 2.2:**
- Data is resplit from scratch (no original train/val/test split)
- Same oversampling strategy (3×/2×, no copy_paste)
- Stratified split ensures balanced class distribution across splits

## Configuration

| Parameter        | Value                              |
|------------------|------------------------------------|
| Model            | YOLOv8s                            |
| Pretrained       | `yolov8s.pt`                       |
| Epochs           | 50                                 |
| Batch size       | 16                                 |
| Image size       | 640×640                            |
| Optimizer        | AdamW (lr=0.001111, momentum=0.9)  |
| Patience         | 20                                 |
| Device           | GPU (Tesla T4, Colab)              |
| AMP              | enabled                            |
| `copy_paste`     | **0.0** (disabled)                 |

## Dataset Resplit (70:15:15)

| Split | Images | Percentage |
|-------|-------:|-----------:|
| Train | 834    | 69.2%      |
| Val   | 174    | 14.4%      |
| Test  | 198    | 16.4%      |
| **Total** | **1,206** | **100%** |

### Class Distribution (Resplit Train Set)

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| helmet | 1,736 | 376 | 431 |
| no-helmet | 80 | 11 | 38 |
| no-vest | 623 | 126 | 143 |
| person | 1,937 | 396 | 484 |
| vest | 917 | 191 | 235 |

**Pre-oversampling imbalance ratio (helmet : no-helmet):** 21.7×  
**Pre-oversampling imbalance ratio (vest : no-vest):** 1.5×

## Oversampling Strategy

| Class         | Original Images | Multiplier | Augmented Copies |
|---------------|----------------:|----------:|-----------------:|
| no-helmet     | 43              | 3×        | ~86              |
| no-vest       | 326             | 2×        | ~326             |

**Post-oversampling dataset:**
- 834 originals + 387 augmented = **1,221 total training images**

### Post-Oversampling Imbalance Ratios

| Metric | Before | After |
|--------|-------:|------:|
| helmet : no-helmet | 21.7× | 10.8× |
| vest : no-vest | 1.5× | 0.8× |

## Test Set Evaluation (best.pt)

| Class      | Precision | Recall | mAP@50 | mAP@50:95 |
|------------|----------:|-------:|-------:|----------:|
| helmet     | 0.777     | 0.930  | 0.932  | 0.510     |
| no-helmet  | 0.752     | 0.638  | 0.660  | 0.257     |
| no-vest    | 0.544     | 0.819  | 0.663  | 0.340     |
| person     | 0.683     | 0.957  | 0.928  | 0.599     |
| vest       | 0.644     | 0.893  | 0.865  | 0.466     |
| **Overall** | **0.680** | **0.847** | **0.810** | **0.434** |

## Comparison: Conservative Oversampling (2.2 vs 3.1)

| Class     | 2.2 Conservative | 3.1 Resplit+Conservative | Δ |
|-----------|----------------:|-------------------------:|--:|
| helmet    | 0.928           | 0.932                    | +0.004 |
| no-helmet | 0.483           | **0.660**                | **+0.177** |
| no-vest   | 0.689           | 0.663                    | -0.026 |
| person    | 0.870           | 0.928                    | +0.058 |
| vest      | 0.839           | 0.865                    | +0.026 |
| **Overall** | **0.762**     | **0.810**                | **+0.048** |

## Comparison Across All Experiments

| Class     | Baseline (2.0) | 10× Oversampled (2.1) | Conservative (2.2) | Resplit+Conservative (3.1) |
|-----------|---------------:|----------------------:|-------------------:|---------------------------:|
| helmet    | 0.929          | 0.926                 | 0.928              | **0.932**                  |
| no-helmet | 0.592          | 0.583                 | 0.483              | **0.660**                  |
| no-vest   | 0.630          | 0.602                 | **0.689**          | 0.663                      |
| person    | 0.890          | 0.873                 | 0.870              | **0.928**                  |
| vest      | 0.850          | 0.823                 | 0.839              | **0.865**                  |
| **Overall** | **0.778**    | **0.761**             | **0.762**          | **0.810**                  |

## Training Summary

- **Best epoch:** epoch 19 (mAP@50=0.861 on val)
- **Early stopping:** triggered at epoch 39 (patience=20)
- **Total time:** 0.276 hours (~16.6 minutes)

## Target Metrics Evaluation

Based on the project targets defined in `README.md`:

| Metric | Target | Achieved | Status |
|--------|-------:|---------:|:------:|
| mAP@50 (overall, PPE Detection) | ≥ 80% | **81.0%** | ✅ Met |
| mAP@50 (no-helmet) | ≥ 80% | **66.0%** | ❌ −14pp |
| mAP@50 (no-vest) | ≥ 80% | **66.3%** | ❌ −13.7pp |
| Recall (no-helmet) | ≥ 90% | **63.8%** | ❌ −26.2pp |
| Recall (no-vest) | ≥ 90% | **81.9%** | ❌ −8.1pp |
| FNR (no-helmet) | ≤ 10% | **36.2%** | ❌ too high |
| FNR (no-vest) | ≤ 10% | **18.1%** | ❌ too high |
| PPE Compliance Accuracy | ≥ 85% | — | ⏳ Measure in notebook 4.0 |
| People Counting Accuracy (MAE) | ≤ 2 persons | — | ⏳ Measure in notebook 4.0 |
| Inference Speed | ≥ 15 FPS | ~96 FPS | ✅ (10.4ms/img) |

> **FNR** = 1 − Recall. A 36.2% FNR for `no-helmet` means ~36 out of every 100 missing-helmet violations go undetected.

### Analysis

1. **Overall mAP@50 met the 80% target** (81.0%) — driven by strong performance on majority classes (`helmet` 0.932, `person` 0.928, `vest` 0.865).

2. **Minority class mAP still below target.** Both `no-helmet` (66.0%) and `no-vest` (66.3%) remain well below the 80% target, though improved from notebook 2.2.

3. **FNR is critically high for `no-helmet`.** At 36.2%, more than 1 in 3 missing-helmet violations are missed — unacceptable for a safety compliance system where missed detections are more dangerous than false alarms.

4. **`no-vest` FNR approaching target.** At 18.1%, it's closer to the 10% threshold. A recall of 81.9% shows the model detects most no-vest cases but needs further improvement.

5. **Two metrics now implemented** — PPE Compliance Accuracy and People Counting MAE are ready to measure via `algear/metrics.py` and notebook `4.0-alg-pipeline-metrics.ipynb`. Run notebook 4.0 on the test set to obtain final values.

### Key Takeaway

**mAP@50 overall is deceptively strong.** The 81% overall mAP hides severe minority class gaps. For a safety system, per-class metrics matter more than aggregate — a model that misses 36% of helmet violations is not deployment-ready regardless of its overall score.

## References

- Notebook: `notebooks/3.1-alg-resplit-oversampling.ipynb`
- Metrics notebook: `notebooks/4.0-alg-pipeline-metrics.ipynb`
- Pipeline module: `algear/modeling/pipeline.py`
- Metrics module: `algear/metrics.py`
- Resplit notebook: `notebooks/3.0-alg-dataset-resplit.ipynb`
- Trained model: `models/resplit-oversample-conservative/weights/best.pt`
- Dataset: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb
- YOLOv8 docs: https://docs.ultralytics.com
