# Oversampling Conservative Report (Notebook 2.2)

## Objective

Retry oversampling with **conservative multipliers** and **no copy_paste** to avoid the over-augmentation issues observed in the 10× experiment (notebook 2.1).

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

## Oversampling Strategy

| Class         | Original Images | Multiplier | Augmented Copies |
|---------------|----------------:|----------:|-----------------:|
| no-helmet     | 49              | 3×        | ~98              |
| no-vest       | 390             | 2×        | ~390             |

**Changes from 2.1:**
- Multiplier reduced: 3× (was 10×) for `no-helmet`, 2× (was 3×) for `no-vest`
- `copy_paste` disabled (was 0.3) — no object pasting during training
- Only YOLOv8's default augmentations (mosaic, flip, HSV, scale, translate) are active

## Dataset After Oversampling

| Split | Images | Instances |
|-------|-------:|----------:|
| Train | 1,455  | —         |
| Val   | 119    | 715       |

**Post-oversampling imbalance ratio (helmet : no-helmet):** 11.0× (down from 22.5×)  
**Post-oversampling imbalance ratio (vest : no-vest):** 0.8× (reversed, but less extreme than 2.1's 0.6×)

## Test Set Evaluation (best.pt)

| Class      | Precision | Recall | mAP@50 | mAP@50:95 |
|------------|----------:|-------:|-------:|----------:|
| helmet     | 0.827     | 0.928  | 0.928  | 0.483     |
| no-helmet  | 0.638     | 0.588  | 0.483  | 0.203     |
| no-vest    | 0.660     | 0.796  | 0.689  | 0.290     |
| person     | 0.804     | 0.911  | 0.870  | 0.550     |
| vest       | 0.844     | 0.853  | 0.839  | 0.461     |
| **Overall** | **0.755** | **0.815** | **0.762** | **0.398** |

## Comparison Across All Experiments

| Class     | Baseline (2.0) | 10× Oversampled (2.1) | Conservative (2.2) |
|-----------|---------------:|----------------------:|-------------------:|
| helmet    | 0.929          | 0.926                 | 0.928              |
| no-helmet | 0.592          | 0.583                 | **0.483**          |
| no-vest   | 0.630          | 0.602                 | **0.689**          |
| person    | 0.890          | 0.873                 | 0.870              |
| vest      | 0.850          | 0.823                 | 0.839              |
| **Overall** | **0.778**    | **0.761**             | **0.762**          |

## Training Summary

- **Best epoch:** epoch 42 (mAP@50=0.911 on val)
- **Early stopping:** not triggered (trained all 50 epochs)
- **Total time:** 0.421 hours

## Analysis

### Results vs Hypotheses

1. **Lower multipliers reduced memorization.** The conservative experiment trained all 50 epochs (no early stopping), compared to 2.1 which stopped at epoch 42. This suggests less overfitting.

2. **`no-vest` improved significantly.** mAP@50 rose from 0.630 (baseline) → 0.602 (10× oversampled) → **0.689** (conservative). This is the only minority class that improved across all experiments.

3. **`no-helmet` dropped further.** mAP@50 went from 0.592 (baseline) → 0.583 (10×) → **0.483** (conservative). The 3× multiplier may still be too aggressive, or the 49 source images are too few to learn from regardless of multiplier.

4. **Overall mAP remained flat** at ~0.76, well below the baseline's 0.778. Neither oversampling approach improved overall performance.

### Key Takeaway

Oversampling alone is not solving the minority class problem. The `no-helmet` class (49 images) is too small for augmentation-based oversampling to help — the model either memorizes (10×) or doesn't see enough variety (3×). A different approach is needed.

## References

- Notebook: `notebooks/2.2-alg-oversampling-conservative.ipynb`
- Trained model: `models/oversample-conservative/weights/best.pt`
- Dataset: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb
- YOLOv8 docs: https://docs.ultralytics.com
