# Oversampling 10× Report (Notebook 2.1)

## Objective

Address the class imbalance problem identified in the baseline by applying **repeat-factor oversampling** with per-copy augmentation for minority classes (`no-helmet`, `no-vest`).

## Configuration

| Parameter        | Value                              |
|------------------|------------------------------------|
| Model            | YOLOv8s                            |
| Pretrained       | `yolov8s.pt`                       |
| Epochs           | 50 (early stopped at epoch 42)     |
| Batch size       | 16                                 |
| Image size       | 640×640                            |
| Optimizer        | AdamW (lr=0.001111, momentum=0.9)  |
| Patience         | 20                                 |
| Device           | GPU (Tesla T4, Colab)              |
| AMP              | enabled                            |
| `copy_paste`     | 0.3                                |

## Oversampling Strategy

| Class         | Original Images | Multiplier | Augmented Copies |
|---------------|----------------:|----------:|-----------------:|
| no-helmet     | 49              | 10×       | ~441             |
| no-vest       | 390             | 3×        | ~780             |

- **Pre-augmentation transforms** (applied to each copy independently): brightness/contrast, hue/saturation/value jitter, Gaussian blur, Gaussian noise, CLAHE, grayscale conversion, gamma correction.
- **YOLOv8 built-in augmentations** (active during training): mosaic, flip, HSV jitter, scale, translate, `copy_paste=0.3`.

## Dataset After Oversampling

| Split | Images | Instances |
|-------|-------:|----------:|
| Train | 2,158  | —         |
| Val   | 119    | 715       |

**Post-oversampling imbalance ratio (helmet : no-helmet):** 4.9× (down from 22.5×)  
**Post-oversampling imbalance ratio (vest : no-vest):** 0.6× (reversed — no-vest now exceeds vest)

## Test Set Evaluation (best.pt)

| Class      | Precision | Recall | mAP@50 | mAP@50:95 |
|------------|----------:|-------:|-------:|----------:|
| helmet     | 0.823     | 0.933  | 0.926  | 0.469     |
| no-helmet  | 0.771     | 0.562  | 0.583  | 0.199     |
| no-vest    | 0.568     | 0.734  | 0.602  | 0.243     |
| person     | 0.764     | 0.916  | 0.873  | 0.561     |
| vest       | 0.797     | 0.822  | 0.823  | 0.424     |
| **Overall** | **0.745** | **0.793** | **0.761** | **0.379** |

## Comparison vs Baseline

| Class     | Baseline mAP@50 | Oversampled mAP@50 | Δ        |
|-----------|----------------:|-------------------:|---------:|
| helmet    | 0.929           | 0.926              | −0.003   |
| no-helmet | 0.592           | 0.583              | −0.009   |
| no-vest   | 0.630           | 0.602              | −0.028   |
| person    | 0.890           | 0.873              | −0.017   |
| vest      | 0.850           | 0.823              | −0.027   |
| **Overall** | **0.778**     | **0.761**          | **−0.017** |

**Result: Overall mAP dropped by 1.7pp. No class improved.**

## Training Summary

- **Best epoch:** epoch 22 (mAP@50=0.891 on val)
- **Early stopping:** triggered at epoch 42 (no improvement for 20 epochs after epoch 22)
- **Total time:** 0.518 hours (~1.7× baseline time due to 2× more training images)

## Analysis

### Why Did Performance Drop?

1. **10× multiplier on 49 images is excessive.** The model sees variations of the same 49 `no-helmet` images ~10 times per epoch. It memorizes these specific images rather than learning generalizable features for the `no-helmet` class.

2. **Double-augmentation degrades image quality.** Pre-augmented copies (brightness, blur, noise, etc.) receive *additional* YOLOv8 augmentations (mosaic, HSV jitter, scale) during training. The resulting images can be too distorted to be useful training signal.

3. **Class imbalance reversal for vest/no-vest.** The vest:no-vest ratio flipped from 1.4× to 0.6×, creating a new imbalance problem in the opposite direction.

4. **Stacking `copy_paste` on oversampled data** causes minority objects to appear at an inflated rate, further encouraging overfitting to the 49 source images.

5. **Early stopping at epoch 22** indicates the model stopped improving much sooner than the baseline (which trained all 50 epochs), a hallmark of overfitting.

## Next Steps (see Notebook 2.2)

- Reduce multiplier to **3× for no-helmet, 2× for no-vest**
- Disable `copy_paste` (set to 0.0) to avoid stacking augmentation strategies

**2.2 results:** `no-vest` improved from 0.602 → **0.689**, but `no-helmet` dropped from 0.583 → **0.483**. Overall mAP remained flat at ~0.76. See `docs/oversampling-conservative-report.md` for full details.

## References

- Notebook: `notebooks/2.1-alg-oversampling.ipynb`
- Trained model: `models/oversample-10x/weights/best.pt`
- Dataset: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb
- YOLOv8 docs: https://docs.ultralytics.com
