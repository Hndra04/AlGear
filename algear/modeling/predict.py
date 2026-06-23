"""Inference and per-image detection extraction for metric evaluation.

Provides functions to run YOLOv8 inference on a directory of images
and return structured results suitable for compliance and counting metrics.
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

from algear.config import MODELS_DIR

app = typer.Typer()


def run_inference(
    model_path: Path,
    image_dir: Path,
    conf: float = 0.25,
    device: str = "cpu",
) -> list[dict]:
    """Run YOLOv8 inference on every image in a directory.

    Parameters
    ----------
    model_path : Path
        Path to the .pt model file.
    image_dir : Path
        Directory containing images.
    conf : float
        Confidence threshold.
    device : str
        'cpu' or 'cuda' / '0'.

    Returns
    -------
    list[dict] — one entry per image:
        'image_name': str
        'detections': np.ndarray (N, 5) — [class_id, cx, cy, w, h] normalised
        'scores': np.ndarray (N,)
        'img_w': int
        'img_h': int
    """
    from ultralytics import YOLO

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
    )

    logger.info(f"Running inference on {len(image_files)} images from {image_dir}")
    results_list = []

    for img_path in tqdm(image_files, desc="Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read {img_path}, skipping")
            continue

        h, w = img.shape[:2]
        preds = model.predict(source=str(img_path), conf=conf, device=device, verbose=False)

        if len(preds) == 0 or preds[0].boxes is None or len(preds[0].boxes) == 0:
            results_list.append(
                {
                    "image_name": img_path.name,
                    "detections": np.empty((0, 5)),
                    "scores": np.empty((0,)),
                    "img_w": w,
                    "img_h": h,
                }
            )
            continue

        boxes = preds[0].boxes
        # boxes.xywhn: normalised (cx, cy, w, h)
        xywhn = boxes.xywhn.cpu().numpy()  # (N, 4)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
        scores = boxes.conf.cpu().numpy()  # (N,)

        detections = np.column_stack([cls_ids, xywhn])  # (N, 5)

        results_list.append(
            {
                "image_name": img_path.name,
                "detections": detections,
                "scores": scores,
                "img_w": w,
                "img_h": h,
            }
        )

    logger.success(f"Inference complete: {len(results_list)} images processed")
    return results_list


@app.command()
def main(
    model_path: Path = MODELS_DIR / "resplit-oversample-conservative" / "weights" / "best.pt",
    image_dir: Path = Path("data/processed/construction-site-safety-resplit/test/images"),
    conf: float = 0.25,
    device: str = "cpu",
):
    """Run inference and print detection summary."""
    results = run_inference(model_path, image_dir, conf=conf, device=device)

    total_detections = sum(len(r["detections"]) for r in results)
    total_persons = sum(
        (r["detections"][:, 0] == 3).sum() if len(r["detections"]) > 0 else 0
        for r in results
    )
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Total person detections: {total_persons}")


if __name__ == "__main__":
    app()
