"""Reproducible evaluation script for AIGear PPE compliance pipeline.

Runs detection-level and pipeline-level evaluation with fixed seeds,
saves a JSON report containing all metrics and configuration for
full reproducibility.

Automatically downloads and prepares the dataset if not found locally.

Usage:
    python -m algear.evaluate [OPTIONS]

Examples:
    # Default: evaluate best resplit model on test set
    python -m algear.evaluate

    # Evaluate baseline model with custom confidence threshold
    python -m algear.evaluate --model models/baseline/weights/best.pt --conf 0.3

    # Evaluate on validation set instead of test
    python -m algear.evaluate --split valid
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from algear.config import MODELS_DIR, PROCESSED_DATA_DIR, ROBOFLOW_DIR

# ── Reproducibility seeds ─────────────────────────────────────────────

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Environment capture ────────────────────────────────────────────────


def capture_environment() -> dict:
    """Capture environment info for reproducibility."""
    env_info = {
        "python": sys.version,
        "platform": sys.platform,
        "numpy": np.__version__,
        "seed": SEED,
    }
    try:
        import torch

        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            env_info["cuda_device"] = torch.cuda.get_device_name(0)
    except (ImportError, AssertionError):
        env_info["torch"] = "not installed"

    try:
        import ultralytics

        env_info["ultralytics"] = ultralytics.__version__
    except ImportError:
        env_info["ultralytics"] = "not installed"

    return env_info


# ── Data preparation ───────────────────────────────────────────────────


RESPLIT_DIR = PROCESSED_DATA_DIR / "construction-site-safety-resplit"


def ensure_data_ready() -> None:
    """Check if dataset exists; download and resplit if missing.

    Flow:
        1. Check if resplit data exists at data/processed/construction-site-safety-resplit/
        2. If missing, check if raw data exists at data/raw/construction-safety-gsnvb/
        3. If raw missing, download from Roboflow (requires ROBOFLOW_API_KEY)
        4. Run stratified resplit (70:15:15)
    """
    if _is_resplit_ready():
        logger.info(f"Dataset found at {RESPLIT_DIR}")
        return

    logger.warning("Resplit dataset not found. Preparing...")

    if not _is_raw_ready():
        logger.info("Raw dataset not found. Downloading from Roboflow...")
        _download_dataset()

    logger.info("Running stratified resplit (70:15:15)...")
    _resplit_dataset()

    if not _is_resplit_ready():
        logger.error("Data preparation failed — resplit directory missing after run.")
        raise typer.Exit(code=1)

    logger.success("Dataset ready for evaluation.")


def _is_resplit_ready() -> bool:
    """Check if resplit dataset has the required splits."""
    for split in ["train", "val", "test"]:
        img_dir = RESPLIT_DIR / split / "images"
        lbl_dir = RESPLIT_DIR / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            return False
        if not any(img_dir.iterdir()):
            return False
    return True


def _is_raw_ready() -> bool:
    """Check if raw Roboflow dataset exists."""
    data_yaml = ROBOFLOW_DIR / "data.yaml"
    return data_yaml.exists() and ROBOFLOW_DIR.is_dir()


def _download_dataset() -> None:
    """Download dataset from Roboflow."""
    from algear.config import ROBOFLOW_API_KEY

    if not ROBOFLOW_API_KEY:
        logger.error(
            "ROBOFLOW_API_KEY not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your Roboflow API key to .env\n"
            "  3. Get your key at https://app.roboflow.com/settings/api"
        )
        raise typer.Exit(code=1)

    from algear.dataset import download_roboflow

    download_roboflow()


def _resplit_dataset() -> None:
    """Run stratified resplit on raw dataset."""
    from algear.dataset import resplit

    resplit()


# ── Detection-level evaluation ─────────────────────────────────────────


def run_detection_eval(
    model_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
) -> dict:
    """Run ultralytics model.val() and return structured metrics."""
    from ultralytics import YOLO

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    logger.info(f"Running detection evaluation on '{split}' split...")
    metrics = model.val(data=str(data_yaml), split=split, device=device, verbose=False)

    per_class = []
    for i, cls_idx in enumerate(metrics.ap_class_index):
        per_class.append(
            {
                "class_id": int(cls_idx),
                "class_name": model.names[cls_idx],
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50_95": float(metrics.box.ap[i]),
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
            }
        )

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }


# ── Pipeline-level evaluation ──────────────────────────────────────────


def run_pipeline_eval(
    model_path: Path,
    image_dir: Path,
    gt_labels_dir: Path,
    conf: float,
    device: str,
) -> dict:
    """Run batch inference + pipeline metrics (compliance, counting)."""
    from algear.metrics import ppe_compliance_accuracy, people_counting_mae
    from algear.modeling.pipeline import associate_ppe_to_persons
    from algear.modeling.predict import run_inference

    pred_results = run_inference(model_path, image_dir, conf=conf, device=device)

    # People counting MAE
    gt_counts = []
    pred_counts = []
    for result in pred_results:
        stem = Path(result["image_name"]).stem
        label_path = gt_labels_dir / f"{stem}.txt"

        gt_label = _parse_yolo_label_simple(label_path)
        gt_person = int((gt_label[:, 0] == 3).sum()) if len(gt_label) > 0 else 0

        workers = associate_ppe_to_persons(
            result["detections"], result["img_w"], result["img_h"]
        )
        gt_counts.append(gt_person)
        pred_counts.append(len(workers))

    counting_metrics = people_counting_mae(gt_counts, pred_counts)

    # PPE compliance accuracy
    compliance_metrics = ppe_compliance_accuracy(gt_labels_dir, pred_results, image_dir)

    return {
        "people_counting_mae": counting_metrics["mae"],
        "people_counting_mean_gt": counting_metrics["mean_gt"],
        "people_counting_mean_pred": counting_metrics["mean_pred"],
        "people_counting_max_error": counting_metrics["max_error"],
        "ppe_compliance_accuracy": compliance_metrics["accuracy"],
        "total_persons_evaluated": compliance_metrics["total_persons"],
        "correct_classifications": compliance_metrics["correct"],
        "per_image_counting": counting_metrics["per_image"],
        "per_image_compliance": compliance_metrics["per_image"],
    }


def _parse_yolo_label_simple(label_path: Path) -> np.ndarray:
    """Read a YOLO label file."""
    if not label_path.exists():
        return np.empty((0, 5))
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(parts[0])] + [float(x) for x in parts[1:5]])
    return np.array(rows) if rows else np.empty((0, 5))


# ── CLI ────────────────────────────────────────────────────────────────

app = typer.Typer()


@app.command()
def evaluate(
    model_path: Path = typer.Option(
        MODELS_DIR / "resplit-oversample-conservative" / "weights" / "best.pt",
        "--model",
        "-m",
        help="Path to .pt model file",
    ),
    data_yaml: Path = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to data.yaml (default: auto-detect from resplit)",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Dataset split: train / val / test",
    ),
    conf: float = typer.Option(
        0.25,
        "--conf",
        "-c",
        help="Confidence threshold for inference",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device: cpu / cuda / 0",
    ),
    output_dir: Path = typer.Option(
        MODELS_DIR / "evaluation",
        "--output",
        "-o",
        help="Directory to save evaluation report",
    ),
    seed: int = typer.Option(
        SEED,
        "--seed",
        help="Random seed for reproducibility",
    ),
    skip_prepare: bool = typer.Option(
        False,
        "--skip-prepare",
        help="Skip automatic data download/resplit",
    ),
) -> None:
    """Run reproducible evaluation of the PPE compliance pipeline.

    Automatically downloads and prepares the dataset if not found locally.
    Evaluates both detection-level metrics (mAP, Precision, Recall)
    and pipeline-level metrics (PPE Compliance Accuracy, People Counting MAE).
    Saves a JSON report with all results, configuration, and environment info.
    """
    global SEED
    SEED = seed
    set_global_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Auto-prepare data if needed ────────────────────────────────────
    if not skip_prepare:
        ensure_data_ready()

    # ── Resolve data paths ─────────────────────────────────────────────
    if data_yaml is None:
        data_yaml = RESPLIT_DIR / "data.yaml"
        if not data_yaml.exists():
            data_yaml = ROBOFLOW_DIR / "data.yaml"

    logger.info("=" * 60)
    logger.info("AIGear Evaluation — Reproducible Run")
    logger.info("=" * 60)
    logger.info(f"Model:  {model_path}")
    logger.info(f"Data:   {data_yaml}")
    logger.info(f"Split:  {split}")
    logger.info(f"Conf:   {conf}")
    logger.info(f"Seed:   {seed}")

    # ── Detection-level evaluation ─────────────────────────────────────
    logger.info("\n[1/2] Detection-level evaluation (mAP, Precision, Recall)")
    detection_metrics = run_detection_eval(model_path, data_yaml, split, device)

    # ── Pipeline-level evaluation ──────────────────────────────────────
    # Map split names: resplit uses "val", some pipelines use "valid"
    split_dir = split
    if split == "valid":
        split_dir = "val"

    image_dir = RESPLIT_DIR / split_dir / "images"
    gt_labels_dir = RESPLIT_DIR / split_dir / "labels"

    if not image_dir.exists():
        logger.warning(f"Image dir not found: {image_dir}, skipping pipeline eval")
        pipeline_metrics = None
    else:
        logger.info("\n[2/2] Pipeline-level evaluation (Compliance, Counting)")
        pipeline_metrics = run_pipeline_eval(model_path, image_dir, gt_labels_dir, conf, device)

    # ── Build report ───────────────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "seed": seed,
        "config": {
            "model_path": str(model_path),
            "data_yaml": str(data_yaml),
            "split": split,
            "conf": conf,
            "device": device,
        },
        "environment": capture_environment(),
        "detection_metrics": detection_metrics,
    }

    if pipeline_metrics is not None:
        report["pipeline_metrics"] = {
            "people_counting_mae": pipeline_metrics["people_counting_mae"],
            "people_counting_mean_gt": pipeline_metrics["people_counting_mean_gt"],
            "people_counting_mean_pred": pipeline_metrics["people_counting_mean_pred"],
            "people_counting_max_error": pipeline_metrics["people_counting_max_error"],
            "ppe_compliance_accuracy": pipeline_metrics["ppe_compliance_accuracy"],
            "total_persons_evaluated": pipeline_metrics["total_persons_evaluated"],
            "correct_classifications": pipeline_metrics["correct_classifications"],
        }
        report["pipeline_metrics_per_image"] = {
            "counting": pipeline_metrics["per_image_counting"],
            "compliance": pipeline_metrics["per_image_compliance"],
        }

    # ── Target compliance check ────────────────────────────────────────
    targets = {
        "mAP50 >= 0.80": detection_metrics["mAP50"] >= 0.80,
        "precision >= 0.75": detection_metrics["precision"] >= 0.75,
        "recall >= 0.75": detection_metrics["recall"] >= 0.75,
    }
    if pipeline_metrics is not None:
        targets["PPE_compliance >= 0.85"] = pipeline_metrics["ppe_compliance_accuracy"] >= 0.85
        targets["counting_MAE <= 2"] = pipeline_metrics["people_counting_mae"] <= 2.0

    report["targets"] = targets

    # ── Save report ────────────────────────────────────────────────────
    report_path = output_dir / f"eval_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.success(f"Report saved: {report_path}")

    # ── Print summary ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  mAP@50:              {detection_metrics['mAP50']:.4f}")
    logger.info(f"  mAP@50:95:           {detection_metrics['mAP50_95']:.4f}")
    logger.info(f"  Precision:           {detection_metrics['precision']:.4f}")
    logger.info(f"  Recall:              {detection_metrics['recall']:.4f}")
    logger.info("  Per-class:")
    for cls in detection_metrics["per_class"]:
        logger.info(
            f"    {cls['class_name']:<15s}  "
            f"mAP50={cls['mAP50']:.3f}  "
            f"P={cls['precision']:.3f}  "
            f"R={cls['recall']:.3f}"
        )

    if pipeline_metrics is not None:
        logger.info(f"  PPE Compliance Acc:  {pipeline_metrics['ppe_compliance_accuracy']:.4f}")
        logger.info(f"  Counting MAE:        {pipeline_metrics['people_counting_mae']:.4f}")

    logger.info("\n  Target compliance:")
    for target, passed in targets.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"    [{status}] {target}")

    all_passed = all(targets.values())
    logger.info(f"\n  Overall: {'ALL TARGETS MET' if all_passed else 'SOME TARGETS MISSED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
