from pathlib import Path

from loguru import logger
import typer

from algear.config import MODELS_DIR, ROBOFLOW_DIR

app = typer.Typer()


@app.command()
def train_yolov8(
    data_yaml: Path = ROBOFLOW_DIR / "data.yaml",
    model_name: str = "yolov8s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    device: str = "cpu",
    output_dir: Path = MODELS_DIR,
    oversample_data: bool = False,
    copy_paste: float = 0.0,
    mixup: float = 0.0,
):
    from ultralytics import YOLO

    from algear.config import PROCESSED_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if oversample_data:
        oversampled_dir = PROCESSED_DATA_DIR / "construction-site-safety-oversampled"
        oversampled_yaml = oversampled_dir / "data.yaml"
        if oversampled_yaml.exists():
            data_yaml = oversampled_yaml
            logger.info(f"Using oversampled dataset: {data_yaml}")
        else:
            logger.warning(
                f"Oversampled dataset not found at {oversampled_yaml}. "
                "Run 'python -m algear.dataset oversample' first. Falling back to raw."
            )

    model = YOLO(model_name)

    logger.info(f"Training YOLOv8s on {data_yaml}")
    logger.info(f"Epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")

    strategies = []
    if oversample_data:
        strategies.append("repeat-factor oversampling")
    if copy_paste > 0:
        strategies.append(f"copy_paste={copy_paste}")
    if mixup > 0:
        strategies.append(f"mixup={mixup}")
    if strategies:
        logger.info(f"Imbalance strategies: {', '.join(strategies)}")
    else:
        logger.info("NOTE: No class imbalance handling — raw training.")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        project=str(output_dir),
        name="baseline",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    if copy_paste > 0:
        train_kwargs["copy_paste"] = copy_paste
    if mixup > 0:
        train_kwargs["mixup"] = mixup

    results = model.train(**train_kwargs)

    best_model_path = output_dir / "baseline" / "weights" / "best.pt"
    if best_model_path.exists():
        logger.success(f"Best model saved to {best_model_path}")

    return results


@app.command()
def evaluate(
    model_path: Path = MODELS_DIR / "baseline" / "weights" / "best.pt",
    data_yaml: Path = ROBOFLOW_DIR / "data.yaml",
    split: str = "test",
    device: str = "cpu",
):
    from ultralytics import YOLO

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise typer.Exit(code=1)

    logger.info(f"Evaluating {model_path} on {split} split")
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split=split, device=device)

    logger.info("=== Per-Class Metrics ===")
    for i, cls_idx in enumerate(metrics.ap_class_index):
        logger.info(
            f"  {model.names[cls_idx]:<15s}  "
            f"mAP@50={metrics.box.ap50[i]:.3f}  "
            f"mAP@50:95={metrics.box.ap[i]:.3f}  "
            f"Precision={metrics.box.p[i]:.3f}  "
            f"Recall={metrics.box.r[i]:.3f}"
        )

    logger.info(f"mAP@50:    {metrics.box.map50:.3f}")
    logger.info(f"mAP@50:95: {metrics.box.map:.3f}")
    logger.info(f"Precision: {metrics.box.mp:.3f}")
    logger.info(f"Recall:    {metrics.box.mr:.3f}")

    return metrics


@app.command()
def main(
    action: str = typer.Argument("train", help="Action: train or evaluate"),
):
    if action == "train":
        train_yolov8()
    elif action == "evaluate":
        evaluate()
    else:
        logger.error(f"Unknown action: {action}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
