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
):
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    logger.info(f"Training YOLOv8s on {data_yaml}")
    logger.info(f"Epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")
    logger.info("NOTE: No class imbalance handling — this is a raw baseline.")

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
    for i, name in enumerate(metrics.ap_class_index):
        logger.info(
            f"  {metrics.names[i]:<15s}  "
            f"mAP@50={metrics.box.mp[i]:.3f}  "
            f"mAP@50:95={metrics.box.mr[i]:.3f}  "
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
