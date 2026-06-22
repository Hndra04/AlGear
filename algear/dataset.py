from pathlib import Path

from loguru import logger
import typer

from algear.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ROBOFLOW_API_KEY,
    ROBOFLOW_DIR,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)

app = typer.Typer()


@app.command()
def download_roboflow(
    output_dir: Path = ROBOFLOW_DIR,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    api_key: str = ROBOFLOW_API_KEY,
):
    from roboflow import Roboflow

    if not api_key:
        logger.error("ROBOFLOW_API_KEY not set. Add it to .env or export it.")
        raise typer.Exit(code=1)

    logger.info(f"Downloading {workspace}/{project} v{version} to {output_dir}")
    rf = Roboflow(api_key=api_key)
    project_handle = rf.workspace(workspace).project(project)
    dataset = project_handle.version(version).download("yolov8", location=str(output_dir))
    logger.success(f"Dataset downloaded to {output_dir}")
    return dataset


@app.command()
def prepare(
    input_path: Path = ROBOFLOW_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "construction-site-safety",
):
    logger.info("Preparing dataset for training...")
    output_path.mkdir(parents=True, exist_ok=True)

    data_yaml = input_path / "data.yaml"
    if data_yaml.exists():
        import shutil
        shutil.copy2(str(data_yaml), str(output_path / "data.yaml"))
        logger.success(f"Copied data.yaml to {output_path}")
    else:
        logger.warning(f"No data.yaml found at {data_yaml}")

    logger.success("Dataset preparation complete.")


@app.command()
def oversample(
    labels_dir: Path = ROBOFLOW_DIR / "train" / "labels",
    img_dir: Path = ROBOFLOW_DIR / "train" / "images",
    output_dir: Path = PROCESSED_DATA_DIR / "construction-site-safety-oversampled",
    multiplier_no_helmet: int = 10,
    multiplier_no_vest: int = 3,
):
    """
    Oversample minority classes by duplicating images with augmentation.

    Images containing no-helmet (class 1) are duplicated `multiplier_no_helmet`
    times, no-vest (class 2) by `multiplier_no_vest` times.
    Each duplicate gets a different random augmentation.
    """
    from algear.modeling.oversample import (
        create_oversampled_dataset,
        scan_classes,
    )

    class_multipliers = {1: multiplier_no_helmet, 2: multiplier_no_vest}

    image_classes = scan_classes(labels_dir)
    for cls_id, mult in class_multipliers.items():
        match_count = sum(1 for c in image_classes.values() if cls_id in c)
        logger.info(
            f"Class {cls_id} (no-{'helmet' if cls_id == 1 else 'vest'}): "
            f"{match_count} images, multiplier={mult}, "
            f"~{match_count * (mult - 1)} augmented copies"
        )

    logger.info(f"Creating oversampled dataset in {output_dir}")
    create_oversampled_dataset(
        src_dir=img_dir.parent,
        dst_dir=output_dir / "train",
        labels_dir=labels_dir,
        class_multipliers=class_multipliers,
    )

    import yaml

    orig_yaml = ROBOFLOW_DIR / "data.yaml"
    with open(orig_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str((output_dir / "train" / "images").resolve())
    cfg["val"] = str((ROBOFLOW_DIR / "valid" / "images").resolve())
    cfg["test"] = str((ROBOFLOW_DIR / "test" / "images").resolve())

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.success(f"Oversampled dataset ready at {output_dir}")


@app.command()
def main(
    action: str = typer.Argument("download", help="Action: download, prepare, or oversample"),
):
    if action == "download":
        download_roboflow()
    elif action == "prepare":
        prepare()
    elif action == "oversample":
        oversample()
    else:
        logger.error(f"Unknown action: {action}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
