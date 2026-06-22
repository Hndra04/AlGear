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
def main(
    action: str = typer.Argument("download", help="Action: download or prepare"),
):
    if action == "download":
        download_roboflow()
    elif action == "prepare":
        prepare()
    else:
        logger.error(f"Unknown action: {action}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
