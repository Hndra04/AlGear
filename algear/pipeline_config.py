"""Pipeline configuration loader.

Reads a YAML config file or returns sensible defaults.

Example config.yaml:
    model: models/resplit-oversample-conservative/weights/best.pt
    conf: 0.25
    iou: 0.45
    imgsz: 640
    device: cpu
    zones:
      - name: zone_a
        polygon: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
      - name: zone_b
        polygon: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
"""

from pathlib import Path

import yaml
from loguru import logger

from algear.core import PipelineConfig
from algear.config import MODELS_DIR

DEFAULT_CONFIG = {
    "model": str(MODELS_DIR / "resplit-oversample-conservative" / "weights" / "best.pt"),
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "device": "cpu",
    "zones": [],
}


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config from YAML file, falling back to defaults."""
    cfg = dict(DEFAULT_CONFIG)

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg:
                cfg.update(user_cfg)
            logger.info(f"Loaded config from {p}")
        else:
            logger.warning(f"Config file not found: {p}, using defaults")

    return cfg


def config_to_pipeline(cfg: dict) -> PipelineConfig:
    """Convert a config dict to a PipelineConfig dataclass."""
    return PipelineConfig(
        conf=cfg.get("conf", 0.25),
        iou=cfg.get("iou", 0.45),
        imgsz=cfg.get("imgsz", 640),
        device=cfg.get("device", "cpu"),
        zone_configs=cfg.get("zones") or None,
    )
