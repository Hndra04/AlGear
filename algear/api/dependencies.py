"""Pipeline singleton and dependency management for FastAPI."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from algear.config import MODELS_DIR
from algear.core import CompliancePipeline, PipelineConfig

_DEFAULT_MODEL = MODELS_DIR / "resplit-oversample-conservative" / "weights" / "best.pt"

_pipeline: CompliancePipeline | None = None
_model_path: Path | None = None


def get_pipeline(
    model_path: Path | str | None = None,
    conf: float = 0.25,
    device: str = "cpu",
) -> CompliancePipeline:
    """Get or initialise the global CompliancePipeline singleton."""
    global _pipeline, _model_path

    resolved = Path(model_path) if model_path else _DEFAULT_MODEL

    if _pipeline is not None and _model_path == resolved:
        return _pipeline

    logger.info(f"Loading model from {resolved}")
    config = PipelineConfig(conf=conf, device=device)
    _pipeline = CompliancePipeline(model_path=resolved, config=config)
    _model_path = resolved
    return _pipeline


def get_model_path() -> Path:
    return _model_path or _DEFAULT_MODEL


def is_model_loaded() -> bool:
    return _pipeline is not None
