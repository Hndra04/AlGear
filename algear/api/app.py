"""FastAPI application for PPE Compliance Inference."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from algear.api.dependencies import get_model_path, get_pipeline, is_model_loaded
from algear.api.inference import infer_image, infer_video
from algear.api.schemas import (
    HealthResponse,
    ImageInferenceResponse,
    ModelInfoResponse,
    VideoInferenceResponse,
)
from algear.api.utils import validate_image_file, validate_video_file
from algear.core import CLASS_NAMES

app = FastAPI(
    title="AlGear PPE Compliance Inference API",
    description=(
        "Real-time PPE (helmet, safety vest) compliance detection using YOLOv8. "
        "Upload an image or video to get compliance analysis results."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load the model at startup."""
    logger.info("Starting AlGear Inference API...")
    try:
        get_pipeline()
        logger.success("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        model_path=str(get_model_path()),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Return model metadata and class definitions."""
    pipeline = get_pipeline()
    return ModelInfoResponse(
        model_name=get_model_path().stem,
        model_path=str(get_model_path()),
        classes=CLASS_NAMES,
        confidence=pipeline.config.conf,
        device=pipeline.config.device,
    )


@app.post("/infer/image", response_model=ImageInferenceResponse)
async def inference_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WebP)"),
):
    """Run PPE compliance inference on a single image.

    Returns JSON results with detections, compliance status, and an annotated image.
    """
    content = await file.read()

    error = validate_image_file(file.filename or "unknown.jpg", len(content))
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        pipeline = get_pipeline()
        result = infer_image(pipeline, content, file.filename or "unknown.jpg")
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/infer/video", response_model=VideoInferenceResponse)
async def inference_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV, WebM)"),
    max_frames: int | None = None,
):
    """Run PPE compliance inference on a video file.

    Processes each frame and returns per-frame results plus a summary.
    The annotated video is returned as a base64-encoded MP4.
    """
    content = await file.read()

    error = validate_video_file(file.filename or "unknown.mp4", len(content))
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        pipeline = get_pipeline()
        result = infer_video(
            pipeline, content, file.filename or "unknown.mp4", max_frames=max_frames
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Video inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
