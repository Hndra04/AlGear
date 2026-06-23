"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_normalized: list[float] = Field(min_length=4, max_length=4)


class WorkerResponse(BaseModel):
    person_idx: int
    head_ppe: str
    body_ppe: str
    has_helmet: bool
    has_no_helmet: bool
    has_vest: bool
    has_no_vest: bool
    is_compliant: bool
    bbox: list[float] = Field(min_length=4, max_length=4)


class ImageResults(BaseModel):
    person_count: int
    compliant_count: int
    violation_count: int
    compliance_rate: float


class ImageInferenceResponse(BaseModel):
    success: bool = True
    filename: str
    image_width: int
    image_height: int
    inference_ms: float
    results: ImageResults
    workers: list[WorkerResponse]
    detections: list[DetectionResponse]
    annotated_image_base64: str


class VideoFrameResult(BaseModel):
    frame_idx: int
    person_count: int
    compliant_count: int
    violation_count: int
    compliance_rate: float


class VideoSummary(BaseModel):
    total_frames: int
    avg_person_count: float
    avg_compliant_count: float
    avg_violation_count: float
    avg_compliance_rate: float


class VideoInferenceResponse(BaseModel):
    success: bool = True
    filename: str
    total_frames: int
    fps: float
    video_width: int
    video_height: int
    summary: VideoSummary
    frame_results: list[VideoFrameResult]
    annotated_video_base64: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    classes: dict[int, str]
    confidence: float
    device: str
