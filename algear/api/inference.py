"""Inference service wrapping the CompliancePipeline for API use."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from algear.api.schemas import (
    DetectionResponse,
    ImageInferenceResponse,
    ImageResults,
    VideoFrameResult,
    VideoInferenceResponse,
    VideoSummary,
    WorkerResponse,
)
from algear.api.utils import (
    encode_image_to_base64,
    encode_video_to_base64,
)
from algear.core import CLASS_NAMES, CompliancePipeline, FrameResult


def _frame_result_to_workers(fr: FrameResult) -> list[WorkerResponse]:
    workers = []
    for idx, wc in enumerate(fr.workers):
        bbox = wc.person_box.tolist()
        workers.append(
            WorkerResponse(
                person_idx=wc.person_idx,
                head_ppe=wc.head_ppe,
                body_ppe=wc.body_ppe,
                has_helmet=wc.has_helmet,
                has_no_helmet=wc.has_no_helmet,
                has_vest=wc.has_vest,
                has_no_vest=wc.has_no_vest,
                is_compliant=wc.is_compliant,
                bbox=bbox,
            )
        )
    return workers


def _frame_result_to_detections(fr: FrameResult) -> list[DetectionResponse]:
    detections = []
    if len(fr.detections) == 0:
        return detections
    for i, det in enumerate(fr.detections):
        cls_id = int(det[0])
        conf = float(fr.scores[i]) if i < len(fr.scores) else 0.0
        detections.append(
            DetectionResponse(
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, str(cls_id)),
                confidence=conf,
                bbox_normalized=[float(det[1]), float(det[2]), float(det[3]), float(det[4])],
            )
        )
    return detections


def infer_image(
    pipeline: CompliancePipeline,
    image_bytes: bytes,
    filename: str,
) -> ImageInferenceResponse:
    """Run inference on a single uploaded image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to decode image: {filename}")

    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    fr = pipeline.process_frame(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated = pipeline.render(frame.copy(), fr)
    annotated_b64 = encode_image_to_base64(annotated)

    return ImageInferenceResponse(
        filename=filename,
        image_width=w,
        image_height=h,
        inference_ms=round(elapsed_ms, 2),
        results=ImageResults(
            person_count=fr.person_count,
            compliant_count=fr.compliant_count,
            violation_count=fr.violation_count,
            compliance_rate=round(fr.compliance_rate, 4),
        ),
        workers=_frame_result_to_workers(fr),
        detections=_frame_result_to_detections(fr),
        annotated_image_base64=annotated_b64,
    )


def infer_video(
    pipeline: CompliancePipeline,
    video_bytes: bytes,
    filename: str,
    max_frames: int | None = None,
) -> VideoInferenceResponse:
    """Run inference on an uploaded video file."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {filename}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tmp_path.parent / f"annotated_{tmp_path.name}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (vw, vh))

        frame_results: list[VideoFrameResult] = []
        frame_idx = 0

        pipeline.reset_tracking()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            fr = pipeline.process_frame(frame, frame_idx)
            annotated = pipeline.render(frame.copy(), fr)
            writer.write(annotated)

            frame_results.append(
                VideoFrameResult(
                    frame_idx=frame_idx,
                    person_count=fr.person_count,
                    compliant_count=fr.compliant_count,
                    violation_count=fr.violation_count,
                    compliance_rate=round(fr.compliance_rate, 4),
                )
            )
            frame_idx += 1

        cap.release()
        writer.release()

        n = len(frame_results) if frame_results else 1
        summary = VideoSummary(
            total_frames=frame_idx,
            avg_person_count=round(sum(r.person_count for r in frame_results) / n, 2),
            avg_compliant_count=round(sum(r.compliant_count for r in frame_results) / n, 2),
            avg_violation_count=round(sum(r.violation_count for r in frame_results) / n, 2),
            avg_compliance_rate=round(
                sum(r.compliance_rate for r in frame_results) / n, 4
            ),
        )

        video_b64 = encode_video_to_base64(out_path) if out_path.exists() else ""

        return VideoInferenceResponse(
            filename=filename,
            total_frames=frame_idx,
            fps=fps,
            video_width=vw,
            video_height=vh,
            summary=summary,
            frame_results=frame_results,
            annotated_video_base64=video_b64,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
        out_path = tmp_path.parent / f"annotated_{tmp_path.name}"
        out_path.unlink(missing_ok=True)
