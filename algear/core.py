"""Core PPE compliance pipeline.

Usage:
    pipeline = CompliancePipeline(model_path="models/.../best.pt")
    result = pipeline.process_frame(frame)
    annotated = pipeline.render(frame, result)

Or process a video with ByteTrack:
    pipeline.process_video("site.mp4", output_path="output.mp4", use_tracker=True)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from algear.modeling.pipeline import (
    HELMET,
    NO_HELMET,
    NO_VEST,
    PERSON,
    VEST,
    WorkerCompliance,
    classify_frame,
)
from algear.tracker import ByteTracker, TrackingResult, TrackState
from algear.zones import Zone, count_per_zone, full_frame_zone, load_zones_from_config

# ── Class mapping ─────────────────────────────────────────────────────

CLASS_NAMES = {
    HELMET: "helmet",
    NO_HELMET: "no-helmet",
    NO_VEST: "no-vest",
    PERSON: "person",
    VEST: "vest",
}


@dataclass
class FrameResult:
    """Structured output of one pipeline pass."""
    detections: np.ndarray       # (N, 5) — raw YOLO output
    scores: np.ndarray           # (N,) — confidence per detection
    workers: list[WorkerCompliance]
    person_count: int
    compliant_count: int
    violation_count: int
    compliance_rate: float
    zone_counts: dict[str, int]
    img_w: int
    img_h: int
    inference_ms: float = 0.0
    # Tracking fields
    tracking_result: TrackingResult | None = None
    track_ids: list[int] = field(default_factory=list)


@dataclass
class PipelineConfig:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    zone_configs: list[dict] | None = None
    # Tracking
    use_tracker: bool = False
    tracker_cfg: str = "bytetrack.yaml"
    tracker_persist: bool = True


class CompliancePipeline:
    """End-to-end PPE compliance detection pipeline.

    Stages:
        1. YOLOv8 object detection (person, helmet, no-helmet, no-vest, vest)
        2. IoU-based PPE → person association
        3. Compliance classification (safe / violation) per worker
        4. Zone-based people counting

    With tracking enabled:
        - ByteTrack assigns persistent IDs across frames
        - Re-identifies occluded/missed detections
        - Enables zone entry/exit counting
    """

    def __init__(
        self,
        model_path: str | Path,
        config: PipelineConfig | None = None,
    ):
        from ultralytics import YOLO

        self.config = config or PipelineConfig()
        self.model = YOLO(str(model_path))
        self._zones: list[Zone] = []
        self._tracker: ByteTracker | None = None

        if self.config.use_tracker:
            self._tracker = ByteTracker(
                tracker_cfg=self.config.tracker_cfg,
                persist=self.config.tracker_persist,
            )

        logger.info(
            f"Pipeline initialised — model={model_path}, conf={self.config.conf}, "
            f"tracking={'on' if self.config.use_tracker else 'off'}"
        )

    def _ensure_zones(self, img_w: int, img_h: int) -> list[Zone]:
        """Lazy-init zones on first frame (need image dimensions)."""
        if not self._zones:
            self._zones = load_zones_from_config(
                self.config.zone_configs, img_w, img_h
            )
        return self._zones

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Run YOLOv8 detection on a frame.

        Returns (detections, scores, inference_ms).
        detections: (N, 5) — [class_id, cx, cy, w, h] normalised.
        scores: (N,) — confidence.
        """
        t0 = time.perf_counter()
        preds = self.model.predict(
            source=frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if len(preds) == 0 or preds[0].boxes is None or len(preds[0].boxes) == 0:
            return np.empty((0, 5)), np.empty((0,)), elapsed_ms

        boxes = preds[0].boxes
        xywhn = boxes.xywhn.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        detections = np.column_stack([cls_ids, xywhn])
        return detections, scores, elapsed_ms

    def track(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, float, TrackingResult]:
        """Run detection + ByteTrack on a frame.

        Returns (detections, scores, inference_ms, tracking_result).
        """
        if self._tracker is None:
            self._tracker = ByteTracker(
                tracker_cfg=self.config.tracker_cfg,
                persist=self.config.tracker_persist,
            )

        t0 = time.perf_counter()
        tracking_result = self._tracker.update(
            model=self.model,
            frame=frame,
            frame_idx=frame_idx,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            device=self.config.device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Convert tracked boxes back to normalised YOLO format for association
        h, w = frame.shape[:2]
        detections = np.empty((0, 5))
        scores_arr = np.empty((0,))

        if tracking_result.tracks:
            cls_ids = []
            xywhn_list = []
            confs = []
            for t in tracking_result.tracks:
                x1, y1, x2, y2 = t.bbox_xyxy
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cls_ids.append(t.class_id)
                xywhn_list.append([cx, cy, bw, bh])
                confs.append(t.confidence)

            detections = np.column_stack([
                np.array(cls_ids),
                np.array(xywhn_list),
            ])
            scores_arr = np.array(confs)

        return detections, scores_arr, elapsed_ms, tracking_result

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> FrameResult:
        """Full pipeline pass on one frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        frame_idx : int
            Frame index (used for tracking state).

        Returns
        -------
        FrameResult with all detections, compliance, zone counts, timing.
        """
        h, w = frame.shape[:2]
        zones = self._ensure_zones(w, h)

        if self.config.use_tracker:
            detections, scores, inf_ms, tracking_result = self.track(frame, frame_idx)
        else:
            detections, scores, inf_ms = self.detect(frame)
            tracking_result = None

        frame_result = classify_frame(detections, w, h)

        # Build worker centres and track IDs
        person_centres = []
        track_ids = []
        for wc in frame_result["workers"]:
            cx = (wc.person_box[0] + wc.person_box[2]) / 2
            cy = (wc.person_box[1] + wc.person_box[3]) / 2
            person_centres.append((cx, cy))

            # Match worker to track by IoU
            if tracking_result and tracking_result.person_tracks:
                best_tid = -1
                best_iou = 0.0
                for pt in tracking_result.person_tracks:
                    tx1, ty1, tx2, ty2 = pt.bbox_xyxy
                    # IoU between worker box and track box
                    wx1, wy1, wx2, wy2 = wc.person_box
                    ix1 = max(wx1, tx1)
                    iy1 = max(wy1, ty1)
                    ix2 = min(wx2, tx2)
                    iy2 = min(wy2, ty2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_w = (wx2 - wx1) * (wy2 - wy1)
                    area_t = (tx2 - tx1) * (ty2 - ty1)
                    union = area_w + area_t - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = pt.track_id
                track_ids.append(best_tid if best_iou > 0.3 else -1)
            else:
                track_ids.append(-1)

        zc = count_per_zone(zones, person_centres)

        return FrameResult(
            detections=detections,
            scores=scores,
            workers=frame_result["workers"],
            person_count=frame_result["person_count"],
            compliant_count=frame_result["compliant_count"],
            violation_count=frame_result["violation_count"],
            compliance_rate=frame_result["compliance_rate"],
            zone_counts=zc,
            img_w=w,
            img_h=h,
            inference_ms=inf_ms,
            tracking_result=tracking_result,
            track_ids=track_ids,
        )

    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Annotate a frame with pipeline results."""
        from algear.visualize import render_frame as _render

        zones = self._ensure_zones(result.img_w, result.img_h)
        return _render(
            frame=frame,
            detections=result.detections,
            scores=result.scores,
            workers=result.workers,
            zones=zones,
            zone_counts=result.zone_counts,
            fps=1000 / result.inference_ms if result.inference_ms > 0 else None,
            class_names=CLASS_NAMES,
            track_ids=result.track_ids,
        )

    def process_video(
        self,
        source: str | Path,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
        show: bool = False,
        use_tracker: bool | None = None,
    ) -> list[FrameResult]:
        """Process a video file or stream.

        Parameters
        ----------
        source : str or Path
            Video path or RTSP/HTTP stream URL.
        output_path : str or Path, optional
            If given, write annotated video to this path.
        max_frames : int, optional
            Stop after this many frames (None = process all).
        show : bool
            If True, display annotated frames in a cv2 window.
        use_tracker : bool, optional
            Override pipeline config tracking for this video.

        Returns
        -------
        list[FrameResult] — one per processed frame.
        """
        if use_tracker is not None:
            self.config.use_tracker = use_tracker
            if use_tracker and self._tracker is None:
                self._tracker = ByteTracker(
                    tracker_cfg=self.config.tracker_cfg,
                    persist=self.config.tracker_persist,
                )
            elif not use_tracker:
                self._tracker = None

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            raise RuntimeError(f"Cannot open video: {source}")

        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps_cap, (frame_w, frame_h))
            logger.info(f"Writing output to {output_path}")

        results: list[FrameResult] = []
        frame_idx = 0

        logger.info(
            f"Processing video: {source} ({total_frames} frames, {fps_cap:.1f} FPS, "
            f"tracking={'on' if self.config.use_tracker else 'off'})"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            fr = self.process_frame(frame, frame_idx)
            results.append(fr)

            annotated = self.render(frame.copy(), fr)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("PPE Compliance", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 100 == 0:
                logger.info(
                    f"Frame {frame_idx}/{total_frames} — "
                    f"persons={fr.person_count}, violations={fr.violation_count}"
                )
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        # Log tracking summary
        if self.config.use_tracker and self._tracker:
            unique_persons = self._tracker.get_unique_person_ids()
            logger.success(
                f"Video processing complete: {frame_idx} frames processed, "
                f"{len(unique_persons)} unique persons tracked"
            )
        else:
            logger.success(f"Video processing complete: {frame_idx} frames processed")

        return results

    def reset_tracking(self):
        """Clear all tracking state (call when starting a new video)."""
        if self._tracker:
            self._tracker.reset()
