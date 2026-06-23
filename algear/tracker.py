"""ByteTrack multi-object tracker wrapper.

Uses Ultralytics' built-in ByteTrack integration via `model.track()`.
Assigns persistent track IDs across frames for temporal consistency.

Benefits over per-frame detection:
  - Persistent person IDs across frames
  - Smooths out detection misses (re-identification)
  - Proper zone entry/exit counting
  - Trajectory analysis possible
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

# ByteTrack tracker configuration
TRACKER_CFG = "bytetrack.yaml"


@dataclass
class TrackState:
    """State of a single tracked object across frames."""
    track_id: int
    class_id: int
    bbox_xyxy: np.ndarray      # (4,) — [x1, y1, x2, y2] pixel coords
    confidence: float
    centre: tuple[float, float] = (0.0, 0.0)
    zone: str = ""
    first_seen: int = 0        # frame index
    last_seen: int = 0         # frame index
    still_active: bool = True

    def __post_init__(self):
        self.centre = (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2,
        )


@dataclass
class TrackingResult:
    """Result of tracking on a single frame."""
    tracks: list[TrackState]
    person_tracks: list[TrackState]   # only person class
    n_active_tracks: int
    frame_idx: int


class ByteTracker:
    """Thin wrapper around Ultralytics ByteTrack integration.

    Usage:
        tracker = ByteTracker()
        for frame_idx, frame in enumerate(video):
            result = tracker.update(model, frame, frame_idx)
            # result.person_tracks has persistent track IDs
    """

    def __init__(
        self,
        tracker_cfg: str = TRACKER_CFG,
        persist: bool = True,
    ):
        self.tracker_cfg = tracker_cfg
        self.persist = persist
        self._active_tracks: dict[int, TrackState] = {}
        self._lost_tracks: dict[int, TrackState] = {}
        self._next_id = 1
        logger.info(f"ByteTracker initialised — cfg={tracker_cfg}, persist={persist}")

    def update(
        self,
        model,
        frame: np.ndarray,
        frame_idx: int,
        conf: float = 0.25,
        classes: list[int] | None = None,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> TrackingResult:
        """Run detection + ByteTrack on a single frame.

        Parameters
        ----------
        model : YOLO
            Loaded ultralytics YOLO model.
        frame : np.ndarray
            BGR image.
        frame_idx : int
            Current frame index (for tracking state).
        conf : float
            Confidence threshold.
        classes : list[int], optional
            Filter to specific classes (None = all).
        imgsz : int
            Inference image size.
        device : str
            'cpu' or 'cuda' / '0'.

        Returns
        -------
        TrackingResult with all active tracks and person-specific tracks.
        """
        # Run model.track() with ByteTrack
        preds = model.track(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            tracker=self.tracker_cfg,
            persist=self.persist,
            verbose=False,
            classes=classes,
        )

        if len(preds) == 0 or preds[0].boxes is None:
            self._mark_all_inactive()
            return TrackingResult(
                tracks=list(self._active_tracks.values()),
                person_tracks=[t for t in self._active_tracks.values() if t.class_id == 3],
                n_active_tracks=0,
                frame_idx=frame_idx,
            )

        boxes = preds[0].boxes
        frame_tracks: list[TrackState] = []

        has_ids = boxes.id is not None
        if has_ids:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes), dtype=int)

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        current_ids = set()

        for i in range(len(boxes)):
            tid = int(track_ids[i])
            current_ids.add(tid)

            ts = TrackState(
                track_id=tid,
                class_id=int(cls_ids[i]),
                bbox_xyxy=xyxy[i],
                confidence=float(confs[i]),
                first_seen=frame_idx,
                last_seen=frame_idx,
                still_active=True,
            )

            self._active_tracks[tid] = ts
            frame_tracks.append(ts)

        # Mark tracks not seen in this frame
        for tid in list(self._active_tracks.keys()):
            if tid not in current_ids:
                self._active_tracks[tid].still_active = False
                self._lost_tracks[tid] = self._active_tracks.pop(tid)

        # Re-identify lost tracks (if ByteTrack re-assigns an old ID)
        if has_ids:
            for tid in list(self._lost_tracks.keys()):
                if tid in current_ids:
                    track = self._lost_tracks.pop(tid)
                    track.still_active = True
                    track.last_seen = frame_idx
                    self._active_tracks[tid] = track

        person_tracks = [t for t in frame_tracks if t.class_id == 3]

        return TrackingResult(
            tracks=frame_tracks,
            person_tracks=person_tracks,
            n_active_tracks=len([t for t in self._active_tracks.values() if t.still_active]),
            frame_idx=frame_idx,
        )

    def _mark_all_inactive(self):
        """Mark all active tracks as inactive (no detections in this frame)."""
        for tid in self._active_tracks:
            self._active_tracks[tid].still_active = False

    def get_unique_person_ids(self) -> set[int]:
        """Return all unique track IDs that were ever classified as person."""
        return {
            tid
            for tid, t in self._active_tracks.items()
            if t.class_id == 3
        } | {
            tid
            for tid, t in self._lost_tracks.items()
            if t.class_id == 3
        }

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        """Return centre positions for a given track ID over time."""
        # This would need frame-by-frame logging — placeholder for now
        return []

    def reset(self):
        """Clear all tracking state."""
        self._active_tracks.clear()
        self._lost_tracks.clear()
        self._next_id = 1
