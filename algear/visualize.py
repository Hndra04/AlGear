"""OpenCV-based visualisation for the PPE compliance pipeline.

Draws bounding boxes, class labels, compliance status, zone overlays,
and a HUD with people count and compliance rate.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from algear.modeling.pipeline import WorkerCompliance
from algear.zones import Zone

# ── Colour palette ────────────────────────────────────────────────────

COLORS = {
    "helmet": (0, 200, 0),       # green
    "no-helmet": (0, 0, 255),    # red
    "vest": (0, 180, 255),       # orange
    "no-vest": (0, 0, 200),      # dark red
    "person": (255, 180, 0),     # light blue
    "compliant": (0, 200, 0),    # green
    "violation": (0, 0, 255),    # red
    "zone": (100, 100, 100),     # grey
    "hud_bg": (40, 40, 40),      # dark grey
    "hud_text": (255, 255, 255), # white
}


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    scores: np.ndarray | None = None,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw raw detection boxes on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in place and returned).
    detections : np.ndarray
        Shape (N, 5) — [class_id, cx, cy, w, h] normalised.
    scores : np.ndarray, optional
        Confidence per detection.
    class_names : dict, optional
        Mapping class_id → name.
    """
    if len(detections) == 0:
        return frame

    h, w = frame.shape[:2]
    if class_names is None:
        class_names = {}

    for i, det in enumerate(detections):
        cls_id = int(det[0])
        cx, cy, bw, bh = det[1], det[2], det[3], det[4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        name = class_names.get(cls_id, str(cls_id))
        color = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = name
        if scores is not None and i < len(scores):
            label = f"{name} {scores[i]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def draw_workers(
    frame: np.ndarray,
    workers: list[WorkerCompliance],
    track_ids: list[int] | None = None,
) -> np.ndarray:
    """Draw person bounding boxes coloured by compliance status.

    Green box = compliant, Red box = violation.
    Also draws PPE icons (small text labels) and track ID near the box.
    """
    h, w = frame.shape[:2]

    for idx, wc in enumerate(workers):
        x1, y1, x2, y2 = wc.person_box.astype(int)
        color = COLORS["compliant"] if wc.is_compliant else COLORS["violation"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Track ID badge
        tid = track_ids[idx] if track_ids and idx < len(track_ids) and track_ids[idx] >= 0 else None
        if tid is not None:
            badge_text = f"ID:{tid}"
            cv2.rectangle(frame, (x1, y1 - 45), (x1 + 60, y1 - 22), color, -1)
            cv2.putText(frame, badge_text, (x1 + 2, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        status = "SAFE" if wc.is_compliant else "VIOLATION"
        cv2.putText(frame, status, (x1, y1 - 25 if tid is None else y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ppe_parts = []
        if wc.has_helmet:
            ppe_parts.append("H")
        if wc.has_no_helmet:
            ppe_parts.append("!H")
        if wc.has_vest:
            ppe_parts.append("V")
        if wc.has_no_vest:
            ppe_parts.append("!V")
        ppe_text = " ".join(ppe_parts)
        cv2.putText(frame, ppe_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def draw_zones(
    frame: np.ndarray,
    zones: list[Zone],
    counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Draw zone polygons and labels on the frame."""
    for z in zones:
        pts = z.polygon.astype(np.int32)
        overlay = frame.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=COLORS["zone"], thickness=2)

        label = z.name
        if counts and z.name in counts:
            label = f"{z.name}: {counts[z.name]}"

        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(overlay, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["zone"], 2)

        # Semi-transparent fill
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    return frame


def draw_hud(
    frame: np.ndarray,
    person_count: int,
    compliant_count: int,
    violation_count: int,
    fps: float | None = None,
) -> np.ndarray:
    """Draw a heads-up display bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 50

    # Dark background bar
    cv2.rectangle(frame, (0, 0), (w, bar_h), COLORS["hud_bg"], -1)

    # People count
    text = f"People: {person_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["hud_text"], 2)

    # Compliance
    text2 = f"Compliant: {compliant_count}"
    cv2.putText(frame, text2, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["compliant"], 2)

    # Violations
    text3 = f"Violations: {violation_count}"
    cv2.putText(frame, text3, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["violation"], 2)

    # FPS
    if fps is not None:
        text4 = f"FPS: {fps:.1f}"
        cv2.putText(frame, text4, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["hud_text"], 2)

    return frame


def render_frame(
    frame: np.ndarray,
    detections: np.ndarray,
    scores: np.ndarray | None,
    workers: list[WorkerCompliance],
    zones: list[Zone],
    zone_counts: dict[str, int],
    fps: float | None = None,
    class_names: dict[int, str] | None = None,
    track_ids: list[int] | None = None,
) -> np.ndarray:
    """Full render pipeline for a single frame.

    Draws zones → raw detections → worker compliance boxes → HUD.
    """
    draw_zones(frame, zones, zone_counts)
    draw_detections(frame, detections, scores, class_names)
    draw_workers(frame, workers, track_ids)
    n_total = len(workers)
    n_ok = sum(w.is_compliant for w in workers)
    draw_hud(frame, n_total, n_ok, n_total - n_ok, fps)
    return frame
