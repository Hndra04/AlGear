"""Person-PPE association and compliance decision engine.

Pipeline per frame:
  1. Detect all objects (person, helmet, no-helmet, no-vest, vest)
  2. Associate PPE detections with nearest person via IoU overlap
  3. Classify each person as Safe or Warning (violation)
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

# Class IDs (must match data.yaml)
PERSON = 3
HELMET = 0
NO_HELMET = 1
NO_VEST = 2
VEST = 4

# IoU thresholds for PPE–person association
IOU_HEAD_THRESHOLD = 0.01
IOU_BODY_THRESHOLD = 0.01


@dataclass
class WorkerCompliance:
    person_idx: int
    person_box: np.ndarray
    head_ppe: str = "unknown"
    body_ppe: str = "unknown"
    has_helmet: bool = False
    has_no_helmet: bool = False
    has_vest: bool = False
    has_no_vest: bool = False
    is_compliant: bool = False


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy(
    box: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """Convert normalized YOLO (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2])


def associate_ppe_to_persons(
    detections: np.ndarray,
    img_w: int,
    img_h: int,
    confidences: np.ndarray | None = None,
) -> list[WorkerCompliance]:
    """Group PPE detections with nearest person via IoU overlap.

    Parameters
    ----------
    detections : np.ndarray
        Shape (N, 5) — [class_id, x_center, y_center, width, height] (normalized).
    img_w, img_h : int
        Image dimensions in pixels.
    confidences : np.ndarray, optional
        Detection confidences (N,). Used for tie-breaking.

    Returns
    -------
    list[WorkerCompliance]
        One entry per person detected, with associated PPE status.
    """
    if len(detections) == 0:
        return []

    class_ids = detections[:, 0].astype(int)
    person_mask = class_ids == PERSON
    if not person_mask.any():
        return []

    person_indices = np.where(person_mask)[0]
    person_boxes_xyxy = np.array(
        [xywhn_to_xyxy(detections[i, 1:5], img_w, img_h) for i in person_indices]
    )

    results: list[WorkerCompliance] = []

    for p_idx, (orig_idx, p_box) in enumerate(
        zip(person_indices, person_boxes_xyxy)
    ):
        comp = WorkerCompliance(person_idx=orig_idx, person_box=p_box)

        # --- Head PPE ---
        for cls_target, attr, ppe_label in [
            (HELMET, "has_helmet", "helmet"),
            (NO_HELMET, "has_no_helmet", "no-helmet"),
        ]:
            cls_mask = class_ids == cls_target
            if not cls_mask.any():
                continue
            cls_indices = np.where(cls_mask)[0]
            best_iou, best_j = 0.0, -1
            for j in cls_indices:
                iou = compute_iou(p_box, xywhn_to_xyxy(detections[j, 1:5], img_w, img_h))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_HEAD_THRESHOLD and best_j >= 0:
                setattr(comp, attr, True)
                comp.head_ppe = ppe_label

        # --- Body PPE ---
        for cls_target, attr, ppe_label in [
            (VEST, "has_vest", "vest"),
            (NO_VEST, "has_no_vest", "no-vest"),
        ]:
            cls_mask = class_ids == cls_target
            if not cls_mask.any():
                continue
            cls_indices = np.where(cls_mask)[0]
            best_iou, best_j = 0.0, -1
            for j in cls_indices:
                iou = compute_iou(p_box, xywhn_to_xyxy(detections[j, 1:5], img_w, img_h))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_BODY_THRESHOLD and best_j >= 0:
                setattr(comp, attr, True)
                comp.body_ppe = ppe_label

        # Compliance: helmet present AND no no-helmet AND vest present AND no no-vest
        comp.is_compliant = (
            comp.has_helmet
            and not comp.has_no_helmet
            and comp.has_vest
            and not comp.has_no_vest
        )

        results.append(comp)

    logger.debug(
        f"Associated {len(results)} persons — "
        f"{sum(c.is_compliant for c in results)} compliant, "
        f"{len(results) - sum(c.is_compliant for c in results)} violations"
    )
    return results


def classify_frame(
    detections: np.ndarray,
    img_w: int,
    img_h: int,
) -> dict:
    """Run full compliance pipeline on one frame's detections.

    Parameters
    ----------
    detections : np.ndarray
        Shape (N, 5) — [class_id, cx, cy, w, h] (normalized YOLO format).
    img_w, img_h : int
        Image dimensions.

    Returns
    -------
    dict with keys:
        'workers': list[WorkerCompliance]
        'person_count': int
        'compliant_count': int
        'violation_count': int
        'compliance_rate': float
    """
    workers = associate_ppe_to_persons(detections, img_w, img_h)
    n_workers = len(workers)
    n_compliant = sum(w.is_compliant for w in workers)
    return {
        "workers": workers,
        "person_count": n_workers,
        "compliant_count": n_compliant,
        "violation_count": n_workers - n_compliant,
        "compliance_rate": n_compliant / n_workers if n_workers > 0 else 0.0,
    }
