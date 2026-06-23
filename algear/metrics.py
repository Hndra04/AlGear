"""Metric calculations for PPE Compliance Accuracy and People Counting MAE.

These metrics complement the detection-level mAP/Precision/Recall
with pipeline-level measurements relevant to the real-world use case.
"""

from pathlib import Path

import numpy as np
from loguru import logger

from algear.modeling.pipeline import (
    HELMET,
    NO_HELMET,
    NO_VEST,
    PERSON,
    VEST,
    compute_iou,
    xywhn_to_xyxy,
)

# ── Ground-truth helpers ──────────────────────────────────────────────


def _parse_yolo_label(label_path: Path) -> np.ndarray:
    """Read a YOLO label file → (N, 5) [class, cx, cy, w, h] or (0, 5)."""
    if not label_path.exists():
        return np.empty((0, 5))
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(parts[0])] + [float(x) for x in parts[1:5]])
    return np.array(rows) if rows else np.empty((0, 5))


def _gt_compliance_from_label(
    label: np.ndarray, img_w: int, img_h: int
) -> dict:
    """Derive ground-truth compliance per person from a YOLO label file.

    Uses proximity: each person is matched to the nearest helmet/no-helmet
    and vest/no-vest by Euclidean distance between centres (in pixel space).

    Returns dict:
        'total_persons': int
        'compliant_persons': int
        'person_details': list[dict]  # per-person info
    """
    if len(label) == 0:
        return {"total_persons": 0, "compliant_persons": 0, "person_details": []}

    class_ids = label[:, 0].astype(int)
    person_mask = class_ids == PERSON
    person_indices = np.where(person_mask)[0]

    if len(person_indices) == 0:
        return {"total_persons": 0, "compliant_persons": 0, "person_details": []}

    details = []
    for p_idx in person_indices:
        p_cx = label[p_idx, 1] * img_w
        p_cy = label[p_idx, 2] * img_h

        has_helmet, has_no_helmet = False, False
        has_vest, has_no_vest = False, False

        # Head PPE — closest helmet / no-helmet
        for cls, flag_name in [(HELMET, "helmet"), (NO_HELMET, "no-helmet")]:
            cls_mask = class_ids == cls
            if not cls_mask.any():
                continue
            if flag_name == "helmet":
                has_helmet = True
            else:
                has_no_helmet = True

        # Body PPE — closest vest / no-vest
        for cls, flag_name in [(VEST, "vest"), (NO_VEST, "no-vest")]:
            cls_mask = class_ids == cls
            if not cls_mask.any():
                continue
            if flag_name == "vest":
                has_vest = True
            else:
                has_no_vest = True

        compliant = has_helmet and not has_no_helmet and has_vest and not has_no_vest
        details.append(
            {
                "person_idx": int(p_idx),
                "compliant": compliant,
                "has_helmet": has_helmet,
                "has_no_helmet": has_no_helmet,
                "has_vest": has_vest,
                "has_no_vest": has_no_vest,
            }
        )

    n_compliant = sum(d["compliant"] for d in details)
    return {
        "total_persons": len(details),
        "compliant_persons": n_compliant,
        "person_details": details,
    }


# ── People Counting MAE ──────────────────────────────────────────────


def people_counting_mae(
    gt_counts: list[int], pred_counts: list[int]
) -> dict:
    """Compute Mean Absolute Error for people counting.

    Parameters
    ----------
    gt_counts : list[int]
        Ground-truth person count per image.
    pred_counts : list[int]
        Predicted person count per image.

    Returns
    -------
    dict with 'mae', 'mean_gt', 'mean_pred', 'max_error', 'per_image'
    """
    assert len(gt_counts) == len(pred_counts), "Mismatched list lengths"
    errors = [abs(g - p) for g, p in zip(gt_counts, pred_counts)]
    mae = float(np.mean(errors))
    mean_gt = float(np.mean(gt_counts))
    mean_pred = float(np.mean(pred_counts))
    max_err = int(max(errors))

    logger.info(f"People Counting MAE: {mae:.2f} (target: ≤ 2)")
    logger.info(f"  Mean GT count: {mean_gt:.1f}, Mean pred count: {mean_pred:.1f}")
    logger.info(f"  Max single-image error: {max_err}")

    return {
        "mae": mae,
        "mean_gt": mean_gt,
        "mean_pred": mean_pred,
        "max_error": max_err,
        "per_image": [{"gt": g, "pred": p, "error": abs(g - p)} for g, p in zip(gt_counts, pred_counts)],
    }


# ── PPE Compliance Accuracy ──────────────────────────────────────────


def ppe_compliance_accuracy(
    gt_labels_dir: Path,
    pred_results: list[dict],
    img_dir: Path,
) -> dict:
    """Compute PPE Compliance Accuracy across a set of images.

    For each image:
      - Derive ground-truth compliance from YOLO labels (proximity-based)
      - Derive predicted compliance from model detections + association
      - Compare per-person compliance status

    Parameters
    ----------
    gt_labels_dir : Path
        Directory containing ground-truth YOLO label .txt files.
    pred_results : list[dict]
        Each dict has keys:
            'image_name': str
            'detections': np.ndarray (N, 5)
            'img_w': int
            'img_h': int
    img_dir : Path
        Directory containing the images (for img_w/img_h lookup if needed).

    Returns
    -------
    dict with 'accuracy', 'total_persons', 'correct', 'per_image'
    """
    total_persons = 0
    correct_classifications = 0
    per_image = []

    for result in pred_results:
        img_name = result["image_name"]
        stem = Path(img_name).stem
        label_path = gt_labels_dir / f"{stem}.txt"

        gt = _gt_compliance_from_label(
            _parse_yolo_label(label_path), result["img_w"], result["img_h"]
        )

        # Predicted compliance from association pipeline
        from algear.modeling.pipeline import associate_ppe_to_persons

        pred_workers = associate_ppe_to_persons(
            result["detections"], result["img_w"], result["img_h"]
        )

        gt_n = gt["total_persons"]
        pred_n = len(pred_workers)

        # Match predicted to GT persons by IoU for compliance comparison
        if gt_n == 0 and pred_n == 0:
            per_image.append(
                {
                    "image": img_name,
                    "gt_persons": 0,
                    "pred_persons": 0,
                    "correct": 0,
                    "total": 0,
                }
            )
            continue

        gt_details = gt["person_details"]

        # Build GT person boxes from label
        label_data = _parse_yolo_label(label_path)
        gt_person_boxes = []
        if len(label_data) > 0:
            person_mask = label_data[:, 0].astype(int) == PERSON
            for row in label_data[person_mask]:
                gt_person_boxes.append(
                    xywhn_to_xyxy(row[1:5], result["img_w"], result["img_h"])
                )

        # Greedy match pred→GT by IoU
        matched_gt = set()
        matched_pred = set()
        matches = []

        for pi, pw in enumerate(pred_workers):
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(gt_person_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pw.person_box, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou > 0.3 and best_gi >= 0:
                matches.append((pi, best_gi))
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        correct = 0
        total = max(gt_n, pred_n)

        for pi, gi in matches:
            gt_compliant = gt_details[gi]["compliant"]
            pred_compliant = pred_workers[pi].is_compliant
            if gt_compliant == pred_compliant:
                correct += 1

        total_persons += total
        correct_classifications += correct

        per_image.append(
            {
                "image": img_name,
                "gt_persons": gt_n,
                "pred_persons": pred_n,
                "correct": correct,
                "total": total,
            }
        )

    accuracy = correct_classifications / total_persons if total_persons > 0 else 0.0
    logger.info(f"PPE Compliance Accuracy: {accuracy:.1%} (target: ≥ 85%)")
    logger.info(f"  Total persons evaluated: {total_persons}")
    logger.info(f"  Correct classifications: {correct_classifications}")

    return {
        "accuracy": accuracy,
        "total_persons": total_persons,
        "correct": correct_classifications,
        "per_image": per_image,
    }
