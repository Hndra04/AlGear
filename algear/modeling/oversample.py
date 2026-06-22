import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def get_augmentation_pipeline():
    import albumentations as A
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),
        A.Blur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(p=0.3),
        A.CLAHE(p=0.3),
        A.ToGray(p=0.1),
        A.RandomGamma(p=0.3),
    ])


def scan_classes(labels_dir: Path):
    """Return {image_stem: {class_ids}} for every label file."""
    image_classes = {}
    for lbl_path in labels_dir.glob("*.txt"):
        with open(lbl_path) as f:
            classes = {int(line.split()[0]) for line in f}
        image_classes[lbl_path.stem] = classes
    return image_classes


def create_oversampled_dataset(
    src_dir: Path,
    dst_dir: Path,
    labels_dir: Path,
    class_multipliers: dict,
):
    """
    Create an augmented oversampled dataset.

    Args:
        src_dir: directory with 'images/' and 'labels/' subdirs (source)
        dst_dir: directory where oversampled data is written
        labels_dir: path to label files
        class_multipliers: {class_id: multiplier} —
            e.g. {1: 10, 2: 3} means images containing class 1 get 10 copies,
            images containing class 2 get 3 copies.
            An image matching multiple minority classes gets the highest multiplier.

    Returns (original_count, augmented_count, final_count).
    """
    src_img_dir = src_dir / "images"
    src_lbl_dir = labels_dir
    dst_img_dir = dst_dir / "images"
    dst_lbl_dir = dst_dir / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    aug_pipeline = get_augmentation_pipeline()
    aug_count = 0
    orig_count = 0

    ext_map = {}
    for p in src_img_dir.iterdir():
        if p.is_file():
            ext_map[p.stem] = p.suffix

    image_classes = scan_classes(labels_dir)

    for lbl_path in tqdm(
        sorted(src_lbl_dir.glob("*.txt")), desc="Oversampling", unit="files"
    ):
        stem = lbl_path.stem
        suffix = ext_map.get(stem, ".jpg")
        img_path = src_img_dir / f"{stem}{suffix}"

        if not img_path.exists():
            logger.warning(f"Image not found for {stem}, skipping")
            continue

        classes = image_classes.get(stem, set())
        multiplier = max(
            (class_multipliers.get(c, 1) for c in classes),
            default=1,
        )
        num_dups = multiplier - 1

        label_text = lbl_path.read_text()

        shutil.copy2(img_path, dst_img_dir / f"{stem}{suffix}")
        shutil.copy2(lbl_path, dst_lbl_dir / f"{stem}.txt")
        orig_count += 1

        if num_dups < 1:
            continue

        img_array = cv2.imread(str(img_path))
        if img_array is None:
            logger.warning(f"Could not read {img_path}, skipping augmentation")
            continue
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        for i in range(num_dups):
            augmented = aug_pipeline(image=img_rgb)
            aug_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)

            aug_stem = f"{stem}_os{i}"
            cv2.imwrite(str(dst_img_dir / f"{aug_stem}{suffix}"), aug_bgr)
            (dst_lbl_dir / f"{aug_stem}.txt").write_text(label_text)
            aug_count += 1

    final_count = orig_count + aug_count
    logger.info(
        f"Oversampling complete: {orig_count} originals + "
        f"{aug_count} augmented = {final_count} total"
    )
    return orig_count, aug_count, final_count
