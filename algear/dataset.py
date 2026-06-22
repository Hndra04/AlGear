from pathlib import Path

from loguru import logger
import typer

from algear.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ROBOFLOW_API_KEY,
    ROBOFLOW_DIR,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = typer.Typer()


@app.command()
def download_roboflow(
    output_dir: Path = ROBOFLOW_DIR,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    api_key: str = ROBOFLOW_API_KEY,
):
    from roboflow import Roboflow

    if not api_key:
        logger.error("ROBOFLOW_API_KEY not set. Add it to .env or export it.")
        raise typer.Exit(code=1)

    logger.info(f"Downloading {workspace}/{project} v{version} to {output_dir}")
    rf = Roboflow(api_key=api_key)
    project_handle = rf.workspace(workspace).project(project)
    dataset = project_handle.version(version).download("yolov8", location=str(output_dir))
    logger.success(f"Dataset downloaded to {output_dir}")
    return dataset


@app.command()
def prepare(
    input_path: Path = ROBOFLOW_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "construction-site-safety",
):
    logger.info("Preparing dataset for training...")
    output_path.mkdir(parents=True, exist_ok=True)

    data_yaml = input_path / "data.yaml"
    if data_yaml.exists():
        import shutil
        shutil.copy2(str(data_yaml), str(output_path / "data.yaml"))
        logger.success(f"Copied data.yaml to {output_path}")
    else:
        logger.warning(f"No data.yaml found at {data_yaml}")

    logger.success("Dataset preparation complete.")


@app.command()
def oversample(
    labels_dir: Path = ROBOFLOW_DIR / "train" / "labels",
    img_dir: Path = ROBOFLOW_DIR / "train" / "images",
    output_dir: Path = PROCESSED_DATA_DIR / "construction-site-safety-oversampled",
    multiplier_no_helmet: int = 10,
    multiplier_no_vest: int = 3,
):
    """
    Oversample minority classes by duplicating images with augmentation.

    Images containing no-helmet (class 1) are duplicated `multiplier_no_helmet`
    times, no-vest (class 2) by `multiplier_no_vest` times.
    Each duplicate gets a different random augmentation.
    """
    from algear.modeling.oversample import (
        create_oversampled_dataset,
        scan_classes,
    )

    class_multipliers = {1: multiplier_no_helmet, 2: multiplier_no_vest}

    image_classes = scan_classes(labels_dir)
    for cls_id, mult in class_multipliers.items():
        match_count = sum(1 for c in image_classes.values() if cls_id in c)
        logger.info(
            f"Class {cls_id} (no-{'helmet' if cls_id == 1 else 'vest'}): "
            f"{match_count} images, multiplier={mult}, "
            f"~{match_count * (mult - 1)} augmented copies"
        )

    logger.info(f"Creating oversampled dataset in {output_dir}")
    create_oversampled_dataset(
        src_dir=img_dir.parent,
        dst_dir=output_dir / "train",
        labels_dir=labels_dir,
        class_multipliers=class_multipliers,
    )

    import yaml

    orig_yaml = ROBOFLOW_DIR / "data.yaml"
    with open(orig_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str((output_dir / "train" / "images").resolve())
    cfg["val"] = str((ROBOFLOW_DIR / "valid" / "images").resolve())
    cfg["test"] = str((ROBOFLOW_DIR / "test" / "images").resolve())

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.success(f"Oversampled dataset ready at {output_dir}")


@app.command()
def resplit(
    src_dir: Path = ROBOFLOW_DIR,
    output_dir: Path = PROCESSED_DATA_DIR / "construction-site-safety-resplit",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Combine all splits (train/valid/test) and resplit with stratification.

    Merges every image-label pair from the original splits, deduplicates
    by image content hash, then performs a stratified split so each new
    split has a representative proportion of all classes.
    """
    import hashlib
    import random
    import shutil
    from collections import Counter, defaultdict

    import yaml

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    random.seed(seed)

    # Load class names from original data.yaml
    orig_yaml = src_dir / "data.yaml"
    with open(orig_yaml) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    num_classes = len(class_names)

    # Collect all image-label pairs, deduping by image hash
    logger.info("Collecting image-label pairs from all splits...")
    all_pairs = []
    seen_hashes = set()
    dup_count = 0

    for split_name in ["train", "valid", "test"]:
        lbl_dir = src_dir / split_name / "labels"
        img_dir = src_dir / split_name / "images"
        if not lbl_dir.exists() or not img_dir.exists():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            stem = lbl_path.stem
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                logger.warning(f"No image found for label {lbl_path.name}, skipping")
                continue
            h = hashlib.md5(img_path.read_bytes()).hexdigest()
            if h in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(h)
            all_pairs.append((img_path, lbl_path))

    logger.info(f"Total unique pairs: {len(all_pairs)}, duplicates skipped: {dup_count}")

    # Tag each pair by its class signature
    def get_label_signature(lbl_path: Path) -> frozenset:
        classes = set()
        with open(lbl_path) as f:
            for line in f:
                classes.add(int(line.strip().split()[0]))
        return frozenset(classes)

    sig_to_pairs = defaultdict(list)
    for img_path, lbl_path in all_pairs:
        sig = get_label_signature(lbl_path)
        sig_to_pairs[sig].append((img_path, lbl_path))

    logger.info(f"Unique class signatures: {len(sig_to_pairs)}")
    for sig, pairs in sorted(sig_to_pairs.items(), key=lambda x: -len(x[1])):
        names = sorted(class_names[c] for c in sig)
        logger.debug(f"  {len(pairs):>5d} images — {names}")

    # Stratified split
    train_pairs, val_pairs, test_pairs = [], [], []

    for sig, pairs in sig_to_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_pairs.extend(pairs[:n_train])
        val_pairs.extend(pairs[n_train:n_train + n_val])
        test_pairs.extend(pairs[n_train + n_val:])

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    logger.info(
        f"Split sizes: train={len(train_pairs)} ({len(train_pairs)/total*100:.1f}%), "
        f"val={len(val_pairs)} ({len(val_pairs)/total*100:.1f}%), "
        f"test={len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)"
    )

    # Write splits to disk
    def write_split(pairs, split_name):
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
        logger.info(f"  {split_name}: {len(pairs)} images written")

    logger.info(f"Writing splits to {output_dir}...")
    write_split(train_pairs, "train")
    write_split(val_pairs, "val")
    write_split(test_pairs, "test")

    # Generate data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": str((output_dir / "train" / "images").resolve()),
        "val": str((output_dir / "val" / "images").resolve()),
        "test": str((output_dir / "test" / "images").resolve()),
        "nc": num_classes,
        "names": class_names,
    }

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Log class distribution per split
    def count_instances(pairs):
        c = Counter()
        for _, lbl_path in pairs:
            with open(lbl_path) as f:
                for line in f:
                    c[int(line.strip().split()[0])] += 1
        return c

    split_counts = {
        "train": count_instances(train_pairs),
        "val": count_instances(val_pairs),
        "test": count_instances(test_pairs),
    }

    logger.info("Class distribution across splits:")
    for cls_id in range(num_classes):
        t = split_counts["train"].get(cls_id, 0)
        v = split_counts["val"].get(cls_id, 0)
        te = split_counts["test"].get(cls_id, 0)
        logger.info(f"  {class_names[cls_id]:<15s}  train={t:>5d}  val={v:>4d}  test={te:>4d}")

    logger.success(f"Resplit dataset ready at {output_dir}")
    return output_dir


@app.command()
def main(
    action: str = typer.Argument("download", help="Action: download, prepare, oversample, or resplit"),
):
    if action == "download":
        download_roboflow()
    elif action == "prepare":
        prepare()
    elif action == "oversample":
        oversample()
    elif action == "resplit":
        resplit()
    else:
        logger.error(f"Unknown action: {action}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
