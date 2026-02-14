"""BDD100K dataset downloader, sampler, and file utilities."""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# BDD100K scene-level attribute values for stratification
WEATHER_VALUES = {"clear", "rainy", "snowy", "foggy", "overcast", "partly cloudy", "undefined"}
SCENE_VALUES = {"city street", "highway", "residential", "parking lot", "gas stations", "tunnel"}
TIMEOFDAY_VALUES = {"daytime", "night", "dawn/dusk", "undefined"}


def load_labels(labels_path: Path | None = None) -> list[dict]:
    """Load BDD100K label JSON file and return list of frame dicts."""
    path = labels_path or settings.labels_file
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {path}.\n"
            "Download BDD100K labels from https://bdd-data.berkeley.edu\n"
            "Place the val labels JSON at: data/raw/labels/bdd100k_labels_images_val.json"
        )
    logger.info(f"Loading labels from {path}")
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} label entries")
    return labels


def get_label_attributes(label: dict) -> dict[str, str]:
    """Extract scene-level attributes (weather, scene, timeofday) from a label entry."""
    attrs = label.get("attributes", {})
    return {
        "weather": attrs.get("weather", "undefined"),
        "scene": attrs.get("scene", "undefined"),
        "timeofday": attrs.get("timeofday", "undefined"),
    }


def stratified_sample(
    labels: list[dict],
    n: int,
    seed: int = 42,
    images_dir: Path | None = None,
) -> list[dict]:
    """
    Sample N labels with stratification across weather, scene, and timeofday.

    Only includes entries whose corresponding image file exists. This ensures
    diverse coverage of driving conditions.
    """
    img_dir = images_dir or settings.images_dir
    rng = random.Random(seed)

    # Filter to entries with existing images
    available = []
    for label in labels:
        img_path = img_dir / label["name"]
        if img_path.exists():
            available.append(label)

    if not available:
        raise FileNotFoundError(
            f"No images found in {img_dir}.\n"
            "Download BDD100K val images from https://bdd-data.berkeley.edu\n"
            "Place images at: data/raw/images/100k/val/"
        )

    logger.info(f"Found {len(available)} images with matching labels")

    if len(available) <= n:
        logger.warning(f"Requested {n} samples but only {len(available)} available, using all")
        return available

    # Group by (weather, timeofday) for stratification
    buckets: dict[tuple[str, str], list[dict]] = {}
    for label in available:
        attrs = get_label_attributes(label)
        key = (attrs["weather"], attrs["timeofday"])
        buckets.setdefault(key, []).append(label)

    # Sample proportionally from each bucket
    sampled = []
    bucket_keys = sorted(buckets.keys())

    # First pass: allocate proportionally
    for key in bucket_keys:
        bucket = buckets[key]
        proportion = len(bucket) / len(available)
        bucket_n = max(1, round(proportion * n))
        rng.shuffle(bucket)
        sampled.extend(bucket[:bucket_n])

    # Trim or pad to exact n
    rng.shuffle(sampled)
    if len(sampled) > n:
        sampled = sampled[:n]
    elif len(sampled) < n:
        # Add more from remaining
        used_names = {s["name"] for s in sampled}
        remaining = [label for label in available if label["name"] not in used_names]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    # Log distribution
    dist = Counter()
    for label in sampled:
        attrs = get_label_attributes(label)
        dist[(attrs["weather"], attrs["timeofday"])] += 1
    logger.info(f"Sampled {len(sampled)} images. Distribution: {dict(dist)}")

    return sampled


def prepare_dataset(
    n: int | None = None,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    """
    Prepare the sampled dataset: load labels, stratify-sample, copy images.

    Returns the path to the sampled directory containing:
      - images/  (copied image files)
      - labels.json  (sampled label entries)
      - metadata.json  (sampling metadata)
    """
    n = n or settings.sample_size
    seed = seed or settings.random_seed
    sampled_dir = settings.sampled_dir

    metadata_path = sampled_dir / "metadata.json"
    if metadata_path.exists() and not force:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        logger.info(
            f"Dataset already prepared: {meta['n_images']} images (seed={meta['seed']}). "
            "Use force=True to re-sample."
        )
        return sampled_dir

    # Load and sample
    labels = load_labels()
    sampled_labels = stratified_sample(labels, n, seed)

    # Create output dirs
    img_out = sampled_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Copy images
    images_dir = settings.images_dir
    copied = 0
    for label in sampled_labels:
        src = images_dir / label["name"]
        dst = img_out / label["name"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            logger.warning(f"Image not found: {src}")

    # Save sampled labels
    labels_out = sampled_dir / "labels.json"
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(sampled_labels, f, indent=2)

    # Save metadata
    metadata = {
        "n_images": copied,
        "seed": seed,
        "source": str(settings.images_dir),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset prepared: {copied} images copied to {sampled_dir}")
    return sampled_dir
