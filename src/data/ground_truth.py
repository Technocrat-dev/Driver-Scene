"""Parse BDD100K labels into ground truth structures for evaluation."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from src.config import settings
from src.models import GroundTruth

logger = logging.getLogger(__name__)

# Canonical object categories in BDD100K
BDD100K_CATEGORIES = {
    "car", "bus", "truck", "train", "motor", "bike",
    "person", "rider", "traffic light", "traffic sign",
}

# Map BDD100K categories to natural language for description templates
CATEGORY_NAMES = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "motor": "motorcycle",
    "bike": "bicycle",
    "person": "pedestrian",
    "rider": "cyclist/rider",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}

# Map BDD100K scene attribute to road_type
SCENE_TO_ROAD_TYPE = {
    "city street": "city_street",
    "highway": "highway",
    "residential": "residential",
    "parking lot": "parking_lot",
    "gas stations": "gas_station",
    "tunnel": "tunnel",
}


def parse_single_label(label_entry: dict) -> GroundTruth:
    """
    Parse a single BDD100K label entry into a GroundTruth object.

    Args:
        label_entry: A dict from the BDD100K labels JSON (one frame).

    Returns:
        GroundTruth with object counts, scene attributes, and templated description.
    """
    image_name = label_entry.get("name", "unknown")
    attrs = label_entry.get("attributes", {})

    # Extract scene-level attributes
    weather = attrs.get("weather", "unknown")
    scene = attrs.get("scene", "unknown")
    timeofday = attrs.get("timeofday", "unknown")

    # Count objects by category
    object_counts: Counter = Counter()
    labels_list = label_entry.get("labels", [])
    for obj in labels_list:
        category = obj.get("category", "")
        if category in BDD100K_CATEGORIES:
            object_counts[category] += 1

    # Build templated description from GT
    description = _build_gt_description(
        objects=dict(object_counts),
        weather=weather,
        scene=scene,
        timeofday=timeofday,
    )

    return GroundTruth(
        image_name=image_name,
        objects=dict(object_counts),
        weather=weather,
        scene=scene,
        timeofday=timeofday,
        description=description,
    )


def _build_gt_description(
    objects: dict[str, int],
    weather: str,
    scene: str,
    timeofday: str,
) -> str:
    """
    Build a natural language description from ground truth labels.

    This serves as the reference text for BERTScore evaluation.
    """
    parts = []

    # Scene overview
    time_str = timeofday if timeofday != "undefined" else "unspecified time"
    weather_str = weather if weather != "undefined" else "unspecified weather"
    scene_str = scene if scene != "undefined" else "a road"
    parts.append(
        f"A driving scene on {scene_str} during {time_str} with {weather_str} conditions."
    )

    # Objects
    if objects:
        obj_parts = []
        for cat, count in sorted(objects.items(), key=lambda x: -x[1]):
            name = CATEGORY_NAMES.get(cat, cat)
            if count == 1:
                obj_parts.append(f"1 {name}")
            else:
                obj_parts.append(f"{count} {name}s")
        parts.append(f"The scene contains {', '.join(obj_parts)}.")
    else:
        parts.append("No annotated objects are present in the scene.")

    return " ".join(parts)


def load_ground_truths(labels_path: Path | None = None) -> dict[str, GroundTruth]:
    """
    Load all ground truths from the sampled labels file.

    Returns:
        Dict mapping image_name → GroundTruth
    """
    path = labels_path or (settings.sampled_dir / "labels.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Sampled labels not found at {path}. Run dataset preparation first."
        )

    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    gts = {}
    for entry in labels:
        gt = parse_single_label(entry)
        gts[gt.image_name] = gt

    logger.info(f"Loaded {len(gts)} ground truth entries")
    return gts
