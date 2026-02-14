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
        labels_list=labels_list,
    )

    return GroundTruth(
        image_name=image_name,
        objects=dict(object_counts),
        weather=weather,
        scene=scene,
        timeofday=timeofday,
        description=description,
        raw_labels=labels_list,
    )


def _build_gt_description(
    objects: dict[str, int],
    weather: str,
    scene: str,
    timeofday: str,
    labels_list: list[dict] | None = None,
) -> str:
    """
    Build a rich natural language description from ground truth labels.

    This serves as the reference text for BERTScore evaluation.
    Enhanced with spatial context derived from bounding box positions.
    """
    parts = []

    # Scene overview — more natural phrasing
    time_str = timeofday if timeofday != "undefined" else "unspecified time"
    weather_str = weather if weather != "undefined" else "unspecified weather"
    scene_str = scene if scene != "undefined" else "a road"

    # Vary sentence structure for richer descriptions
    if weather_str == "clear":
        parts.append(f"A {time_str} driving scene on a {scene_str} under clear skies.")
    elif weather_str in ("rainy", "rain"):
        parts.append(f"A {time_str} driving scene on a {scene_str} with rainy conditions and wet road surfaces.")
    elif weather_str in ("foggy", "fog"):
        parts.append(f"A {time_str} driving scene on a {scene_str} with foggy conditions reducing visibility.")
    elif weather_str in ("snowy", "snow"):
        parts.append(f"A {time_str} driving scene on a {scene_str} with snowy conditions.")
    else:
        parts.append(f"A {time_str} driving scene on a {scene_str} with {weather_str} conditions.")

    # Objects with spatial context from bounding boxes
    if objects:
        obj_parts = []
        for cat, count in sorted(objects.items(), key=lambda x: -x[1]):
            name = CATEGORY_NAMES.get(cat, cat)
            spatial_hint = _get_spatial_hint(cat, labels_list) if labels_list else ""
            if count == 1:
                obj_parts.append(f"1 {name}{spatial_hint}")
            else:
                obj_parts.append(f"{count} {name}s{spatial_hint}")
        parts.append(f"The scene contains {', '.join(obj_parts)}.")

        # Add driving relevance
        total_objects = sum(objects.values())
        if total_objects > 10:
            parts.append("The road is busy with significant traffic.")
        elif total_objects > 5:
            parts.append("Moderate traffic is observed.")
        else:
            parts.append("Light traffic conditions.")

        # Add safety-relevant details
        if objects.get("person", 0) > 0 or objects.get("rider", 0) > 0:
            parts.append("Vulnerable road users are present, requiring caution.")
    else:
        parts.append("No annotated objects are present in the scene.")

    return " ".join(parts)


def _get_spatial_hint(category: str, labels_list: list[dict]) -> str:
    """Extract a spatial hint from bounding boxes for a given category."""
    boxes = [
        label["box2d"]
        for label in labels_list
        if label.get("category") == category and label.get("box2d")
    ]
    if not boxes:
        return ""

    # Compute average horizontal position (0=left, 1=right) for the category
    avg_cx = sum((b["x1"] + b["x2"]) / 2 for b in boxes) / len(boxes) / 1280

    if avg_cx < 0.35:
        return " on the left side"
    elif avg_cx > 0.65:
        return " on the right side"
    else:
        return " ahead"


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
