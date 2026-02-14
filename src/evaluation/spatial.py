"""Spatial grounding evaluation: compare VLM-described object positions against BDD100K bounding boxes.

Divides the frame into a 3×3 zone grid (left/center/right × near/mid/far) and computes
how well the VLM's spatial descriptions match the ground truth object positions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from src.evaluation.hallucination import normalize_category
from src.models import DetectedObject

logger = logging.getLogger(__name__)

# BDD100K standard frame dimensions
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Horizontal zone boundaries (relative to image width)
LEFT_BOUNDARY = 0.33
RIGHT_BOUNDARY = 0.67

# Vertical zone boundaries (relative to image height)
#   Near = bottom third (close to camera)
#   Mid  = middle third
#   Far  = top third (far from camera, higher in the image)
FAR_BOUNDARY = 0.40
MID_BOUNDARY = 0.70


class HorizontalZone(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VerticalZone(str, Enum):
    FAR = "far"
    MID = "mid"
    NEAR = "near"


@dataclass
class Zone:
    """A spatial zone in the frame."""

    horizontal: HorizontalZone
    vertical: VerticalZone

    @property
    def label(self) -> str:
        return f"{self.horizontal.value}_{self.vertical.value}"


# ── Keyword maps for parsing VLM spatial descriptions ────────────────────────

HORIZONTAL_KEYWORDS: dict[HorizontalZone, list[str]] = {
    HorizontalZone.LEFT: [
        "left", "left lane", "left side", "leftward", "driver side",
        "oncoming", "opposite lane", "median",
    ],
    HorizontalZone.CENTER: [
        "ahead", "center", "front", "middle", "straight",
        "in front", "directly ahead", "ego lane", "our lane",
    ],
    HorizontalZone.RIGHT: [
        "right", "right lane", "right side", "rightward", "passenger side",
        "shoulder", "curb", "sidewalk", "parked",
    ],
}

VERTICAL_KEYWORDS: dict[VerticalZone, list[str]] = {
    VerticalZone.NEAR: [
        "near", "close", "nearby", "immediately", "adjacent",
        "next to", "beside", "following", "behind us",
    ],
    VerticalZone.MID: [
        "mid", "moderate", "several car lengths", "intersection",
        "approaching",
    ],
    VerticalZone.FAR: [
        "far", "distant", "horizon", "far ahead", "way ahead",
        "background", "down the road",
    ],
}


def bbox_to_zone(box: dict, img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT) -> Zone:
    """Convert a BDD100K bounding box to a spatial zone.

    Args:
        box: Dict with x1, y1, x2, y2 (pixel coordinates).
        img_width: Frame width in pixels.
        img_height: Frame height in pixels.

    Returns:
        Zone indicating where the object is in the frame.
    """
    cx = (box["x1"] + box["x2"]) / 2 / img_width
    cy = (box["y1"] + box["y2"]) / 2 / img_height

    # Horizontal
    if cx < LEFT_BOUNDARY:
        h = HorizontalZone.LEFT
    elif cx > RIGHT_BOUNDARY:
        h = HorizontalZone.RIGHT
    else:
        h = HorizontalZone.CENTER

    # Vertical (remember: higher y = closer to camera = NEAR)
    if cy < FAR_BOUNDARY:
        v = VerticalZone.FAR
    elif cy > MID_BOUNDARY:
        v = VerticalZone.NEAR
    else:
        v = VerticalZone.MID

    return Zone(horizontal=h, vertical=v)


def text_to_zones(details: str) -> list[Zone]:
    """Parse a VLM object details string into spatial zones using keyword matching.

    Args:
        details: Free-text description, e.g. "2 cars ahead in the left lane".

    Returns:
        List of inferred zones. Returns [CENTER_MID] if no keywords match.
    """
    if not details:
        return [Zone(HorizontalZone.CENTER, VerticalZone.MID)]

    text = details.lower()

    # Find horizontal zone
    h_zone = None
    best_h_pos = len(text)
    for zone, keywords in HORIZONTAL_KEYWORDS.items():
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1 and pos < best_h_pos:
                best_h_pos = pos
                h_zone = zone

    # Find vertical zone
    v_zone = None
    best_v_pos = len(text)
    for zone, keywords in VERTICAL_KEYWORDS.items():
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1 and pos < best_v_pos:
                best_v_pos = pos
                v_zone = zone

    # Default to center/mid if no keywords found
    h_zone = h_zone or HorizontalZone.CENTER
    v_zone = v_zone or VerticalZone.MID

    return [Zone(h_zone, v_zone)]


def _build_gt_zone_distribution(
    gt_labels: list[dict],
    gt_objects: dict[str, int],
    img_width: int = IMG_WIDTH,
    img_height: int = IMG_HEIGHT,
) -> dict[str, dict[str, int]]:
    """Build a per-category zone distribution from BDD100K labels.

    Returns:
        Dict mapping normalized category → {zone_label: count}.
    """
    distribution: dict[str, dict[str, int]] = {}

    for label in gt_labels:
        cat = normalize_category(label.get("category", ""))
        if cat not in gt_objects:
            continue

        box = label.get("box2d")
        if not box:
            continue

        zone = bbox_to_zone(box, img_width, img_height)

        if cat not in distribution:
            distribution[cat] = {}
        distribution[cat][zone.label] = distribution[cat].get(zone.label, 0) + 1

    return distribution


def _build_pred_zone_distribution(
    predicted_objects: list[DetectedObject],
    gt_objects: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Build a per-category zone distribution from VLM predictions.

    Returns:
        Dict mapping normalized category → {zone_label: count}.
    """
    distribution: dict[str, dict[str, int]] = {}

    for obj in predicted_objects:
        cat = normalize_category(obj.category)
        if cat not in gt_objects:
            continue

        zones = text_to_zones(obj.details or "")
        if cat not in distribution:
            distribution[cat] = {}

        for zone in zones:
            count = obj.count if len(zones) == 1 else max(1, obj.count // len(zones))
            distribution[cat][zone.label] = distribution[cat].get(zone.label, 0) + count

    return distribution


@dataclass
class SpatialResult:
    """Results of spatial grounding evaluation."""

    zone_accuracy: float = 0.0
    per_category_accuracy: dict[str, float] = field(default_factory=dict)
    gt_zone_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    pred_zone_distribution: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_spatial_accuracy(
    predicted_objects: list[DetectedObject],
    gt_objects: dict[str, int],
    gt_labels: list[dict],
    img_width: int = IMG_WIDTH,
    img_height: int = IMG_HEIGHT,
) -> SpatialResult:
    """Compare VLM-predicted spatial zones against BDD100K ground truth zones.

    Uses zone-level IoU (Intersection over Union) to measure spatial accuracy.

    Args:
        predicted_objects: VLM-detected objects with details text.
        gt_objects: Ground truth object counts by category.
        gt_labels: Raw BDD100K label entries with box2d coordinates.
        img_width: Frame width in pixels.
        img_height: Frame height in pixels.

    Returns:
        SpatialResult with overall and per-category zone accuracy.
    """
    if not gt_objects or not gt_labels:
        return SpatialResult()

    gt_dist = _build_gt_zone_distribution(gt_labels, gt_objects, img_width, img_height)
    pred_dist = _build_pred_zone_distribution(predicted_objects, gt_objects)

    if not gt_dist:
        return SpatialResult(
            gt_zone_distribution=gt_dist,
            pred_zone_distribution=pred_dist,
        )

    per_cat_accuracy: dict[str, float] = {}
    total_iou = 0.0
    n_categories = 0

    # For each category present in GT, compute zone overlap
    for cat, gt_zones in gt_dist.items():
        pred_zones = pred_dist.get(cat, {})
        if not gt_zones:
            continue

        # Compute zone-level IoU
        all_zones = set(gt_zones.keys()) | set(pred_zones.keys())
        intersection = 0
        union = 0

        for zone in all_zones:
            gt_count = gt_zones.get(zone, 0)
            pred_count = pred_zones.get(zone, 0)
            intersection += min(gt_count, pred_count)
            union += max(gt_count, pred_count)

        iou = intersection / union if union > 0 else 0.0
        per_cat_accuracy[cat] = round(iou, 4)
        total_iou += iou
        n_categories += 1

    overall = round(total_iou / n_categories, 4) if n_categories > 0 else 0.0

    return SpatialResult(
        zone_accuracy=overall,
        per_category_accuracy=per_cat_accuracy,
        gt_zone_distribution=gt_dist,
        pred_zone_distribution=pred_dist,
    )
