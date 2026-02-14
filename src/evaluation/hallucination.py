"""Hallucination detection: compare VLM-detected objects against BDD100K ground truth.

Measures:
- False positives (hallucinated objects not in GT)
- False negatives (GT objects missed by VLM)
- Hallucination rate as a composite metric
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models import DetectedObject

logger = logging.getLogger(__name__)

# Normalize VLM output categories to match BDD100K vocabulary
CATEGORY_ALIASES = {
    # VLM might say these → map to BDD100K category
    "pedestrian": "person",
    "motorcycle": "motor",
    "motorbike": "motor",
    "bicycle": "bike",
    "cyclist": "rider",
    "biker": "rider",
    "vehicle": "car",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "pickup": "truck",
    "semi": "truck",
    "lorry": "truck",
    "traffic_light": "traffic light",
    "trafficlight": "traffic light",
    "traffic_sign": "traffic sign",
    "trafficsign": "traffic sign",
    "sign": "traffic sign",
    "streetlight": "traffic light",
    "stop_sign": "traffic sign",
    "stopsign": "traffic sign",
}

BDD100K_CATEGORIES = {
    "car", "bus", "truck", "train", "motor", "bike",
    "person", "rider", "traffic light", "traffic sign",
}


@dataclass
class HallucinationResult:
    """Result of hallucination analysis for a single image."""

    gt_objects: dict[str, int] = field(default_factory=dict)
    pred_objects: dict[str, int] = field(default_factory=dict)
    false_positives: list[str] = field(default_factory=list)  # Categories hallucinated
    false_negatives: list[str] = field(default_factory=list)  # Categories missed
    hallucination_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


def normalize_category(category: str) -> str:
    """Normalize a VLM-generated category to BDD100K vocabulary."""
    cat_lower = category.lower().strip().replace(" ", "_")

    # Direct match
    cat_spaces = cat_lower.replace("_", " ")
    if cat_spaces in BDD100K_CATEGORIES:
        return cat_spaces

    # Alias lookup
    if cat_lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat_lower]
    if cat_spaces in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat_spaces]

    # Return as-is for non-BDD100K categories (will be tracked as noise)
    return cat_spaces


def compute_hallucination(
    predicted_objects: list[DetectedObject],
    gt_objects: dict[str, int],
) -> HallucinationResult:
    """
    Compare VLM predictions against ground truth to detect hallucinations.

    Args:
        predicted_objects: List of DetectedObject from VLM output.
        gt_objects: Dict of category → count from BDD100K ground truth.

    Returns:
        HallucinationResult with false positives, negatives, and rates.
    """
    # Normalize predicted objects
    pred_counts: dict[str, int] = {}
    for obj in predicted_objects:
        norm_cat = normalize_category(obj.category)
        pred_counts[norm_cat] = pred_counts.get(norm_cat, 0) + obj.count

    # Normalize GT objects (should already be normalized, but just in case)
    gt_counts: dict[str, int] = {}
    for cat, count in gt_objects.items():
        norm_cat = normalize_category(cat)
        gt_counts[norm_cat] = gt_counts.get(norm_cat, 0) + count

    # All categories present in either set
    all_categories = set(pred_counts.keys()) | set(gt_counts.keys())

    # False positives: predicted but not in GT
    false_positives = []
    for cat in pred_counts:
        if cat not in gt_counts:
            false_positives.append(cat)

    # False negatives: in GT but not predicted
    false_negatives = []
    for cat in gt_counts:
        if cat not in pred_counts:
            false_negatives.append(cat)

    # Compute rates
    n_predicted = len([c for c in pred_counts if c in BDD100K_CATEGORIES or c in gt_counts])
    n_gt = len(gt_counts)
    n_union = len(all_categories)

    if n_union == 0:
        hallucination_rate = 0.0
        precision = 1.0
        recall = 1.0
    else:
        n_errors = len(false_positives) + len(false_negatives)
        hallucination_rate = n_errors / n_union

        # Category-level precision/recall
        true_positives = set(pred_counts.keys()) & set(gt_counts.keys())
        precision = len(true_positives) / max(len(pred_counts), 1)
        recall = len(true_positives) / max(len(gt_counts), 1)

    return HallucinationResult(
        gt_objects=gt_counts,
        pred_objects=pred_counts,
        false_positives=sorted(false_positives),
        false_negatives=sorted(false_negatives),
        hallucination_rate=round(hallucination_rate, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
    )
