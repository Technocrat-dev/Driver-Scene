"""Object count accuracy: compare VLM-predicted counts against BDD100K ground truth.

Measures per-category count accuracy using:
- Mean Absolute Error (MAE) across matched categories
- Root Mean Squared Error (RMSE) for sensitivity to large count deviations
- Total count accuracy ratio

This complements category-level hallucination detection by evaluating
whether the VLM correctly *counts* detected objects, not just identifies them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from src.models import DetectedObject

logger = logging.getLogger(__name__)


@dataclass
class CountAccuracyResult:
    """Result of object count accuracy analysis for a single image."""

    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    mae: float = 0.0
    rmse: float = 0.0
    total_gt_count: int = 0
    total_pred_count: int = 0
    count_ratio: float = 0.0  # pred_total / gt_total, 1.0 = perfect


def compute_count_accuracy(
    predicted_objects: list[DetectedObject],
    gt_objects: dict[str, int],
) -> CountAccuracyResult:
    """
    Compare predicted object counts against ground truth counts.

    Only evaluates categories present in ground truth (to avoid penalizing
    valid VLM detections of non-BDD100K categories like buildings/trees).

    Args:
        predicted_objects: List of DetectedObject from VLM output.
        gt_objects: Dict of category → count from BDD100K ground truth.

    Returns:
        CountAccuracyResult with MAE, RMSE, and per-category breakdowns.
    """
    from src.evaluation.hallucination import normalize_category

    # Aggregate predicted counts by normalized category
    pred_counts: dict[str, int] = {}
    for obj in predicted_objects:
        norm_cat = normalize_category(obj.category)
        pred_counts[norm_cat] = pred_counts.get(norm_cat, 0) + obj.count

    # Normalize GT categories
    gt_counts: dict[str, int] = {}
    for cat, count in gt_objects.items():
        norm_cat = normalize_category(cat)
        gt_counts[norm_cat] = gt_counts.get(norm_cat, 0) + count

    if not gt_counts:
        return CountAccuracyResult()

    # Per-category comparison (only for GT categories)
    per_category: dict[str, dict[str, float]] = {}
    errors = []

    for cat, gt_count in gt_counts.items():
        pred_count = pred_counts.get(cat, 0)
        error = abs(pred_count - gt_count)
        errors.append(error)

        per_category[cat] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "abs_error": error,
            "accuracy": max(0, 1 - error / max(gt_count, 1)),
        }

    # Also account for hallucinated categories (predicted but not in GT)
    from src.evaluation.hallucination import BDD100K_CATEGORIES
    for cat, pred_count in pred_counts.items():
        if cat not in gt_counts and cat in BDD100K_CATEGORIES:
            errors.append(pred_count)
            per_category[cat] = {
                "gt_count": 0,
                "pred_count": pred_count,
                "abs_error": pred_count,
                "accuracy": 0.0,
            }

    mae = sum(errors) / len(errors) if errors else 0.0
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0

    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.get(c, 0) for c in gt_counts)
    count_ratio = total_pred / max(total_gt, 1)

    return CountAccuracyResult(
        per_category=per_category,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        total_gt_count=total_gt,
        total_pred_count=total_pred,
        count_ratio=round(count_ratio, 4),
    )
