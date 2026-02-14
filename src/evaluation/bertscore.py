"""BERTScore-based semantic similarity evaluation.

Compares VLM-generated scene descriptions against ground truth
descriptions built from BDD100K annotations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid slow import at startup
_scorer = None


def _get_scorer():
    """Lazy-load BERTScorer to avoid slow startup."""
    global _scorer
    if _scorer is None:
        logger.info("Loading BERTScore model (first call only)...")
        from bert_score import BERTScorer
        _scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
        )
        logger.info("BERTScore model loaded")
    return _scorer


def compute_bertscore(
    prediction: str,
    reference: str,
) -> dict[str, float]:
    """
    Compute BERTScore between a predicted description and a reference.

    Args:
        prediction: VLM-generated scene description summary.
        reference: Ground truth description from BDD100K labels.

    Returns:
        Dict with precision, recall, f1 scores.
    """
    if not prediction or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    scorer = _get_scorer()
    P, R, F1 = scorer.score([prediction], [reference])

    return {
        "precision": round(P.item(), 4),
        "recall": round(R.item(), 4),
        "f1": round(F1.item(), 4),
    }


def compute_bertscore_batch(
    predictions: list[str],
    references: list[str],
) -> list[dict[str, float]]:
    """
    Batch-compute BERTScore for efficiency.

    Args:
        predictions: List of VLM-generated descriptions.
        references: List of ground truth descriptions.

    Returns:
        List of dicts with precision, recall, f1 for each pair.
    """
    if not predictions or not references:
        return []

    assert len(predictions) == len(references), "Predictions and references must have same length"

    scorer = _get_scorer()
    P, R, F1 = scorer.score(predictions, references)

    results = []
    for i in range(len(predictions)):
        results.append({
            "precision": round(P[i].item(), 4),
            "recall": round(R[i].item(), 4),
            "f1": round(F1[i].item(), 4),
        })

    return results
