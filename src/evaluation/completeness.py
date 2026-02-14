"""Completeness scoring: evaluate whether the VLM output covers all required fields.

Checks that each field in the SceneDescription is populated with meaningful content,
weighted by importance for autonomous driving applications.
"""

from __future__ import annotations

import logging

from src.models import SceneDescription

logger = logging.getLogger(__name__)

# Field weights reflecting importance for AD applications
FIELD_WEIGHTS = {
    "summary": 0.15,
    "objects": 0.30,
    "weather": 0.10,
    "lighting": 0.10,
    "road_type": 0.10,
    "hazards": 0.10,
    "meta_actions": 0.15,
}


def compute_completeness(description: SceneDescription) -> dict[str, float]:
    """
    Evaluate the completeness of a scene description.

    Checks each field for:
    - Presence (is the field populated?)
    - Meaningfulness (is the content substantive, not just placeholders?)

    Args:
        description: The VLM-generated SceneDescription.

    Returns:
        Dict with per-field scores and weighted total.
    """
    scores: dict[str, float] = {}

    # Summary: present and at least 20 characters
    if description.summary and len(description.summary.strip()) >= 20:
        scores["summary"] = 1.0
    elif description.summary and len(description.summary.strip()) >= 5:
        scores["summary"] = 0.5
    else:
        scores["summary"] = 0.0

    # Objects: at least 1 detected, with counts
    if description.objects:
        valid_objects = [o for o in description.objects if o.count > 0 and o.category.strip()]
        if len(valid_objects) >= 3:
            scores["objects"] = 1.0
        elif len(valid_objects) >= 1:
            scores["objects"] = 0.5 + 0.5 * (len(valid_objects) / 3)
        else:
            scores["objects"] = 0.1
    else:
        scores["objects"] = 0.0

    # Weather: must be a valid value
    valid_weather = {"clear", "rainy", "foggy", "snowy", "overcast", "partly cloudy"}
    if description.weather and description.weather.lower().strip() in valid_weather:
        scores["weather"] = 1.0
    elif description.weather and description.weather.strip():
        scores["weather"] = 0.5  # Non-standard but present
    else:
        scores["weather"] = 0.0

    # Lighting: must be a valid value
    valid_lighting = {"daytime", "night", "dawn", "dusk", "dawn/dusk"}
    if description.lighting and description.lighting.lower().strip() in valid_lighting:
        scores["lighting"] = 1.0
    elif description.lighting and description.lighting.strip():
        scores["lighting"] = 0.5
    else:
        scores["lighting"] = 0.0

    # Road type: must be a valid value
    valid_road = {"highway", "city_street", "city street", "residential", "intersection",
                  "parking_lot", "parking lot", "gas_station", "tunnel"}
    if description.road_type and description.road_type.lower().strip() in valid_road:
        scores["road_type"] = 1.0
    elif description.road_type and description.road_type.strip():
        scores["road_type"] = 0.5
    else:
        scores["road_type"] = 0.0

    # Hazards: at least 1 identified
    if description.hazards and len(description.hazards) >= 2:
        scores["hazards"] = 1.0
    elif description.hazards and len(description.hazards) >= 1:
        scores["hazards"] = 0.7
    else:
        scores["hazards"] = 0.0

    # Meta-actions: at least 1 recommended
    valid_actions = {"brake", "accelerate", "slow_down", "maintain_speed",
                     "lane_change_left", "lane_change_right", "yield", "stop"}
    if description.meta_actions:
        valid = [a for a in description.meta_actions if a.lower().strip() in valid_actions]
        if len(valid) >= 1:
            scores["meta_actions"] = 1.0
        else:
            scores["meta_actions"] = 0.5  # Actions present but non-standard
    else:
        scores["meta_actions"] = 0.0

    # Weighted total
    total = sum(scores[field] * FIELD_WEIGHTS[field] for field in FIELD_WEIGHTS)
    scores["total"] = round(total, 4)

    return scores
