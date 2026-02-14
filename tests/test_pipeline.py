"""Tests for the driving scene description generator pipeline."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.models import (
    DetectedObject,
    EvaluationResult,
    GroundTruth,
    PromptComparisonRow,
    SceneDescription,
)


# ── Model Validation Tests ────────────────────────────────────────────────────


class TestSceneDescription:
    """Test SceneDescription Pydantic model validation."""

    def test_valid_description(self):
        desc = SceneDescription(
            summary="A clear daytime highway scene with moderate traffic.",
            objects=[
                DetectedObject(category="car", count=3, details="3 cars ahead"),
                DetectedObject(category="truck", count=1, details="1 truck in right lane"),
            ],
            weather="clear",
            lighting="daytime",
            road_type="highway",
            hazards=["moderate traffic density"],
            meta_actions=["maintain_speed"],
        )
        assert desc.summary.startswith("A clear")
        assert len(desc.objects) == 2
        assert desc.objects[0].count == 3
        assert desc.weather == "clear"

    def test_minimal_description(self):
        desc = SceneDescription(
            summary="A scene.",
            weather="clear",
            lighting="daytime",
            road_type="highway",
        )
        assert desc.objects == []
        assert desc.hazards == []
        assert desc.meta_actions == []

    def test_json_roundtrip(self):
        desc = SceneDescription(
            summary="Test scene",
            objects=[DetectedObject(category="car", count=1)],
            weather="rainy",
            lighting="night",
            road_type="city_street",
            hazards=["wet road"],
            meta_actions=["slow_down"],
        )
        json_str = desc.model_dump_json()
        recovered = SceneDescription.model_validate_json(json_str)
        assert recovered.summary == desc.summary
        assert recovered.weather == desc.weather
        assert len(recovered.objects) == 1


class TestGroundTruth:
    """Test GroundTruth model."""

    def test_valid_ground_truth(self):
        gt = GroundTruth(
            image_name="test.jpg",
            objects={"car": 5, "person": 2},
            weather="clear",
            scene="highway",
            timeofday="daytime",
            description="A clear daytime highway scene.",
        )
        assert gt.objects["car"] == 5
        assert gt.weather == "clear"

    def test_default_values(self):
        gt = GroundTruth(image_name="test.jpg")
        assert gt.objects == {}
        assert gt.weather == "unknown"


# ── Ground Truth Parser Tests ─────────────────────────────────────────────────


class TestGroundTruthParser:
    """Test BDD100K label parsing."""

    def test_parse_single_label(self):
        from src.data.ground_truth import parse_single_label

        label = {
            "name": "test_image.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "labels": [
                {"category": "car"},
                {"category": "car"},
                {"category": "person"},
                {"category": "traffic light"},
            ],
        }

        gt = parse_single_label(label)
        assert gt.image_name == "test_image.jpg"
        assert gt.objects["car"] == 2
        assert gt.objects["person"] == 1
        assert gt.weather == "clear"
        assert "car" in gt.description

    def test_empty_labels(self):
        from src.data.ground_truth import parse_single_label

        label = {
            "name": "empty.jpg",
            "attributes": {},
            "labels": [],
        }
        gt = parse_single_label(label)
        assert gt.objects == {}
        assert "No annotated objects" in gt.description


# ── Hallucination Tests ───────────────────────────────────────────────────────


class TestHallucination:
    """Test hallucination detection."""

    def test_perfect_match(self):
        from src.evaluation.hallucination import compute_hallucination

        predicted = [
            DetectedObject(category="car", count=3),
            DetectedObject(category="person", count=2),
        ]
        gt = {"car": 3, "person": 2}

        result = compute_hallucination(predicted, gt)
        assert result.hallucination_rate == 0.0
        assert result.false_positives == []
        assert result.false_negatives == []

    def test_hallucinated_objects(self):
        from src.evaluation.hallucination import compute_hallucination

        predicted = [
            DetectedObject(category="car", count=3),
            DetectedObject(category="bus", count=1),  # Not in GT
        ]
        gt = {"car": 3}

        result = compute_hallucination(predicted, gt)
        assert result.hallucination_rate > 0
        assert "bus" in result.false_positives

    def test_missed_objects(self):
        from src.evaluation.hallucination import compute_hallucination

        predicted = [
            DetectedObject(category="car", count=3),
        ]
        gt = {"car": 3, "person": 2}  # Person missed

        result = compute_hallucination(predicted, gt)
        assert result.hallucination_rate > 0
        assert "person" in result.false_negatives

    def test_category_normalization(self):
        from src.evaluation.hallucination import normalize_category

        assert normalize_category("pedestrian") == "person"
        assert normalize_category("motorcycle") == "motor"
        assert normalize_category("bicycle") == "bike"
        assert normalize_category("traffic_light") == "traffic light"


# ── Completeness Tests ────────────────────────────────────────────────────────


class TestCompleteness:
    """Test completeness scoring."""

    def test_fully_complete(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="A clear daytime highway scene with moderate traffic and good visibility.",
            objects=[
                DetectedObject(category="car", count=3),
                DetectedObject(category="truck", count=1),
                DetectedObject(category="traffic sign", count=2),
            ],
            weather="clear",
            lighting="daytime",
            road_type="highway",
            hazards=["following too closely", "lane change ahead"],
            meta_actions=["maintain_speed"],
        )
        scores = compute_completeness(desc)
        assert scores["total"] >= 0.9

    def test_empty_description(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="",
            weather="",
            lighting="",
            road_type="",
        )
        scores = compute_completeness(desc)
        assert scores["total"] < 0.1


# ── Prompt Registry Tests ────────────────────────────────────────────────────


class TestPromptRegistry:
    """Test prompt variant registry."""

    def test_all_prompts_load(self):
        from src.prompts.registry import registry

        assert len(registry) == 8

    def test_prompt_ids(self):
        from src.prompts.registry import registry

        expected_ids = [
            "v1_baseline", "v2_structured", "v3_role", "v4_cot",
            "v5_few_shot", "v6_safety", "v7_grounded", "v8_combined",
        ]
        for pid in expected_ids:
            variant = registry.get(pid)
            assert variant.template
            assert variant.strategy

    def test_invalid_prompt(self):
        from src.prompts.registry import registry

        with pytest.raises(KeyError):
            registry.get("v99_nonexistent")

    def test_list_all(self):
        from src.prompts.templates import list_prompts

        prompts = list_prompts()
        assert len(prompts) == 8
        assert all("id" in p and "strategy" in p for p in prompts)
