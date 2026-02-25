"""Tests for the driving scene description generator pipeline."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.models import (
    DetectedObject,
    GroundTruth,
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


# ── AI Agent Tests ────────────────────────────────────────────────────────────


class TestErrorAnalyzerAgent:
    """Test the AI agent error analysis module."""

    def _make_eval(
        self, image="img.jpg", prompt="v1", bert_f1=0.5,
        hall_rate=0.1, comp=0.8, fps=None, fns=None,
        weather_match=True, lighting_match=True,
    ):
        return {
            "image_name": image,
            "prompt_id": prompt,
            "bert_score_f1": bert_f1,
            "hallucination_rate": hall_rate,
            "completeness_score": comp,
            "false_positive_objects": fps or [],
            "false_negative_objects": fns or [],
            "weather_match": weather_match,
            "lighting_match": lighting_match,
        }

    def test_no_errors(self):
        from src.agent.analyzer import ErrorAnalyzerAgent
        import tempfile

        evals = [self._make_eval(image=f"img_{i}.jpg") for i in range(5)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(evals, f)
            f.flush()
            agent = ErrorAnalyzerAgent()
            report = agent.analyze(Path(f.name))

        assert report.n_images == 5
        assert len(report.error_patterns) == 0

    def test_hallucination_bias_detected(self):
        from src.agent.analyzer import ErrorAnalyzerAgent
        import tempfile

        # 4 out of 5 images hallucinate 'bus'
        evals = [
            self._make_eval(image=f"img_{i}.jpg", fps=["bus"] if i < 4 else [])
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(evals, f)
            f.flush()
            agent = ErrorAnalyzerAgent()
            report = agent.analyze(Path(f.name))

        hall_patterns = [p for p in report.error_patterns if p.pattern_type == "hallucination_bias"]
        assert len(hall_patterns) >= 1
        assert hall_patterns[0].severity == "high"

    def test_weather_confusion_detected(self):
        from src.agent.analyzer import ErrorAnalyzerAgent
        import tempfile

        # 3 out of 5 images have wrong weather
        evals = [
            self._make_eval(image=f"img_{i}.jpg", weather_match=(i >= 3))
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(evals, f)
            f.flush()
            agent = ErrorAnalyzerAgent()
            report = agent.analyze(Path(f.name))

        weather_patterns = [p for p in report.error_patterns if p.pattern_type == "weather_confusion"]
        assert len(weather_patterns) >= 1

    def test_improvement_suggestions_generated(self):
        from src.agent.analyzer import ErrorAnalyzerAgent
        import tempfile

        evals = [
            self._make_eval(image=f"img_{i}.jpg", fps=["bus"], hall_rate=0.5)
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(evals, f)
            f.flush()
            agent = ErrorAnalyzerAgent()
            report = agent.analyze(Path(f.name))

        assert len(report.improvement_suggestions) > 0

    def test_auto_improve_prompt(self):
        from src.agent.analyzer import ErrorAnalyzerAgent, AnalysisReport, ErrorPattern

        agent = ErrorAnalyzerAgent()
        report = AnalysisReport(
            prompt_id="v1_baseline",
            n_images=10,
            error_patterns=[
                ErrorPattern(
                    pattern_type="hallucination_bias",
                    severity="high",
                    description="Hallucinates buses",
                    suggestion="Do not report buses unless clearly visible.",
                )
            ],
        )
        original = "Describe this driving scene."
        improved = agent.generate_improved_prompt(original, report)
        assert "CORRECTIONS" in improved
        assert "buses" in improved


# ── Spatial Grounding Tests ──────────────────────────────────────────────────


class TestSpatialGrounding:
    """Test spatial zone evaluation."""

    def test_bbox_to_zone_center(self):
        from src.evaluation.spatial import bbox_to_zone, HorizontalZone, VerticalZone

        # Center of frame (640, 360 in 1280×720)
        box = {"x1": 600, "y1": 340, "x2": 680, "y2": 380}
        zone = bbox_to_zone(box)
        assert zone.horizontal == HorizontalZone.CENTER
        assert zone.vertical == VerticalZone.MID

    def test_bbox_to_zone_left_near(self):
        from src.evaluation.spatial import bbox_to_zone, HorizontalZone, VerticalZone

        # Bottom-left (close to camera, left side)
        box = {"x1": 0, "y1": 600, "x2": 100, "y2": 700}
        zone = bbox_to_zone(box)
        assert zone.horizontal == HorizontalZone.LEFT
        assert zone.vertical == VerticalZone.NEAR

    def test_bbox_to_zone_right_far(self):
        from src.evaluation.spatial import bbox_to_zone, HorizontalZone, VerticalZone

        # Top-right (far from camera, right side)
        box = {"x1": 1000, "y1": 50, "x2": 1100, "y2": 150}
        zone = bbox_to_zone(box)
        assert zone.horizontal == HorizontalZone.RIGHT
        assert zone.vertical == VerticalZone.FAR

    def test_text_to_zones_ahead(self):
        from src.evaluation.spatial import text_to_zones, HorizontalZone

        zones = text_to_zones("2 cars ahead in the same lane")
        assert len(zones) == 1
        assert zones[0].horizontal == HorizontalZone.CENTER

    def test_text_to_zones_left(self):
        from src.evaluation.spatial import text_to_zones, HorizontalZone

        zones = text_to_zones("1 truck in the left lane")
        assert zones[0].horizontal == HorizontalZone.LEFT

    def test_text_to_zones_parked_right(self):
        from src.evaluation.spatial import text_to_zones, HorizontalZone

        zones = text_to_zones("parked on the right side")
        assert zones[0].horizontal == HorizontalZone.RIGHT

    def test_text_to_zones_empty(self):
        from src.evaluation.spatial import text_to_zones, HorizontalZone, VerticalZone

        # Empty string defaults to center/mid
        zones = text_to_zones("")
        assert zones[0].horizontal == HorizontalZone.CENTER
        assert zones[0].vertical == VerticalZone.MID

    def test_perfect_spatial_match(self):
        from src.evaluation.spatial import compute_spatial_accuracy

        predicted = [
            DetectedObject(category="car", count=2, details="2 cars ahead"),
        ]
        gt_objects = {"car": 2}
        # Two cars in center of frame
        gt_labels = [
            {"category": "car", "box2d": {"x1": 500, "y1": 350, "x2": 600, "y2": 400}},
            {"category": "car", "box2d": {"x1": 620, "y1": 340, "x2": 720, "y2": 390}},
        ]

        result = compute_spatial_accuracy(predicted, gt_objects, gt_labels)
        assert result.zone_accuracy > 0

    def test_spatial_mismatch(self):
        from src.evaluation.spatial import compute_spatial_accuracy

        # Predict objects on the right, but GT has them on the left
        predicted = [
            DetectedObject(category="car", count=2, details="2 cars on the right side"),
        ]
        gt_objects = {"car": 2}
        gt_labels = [
            {"category": "car", "box2d": {"x1": 50, "y1": 350, "x2": 150, "y2": 400}},
            {"category": "car", "box2d": {"x1": 100, "y1": 340, "x2": 200, "y2": 390}},
        ]

        result = compute_spatial_accuracy(predicted, gt_objects, gt_labels)
        assert result.zone_accuracy < 1.0

    def test_empty_gt(self):
        from src.evaluation.spatial import compute_spatial_accuracy

        predicted = [DetectedObject(category="car", count=1, details="ahead")]
        result = compute_spatial_accuracy(predicted, {}, [])
        assert result.zone_accuracy == 0.0


# ── Temporal Tests ───────────────────────────────────────────────────────────


class TestTemporal:
    """Test temporal scene description module."""

    def test_extract_video_id(self):
        from src.data.temporal import extract_video_id

        vid, fid = extract_video_id("c0035eda-6e1b34d6.jpg")
        assert vid == "c0035eda"
        assert fid == "6e1b34d6"

    def test_find_video_sequences(self):
        from src.data.temporal import find_video_sequences

        images = [
            "aaa-001.jpg", "aaa-002.jpg", "aaa-003.jpg",  # 3-frame sequence
            "bbb-001.jpg", "bbb-002.jpg",                  # 2-frame sequence
            "ccc-001.jpg",                                   # single frame
        ]
        seqs = find_video_sequences(images, min_frames=2)
        assert len(seqs) == 2
        assert seqs[0].video_id == "aaa"
        assert seqs[0].n_frames == 3
        assert seqs[1].video_id == "bbb"
        assert seqs[1].n_frames == 2

    def test_find_no_sequences(self):
        from src.data.temporal import find_video_sequences

        images = ["aaa-001.jpg", "bbb-002.jpg", "ccc-003.jpg"]
        seqs = find_video_sequences(images, min_frames=2)
        assert len(seqs) == 0

    def test_temporal_prompt_first_frame(self):
        from src.data.temporal import build_temporal_prompt

        prompt = build_temporal_prompt(previous_description=None, frame_number=1, total_frames=3)
        assert "frame 1 of 3" in prompt
        assert "JSON" in prompt

    def test_temporal_prompt_subsequent_frame(self):
        from src.data.temporal import build_temporal_prompt

        prev = '{"summary": "A clear daytime city street."}'
        prompt = build_temporal_prompt(previous_description=prev, frame_number=2, total_frames=3)
        assert "frame 2 of 3" in prompt
        assert "PREVIOUS" in prompt
        assert "scene_changes" in prompt


# ── Weather / Lighting Match Tests ───────────────────────────────────────────


class TestWeatherLightingMatch:
    """Test the _check_weather_match and _check_lighting_match helpers in pipeline.py."""

    def test_exact_weather_match(self):
        from src.pipeline import _check_weather_match
        assert _check_weather_match("clear", "clear") is True

    def test_weather_synonym_rainy(self):
        from src.pipeline import _check_weather_match
        assert _check_weather_match("rain", "rainy") is True
        assert _check_weather_match("rainy", "rain") is True

    def test_weather_overcast_cloudy(self):
        from src.pipeline import _check_weather_match
        assert _check_weather_match("cloudy", "overcast") is True

    def test_weather_mismatch(self):
        from src.pipeline import _check_weather_match
        assert _check_weather_match("clear", "rainy") is False

    def test_weather_undefined_gt_is_always_true(self):
        from src.pipeline import _check_weather_match
        assert _check_weather_match("clear", "undefined") is True
        assert _check_weather_match("anything", "unknown") is True
        assert _check_weather_match("foggy", "") is True

    def test_lighting_exact_match(self):
        from src.pipeline import _check_lighting_match
        assert _check_lighting_match("night", "night") is True

    def test_lighting_synonyms(self):
        from src.pipeline import _check_lighting_match
        assert _check_lighting_match("nighttime", "night") is True
        assert _check_lighting_match("day", "daytime") is True

    def test_lighting_dawn_dusk(self):
        from src.pipeline import _check_lighting_match
        assert _check_lighting_match("dawn", "dawn/dusk") is True
        assert _check_lighting_match("dusk", "dawn/dusk") is True
        assert _check_lighting_match("twilight", "dawn/dusk") is True

    def test_lighting_mismatch(self):
        from src.pipeline import _check_lighting_match
        assert _check_lighting_match("daytime", "night") is False

    def test_lighting_undefined_gt_is_always_true(self):
        from src.pipeline import _check_lighting_match
        assert _check_lighting_match("daytime", "undefined") is True
        assert _check_lighting_match("night", "") is True


# ── Rate Limiter Tests ───────────────────────────────────────────────────────


class TestRateLimiter:
    """Test the RateLimiter class from vlm.client."""

    def test_requests_remaining(self):
        from src.vlm.client import RateLimiter
        limiter = RateLimiter(rpm=10, rpd=100)
        assert limiter.requests_remaining_today == 100

    def test_count_decrements_after_wait(self):
        from src.vlm.client import RateLimiter
        limiter = RateLimiter(rpm=60, rpd=100)
        limiter.wait_if_needed()
        assert limiter.requests_remaining_today == 99

    def test_daily_limit_raises(self):
        from src.vlm.client import RateLimiter
        limiter = RateLimiter(rpm=100, rpd=2)
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        with pytest.raises(RuntimeError, match="Daily rate limit"):
            limiter.wait_if_needed()


# ── Fallback Parse Tests ─────────────────────────────────────────────────────


class TestFallbackParse:
    """Test VLM client _fallback_parse methods."""

    def test_gemini_fallback_extracts_json_from_text(self):
        from src.vlm.client import GeminiClient

        # Create a client without API key for parse testing only
        client = object.__new__(GeminiClient)

        text = 'Here is the result: {"summary": "Test", "weather": "clear", "lighting": "daytime", "road_type": "highway", "objects": [], "hazards": [], "meta_actions": []} done!'
        result = client._fallback_parse(text)
        assert result is not None
        assert result.summary == "Test"
        assert result.weather == "clear"

    def test_fallback_returns_none_on_garbage(self):
        from src.vlm.client import GeminiClient

        client = object.__new__(GeminiClient)
        result = client._fallback_parse("this is not json at all")
        assert result is None


# ── Partial Completeness Tests ───────────────────────────────────────────────


class TestCompletenessPartial:
    """Test partial scoring edge cases for completeness."""

    def test_short_summary_scores_half(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="Short",
            weather="clear",
            lighting="daytime",
            road_type="highway",
        )
        scores = compute_completeness(desc)
        assert scores["summary"] == 0.5

    def test_one_object_partial_score(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="A clear daytime highway scene with one vehicle visible.",
            objects=[DetectedObject(category="car", count=1)],
            weather="clear",
            lighting="daytime",
            road_type="highway",
        )
        scores = compute_completeness(desc)
        # 1 valid object out of 3 needed for full score
        assert 0.5 < scores["objects"] < 1.0

    def test_nonstandard_weather_scores_half(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="A driving scene with unusual conditions.",
            weather="hazy",  # Non-standard but present
            lighting="daytime",
            road_type="highway",
        )
        scores = compute_completeness(desc)
        assert scores["weather"] == 0.5

    def test_single_hazard_scores_partial(self):
        from src.evaluation.completeness import compute_completeness

        desc = SceneDescription(
            summary="A driving scene with one hazard identified.",
            weather="clear",
            lighting="daytime",
            road_type="highway",
            hazards=["wet road"],
        )
        scores = compute_completeness(desc)
        assert scores["hazards"] == 0.7


# ── Agent No-Patterns Test ───────────────────────────────────────────────────


class TestAgentNoPatterns:
    """Test agent behavior when no error patterns exist."""

    def test_generate_improved_prompt_returns_original(self):
        from src.agent.analyzer import AnalysisReport, ErrorAnalyzerAgent

        agent = ErrorAnalyzerAgent()
        report = AnalysisReport(
            prompt_id="v1_baseline",
            n_images=10,
            error_patterns=[],
        )
        original = "Describe this driving scene."
        improved = agent.generate_improved_prompt(original, report)
        assert improved == original  # No changes when no patterns


# ── Count Accuracy Hallucinated Category Test ────────────────────────────────


class TestCountAccuracyHallucinatedCategory:
    """Test that hallucinated BDD100K categories are penalized in count accuracy."""

    def test_hallucinated_bdd100k_category_penalized(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [
            DetectedObject(category="car", count=3),
            DetectedObject(category="bus", count=2),  # Not in GT
        ]
        gt = {"car": 3}

        result = compute_count_accuracy(predicted, gt)
        # bus is a BDD100K category predicted but not in GT → penalized
        assert result.mae > 0
        assert "bus" in result.per_category
        assert result.per_category["bus"]["accuracy"] == 0.0
