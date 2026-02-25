"""Integration tests for the driving scene pipeline with mocked Gemini API.

Tests the full generate → evaluate flow without real API calls.
"""

from __future__ import annotations

import json

import pytest

from src.models import DetectedObject, SceneDescription


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_scene_description():
    """A realistic scene description that the mock VLM returns."""
    return SceneDescription(
        summary="A clear daytime city street with moderate traffic and pedestrians on the sidewalk.",
        objects=[
            DetectedObject(category="car", count=3, details="2 ahead, 1 parked"),
            DetectedObject(category="person", count=2, details="on sidewalk"),
            DetectedObject(category="traffic light", count=1, details="green"),
        ],
        weather="clear",
        lighting="daytime",
        road_type="city_street",
        hazards=["pedestrians near roadway"],
        meta_actions=["maintain_speed", "yield"],
    )


@pytest.fixture
def sample_labels():
    """Minimal BDD100K labels for 2 images."""
    return [
        {
            "name": "test_img_001.jpg",
            "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
            "labels": [
                {"category": "car"},
                {"category": "car"},
                {"category": "car"},
                {"category": "person"},
                {"category": "person"},
                {"category": "traffic light"},
            ],
        },
        {
            "name": "test_img_002.jpg",
            "attributes": {"weather": "rainy", "scene": "highway", "timeofday": "night"},
            "labels": [
                {"category": "car"},
                {"category": "truck"},
                {"category": "traffic sign"},
            ],
        },
    ]


@pytest.fixture
def prepared_dataset(tmp_path, sample_labels):
    """Create a minimal prepared dataset in a temp directory."""
    sampled_dir = tmp_path / "data" / "sampled"
    sampled_dir.mkdir(parents=True)

    # Write labels
    labels_file = sampled_dir / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(sample_labels, f)

    # Create dummy images
    images_dir = sampled_dir / "images"
    images_dir.mkdir()
    for label in sample_labels:
        (images_dir / label["name"]).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    return sampled_dir


# ── Integration Tests ─────────────────────────────────────────────────────────


class TestEvaluateAndExport:
    """Test evaluate → export flow using pre-built fixture data."""

    def test_evaluate_metrics_are_populated(self, tmp_path, mock_scene_description):
        """Test that all evaluation metrics including count accuracy are computed."""
        from src.evaluation.bertscore import compute_bertscore
        from src.evaluation.completeness import compute_completeness
        from src.evaluation.count_accuracy import compute_count_accuracy
        from src.evaluation.hallucination import compute_hallucination
        from src.models import EvaluationResult, GroundTruth

        # Build a ground truth
        gt = GroundTruth(
            image_name="test.jpg",
            weather="clear",
            timeofday="daytime",
            scene="city street",
            objects={"car": 3, "person": 2, "traffic light": 1},
            description="A driving scene on city street during daytime with clear conditions.",
        )

        desc = mock_scene_description

        # Run all metrics
        bs = compute_bertscore(desc.summary, gt.description)
        hall = compute_hallucination(desc.objects, gt.objects)
        comp = compute_completeness(desc)
        count_acc = compute_count_accuracy(desc.objects, gt.objects)

        # Verify all metrics are valid
        assert 0 <= bs["f1"] <= 1
        assert 0 <= hall.hallucination_rate <= 1
        assert 0 <= comp["total"] <= 1
        assert count_acc.mae >= 0

        # Build EvaluationResult with all fields
        eval_result = EvaluationResult(
            image_name="test.jpg",
            prompt_id="v1_baseline",
            bert_score_precision=bs["precision"],
            bert_score_recall=bs["recall"],
            bert_score_f1=bs["f1"],
            hallucination_rate=hall.hallucination_rate,
            false_positive_objects=hall.false_positives,
            false_negative_objects=hall.false_negatives,
            completeness_score=comp["total"],
            count_accuracy_mae=count_acc.mae,
            count_accuracy_ratio=count_acc.count_ratio,
            weather_match=True,
            lighting_match=True,
        )

        assert eval_result.count_accuracy_mae >= 0
        assert eval_result.count_accuracy_ratio > 0


class TestExportTraining:
    """Test the training data export functionality."""

    def test_export_creates_jsonl(self, tmp_path, mock_scene_description):
        """Test that export produces valid JSONL output."""
        from src.pipeline import _export_training_data

        # Create mock results
        results = [
            {
                "image_name": "img1.jpg",
                "prompt_id": "v1_baseline",
                "description": mock_scene_description.model_dump(),
            },
            {
                "image_name": "img2.jpg",
                "prompt_id": "v1_baseline",
                "description": None,  # Failed generation — should be skipped
            },
        ]

        results_file = tmp_path / "results_v1_baseline.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        output_file = tmp_path / "training.jsonl"
        _export_training_data(results_file, output_file, prompt_id=None)

        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 1  # Only 1 valid result
        entry = json.loads(lines[0])
        assert "messages" in entry
        assert len(entry["messages"]) == 2
        assert entry["messages"][0]["role"] == "user"
        assert entry["messages"][1]["role"] == "assistant"


class TestCountAccuracy:
    """Test the count accuracy evaluation metric."""

    def test_perfect_counts(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [
            DetectedObject(category="car", count=3),
            DetectedObject(category="person", count=2),
        ]
        gt = {"car": 3, "person": 2}

        result = compute_count_accuracy(predicted, gt)
        assert result.mae == 0.0
        assert result.count_ratio == 1.0

    def test_overcounting(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [
            DetectedObject(category="car", count=5),
            DetectedObject(category="person", count=2),
        ]
        gt = {"car": 3, "person": 2}

        result = compute_count_accuracy(predicted, gt)
        assert result.mae > 0
        assert result.count_ratio > 1.0  # Overcounted

    def test_undercounting(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [
            DetectedObject(category="car", count=1),
        ]
        gt = {"car": 3, "person": 2}

        result = compute_count_accuracy(predicted, gt)
        assert result.mae > 0
        assert result.count_ratio < 1.0  # Undercounted

    def test_empty_gt(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [DetectedObject(category="car", count=2)]
        gt = {}

        result = compute_count_accuracy(predicted, gt)
        assert result.mae == 0.0  # No GT to compare against

    def test_category_normalization(self):
        from src.evaluation.count_accuracy import compute_count_accuracy

        predicted = [
            DetectedObject(category="pedestrian", count=3),
        ]
        gt = {"person": 3}

        result = compute_count_accuracy(predicted, gt)
        assert result.mae == 0.0
