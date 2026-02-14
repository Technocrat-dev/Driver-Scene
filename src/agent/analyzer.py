"""AI Agent for automated error pattern analysis and prompt improvement.

This module implements an AI agent that:
1. Analyzes evaluation results to identify systematic error patterns
2. Classifies errors (hallucination bias, missing objects, weather confusion)
3. Generates actionable suggestions for prompt improvement
4. Can auto-generate improved prompt variants based on error analysis

Maps to Woven JD: "Development of AI agents to support MLOps tasks"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """A systematic error pattern discovered in evaluation results."""

    pattern_type: str  # e.g. "hallucination_bias", "weather_confusion", "missing_category"
    severity: str  # "high", "medium", "low"
    description: str
    affected_images: list[str] = field(default_factory=list)
    frequency: float = 0.0  # Fraction of images affected
    suggestion: str = ""


@dataclass
class AnalysisReport:
    """Complete analysis report for a prompt variant."""

    prompt_id: str
    n_images: int
    error_patterns: list[ErrorPattern] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    overall_assessment: str = ""


class ErrorAnalyzerAgent:
    """Agent that analyzes evaluation results and suggests prompt improvements.

    This agent acts as an MLOps diagnostic tool:
    - Reads evaluation JSON results
    - Identifies systematic error patterns
    - Generates human-readable analysis reports
    - Proposes prompt modifications to address found issues
    """

    def __init__(self):
        self.patterns: list[ErrorPattern] = []

    def analyze(self, eval_file: Path) -> AnalysisReport:
        """
        Run full analysis on evaluation results.

        Args:
            eval_file: Path to evaluation JSON file from pipeline run.

        Returns:
            AnalysisReport with patterns, strengths, weaknesses, and suggestions.
        """
        with open(eval_file, "r") as f:
            evaluations = json.load(f)

        if not evaluations:
            return AnalysisReport(prompt_id="unknown", n_images=0)

        prompt_id = evaluations[0].get("prompt_id", "unknown")
        n_images = len(evaluations)

        logger.info(f"Analyzing {n_images} evaluation results for prompt '{prompt_id}'")

        # Run all analysis passes
        patterns = []
        patterns.extend(self._analyze_hallucinations(evaluations))
        patterns.extend(self._analyze_weather_lighting(evaluations))
        patterns.extend(self._analyze_completeness(evaluations))
        patterns.extend(self._analyze_bertscore_distribution(evaluations))

        # Generate strengths and weaknesses
        strengths = self._identify_strengths(evaluations)
        weaknesses = self._identify_weaknesses(evaluations, patterns)

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(patterns, evaluations)

        # Overall assessment
        assessment = self._generate_assessment(evaluations, patterns)

        return AnalysisReport(
            prompt_id=prompt_id,
            n_images=n_images,
            error_patterns=patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            overall_assessment=assessment,
        )

    def _analyze_hallucinations(self, evaluations: list[dict]) -> list[ErrorPattern]:
        """Detect systematic hallucination patterns."""
        patterns = []

        # Count hallucinated category frequency
        fp_counter: dict[str, list[str]] = {}  # category → [image_names]
        fn_counter: dict[str, list[str]] = {}

        for ev in evaluations:
            image = ev.get("image_name", "")
            for fp in ev.get("false_positive_objects", []):
                fp_counter.setdefault(fp, []).append(image)
            for fn in ev.get("false_negative_objects", []):
                fn_counter.setdefault(fn, []).append(image)

        # Systematic false positives (hallucination bias)
        for cat, images in fp_counter.items():
            freq = len(images) / len(evaluations)
            if freq >= 0.3:  # Hallucinated in ≥30% of images
                patterns.append(ErrorPattern(
                    pattern_type="hallucination_bias",
                    severity="high" if freq >= 0.5 else "medium",
                    description=(
                        f"VLM systematically hallucinates '{cat}' objects — "
                        f"reported in {len(images)}/{len(evaluations)} images ({freq:.0%}) "
                        f"where they don't exist in ground truth."
                    ),
                    affected_images=images,
                    frequency=freq,
                    suggestion=(
                        f"Add explicit instruction: 'Only report {cat} if clearly visible. "
                        f"Do not assume {cat} presence without visual evidence.'"
                    ),
                ))

        # Systematic false negatives (blindspots)
        for cat, images in fn_counter.items():
            freq = len(images) / len(evaluations)
            if freq >= 0.3:
                patterns.append(ErrorPattern(
                    pattern_type="missing_category",
                    severity="high" if freq >= 0.5 else "medium",
                    description=(
                        f"VLM consistently misses '{cat}' objects — "
                        f"not detected in {len(images)}/{len(evaluations)} images ({freq:.0%}) "
                        f"where they exist in ground truth."
                    ),
                    affected_images=images,
                    frequency=freq,
                    suggestion=(
                        f"Add explicit instruction: 'Pay special attention to {cat}. "
                        f"Scan the image carefully for any {cat} instances.'"
                    ),
                ))

        return patterns

    def _analyze_weather_lighting(self, evaluations: list[dict]) -> list[ErrorPattern]:
        """Detect weather/lighting classification errors."""
        patterns = []

        weather_wrong = [ev["image_name"] for ev in evaluations if not ev.get("weather_match", True)]
        lighting_wrong = [ev["image_name"] for ev in evaluations if not ev.get("lighting_match", True)]

        if weather_wrong:
            freq = len(weather_wrong) / len(evaluations)
            if freq >= 0.2:
                patterns.append(ErrorPattern(
                    pattern_type="weather_confusion",
                    severity="high" if freq >= 0.4 else "medium",
                    description=(
                        f"Weather misclassified in {len(weather_wrong)}/{len(evaluations)} "
                        f"images ({freq:.0%})."
                    ),
                    affected_images=weather_wrong,
                    frequency=freq,
                    suggestion=(
                        "Strengthen weather assessment instructions: 'Determine weather by "
                        "examining: sky visibility, road surface wetness, windshield droplets, "
                        "fog/haze density. Choose from: clear, rainy, foggy, snowy, overcast.'"
                    ),
                ))

        if lighting_wrong:
            freq = len(lighting_wrong) / len(evaluations)
            if freq >= 0.2:
                patterns.append(ErrorPattern(
                    pattern_type="lighting_confusion",
                    severity="medium",
                    description=(
                        f"Lighting/time-of-day misclassified in {len(lighting_wrong)}/{len(evaluations)} "
                        f"images ({freq:.0%})."
                    ),
                    affected_images=lighting_wrong,
                    frequency=freq,
                    suggestion=(
                        "Add lighting assessment guidance: 'Determine time-of-day from: sky "
                        "brightness, shadow length/direction, headlight usage, artificial "
                        "lighting. Choose from: daytime, night, dawn, dusk.'"
                    ),
                ))

        return patterns

    def _analyze_completeness(self, evaluations: list[dict]) -> list[ErrorPattern]:
        """Detect completeness issues."""
        patterns = []

        low_completeness = [
            ev["image_name"] for ev in evaluations
            if ev.get("completeness_score", 0) < 0.5
        ]

        if low_completeness:
            freq = len(low_completeness) / len(evaluations)
            if freq >= 0.2:
                patterns.append(ErrorPattern(
                    pattern_type="low_completeness",
                    severity="high" if freq >= 0.4 else "medium",
                    description=(
                        f"Low completeness scores (<0.5) in {len(low_completeness)}/{len(evaluations)} "
                        f"images ({freq:.0%}). VLM is not filling all required fields."
                    ),
                    affected_images=low_completeness,
                    frequency=freq,
                    suggestion=(
                        "Add explicit field requirements: 'You MUST fill ALL of these fields: "
                        "summary, objects, weather, lighting, road_type, hazards, meta_actions. "
                        "Empty fields are NOT acceptable.'"
                    ),
                ))

        return patterns

    def _analyze_bertscore_distribution(self, evaluations: list[dict]) -> list[ErrorPattern]:
        """Analyze BERTScore distribution for outliers."""
        patterns = []

        scores = [ev.get("bert_score_f1", 0) for ev in evaluations]
        if not scores:
            return patterns

        avg = sum(scores) / len(scores)
        low_outliers = [
            ev["image_name"] for ev in evaluations
            if ev.get("bert_score_f1", 0) < avg - 0.2
        ]

        if low_outliers and len(low_outliers) >= 2:
            patterns.append(ErrorPattern(
                pattern_type="semantic_drift",
                severity="low",
                description=(
                    f"{len(low_outliers)} images have BERTScore F1 significantly below "
                    f"average ({avg:.3f}). The VLM's descriptions diverge from expected style."
                ),
                affected_images=low_outliers,
                frequency=len(low_outliers) / len(evaluations),
                suggestion=(
                    "Consider adding an example of the expected description style to anchor "
                    "the VLM's output format (few-shot prompting)."
                ),
            ))

        return patterns

    def _identify_strengths(self, evaluations: list[dict]) -> list[str]:
        """Identify what the prompt does well."""
        strengths = []
        n = len(evaluations)
        if not n:
            return strengths

        avg_bert = sum(ev.get("bert_score_f1", 0) for ev in evaluations) / n
        avg_hall = sum(ev.get("hallucination_rate", 0) for ev in evaluations) / n
        avg_comp = sum(ev.get("completeness_score", 0) for ev in evaluations) / n
        weather_acc = sum(1 for ev in evaluations if ev.get("weather_match", False)) / n
        lighting_acc = sum(1 for ev in evaluations if ev.get("lighting_match", False)) / n

        if avg_bert >= 0.4:
            strengths.append(f"Strong semantic similarity (BERTScore F1: {avg_bert:.3f})")
        if avg_hall <= 0.2:
            strengths.append(f"Low hallucination rate ({avg_hall:.1%})")
        if avg_comp >= 0.8:
            strengths.append(f"High completeness ({avg_comp:.1%})")
        if weather_acc >= 0.8:
            strengths.append(f"Accurate weather detection ({weather_acc:.0%})")
        if lighting_acc >= 0.8:
            strengths.append(f"Accurate lighting detection ({lighting_acc:.0%})")

        return strengths

    def _identify_weaknesses(
        self, evaluations: list[dict], patterns: list[ErrorPattern]
    ) -> list[str]:
        """Identify weaknesses from metrics and patterns."""
        weaknesses = []
        n = len(evaluations)
        if not n:
            return weaknesses

        avg_bert = sum(ev.get("bert_score_f1", 0) for ev in evaluations) / n
        avg_hall = sum(ev.get("hallucination_rate", 0) for ev in evaluations) / n
        avg_comp = sum(ev.get("completeness_score", 0) for ev in evaluations) / n

        if avg_bert < 0.3:
            weaknesses.append(f"Low semantic similarity (BERTScore F1: {avg_bert:.3f})")
        if avg_hall > 0.4:
            weaknesses.append(f"High hallucination rate ({avg_hall:.1%})")
        if avg_comp < 0.6:
            weaknesses.append(f"Low completeness ({avg_comp:.1%})")

        # Add pattern-based weaknesses
        for p in patterns:
            if p.severity == "high":
                weaknesses.append(p.description)

        return weaknesses

    def _generate_suggestions(
        self, patterns: list[ErrorPattern], evaluations: list[dict]
    ) -> list[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []

        # Pattern-specific suggestions
        for p in sorted(patterns, key=lambda x: -x.frequency):
            if p.suggestion:
                suggestions.append(p.suggestion)

        # General suggestions based on overall metrics
        n = len(evaluations)
        if n:
            avg_hall = sum(ev.get("hallucination_rate", 0) for ev in evaluations) / n
            if avg_hall > 0.3:
                suggestions.append(
                    "Consider using the 'grounded' prompt strategy (v7) which explicitly "
                    "instructs the VLM to only report clearly visible objects."
                )

        return suggestions

    def _generate_assessment(
        self, evaluations: list[dict], patterns: list[ErrorPattern]
    ) -> str:
        """Generate overall assessment summary."""
        n = len(evaluations)
        if not n:
            return "No evaluation data available."

        avg_bert = sum(ev.get("bert_score_f1", 0) for ev in evaluations) / n
        avg_hall = sum(ev.get("hallucination_rate", 0) for ev in evaluations) / n
        avg_comp = sum(ev.get("completeness_score", 0) for ev in evaluations) / n

        high_severity = sum(1 for p in patterns if p.severity == "high")

        if high_severity == 0 and avg_bert >= 0.4 and avg_hall <= 0.2:
            grade = "GOOD"
        elif high_severity <= 1 and avg_bert >= 0.3:
            grade = "MODERATE"
        else:
            grade = "NEEDS IMPROVEMENT"

        return (
            f"Overall grade: {grade}. "
            f"BERTScore F1: {avg_bert:.3f}, "
            f"Hallucination rate: {avg_hall:.1%}, "
            f"Completeness: {avg_comp:.1%}. "
            f"Found {len(patterns)} error patterns ({high_severity} high severity)."
        )

    def generate_improved_prompt(
        self, original_prompt: str, report: AnalysisReport
    ) -> str:
        """
        Generate an improved prompt based on error analysis.

        Uses findings from the analysis report to add guardrails
        and instructions that address specific error patterns.
        """
        additions = []

        for pattern in report.error_patterns:
            if pattern.severity in ("high", "medium") and pattern.suggestion:
                additions.append(pattern.suggestion)

        if not additions:
            return original_prompt

        improvement_block = (
            "\n\nIMPORTANT CORRECTIONS (based on error analysis):\n"
            + "\n".join(f"- {s}" for s in additions)
        )

        return original_prompt + improvement_block

    def print_report(self, report: AnalysisReport) -> None:
        """Pretty-print an analysis report."""
        print(f"\n{'='*70}")
        print(f"  ERROR ANALYSIS REPORT -- {report.prompt_id}")
        print(f"  Images analyzed: {report.n_images}")
        print(f"{'='*70}")

        print(f"\n[ASSESSMENT] {report.overall_assessment}\n")

        if report.strengths:
            print("[+] Strengths:")
            for s in report.strengths:
                print(f"    - {s}")

        if report.weaknesses:
            print("\n[!] Weaknesses:")
            for w in report.weaknesses:
                print(f"    - {w}")

        if report.error_patterns:
            print(f"\n[?] Error Patterns ({len(report.error_patterns)} found):")
            for i, p in enumerate(report.error_patterns, 1):
                print(f"\n   {i}. [{p.severity.upper()}] {p.pattern_type}")
                print(f"      {p.description}")
                if p.suggestion:
                    print(f"      >> Suggestion: {p.suggestion}")

        if report.improvement_suggestions:
            print("\n[>>] Improvement Suggestions:")
            for i, s in enumerate(report.improvement_suggestions, 1):
                print(f"   {i}. {s}")

        print(f"\n{'='*70}\n")

    def save_report(self, report: AnalysisReport, output_path: Path) -> None:
        """Save analysis report to JSON."""
        data = {
            "prompt_id": report.prompt_id,
            "n_images": report.n_images,
            "overall_assessment": report.overall_assessment,
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "improvement_suggestions": report.improvement_suggestions,
            "error_patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "severity": p.severity,
                    "description": p.description,
                    "frequency": p.frequency,
                    "n_affected": len(p.affected_images),
                    "suggestion": p.suggestion,
                }
                for p in report.error_patterns
            ],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Analysis report saved to {output_path}")
