"""LLM-as-judge evaluation: use a VLM to rate description quality.

Supports both Gemini and Groq backends. Evaluates VLM outputs on:
- Accuracy: Are the objects and conditions described correctly?
- Completeness: Does the description cover all important elements?
- Safety relevance: Are hazards and actions appropriate?
- Actionability: Would the meta-actions be useful for an AD system?
"""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator for autonomous driving scene descriptions.

You are given:
1. A ground truth summary of a driving scene (from human annotations)
2. An AI-generated scene description of the same image

Rate the AI-generated description on these criteria (1-5 scale):

1. **Accuracy** (1-5): Are the objects, weather, and lighting correctly identified?
2. **Completeness** (1-5): Does it cover all important scene elements?
3. **Safety Relevance** (1-5): Are hazards identified and appropriately serious?
4. **Actionability** (1-5): Are the recommended meta-actions reasonable and safe?

GROUND TRUTH:
{ground_truth}

AI-GENERATED DESCRIPTION:
{prediction}

Respond with JSON only:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "safety_relevance": <1-5>,
  "actionability": <1-5>,
  "overall": <1-5 weighted average>,
  "reasoning": "Brief explanation of your ratings"
}}"""

_EMPTY_RESULT: dict[str, Any] = {
    "accuracy": 0,
    "completeness": 0,
    "safety_relevance": 0,
    "actionability": 0,
    "overall": 0,
    "reasoning": "Evaluation failed",
}


def _judge_with_gemini(client, prompt: str) -> dict[str, Any]:
    """Run judge evaluation using the Gemini API."""
    from google.genai import types

    client.rate_limiter.wait_if_needed()

    response = client.client.models.generate_content(
        model=client.model_name,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    if response.text:
        return json.loads(response.text)
    return {}


def _judge_with_groq(client, prompt: str) -> dict[str, Any]:
    """Run judge evaluation using the Groq API."""
    client.rate_limiter.wait_if_needed()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert evaluator. Respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content
    if text:
        return json.loads(text)
    return {}


def evaluate_with_judge(
    prediction_text: str,
    ground_truth_text: str,
    client: Any = None,
) -> dict[str, Any]:
    """
    Use a VLM as a judge to evaluate a scene description.

    Supports both Gemini and Groq backends — the correct API is selected
    automatically based on the client type.

    Args:
        prediction_text: The VLM-generated scene description (JSON string or summary).
        ground_truth_text: The ground truth description from BDD100K.
        client: Optional VLM client instance (creates one via factory if not provided).

    Returns:
        Dict with accuracy, completeness, safety_relevance, actionability,
        overall scores, and reasoning.
    """
    if client is None:
        from src.vlm import create_vlm_client
        client = create_vlm_client()

    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth_text,
        prediction=prediction_text,
    )

    try:
        from src.vlm.groq_client import GroqClient

        if isinstance(client, GroqClient):
            result = _judge_with_groq(client, prompt)
        else:
            result = _judge_with_gemini(client, prompt)

        return {
            "accuracy": result.get("accuracy", 0),
            "completeness": result.get("completeness", 0),
            "safety_relevance": result.get("safety_relevance", 0),
            "actionability": result.get("actionability", 0),
            "overall": result.get("overall", 0),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        logger.warning(f"Judge evaluation failed: {e}")

    return dict(_EMPTY_RESULT)
