"""Groq VLM client with rate limiting and structured output parsing.

Uses the Groq API (OpenAI-compatible) with Llama Vision models for
scene description generation. Much more generous free tier than Gemini
(14,400 RPD vs ~20 effective RPD).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import SceneDescription

logger = logging.getLogger(__name__)

# Groq vision models ranked by capability
GROQ_VISION_MODELS = [
    "llama-3.2-90b-vision-preview",   # Best quality
    "llama-3.2-11b-vision-preview",   # Faster, still good
]


class GroqClient:
    """Wrapper around the Groq API for structured scene description generation.

    Uses Llama 3.2 Vision models via Groq's ultra-fast inference.
    Free tier: 14,400 RPD, 30 RPM — far more generous than Gemini.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )

        self.api_key = api_key or settings.groq_api_key
        self.model_name = model or settings.groq_model
        if not self.api_key:
            raise ValueError(
                "Groq API key not set. Get a free key at https://console.groq.com/keys\n"
                "Then set GROQ_API_KEY in your .env file."
            )

        self.client = Groq(api_key=self.api_key)
        self._request_count = 0

    def generate_scene_description(
        self,
        image_path: Path,
        prompt_text: str,
        max_retries: int | None = None,
    ) -> SceneDescription | None:
        """
        Send an image + prompt to Groq and parse the structured response.

        Args:
            image_path: Path to the driving scene image.
            prompt_text: The full prompt text (from a prompt template).
            max_retries: Number of retries on transient errors.

        Returns:
            Parsed SceneDescription or None if all retries fail.
        """
        retries = max_retries or settings.max_retries
        image_b64 = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        # Build the system prompt to enforce JSON output
        system_prompt = (
            "You are an autonomous driving perception system. "
            "You MUST respond with valid JSON only, no markdown fences, no explanation. "
            "The JSON must match the schema requested in the user prompt."
        )

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_b64}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                },
                            ],
                        },
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )

                self._request_count += 1
                text = response.choices[0].message.content

                if text:
                    return self._parse_response(text)
                else:
                    logger.warning(f"Empty response for {image_path.name}")
                    return None

            except Exception as e:
                error_msg = str(e)
                wait_time = min(settings.retry_delay * (2**attempt), 60.0)

                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{retries + 1}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                elif attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {error_msg}. "
                        f"Retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {retries + 1} attempts: {error_msg}")
                    return None

        return None

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type from file extension."""
        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return mime_map.get(suffix, "image/jpeg")

    def _parse_response(self, response_text: str) -> SceneDescription | None:
        """Parse JSON response into SceneDescription."""
        try:
            # Strip markdown fences if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            data = json.loads(text)
            return SceneDescription.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse response: {e}")
            return self._fallback_parse(response_text)

    def _fallback_parse(self, text: str) -> SceneDescription | None:
        """Attempt to extract structured data from malformed JSON."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return SceneDescription.model_validate(data)
        except Exception:
            pass

        logger.error(f"Fallback parsing also failed. Raw response:\n{text[:500]}")
        return None

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def requests_remaining(self) -> int:
        return 14400 - self._request_count  # Approximate
