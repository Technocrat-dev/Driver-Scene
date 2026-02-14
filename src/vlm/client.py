"""Gemini VLM client with rate limiting and structured output parsing."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from src.config import settings
from src.models import SceneDescription

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter for RPM and RPD constraints."""

    def __init__(self, rpm: int = 10, rpd: int = 250):
        self.rpm = rpm
        self.rpd = rpd
        self._minute_timestamps: list[float] = []
        self._day_count = 0
        self._day_start = time.time()

    def wait_if_needed(self) -> None:
        """Block until the next request is allowed."""
        now = time.time()

        # Reset daily counter if 24h passed
        if now - self._day_start > 86400:
            self._day_count = 0
            self._day_start = now

        # Check daily limit
        if self._day_count >= self.rpd:
            sleep_time = 86400 - (now - self._day_start)
            logger.warning(f"Daily rate limit reached. Sleeping {sleep_time:.0f}s")
            raise RuntimeError(
                f"Daily rate limit of {self.rpd} requests reached. "
                "Resume tomorrow or increase limit with a paid plan."
            )

        # Check per-minute limit
        self._minute_timestamps = [t for t in self._minute_timestamps if now - t < 60]
        if len(self._minute_timestamps) >= self.rpm:
            oldest = self._minute_timestamps[0]
            sleep_time = 60 - (now - oldest) + 0.5
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self._minute_timestamps.append(time.time())
        self._day_count += 1

    @property
    def requests_remaining_today(self) -> int:
        return max(0, self.rpd - self._day_count)


class GeminiClient:
    """Wrapper around the Gemini API for structured scene description generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rpm: int | None = None,
        rpd: int | None = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model
        if not self.api_key:
            raise ValueError(
                "Gemini API key not set. Get a free key at https://aistudio.google.com/apikey\n"
                "Then set GEMINI_API_KEY in your .env file."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.rate_limiter = RateLimiter(
            rpm=rpm or settings.rate_limit_rpm,
            rpd=rpd or settings.rate_limit_rpd,
        )
        self._request_count = 0

    def generate_scene_description(
        self,
        image_path: Path,
        prompt_text: str,
        max_retries: int | None = None,
    ) -> SceneDescription | None:
        """
        Send an image + prompt to Gemini and parse the structured response.

        Args:
            image_path: Path to the driving scene image.
            prompt_text: The full prompt text (from a prompt template).
            max_retries: Number of retries on transient errors.

        Returns:
            Parsed SceneDescription or None if all retries fail.
        """
        retries = max_retries or settings.max_retries
        image_data = self._load_image(image_path)

        for attempt in range(retries + 1):
            try:
                self.rate_limiter.wait_if_needed()

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(
                                    data=image_data["bytes"],
                                    mime_type=image_data["mime_type"],
                                ),
                                types.Part.from_text(text=prompt_text),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=SceneDescription,
                    ),
                )

                self._request_count += 1

                if response.text:
                    return self._parse_response(response.text)
                else:
                    logger.warning(f"Empty response for {image_path.name}")
                    return None

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = settings.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{retries + 1}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                elif attempt < retries:
                    wait_time = settings.retry_delay * (2**attempt)
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {error_msg}. "
                        f"Retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {retries + 1} attempts: {error_msg}")
                    return None

        return None

    def _load_image(self, image_path: Path) -> dict[str, Any]:
        """Load image bytes and determine MIME type."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        return {"bytes": img_bytes, "mime_type": mime_type}

    def _parse_response(self, response_text: str) -> SceneDescription | None:
        """Parse JSON response into SceneDescription."""
        try:
            data = json.loads(response_text)
            return SceneDescription.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse response: {e}")
            # Try to salvage partial data
            return self._fallback_parse(response_text)

    def _fallback_parse(self, text: str) -> SceneDescription | None:
        """Attempt to extract structured data from malformed JSON."""
        try:
            # Try to find JSON object in the response
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
        return self.rate_limiter.requests_remaining_today
