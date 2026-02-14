"""VLM client abstraction layer supporting Gemini and Groq backends."""

from __future__ import annotations

from src.config import settings


def create_vlm_client():
    """Create a VLM client based on the configured provider.

    Returns:
        GeminiClient or GroqClient instance.

    Raises:
        ValueError: If VLM_PROVIDER is not 'gemini' or 'groq'.
    """
    provider = settings.vlm_provider.lower()

    if provider == "gemini":
        from src.vlm.client import GeminiClient
        return GeminiClient()
    elif provider == "groq":
        from src.vlm.groq_client import GroqClient
        return GroqClient()
    else:
        raise ValueError(
            f"Unknown VLM provider: '{provider}'. "
            "Set VLM_PROVIDER to 'gemini' or 'groq' in your .env file."
        )


__all__ = ["create_vlm_client"]
