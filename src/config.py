"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration loaded from .env file."""

    # VLM Provider: "gemini" or "groq"
    vlm_provider: str = "gemini"

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    # Groq API (free tier: 14,400 RPD, 30 RPM)
    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Rate limiting (free tier)
    rate_limit_rpm: int = 15
    rate_limit_rpd: int = 1000

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # Dataset
    sample_size: int = 200
    eval_subset_size: int = 50
    random_seed: int = 42

    # Processing
    max_retries: int = 6
    retry_delay: float = 4.0
    rate_limit_cooldown: float = 2.0  # Extra delay between requests to avoid 429s

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def sampled_dir(self) -> Path:
        return self.data_dir / "sampled"

    @property
    def images_dir(self) -> Path:
        return self.raw_dir / "images" / "100k" / "val"

    @property
    def labels_file(self) -> Path:
        return self.raw_dir / "labels" / "bdd100k_labels_images_val.json"


settings = Settings()
