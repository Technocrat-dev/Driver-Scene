"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration loaded from .env file."""

    # API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Rate limiting (free tier)
    rate_limit_rpm: int = 10
    rate_limit_rpd: int = 250

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # Dataset
    sample_size: int = 200
    eval_subset_size: int = 50
    random_seed: int = 42

    # Processing
    max_retries: int = 3
    retry_delay: float = 2.0

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
