from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent
    model_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"

    # Model
    active_model: str = "distilbert-base-uncased"
    use_onnx: bool = False
    max_seq_length: int = 128

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    rate_limit: str = "60/minute"
    cors_origins: list[str] = ["*"]

    # Inference
    confidence_threshold: float = 0.5
    max_batch_size: int = 100
    max_input_length: int = 5000

    # Feedback
    feedback_db_path: Path = project_root / "feedback.db"
    log_uncertain_threshold: float = 0.4

    model_config = {"env_prefix": "CLAIM_", "env_file": ".env"}


settings = Settings()
