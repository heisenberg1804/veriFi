"""Centralized configuration with Pydantic Settings + YAML override."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class DeviceConfig(BaseSettings):
    preferred: str = "auto"

    def resolve(self) -> str:
        if self.preferred != "auto":
            return self.preferred
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class SamplingConfig(BaseSettings):
    frame_budget: int = 30
    scene_threshold: float = 30.0
    transition_margin: int = 2
    min_laplacian_var: float = 100.0
    min_face_confidence: float = 0.95


class DetectorConfig(BaseSettings):
    clip_weight_path: Path = Path("data/weights/clip_vit_l14_deepfake.torchscript")
    effnet_weight_path: Path = Path("data/weights/efficientnet_b4_ff.pth")
    clip_input_size: int = 224
    effnet_input_size: int = 380


class EnsembleConfig(BaseSettings):
    clip_weight: float = 0.45
    effnet_weight: float = 0.30
    frequency_weight: float = 0.15
    temporal_weight: float = 0.10
    suspicious_threshold: float = 0.30
    manipulated_threshold: float = 0.70


class ExplainerConfig(BaseSettings):
    model: str = "claude-sonnet-4-20250514"
    max_heatmap_frames: int = 5
    max_tokens: int = 1500


class AppConfig(BaseSettings):
    project_root: Path = Path(".")
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    explainer: ExplainerConfig = Field(default_factory=ExplainerConfig)
    max_video_duration_sec: int = 600

    class Config:
        env_prefix = "VERIFI_"
        env_nested_delimiter = "__"


def load_config() -> AppConfig:
    """Load config from environment variables (+ .env file)."""
    from dotenv import load_dotenv
    load_dotenv()
    return AppConfig()
