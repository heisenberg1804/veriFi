"""Abstract base class for all detection signals."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DetectionResult:
    """Result from a single detector on a single frame."""
    score: float
    raw_logits: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class BaseDetector(ABC):
    """Interface all detectors must implement."""

    def __init__(self, device: str):
        self.device = device
        self._model = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        ...

    @abstractmethod
    def get_gradcam_target_layer(self):
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
