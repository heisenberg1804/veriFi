"""EfficientNet-B4 artifact-focused deepfake detector."""
import numpy as np
import structlog
import torch
from torch import nn
from torchvision import transforms

from verifi.detectors.base import BaseDetector, DetectionResult

logger = structlog.get_logger()


class EfficientNetDetector(BaseDetector):
    """
    EfficientNet-B4 trained on FF++ (DeepfakeBench weights).
    Focuses on pixel-level artifacts: blending boundaries,
    compression inconsistencies, texture anomalies.

    Uses efficientnet_pytorch library to match DeepfakeBench checkpoint format.
    """

    def __init__(self, weight_path: str, device: str, input_size: int = 380):
        super().__init__(device)
        self.weight_path = weight_path
        self.input_size = input_size
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        from efficientnet_pytorch import EfficientNet

        logger.info("loading_effnet_detector", path=self.weight_path, device=self.device)

        self._model = EfficientNet.from_name("efficientnet-b4")
        # Replace final FC with 2-class head (real vs fake)
        self._model._fc = nn.Linear(1792, 2)

        try:
            raw_state = torch.load(
                self.weight_path, map_location=self.device, weights_only=True
            )
            mapped_state = self._map_deepfakebench_keys(raw_state)
            self._model.load_state_dict(mapped_state, strict=True)
            logger.info("effnet_loaded", weights="deepfakebench_ff++")
        except FileNotFoundError:
            logger.warn("effnet_weights_not_found_using_imagenet", path=self.weight_path)
            # Fall back to ImageNet pretrained
            self._model = EfficientNet.from_pretrained("efficientnet-b4")
            self._model._fc = nn.Linear(1792, 2)
            logger.info("effnet_loaded", weights="imagenet_pretrained")

        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    @staticmethod
    def _map_deepfakebench_keys(raw_state: dict) -> dict:
        """Map DeepfakeBench checkpoint keys to efficientnet_pytorch keys.

        DeepfakeBench prefixes backbone keys with "backbone.efficientnet."
        and stores the classifier as "backbone.last_layer".
        """
        mapped = {}
        for key, value in raw_state.items():
            if key.startswith("backbone.efficientnet."):
                new_key = key[len("backbone.efficientnet."):]
                mapped[new_key] = value
            elif key == "backbone.last_layer.weight":
                mapped["_fc.weight"] = value
            elif key == "backbone.last_layer.bias":
                mapped["_fc.bias"] = value
            else:
                # Pass through any unexpected keys as-is
                mapped[key] = value
        return mapped

    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        if not face_crops or not self._loaded:
            return []

        tensors = [self._transform(c[:, :, ::-1].copy()) for c in face_crops]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)
            # DeepfakeBench convention: index 1 = fake
            fake_probs = probs[:, 1].cpu().numpy()

        return [
            DetectionResult(score=float(p), raw_logits=logits[i].cpu().numpy())
            for i, p in enumerate(fake_probs)
        ]

    def get_gradcam_target_layer(self):
        return self._model._conv_head
