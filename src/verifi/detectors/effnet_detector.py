"""EfficientNet-B4 artifact-focused deepfake detector."""
import numpy as np
import structlog
import timm
import torch
from torchvision import transforms

from verifi.detectors.base import BaseDetector, DetectionResult

logger = structlog.get_logger()


class EfficientNetDetector(BaseDetector):
    """
    EfficientNet-B4 trained on FF++ (DeepfakeBench weights).
    Focuses on pixel-level artifacts: blending boundaries,
    compression inconsistencies, texture anomalies.
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
        logger.info("loading_effnet_detector", path=self.weight_path, device=self.device)
        self._model = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=2
        )
        try:
            state = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state, strict=False)
        except FileNotFoundError:
            logger.warn("effnet_weights_not_found_using_imagenet", path=self.weight_path)
            self._model = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=2
            )
        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("effnet_loaded")

    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        if not face_crops or not self._loaded:
            return []

        tensors = [self._transform(c[:, :, ::-1].copy()) for c in face_crops]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 0].cpu().numpy()

        return [
            DetectionResult(score=float(p), raw_logits=logits[i].cpu().numpy())
            for i, p in enumerate(fake_probs)
        ]

    def get_gradcam_target_layer(self):
        return self._model.conv_head
