"""CLIP ViT-L/14 deepfake detector (LN-tuned, WACV 2026)."""
import numpy as np
import structlog
import torch
from torchvision import transforms

from verifi.detectors.base import BaseDetector, DetectionResult

logger = structlog.get_logger()


class CLIPDeepfakeDetector(BaseDetector):
    """
    CLIP ViT-L/14 fine-tuned via LayerNorm tuning on FaceForensics++.
    Best cross-dataset generalization of all open-source detectors.
    """

    def __init__(self, weight_path: str, device: str, input_size: int = 224):
        super().__init__(device)
        self.weight_path = weight_path
        self.input_size = input_size
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def load(self) -> None:
        logger.info("loading_clip_detector", path=self.weight_path, device=self.device)
        try:
            # Try TorchScript first
            self._model = torch.jit.load(self.weight_path, map_location=self.device)
            self._model.eval()
            self._model_format = "torchscript"
            logger.info("clip_loaded", format="torchscript")
        except Exception as e:
            # Fallback: load via open_clip + state dict
            logger.warn("torchscript_failed_trying_open_clip", error=str(e))
            self._load_via_open_clip()
        self._loaded = True

    def _load_via_open_clip(self) -> None:
        """Fallback: load CLIP via open_clip and apply saved weights."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        # If weight_path is a state dict (not torchscript), load it
        if self.weight_path.endswith(".pth"):
            state = torch.load(self.weight_path, map_location=self.device)
            model.load_state_dict(state, strict=False)

        self._model = model.visual.to(self.device)
        self._model.eval()
        self._model_format = "open_clip"

        # Add a classification head (2-class: fake/real)
        embed_dim = 768  # ViT-L/14 output dim
        self._classifier = torch.nn.Linear(embed_dim, 2).to(self.device)
        # NOTE: You'll need to load the classifier weights too
        # or train this head on FF++ — see Phase 3 benchmarking
        logger.info("clip_loaded", format="open_clip_fallback")

    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        if not face_crops or not self._loaded:
            return []

        tensors = []
        for crop in face_crops:
            rgb = crop[:, :, ::-1].copy()
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            if self._model_format == "torchscript":
                logits = self._model(batch)
            else:
                features = self._model(batch)
                logits = self._classifier(features)

            probs = torch.softmax(logits, dim=1)
            # Verify label order: index 0 = fake, index 1 = real
            fake_probs = probs[:, 0].cpu().numpy()

        return [
            DetectionResult(
                score=float(p),
                raw_logits=logits[i].cpu().numpy(),
            )
            for i, p in enumerate(fake_probs)
        ]

    def get_gradcam_target_layer(self):
        if self._model_format == "torchscript":
            # TorchScript may not expose named layers — needs testing
            logger.warn("gradcam_torchscript_limited")
            return None
        # For open_clip ViT-L/14:
        return self._model.transformer.resblocks[-1].ln_2
