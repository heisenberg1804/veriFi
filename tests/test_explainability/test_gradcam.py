
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE 2: tests/test_explainability/test_gradcam.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for GradCAM generation."""
import numpy as np
import torch
from torchvision import transforms

from verifi.explainability.gradcam import (
    GradCAMGenerator,
    HeatmapResult,
    reshape_transform_vit,
)


def test_reshape_transform_correct_shape():
    """ViT reshape should produce (B, D, H, W) from (B, 1+HW, D)."""
    B, H, W, D = 2, 16, 16, 1024
    # Input: (B, 1 + H*W, D) — class token + patch tokens
    tensor = torch.randn(B, 1 + H * W, D)
    result = reshape_transform_vit(tensor, height=H, width=W)
    assert result.shape == (B, D, H, W)


def test_reshape_transform_strips_cls_token():
    """Should remove the first token (class token)."""
    tensor = torch.randn(1, 257, 768)  # 1 + 16*16 = 257
    result = reshape_transform_vit(tensor, height=16, width=16)
    # Class token stripped: 257-1 = 256 = 16*16 patches
    assert result.shape == (1, 768, 16, 16)


def test_heatmap_result_dataclass():
    result = HeatmapResult(
        overlay=np.zeros((224, 224, 3), dtype=np.uint8),
        raw_cam=np.zeros((224, 224), dtype=np.float32),
        source_image=np.zeros((224, 224, 3), dtype=np.uint8),
        frame_idx=42,
        model_name="clip",
    )
    assert result.frame_idx == 42
    assert result.model_name == "clip"


def test_gradcam_generator_handles_none_layer():
    """Should return None if target_layer is None."""
    gen = GradCAMGenerator(device="cpu")
    result = gen.generate(
        model=torch.nn.Linear(10, 2),
        target_layer=None,
        image_bgr=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        preprocess_fn=transforms.ToTensor(),
        model_name="test",
    )
    assert result is None

