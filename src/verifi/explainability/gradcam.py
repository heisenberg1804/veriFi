"""
GradCAM heatmap generation for CLIP ViT-L/14 and EfficientNet-B4.
Supports both face crops and full frames.

Save to: src/verifi/explainability/gradcam.py

Key challenge: ViT outputs flattened patch tokens, not spatial feature maps.
We use a reshape_transform to convert (B, 1+HW, D) → (B, D, H, W).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = structlog.get_logger()


@dataclass
class HeatmapResult:
    """Output from a GradCAM computation."""
    overlay: np.ndarray          # BGR uint8 (H, W, 3) — heatmap overlaid on image
    raw_cam: np.ndarray          # Grayscale float (H, W) in [0, 1]
    source_image: np.ndarray     # Original input image (BGR)
    frame_idx: int
    model_name: str              # "clip" or "effnet"
    target_class: int = 0        # 0 = fake


def reshape_transform_vit(tensor: torch.Tensor, height: int = 16, width: int = 16):
    """
    Reshape ViT output for GradCAM compatibility.

    ViT-L/14 outputs: (B, 1 + H*W, D) where the first token is [CLS].
    For 224px input with patch_size=14: H=W=16, D=1024.

    We strip [CLS], reshape patches to a spatial grid, and permute to (B, D, H, W).
    """
    # Strip class token, keep only patch tokens
    result = tensor[:, 1:, :]
    # Reshape to spatial grid
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # Permute to channel-first: (B, H, W, D) → (B, D, H, W)
    result = result.permute(0, 3, 1, 2)
    return result


class GradCAMGenerator:
    """
    Generate GradCAM heatmaps for detection models.

    Handles both ViT (needs reshape_transform) and CNN (standard) architectures.
    Supports face crops and full frames.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def generate(
        self,
        model: torch.nn.Module,
        target_layer,
        image_bgr: np.ndarray,
        preprocess_fn,
        frame_idx: int = 0,
        model_name: str = "unknown",
        is_vit: bool = False,
        target_class: int = 0,
        input_size: int | None = None,
    ) -> HeatmapResult | None:
        """
        Generate a GradCAM heatmap for a single image.

        Args:
            model: The detection model (must be in eval mode).
            target_layer: The layer to hook for gradient computation.
            image_bgr: Input image as BGR numpy array.
            preprocess_fn: Transform that converts BGR numpy → model input tensor.
            frame_idx: Frame index for tracking.
            model_name: "clip" or "effnet" for labeling.
            is_vit: If True, applies ViT reshape transform.
            target_class: Class index to explain (0 = fake).
            input_size: Expected input size (for ViT grid calculation).

        Returns:
            HeatmapResult or None if generation fails.
        """
        if target_layer is None:
            logger.warn("gradcam_no_target_layer", model=model_name)
            return None

        try:
            # Prepare input
            rgb = image_bgr[:, :, ::-1].copy()
            input_tensor = preprocess_fn(rgb).unsqueeze(0)

            # Move to CPU for GradCAM (MPS has issues with hooks)
            model_cpu = model.cpu()
            model_cpu.eval()
            input_tensor = input_tensor.cpu()

            # Build GradCAM kwargs
            cam_kwargs = {
                "model": model_cpu,
                "target_layers": [target_layer],
            }

            if is_vit:
                patch_size = 14  # ViT-L/14
                img_size = input_size or 224
                grid_size = img_size // patch_size  # 16 for 224px
                cam_kwargs["reshape_transform"] = (
                    lambda t: reshape_transform_vit(t, height=grid_size, width=grid_size)
                )

            # Run GradCAM
            # If model outputs a single score (like zero-shot similarity),
            # use None targets (maximizes the output)
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]
            else:
                targets = None

            with GradCAM(**cam_kwargs) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets,
                )
                grayscale_cam = grayscale_cam[0, :]  # (H, W) float [0,1]

            # Generate overlay on original-size image
            orig_h, orig_w = image_bgr.shape[:2]
            cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))

            # Normalize RGB to [0,1] for show_cam_on_image
            rgb_normalized = rgb.astype(np.float32) / 255.0
            rgb_resized = cv2.resize(rgb_normalized, (orig_w, orig_h))

            overlay_rgb = show_cam_on_image(rgb_resized, cam_resized, use_rgb=True)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            # Move model back to original device
            model.to(self.device)

            return HeatmapResult(
                overlay=overlay_bgr,
                raw_cam=cam_resized,
                source_image=image_bgr,
                frame_idx=frame_idx,
                model_name=model_name,
                target_class=target_class,
            )

        except Exception as e:
            logger.error(
                "gradcam_failed",
                model=model_name,
                frame_idx=frame_idx,
                error=str(e),
            )
            # Move model back even on failure
            try:
                model.to(self.device)
            except Exception:
                pass
            return None

    def generate_batch(
        self,
        model: torch.nn.Module,
        target_layer,
        images: list[tuple[np.ndarray, int]],
        preprocess_fn,
        model_name: str = "unknown",
        is_vit: bool = False,
        target_class: int = 0,
        input_size: int | None = None,
        max_heatmaps: int = 5,
    ) -> list[HeatmapResult]:
        """
        Generate GradCAM heatmaps for multiple images.
        Only processes top max_heatmaps to save compute.

        Args:
            images: List of (image_bgr, frame_idx) tuples.
            Other args: same as generate().

        Returns:
            List of HeatmapResult (may be shorter than input if some fail).
        """
        results = []
        for image_bgr, frame_idx in images[:max_heatmaps]:
            result = self.generate(
                model=model,
                target_layer=target_layer,
                image_bgr=image_bgr,
                preprocess_fn=preprocess_fn,
                frame_idx=frame_idx,
                model_name=model_name,
                is_vit=is_vit,
                target_class=target_class,
                input_size=input_size,
            )
            if result is not None:
                results.append(result)

        logger.info(
            "gradcam_batch_complete",
            model=model_name,
            requested=min(len(images), max_heatmaps),
            generated=len(results),
        )
        return results

