"""
Detection tools: wrap each detector as a callable Tool.

Save to: src/verifi/tools/detection_tools.py

These tools can be called by:
- Tier 1 (fixed pipeline): called in sequence on all frames
- Tier 3 (agent): called selectively on specific frames/regions
"""
from __future__ import annotations

import numpy as np

from verifi.detectors.clip_detector import CLIPDeepfakeDetector
from verifi.detectors.effnet_detector import EfficientNetDetector
from verifi.detectors.frequency import FrequencyAnalyzer
from verifi.detectors.temporal import TemporalAnalyzer
from verifi.tools.base import Tool, ToolResult


class CLIPDetectionTool(Tool):
    """Run zero-shot CLIP deepfake detection on an image."""

    def __init__(self, detector: CLIPDeepfakeDetector):
        self._detector = detector

    @property
    def name(self) -> str:
        return "run_clip_detection"

    @property
    def description(self) -> str:
        return (
            "Run CLIP ViT-L/14 zero-shot deepfake detection on an image. "
            "Works on both face crops and full frames. Returns a score from "
            "0.0 (authentic) to 1.0 (AI-generated). Higher scores indicate "
            "the image looks more like AI-generated content to CLIP. "
            "This is the primary detection signal with best generalization."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the image from the shared context (e.g., 'frame_42', 'face_0_frame_42')",
            },
        }

    def execute(self, image: np.ndarray | None = None, **kwargs) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        results = self._detector.predict([image])
        if not results:
            return ToolResult(
                tool_name=self.name, success=False,
                error="CLIP prediction returned empty",
            )

        r = results[0]
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "score": r.score,
                "metadata": r.metadata,
                "summary": f"CLIP score: {r.score:.3f} ({'suspicious' if r.score > 0.5 else 'likely authentic'})",
            },
        )


class EfficientNetDetectionTool(Tool):
    """Run EfficientNet-B4 artifact detection on an image."""

    def __init__(self, detector: EfficientNetDetector):
        self._detector = detector

    @property
    def name(self) -> str:
        return "run_effnet_detection"

    @property
    def description(self) -> str:
        return (
            "Run EfficientNet-B4 on an image to detect pixel-level artifacts "
            "like blending boundaries, texture anomalies, and compression "
            "inconsistencies. Best for face crops where manipulation boundaries "
            "are visible. NOT reliable on images smaller than 80x80 pixels "
            "(upscaling artifacts create false positives). Returns 0.0-1.0."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the image from shared context",
            },
        }

    def execute(self, image: np.ndarray | None = None, **kwargs) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        # Size check — skip tiny images
        h, w = image.shape[:2]
        if h < 80 or w < 80:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "score": 0.5,
                    "skipped": True,
                    "reason": f"Image too small ({w}x{h}), skipping to avoid upscaling artifacts",
                    "summary": f"EfficientNet skipped: image too small ({w}x{h}px)",
                },
            )

        results = self._detector.predict([image])
        if not results:
            return ToolResult(
                tool_name=self.name, success=False,
                error="EfficientNet prediction returned empty",
            )

        r = results[0]
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "score": r.score,
                "summary": f"EfficientNet score: {r.score:.3f}",
            },
        )


class DCTFrequencyTool(Tool):
    """Analyze frequency-domain characteristics of an image."""

    def __init__(self, analyzer: FrequencyAnalyzer | None = None):
        self._analyzer = analyzer or FrequencyAnalyzer()

    @property
    def name(self) -> str:
        return "run_dct_analysis"

    @property
    def description(self) -> str:
        return (
            "Run DCT (Discrete Cosine Transform) frequency analysis on an image. "
            "This is a physics-based detector with NO machine learning — it measures "
            "the frequency spectrum of the image. AI generators suppress high-frequency "
            "detail and produce unnaturally smooth spectral rolloff. Returns three "
            "sub-scores: band_energy (HF suppression), spectral_smoothness, and "
            "periodic_artifacts (GAN checkerboard). Combined score 0.0-1.0. "
            "Use on full frames for best results — also works on face crops."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the image from shared context",
            },
        }

    def execute(self, image: np.ndarray | None = None, **kwargs) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        result = self._analyzer.analyze(image)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "score": result.score,
                **result.metadata,
                "summary": (
                    f"DCT score: {result.score:.3f} "
                    f"(HF ratio: {result.metadata.get('high_ratio', 0):.3f}, "
                    f"smoothness: {result.metadata.get('spectral_smoothness', 0):.3f})"
                ),
            },
        )


class TemporalConsistencyTool(Tool):
    """Check motion consistency between two frames."""

    def __init__(self, analyzer: TemporalAnalyzer | None = None):
        self._analyzer = analyzer or TemporalAnalyzer()

    @property
    def name(self) -> str:
        return "run_temporal_analysis"

    @property
    def description(self) -> str:
        return (
            "Compare optical flow consistency between two consecutive frames. "
            "Detects motion discontinuities where the face region moves "
            "differently from the background — a sign of face-swap manipulation. "
            "Fully AI-generated video (Sora, Veo) is usually temporally smooth "
            "and will score LOW on this tool. Returns divergence score 0.0-1.0."
        )

    @property
    def parameters(self) -> dict:
        return {
            "frame_a_key": {
                "type": "string",
                "description": "Key for the first frame",
            },
            "frame_b_key": {
                "type": "string",
                "description": "Key for the second (next) frame",
            },
        }

    def execute(
        self,
        frame_a: np.ndarray | None = None,
        frame_b: np.ndarray | None = None,
        face_bbox: tuple | None = None,
        **kwargs,
    ) -> ToolResult:
        if frame_a is None or frame_b is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Two frames required for temporal analysis",
            )

        result = self._analyzer.analyze_pair(frame_a, frame_b, face_bbox)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "score": result.score,
                **result.metadata,
                "summary": (
                    f"Temporal divergence: {result.metadata.get('divergence_sigma', 0):.2f}σ "
                    f"({'anomalous' if result.score > 0.5 else 'consistent'})"
                ),
            },
        )
