"""
Analysis and explainability tools.

Save to: src/verifi/tools/analysis_tools.py

These tools generate visual evidence and explanations.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from verifi.detectors.frequency import FrequencyAnalyzer
from verifi.explainability.gradcam import GradCAMGenerator
from verifi.explainability.heatmap_renderer import create_forensic_view
from verifi.tools.base import Tool, ToolResult


class GradCAMTool(Tool):
    """Generate GradCAM heatmap showing where a model detects anomalies."""

    def __init__(self, gradcam_gen: GradCAMGenerator):
        self._gen = gradcam_gen

    @property
    def name(self) -> str:
        return "generate_gradcam"

    @property
    def description(self) -> str:
        return (
            "Generate a GradCAM heatmap showing which spatial regions of an image "
            "triggered the detection model. Bright red = high anomaly signal. "
            "Produces a visual overlay image. Use this AFTER running a detection "
            "tool to understand WHERE in the image the signal is coming from. "
            "Helps distinguish boundary artifacts (face-swap) from full-frame "
            "anomalies (synthetic content)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the image",
            },
            "model_name": {
                "type": "string",
                "description": "Which model to explain: 'clip' or 'effnet'",
                "enum": ["clip", "effnet"],
            },
        }

    def execute(
        self,
        image: np.ndarray | None = None,
        model=None,
        target_layer=None,
        preprocess_fn=None,
        model_name: str = "clip",
        is_vit: bool = True,
        input_size: int = 224,
        frame_idx: int = 0,
        **kwargs,
    ) -> ToolResult:
        if image is None or model is None or target_layer is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Image, model, and target_layer required",
            )

        result = self._gen.generate(
            model=model,
            target_layer=target_layer,
            image_bgr=image,
            preprocess_fn=preprocess_fn,
            frame_idx=frame_idx,
            model_name=model_name,
            is_vit=is_vit,
            input_size=input_size,
        )

        if result is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"GradCAM generation failed for {model_name}",
            )

        # Compute heat concentration — where is the attention focused?
        cam = result.raw_cam
        hot_pixels = float((cam > 0.5).sum() / cam.size) * 100  # % of image that's "hot"

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "overlay": result.overlay,
                "raw_cam": result.raw_cam,
                "model_name": model_name,
                "frame_idx": frame_idx,
                "hot_region_pct": round(hot_pixels, 1),
                "summary": (
                    f"GradCAM ({model_name}): "
                    f"{hot_pixels:.1f}% of image is 'hot'. "
                    + (
                        "Diffuse attention = full-frame anomaly"
                        if hot_pixels > 30
                        else "Focused attention = localized anomaly"
                    )
                ),
            },
        )


class ForensicViewTool(Tool):
    """Create a three-panel forensic comparison view."""

    def __init__(self, freq_analyzer: FrequencyAnalyzer | None = None):
        self._freq = freq_analyzer or FrequencyAnalyzer()

    @property
    def name(self) -> str:
        return "create_forensic_view"

    @property
    def description(self) -> str:
        return (
            "Create a three-panel forensic view: Original | GradCAM Heatmap | DCT Spectrum. "
            "Use this to generate visual evidence for the forensic report. "
            "The side-by-side comparison makes it easy for human reviewers to see "
            "what the detectors found."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the original frame",
            },
            "frame_idx": {
                "type": "integer",
                "description": "Frame index for labeling",
            },
        }

    def execute(
        self,
        image: np.ndarray | None = None,
        heatmap_overlay: np.ndarray | None = None,
        frame_idx: int = 0,
        output_dir: str | Path | None = None,
        **kwargs,
    ) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        from verifi.explainability.gradcam import HeatmapResult

        hm_result = None
        if heatmap_overlay is not None:
            hm_result = HeatmapResult(
                overlay=heatmap_overlay,
                raw_cam=np.zeros((1, 1)),
                source_image=image,
                frame_idx=frame_idx,
                model_name="clip",
            )

        forensic = create_forensic_view(
            original=image,
            heatmap_result=hm_result,
            frame_idx=frame_idx,
            freq_analyzer=self._freq,
        )

        path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"forensic_frame{frame_idx:04d}.png"
            cv2.imwrite(str(path), forensic)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "forensic_image": forensic,
                "saved_path": str(path) if path else None,
                "summary": f"Forensic view created for frame {frame_idx}"
                           + (f", saved to {path}" if path else ""),
            },
        )


class QuickScanTool(Tool):
    """
    Run the full Tier 1 pipeline as a single tool call.

    This is the bridge between the fixed pipeline and the agent.
    The agent calls this first, then decides what to investigate further.
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: VeriFiPipeline instance (already initialized with models).
        """
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "quick_scan"

    @property
    def description(self) -> str:
        return (
            "Run the complete Tier 1 detection pipeline on the video: "
            "smart frame sampling, face detection, dual-path analysis "
            "(face-level + frame-level), ensemble aggregation. "
            "Returns structured detection results including per-frame scores, "
            "flagged frames, verdict, and signal statistics. "
            "This should ALWAYS be your first tool call. Use the results to "
            "decide what to investigate further."
        )

    @property
    def parameters(self) -> dict:
        return {}  # Uses the video already in context

    def execute(self, video_path: str | None = None, **kwargs) -> ToolResult:
        if video_path is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No video path provided",
            )

        try:
            report = self._pipeline.analyze(video_path)

            analysis = report.analysis
            stats = report.signal_stats

            # Build a concise summary for the LLM
            frame_scores = [
                {"frame_idx": a.frame_idx, "timestamp": a.timestamp_sec,
                 "score": round(a.ensemble_score, 3), "flagged": a.flagged}
                for a in analysis.frame_analyses
            ]

            face_summary = "No faces detected"
            if analysis.face_analyses:
                face_ids = set(a.face_id for a in analysis.face_analyses)
                face_summary = (
                    f"{len(analysis.face_analyses)} face detections across "
                    f"{len(face_ids)} unique face(s). "
                    f"Face path score: {analysis.face_path_score:.3f}"
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "video_score": analysis.video_score,
                    "verdict": analysis.verdict.value,
                    "manipulation_type": analysis.manipulation_type.value,
                    "dominant_path": analysis.dominant_path,
                    "face_path_score": analysis.face_path_score,
                    "frame_path_score": analysis.frame_path_score,
                    "face_summary": face_summary,
                    "signal_stats": stats,
                    "frame_scores": frame_scores,
                    "num_frames_analyzed": len(analysis.frame_analyses),
                    "num_frames_flagged": len(analysis.flagged_frame_indices),
                    "heatmap_paths": report.heatmap_paths,
                    "forensic_view_paths": report.forensic_view_paths,
                    "output_dir": report.output_dir,
                    "timings": report.timings.to_dict(),
                    "summary": (
                        f"Verdict: {analysis.verdict.value} "
                        f"(score: {analysis.video_score:.3f}, "
                        f"dominant: {analysis.dominant_path} path). "
                        f"{face_summary}. "
                        f"Frame path: {analysis.frame_path_score:.3f} "
                        f"({len(analysis.flagged_frame_indices)}"
                        f"/{len(analysis.frame_analyses)} flagged). "
                        f"DCT: {stats.get('freq_score', 0):.3f}. "
                        f"CLIP frame mean: {stats.get('frame_clip_mean', 0):.3f}."
                    ),
                },
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Pipeline failed: {str(e)}",
            )
