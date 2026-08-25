"""
End-to-end pipeline orchestrator: video → forensic report.

Save to: src/verifi/pipeline/orchestrator.py

Runs the complete dual-path detection pipeline:
1. Validate video
2. Smart frame sampling
3. Face detection
4. Path A: face-level detection (CLIP + EfficientNet + DCT on face crops)
5. Path B: frame-level detection (CLIP + DCT on full frames)
6. Temporal consistency analysis
7. Ensemble aggregation
8. GradCAM heatmaps on flagged frames
9. Forensic view assembly
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog
import torch

from verifi.config import AppConfig
from verifi.detectors.clip_detector import CLIPDeepfakeDetector
from verifi.detectors.effnet_detector import EfficientNetDetector
from verifi.detectors.frequency import FrequencyAnalyzer
from verifi.detectors.noise_residual import NoiseResidualAnalyzer
from verifi.detectors.physics_reasoner import PhysicsReasoner
from verifi.detectors.temporal import TemporalAnalyzer
from verifi.ensemble.aggregator import (
    EnsembleWeights,
    FaceFrameAnalysis,
    FullFrameAnalysis,
    SignalScore,
    Verdict,
    VideoAnalysis,
    aggregate,
    compute_signal_stats,
)
from verifi.explainability.gradcam import GradCAMGenerator
from verifi.explainability.heatmap_renderer import (
    create_confidence_timeline,
    create_forensic_view,
    save_forensic_view,
    save_heatmap,
    save_timeline,
)
from verifi.ingestion.validator import validate_video
from verifi.preprocessing.face_detector import FaceDetectionPipeline
from verifi.sampling.frame_selector import (
    SelectedFrame,
    # SelectionResult,
    _pass1_sequential_decode,
    _pass2_load_pixels,
    _select_indices,
)

logger = structlog.get_logger()


@dataclass
class StageTimings:
    """Timing metrics for each pipeline stage."""
    validation: float = 0.0
    scene_detection: float = 0.0
    frame_selection: float = 0.0
    face_detection: float = 0.0
    face_path_inference: float = 0.0
    frame_path_inference: float = 0.0
    temporal_analysis: float = 0.0
    ensemble: float = 0.0
    physics_reasoning: float = 0.0
    gradcam: float = 0.0
    forensic_views: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


@dataclass
class ForensicReport:
    """Complete output of the pipeline."""
    video_metadata: dict
    analysis: VideoAnalysis
    signal_stats: dict
    heatmap_paths: list[str] = field(default_factory=list)
    forensic_view_paths: list[str] = field(default_factory=list)
    timeline_path: str = ""
    timings: StageTimings = field(default_factory=StageTimings)
    output_dir: str = ""
    low_sharpness_fallback: bool = False

    def summary(self) -> dict:
        d = {
            "video": self.video_metadata.get("filename", "unknown"),
            "duration": self.video_metadata.get("duration_sec", 0),
            **self.analysis.to_dict(),
            "signal_stats": self.signal_stats,
            "heatmaps_generated": len(self.heatmap_paths),
            "forensic_views_generated": len(self.forensic_view_paths),
            "timings": self.timings.to_dict(),
        }
        if self.low_sharpness_fallback:
            d["low_sharpness_fallback"] = True
        return d


class VeriFiPipeline:
    """
    Main pipeline orchestrator.

    Usage:
        pipeline = VeriFiPipeline(config)
        pipeline.load_models()
        report = pipeline.analyze("path/to/video.mp4")
        pipeline.unload_models()
    """

    def __init__(self, config: AppConfig | None = None):
        if config is None:
            config = AppConfig()
        self.config = config
        self.device = config.device.resolve()

        # Models (loaded lazily)
        self._clip: CLIPDeepfakeDetector | None = None
        self._effnet: EfficientNetDetector | None = None
        self._face_pipeline: FaceDetectionPipeline | None = None
        self._freq = FrequencyAnalyzer()
        self._noise_residual = NoiseResidualAnalyzer()
        self._temporal = TemporalAnalyzer()
        self._physics = PhysicsReasoner(
            backend=config.explainer.physics_backend,
            ollama_model=config.explainer.physics_model,
        )
        self._gradcam = GradCAMGenerator(device=self.device)
        self._models_loaded = False

    def load_models(self) -> None:
        """Load all ML models into memory."""
        logger.info("loading_models", device=self.device)
        t0 = time.perf_counter()

        self._clip = CLIPDeepfakeDetector(
            weight_path=str(self.config.detector.clip_weight_path),
            device=self.device,
            input_size=self.config.detector.clip_input_size,
        )
        self._clip.load()

        self._effnet = EfficientNetDetector(
            weight_path=str(self.config.detector.effnet_weight_path),
            device=self.device,
            input_size=self.config.detector.effnet_input_size,
        )
        self._effnet.load()

        self._face_pipeline = FaceDetectionPipeline(
            device="cpu",  # MTCNN faster on CPU
            target_size=self.config.detector.clip_input_size,
            margin_ratio=0.3,
            min_confidence=0.90,
        )
        self._face_pipeline.load()

        self._models_loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("models_loaded", elapsed=f"{elapsed:.1f}s")

    def unload_models(self) -> None:
        """Free GPU/memory."""
        if self._clip:
            self._clip.unload()
        if self._effnet:
            self._effnet.unload()
        self._models_loaded = False
        logger.info("models_unloaded")

    def analyze(
        self,
        video_path: str,
        output_dir: str | Path | None = None,
        skip_explainability: bool = False,
    ) -> ForensicReport:
        """
        Run the complete analysis pipeline on a video.

        Args:
            video_path: Path to video file.
            output_dir: Directory for heatmaps, forensic views, timeline.
                        Defaults to data/reports/<file_hash>/

        Returns:
            ForensicReport with all results.
        """
        if not self._models_loaded:
            self.load_models()

        timings = StageTimings()
        total_t0 = time.perf_counter()

        # ── Stage 1: Validate ──
        t0 = time.perf_counter()
        validation = validate_video(
            video_path,
            max_duration_sec=self.config.max_video_duration_sec,
        )
        timings.validation = time.perf_counter() - t0

        if not validation.valid:
            raise ValueError(f"Invalid video: {validation.errors}")

        meta = validation.metadata
        if output_dir is None:
            output_dir = Path(f"data/reports/{meta.file_hash.replace(':', '_')}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("pipeline_start", video=meta.filename, duration=f"{meta.duration_sec:.1f}s")

        # ── Stage 2: Pass 1 — sequential decode for stats + scene detection ──
        t0 = time.perf_counter()
        pass1 = _pass1_sequential_decode(
            video_path,
            scene_threshold=self.config.sampling.scene_threshold,
        )
        timings.scene_detection = time.perf_counter() - t0

        # ── Stage 3: Select + Pass 2 — load pixels for selected frames ──
        t0 = time.perf_counter()
        selection = _select_indices(
            pass1,
            frame_budget=self.config.sampling.frame_budget,
            transition_margin=self.config.sampling.transition_margin,
            min_laplacian_var=self.config.sampling.min_laplacian_var,
        )
        frames = _pass2_load_pixels(video_path, selection.indices, pass1)
        low_sharpness_fallback = selection.low_sharpness_fallback
        timings.frame_selection = time.perf_counter() - t0

        if not frames:
            raise ValueError("No frames selected — video may be too short")

        # ── Stage 4: Face detection ──
        t0 = time.perf_counter()
        frame_tuples = [(f.image, f.frame_idx, f.timestamp_sec) for f in frames]
        face_results = self._face_pipeline.detect_batch(frame_tuples)
        timings.face_detection = time.perf_counter() - t0

        # ── Stage 5: Path A — Face-level detection ──
        t0 = time.perf_counter()
        face_analyses = self._run_face_path(frames, face_results)
        timings.face_path_inference = time.perf_counter() - t0

        # ── Stage 6: Path B — Frame-level detection ──
        t0 = time.perf_counter()
        frame_analyses = self._run_frame_path(frames)
        timings.frame_path_inference = time.perf_counter() - t0

        # ── Stage 7: Temporal analysis ──
        t0 = time.perf_counter()
        self._run_temporal_analysis(frames, frame_analyses)
        timings.temporal_analysis = time.perf_counter() - t0

        # ── Stage 8: Ensemble aggregation ──
        t0 = time.perf_counter()
        weights = EnsembleWeights(
            face_clip=self.config.ensemble.clip_weight,
            face_effnet=self.config.ensemble.effnet_weight,
            face_dct=self.config.ensemble.frequency_weight,
            face_noise_residual=self.config.ensemble.noise_residual_weight,
            frame_clip=self.config.ensemble.clip_weight,
            frame_dct=self.config.ensemble.frequency_weight,
            frame_noise_residual=self.config.ensemble.noise_residual_weight,
            frame_channel_corr=self.config.ensemble.channel_corr_weight,
            frame_temporal=self.config.ensemble.temporal_weight,
            suspicious_threshold=self.config.ensemble.suspicious_threshold,
            manipulated_threshold=self.config.ensemble.manipulated_threshold,
        )
        analysis = aggregate(face_analyses, frame_analyses, weights)
        signal_stats = compute_signal_stats(analysis)
        timings.ensemble = time.perf_counter() - t0

        # ── Stage 9: Physics reasoning (Tier 2) ──
        if not skip_explainability and analysis.verdict != Verdict.LIKELY_AUTHENTIC:
            t0 = time.perf_counter()
            physics_result = self._run_physics_reasoning(frames, analysis)
            timings.physics_reasoning = time.perf_counter() - t0

            if physics_result and physics_result.num_frames_analyzed > 0:
                combined = 0.60 * analysis.video_score + 0.40 * physics_result.aggregate_score
                if combined >= self.config.ensemble.manipulated_threshold:
                    analysis.verdict = Verdict.LIKELY_MANIPULATED
                elif combined < self.config.ensemble.suspicious_threshold:
                    analysis.verdict = Verdict.LIKELY_AUTHENTIC
                analysis.video_score = combined
                signal_stats["physics_score"] = round(physics_result.aggregate_score, 4)
                signal_stats["physics_confidence"] = round(physics_result.confidence, 4)

        # ── Stage 10: GradCAM on flagged frames ──
        heatmap_paths = []
        forensic_paths = []
        timeline_path = None
        if not skip_explainability:
            t0 = time.perf_counter()
            heatmap_paths = self._generate_heatmaps(frames, analysis, output_dir)
            timings.gradcam = time.perf_counter() - t0

            # ── Stage 10: Forensic views + timeline ──
            t0 = time.perf_counter()
            forensic_paths = self._generate_forensic_views(
                frames, analysis, heatmap_paths, output_dir,
            )
            timeline_path = self._generate_timeline(analysis, output_dir)
            timings.forensic_views = time.perf_counter() - t0

        timings.total = time.perf_counter() - total_t0

        report = ForensicReport(
            video_metadata=meta.to_dict(),
            analysis=analysis,
            signal_stats=signal_stats,
            heatmap_paths=[str(p) for p in heatmap_paths],
            forensic_view_paths=[str(p) for p in forensic_paths],
            timeline_path=str(timeline_path) if timeline_path else "",
            timings=timings,
            output_dir=str(output_dir),
            low_sharpness_fallback=low_sharpness_fallback,
        )

        logger.info(
            "pipeline_complete",
            verdict=analysis.verdict.value,
            score=f"{analysis.video_score:.3f}",
            manipulation=analysis.manipulation_type.value,
            dominant_path=analysis.dominant_path,
            total_time=f"{timings.total:.1f}s",
        )

        return report

    # ── Internal pipeline stages ──

    _BATCH_CHUNK = 32

    def _run_face_path(
        self,
        frames: list,  # list[SelectedFrame]
        face_results: list,  # list[FrameFaces]
    ) -> list:  # list[FaceFrameAnalysis]
        """Path A: run detectors on face crops. Skips tiny crops."""
        from verifi.ensemble.aggregator import SignalScore

        min_crop_pixels = 80

        face_entries = []
        for sf, fr in zip(frames, face_results):
            for face in fr.faces:
                raw_h, raw_w = face.crop_raw.shape[:2]
                is_small = raw_h < min_crop_pixels or raw_w < min_crop_pixels
                face_entries.append((sf, face, is_small, raw_w, raw_h))

        if not face_entries:
            logger.info("face_path_complete", num_analyses=0)
            return []

        all_crops = [e[1].crop for e in face_entries]
        clip_scores = self._batched_predict(self._clip, all_crops)

        effnet_crops = [e[1].crop for e in face_entries if not e[2]]
        effnet_scores = self._batched_predict(self._effnet, effnet_crops) if effnet_crops else []

        analyses = []
        effnet_idx = 0
        for i, (sf, face, is_small, raw_w, raw_h) in enumerate(face_entries):
            signals = []

            if clip_scores[i] is not None:
                signals.append(SignalScore(
                    name="clip", score=clip_scores[i].score,
                    metadata=clip_scores[i].metadata,
                ))

            if not is_small:
                if effnet_idx < len(effnet_scores) and effnet_scores[effnet_idx] is not None:
                    signals.append(SignalScore(
                        name="effnet", score=effnet_scores[effnet_idx].score,
                    ))
                effnet_idx += 1
            else:
                signals.append(SignalScore(
                    name="effnet", score=0.5,
                    metadata={"skipped": True, "reason": "face_too_small",
                            "raw_size": f"{raw_w}x{raw_h}"},
                ))

            face_gray = cv2.cvtColor(face.crop, cv2.COLOR_BGR2GRAY)
            face_sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
            dct_result = self._freq.analyze(face.crop, sharpness=face_sharpness)
            signals.append(SignalScore(
                name="dct", score=dct_result.score,
                metadata=dct_result.metadata,
            ))
            signals.append(SignalScore(
                name="channel_corr",
                score=dct_result.metadata.get("channel_corr_score", 0.5),
            ))

            nr_result = self._noise_residual.analyze(face.crop)
            signals.append(SignalScore(
                name="noise_residual", score=nr_result.score,
                metadata=nr_result.metadata,
            ))

            analyses.append(FaceFrameAnalysis(
                face_id=face.face_id,
                frame_idx=sf.frame_idx,
                timestamp_sec=sf.timestamp_sec,
                signals=signals,
            ))

        logger.info("face_path_complete", num_analyses=len(analyses))
        return analyses


    def _run_frame_path(
        self,
        frames: list[SelectedFrame],
    ) -> list[FullFrameAnalysis]:
        """Path B: run detectors on full frames (no face crop)."""
        clip_size = self.config.detector.clip_input_size
        resized = [
            cv2.resize(sf.image, (clip_size, clip_size)) for sf in frames
        ]

        clip_scores = self._batched_predict(self._clip, resized)

        analyses = []
        for i, sf in enumerate(frames):
            signals = []

            if clip_scores[i] is not None:
                signals.append(SignalScore(
                    name="clip", score=clip_scores[i].score,
                ))

            frame_gray = cv2.cvtColor(sf.image, cv2.COLOR_BGR2GRAY)
            frame_sharpness = float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
            dct_result = self._freq.analyze(sf.image, sharpness=frame_sharpness)
            signals.append(SignalScore(
                name="dct", score=dct_result.score,
                metadata=dct_result.metadata,
            ))
            signals.append(SignalScore(
                name="channel_corr",
                score=dct_result.metadata.get("channel_corr_score", 0.5),
            ))

            nr_result = self._noise_residual.analyze(sf.image)
            signals.append(SignalScore(
                name="noise_residual", score=nr_result.score,
                metadata=nr_result.metadata,
            ))

            analyses.append(FullFrameAnalysis(
                frame_idx=sf.frame_idx,
                timestamp_sec=sf.timestamp_sec,
                signals=signals,
            ))

        logger.info("frame_path_complete", num_analyses=len(analyses))
        return analyses

    def _batched_predict(self, detector, images: list[np.ndarray]):
        """Run a detector on images in chunks of _BATCH_CHUNK, return flat results list."""
        results = []
        for start in range(0, len(images), self._BATCH_CHUNK):
            chunk = images[start : start + self._BATCH_CHUNK]
            results.extend(detector.predict(chunk))
        return results

    def _run_temporal_analysis(
        self,
        frames: list[SelectedFrame],
        frame_analyses: list[FullFrameAnalysis],
    ) -> None:
        """Run temporal consistency on adjacent frame pairs."""
        pairs_analyzed = 0
        for i in range(len(frames) - 1):
            fa, fb = frames[i], frames[i + 1]
            # Only analyze closely adjacent frames
            if abs(fa.frame_idx - fb.frame_idx) > 10:
                continue

            result = self._temporal.analyze_pair(fa.image, fb.image)
            # Add temporal score to the second frame's analysis
            if i + 1 < len(frame_analyses):
                frame_analyses[i + 1].signals.append(SignalScore(
                    name="temporal", score=result.score,
                    metadata=result.metadata,
                ))
            pairs_analyzed += 1

        logger.info("temporal_analysis_complete", pairs=pairs_analyzed)

    def _run_physics_reasoning(
        self,
        frames: list[SelectedFrame],
        analysis: VideoAnalysis,
    ):
        """Run physics-based reasoning on selected frames."""
        tier1_context = {
            "video_score": round(analysis.video_score, 4),
            "verdict": analysis.verdict.value,
            "dominant_path": analysis.dominant_path,
            "frame_scores": {
                str(fa.frame_idx): round(fa.ensemble_score, 4)
                for fa in analysis.frame_analyses
            },
        }

        frame_images = [f.image for f in frames]
        frame_indices = [f.frame_idx for f in frames]
        timestamps = [f.timestamp_sec for f in frames]

        try:
            result = self._physics.analyze(
                frame_images, frame_indices, timestamps, tier1_context
            )
            logger.info(
                "physics_reasoning_complete",
                score=f"{result.aggregate_score:.3f}",
                confidence=f"{result.confidence:.3f}",
                verdict=result.verdict_suggestion,
                frames=result.num_frames_analyzed,
            )
            return result
        except Exception as e:
            logger.error("physics_reasoning_failed", error=str(e))
            return None

    def _generate_heatmaps(
        self,
        frames: list[SelectedFrame],
        analysis: VideoAnalysis,
        output_dir: Path,
    ) -> list[Path]:
        """Generate GradCAM heatmaps on flagged frames."""
        heatmap_dir = output_dir / "heatmaps"
        paths = []

        # Determine which frames to explain based on dominant path
        if analysis.dominant_path == "frame":
            # Use top flagged full frames
            flagged = sorted(
                [(i, a) for i, a in enumerate(analysis.frame_analyses) if a.flagged],
                key=lambda x: x[1].ensemble_score,
                reverse=True,
            )[:5]

            images = []
            for _, fa in flagged:
                matching = [f for f in frames if f.frame_idx == fa.frame_idx]
                if matching:
                    images.append((matching[0].image, fa.frame_idx))

        else:
            # Use top flagged face frames
            flagged = sorted(
                [(i, a) for i, a in enumerate(analysis.face_analyses) if a.flagged],
                key=lambda x: x[1].ensemble_score,
                reverse=True,
            )[:5]

            images = []
            for _, fa in flagged:
                matching = [f for f in frames if f.frame_idx == fa.frame_idx]
                if matching:
                    images.append((matching[0].image, fa.frame_idx))

        if not images:
            logger.info("no_frames_to_explain")
            return paths

        # Generate CLIP GradCAM — use visual encoder only
        clip_layer = self._clip.get_gradcam_target_layer()
        if clip_layer is not None:
            # For zero-shot CLIP, we need the visual encoder
            # not the full model (which expects text input too)
            visual_model = self._clip.get_visual_model()

            # Wrap visual model to output a score from embeddings
            class VisualScoreWrapper(torch.nn.Module):
                def __init__(self, visual, fake_text_feat):
                    super().__init__()
                    self.visual = visual
                    self.fake_text_feat = fake_text_feat

                def forward(self, x):
                    features = self.visual(x)
                    features = features / features.norm(dim=-1, keepdim=True)
                    # Similarity to "fake" text embedding
                    sim = (features @ self.fake_text_feat.T) * 100.0
                    return sim

            wrapper = VisualScoreWrapper(
                visual_model,
                self._clip._fake_features.cpu()
            ).cpu()
            wrapper.eval()

            # Target layer is relative to visual model
            clip_target = wrapper.visual.transformer.resblocks[-1].ln_2

            clip_heatmaps = self._gradcam.generate_batch(
                model=wrapper,
                target_layer=clip_target,
                images=images,
                preprocess_fn=self._clip._transform,
                model_name="clip",
                is_vit=True,
                input_size=self.config.detector.clip_input_size,
                max_heatmaps=5,
            )
            for hm in clip_heatmaps:
                p = save_heatmap(hm, heatmap_dir)
                paths.append(p)

            # Move CLIP back to device
            self._clip._model.to(self.device)

        # Generate EfficientNet GradCAM
        effnet_layer = self._effnet.get_gradcam_target_layer()
        if effnet_layer is not None:
            # EfficientNet needs its own preprocessing
            effnet_images = []
            for img, idx in images[:3]:  # fewer for effnet (secondary signal)
                resized = cv2.resize(
                    img,
                    (self.config.detector.effnet_input_size,
                      self.config.detector.effnet_input_size),
                )
                effnet_images.append((resized, idx))

            effnet_heatmaps = self._gradcam.generate_batch(
                model=self._effnet._model,
                target_layer=effnet_layer,
                images=effnet_images,
                preprocess_fn=self._effnet._transform,
                model_name="effnet",
                is_vit=False,
                max_heatmaps=3,
            )
            for hm in effnet_heatmaps:
                p = save_heatmap(hm, heatmap_dir)
                paths.append(p)

        return paths

    def _generate_forensic_views(
        self,
        frames: list[SelectedFrame],
        analysis: VideoAnalysis,
        heatmap_paths: list[Path],
        output_dir: Path,
    ) -> list[Path]:
        """Create three-panel forensic views for flagged frames."""
        from verifi.explainability.gradcam import HeatmapResult

        forensic_dir = output_dir / "forensic"
        paths = []

        # Match heatmaps to frames by parsing filename
        heatmap_map: dict[int, Path] = {}
        for p in heatmap_paths:
            # filename format: heatmap_clip_frame0042.png
            name = p.stem
            parts = name.split("frame")
            if len(parts) == 2:
                try:
                    idx = int(parts[1])
                    heatmap_map[idx] = p
                except ValueError:
                    pass

        # Get flagged frames
        if analysis.dominant_path == "frame":
            flagged_indices = [
                a.frame_idx for a in analysis.frame_analyses if a.flagged
            ]
        else:
            flagged_indices = [
                a.frame_idx for a in analysis.face_analyses if a.flagged
            ]

        # Sort by score, take top 5
        flagged_indices = list(set(flagged_indices))[:5]

        for frame_idx in flagged_indices:
            matching = [f for f in frames if f.frame_idx == frame_idx]
            if not matching:
                continue
            frame = matching[0]

            # Load heatmap if available
            hm_result = None
            if frame_idx in heatmap_map:
                hm_img = cv2.imread(str(heatmap_map[frame_idx]))
                if hm_img is not None:
                    hm_result = HeatmapResult(
                        overlay=hm_img,
                        raw_cam=np.zeros((1, 1)),
                        source_image=frame.image,
                        frame_idx=frame_idx,
                        model_name="clip",
                    )

            forensic = create_forensic_view(
                original=frame.image,
                heatmap_result=hm_result,
                frame_idx=frame_idx,
                freq_analyzer=self._freq,
            )
            p = save_forensic_view(forensic, forensic_dir, frame_idx)
            paths.append(p)

        return paths

    def _generate_timeline(
        self,
        analysis: VideoAnalysis,
        output_dir: Path,
    ) -> Path | None:
        """Generate confidence timeline visualization."""
        # Use frame-path scores (always available)
        scores = [
            (a.timestamp_sec, a.ensemble_score)
            for a in analysis.frame_analyses
        ]
        if not scores:
            return None

        timeline = create_confidence_timeline(
            scores,
            suspicious_threshold=self.config.ensemble.suspicious_threshold,
            manipulated_threshold=self.config.ensemble.manipulated_threshold,
        )
        return save_timeline(timeline, output_dir)


