"""
Dual-path ensemble aggregator: face-level + full-frame analysis.

Save to: src/verifi/ensemble/aggregator.py

Path A (face-level): CLIP + EfficientNet + DCT on face crops
Path B (frame-level): CLIP + DCT on full frames (no face required)

Both paths run in parallel. The stronger signal determines the verdict.
This handles face-swap deepfakes AND fully synthetic video (Sora, Veo, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import structlog

logger = structlog.get_logger()


class ManipulationType(StrEnum):
    NONE = "none"
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FULL_SYNTHESIS = "full_synthesis"
    UNKNOWN = "unknown"


class Verdict(StrEnum):
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"


@dataclass
class SignalScore:
    """A single detection signal score for one frame."""
    name: str           # "clip", "effnet", "dct", "channel_corr", "temporal"
    score: float        # 0.0 (authentic) → 1.0 (manipulated)
    metadata: dict = field(default_factory=dict)


@dataclass
class FaceFrameAnalysis:
    """Face-level analysis for one face in one frame."""
    face_id: int
    frame_idx: int
    timestamp_sec: float
    signals: list[SignalScore] = field(default_factory=list)
    ensemble_score: float = 0.0
    flagged: bool = False


@dataclass
class FullFrameAnalysis:
    """Frame-level analysis (no face crop — entire frame)."""
    frame_idx: int
    timestamp_sec: float
    signals: list[SignalScore] = field(default_factory=list)
    ensemble_score: float = 0.0
    flagged: bool = False


@dataclass
class VideoAnalysis:
    """Complete dual-path analysis for a video."""
    # Path A: face-level results
    face_analyses: list[FaceFrameAnalysis] = field(default_factory=list)
    face_path_score: float = 0.0
    face_path_active: bool = False

    # Path B: frame-level results
    frame_analyses: list[FullFrameAnalysis] = field(default_factory=list)
    frame_path_score: float = 0.0
    frame_path_active: bool = True  # always runs

    # Combined verdict
    video_score: float = 0.0
    confidence: float = 0.0
    verdict: Verdict = Verdict.LIKELY_AUTHENTIC
    manipulation_type: ManipulationType = ManipulationType.NONE
    dominant_path: str = "frame"  # "face" or "frame"

    # Flagged items for GradCAM / explainer
    flagged_face_indices: list[int] = field(default_factory=list)
    flagged_frame_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_score": round(self.video_score, 4),
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict.value,
            "manipulation_type": self.manipulation_type.value,
            "dominant_path": self.dominant_path,
            "face_path": {
                "active": self.face_path_active,
                "score": round(self.face_path_score, 4),
                "num_analyses": len(self.face_analyses),
                "num_flagged": len(self.flagged_face_indices),
            },
            "frame_path": {
                "active": self.frame_path_active,
                "score": round(self.frame_path_score, 4),
                "num_analyses": len(self.frame_analyses),
                "num_flagged": len(self.flagged_frame_indices),
            },
        }


@dataclass
class EnsembleWeights:
    """Configurable weights for each signal in each path."""
    # Face-level path weights (sum = 1.0)
    face_clip: float = 0.20
    face_effnet: float = 0.15
    face_dct: float = 0.30
    face_noise_residual: float = 0.20
    face_channel_corr: float = 0.05
    face_temporal: float = 0.10

    # Frame-level path weights (sum = 1.0)
    frame_clip: float = 0.15
    frame_dct: float = 0.30
    frame_noise_residual: float = 0.25
    frame_channel_corr: float = 0.05
    frame_temporal: float = 0.25

    # Verdict thresholds — wide SUSPICIOUS band to avoid overconfident wrong answers
    suspicious_threshold: float = 0.35
    manipulated_threshold: float = 0.70

    # Signal agreement bonus/penalty
    agreement_bonus: float = 0.05
    disagreement_penalty: float = 0.03

    # Use mean of all frames — robust to per-frame noise and edge frames
    top_k_ratio: float = 1.0
    min_top_k: int = 3


def _weighted_score(
    signals: list[SignalScore], weight_map: dict[str, float],
) -> float:
    total_w = 0.0
    weighted_sum = 0.0
    for sig in signals:
        w = weight_map.get(sig.name, 0.0)
        weighted_sum += w * sig.score
        total_w += w
    return weighted_sum / total_w if total_w > 0 else 0.0


def _apply_agreement_bonus(
    signals: list[SignalScore],
    base_score: float,
    weights: EnsembleWeights,
) -> float:
    """Reward forensic signal agreement, penalize disagreement."""
    dct_score = None
    nr_score = None
    for s in signals:
        if s.name == "dct":
            dct_score = s.score
        elif s.name == "noise_residual":
            nr_score = s.score

    if dct_score is None or nr_score is None:
        return base_score

    both_high = dct_score > 0.55 and nr_score > 0.55
    disagree = (dct_score > 0.55 and nr_score < 0.40) or (
        nr_score > 0.55 and dct_score < 0.40
    )

    if both_high:
        return float(np.clip(base_score + weights.agreement_bonus, 0.0, 1.0))
    if disagree:
        return float(np.clip(base_score - weights.disagreement_penalty, 0.0, 1.0))
    return base_score


def _compute_confidence(score: float, weights: EnsembleWeights) -> float:
    """Distance from nearest threshold, normalized to 0.0-1.0."""
    lo = weights.suspicious_threshold
    hi = weights.manipulated_threshold

    if score < lo:
        dist = lo - score
        max_dist = lo
    elif score >= hi:
        dist = score - hi
        max_dist = 1.0 - hi
    else:
        dist_lo = score - lo
        dist_hi = hi - score
        dist = min(dist_lo, dist_hi)
        half_band = (hi - lo) / 2.0
        return float(np.clip(1.0 - dist / half_band, 0.0, 1.0)) * 0.5

    return float(np.clip(dist / max_dist, 0.0, 1.0)) if max_dist > 0 else 0.0


def aggregate_face_signals(
    analyses: list[FaceFrameAnalysis],
    weights: EnsembleWeights,
) -> float:
    """
    Compute face-path video score from per-face-per-frame signals.
    Uses weighted average per face, then top-K across frames.
    """
    if not analyses:
        return 0.0

    weight_map = {
        "clip": weights.face_clip,
        "effnet": weights.face_effnet,
        "dct": weights.face_dct,
        "noise_residual": weights.face_noise_residual,
        "channel_corr": weights.face_channel_corr,
        "temporal": weights.face_temporal,
    }

    for a in analyses:
        a.ensemble_score = _weighted_score(a.signals, weight_map)
        a.ensemble_score = _apply_agreement_bonus(a.signals, a.ensemble_score, weights)
        a.flagged = a.ensemble_score > weights.suspicious_threshold

    scores = sorted([a.ensemble_score for a in analyses], reverse=True)
    k = max(weights.min_top_k, int(len(scores) * weights.top_k_ratio))
    return float(np.mean(scores[:k]))


def aggregate_frame_signals(
    analyses: list[FullFrameAnalysis],
    weights: EnsembleWeights,
) -> float:
    """
    Compute frame-path video score from per-frame signals.
    """
    if not analyses:
        return 0.0

    weight_map = {
        "clip": weights.frame_clip,
        "dct": weights.frame_dct,
        "noise_residual": weights.frame_noise_residual,
        "channel_corr": weights.frame_channel_corr,
        "temporal": weights.frame_temporal,
    }

    for a in analyses:
        a.ensemble_score = _weighted_score(a.signals, weight_map)
        a.ensemble_score = _apply_agreement_bonus(a.signals, a.ensemble_score, weights)
        a.flagged = a.ensemble_score > weights.suspicious_threshold

    scores = sorted([a.ensemble_score for a in analyses], reverse=True)
    k = max(weights.min_top_k, int(len(scores) * weights.top_k_ratio))
    return float(np.mean(scores[:k]))


def infer_manipulation_type(
    face_score: float,
    frame_score: float,
    face_analyses: list[FaceFrameAnalysis],
    frame_analyses: list[FullFrameAnalysis],
) -> ManipulationType:
    """
    Heuristic to determine the type of manipulation.

    Uses forensic signals (noise_residual, DCT) for full-synthesis detection,
    and face-level signals (effnet, temporal) for face manipulation.
    """
    if face_score < 0.25 and frame_score < 0.25:
        return ManipulationType.NONE

    # Check frame-path forensic signals for full synthesis
    if frame_score > 0.35 and frame_analyses:
        nr_scores = [
            s.score for a in frame_analyses
            for s in a.signals if s.name == "noise_residual"
        ]
        dct_scores = [
            s.score for a in frame_analyses
            for s in a.signals if s.name == "dct"
        ]
        avg_nr = float(np.mean(nr_scores)) if nr_scores else 0
        avg_dct = float(np.mean(dct_scores)) if dct_scores else 0
        if avg_nr > 0.55 and avg_dct > 0.45:
            return ManipulationType.FULL_SYNTHESIS

    if frame_score > face_score + 0.10:
        return ManipulationType.FULL_SYNTHESIS

    # Face-dominant signal → face swap or reenactment
    if face_score > 0.4 and face_analyses:
        flagged = [a for a in face_analyses if a.flagged]
        if not flagged:
            return ManipulationType.UNKNOWN

        effnet_scores = []
        temporal_scores = []
        for a in flagged:
            for sig in a.signals:
                if sig.name == "effnet":
                    effnet_scores.append(sig.score)
                if sig.name == "temporal":
                    temporal_scores.append(sig.score)

        avg_effnet = float(np.mean(effnet_scores)) if effnet_scores else 0
        avg_temporal = float(np.mean(temporal_scores)) if temporal_scores else 0

        if avg_effnet > 0.6:
            return ManipulationType.FACE_SWAP
        if avg_temporal > 0.5:
            return ManipulationType.FACE_REENACTMENT

    return ManipulationType.UNKNOWN


def aggregate(
    face_analyses: list[FaceFrameAnalysis],
    frame_analyses: list[FullFrameAnalysis],
    weights: EnsembleWeights | None = None,
) -> VideoAnalysis:
    """
    Main aggregation function: combine both paths into a single verdict.

    Args:
        face_analyses: Per-face-per-frame results from Path A.
        frame_analyses: Per-frame results from Path B.
        weights: Ensemble configuration. Uses defaults if None.

    Returns:
        VideoAnalysis with combined verdict and flagged items.
    """
    if weights is None:
        weights = EnsembleWeights()

    # Aggregate each path independently
    face_score = aggregate_face_signals(face_analyses, weights)
    frame_score = aggregate_frame_signals(frame_analyses, weights)

    face_active = len(face_analyses) > 0

    # Combined score: determine dominant path
    # When frame-path has overwhelming consensus (>70% flagged) and exceeds
    # suspicious threshold, it wins regardless — this catches fully synthetic
    # content where face-path scores are noise from small background faces.
    frame_flagged_ratio = (
        sum(1 for a in frame_analyses if a.flagged) / len(frame_analyses)
        if frame_analyses else 0.0
    )
    frame_consensus = (
        frame_flagged_ratio > 0.70
        and frame_score >= weights.suspicious_threshold
    )

    if frame_consensus:
        video_score = frame_score
        dominant_path = "frame"
    elif face_active and face_score > frame_score:
        video_score = face_score
        dominant_path = "face"
    else:
        video_score = frame_score
        dominant_path = "frame"

    # Verdict
    if video_score >= weights.manipulated_threshold:
        verdict = Verdict.LIKELY_MANIPULATED
    elif video_score >= weights.suspicious_threshold:
        verdict = Verdict.SUSPICIOUS
    else:
        verdict = Verdict.LIKELY_AUTHENTIC

    confidence = _compute_confidence(video_score, weights)

    # Manipulation type
    manipulation_type = infer_manipulation_type(
        face_score, frame_score, face_analyses, frame_analyses,
    )

    # Collect flagged indices (for GradCAM targeting)
    flagged_face = [i for i, a in enumerate(face_analyses) if a.flagged]
    flagged_frame = [i for i, a in enumerate(frame_analyses) if a.flagged]

    result = VideoAnalysis(
        face_analyses=face_analyses,
        face_path_score=face_score,
        face_path_active=face_active,
        frame_analyses=frame_analyses,
        frame_path_score=frame_score,
        frame_path_active=True,
        video_score=video_score,
        confidence=confidence,
        verdict=verdict,
        manipulation_type=manipulation_type,
        dominant_path=dominant_path,
        flagged_face_indices=flagged_face,
        flagged_frame_indices=flagged_frame,
    )

    logger.info(
        "ensemble_aggregated",
        face_score=f"{face_score:.3f}",
        frame_score=f"{frame_score:.3f}",
        video_score=f"{video_score:.3f}",
        verdict=verdict.value,
        manipulation_type=manipulation_type.value,
        dominant_path=dominant_path,
        face_flagged=len(flagged_face),
        frame_flagged=len(flagged_frame),
    )

    return result


def compute_signal_stats(analysis: VideoAnalysis) -> dict:
    """
    Compute aggregate statistics across all signals for the LLM explainer.
    Returns a flat dict suitable for prompt template formatting.
    """
    stats = {}

    # Face path stats
    if analysis.face_analyses:
        clip_s = [s.score for a in analysis.face_analyses for s in a.signals if s.name == "clip"]
        effnet_s = [
            s.score for a in analysis.face_analyses
            for s in a.signals if s.name == "effnet"
        ]
        stats["clip_mean"] = float(np.mean(clip_s)) if clip_s else 0
        stats["clip_max"] = float(np.max(clip_s)) if clip_s else 0
        stats["clip_std"] = float(np.std(clip_s)) if clip_s else 0
        stats["effnet_mean"] = float(np.mean(effnet_s)) if effnet_s else 0
        stats["effnet_max"] = float(np.max(effnet_s)) if effnet_s else 0
        stats["effnet_std"] = float(np.std(effnet_s)) if effnet_s else 0
    else:
        stats.update({
            "clip_mean": 0, "clip_max": 0, "clip_std": 0,
            "effnet_mean": 0, "effnet_max": 0, "effnet_std": 0,
        })

    # Frame path stats
    dct_s = [s.score for a in analysis.frame_analyses for s in a.signals if s.name == "dct"]
    frame_clip_s = [s.score for a in analysis.frame_analyses for s in a.signals if s.name == "clip"]
    temporal_s = [
        s.score for a in analysis.frame_analyses
        for s in a.signals if s.name == "temporal"
    ]

    stats["freq_score"] = float(np.mean(dct_s)) if dct_s else 0
    dct_meta = [s.metadata for a in analysis.frame_analyses for s in a.signals if s.name == "dct"]
    hf_vals = [m.get("hf_suppression_pct", 0) for m in dct_meta if m]
    stats["hf_suppression"] = float(np.mean(hf_vals)) if hf_vals else 0

    corr_s = [
        s.score for a in analysis.frame_analyses
        for s in a.signals if s.name == "channel_corr"
    ]
    stats["channel_corr_mean"] = float(np.mean(corr_s)) if corr_s else 0
    stats["channel_corr_max"] = float(np.max(corr_s)) if corr_s else 0

    stats["frame_clip_mean"] = float(np.mean(frame_clip_s)) if frame_clip_s else 0
    stats["frame_clip_max"] = float(np.max(frame_clip_s)) if frame_clip_s else 0

    if temporal_s:
        anomalies = [s for s in temporal_s if s > 0.5]
        stats["temporal_summary"] = (
            f"{len(anomalies)} anomalous pairs out of {len(temporal_s)} analyzed"
        )
    else:
        stats["temporal_summary"] = "Not analyzed"

    # Noise residual stats
    nr_s = [
        s.score for a in analysis.frame_analyses
        for s in a.signals if s.name == "noise_residual"
    ]
    stats["noise_residual_mean"] = float(np.mean(nr_s)) if nr_s else 0
    stats["noise_residual_max"] = float(np.max(nr_s)) if nr_s else 0

    stats["av_sync_summary"] = "Not yet implemented"

    return stats
