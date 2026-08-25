"""Pydantic models for API request/response schemas."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze when submitting a path or URL."""

    video_path: str | None = Field(None, description="Local path to video file")
    url: str | None = Field(None, description="URL to download video from (YouTube, etc.)")
    skip_explainability: bool = Field(
        False, description="Skip GradCAM heatmaps and forensic views for faster analysis"
    )


class Verdict(StrEnum):
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"


class ManipulationType(StrEnum):
    NONE = "none"
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FULL_SYNTHESIS = "full_synthesis"
    UNKNOWN = "unknown"


class SignalDetail(BaseModel):
    name: str
    mean_score: float
    max_score: float


class AnalysisSummary(BaseModel):
    """Core analysis result returned from POST /analyze."""

    report_id: str
    video: str
    duration_sec: float
    verdict: Verdict
    video_score: float
    confidence: float
    manipulation_type: ManipulationType
    dominant_path: str
    face_path_score: float
    frame_path_score: float
    face_path_active: bool
    frame_path_active: bool
    signal_stats: dict[str, float] = Field(default_factory=dict)
    processing_time_sec: float = 0.0


class TimingDetail(BaseModel):
    validation: float = 0.0
    scene_detection: float = 0.0
    frame_selection: float = 0.0
    face_detection: float = 0.0
    face_path_inference: float = 0.0
    frame_path_inference: float = 0.0
    temporal_analysis: float = 0.0
    ensemble: float = 0.0
    gradcam: float = 0.0
    forensic_views: float = 0.0
    total: float = 0.0


class FrameSignal(BaseModel):
    name: str
    score: float
    metadata: dict = Field(default_factory=dict)


class FrameAnalysisDetail(BaseModel):
    frame_idx: int
    timestamp_sec: float
    signals: list[FrameSignal]
    ensemble_score: float
    flagged: bool


class FaceAnalysisDetail(BaseModel):
    face_id: int
    frame_idx: int
    timestamp_sec: float
    signals: list[FrameSignal]
    ensemble_score: float
    flagged: bool


class FullReport(BaseModel):
    """Complete forensic report with all details."""

    report_id: str
    video_metadata: dict
    verdict: Verdict
    video_score: float
    confidence: float
    manipulation_type: ManipulationType
    dominant_path: str
    face_path_score: float
    face_path_active: bool
    frame_path_score: float
    frame_path_active: bool
    frame_analyses: list[FrameAnalysisDetail]
    face_analyses: list[FaceAnalysisDetail]
    signal_stats: dict[str, float]
    heatmap_paths: list[str]
    forensic_view_paths: list[str]
    timeline_path: str
    timings: TimingDetail
    output_dir: str


class ReportListItem(BaseModel):
    report_id: str
    video: str
    verdict: Verdict
    video_score: float
    created_at: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
