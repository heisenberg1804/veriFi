"""Analysis API routes: submit videos, retrieve reports."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, UploadFile

from verifi.api.schemas import (
    AnalysisSummary,
    AnalyzeRequest,
    FaceAnalysisDetail,
    FrameAnalysisDetail,
    FrameSignal,
    FullReport,
    ReportListItem,
    TimingDetail,
)

logger = structlog.get_logger()
router = APIRouter(tags=["analysis"])

REPORTS_DIR = Path("data/reports")


def _report_id_from_path(output_dir: str) -> str:
    return Path(output_dir).name


def _build_summary(report, report_id: str) -> AnalysisSummary:
    analysis = report.analysis
    return AnalysisSummary(
        report_id=report_id,
        video=report.video_metadata.get("filename", "unknown"),
        duration_sec=report.video_metadata.get("duration_sec", 0.0),
        verdict=analysis.verdict.value,
        video_score=round(analysis.video_score, 4),
        confidence=round(analysis.confidence, 4),
        manipulation_type=analysis.manipulation_type.value,
        dominant_path=analysis.dominant_path,
        face_path_score=round(analysis.face_path_score, 4),
        frame_path_score=round(analysis.frame_path_score, 4),
        face_path_active=analysis.face_path_active,
        frame_path_active=analysis.frame_path_active,
        signal_stats={k: round(v, 4) for k, v in report.signal_stats.items()},
        processing_time_sec=round(report.timings.total, 3),
    )


def _build_full_report(report, report_id: str) -> FullReport:
    analysis = report.analysis

    frame_details = [
        FrameAnalysisDetail(
            frame_idx=fa.frame_idx,
            timestamp_sec=round(fa.timestamp_sec, 3),
            signals=[
                FrameSignal(name=s.name, score=round(s.score, 4), metadata=s.metadata)
                for s in fa.signals
            ],
            ensemble_score=round(fa.ensemble_score, 4),
            flagged=fa.flagged,
        )
        for fa in analysis.frame_analyses
    ]

    face_details = [
        FaceAnalysisDetail(
            face_id=fa.face_id,
            frame_idx=fa.frame_idx,
            timestamp_sec=round(fa.timestamp_sec, 3),
            signals=[
                FrameSignal(name=s.name, score=round(s.score, 4), metadata=s.metadata)
                for s in fa.signals
            ],
            ensemble_score=round(fa.ensemble_score, 4),
            flagged=fa.flagged,
        )
        for fa in analysis.face_analyses
    ]

    return FullReport(
        report_id=report_id,
        video_metadata=report.video_metadata,
        verdict=analysis.verdict.value,
        video_score=round(analysis.video_score, 4),
        confidence=round(analysis.confidence, 4),
        manipulation_type=analysis.manipulation_type.value,
        dominant_path=analysis.dominant_path,
        face_path_score=round(analysis.face_path_score, 4),
        face_path_active=analysis.face_path_active,
        frame_path_score=round(analysis.frame_path_score, 4),
        frame_path_active=analysis.frame_path_active,
        frame_analyses=frame_details,
        face_analyses=face_details,
        signal_stats={k: round(v, 4) for k, v in report.signal_stats.items()},
        heatmap_paths=report.heatmap_paths,
        forensic_view_paths=report.forensic_view_paths,
        timeline_path=report.timeline_path,
        timings=TimingDetail(**report.timings.to_dict()),
        output_dir=report.output_dir,
    )


@router.post("/analyze", response_model=AnalysisSummary)
async def analyze_video(request: AnalyzeRequest):
    """Analyze a video for AI-generated content.

    Provide either a local video_path or a URL to download.
    Returns a summary with verdict, scores, and a report_id for full details.
    """
    from verifi.api.app import get_pipeline

    pipeline = get_pipeline()
    video_path: str | None = None
    tmp_dir: str | None = None

    try:
        if request.url:
            from verifi.ingestion.downloader import download_video

            tmp_dir = tempfile.mkdtemp(prefix="verifi_")
            downloaded = download_video(request.url, output_dir=tmp_dir)
            video_path = str(downloaded)
        elif request.video_path:
            video_path = request.video_path
            if not Path(video_path).exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        else:
            raise HTTPException(
                status_code=400, detail="Provide either 'video_path' or 'url'"
            )

        report = pipeline.analyze(
            video_path,
            skip_explainability=request.skip_explainability,
        )
        report_id = _report_id_from_path(report.output_dir)

        _save_report_metadata(report, report_id)

        return _build_summary(report, report_id)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("analyze_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/analyze/upload", response_model=AnalysisSummary)
async def analyze_upload(file: UploadFile, skip_explainability: bool = False):
    """Analyze an uploaded video file."""
    from verifi.api.app import get_pipeline

    pipeline = get_pipeline()
    tmp_dir = tempfile.mkdtemp(prefix="verifi_")

    try:
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        report = pipeline.analyze(
            str(tmp_path),
            skip_explainability=skip_explainability,
        )
        report_id = _report_id_from_path(report.output_dir)

        _save_report_metadata(report, report_id)

        return _build_summary(report, report_id)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("analyze_upload_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/report/{report_id}", response_model=FullReport)
async def get_report(report_id: str):
    """Retrieve a complete forensic report by ID."""
    report_dir = REPORTS_DIR / report_id
    meta_path = report_dir / "report.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")

    with open(meta_path) as f:
        data = json.load(f)

    return FullReport(**data)


@router.get("/reports", response_model=list[ReportListItem])
async def list_reports(limit: int = 20):
    """List available reports, most recent first."""
    if not REPORTS_DIR.exists():
        return []

    items = []
    for report_dir in sorted(REPORTS_DIR.iterdir(), reverse=True):
        if not report_dir.is_dir():
            continue
        meta_path = report_dir / "report.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            data = json.load(f)

        items.append(ReportListItem(
            report_id=data.get("report_id", report_dir.name),
            video=data.get("video_metadata", {}).get("filename", "unknown"),
            verdict=data.get("verdict", "SUSPICIOUS"),
            video_score=data.get("video_score", 0.0),
            created_at=data.get("video_metadata", {}).get("analyzed_at", ""),
        ))

        if len(items) >= limit:
            break

    return items


def _save_report_metadata(report, report_id: str) -> None:
    """Persist the full report as JSON for later retrieval."""
    report_dir = Path(report.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    full = _build_full_report(report, report_id)
    data = full.model_dump(mode="json")
    data["video_metadata"]["analyzed_at"] = datetime.now(UTC).isoformat()

    with open(report_dir / "report.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
