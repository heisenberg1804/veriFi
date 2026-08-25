"""Tests for the FastAPI endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from verifi.ensemble.aggregator import (
    FullFrameAnalysis,
    ManipulationType,
    SignalScore,
    VideoAnalysis,
)
from verifi.ensemble.aggregator import Verdict as AggVerdict
from verifi.pipeline.orchestrator import ForensicReport, StageTimings


def _make_mock_report(output_dir: str = "/tmp/test_report") -> ForensicReport:
    frame_analyses = [
        FullFrameAnalysis(
            frame_idx=0,
            timestamp_sec=0.0,
            signals=[
                SignalScore(name="clip", score=0.3),
                SignalScore(name="dct", score=0.4),
                SignalScore(name="noise_residual", score=0.35),
            ],
            ensemble_score=0.35,
            flagged=False,
        ),
    ]

    analysis = VideoAnalysis(
        frame_analyses=frame_analyses,
        frame_path_score=0.35,
        frame_path_active=True,
        video_score=0.35,
        confidence=0.5,
        verdict=AggVerdict.SUSPICIOUS,
        manipulation_type=ManipulationType.UNKNOWN,
        dominant_path="frame",
    )

    return ForensicReport(
        video_metadata={
            "filename": "test_video.mp4",
            "duration_sec": 10.5,
            "file_hash": "sha256:abc123",
        },
        analysis=analysis,
        signal_stats={
            "clip_mean": 0.3,
            "dct_mean": 0.4,
            "noise_residual_mean": 0.35,
        },
        timings=StageTimings(total=2.5),
        output_dir=output_dir,
    )


def _make_test_video(path: Path) -> None:
    """Create a minimal mp4 via OpenCV."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (320, 240))
    for _ in range(24):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def mock_pipeline():
    """Fixture that patches the pipeline singleton so no models are loaded."""
    mock = MagicMock()
    mock._models_loaded = True

    with patch("verifi.api.app._pipeline", mock), \
         patch("verifi.api.app._config", MagicMock(device=MagicMock(resolve=lambda: "cpu"))):
        yield mock


@pytest.fixture
def client(mock_pipeline):
    """FastAPI test client with mocked pipeline."""
    from verifi.api.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_health_shows_device(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "device" in data


class TestAnalyzeEndpoint:
    def test_analyze_missing_input(self, client):
        resp = client.post("/api/v1/analyze", json={})
        assert resp.status_code == 400

    def test_analyze_nonexistent_path(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={"video_path": "/nonexistent/video.mp4"},
        )
        assert resp.status_code == 404

    def test_analyze_with_path(self, client, mock_pipeline, tmp_path):
        video_path = tmp_path / "test.mp4"
        _make_test_video(video_path)

        report_dir = tmp_path / "report_out"
        report_dir.mkdir()
        mock_report = _make_mock_report(str(report_dir))
        mock_pipeline.analyze.return_value = mock_report

        resp = client.post(
            "/api/v1/analyze",
            json={"video_path": str(video_path), "skip_explainability": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "SUSPICIOUS"
        assert data["video_score"] == 0.35
        assert "report_id" in data
        assert data["processing_time_sec"] == 2.5

        mock_pipeline.analyze.assert_called_once_with(
            str(video_path),
            skip_explainability=True,
        )

    def test_analyze_returns_signal_stats(self, client, mock_pipeline, tmp_path):
        video_path = tmp_path / "test.mp4"
        _make_test_video(video_path)

        report_dir = tmp_path / "report_out"
        report_dir.mkdir()
        mock_pipeline.analyze.return_value = _make_mock_report(str(report_dir))

        resp = client.post(
            "/api/v1/analyze",
            json={"video_path": str(video_path)},
        )
        data = resp.json()
        assert "signal_stats" in data
        assert "clip_mean" in data["signal_stats"]

    def test_analyze_pipeline_error(self, client, mock_pipeline, tmp_path):
        video_path = tmp_path / "test.mp4"
        _make_test_video(video_path)

        mock_pipeline.analyze.side_effect = ValueError("Video too short")

        resp = client.post(
            "/api/v1/analyze",
            json={"video_path": str(video_path)},
        )
        assert resp.status_code == 422
        assert "Video too short" in resp.json()["detail"]


class TestUploadEndpoint:
    def test_upload_analyze(self, client, mock_pipeline, tmp_path):
        video_path = tmp_path / "upload_test.mp4"
        _make_test_video(video_path)

        report_dir = tmp_path / "report_out"
        report_dir.mkdir()
        mock_pipeline.analyze.return_value = _make_mock_report(str(report_dir))

        with open(video_path, "rb") as f:
            resp = client.post(
                "/api/v1/analyze/upload",
                files={"file": ("test.mp4", f, "video/mp4")},
                params={"skip_explainability": True},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "SUSPICIOUS"
        mock_pipeline.analyze.assert_called_once()


class TestReportEndpoints:
    def test_get_report_not_found(self, client):
        resp = client.get("/api/v1/report/nonexistent_id")
        assert resp.status_code == 404

    def test_get_report_exists(self, client, tmp_path):
        report_id = "test_report_123"
        report_dir = tmp_path / "reports" / report_id
        report_dir.mkdir(parents=True)

        report_data = {
            "report_id": report_id,
            "video_metadata": {"filename": "test.mp4", "duration_sec": 10.0},
            "verdict": "SUSPICIOUS",
            "video_score": 0.45,
            "confidence": 0.6,
            "manipulation_type": "unknown",
            "dominant_path": "frame",
            "face_path_score": 0.0,
            "face_path_active": False,
            "frame_path_score": 0.45,
            "frame_path_active": True,
            "frame_analyses": [],
            "face_analyses": [],
            "signal_stats": {"dct_mean": 0.4},
            "heatmap_paths": [],
            "forensic_view_paths": [],
            "timeline_path": "",
            "timings": {"total": 3.0},
            "output_dir": str(report_dir),
        }

        with open(report_dir / "report.json", "w") as f:
            json.dump(report_data, f)

        with patch("verifi.api.routes.analysis.REPORTS_DIR", tmp_path / "reports"):
            resp = client.get(f"/api/v1/report/{report_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == report_id
        assert data["verdict"] == "SUSPICIOUS"

    def test_list_reports_empty(self, client, tmp_path):
        with patch("verifi.api.routes.analysis.REPORTS_DIR", tmp_path / "no_reports"):
            resp = client.get("/api/v1/reports")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_reports(self, client, tmp_path):
        reports_dir = tmp_path / "reports"
        for i in range(3):
            rd = reports_dir / f"report_{i}"
            rd.mkdir(parents=True)
            data = {
                "report_id": f"report_{i}",
                "video_metadata": {"filename": f"vid{i}.mp4", "analyzed_at": f"2026-01-0{i+1}"},
                "verdict": "SUSPICIOUS",
                "video_score": 0.5,
            }
            with open(rd / "report.json", "w") as f:
                json.dump(data, f)

        with patch("verifi.api.routes.analysis.REPORTS_DIR", reports_dir):
            resp = client.get("/api/v1/reports?limit=2")

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
