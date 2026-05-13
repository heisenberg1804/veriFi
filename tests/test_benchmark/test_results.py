"""Tests for benchmark result I/O."""
import json

from verifi.benchmark.results import ResultsReader, ResultsWriter, VideoResult


def _make_result(**overrides) -> VideoResult:
    defaults = {
        "video_path": "/fake/video.mp4",
        "dataset": "ff++",
        "label": 1,
        "method": "Deepfakes",
        "predicted_score": 0.75,
        "predicted_verdict": "LIKELY_MANIPULATED",
        "confidence": 0.8,
        "face_path_score": 0.6,
        "frame_path_score": 0.75,
        "dominant_path": "frame",
        "manipulation_type": "full_synthesis",
        "signal_scores": {"dct": 0.7, "noise_residual": 0.8},
        "processing_time_sec": 30.0,
    }
    defaults.update(overrides)
    return VideoResult(**defaults)


def test_video_result_roundtrip():
    r = _make_result()
    d = r.to_dict()
    r2 = VideoResult.from_dict(d)
    assert r2.predicted_score == r.predicted_score
    assert r2.signal_scores == r.signal_scores
    assert r2.dataset == "ff++"


def test_results_writer_jsonl(tmp_path):
    writer = ResultsWriter(tmp_path / "run1")
    r1 = _make_result(video_path="/a.mp4", predicted_score=0.8)
    r2 = _make_result(video_path="/b.mp4", predicted_score=0.3, label=0)
    writer.write(r1)
    writer.write(r2)

    jsonl = tmp_path / "run1" / "results.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["predicted_score"] == 0.8


def test_results_writer_csv(tmp_path):
    writer = ResultsWriter(tmp_path / "run1")
    writer.write(_make_result(video_path="/a.mp4"))

    csv_path = tmp_path / "run1" / "results.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2  # header + 1 row


def test_get_processed_paths(tmp_path):
    writer = ResultsWriter(tmp_path / "run1")
    writer.write(_make_result(video_path="/a.mp4"))
    writer.write(_make_result(video_path="/b.mp4"))

    processed = writer.get_processed_paths()
    assert "/a.mp4" in processed
    assert "/b.mp4" in processed
    assert len(processed) == 2


def test_results_reader(tmp_path):
    writer = ResultsWriter(tmp_path / "run1")
    writer.write(_make_result(video_path="/a.mp4"))
    writer.write(_make_result(video_path="/b.mp4"))

    reader = ResultsReader(tmp_path / "run1")
    results = reader.load_all()
    assert len(results) == 2
    assert results[0].video_path == "/a.mp4"


def test_results_reader_empty(tmp_path):
    reader = ResultsReader(tmp_path / "nonexistent")
    assert reader.load_all() == []
    assert reader.count() == 0


def test_config_save_load(tmp_path):
    writer = ResultsWriter(tmp_path / "run1")
    writer.save_config({"dataset": "ff++", "max_videos": 100})

    reader = ResultsReader(tmp_path / "run1")
    config = reader.load_config()
    assert config["dataset"] == "ff++"
    assert config["max_videos"] == 100
