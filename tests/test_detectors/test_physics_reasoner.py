"""Tests for physics-based reasoning detector."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verifi.detectors.physics_reasoner import (
    PHYSICS_ANALYSIS_PROMPT,
    PhysicsAnalysisResult,
    PhysicsFrameResult,
    PhysicsReasoner,
    _encode_frame_b64,
    _extract_json,
    _resize_for_vision,
)


class TestDataclasses:
    def test_physics_frame_result_defaults(self):
        r = PhysicsFrameResult(frame_idx=0, timestamp_sec=1.5)
        assert r.shadow_score == 0.0
        assert r.overall_score == 0.0
        assert r.evidence == []
        assert r.reasoning == ""

    def test_physics_frame_result_with_scores(self):
        r = PhysicsFrameResult(
            frame_idx=10,
            timestamp_sec=3.0,
            shadow_score=0.8,
            reflection_score=0.3,
            perspective_score=0.1,
            anatomy_score=0.6,
            texture_score=0.4,
            overall_score=0.7,
            evidence=["shadow direction inconsistent"],
            reasoning="Multiple physics violations detected",
        )
        assert r.shadow_score == 0.8
        assert r.overall_score == 0.7
        assert len(r.evidence) == 1

    def test_physics_analysis_result_defaults(self):
        r = PhysicsAnalysisResult()
        assert r.aggregate_score == 0.5
        assert r.confidence == 0.0
        assert r.verdict_suggestion == "suspicious"
        assert r.num_frames_analyzed == 0

    def test_physics_analysis_result_to_dict(self):
        fr = PhysicsFrameResult(frame_idx=0, timestamp_sec=0.0, overall_score=0.8)
        r = PhysicsAnalysisResult(
            per_frame=[fr],
            aggregate_score=0.8,
            confidence=0.9,
            evidence_summary=["test evidence"],
            verdict_suggestion="likely_manipulated",
            num_frames_analyzed=1,
            backend_used="ollama",
        )
        d = r.to_dict()
        assert d["aggregate_score"] == 0.8
        assert d["confidence"] == 0.9
        assert d["verdict_suggestion"] == "likely_manipulated"
        assert len(d["per_frame"]) == 1
        assert d["per_frame"][0]["overall_score"] == 0.8


class TestParseResponse:
    def test_parse_valid_json(self):
        reasoner = PhysicsReasoner()
        raw = json.dumps({
            "frames": [
                {
                    "frame_index": 0,
                    "shadow_score": 0.3,
                    "reflection_score": 0.2,
                    "perspective_score": 0.1,
                    "anatomy_score": 0.5,
                    "texture_score": 0.4,
                    "overall_score": 0.7,
                    "evidence": ["hands have 6 fingers"],
                    "reasoning": "Anatomical anomaly detected",
                },
                {
                    "frame_index": 1,
                    "shadow_score": 0.1,
                    "reflection_score": 0.0,
                    "perspective_score": 0.0,
                    "anatomy_score": 0.0,
                    "texture_score": 0.2,
                    "overall_score": 0.15,
                    "evidence": [],
                    "reasoning": "No issues found",
                },
            ],
            "aggregate_assessment": "One frame with anomaly",
            "verdict_suggestion": "suspicious",
        })

        results = reasoner._parse_response(raw, [0, 42], [0.0, 1.5])
        assert len(results) == 2
        assert results[0].frame_idx == 0
        assert results[0].shadow_score == 0.3
        assert results[0].overall_score == 0.7
        assert results[0].evidence == ["hands have 6 fingers"]
        assert results[1].frame_idx == 42
        assert results[1].overall_score == 0.15

    def test_parse_json_in_code_block(self):
        reasoner = PhysicsReasoner()
        raw = """Here is my analysis:
```json
{
  "frames": [
    {
      "frame_index": 0,
      "shadow_score": 0.5,
      "reflection_score": 0.0,
      "perspective_score": 0.0,
      "anatomy_score": 0.0,
      "texture_score": 0.0,
      "overall_score": 0.4,
      "evidence": ["shadow mismatch"],
      "reasoning": "Minor shadow issue"
    }
  ],
  "aggregate_assessment": "Minor issue",
  "verdict_suggestion": "suspicious"
}
```
"""
        results = reasoner._parse_response(raw, [5], [2.0])
        assert len(results) == 1
        assert results[0].shadow_score == 0.5
        assert results[0].frame_idx == 5

    def test_parse_invalid_json_returns_defaults(self):
        reasoner = PhysicsReasoner()
        raw = "I cannot analyze these frames because they are blurry."
        results = reasoner._parse_response(raw, [0, 1], [0.0, 0.5])
        assert len(results) == 2
        assert results[0].overall_score == 0.5
        assert "Failed to parse" in results[0].reasoning

    def test_parse_fewer_frames_than_expected(self):
        reasoner = PhysicsReasoner()
        raw = json.dumps({
            "frames": [
                {
                    "frame_index": 0,
                    "shadow_score": 0.3,
                    "overall_score": 0.4,
                    "evidence": [],
                    "reasoning": "ok",
                }
            ],
            "verdict_suggestion": "likely_authentic",
        })
        results = reasoner._parse_response(raw, [0, 10, 20], [0.0, 1.0, 2.0])
        assert len(results) == 3
        assert results[0].overall_score == 0.4
        assert results[1].overall_score == 0.5
        assert results[2].overall_score == 0.5


class TestSelectFrames:
    def _make_frames(self, n: int):
        frames = []
        for i in range(n):
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frames.append((img, i * 10, float(i * 2)))
        return frames

    def test_fewer_than_max_returns_all(self):
        reasoner = PhysicsReasoner(max_frames=6)
        frames = self._make_frames(3)
        selected = reasoner._select_frames(frames, None, 6)
        assert len(selected) == 3

    def test_selects_max_frames(self):
        reasoner = PhysicsReasoner(max_frames=4)
        frames = self._make_frames(20)
        selected = reasoner._select_frames(frames, None, 4)
        assert len(selected) <= 4

    def test_includes_start_and_end(self):
        reasoner = PhysicsReasoner(max_frames=4)
        frames = self._make_frames(10)
        selected = reasoner._select_frames(frames, None, 4)
        indices = [s[1] for s in selected]
        assert frames[0][1] in indices
        assert frames[-1][1] in indices

    def test_temporal_diversity(self):
        reasoner = PhysicsReasoner(max_frames=6)
        frames = self._make_frames(10)
        selected = reasoner._select_frames(frames, None, 6)
        timestamps = [s[2] for s in selected]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1]

    def test_tier1_context_prioritizes_high_scores(self):
        reasoner = PhysicsReasoner(max_frames=3)
        frames = self._make_frames(10)
        context = {
            "frame_scores": {str(i * 10): 0.1 + i * 0.08 for i in range(10)}
        }
        selected = reasoner._select_frames(frames, context, 3)
        assert len(selected) == 3


class TestPrompt:
    def test_prompt_contains_all_categories(self):
        for category in ["SHADOWS", "REFLECTIONS", "PERSPECTIVE", "ANATOMY", "TEXTURE"]:
            assert category in PHYSICS_ANALYSIS_PROMPT

    def test_prompt_requests_json(self):
        assert '"frames"' in PHYSICS_ANALYSIS_PROMPT
        assert '"overall_score"' in PHYSICS_ANALYSIS_PROMPT
        assert '"evidence"' in PHYSICS_ANALYSIS_PROMPT

    def test_build_prompt_with_context(self):
        reasoner = PhysicsReasoner()
        prompt = reasoner._build_physics_prompt({"video_score": 0.6, "verdict": "SUSPICIOUS"})
        assert "Tier 1 statistical analysis" in prompt
        assert "0.6" in prompt

    def test_build_prompt_without_context(self):
        reasoner = PhysicsReasoner()
        prompt = reasoner._build_physics_prompt(None)
        assert "Tier 1" not in prompt
        assert "SHADOWS" in prompt


class TestHelpers:
    def test_resize_for_vision_small_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = _resize_for_vision(img, max_side=720)
        assert resized.shape == (480, 640, 3)

    def test_resize_for_vision_large_image(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = _resize_for_vision(img, max_side=720)
        assert max(resized.shape[:2]) == 720

    def test_encode_frame_b64(self):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        b64 = _encode_frame_b64(img)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_extract_json_plain(self):
        raw = '{"key": "value"}'
        result = _extract_json(raw)
        assert result == {"key": "value"}

    def test_extract_json_code_block(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _extract_json(raw)
        assert result == {"key": "value"}

    def test_extract_json_with_surrounding_text(self):
        raw = "Here is the result: {\"key\": 1} and that's it."
        result = _extract_json(raw)
        assert result == {"key": 1}

    def test_extract_json_invalid(self):
        result = _extract_json("no json here")
        assert result is None


class TestBuildResult:
    def test_build_result_high_score(self):
        reasoner = PhysicsReasoner()
        frames = [
            PhysicsFrameResult(frame_idx=0, timestamp_sec=0.0, overall_score=0.9),
            PhysicsFrameResult(frame_idx=1, timestamp_sec=1.0, overall_score=0.8),
        ]
        raw = json.dumps({"verdict_suggestion": "likely_manipulated"})
        result = reasoner._build_result(frames, raw, "ollama")
        assert result.aggregate_score == pytest.approx(0.85, abs=0.01)
        assert result.verdict_suggestion == "likely_manipulated"

    def test_build_result_low_score(self):
        reasoner = PhysicsReasoner()
        frames = [
            PhysicsFrameResult(frame_idx=0, timestamp_sec=0.0, overall_score=0.1),
            PhysicsFrameResult(frame_idx=1, timestamp_sec=1.0, overall_score=0.2),
        ]
        raw = json.dumps({"verdict_suggestion": "likely_authentic"})
        result = reasoner._build_result(frames, raw, "claude")
        assert result.aggregate_score == pytest.approx(0.15, abs=0.01)
        assert result.verdict_suggestion == "likely_authentic"
        assert result.backend_used == "claude"

    def test_build_result_empty(self):
        reasoner = PhysicsReasoner()
        result = reasoner._build_result([], "", "ollama")
        assert result.num_frames_analyzed == 0

    def test_confidence_high_when_consistent(self):
        reasoner = PhysicsReasoner()
        frames = [
            PhysicsFrameResult(frame_idx=i, timestamp_sec=float(i), overall_score=0.8)
            for i in range(5)
        ]
        result = reasoner._build_result(frames, "{}", "ollama")
        assert result.confidence == 1.0

    def test_confidence_low_when_inconsistent(self):
        reasoner = PhysicsReasoner()
        frames = [
            PhysicsFrameResult(frame_idx=0, timestamp_sec=0.0, overall_score=0.1),
            PhysicsFrameResult(frame_idx=1, timestamp_sec=1.0, overall_score=0.9),
        ]
        result = reasoner._build_result(frames, "{}", "ollama")
        assert result.confidence < 0.5
