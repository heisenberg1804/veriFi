
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE 2: tests/test_agent/test_investigator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for the LangGraph investigation agent."""

from verifi.agent.investigator import InvestigationReport


def test_investigation_report_to_dict():
    report = InvestigationReport(
        verdict="LIKELY_MANIPULATED",
        confidence=0.85,
        manipulation_type="full_synthesis",
        summary="Test summary",
        evidence=["evidence 1"],
        investigation_trace=["step 1"],
        caveats=["caveat 1"],
        recommended_action="review",
        total_time_sec=10.5,
        num_tool_calls=3,
    )
    d = report.to_dict()
    assert d["verdict"] == "LIKELY_MANIPULATED"
    assert d["confidence"] == 0.85
    assert d["total_time_sec"] == 10.5
    assert d["num_tool_calls"] == 3


def test_empty_report():
    report = InvestigationReport(
        verdict="LIKELY_AUTHENTIC",
        confidence=0.9,
        manipulation_type="none",
        summary="No issues found",
        evidence=[],
        investigation_trace=[],
        caveats=[],
        recommended_action="none",
    )
    d = report.to_dict()
    assert d["verdict"] == "LIKELY_AUTHENTIC"


def test_report_defaults():
    """Report should accept all required fields."""
    report = InvestigationReport(
        verdict="SUSPICIOUS",
        confidence=0.5,
        manipulation_type="unknown",
        summary="default",
        evidence=[],
        investigation_trace=[],
        caveats=[],
        recommended_action="Manual review",
    )
    assert report.verdict == "SUSPICIOUS"
    assert report.confidence == 0.5

def test_report_json_fields():
    report = InvestigationReport(
        verdict="SUSPICIOUS",
        confidence=0.6,
        manipulation_type="unknown",
        summary="test",
        evidence=["a", "b"],
        investigation_trace=["step 1", "step 2"],
        caveats=["c"],
        recommended_action="review",
    )
    d = report.to_dict()
    assert len(d["evidence"]) == 2
    assert len(d["investigation_trace"]) == 2
    assert "recommended_action" in d
