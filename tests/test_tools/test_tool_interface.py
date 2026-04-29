# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE 1: tests/test_tools/test_tool_interface.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for tool interface and registry."""
import numpy as np
import pytest

from verifi.tools.base import Tool, ToolRegistry, ToolResult


class MockTool(Tool):
    @property
    def name(self):
        return "mock_tool"

    @property
    def description(self):
        return "A mock tool for testing"

    @property
    def parameters(self):
        return {"value": {"type": "integer", "description": "test value"}}

    def execute(self, value: int = 0, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": value * 2, "summary": f"Doubled: {value * 2}"},
        )


class FailingTool(Tool):
    @property
    def name(self):
        return "failing_tool"

    @property
    def description(self):
        return "Always fails"

    @property
    def parameters(self):
        return {}

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Intentional failure")


def test_tool_result_summary():
    r = ToolResult(tool_name="test", success=True, data={"summary": "hello"})
    assert r.summary() == "hello"


def test_tool_result_failure_summary():
    r = ToolResult(tool_name="test", success=False, error="broken")
    assert "FAILED" in r.summary()
    assert "broken" in r.summary()


def test_tool_result_to_dict():
    r = ToolResult(tool_name="test", success=True, data={"x": 1}, execution_time_ms=42.5)
    d = r.to_dict()
    assert d["tool_name"] == "test"
    assert d["execution_time_ms"] == 42.5


def test_registry_register_and_get():
    reg = ToolRegistry()
    tool = MockTool()
    reg.register(tool)
    assert "mock_tool" in reg
    assert reg.get("mock_tool") is tool


def test_registry_execute():
    reg = ToolRegistry()
    reg.register(MockTool())
    result = reg.execute("mock_tool", value=5)
    assert result.success
    assert result.data["result"] == 10


def test_registry_execute_missing_tool():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.execute("nonexistent")


def test_registry_handles_tool_failure():
    reg = ToolRegistry()
    reg.register(FailingTool())
    result = reg.execute("failing_tool")
    assert not result.success
    assert "Intentional failure" in result.error


def test_registry_tracks_execution_time():
    reg = ToolRegistry()
    reg.register(MockTool())
    result = reg.execute("mock_tool", value=1)
    assert result.execution_time_ms >= 0


def test_registry_list_tools():
    reg = ToolRegistry()
    reg.register(MockTool())
    assert reg.list_tools() == ["mock_tool"]


def test_registry_all_schemas():
    reg = ToolRegistry()
    reg.register(MockTool())
    schemas = reg.all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "mock_tool"
    assert "parameters" in schemas[0]


def test_registry_ollama_schema_format():
    reg = ToolRegistry()
    reg.register(MockTool())
    schemas = reg.schemas_for_ollama()
    assert schemas[0]["type"] == "function"
    assert "function" in schemas[0]
    assert schemas[0]["function"]["name"] == "mock_tool"


def test_registry_claude_schema_format():
    reg = ToolRegistry()
    reg.register(MockTool())
    schemas = reg.schemas_for_claude()
    assert schemas[0]["name"] == "mock_tool"
    assert "input_schema" in schemas[0]


def test_tool_schema():
    tool = MockTool()
    s = tool.schema()
    assert s["name"] == "mock_tool"
    assert "description" in s
    assert "parameters" in s


def test_dct_tool_with_image():
    from verifi.tools.detection_tools import DCTFrequencyTool
    tool = DCTFrequencyTool()
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = tool.execute(image=img)
    assert result.success
    assert 0 <= result.data["score"] <= 1


def test_dct_tool_no_image():
    from verifi.tools.detection_tools import DCTFrequencyTool
    tool = DCTFrequencyTool()
    result = tool.execute()
    assert not result.success


def test_zoom_tool():
    from verifi.tools.sampling_tools import ZoomRegionTool
    tool = ZoomRegionTool()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = tool.execute(image=img, x=100, y=100, width=200, height=200, target_size=224)
    assert result.success
    assert result.data["image"].shape == (224, 224, 3)


def test_zoom_tool_clamps_bounds():
    from verifi.tools.sampling_tools import ZoomRegionTool
    tool = ZoomRegionTool()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = tool.execute(image=img, x=600, y=400, width=500, height=500)
    assert result.success


def test_effnet_tool_skips_small():
    from unittest.mock import MagicMock

    from verifi.tools.detection_tools import EfficientNetDetectionTool
    mock_det = MagicMock()
    tool = EfficientNetDetectionTool(mock_det)
    tiny = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = tool.execute(image=tiny)
    assert result.success
    assert result.data.get("skipped") is True
    mock_det.predict.assert_not_called()

