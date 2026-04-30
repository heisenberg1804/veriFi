"""
Tool registry assembly — the single point where tools meet models.

Save to: src/verifi/tools/factory.py

This module is the ONLY place that imports both tools and pipeline.
It receives pre-loaded model instances and wires them into tools.
Neither tools nor agent import this module — it's called from the
app entry point (scripts, API, tests).

Dependency flow:
  detectors → tools (wraps detectors)
  detectors → pipeline (uses detectors)
  tools + pipeline → factory (assembles registry)
  factory → app entry point (scripts/test_phase4.py, api/app.py)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from verifi.detectors.frequency import FrequencyAnalyzer
from verifi.explainability.gradcam import GradCAMGenerator
from verifi.tools.analysis_tools import (
    ForensicViewTool,
    GradCAMTool,
    QuickScanTool,
)
from verifi.tools.base import ToolRegistry
from verifi.tools.detection_tools import (
    CLIPDetectionTool,
    DCTFrequencyTool,
    EfficientNetDetectionTool,
    TemporalConsistencyTool,
)
from verifi.tools.sampling_tools import (
    CheckMetadataTool,
    FaceDetectionTool,
    SampleFramesTool,
    ZoomRegionTool,
)

if TYPE_CHECKING:
    from verifi.pipeline.orchestrator import VeriFiPipeline

logger = structlog.get_logger()


def create_tool_registry(
    pipeline: VeriFiPipeline,
) -> ToolRegistry:
    """
    Create a fully populated ToolRegistry from a loaded pipeline.

    Args:
        pipeline: VeriFiPipeline with models already loaded.

    Returns:
        ToolRegistry with all forensic tools registered.
    """
    registry = ToolRegistry()
    device = pipeline.device

    # ── Tier 1 entry point ──
    registry.register(QuickScanTool(pipeline))

    # ── Detection tools ──
    registry.register(CLIPDetectionTool(pipeline._clip))
    registry.register(EfficientNetDetectionTool(pipeline._effnet))
    registry.register(DCTFrequencyTool(FrequencyAnalyzer()))
    registry.register(TemporalConsistencyTool())

    # ── Sampling tools ──
    registry.register(ZoomRegionTool())
    registry.register(SampleFramesTool())
    registry.register(FaceDetectionTool(pipeline._face_pipeline))
    registry.register(CheckMetadataTool())

    # ── Analysis tools ──
    registry.register(GradCAMTool(GradCAMGenerator(device=device)))
    registry.register(ForensicViewTool(FrequencyAnalyzer()))

    logger.info("tool_registry_created", num_tools=len(registry), tools=registry.list_tools())
    return registry


def get_langchain_tools(registry: ToolRegistry) -> list:
    """
    Convert ToolRegistry tools into LangChain-compatible tools
    for use with LangGraph.

    LangGraph expects tools decorated with @tool or as StructuredTool.
    We wrap our Tool.execute() methods with the LangChain interface.
    """
    from langchain_core.tools import StructuredTool

    lc_tools = []

    # Map our tool names to the subset the agent can call
    # (quick_scan is always first, then investigation tools)
    agent_tool_names = [
        "quick_scan",
        "run_clip_detection",
        "run_dct_analysis",
        "run_temporal_analysis",
        "zoom_region",
        "sample_more_frames",
        "detect_faces",
        "check_metadata",
        # NOTE: generate_gradcam and create_forensic_view are excluded —
        # they need model objects that can't pass through LangGraph's
        # text interface. They remain in the registry for Tier 1 use.
    ]

    for name in agent_tool_names:
        if name not in registry:
            continue

        tool = registry.get(name)

        # Build a wrapper function that the agent can call
        # We use a closure to capture the tool instance
        def _make_executor(t):
            def executor(**kwargs) -> str:
                result = t.execute(**kwargs)
                return result.summary()
            return executor

        lc_tool = StructuredTool.from_function(
            func=_make_executor(tool),
            name=tool.name,
            description=tool.description,
        )
        lc_tools.append(lc_tool)

    logger.info("langchain_tools_created", count=len(lc_tools))
    return lc_tools
