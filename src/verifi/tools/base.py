"""
Tool interface and registry for the forensic investigation system.

Save to: src/verifi/tools/base.py

Every detection/analysis capability is wrapped as a Tool with:
- A name and description (for the LLM to understand when to use it)
- A JSON-serializable parameter schema (for LLM tool calling)
- An execute() method that takes kwargs and returns ToolResult

The ToolRegistry holds all available tools and provides:
- Lookup by name
- Schema generation for LLM tool-use prompts
- Tier 1 (fixed sequence) and Tier 3 (agent-chosen) execution modes
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ToolResult:
    """Standardized output from any tool execution."""
    tool_name: str
    success: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0

    def summary(self) -> str:
        """One-line summary for the LLM to read."""
        if not self.success:
            return f"[{self.tool_name}] FAILED: {self.error}"
        # Subclasses override via data["summary"]
        return self.data.get("summary", f"[{self.tool_name}] completed")

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 1),
        }


class Tool(ABC):
    """
    Base class for all forensic investigation tools.

    Each tool has:
    - name: unique identifier (used in tool calls)
    - description: what it does (shown to the LLM)
    - parameters: JSON schema of accepted inputs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Run the tool with given parameters."""
        ...

    def schema(self) -> dict:
        """Generate LLM-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
        }


class ToolRegistry:
    """
    Central registry of all available forensic tools.

    Used by:
    - Tier 1: fixed pipeline calls tools in predetermined order
    - Tier 2: LLM explainer sees tool results (read-only)
    - Tier 3: agent selects and calls tools dynamically
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warn("tool_already_registered", name=tool.name)
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters."""
        import time
        tool = self.get(name)
        t0 = time.perf_counter()
        try:
            result = tool.execute(**kwargs)
            result.execution_time_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed,
            )

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def all_schemas(self) -> list[dict]:
        """Generate schemas for all tools (for LLM tool-use prompt)."""
        return [t.schema() for t in self._tools.values()]

    def schemas_for_ollama(self) -> list[dict]:
        """
        Format tool schemas for Ollama's function calling API.

        Ollama expects:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        return [
            {
                "type": "function",
                "function": t.schema(),
            }
            for t in self._tools.values()
        ]

    def schemas_for_claude(self) -> list[dict]:
        """
        Format tool schemas for Claude's tool_use API.

        Claude expects:
        {"name": ..., "description": ..., "input_schema": {"type": "object", "properties": ...}}
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
