#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run from project root"
    exit 1
fi

# ── 1. Directories ──
mkdir -p src/verifi/tools
mkdir -p src/verifi/agent
mkdir -p tests/test_tools
mkdir -p tests/test_agent

# ── 2. Source files ──
touch src/verifi/tools/__init__.py
touch src/verifi/tools/base.py              # Tool interface + registry
touch src/verifi/tools/detection_tools.py   # CLIP, EfficientNet, DCT, temporal
touch src/verifi/tools/sampling_tools.py    # Frame sampling, zoom, face detection
touch src/verifi/tools/analysis_tools.py    # GradCAM, metadata, forensic view
touch src/verifi/tools/factory.py           # Tool registry assembly (no circular imports)
touch src/verifi/agent/__init__.py
touch src/verifi/agent/investigator.py      # Tier 3 agent loop (LangGraph)
touch src/verifi/agent/planner.py           # Investigation strategy prompts
touch scripts/test_phase4.py
touch tests/test_tools/__init__.py
touch tests/test_tools/test_tool_interface.py
touch tests/test_agent/__init__.py
touch tests/test_agent/test_investigator.py

# ── 3. Install LangGraph + LangChain deps ──
echo "Installing LangGraph dependencies..."
pip install langgraph langchain-core langchain-ollama langchain-anthropic

# ── 4. Add to pyproject.toml if not present ──
if ! grep -q "langgraph" pyproject.toml; then
    sed -i '' 's/"structlog>=24.1",/"structlog>=24.1",\n    "langgraph>=0.2",\n    "langchain-core>=0.3",\n    "langchain-ollama>=0.2",\n    "langchain-anthropic>=0.3",/' pyproject.toml
    echo "[✓] Added LangGraph deps to pyproject.toml"
fi

pip install -e ".[dev]"

echo ""
echo "Phase 4 scaffold created. Paste content into:"
echo "  src/verifi/tools/base.py              ← 4A (tool interface)"
echo "  src/verifi/tools/detection_tools.py   ← 4B (detection wrappers)"
echo "  src/verifi/tools/sampling_tools.py    ← 4C (sampling wrappers)"
echo "  src/verifi/tools/analysis_tools.py    ← 4D (explainability wrappers)"
echo "  src/verifi/tools/factory.py           ← 4E (registry assembly)"
echo "  src/verifi/agent/planner.py           ← 4F (prompts)"
echo "  src/verifi/agent/investigator.py      ← 4G (LangGraph agent)"
echo "  scripts/test_phase4.py                ← 4H (integration test)"
echo "  tests/test_tools/test_tool_interface.py  ← 4I (section 1)"
echo "  tests/test_agent/test_investigator.py    ← 4I (section 2)"