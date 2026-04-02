#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run from project root"
    exit 1
fi

# Directories
mkdir -p src/verifi/ensemble
mkdir -p src/verifi/explainability
mkdir -p src/verifi/pipeline
mkdir -p tests/test_ensemble
mkdir -p tests/test_explainability
mkdir -p tests/test_pipeline

# Source files
touch src/verifi/ensemble/__init__.py
touch src/verifi/ensemble/aggregator.py          # 3A — dual-path ensemble
touch src/verifi/explainability/__init__.py
touch src/verifi/explainability/gradcam.py        # 3B — GradCAM for ViT + CNN
touch src/verifi/explainability/heatmap_renderer.py  # 3C — overlay + forensic view
touch src/verifi/pipeline/__init__.py
touch src/verifi/pipeline/orchestrator.py         # 3D — end-to-end pipeline

# Integration test
touch scripts/test_phase3.py                      # 3E

# Unit tests
touch tests/test_ensemble/__init__.py
touch tests/test_ensemble/test_aggregator.py
touch tests/test_explainability/__init__.py
touch tests/test_explainability/test_gradcam.py
touch tests/test_pipeline/__init__.py
touch tests/test_pipeline/test_orchestrator.py

echo "Phase 3 scaffold created. Paste content into:"
echo "  src/verifi/ensemble/aggregator.py          ← 3A"
echo "  src/verifi/explainability/gradcam.py        ← 3B"
echo "  src/verifi/explainability/heatmap_renderer.py ← 3C"
echo "  src/verifi/pipeline/orchestrator.py         ← 3D"
echo "  scripts/test_phase3.py                      ← 3E"
echo "  tests/test_ensemble/test_aggregator.py      ← 3F (section 1)"
echo "  tests/test_explainability/test_gradcam.py   ← 3F (section 2)"
echo "  tests/test_pipeline/test_orchestrator.py    ← 3F (section 3)"