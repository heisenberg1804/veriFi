#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# VeriFi — Phase 2 Scaffold (empty files + directories)
# Run from verifi/ project root:
#   chmod +x scaffold_phase2.sh && ./scaffold_phase2.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run this from your verifi/ project root"
    exit 1
fi

# Directories
mkdir -p src/verifi/ingestion
mkdir -p src/verifi/sampling
mkdir -p src/verifi/preprocessing
mkdir -p src/verifi/explanation
mkdir -p tests/test_sampling
mkdir -p tests/test_preprocessing
mkdir -p data/reports/phase2_debug

# Source files (paste content into these)
touch src/verifi/ingestion/validator.py          # Phase 2A
touch src/verifi/sampling/scene_detector.py      # Phase 2B
touch src/verifi/sampling/frame_selector.py      # Phase 2C
touch src/verifi/preprocessing/face_detector.py  # Phase 2D
touch src/verifi/explanation/llm_explainer.py    # Ollama explainer

# Scripts
touch scripts/test_phase2.py                     # Phase 2E

# Tests
touch tests/test_sampling/__init__.py
touch tests/test_sampling/test_scene_detector.py
touch tests/test_sampling/test_frame_selector.py
touch tests/test_preprocessing/__init__.py
touch tests/test_preprocessing/test_face_detector.py

echo "Done. Paste artifact content into:"
echo "  src/verifi/ingestion/validator.py          ← Phase 2A"
echo "  src/verifi/sampling/scene_detector.py      ← Phase 2B"
echo "  src/verifi/sampling/frame_selector.py      ← Phase 2C"
echo "  src/verifi/preprocessing/face_detector.py  ← Phase 2D"
echo "  src/verifi/explanation/llm_explainer.py    ← Ollama explainer"
echo "  scripts/test_phase2.py                     ← Phase 2E"
echo "  tests/test_sampling/test_scene_detector.py ← Phase 2F (section 1)"
echo "  tests/test_sampling/test_frame_selector.py ← Phase 2F (section 2)"
echo "  tests/test_preprocessing/test_face_detector.py ← Phase 2F (section 3)"