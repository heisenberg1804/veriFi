.PHONY: setup install test lint format run validate weights clean

# Full setup from scratch
setup: install weights validate
	@echo "\n\033[92m✓ Setup complete. Run 'make test' to verify.\033[0m"

# Install dependencies
install:
	python3 -m pip install -e ".[dev]"
	python3 -m pip install python-dotenv

# Download model weights
weights:
	python3 scripts/download_weights.py

# Validate entire setup
validate:
	python3 scripts/validate_setup.py

# Run tests
test:
	python3 -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	python3 -m pytest tests/ -v --cov=verifi --cov-report=term-missing

# Lint
lint:
	python3 -m ruff check src/ tests/

# Format
format:
	python3 -m ruff format src/ tests/

# Run API server (development)
run:
	python3 -m uvicorn verifi.api.app:app --reload --port 8000

# Quick smoke test: run frequency analyzer on a dummy image
smoke:
	python3 -c "import numpy as np; from verifi.detectors.frequency import FrequencyAnalyzer; \
		a = FrequencyAnalyzer(); r = a.analyze(np.random.randint(0,255,(224,224,3),dtype=np.uint8)); \
		print(f'Score: {r.score:.3f}, HF ratio: {r.metadata[\"hf_ratio\"]:.3f}')"

# Clean caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist
