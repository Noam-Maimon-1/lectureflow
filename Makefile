# ── Lecture Pipeline ──────────────────────────────────────────────
# Usage:
#   make process FILE=lecture.mp4
#   make process FILE=lecture.mp4 ARGS="--skip-analysis --whisper-model medium"
#   make test
#   make docker-build
#   make docker-run FILE=lecture.mp4   (file must be inside ./input)

.PHONY: process test lint docker-build docker-run clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make process FILE=...     Run the pipeline on a video/audio file"
	@echo "  make test                Run tests"
	@echo "  make lint                Run ruff"
	@echo "  make docker-build        Build the Docker image"
	@echo "  make docker-run FILE=...  Run via Docker (file in ./input)"
	@echo "  make clean               Remove output files"
	@echo "  make setup               Install dependencies"

# Run the pipeline locally
process:
	python main.py "$(FILE)" $(ARGS)

# Run tests
test:
	python -m pytest -v

# Lint
lint:
	ruff check .

# Install dependencies
setup:
	pip install -r requirements.txt -r requirements-dev.txt

# Docker
docker-build:
	docker build -t lecture-pipeline .

docker-run:
	docker compose run --rm pipeline "/input/$(notdir $(FILE))" $(ARGS)

# Clean output files
clean:
	rm -rf output/*
	@echo "Output directory cleaned."
