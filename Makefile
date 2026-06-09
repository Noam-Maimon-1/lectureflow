# ── Lecture Pipeline ──────────────────────────────────────────────
# Usage:
#   make process URL="https://example.com/lecture.mp4"
#   make process URL="..." ARGS="--skip-analysis --whisper-model medium"
#   make test
#   make docker-build
#   make docker-run URL="..."

.PHONY: process test lint docker-build docker-run clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make process URL=\"...\"   Run the pipeline on a lecture URL"
	@echo "  make test                Run tests"
	@echo "  make docker-build        Build the Docker image"
	@echo "  make docker-run URL=\"...\" Run via Docker"
	@echo "  make clean               Remove output files"
	@echo "  make setup               Install dependencies"

# Run the pipeline locally
process:
	python main.py "$(URL)" $(ARGS)

# Run tests
test:
	python -m pytest tests/ -v

# Install dependencies
setup:
	pip install -r requirements.txt

# Docker
docker-build:
	docker build -t lecture-pipeline .

docker-run:
	docker compose run pipeline "$(URL)" $(ARGS)

# Clean output files
clean:
	rm -rf output/*
	@echo "Output directory cleaned."
