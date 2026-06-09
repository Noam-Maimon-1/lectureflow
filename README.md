# Lecture Pipeline

An ETL pipeline that transforms university lecture recordings into AI-generated study guides.

**Extract** a lecture video from a URL → **Transform** it through audio conversion and speech-to-text transcription → **Load** the transcript into Claude for structured study material.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│ Download │───>│ Convert  │───>│ Transcribe │───>│ Analyze  │
│ (yt-dlp) │    │ (ffmpeg) │    │ (Whisper)  │    │ (Claude) │
└──────────┘    └──────────┘    └────────────┘    └──────────┘
   EXTRACT       TRANSFORM 1     TRANSFORM 2        LOAD

Input: Video URL
Output: MP3 audio + text transcript + study guide (Markdown)
```

Each stage is an independent module in `src/` — testable, replaceable, and extensible on its own.

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/lecture-pipeline.git
cd lecture-pipeline

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Usage

```bash
# Full pipeline — paste a lecture video URL:
python main.py "https://moodle.example.com/media/lecture.mp4"

# Just download + transcribe (no API key needed):
python main.py "URL" --skip-analysis

# Non-English lectures — use a larger Whisper model:
python main.py "URL" --whisper-model medium

# Custom study instructions:
python main.py "URL" --instructions instructions/my_prompt.txt

# Using Make:
make process URL="https://..."
```

### Docker

```bash
docker build -t lecture-pipeline .
docker compose run pipeline "VIDEO_URL"
```

## Project Structure

```
lecture-pipeline/
├── main.py                  # CLI entry point
├── src/
│   ├── config.py            # Configuration management (env vars)
│   ├── logger.py            # Structured logging setup
│   ├── downloader.py        # EXTRACT — video download with retries
│   ├── converter.py         # TRANSFORM — video to audio (ffmpeg)
│   ├── transcriber.py       # TRANSFORM — audio to text (Whisper)
│   ├── analyzer.py          # LOAD — transcript to Claude API
│   └── pipeline.py          # Orchestrator — sequences all stages
├── tests/
│   └── test_pipeline.py     # Unit tests for each module
├── instructions/
│   └── default.txt          # Default study prompt for Claude
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
```

## Design Decisions

**Why not Airflow/Prefect?** This pipeline is on-demand, single-user, and linear — an orchestration framework would add operational overhead without solving a problem that exists at this scale. If extended to scheduled batch processing, Prefect would be a natural next step.

**Why modular stages?** Each stage has a single responsibility and a clean interface (input path → output path). This makes them independently testable, replaceable (swap Whisper for another STT engine), and composable (skip stages with flags).

**Why Docker?** The pipeline has non-trivial system dependencies (ffmpeg, Whisper model weights). Containerizing ensures reproducible runs across machines — a legitimate engineering reason, not resume padding.

## Configuration

All settings can be controlled via environment variables (see `.env.example`) or CLI flags. CLI flags override env vars. This separation of configuration from code follows [twelve-factor app](https://12factor.net/config) principles.

## Testing

```bash
make test
# or
python -m pytest tests/ -v
```

## Future Roadmap

- [ ] **Moodle scraper**: Authenticate and automatically discover new lecture recordings
- [ ] **Scheduled runs**: Weekly cron or Prefect flow to process new lectures automatically
- [ ] **State tracking**: SQLite log of processed lectures for idempotency
- [ ] **Parallel processing**: Batch-process multiple lectures concurrently
- [ ] **Notification**: Send study guide via email/Telegram when processing completes
