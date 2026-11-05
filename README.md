# Lecture Pipeline

[![CI](https://github.com/Noam-Maimon-1/lectureflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Noam-Maimon-1/lectureflow/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An ETL pipeline that transforms university lecture recordings into AI-generated study guides.

**Extract** a lecture video from a URL → **Transform** it through audio conversion and speech-to-text transcription → **Load** the transcript into Claude for structured study material.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
│ Download │───>│ Convert  │───>│ Transcribe │───>│ Analyze  │───>│  Render  │
│ (yt-dlp) │    │ (ffmpeg) │    │ (Whisper)  │    │ (Claude) │    │  (HTML)  │
└──────────┘    └──────────┘    └────────────┘    └──────────┘    └──────────┘
   EXTRACT       TRANSFORM 1     TRANSFORM 2        LOAD

Input:  Video URL, or a local video/audio file (--local-file)
Output: MP3 audio, text transcript, and a study guide as JSON, Markdown and HTML
```

Each stage is an independent module in `src/` — testable, replaceable, and extensible on its own.

## Output

Each run writes to the output directory (default `~/lecture_pipeline_output`):

| File | Contents |
|---|---|
| `<lecture>.mp3` | Extracted audio (128 kbps — enough for speech) |
| `<lecture>.txt` | Raw Whisper transcript |
| `study_guide.json` | Structured guide: summary, topics → subtopics, key terms, misconceptions, review Q&A, lecture notes |
| `study_guide.md` | The same guide rendered as Markdown |
| `study_guide.html` | Self-contained page (no external assets) with a sidebar table of contents, light/dark toggle and collapsible review answers |

Claude is instructed to return JSON against a fixed schema ([instructions/default.txt](instructions/default.txt)). The analyzer validates the required keys and, if parsing fails, saves the raw response to `study_guide_raw.txt` for debugging.

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

### Setup

```bash
git clone https://github.com/Noam-Maimon-1/lectureflow.git
cd lectureflow

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

# Process a recording already on disk (your file is never deleted):
python main.py --local-file ~/Downloads/lecture.mp4

# CDN-hosted videos that need a Referer (e.g. Moodle):
python main.py "URL" --referer https://moodle.example.ac.il/

# Skip the HTML page (JSON + Markdown only):
python main.py "URL" --no-html

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
│   ├── analyzer.py          # LOAD — transcript to Claude API, JSON validation
│   ├── renderer.py          # Study guide JSON to self-contained HTML
│   └── pipeline.py          # Orchestrator — sequences all stages
├── tests/
│   └── test_pipeline.py     # Unit tests (ffmpeg, yt-dlp, Whisper and Claude are mocked)
├── .github/workflows/ci.yml # Lint + tests on Python 3.11–3.13
├── instructions/
│   └── default.txt          # Default study prompt for Claude
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml           # Ruff and pytest configuration
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # pytest + ruff (all CI needs)
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
pip install -r requirements-dev.txt
make test        # or: python -m pytest -v
ruff check .
```

The tests mock every external dependency (subprocess calls to ffmpeg/yt-dlp/curl, Whisper, and the Anthropic client), so they run in well under a second and don't need a GPU or an API key. CI runs lint and tests on every push and pull request.

## Future Roadmap

- [ ] **Moodle scraper**: Authenticate and automatically discover new lecture recordings
- [ ] **Scheduled runs**: Weekly cron or Prefect flow to process new lectures automatically
- [ ] **State tracking**: SQLite log of processed lectures for idempotency
- [ ] **Parallel processing**: Batch-process multiple lectures concurrently
- [ ] **Notification**: Send study guide via email/Telegram when processing completes

## License

[MIT](LICENSE)
