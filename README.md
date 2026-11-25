# Lecture Pipeline

[![CI](https://github.com/Noam-Maimon-1/lectureflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Noam-Maimon-1/lectureflow/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An ETL pipeline that transforms university lecture recordings into AI-generated study guides.

Give it a lecture recording (video or audio) → it **extracts** the audio, **transforms** it into a transcript with speech-to-text, and **loads** the transcript into Claude to produce structured study material.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
│  Input   │───>│ Convert  │───>│ Transcribe │───>│ Analyze  │───>│  Render  │
│  file    │    │ (ffmpeg) │    │ (Whisper)  │    │ (Claude) │    │  (HTML)  │
└──────────┘    └──────────┘    └────────────┘    └──────────┘    └──────────┘
                  EXTRACT         TRANSFORM          LOAD

Input:  A video (.mp4, .mkv, .mov, ...) or audio file (.mp3, .wav, .m4a, .flac, .ogg)
Output: MP3 audio, text transcript, and a study guide as JSON, Markdown and HTML
```

Each stage is an independent module in `src/` — testable, replaceable, and extensible on its own. Audio input skips the ffmpeg step. Your input file is never modified or deleted.

## Output

Each run writes to the output directory (default `~/lecture_pipeline_output`):

| File | Contents |
|---|---|
| `<lecture>.mp3` | Extracted audio, video input only (128 kbps — enough for speech) |
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
# Full pipeline on a video:
python main.py ~/Downloads/lecture.mp4

# Audio works too (skips ffmpeg):
python main.py lecture.m4a

# Just transcribe (no API key needed):
python main.py lecture.mp4 --skip-analysis

# Non-English lectures — use a larger Whisper model:
python main.py lecture.mp4 --whisper-model medium

# Custom study instructions:
python main.py lecture.mp4 --instructions instructions/my_prompt.txt

# Choose where outputs go, skip the HTML page:
python main.py lecture.mp4 --output-dir ./output --no-html

# Using Make:
make process FILE=lecture.mp4
```

### Docker

Put the recording in `./input` (mounted read-only); results appear in `./output`:

```bash
docker compose build
docker compose run --rm pipeline /input/lecture.mp4
```

## Project Structure

```
lecture-pipeline/
├── main.py                  # CLI entry point
├── src/
│   ├── config.py            # Configuration management (env vars)
│   ├── logger.py            # Structured logging setup
│   ├── converter.py         # EXTRACT — audio from video (ffmpeg)
│   ├── transcriber.py       # TRANSFORM — audio to text (Whisper)
│   ├── analyzer.py          # LOAD — transcript to Claude API, JSON validation
│   ├── renderer.py          # Study guide JSON to self-contained HTML
│   └── pipeline.py          # Orchestrator — sequences all stages
├── tests/
│   └── test_pipeline.py     # Unit tests (ffmpeg, Whisper and Claude are mocked)
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

**Why take a file instead of a URL?** An earlier version downloaded lectures from URLs (yt-dlp with a curl fallback, retries and custom headers for CDN-hosted video). That made the pipeline depend on how each site serves video, which kept changing. Taking a file keeps the tool focused on its core job: it works with any recording, from any source. Getting the recording is left to the user.

**Why Docker?** The pipeline has non-trivial system dependencies (ffmpeg, Whisper model weights). Containerizing ensures reproducible runs across machines — a legitimate engineering reason, not resume padding.

## Configuration

All settings can be controlled via environment variables (see `.env.example`) or CLI flags. CLI flags override env vars. This separation of configuration from code follows [twelve-factor app](https://12factor.net/config) principles.

## Testing

```bash
pip install -r requirements-dev.txt
make test        # or: python -m pytest -v
ruff check .
```

The tests mock every external dependency (the ffmpeg subprocess, Whisper, and the Anthropic client), so they run in well under a second and don't need a GPU or an API key. CI runs lint and tests on every push and pull request.

## Future Roadmap

- [ ] **Batch mode**: Process every recording in a folder
- [ ] **Watch folder**: Pick up new recordings automatically as they appear
- [ ] **State tracking**: SQLite log of processed lectures for idempotency
- [ ] **Parallel processing**: Process multiple lectures concurrently
- [ ] **Notification**: Send study guide via email/Telegram when processing completes

## License

[MIT](LICENSE)
