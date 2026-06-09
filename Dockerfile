# ── Lecture Pipeline Container ───────────────────────────────────
# Bundles Python, ffmpeg, and all dependencies so the pipeline
# runs identically on any machine. No "works on my laptop" issues.
#
# Build:  docker build -t lecture-pipeline .
# Run:    docker run --env-file .env -v ~/lecture_output:/output lecture-pipeline "VIDEO_URL"

FROM python:3.12-slim

# Install system dependencies
# ffmpeg: audio/video conversion
# curl: fallback downloader
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Output directory — mount a volume here to access files from the host
ENV LP_OUTPUT_DIR=/output
RUN mkdir -p /output

ENTRYPOINT ["python", "main.py"]
