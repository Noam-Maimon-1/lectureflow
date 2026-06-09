"""
Tests for the lecture pipeline.

These test individual stages in isolation — each module should be
testable without running the full pipeline or making external calls.

Run with: python -m pytest tests/ -v
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.config import PipelineConfig
from src.pipeline import PipelineResult


# ── Config tests ─────────────────────────────────────────────────

class TestConfig:
    """Config should validate correctly and read from environment."""

    def test_default_config_requires_api_key(self):
        config = PipelineConfig()
        errors = config.validate()
        assert any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_skip_analysis_removes_api_key_requirement(self):
        config = PipelineConfig(skip_analysis=True)
        errors = config.validate()
        assert not any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_invalid_whisper_model_caught(self):
        config = PipelineConfig(skip_analysis=True, whisper_model="nonexistent")
        errors = config.validate()
        assert any("whisper model" in e.lower() for e in errors)

    def test_missing_instructions_file_caught(self):
        config = PipelineConfig(
            skip_analysis=True,
            instructions_file=Path("/tmp/does_not_exist.txt"),
        )
        errors = config.validate()
        assert any("not found" in e.lower() for e in errors)

    def test_from_env_reads_environment(self):
        with patch.dict(os.environ, {
            "LP_WHISPER_MODEL": "medium",
            "LP_KEEP_VIDEO": "true",
            "ANTHROPIC_API_KEY": "test-key",
        }):
            config = PipelineConfig.from_env()
            assert config.whisper_model == "medium"
            assert config.keep_video is True
            assert config.anthropic_api_key == "test-key"


# ── PipelineResult tests ────────────────────────────────────────

class TestPipelineResult:
    """PipelineResult should compute totals and format summaries."""

    def test_total_duration(self):
        result = PipelineResult(
            success=True,
            stage_durations={"download": 5.0, "convert": 2.0, "transcribe": 30.0},
        )
        assert result.total_duration == 37.0

    def test_summary_contains_status(self):
        result = PipelineResult(success=True, stage_durations={})
        assert "SUCCESS" in result.summary()

        result = PipelineResult(success=False, error="boom", stage_durations={})
        assert "FAILED" in result.summary()
        assert "boom" in result.summary()


# ── Converter tests ──────────────────────────────────────────────

class TestConverter:
    """Converter should call ffmpeg correctly."""

    @patch("src.converter.subprocess.run")
    def test_extract_audio_calls_ffmpeg(self, mock_run, tmp_path):
        from src.converter import extract_audio

        # Create a fake video file
        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"fake video content")

        # Mock ffmpeg success + create the expected output
        audio_path = tmp_path / "lecture.mp3"
        audio_path.write_bytes(b"fake audio")

        mock_run.return_value = MagicMock(returncode=0)

        result = extract_audio(video, keep_video=True)

        assert result.suffix == ".mp3"
        mock_run.assert_called_once()

        # Verify ffmpeg was called with -vn (no video)
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-vn" in cmd

    @patch("src.converter.subprocess.run")
    def test_extract_audio_raises_on_failure(self, mock_run, tmp_path):
        from src.converter import extract_audio, ConversionError

        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"fake")

        mock_run.return_value = MagicMock(returncode=1, stderr="codec error")

        with pytest.raises(ConversionError):
            extract_audio(video)


# ── Downloader tests ────────────────────────────────────────────

class TestDownloader:
    """Downloader should try yt-dlp first, then fall back to curl."""

    @patch("src.downloader._try_curl")
    @patch("src.downloader._try_ytdlp")
    def test_tries_ytdlp_first(self, mock_ytdlp, mock_curl, tmp_path):
        from src.downloader import download_video

        expected = tmp_path / "lecture.mp4"
        expected.write_bytes(b"video")
        mock_ytdlp.return_value = expected

        config = PipelineConfig(output_dir=tmp_path)
        result = download_video("https://example.com/video.mp4", config)

        assert result == expected
        mock_ytdlp.assert_called_once()
        mock_curl.assert_not_called()

    @patch("src.downloader._try_curl")
    @patch("src.downloader._try_ytdlp")
    def test_falls_back_to_curl(self, mock_ytdlp, mock_curl, tmp_path):
        from src.downloader import download_video

        mock_ytdlp.side_effect = RuntimeError("yt-dlp not found")

        expected = tmp_path / "lecture.mp4"
        expected.write_bytes(b"video")
        mock_curl.return_value = expected

        config = PipelineConfig(output_dir=tmp_path, download_retries=1)
        result = download_video("https://example.com/video.mp4", config)

        assert result == expected
