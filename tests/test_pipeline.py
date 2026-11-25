"""
Tests for the lecture pipeline.

These test individual stages in isolation — each module should be
testable without running the full pipeline or making external calls.

Run with: python -m pytest tests/ -v
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            "LP_SKIP_HTML": "true",
            "ANTHROPIC_API_KEY": "test-key",
        }):
            config = PipelineConfig.from_env()
            assert config.whisper_model == "medium"
            assert config.skip_html is True
            assert config.anthropic_api_key == "test-key"


# ── PipelineResult tests ────────────────────────────────────────

class TestPipelineResult:
    """PipelineResult should compute totals and format summaries."""

    def test_total_duration(self):
        result = PipelineResult(
            success=True,
            stage_durations={"convert": 2.0, "transcribe": 30.0, "analyze": 5.0},
        )
        assert result.total_duration == 37.0

    def test_summary_contains_status(self):
        result = PipelineResult(success=True, stage_durations={})
        assert "SUCCESS" in result.summary()

        result = PipelineResult(success=False, error="boom", stage_durations={})
        assert "FAILED" in result.summary()
        assert "boom" in result.summary()


# ── Pipeline orchestration tests ───────────────────────────────────

def _fake_ffmpeg(cmd, **kwargs):
    """Stand-in for subprocess.run: writes the output file ffmpeg would create."""
    Path(cmd[-1]).write_bytes(b"fake audio")
    return MagicMock(returncode=0)


class TestPipelineRun:
    """The pipeline runs from a user-provided file and never touches that file."""

    @patch("src.pipeline.transcribe")
    def test_video_input_runs_and_keeps_original(self, mock_transcribe, tmp_path):
        from src.pipeline import run

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video = input_dir / "lecture.mp4"
        video.write_bytes(b"fake video content")
        output_dir = tmp_path / "out"
        mock_transcribe.return_value = ("transcript text", output_dir / "lecture.txt")

        with patch("src.converter.subprocess.run", side_effect=_fake_ffmpeg):
            config = PipelineConfig(skip_analysis=True, output_dir=output_dir)
            result = run(video, config)

        assert result.success is True
        assert result.input_path == video
        assert result.audio_path == output_dir / "lecture.mp3"
        assert list(result.stage_durations) == ["convert", "transcribe"]
        # The user's file is untouched and nothing was written next to it.
        assert video.read_bytes() == b"fake video content"
        assert sorted(p.name for p in input_dir.iterdir()) == ["lecture.mp4"]
        mock_transcribe.assert_called_once_with(
            output_dir / "lecture.mp3", model_name="base", output_dir=output_dir
        )

    @patch("src.pipeline.transcribe")
    def test_audio_input_skips_ffmpeg(self, mock_transcribe, tmp_path):
        from src.pipeline import run

        audio = tmp_path / "lecture.m4a"
        audio.write_bytes(b"fake audio")
        mock_transcribe.return_value = ("transcript text", tmp_path / "out" / "lecture.txt")

        with patch("src.converter.subprocess.run") as mock_run:
            result = run(audio, PipelineConfig(skip_analysis=True, output_dir=tmp_path / "out"))

        assert result.success is True
        assert result.audio_path == audio
        mock_run.assert_not_called()

    @patch("src.pipeline.transcribe")
    def test_missing_input_fails_at_convert(self, mock_transcribe, tmp_path):
        from src.pipeline import run

        result = run(tmp_path / "nope.mp4", PipelineConfig(skip_analysis=True, output_dir=tmp_path))

        assert result.success is False
        assert "not found" in result.error
        mock_transcribe.assert_not_called()


# ── Converter tests ──────────────────────────────────────────────

class TestConverter:
    """Converter should call ffmpeg correctly and write into the output directory."""

    @patch("src.converter.subprocess.run", side_effect=_fake_ffmpeg)
    def test_extract_audio_calls_ffmpeg(self, mock_run, tmp_path):
        from src.converter import extract_audio

        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"fake video content")
        output_dir = tmp_path / "out"

        result = extract_audio(video, output_dir)

        assert result == output_dir / "lecture.mp3"
        assert video.exists()
        mock_run.assert_called_once()

        # Verify ffmpeg was called with -vn (no video)
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-vn" in cmd

    @patch("src.converter.subprocess.run")
    def test_extract_audio_skips_ffmpeg_for_mp3_input(self, mock_run, tmp_path):
        from src.converter import extract_audio

        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"fake audio content")

        result = extract_audio(audio, tmp_path / "out")

        assert result == audio
        mock_run.assert_not_called()

    @patch("src.converter.subprocess.run")
    def test_extract_audio_raises_on_failure(self, mock_run, tmp_path):
        from src.converter import ConversionError, extract_audio

        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"fake")

        mock_run.return_value = MagicMock(returncode=1, stderr="codec error")

        with pytest.raises(ConversionError):
            extract_audio(video, tmp_path)

    @patch("src.converter.subprocess.run", side_effect=FileNotFoundError)
    def test_extract_audio_reports_missing_ffmpeg(self, mock_run, tmp_path):
        from src.converter import ConversionError, extract_audio

        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"fake")

        with pytest.raises(ConversionError, match="ffmpeg not found"):
            extract_audio(video, tmp_path)


# ── Transcriber tests ────────────────────────────────────────────

class TestTranscriber:
    """Transcriber should save the transcript to the output directory."""

    def test_transcript_written_to_output_dir(self, tmp_path):
        import sys
        import types

        from src.transcriber import transcribe

        model = MagicMock()
        model.transcribe.return_value = {"text": "  hello lecture  "}
        fake_whisper = types.SimpleNamespace(load_model=lambda name: model)

        audio = tmp_path / "input" / "lecture.m4a"
        audio.parent.mkdir()
        audio.write_bytes(b"fake audio")
        output_dir = tmp_path / "out"

        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            text, path = transcribe(audio, model_name="tiny", output_dir=output_dir)

        assert text == "hello lecture"
        assert path == output_dir / "lecture.txt"
        assert path.read_text(encoding="utf-8") == "hello lecture"
        assert sorted(p.name for p in audio.parent.iterdir()) == ["lecture.m4a"]


# ── Analyzer tests ───────────────────────────────────────────────

class TestAnalyzer:
    """Analyzer should parse Claude JSON, validate keys, and write outputs."""

    def _make_client(self, text: str):
        from unittest.mock import MagicMock

        response = MagicMock()
        response.content = [MagicMock(text=text)]

        client = MagicMock()
        messages = MagicMock()
        messages.create.return_value = response
        client.messages = messages
        return client

    def test_analyzer_valid_json(self, tmp_path):
        from src.analyzer import analyze

        # Prepare a valid JSON response with required keys
        payload = {
            "title": "Intro to Testing",
            "summary": "This lecture covers testing basics.",
            "topics": [{"title": "Unit tests", "subtopics": ["what", "why"]}],
            "review_questions": [{"question": "What is a unit test?", "answer": "A small test."}],
            "key_terms_glossary": {"unit test": "isolated test"},
        }

        import sys
        import types
        client = self._make_client(text=__import__("json").dumps(payload))
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: client)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            config = PipelineConfig(anthropic_api_key="k", output_dir=tmp_path, instructions_file=tmp_path / "inst.txt")
            (config.instructions_file).write_text("instructions")
            md_path, parsed = analyze("transcript text", config, output_dir=tmp_path)

        assert md_path == tmp_path / "study_guide.md"
        assert parsed["title"] == "Intro to Testing"
        assert (tmp_path / "study_guide.json").exists()
        assert (tmp_path / "study_guide.md").exists()
        text = (tmp_path / "study_guide.md").read_text(encoding="utf-8")
        assert "# Intro to Testing" in text
        assert "1. What is a unit test?" in text
        assert "Answer" in text

    def test_analyzer_missing_keys(self, tmp_path):
        from src.analyzer import AnalysisError, analyze

        payload = {"title": "No Summary"}  # missing summary, topics, review_questions

        import sys
        import types
        client = self._make_client(text=__import__("json").dumps(payload))
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: client)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            config = PipelineConfig(anthropic_api_key="k", output_dir=tmp_path, instructions_file=tmp_path / "inst.txt")
            (config.instructions_file).write_text("instructions")
            with pytest.raises(AnalysisError) as exc:
                analyze("t", config, output_dir=tmp_path)

        assert "missing required keys" in str(exc.value)

    def test_analyzer_malformed_json(self, tmp_path):
        from src.analyzer import AnalysisError, analyze

        bad_text = "{ not valid json..."

        import sys
        import types
        client = self._make_client(text=bad_text)
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: client)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            config = PipelineConfig(anthropic_api_key="k", output_dir=tmp_path, instructions_file=tmp_path / "inst.txt")
            (config.instructions_file).write_text("instructions")
            with pytest.raises(AnalysisError):
                analyze("t", config, output_dir=tmp_path)

        # Raw file should be saved
        assert (tmp_path / "study_guide_raw.txt").exists()

    def test_analyzer_strips_code_fences(self, tmp_path):
        from src.analyzer import analyze

        payload = {
            "title": "Fenced JSON Lecture",
            "summary": "Testing code fence stripping.",
            "topics": [{"title": "Code Fences", "subtopics": []}],
            "review_questions": [{"question": "What are fences?", "answer": "Backtick wrappers."}],
        }

        import json
        import sys
        import types
        fenced_text = "```json\n" + json.dumps(payload) + "\n```"
        client = self._make_client(text=fenced_text)
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: client)

        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            config = PipelineConfig(anthropic_api_key="k", output_dir=tmp_path, instructions_file=tmp_path / "inst.txt")
            (config.instructions_file).write_text("instructions")
            _, parsed = analyze("transcript text", config, output_dir=tmp_path)

        assert parsed["title"] == "Fenced JSON Lecture"
        assert (tmp_path / "study_guide.json").exists()
        assert (tmp_path / "study_guide.md").exists()


# ── Renderer tests ───────────────────────────────────────────────────────────

class TestRenderer:
    """Renderer should produce a self-contained HTML file from parsed JSON."""

    def test_renderer_creates_html_file(self, tmp_path):
        from src.renderer import render_html

        parsed = {
            "title": "My Test Lecture",
            "summary": "A brief summary.",
            "topics": [],
            "review_questions": [],
        }

        path = render_html(parsed, tmp_path)

        assert path.exists()
        assert path.suffix == ".html"
        text = path.read_text(encoding="utf-8")
        assert "My Test Lecture" in text

    def test_renderer_missing_title_raises(self, tmp_path):
        from src.renderer import RenderError, render_html

        with pytest.raises(RenderError):
            render_html({}, tmp_path)
