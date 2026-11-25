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

    @patch("src.pipeline.get_transcriber")
    def test_video_input_runs_and_keeps_original(self, mock_transcribe, tmp_path):
        from src.pipeline import run

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video = input_dir / "lecture.mp4"
        video.write_bytes(b"fake video content")
        output_dir = tmp_path / "out"
        mock_transcribe.return_value.transcribe.return_value = ("transcript text", output_dir / "lecture.txt")

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
        mock_transcribe.assert_called_once_with(config)
        mock_transcribe.return_value.transcribe.assert_called_once_with(
            output_dir / "lecture.mp3", output_dir=output_dir
        )

    @patch("src.pipeline.get_transcriber")
    def test_audio_input_skips_ffmpeg(self, mock_transcribe, tmp_path):
        from src.pipeline import run

        audio = tmp_path / "lecture.m4a"
        audio.write_bytes(b"fake audio")
        mock_transcribe.return_value.transcribe.return_value = ("transcript text", tmp_path / "out" / "lecture.txt")

        with patch("src.converter.subprocess.run") as mock_run:
            result = run(audio, PipelineConfig(skip_analysis=True, output_dir=tmp_path / "out"))

        assert result.success is True
        assert result.audio_path == audio
        mock_run.assert_not_called()

    @patch("src.pipeline.get_transcriber")
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

def _fake_whisper(segments):
    """A stand-in for the `whisper` module whose model returns the given segments."""
    import types

    model = MagicMock()
    model.transcribe.return_value = {"text": " ".join(s["text"] for s in segments), "segments": segments}
    return types.SimpleNamespace(load_model=lambda name: model), model


class TestTranscriber:
    """Shared output format and the local Whisper backend."""

    def test_local_transcript_written_to_output_dir(self, tmp_path):
        import sys

        from src.transcriber import LocalWhisperTranscriber

        fake_whisper, model = _fake_whisper([
            {"start": 0.0, "end": 2.0, "text": "  hello  "},
            {"start": 65.4, "end": 70.0, "text": "lecture"},
        ])
        audio = tmp_path / "input" / "lecture.m4a"
        audio.parent.mkdir()
        audio.write_bytes(b"fake audio")
        output_dir = tmp_path / "out"

        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            text, path = LocalWhisperTranscriber("tiny").transcribe(audio, output_dir=output_dir)

        assert text == "[00:00:00] hello\n[00:01:05] lecture"
        assert path == output_dir / "lecture.txt"
        assert path.read_text(encoding="utf-8") == text + "\n"
        assert sorted(p.name for p in audio.parent.iterdir()) == ["lecture.m4a"]
        # Language is always passed explicitly, never auto-detected.
        assert model.transcribe.call_args.kwargs["language"] == "he"

    def test_output_one_segment_per_line_without_rtl_marks(self):
        from src.transcriber import format_transcript

        text = format_transcript([
            {"start": 0.0, "end": 1.0, "text": "‫שלום‬ עולם"},
            {"start": 3661.9, "end": 3663.0, "text": " ‏שורה\n  שנייה‎ "},
            {"start": 3664.0, "end": 3665.0, "text": "   "},  # empty segments are dropped
        ])

        assert text.splitlines() == [
            "[00:00:00] שלום עולם",
            "[01:01:01] שורה שנייה",
        ]
        bidi = "؜‎‏‪‫‬‭‮⁦⁧⁨⁩"
        assert not any(ch in text for ch in bidi)

    def test_empty_transcript_raises(self, tmp_path):
        import sys

        from src.transcriber import LocalWhisperTranscriber, TranscriptionError

        fake_whisper, _ = _fake_whisper([{"start": 0.0, "end": 1.0, "text": "  "}])
        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"fake audio")

        with patch.dict(sys.modules, {"whisper": fake_whisper}), pytest.raises(TranscriptionError):
            LocalWhisperTranscriber().transcribe(audio, output_dir=tmp_path)


class TestTranscriberFactory:
    """get_transcriber() should build the backend named in config."""

    def test_default_is_local(self):
        from src.transcriber import LocalWhisperTranscriber, get_transcriber

        transcriber = get_transcriber(PipelineConfig(whisper_model="small"))

        assert isinstance(transcriber, LocalWhisperTranscriber)
        assert transcriber.model_name == "small"

    def test_groq_backend(self):
        from src.groq_transcriber import GroqTranscriber
        from src.transcriber import get_transcriber

        config = PipelineConfig(
            transcriber_backend="groq", groq_api_key="gsk_test", groq_model="whisper-large-v3-turbo"
        )
        transcriber = get_transcriber(config)

        assert isinstance(transcriber, GroqTranscriber)
        assert transcriber.model == "whisper-large-v3-turbo"
        assert transcriber.language == "he"

    def test_unknown_backend_raises(self):
        from src.transcriber import TranscriptionError, get_transcriber

        with pytest.raises(TranscriptionError):
            get_transcriber(PipelineConfig(transcriber_backend="azure"))


class TestTranscriberConfig:
    """Backend selection is validated before the pipeline runs."""

    def test_groq_without_api_key_fails_validation(self):
        config = PipelineConfig(skip_analysis=True, transcriber_backend="groq", groq_api_key=None)
        errors = config.validate()
        assert any("GROQ_API_KEY" in e and "https://console.groq.com/keys" in e for e in errors)

    def test_groq_with_api_key_passes_validation(self):
        config = PipelineConfig(skip_analysis=True, transcriber_backend="groq", groq_api_key="gsk_test")
        assert config.validate() == []

    def test_invalid_backend_fails_validation(self):
        errors = PipelineConfig(skip_analysis=True, transcriber_backend="azure").validate()
        assert any("transcriber backend" in e for e in errors)

    def test_backend_read_from_environment(self):
        with patch.dict(os.environ, {
            "LP_TRANSCRIBER_BACKEND": "GROQ",
            "GROQ_API_KEY": "gsk_env",
            "LP_GROQ_MODEL": "whisper-large-v3-turbo",
        }):
            config = PipelineConfig.from_env()
        assert config.transcriber_backend == "groq"
        assert config.groq_api_key == "gsk_env"
        assert config.groq_model == "whisper-large-v3-turbo"


# ── Groq backend tests ───────────────────────────────────────────

def _fake_groq_module(client=None):
    """A stand-in for the `groq` SDK: the same exception hierarchy, no network."""
    import types

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        status_code = 0

        def __init__(self, message="error", headers=None):
            super().__init__(message)
            self.response = types.SimpleNamespace(headers=headers or {})

    class APIConnectionError(APIError):
        pass

    def status_error(name, code):
        return type(name, (APIStatusError,), {"status_code": code})

    return types.SimpleNamespace(
        Groq=MagicMock(return_value=client or MagicMock()),
        APIError=APIError,
        APIStatusError=APIStatusError,
        APIConnectionError=APIConnectionError,
        APITimeoutError=type("APITimeoutError", (APIConnectionError,), {}),
        AuthenticationError=status_error("AuthenticationError", 401),
        PermissionDeniedError=status_error("PermissionDeniedError", 403),
        BadRequestError=status_error("BadRequestError", 400),
        RateLimitError=status_error("RateLimitError", 429),
        InternalServerError=status_error("InternalServerError", 500),
    )


class TestGroqChunking:
    """Chunk timestamps must be shifted onto the full recording's timeline."""

    def test_offsets_are_cumulative_real_durations(self):
        from src.groq_transcriber import compute_offsets

        # Real chunk lengths drift from exactly 600s; offsets must follow the real values.
        assert compute_offsets([600.04, 599.98, 120.5]) == pytest.approx([0.0, 600.04, 1200.02])

    def test_offset_segments(self):
        from src.groq_transcriber import offset_segments

        shifted = offset_segments([{"start": 1.5, "end": 4.0, "text": "a"}], 600.0)
        assert shifted == [{"start": 601.5, "end": 604.0, "text": "a"}]

    def test_chunked_transcription_is_merged_with_offsets(self, tmp_path):
        import sys
        import types

        from src.groq_transcriber import GroqTranscriber

        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"fake audio")
        seen_tmp_dirs = []

        def fake_reencode(src, out):
            seen_tmp_dirs.append(out.parent)
            out.write_bytes(b"x" * 100)  # "too big" for the 50-byte limit below
            return out

        def fake_split(src, out_dir, chunk_seconds):
            chunks = [out_dir / f"chunk_{i:03d}.mp3" for i in range(3)]
            for c in chunks:
                c.write_bytes(b"chunk")
            return chunks

        durations = {"chunk_000.mp3": 600.04, "chunk_001.mp3": 599.98, "chunk_002.mp3": 42.0}
        # Each chunk's segment timestamps are relative to the start of that chunk.
        responses = [
            types.SimpleNamespace(text="a", segments=[{"start": 0.0, "end": 5.0, "text": "first"}]),
            types.SimpleNamespace(text="b", segments=[{"start": 2.0, "end": 6.0, "text": "second"}]),
            types.SimpleNamespace(text="c", segments=[{"start": 10.0, "end": 12.0, "text": "third"}]),
        ]
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = responses
        fake_groq = _fake_groq_module(client)

        with (
            patch.dict(sys.modules, {"groq": fake_groq}),
            patch("src.groq_transcriber._reencode_for_upload", side_effect=fake_reencode),
            patch("src.groq_transcriber._split_into_chunks", side_effect=fake_split),
            patch("src.groq_transcriber._probe_duration", side_effect=lambda p: durations[p.name]),
        ):
            segments = GroqTranscriber(api_key="gsk_test", max_upload_bytes=50).transcribe_segments(audio)

        assert [s["text"] for s in segments] == ["first", "second", "third"]
        assert [s["start"] for s in segments] == pytest.approx([0.0, 602.04, 1210.02])
        # Every request pins the language and deterministic decoding, and asks for segments.
        assert client.audio.transcriptions.create.call_count == 3
        for call in client.audio.transcriptions.create.call_args_list:
            assert call.kwargs["language"] == "he"
            assert call.kwargs["temperature"] == 0
            assert call.kwargs["response_format"] == "verbose_json"
        # The SDK's own retries are off, so ours is the only retry loop.
        fake_groq.Groq.assert_called_once_with(api_key="gsk_test", max_retries=0)
        # The temp directory (re-encoded audio + chunks) was removed.
        assert seen_tmp_dirs and not seen_tmp_dirs[0].exists()

    def test_small_file_is_uploaded_once_without_splitting(self, tmp_path):
        import sys
        import types

        from src.groq_transcriber import GroqTranscriber

        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"fake audio")
        client = MagicMock()
        client.audio.transcriptions.create.return_value = types.SimpleNamespace(
            text="hi", segments=[{"start": 0.0, "end": 1.0, "text": "hi"}]
        )

        def fake_reencode(src, out):
            out.write_bytes(b"small")
            return out

        with (
            patch.dict(sys.modules, {"groq": _fake_groq_module(client)}),
            patch("src.groq_transcriber._reencode_for_upload", side_effect=fake_reencode),
            patch("src.groq_transcriber._split_into_chunks") as mock_split,
        ):
            segments = GroqTranscriber(api_key="gsk_test").transcribe_segments(audio)

        assert segments == [{"start": 0.0, "end": 1.0, "text": "hi"}]
        mock_split.assert_not_called()
        client.audio.transcriptions.create.assert_called_once()


class TestGroqRetry:
    """Rate limits and transient errors are retried; auth errors are not."""

    def test_rate_limit_is_retried_with_backoff(self):
        from src.groq_transcriber import GroqTranscriber

        groq = _fake_groq_module()
        waits = []
        fn = MagicMock(side_effect=[groq.RateLimitError("429"), groq.RateLimitError("429"), ["ok"]])

        result = GroqTranscriber(api_key="k", sleep=waits.append)._call_with_retry(groq, fn)

        assert result == ["ok"]
        assert fn.call_count == 3
        assert waits == [2, 4]  # exponential backoff: 2s, 4s

    def test_retry_after_header_is_honored(self):
        from src.groq_transcriber import GroqTranscriber

        groq = _fake_groq_module()
        waits = []
        fn = MagicMock(side_effect=[groq.RateLimitError("429", headers={"retry-after": "30"}), ["ok"]])

        GroqTranscriber(api_key="k", sleep=waits.append)._call_with_retry(groq, fn)

        assert waits == [30]

    def test_network_errors_are_retried(self):
        from src.groq_transcriber import GroqTranscriber

        groq = _fake_groq_module()
        fn = MagicMock(side_effect=[groq.APITimeoutError("timeout"), groq.InternalServerError("503"), ["ok"]])

        assert GroqTranscriber(api_key="k", sleep=lambda s: None)._call_with_retry(groq, fn) == ["ok"]

    def test_gives_up_after_max_attempts(self):
        from src.groq_transcriber import GroqTranscriber
        from src.transcriber import TranscriptionError

        groq = _fake_groq_module()
        fn = MagicMock(side_effect=groq.RateLimitError("429"))

        with pytest.raises(TranscriptionError, match="after 3 attempts"):
            GroqTranscriber(api_key="k", max_attempts=3, sleep=lambda s: None)._call_with_retry(groq, fn)
        assert fn.call_count == 3

    def test_auth_error_raises_without_retry(self):
        from src.groq_transcriber import GroqTranscriber
        from src.transcriber import TranscriptionError

        groq = _fake_groq_module()
        sleep = MagicMock()
        fn = MagicMock(side_effect=groq.AuthenticationError("401 invalid key"))

        with pytest.raises(TranscriptionError, match="GROQ_API_KEY"):
            GroqTranscriber(api_key="bad", sleep=sleep)._call_with_retry(groq, fn)
        assert fn.call_count == 1
        sleep.assert_not_called()

    def test_other_client_errors_raise_without_retry(self):
        from src.groq_transcriber import GroqTranscriber
        from src.transcriber import TranscriptionError

        groq = _fake_groq_module()
        fn = MagicMock(side_effect=groq.BadRequestError("400 file too large"))

        with pytest.raises(TranscriptionError, match="400"):
            GroqTranscriber(api_key="k", sleep=lambda s: None)._call_with_retry(groq, fn)
        assert fn.call_count == 1


# ── Analyzer tests ───────────────────────────────────────────────

class TestAnalyzer:
    """Analyzer should parse Claude JSON, validate keys, and write outputs."""

    def _make_client(self, text: str, stop_reason: str = "end_turn", stop_details=None):
        """Fake client whose streamed reply looks like Sonnet 5.5's: an empty thinking block, then text."""
        import types

        response = types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="thinking", thinking=""),  # display defaults to "omitted"
                types.SimpleNamespace(type="text", text=text),
            ],
            stop_reason=stop_reason,
            stop_details=stop_details,
        )
        client = MagicMock()
        client.beta.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = response
        return client

    def _run(self, tmp_path, client):
        import sys
        import types

        from src.analyzer import analyze

        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: client)
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            config = PipelineConfig(anthropic_api_key="k", output_dir=tmp_path, instructions_file=tmp_path / "inst.txt")
            config.instructions_file.write_text("instructions")
            return analyze("transcript text", config, output_dir=tmp_path)

    _VALID = {
        "title": "T",
        "summary": "S",
        "topics": [],
        "review_questions": [{"question": "Q?", "answer": "A."}],
    }

    def test_request_uses_streaming_effort_and_refusal_fallback(self, tmp_path):
        import json

        client = self._make_client(json.dumps(self._VALID))
        self._run(tmp_path, client)

        kwargs = client.beta.messages.stream.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5-5"
        assert kwargs["max_tokens"] == 64000
        assert kwargs["output_config"] == {"effort": "low"}
        assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
        assert kwargs["extra_body"] == {"fallbacks": "default"}

    def test_refusal_raises_with_category(self, tmp_path):
        import types

        from src.analyzer import AnalysisError

        client = self._make_client("", stop_reason="refusal", stop_details=types.SimpleNamespace(category="cyber"))
        with pytest.raises(AnalysisError, match="refusal category: cyber"):
            self._run(tmp_path, client)
        assert not (tmp_path / "study_guide.json").exists()

    def test_truncated_response_raises(self, tmp_path):
        from src.analyzer import AnalysisError

        client = self._make_client('{"title": "cut off', stop_reason="max_tokens")
        with pytest.raises(AnalysisError, match="LP_CLAUDE_MAX_TOKENS"):
            self._run(tmp_path, client)

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
