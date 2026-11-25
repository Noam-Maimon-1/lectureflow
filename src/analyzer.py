"""
Analyzer — the LOAD stage of the pipeline.

Sends the processed transcript to Claude's API along with
study instructions, and saves the resulting study guide.

This is "Load" because we're delivering transformed data to
the system that will consume it (the LLM). The output is the
final deliverable: a structured study guide.
"""

import json
import logging
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger("lecture_pipeline")

# Bundled with the project — user can override via --instructions
DEFAULT_INSTRUCTIONS_PATH = Path(__file__).parent.parent / "instructions" / "default.txt"


class AnalysisError(Exception):
    """Raised when the Claude API call fails."""


def analyze(transcript: str, config: PipelineConfig, output_dir: Path) -> tuple[Path, dict]:
    """Send transcript to Claude with study instructions, save the result.

    Args:
        transcript: The lecture transcript text.
        config: Pipeline configuration (API key, model, etc.).
        output_dir: Where to save the study guide.

    Returns:
        Tuple of (path to the saved markdown file, parsed study guide dict).

    Raises:
        AnalysisError: If the API call fails.
    """
    instructions = _load_instructions(config)

    logger.info(
        "Sending %d words to Claude (%s)...",
        len(transcript.split()),
        config.claude_model,
    )

    try:
        import anthropic
    except ImportError as e:
        raise AnalysisError("Anthropic SDK not installed. Run: pip install anthropic") from e

    try:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        # Streaming: thinking counts toward max_tokens, so the budget is large (64K by default),
        # and long non-streaming requests risk HTTP timeouts.
        with client.beta.messages.stream(
            model=config.claude_model,
            max_tokens=config.claude_max_tokens,
            output_config={"effort": config.claude_effort},
            # If a safety classifier declines (e.g. a security lecture flagged as "cyber"),
            # the API retries on a fallback model inside the same request.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{instructions}\n\n"
                        f"--- LECTURE TRANSCRIPT ---\n\n"
                        f"{transcript}"
                    ),
                }
            ],
        ) as stream:
            message = stream.get_final_message()
    except Exception as e:
        raise AnalysisError(f"Claude API call failed: {e}") from e

    # Check why the model stopped before reading the content.
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        raise AnalysisError(f"Claude declined to process this transcript (refusal category: {category}).")
    if message.stop_reason == "max_tokens":
        raise AnalysisError(
            f"Claude's response was cut off at max_tokens={config.claude_max_tokens}. "
            "Raise LP_CLAUDE_MAX_TOKENS and run again."
        )

    # Read text blocks by type: the response can start with a (possibly empty) thinking block.
    study_guide_text = "".join(block.text for block in message.content if block.type == "text")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to parse Claude's response as JSON per new instructions
    cleaned = study_guide_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[cleaned.index("\n") + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:cleaned.rindex("```")]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Save raw text for debugging, log and raise
        raw_path = output_dir / "study_guide_raw.txt"
        raw_path.write_text(study_guide_text, encoding="utf-8")
        logger.warning("Claude returned malformed JSON; saved raw output: %s", raw_path)
        raise AnalysisError(f"Failed to parse Claude JSON response: {e}") from e

    # Validate required top-level keys
    required = ["title", "summary", "topics", "review_questions"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise AnalysisError(f"Claude response missing required keys: {', '.join(missing)}")

    # Save the raw parsed JSON
    json_path = output_dir / "study_guide.json"
    json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render a human-readable markdown fallback
    md = _render_markdown(parsed)
    md_path = output_dir / "study_guide.md"
    md_path.write_text(md, encoding="utf-8")

    logger.info("Study guide saved: %s and %s", json_path, md_path)
    return md_path, parsed


def _render_markdown(obj: dict) -> str:
    """Render the structured study guide JSON into readable Markdown."""
    parts: list[str] = []

    title = obj.get("title")
    if title:
        parts.append(f"# {title}\n")

    if obj.get("course_context"):
        parts.append(f"**Course Context:** {obj.get('course_context')}\n")

    if obj.get("summary"):
        parts.append("## Summary\n")
        parts.append(obj.get("summary") + "\n")

    # Topics: expect a list of topic dicts
    topics = obj.get("topics")
    if topics:
        parts.append("## Topics\n")
        if isinstance(topics, dict):
            topics = [topics]
        for topic in topics:
            title = topic.get("title") or topic.get("name") or "Topic"
            parts.append(f"### {title}\n")

            if topic.get("subtopics"):
                for st in topic.get("subtopics"):
                    if not isinstance(st, dict):
                        continue
                    parts.append(f"#### {st.get('title', '')}\n")
                    if st.get("explanation"):
                        parts.append(st.get("explanation") + "\n")
                    if st.get("examples"):
                        parts.append("**Examples:**\n")
                        for ex in st.get("examples"):
                            parts.append(f"- {ex}")
                        parts.append("\n")
                    if st.get("key_terms"):
                        parts.append("**Key Terms:**\n")
                        for kt in st.get("key_terms"):
                            parts.append(f"- **{kt.get('term')}**: {kt.get('definition')}")
                        parts.append("\n")
                    if st.get("common_misconceptions"):
                        parts.append("**Watch Out:**\n")
                        for cm in st.get("common_misconceptions"):
                            parts.append(f"- {cm}")
                        parts.append("\n")
                    if st.get("connections"):
                        parts.append("**Connections:**\n")
                        for c in st.get("connections"):
                            parts.append(f"- {c}")
                        parts.append("\n")

            if topic.get("key_terms"):
                parts.append("**Key terms:**\n")
                parts.append(", ".join(topic.get("key_terms")) + "\n")

            if topic.get("examples"):
                parts.append("**Examples:**\n")
                for ex in topic.get("examples"):
                    parts.append(f"- {ex}")
                parts.append("\n")

            if topic.get("common_misconceptions"):
                parts.append("**Common misconceptions:**\n")
                for cm in topic.get("common_misconceptions"):
                    parts.append(f"- {cm}")
                parts.append("\n")

            if topic.get("connections"):
                parts.append("**Connections:**\n")
                for c in topic.get("connections"):
                    parts.append(f"- {c}")
                parts.append("\n")

    # Glossary
    glossary = obj.get("key_terms_glossary")
    if glossary:
        parts.append("## Glossary\n")
        for entry in glossary:
            if isinstance(entry, dict):
                parts.append(f"- **{entry.get('term')}**: {entry.get('definition')}")
        parts.append("\n")

    if obj.get("big_picture"):
        parts.append("## Big Picture\n")
        parts.append(obj.get("big_picture") + "\n")

    if obj.get("lecture_notes"):
        parts.append("## Lecture Notes\n")
        for note in obj.get("lecture_notes"):
            parts.append(f"- {note}")
        parts.append("\n")

    # Review questions: numbered list with answers
    review = obj.get("review_questions")
    if review:
        parts.append("## Review Questions\n")
        for i, rq in enumerate(review, start=1):
            if isinstance(rq, dict):
                q = rq.get("question") or rq.get("q") or "Question"
                a = rq.get("answer") or rq.get("a") or ""
            else:
                q = str(rq)
                a = ""
            parts.append(f"{i}. {q}")
            if a:
                parts.append(f"\n   **Answer:** {a}")
        parts.append("\n")

    return "\n".join(parts)


def _load_instructions(config: PipelineConfig) -> str:
    """Load study instructions from file — user-provided or default."""
    if config.instructions_file and config.instructions_file.exists():
        logger.info("Using custom instructions: %s", config.instructions_file)
        return config.instructions_file.read_text(encoding="utf-8")

    if DEFAULT_INSTRUCTIONS_PATH.exists():
        logger.debug("Using default instructions")
        return DEFAULT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")

    # Hardcoded fallback (shouldn't happen if project is intact)
    logger.warning("No instructions file found, using minimal fallback")
    return "Summarize this lecture and explain the key concepts clearly."
