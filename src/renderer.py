"""
Renderer — HTML output stage.

Takes the structured study guide JSON produced by the analyzer and
renders a polished, self-contained HTML file with no external dependencies.
"""

import html as _html_lib
import logging
import re
from pathlib import Path

logger = logging.getLogger("lecture_pipeline")


class RenderError(Exception):
    """Raised when HTML rendering fails."""


def render_html(parsed: dict, output_dir: Path) -> Path:
    """Render the parsed study guide JSON to a self-contained HTML file.

    Args:
        parsed: The structured study guide dict produced by the analyzer.
        output_dir: Directory where study_guide.html will be written.

    Returns:
        Path to the written HTML file.

    Raises:
        RenderError: If the title is missing or writing fails.
    """
    if not parsed.get("title"):
        raise RenderError("parsed study guide is missing required 'title' key")

    output_dir.mkdir(parents=True, exist_ok=True)

    content = _build_document(parsed)
    out_path = output_dir / "study_guide.html"
    try:
        out_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise RenderError(f"Failed to write HTML file: {e}")

    logger.info("HTML study guide saved: %s", out_path)
    return out_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _e(value) -> str:
    """HTML-escape a value; returns empty string for None."""
    if value is None:
        return ""
    return _html_lib.escape(str(value))


def _slug(text: str, index: int) -> str:
    """Build a URL-safe, index-unique section ID."""
    base = re.sub(r"\s+", "-", str(text).lower().strip())
    base = re.sub(r"[^\w-]", "", base).strip("-")
    return f"topic-{index}-{base}" if base else f"topic-{index}"


# ── TOC ───────────────────────────────────────────────────────────────────────

def _build_toc(parsed: dict, topic_info: list) -> str:
    """Render the sidebar table-of-contents <ul>."""
    lines = ["    <ul>"]

    def _li(href: str, label: str) -> str:
        return f'      <li><a href="#{href}">{_e(label)}</a></li>'

    if parsed.get("summary"):
        lines.append(_li("summary", "Summary"))
    for _, t_title, slug in topic_info:
        lines.append(_li(slug, t_title))
    if parsed.get("key_terms_glossary"):
        lines.append(_li("glossary", "Glossary"))
    if parsed.get("big_picture"):
        lines.append(_li("big-picture", "Big Picture"))
    if parsed.get("review_questions"):
        lines.append(_li("review-questions", "Review Questions"))
    if parsed.get("lecture_notes"):
        lines.append(_li("lecture-notes", "Lecture Notes"))

    lines.append("    </ul>")
    return "\n".join(lines)


# ── Section builders ──────────────────────────────────────────────────────────

def _render_subtopic(st: dict) -> list[str]:
    """Return HTML lines for one subtopic dict."""
    lines = ['      <div class="subtopic">']

    if st.get("title"):
        lines.append(f"        <h3>{_e(st['title'])}</h3>")
    if st.get("explanation"):
        lines.append(f"        <p>{_e(st['explanation'])}</p>")

    examples = st.get("examples") or []
    if examples:
        lines.append('        <p class="sub-label">Examples</p>')
        lines.append("        <ul>")
        for ex in examples:
            lines.append(f"          <li>{_e(ex)}</li>")
        lines.append("        </ul>")

    key_terms = st.get("key_terms") or []
    if key_terms:
        lines.append('        <p class="sub-label">Key Terms</p>')
        lines.append('        <ul class="key-terms-list">')
        for kt in key_terms:
            if isinstance(kt, dict):
                lines.append(
                    f'          <li><strong>{_e(kt.get("term"))}</strong>: {_e(kt.get("definition"))}</li>'
                )
            else:
                lines.append(f"          <li>{_e(kt)}</li>")
        lines.append("        </ul>")

    misconceptions = st.get("common_misconceptions") or []
    if misconceptions:
        lines.append('        <div class="misconceptions">')
        lines.append('          <p class="misconceptions-label">⚠️ Watch Out</p>')
        lines.append("          <ul>")
        for mc in misconceptions:
            lines.append(f"            <li>{_e(mc)}</li>")
        lines.append("          </ul>")
        lines.append("        </div>")

    connections = st.get("connections") or []
    if connections:
        lines.append('        <p class="sub-label">Connections</p>')
        lines.append("        <ul>")
        for c in connections:
            lines.append(f"          <li>{_e(c)}</li>")
        lines.append("        </ul>")

    lines.append("      </div>")
    return lines


def _section_topic(topic: dict, slug: str) -> list[str]:
    """Return HTML lines for one topic card section."""
    title = _e(topic.get("title") or topic.get("name") or "Topic")
    lines = [
        f'    <section id="{slug}" class="topic-card">',
        f"      <h2>{title}</h2>",
    ]
    if topic.get("overview"):
        lines.append(f'      <p class="topic-overview">{_e(topic["overview"])}</p>')
    for st in topic.get("subtopics") or []:
        if isinstance(st, dict):
            lines.extend(_render_subtopic(st))
    lines.append("    </section>")
    return lines


def _section_glossary(glossary) -> list[str]:
    """Return HTML lines for the glossary section, alphabetically sorted."""
    entries: list[tuple[str, str]] = []
    for entry in glossary:
        if isinstance(entry, dict):
            entries.append((entry.get("term") or "", entry.get("definition") or ""))
    entries.sort(key=lambda x: x[0].lower())

    lines = ['    <section id="glossary">', "      <h2>Glossary</h2>", "      <dl>"]
    for term, defn in entries:
        lines.append(f"        <dt>{_e(term)}</dt>")
        lines.append(f"        <dd>{_e(defn)}</dd>")
    lines += ["      </dl>", "    </section>"]
    return lines


def _section_review_questions(questions: list) -> list[str]:
    """Return HTML lines for the review questions section."""
    lines = ['    <section id="review-questions">', "      <h2>Review Questions</h2>", "      <ol>"]
    for rq in questions:
        if isinstance(rq, dict):
            q = rq.get("question") or rq.get("q") or ""
            a = rq.get("answer") or rq.get("a") or ""
        else:
            q, a = str(rq), ""
        lines += ["        <li>", "          <details>", f"            <summary>{_e(q)}</summary>"]
        if a:
            lines.append(f'            <div class="answer">{_e(a)}</div>')
        lines += ["          </details>", "        </li>"]
    lines += ["      </ol>", "    </section>"]
    return lines


# ── Document assembly ─────────────────────────────────────────────────────────

def _build_document(parsed: dict) -> str:
    """Assemble the complete self-contained HTML document."""
    title = parsed.get("title", "")
    course_context = parsed.get("course_context", "")
    summary = parsed.get("summary", "")
    topics = parsed.get("topics") or []
    if isinstance(topics, dict):
        topics = [topics]
    glossary = parsed.get("key_terms_glossary") or []
    big_picture = parsed.get("big_picture", "")
    review_questions = parsed.get("review_questions") or []
    lecture_notes = parsed.get("lecture_notes") or []

    topic_info = [
        (i, topic.get("title") or topic.get("name") or f"Topic {i + 1}",
         _slug(topic.get("title") or topic.get("name") or f"topic {i + 1}", i))
        for i, topic in enumerate(topics)
    ]

    toc = _build_toc(parsed, topic_info)

    body: list[str] = [f"    <h1>{_e(title)}</h1>"]
    if course_context:
        body.append(f'    <p class="course-context">{_e(course_context)}</p>')

    if summary:
        body += [
            '    <section id="summary">',
            "      <h2>Summary</h2>",
            f"      <p>{_e(summary)}</p>",
            "    </section>",
        ]

    for i, _, slug in topic_info:
        body.extend(_section_topic(topics[i], slug))

    if glossary:
        body.extend(_section_glossary(glossary))

    if big_picture:
        body += [
            '    <section id="big-picture">',
            "      <h2>Big Picture</h2>",
            f"      <p>{_e(big_picture)}</p>",
            "    </section>",
        ]

    if review_questions:
        body.extend(_section_review_questions(review_questions))

    if lecture_notes:
        body += ['    <section id="lecture-notes" class="lecture-notes">', "      <h2>Lecture Notes</h2>", "      <ul>"]
        for note in lecture_notes:
            body.append(f"        <li>{_e(note)}</li>")
        body += ["      </ul>", "    </section>"]

    doc = [
        "<!DOCTYPE html>",
        '<html lang="en" data-theme="light">',
        "<head>",
        '  <meta charset="UTF-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
        f"  <title>{_e(title)}</title>",
        "  <style>",
        _CSS,
        "  </style>",
        "</head>",
        "<body>",
        '  <button id="theme-toggle" title="Toggle light/dark mode">\U0001f319</button>',
        '  <nav id="sidebar">',
        "    <h2>Contents</h2>",
        toc,
        "  </nav>",
        '  <main id="content">',
        "\n".join(body),
        "  </main>",
        "  <script>",
        _JS,
        "  </script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(doc)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """\
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:            #ffffff;
      --bg-sec:        #f8f9fa;
      --text:          #212529;
      --muted:         #6c757d;
      --border:        #dee2e6;
      --accent:        #2563eb;
      --card-bg:       #ffffff;
      --card-border:   #e2e8f0;
      --shadow:        0 1px 4px rgba(0,0,0,.07);
      --warn-bg:       #fffbeb;
      --warn-border:   #f59e0b;
      --warn-text:     #78350f;
      --sidebar-bg:    #f8f9fa;
      --sidebar-w:     256px;
      --nav-active:    #2563eb;
      --nav-active-bg: rgba(37,99,235,.08);
    }

    [data-theme="dark"] {
      --bg:            #0f172a;
      --bg-sec:        #1e293b;
      --text:          #e2e8f0;
      --muted:         #94a3b8;
      --border:        #334155;
      --accent:        #60a5fa;
      --card-bg:       #1e293b;
      --card-border:   #334155;
      --shadow:        0 1px 4px rgba(0,0,0,.3);
      --warn-bg:       #431407;
      --warn-border:   #d97706;
      --warn-text:     #fcd34d;
      --sidebar-bg:    #1e293b;
      --nav-active:    #60a5fa;
      --nav-active-bg: rgba(96,165,250,.12);
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 16px;
      line-height: 1.6;
      color: var(--text);
      background: var(--bg);
      display: flex;
      min-height: 100vh;
      transition: background .2s, color .2s;
    }

    #sidebar {
      position: fixed;
      top: 0; left: 0;
      width: var(--sidebar-w);
      height: 100vh;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      overflow-y: auto;
      padding: 1.5rem 1rem;
      z-index: 100;
      transition: background .2s, border-color .2s;
    }
    #sidebar > h2 {
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: .75rem;
    }
    #sidebar ul { list-style: none; }
    #sidebar li { margin: .1rem 0; }
    #sidebar a {
      display: block;
      padding: .3rem .6rem;
      color: var(--muted);
      text-decoration: none;
      font-size: .85rem;
      border-radius: 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      transition: color .15s, background .15s;
    }
    #sidebar a:hover { color: var(--text); background: var(--border); }
    #sidebar a.active { color: var(--nav-active); font-weight: 600; background: var(--nav-active-bg); }
    #sidebar::-webkit-scrollbar { width: 4px; }
    #sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    #content {
      margin-left: var(--sidebar-w);
      max-width: 720px;
      padding: 2.5rem 2rem 4rem;
      width: 100%;
    }

    #theme-toggle {
      position: fixed;
      top: .9rem; right: 1rem;
      z-index: 200;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 50%;
      width: 2.25rem; height: 2.25rem;
      font-size: 1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow);
      transition: background .2s, border-color .2s;
    }
    #theme-toggle:hover { background: var(--bg-sec); }

    h1 { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; }
    h2 { font-size: 1.35rem; font-weight: 700; margin: 2rem 0 .6rem; }
    h3 { font-size: 1.1rem; font-weight: 600; margin: 1.25rem 0 .4rem; }
    p  { margin-bottom: .7rem; }
    ul, ol { padding-left: 1.5rem; margin-bottom: .7rem; }
    li { margin: .2rem 0; }

    section { scroll-margin-top: 1rem; margin-bottom: 1.5rem; }

    .course-context { color: var(--muted); font-size: .95rem; font-style: italic; margin-bottom: 1.75rem; }

    .topic-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 1.5rem 1.75rem;
      margin: 1.5rem 0;
      box-shadow: var(--shadow);
      transition: background .2s, border-color .2s;
    }
    .topic-card > h2 { margin-top: 0; padding-bottom: .6rem; border-bottom: 2px solid var(--border); }
    .topic-overview { color: var(--muted); margin-top: .5rem; }

    .subtopic { margin-top: 1.25rem; }

    .sub-label {
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: .9rem 0 .3rem;
    }

    .key-terms-list { list-style: none; padding-left: 0; }
    .key-terms-list li { margin: .25rem 0; }
    .key-terms-list li strong { color: var(--accent); }

    .misconceptions {
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-border);
      padding: .75rem 1rem;
      margin: .75rem 0;
      border-radius: 0 6px 6px 0;
      color: var(--warn-text);
      transition: background .2s;
    }
    .misconceptions-label { font-weight: 700; margin-bottom: .25rem; }
    .misconceptions ul { margin-bottom: 0; }

    dl { margin: .5rem 0; }
    dt { font-weight: 700; margin-top: .75rem; }
    dd { margin-left: 1.5rem; margin-bottom: .2rem; }

    details { margin: .6rem 0; }
    summary {
      cursor: pointer;
      font-weight: 500;
      padding: .45rem 0;
      list-style: none;
      display: flex;
      align-items: baseline;
      gap: .5rem;
    }
    summary::-webkit-details-marker { display: none; }
    summary::before { content: "\25b6"; font-size: .65em; color: var(--muted); flex-shrink: 0; }
    details[open] summary::before { content: "\25bc"; }

    .answer {
      background: var(--bg-sec);
      border-left: 3px solid var(--accent);
      padding: .75rem 1rem;
      margin-top: .4rem;
      border-radius: 0 6px 6px 0;
      font-size: .95rem;
      transition: background .2s;
    }

    .lecture-notes ul { list-style: disc; }"""


# ── JavaScript ────────────────────────────────────────────────────────────────

_JS = """\
    // Highlight the active TOC section via IntersectionObserver
    (function () {
      var links = document.querySelectorAll('#sidebar a[href^="#"]');
      var map = new Map();
      links.forEach(function (a) {
        var el = document.getElementById(a.getAttribute('href').slice(1));
        if (el) map.set(el, a);
      });
      var active = null;
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            if (active) active.classList.remove('active');
            active = map.get(e.target) || null;
            if (active) active.classList.add('active');
          }
        });
      }, { threshold: 0, rootMargin: '0px 0px -60% 0px' });
      map.forEach(function (a, el) { obs.observe(el); });
    }());

    // Light/dark mode toggle
    (function () {
      var btn = document.getElementById('theme-toggle');
      btn.addEventListener('click', function () {
        var html = document.documentElement;
        var next = html.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        html.setAttribute('data-theme', next);
        btn.textContent = next === 'dark' ? '☀️' : '\U0001f319';
      });
    }());"""
