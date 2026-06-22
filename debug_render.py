# debug_render.py  (delete after use)
import json
from pathlib import Path
from src.renderer import render_html

raw = Path("~/lecture_pipeline_output/study_guide_raw.txt").expanduser().read_text(encoding="utf-8")

# Strip the fences manually to test
clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
parsed = json.loads(clean)

output_dir = Path("~/lecture_pipeline_output").expanduser()
render_html(parsed, output_dir)

print("Done. Open:", output_dir / "study_guide.html")