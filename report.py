"""Build a self-contained HTML dashboard from the recorded evaluation runs.

Usage:
    python report.py            # writes reports/index.html
    python report.py --open     # ... and opens it in the browser

ai_evaluator.py appends one record per run to evaluations/runs.jsonl and calls this
automatically, so the dashboard always shows the full history.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from config import (
    EVALUATIONS_DIR,
    JUDGE_RESULTS_FILE,
    REPEAT_RESULTS_FILE,
    REPORTS_DIR,
    RUNS_FILENAME,
)

TEMPLATE = Path(__file__).resolve().parent / "report_template.html"
PLACEHOLDER = "__RUNS_JSON__"


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path: Path | None) -> dict | None:
    return json.loads(path.read_text()) if path and path.exists() else None


def render_report(runs: list[dict], repeat: dict | None = None, judge: dict | None = None) -> str:
    # Transcript text is model output: escape "<" so it can never close the <script> tag.
    data = json.dumps({"runs": runs, "repeat": repeat, "judge": judge}).replace("<", "\\u003c")
    return TEMPLATE.read_text().replace(PLACEHOLDER, data)


def build_report(
    runs_file: Path,
    out_file: Path,
    repeat_file: Path | None = None,
    judge_file: Path | None = None,
) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    html = render_report(load_runs(runs_file), load_json(repeat_file), load_json(judge_file))
    out_file.write_text(html)
    return out_file


def build_dashboard() -> Path:
    """The dashboard from the project's real result files (runs, repeat runs, judge vs human)."""
    return build_report(
        EVALUATIONS_DIR / RUNS_FILENAME,
        REPORTS_DIR / "index.html",
        REPEAT_RESULTS_FILE,
        JUDGE_RESULTS_FILE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="open the report in the browser")
    args = parser.parse_args()

    out = build_dashboard()
    print(f"Report written to {out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
