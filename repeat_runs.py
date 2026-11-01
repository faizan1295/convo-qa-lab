"""Repeat the live run N times and report a per-case pass rate with a 95% interval.

One run of a non-deterministic system is noise. Each repeat uses its own transcript folder
(transcripts_repeat/run_<i>/), so the main transcripts/ folder and its labels stay untouched.

Usage (demo app and Ollama must be running):
    python repeat_runs.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from ai_evaluator import evaluate_transcript
from config import EVALUATIONS_DIR, REPEAT_RESULTS_FILE, ROOT
from judge_benchmark import wilson
from report import build_dashboard

REPEAT_DIR = ROOT / "transcripts_repeat"
RESULTS_FILE = REPEAT_RESULTS_FILE


def summarize(runs: list[dict[str, str]]) -> list[dict]:
    """Per case: PASS/FAIL/INVALID counts. Rate = PASS / (PASS + FAIL); INVALID is a broken test."""
    names = sorted({name for run in runs for name in run})
    rows = []
    for name in names:
        verdicts = [run[name] for run in runs if name in run]
        passed, failed = verdicts.count("PASS"), verdicts.count("FAIL")
        valid = passed + failed
        lo, hi = wilson(passed, valid)
        rows.append(
            {
                "test_name": name,
                "pass": passed,
                "fail": failed,
                "invalid": verdicts.count("INVALID"),
                "crashed": verdicts.count("CRASHED"),
                "rate": passed / valid if valid else None,
                "ci95": [lo, hi],
            }
        )
    return rows


def format_summary(rows: list[dict], n_runs: int) -> str:
    lines = [
        f"Per-case pass rate over {n_runs} run(s) (INVALID and CRASHED excluded from the rate):",
        "",
    ]
    for r in sorted(rows, key=lambda r: (r["rate"] is None, r["rate"] or 0)):
        if r["rate"] is None:
            rate = "n/a"
        else:
            rate = f"{r['rate']:.0%} (95% CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})"
        counts = f"pass {r['pass']} fail {r['fail']} invalid {r['invalid']} crashed {r['crashed']}"
        lines.append(f"  {r['test_name']:36} {counts}  {rate}")
    return "\n".join(lines)


def failure_reasons(evaluation: dict) -> list[str]:
    """Why a case did not pass, tagged by layer: rule (code), judge (LLM) or test (broken case)."""
    if evaluation["verdict"] == "INVALID":
        return [
            f"test:goal_not_said {g['id']}"
            for g in evaluation["goal_coverage"]
            if g["status"] != "said"
        ]
    reasons = [f"rule:{v['rule']}" for v in evaluation["violations"]]
    reasons += [
        f"judge:{c['id']} {c['status']} - {c['statement']}"
        for c in evaluation["criteria"]
        if c["status"] != "pass"
    ]
    return reasons


def reason_counts(reasons_per_run: list[dict[str, list[str]]]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for run in reasons_per_run:
        for name, reasons in run.items():
            counts.setdefault(name, Counter()).update(set(reasons))
    return counts


def format_reasons(counts: dict[str, Counter], n_runs: int) -> str:
    lines = [f"Why cases did not pass (in how many of {n_runs} runs each reason appeared):", ""]
    for name in sorted(counts):
        if not counts[name]:
            continue
        lines.append(f"  {name}")
        lines += [f"      {n}x  {reason}" for reason, n in counts[name].most_common()]
    return "\n".join(lines)


def _run_once(index: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    folder = REPEAT_DIR / f"run_{index}"
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TRANSCRIPTS_DIR": str(folder)}
    subprocess.run(
        [sys.executable, "-m", "pytest", "e2e_assistant_runner.py", "-q"], env=env, check=False
    )
    verdicts, reasons = {}, {}
    for f in sorted(Path(folder).glob("*.json")):
        evaluation = evaluate_transcript(json.loads(f.read_text()))
        verdicts[evaluation["test_name"]] = evaluation["verdict"]
        reasons[evaluation["test_name"]] = failure_reasons(evaluation)
    for f in sorted((folder / "incomplete").glob("*.json")):
        verdicts.setdefault(json.loads(f.read_text())["test_case"]["test_name"], "CRASHED")
    return verdicts, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    EVALUATIONS_DIR.mkdir(exist_ok=True)
    runs: list[dict[str, str]] = []
    reasons: list[dict[str, list[str]]] = []
    for i in range(1, args.runs + 1):
        print(f"\n=== repeat run {i}/{args.runs} ===")
        verdicts, why = _run_once(i)
        runs.append(verdicts)
        reasons.append(why)
        data = {"runs": runs, "reasons": reasons, "summary": summarize(runs)}
        RESULTS_FILE.write_text(json.dumps(data, indent=2))
    print("\n" + format_summary(summarize(runs), len(runs)))
    print("\n" + format_reasons(reason_counts(reasons), len(runs)))
    print(f"\nSaved {RESULTS_FILE}\nDashboard: {build_dashboard()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
