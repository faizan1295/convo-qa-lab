"""Quality gate for recorded conversations: deterministic checks + per-criterion LLM judge.

Usage:
    python ai_evaluator.py transcripts/some_test.json
    python ai_evaluator.py transcripts/            # evaluate every transcript

Exit code is 0 when every transcript passes, 1 otherwise (CI friendly).

Before judging the assistant, the test itself is checked: did the simulated user really say
every one of the case's user_goals? If not, the case is INVALID (a broken test, not a model
failure) and the assistant is not judged. INVALID never counts as PASS or FAIL.

A valid transcript PASSES only if
  1. behavior_checks.py finds no rule violation (no LLM involved), and
  2. every expected_behavior criterion is judged to hold, with a quote that really appears in
     the transcript, and
  3. no must_not / universal criterion is judged violated.
Two kinds of criteria, because absence cannot be quoted:
  - expected ("holds"): positive proof needed. holds=true without a verifiable quote is
    "unverified" and fails the case.
  - violation ("violated"): written as the forbidden action itself (no double negatives).
    violated=true needs a verifiable quote of the offending line, else "unverified" (fails);
    violated=false needs no quote. This side is weaker by design, which is why the
    deterministic rules and forbidden_phrases carry the hard guarantees.
The verdict is computed here in code, never by the judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from behavior_checks import check_case, violations_as_dicts
from config import (
    EVALUATIONS_DIR,
    JUDGE_RESULTS_FILE,
    REPEAT_FILENAME,
    REPORTS_DIR,
    RUNS_FILENAME,
    load_settings,
)
from llm import complete_json
from report import build_report

MIN_QUOTE_CHARS = 8
JUDGE_TEMPERATURE = 0

UNIVERSAL_CRITERIA = (
    "Is rude, wordy or changes tone abruptly in a way that does not suit the persona.",
    "Forgets, or asks again for, information the user already gave in earlier turns.",
    "Invents specifics it cannot know (addresses, distances, times, directions).",
)

VIOLATION_KINDS = ("must_not", "universal")

SYSTEM_PROMPT = """You are a strict QA judge for conversational assistants. \
Check each criterion against the transcript and respond with JSON only."""

EVAL_PROMPT = """Test case:
- Description: {description}
- User persona: {persona}

Transcript:
{transcript}

There are two kinds of criteria, judged only on what is literally in the transcript.

EXPECTED criteria (ids starting with E) describe something the ASSISTANT should have done. \
Do not give credit for behaviour that did not happen. Return:
- "holds": true or false
- "quote": an exact, word-for-word excerpt copied from the transcript that proves it. A \
"holds": true answer without a real quote is discarded.

VIOLATION criteria (ids starting with N or U) describe something bad the ASSISTANT must \
not do. Return:
- "violated": true if the assistant did it at any point, otherwise false
- "quote": if violated, the exact, word-for-word assistant excerpt that shows it; otherwise ""

Criteria:
{criteria}

Return JSON:
{{"results": [{{"id": "E1", "holds": true, "quote": "..."}}, \
{{"id": "N1", "violated": false, "quote": ""}}]}}"""


GOAL_PROMPT = """Below are ONLY the messages a simulated user sent to an assistant.

User messages:
{user_lines}

Goals the user was supposed to express during the conversation:
{goals}

For every goal decide whether the user actually expressed it in the messages above, in any \
wording. Judge only what is literally there.

For each goal return:
- "said": true or false
- "quote": an exact, word-for-word excerpt copied from the user messages that expresses the \
goal. A "said": true answer without a real quote is discarded.

Return JSON:
{{"results": [{{"id": "G1", "said": true, "quote": "..."}}]}}"""


def build_criteria(case: dict) -> list[dict]:
    """Turn a test case into an ordered list of checkable statements with stable ids."""
    if "expected_behavior" not in case:
        raise ValueError(
            f"Test case '{case.get('test_name')}' has no expected_behavior. "
            "Regenerate test cases with test_generator.py and re-run the runner."
        )
    criteria = [
        {"id": f"E{i}", "kind": "expected", "statement": s}
        for i, s in enumerate(case["expected_behavior"], 1)
    ]
    criteria += [
        {"id": f"N{i}", "kind": "must_not", "statement": s}
        for i, s in enumerate(case.get("must_not", []), 1)
    ]
    criteria += [
        {"id": f"U{i}", "kind": "universal", "statement": s}
        for i, s in enumerate(UNIVERSAL_CRITERIA, 1)
    ]
    return criteria


def _normalize(text: str) -> str:
    return " ".join(text.replace("’", "'").lower().split())


def _quote_is_real(quote: object, transcript: str) -> bool:
    if not isinstance(quote, str):
        return False
    cleaned = _normalize(quote).strip(" \"'.…")
    cleaned = cleaned.removeprefix("...").removesuffix("...").strip()
    return len(cleaned) >= MIN_QUOTE_CHARS and cleaned in _normalize(transcript)


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _grade_expected(holds: bool | None, quote: object, transcript: str) -> tuple[str, str]:
    if holds is None:
        return "unverified", "judge gave no true/false answer"
    if not holds:
        return "fail", ""
    if not _quote_is_real(quote, transcript):
        return "unverified", "claimed to hold, but its quote is not in the transcript"
    return "pass", ""


def _grade_violation(violated: bool | None, quote: object, transcript: str) -> tuple[str, str]:
    if violated is None:
        return "unverified", "judge gave no true/false answer"
    if not violated:
        return "pass", ""
    if not _quote_is_real(quote, transcript):
        return "unverified", "claimed a violation, but its quote is not in the transcript"
    return "fail", ""


def grade_criteria(criteria: list[dict], judge_results: object, transcript: str) -> list[dict]:
    """Attach status pass/fail/unverified to each criterion. Unverifiable answers never pass.

    Expected criteria need holds=true plus a real quote. Violation criteria fail on a quoted
    violation, pass on violated=false, and are unverified on anything else.
    """
    by_id = (
        {r.get("id"): r for r in judge_results if isinstance(r, dict)}
        if isinstance(judge_results, list)
        else {}
    )
    graded = []
    for c in criteria:
        result = by_id.get(c["id"], {})
        quote = result.get("quote", "")
        if c["kind"] in VIOLATION_KINDS:
            status, reason = _grade_violation(_as_bool(result.get("violated")), quote, transcript)
        else:
            status, reason = _grade_expected(_as_bool(result.get("holds")), quote, transcript)
        graded.append({**c, "status": status, "quote": quote, "reason": reason})
    return graded


def check_goal_coverage(case: dict, turns: list[dict]) -> list[dict]:
    """'Test the test': was each user_goal really said by the simulated user?

    Status is said / not_said / unverified. Fail-closed like the criteria: an unverifiable
    answer is not coverage, so the case is INVALID rather than silently trusted.
    """
    goals = case.get("user_goals", [])
    if not goals:
        return []
    user_lines = "\n".join(f"[turn {t['turn']}] {t['text']}" for t in turns if t["role"] == "user")
    payload = complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GOAL_PROMPT.format(
                    user_lines=user_lines,
                    goals="\n".join(f"G{i}: {g}" for i, g in enumerate(goals, 1)),
                ),
            },
        ],
        temperature=0,
    )
    results = payload.get("results")
    by_id = (
        {r.get("id"): r for r in results if isinstance(r, dict)}
        if isinstance(results, list)
        else {}
    )
    coverage = []
    for i, goal in enumerate(goals, 1):
        result = by_id.get(f"G{i}", {})
        said = _as_bool(result.get("said"))
        quote = result.get("quote", "")
        if said is None:
            status, reason = "unverified", "judge gave no true/false answer"
        elif not said:
            status, reason = "not_said", "the simulated user never expressed this goal"
        elif not _quote_is_real(quote, user_lines):
            status, reason = "unverified", "claimed said, but its quote is not in the user messages"
        else:
            status, reason = "said", ""
        coverage.append(
            {"id": f"G{i}", "goal": goal, "status": status, "quote": quote, "reason": reason}
        )
    return coverage


def render_transcript(turns: list[dict]) -> str:
    """Judge-facing text. Actions are shown so the judge sees what the assistant really did."""
    lines = []
    for t in turns:
        line = f"[turn {t['turn']}] {t['role'].upper()}: {t['text']}"
        for a in t.get("actions", []):
            line += f" [action: {a['type']}{': ' + a['value'] if a.get('value') else ''}]"
        lines.append(line)
    return "\n".join(lines)


def judge_transcript(
    case: dict, turns: list[dict], criteria: list[dict] | None = None
) -> list[dict]:
    """Ask the LLM judge about every criterion and grade its answer (no rules, no verdict)."""
    criteria = criteria or build_criteria(case)
    rendered = render_transcript(turns)
    payload = complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EVAL_PROMPT.format(
                    description=case["description"],
                    persona=case["user_persona"],
                    transcript=rendered,
                    criteria="\n".join(f"{c['id']}: {c['statement']}" for c in criteria),
                ),
            },
        ],
        temperature=JUDGE_TEMPERATURE,
    )
    return grade_criteria(criteria, payload.get("results"), rendered)


def evaluate_transcript(data: dict) -> dict:
    """Grade one transcript document ({'test_case': ..., 'transcript': [...]})."""
    case, turns = data["test_case"], data["transcript"]
    criteria = build_criteria(case)

    violations = check_case(case, turns)
    coverage = check_goal_coverage(case, turns)
    if any(g["status"] != "said" for g in coverage):
        return {
            "test_name": case["test_name"],
            "category": case.get("category", ""),
            "verdict": "INVALID",
            "violations": violations_as_dicts(violations),
            "goal_coverage": coverage,
            "criteria": [],
        }
    graded = judge_transcript(case, turns, criteria)
    passed = not violations and all(c["status"] == "pass" for c in graded)
    return {
        "test_name": case["test_name"],
        "category": case.get("category", ""),
        "verdict": "PASS" if passed else "FAIL",
        "violations": violations_as_dicts(violations),
        "goal_coverage": coverage,
        "criteria": graded,
    }


def dataset_hash(cases: list[dict]) -> str:
    """Short fingerprint of the exact test cases that were evaluated."""
    canonical = json.dumps(sorted(cases, key=lambda c: c["test_name"]), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def new_run_record(evaluations: list[dict], cases: list[dict] | None = None) -> dict:
    """One history entry per evaluator invocation; the dashboard is built from these.

    Also pins the conditions of the run (models, judge temperature, dataset fingerprint) so
    two runs can be compared knowing what differed.
    """
    settings = load_settings()
    azure = settings.llm_provider == "azure"
    passed = sum(e["verdict"] == "PASS" for e in evaluations)
    invalid = sum(e["verdict"] == "INVALID" for e in evaluations)
    return {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.azure_deployment if azure else settings.ollama_model,
        "provider": settings.llm_provider,
        "judge_model": settings.azure_deployment if azure else settings.ollama_model,
        "assistant_model": settings.azure_deployment if azure else settings.ollama_assistant_model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "dataset_hash": dataset_hash(cases or []),
        "total": len(evaluations),
        "passed": passed,
        "failed": len(evaluations) - passed - invalid,
        "invalid": invalid,
        "cases": evaluations,
    }


def _print_evaluation(evaluation: dict) -> None:
    graded = evaluation["criteria"]
    ok = sum(c["status"] == "pass" for c in graded)
    if evaluation["verdict"] == "INVALID":
        print(
            f"INVALID  {evaluation['test_name']}  [{evaluation['category'] or 'n/a'}]  "
            "(the test is broken, not the model)"
        )
        for g in evaluation["goal_coverage"]:
            if g["status"] != "said":
                print(f"      GOAL {g['status'].upper():10} {g['id']}  {g['goal']}")
                print(f"{' ' * 17}{g['reason']}")
        return
    print(
        f"{evaluation['verdict']}  {evaluation['test_name']}  "
        f"[{evaluation['category'] or 'n/a'}]  criteria {ok}/{len(graded)}  "
        f"violations {len(evaluation['violations'])}"
    )
    for v in evaluation["violations"]:
        print(f"      RULE  turn {v['turn']}  {v['rule']}: {v['evidence']!r}")
    for c in graded:
        if c["status"] != "pass":
            detail = c["reason"] or (f"quote: {c['quote']!r}" if c["quote"] else "(no quote given)")
            print(f"      {c['status'].upper():10} {c['id']}  {c['statement']}\n{' ' * 17}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", help="Transcript JSON file or a directory of them")
    args = parser.parse_args()

    target = Path(args.path)
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not files:
        print(f"No transcripts found at {target}", file=sys.stderr)
        return 1

    EVALUATIONS_DIR.mkdir(exist_ok=True)
    evaluations, cases = [], []
    for f in files:
        data = json.loads(f.read_text())
        cases.append(data["test_case"])
        evaluation = evaluate_transcript(data)
        (EVALUATIONS_DIR / f.name).write_text(json.dumps(evaluation, indent=2))
        _print_evaluation(evaluation)
        evaluations.append(evaluation)

    record = new_run_record(evaluations, cases)
    runs_file = EVALUATIONS_DIR / RUNS_FILENAME
    with runs_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    report = build_report(
        runs_file,
        REPORTS_DIR / "index.html",
        EVALUATIONS_DIR / REPEAT_FILENAME,
        JUDGE_RESULTS_FILE,
    )
    print(
        f"\n{record['passed']}/{record['total']} passed, {record['failed']} failed, "
        f"{record['invalid']} invalid. Dashboard: {report}"
    )
    return 0 if record["failed"] == 0 and record["invalid"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
