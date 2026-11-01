"""Opens the generated dashboard in a real browser and checks what a reader would see."""

import json

import pytest
from playwright.sync_api import Page, expect

import report


def _case(name: str, category: str, ok: bool) -> dict:
    return {
        "test_name": name,
        "category": category,
        "verdict": "PASS" if ok else "FAIL",
        "violations": []
        if ok
        else [{"rule": "invented_quantity", "turn": 1, "evidence": "a block"}],
        "criteria": [
            {
                "id": "E1",
                "kind": "expected",
                "statement": "Asks which cinema",
                "status": "pass" if ok else "fail",
                "quote": "Which cinema do you mean?",
                "reason": "",
            }
        ],
    }


def _run(finished_at: str, outcomes: list[bool]) -> dict:
    cases = [_case(f"case_{i}", f"cat_{i}", ok) for i, ok in enumerate(outcomes)]
    passed = sum(outcomes)
    return {
        "finished_at": finished_at,
        "model": "llama3.1:8b",
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "cases": cases,
    }


@pytest.fixture
def dashboard(tmp_path, page: Page):
    def open_with(runs: list[dict]) -> Page:
        runs_file = tmp_path / "runs.jsonl"
        runs_file.write_text("".join(json.dumps(r) + "\n" for r in runs))
        out = report.build_report(runs_file, tmp_path / "reports" / "index.html")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(out.as_uri())
        page.errors = errors
        return page

    return open_with


def test_shows_latest_pass_rate_delta_and_marks(dashboard):
    page = dashboard(
        [
            _run("2026-10-10T10:00:00", [True, False, False, False]),
            _run("2026-10-10T11:00:00", [True, True, True, False]),
        ]
    )

    expect(page.locator(".hero-value")).to_have_text("75%")
    expect(page.locator(".hero .delta")).to_contain_text("+50 pts vs previous run")
    expect(page.locator("details.case")).to_have_count(4)
    assert page.errors == []


def test_failed_case_lists_rule_and_failed_criterion_with_evidence(dashboard):
    page = dashboard([_run("2026-10-10T10:00:00", [True, False])])

    page.locator("details.case", has_text="case_1").locator("summary").click()

    body = page.locator("details.case", has_text="case_1").locator(".case-body")
    expect(body).to_contain_text("invented quantity")
    expect(body).to_contain_text("a block")
    expect(body).to_contain_text("Failed: Asks which cinema")


def test_single_run_renders_without_errors(dashboard):
    page = dashboard([_run("2026-10-10T10:00:00", [False])])

    expect(page.locator(".hero-value")).to_have_text("0%")
    expect(page.locator(".hero .delta")).to_have_text("First recorded run")
    assert page.errors == []


def test_empty_history_explains_what_to_run(dashboard):
    page = dashboard([])

    expect(page.locator(".empty")).to_contain_text("python ai_evaluator.py transcripts/")
    assert page.errors == []


def test_untrusted_text_is_rendered_as_text_not_html(dashboard):
    run = _run("2026-10-10T10:00:00", [False])
    run["cases"][0]["violations"][0]["evidence"] = "<img src=x onerror=alert(1)>"
    run["cases"][0]["test_name"] = "<b>bold</b>"
    page = dashboard([run])

    expect(page.locator("details.case b")).to_have_count(0)
    expect(page.locator("details.case img")).to_have_count(0)
    expect(page.locator(".case-name")).to_have_text("<b>bold</b>")


def _invalid_case(name: str, category: str) -> dict:
    return {
        "test_name": name,
        "category": category,
        "verdict": "INVALID",
        "violations": [],
        "goal_coverage": [
            {
                "id": "G1",
                "goal": "Ask for the beach",
                "status": "not_said",
                "quote": "",
                "reason": "the simulated user never expressed this goal",
            }
        ],
        "criteria": [],
    }


def test_invalid_cases_are_excluded_from_pass_rate_and_shown_as_invalid(dashboard):
    run = _run("2026-10-10T10:00:00", [True, False])
    run["cases"].append(_invalid_case("broken_case", "cat_9"))
    run.update(total=3, invalid=1)
    page = dashboard([run])

    expect(page.locator(".hero-value")).to_have_text("50%")
    expect(page.locator(".kpis")).to_contain_text("Invalid")
    expect(page.locator("details.case", has_text="broken_case")).to_contain_text("Invalid test")
    assert page.errors == []


def test_invalid_case_explains_which_goal_was_never_said(dashboard):
    run = _run("2026-10-10T10:00:00", [])
    run["cases"] = [_invalid_case("broken_case", "ambiguity")]
    run.update(total=1, passed=0, failed=0, invalid=1)
    page = dashboard([run])

    page.locator("details.case", has_text="broken_case").locator("summary").click()

    body = page.locator("details.case", has_text="broken_case").locator(".case-body")
    expect(body).to_contain_text("Goal never said by the simulated user: Ask for the beach")
    expect(page.locator("details.case summary")).to_contain_text("Invalid test")
    assert page.errors == []


def _basis_case(name: str, verdict: str, rule: bool, judge_fail: bool) -> dict:
    return {
        "test_name": name,
        "category": "cat",
        "verdict": verdict,
        "violations": [{"rule": "invented_quantity", "turn": 1, "evidence": "4 miles"}]
        if rule
        else [],
        "criteria": [
            {
                "id": "E1",
                "kind": "expected",
                "statement": "Asks which cinema",
                "status": "fail" if judge_fail else "pass",
                "quote": "Which cinema?",
                "reason": "",
            }
        ],
    }


def _basis_run() -> dict:
    cases = [
        _basis_case("only_rules", "FAIL", rule=True, judge_fail=False),
        _basis_case("both_layers", "FAIL", rule=True, judge_fail=True),
        _basis_case("only_judge", "FAIL", rule=False, judge_fail=True),
        _basis_case("clean", "PASS", rule=False, judge_fail=False),
        _invalid_case("broken", "cat"),
    ]
    return {
        "finished_at": "2026-10-10T10:00:00",
        "model": "llama3.1:8b",
        "total": 5,
        "passed": 1,
        "failed": 3,
        "invalid": 1,
        "cases": cases,
    }


def test_verdict_basis_chart_counts_who_caught_each_case(dashboard):
    page = dashboard([_basis_run()])

    bars = page.locator("svg[aria-label='Cases by what decided the verdict'] .bar-g")
    labels = [bars.nth(i).get_attribute("aria-label") for i in range(bars.count())]
    assert labels == [
        "Rules only: 1",
        "Rules + judge: 1",
        "Judge only: 1",
        "Passed: 1",
        "Invalid test: 1",
    ]
    assert page.errors == []


def test_each_case_names_what_decided_its_verdict(dashboard):
    page = dashboard([_basis_run()])

    only_rules = page.locator("details.case", has_text="only_rules")
    only_rules.locator("summary").click()

    expect(only_rules.locator("summary .chip").last).to_have_text("Rules only")
    expect(only_rules.locator(".case-body")).to_contain_text("The judge alone would have passed")
    clean = page.locator("details.case", has_text="clean")
    expect(clean.locator("summary .chip").last).to_have_text("Passed")
    expect(
        page.locator("details.case", has_text="broken").locator("summary .chip").last
    ).to_have_text("Invalid test")


def test_repeat_runs_and_judge_sections_render_with_reasons(tmp_path, page: Page):
    repeat = {
        "runs": [{"a": "PASS", "b": "INVALID"}, {"a": "FAIL", "b": "INVALID"}],
        "reasons": [{"a": [], "b": ["test:goal_not_said G1"]}, {"a": ["rule:reply_too_long"]}],
        "summary": [
            {
                "test_name": "a",
                "pass": 1,
                "fail": 1,
                "invalid": 0,
                "crashed": 0,
                "rate": 0.5,
                "ci95": [0.09, 0.91],
            },
            {
                "test_name": "b",
                "pass": 0,
                "fail": 0,
                "invalid": 2,
                "crashed": 0,
                "rate": None,
                "ci95": [0.0, 1.0],
            },
        ],
    }
    judge = {
        "items_used": 4, "criteria_judged": 29, "judge_model": "m", "kappa": 0.46,
        "failures_caught": {"k": 3, "n": 8, "rate": 0.375, "ci95": [0.14, 0.69]},
        "false_pass": {"k": 5, "n": 8, "rate": 0.625, "ci95": [0.31, 0.86]},
        "false_alarm": {"k": 0, "n": 21, "rate": 0.0, "ci95": [0.0, 0.15]},
    }  # fmt: skip
    (tmp_path / "repeat.json").write_text(json.dumps(repeat))
    (tmp_path / "judge.json").write_text(json.dumps(judge))
    out = report.build_report(
        tmp_path / "none.jsonl",
        tmp_path / "i.html",
        tmp_path / "repeat.json",
        tmp_path / "judge.json",
    )
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(out.as_uri())

    expect(page.locator("h2").first).to_have_text("Repeat runs - 2 cases x 2 runs")
    expect(page.locator("h2").nth(1)).to_contain_text("Judge vs hand labels")
    expect(page.locator("section").first.locator(".bar-g")).to_have_count(2)
    expect(page.locator("section").first).to_contain_text("no valid run")
    expect(page.locator(".grid4").first).to_contain_text("Broken tests")
    expect(page.locator(".grid4").last).to_contain_text("3 of 8")
    expect(page.locator(".grid4").last).to_contain_text("0.46")
    page.locator("section").first.locator(".bar-g").first.focus()
    expect(page.locator("#tip")).to_contain_text("1 of 2 valid runs passed")
    page.locator("summary", has_text="Why each case did not pass").click()
    page.locator("details.case", has_text="a").first.locator("summary").click()
    expect(page.locator("details.case").first).to_contain_text("Rule (code)")
    assert errors == []
