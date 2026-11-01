import repeat_runs as rr


def test_pass_rate_excludes_invalid_runs():
    runs = [{"a": "PASS", "b": "INVALID"}, {"a": "FAIL", "b": "INVALID"}, {"a": "PASS"}]

    rows = {r["test_name"]: r for r in rr.summarize(runs)}

    assert (rows["a"]["pass"], rows["a"]["fail"]) == (2, 1)
    assert abs(rows["a"]["rate"] - 2 / 3) < 1e-9
    assert rows["b"]["invalid"] == 2 and rows["b"]["rate"] is None


def test_interval_is_wide_for_few_runs():
    (row,) = rr.summarize([{"a": "PASS"}, {"a": "PASS"}, {"a": "PASS"}])

    assert row["rate"] == 1.0
    assert row["ci95"][0] < 0.5


def test_summary_lists_worst_cases_first():
    rows = rr.summarize([{"good": "PASS", "bad": "FAIL"}])

    text = rr.format_summary(rows, 1)

    assert text.index("bad") < text.index("good")


def test_crashed_runs_are_counted_not_hidden():
    (row,) = rr.summarize([{"a": "CRASHED"}, {"a": "PASS"}])

    assert row["crashed"] == 1 and row["rate"] == 1.0
    assert "crashed 1" in rr.format_summary([row], 2)


def _evaluation(verdict, violations=(), criteria=(), goals=()):
    return {
        "verdict": verdict,
        "violations": [{"rule": r} for r in violations],
        "criteria": [{"id": i, "status": st, "statement": "x"} for i, st in criteria],
        "goal_coverage": [{"id": i, "status": st} for i, st in goals],
    }


def test_reasons_are_tagged_by_layer():
    rule = rr.failure_reasons(_evaluation("FAIL", violations=["reply_too_long"]))
    judge = rr.failure_reasons(_evaluation("FAIL", criteria=[("N2", "fail"), ("E1", "pass")]))
    broken = rr.failure_reasons(_evaluation("INVALID", goals=[("G2", "not_said")]))

    assert rule == ["rule:reply_too_long"]
    assert judge == ["judge:N2 fail - x"]
    assert broken == ["test:goal_not_said G2"]


def test_reason_counts_count_runs_not_repeats_within_a_run():
    runs = [{"a": ["rule:x", "rule:x"]}, {"a": ["rule:x", "judge:N1 fail - y"]}, {"a": []}]

    counts = rr.reason_counts(runs)

    assert counts["a"]["rule:x"] == 2 and counts["a"]["judge:N1 fail - y"] == 1
    assert "rule:x" in rr.format_reasons(counts, 3)
