import json

import report


def _run(finished_at: str, passed: int, total: int) -> dict:
    return {
        "finished_at": finished_at,
        "model": "llama3.1:8b",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "cases": [],
    }


def test_load_runs_reads_one_record_per_line(tmp_path):
    f = tmp_path / "runs.jsonl"
    f.write_text(json.dumps(_run("2026-10-10T10:00:00", 1, 2)) + "\n\n")

    assert [r["passed"] for r in report.load_runs(f)] == [1]


def test_load_runs_of_missing_file_is_empty(tmp_path):
    assert report.load_runs(tmp_path / "nope.jsonl") == []


def test_model_output_cannot_break_out_of_the_script_tag():
    run = _run("2026-10-10T10:00:00", 0, 1)
    run["model"] = "</script><script>alert(1)</script>"

    html = report.render_report([run])

    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script>" in html


def test_report_embeds_runs_and_replaces_placeholder():
    html = report.render_report([_run("2026-10-10T10:00:00", 3, 4)])

    assert report.PLACEHOLDER not in html
    assert '"passed": 3' in html


def test_build_report_creates_parent_directory(tmp_path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(_run("2026-10-10T10:00:00", 1, 1)) + "\n")

    out = report.build_report(runs, tmp_path / "reports" / "index.html")

    assert out.exists()


def test_repeat_and_judge_results_are_embedded_when_given(tmp_path):
    repeat = tmp_path / "repeat.json"
    repeat.write_text(json.dumps({"runs": [], "summary": [], "marker": "repeat-marker"}))
    judge = tmp_path / "judge.json"
    judge.write_text(json.dumps({"kappa": 0.46, "marker": "judge-marker"}))

    out = report.build_report(tmp_path / "none.jsonl", tmp_path / "i.html", repeat, judge)

    html = out.read_text()
    assert "repeat-marker" in html and "judge-marker" in html


def test_missing_repeat_and_judge_files_are_simply_left_out(tmp_path):
    out = report.build_report(tmp_path / "none.jsonl", tmp_path / "i.html", tmp_path / "x", None)

    assert '"repeat": null' in out.read_text()
