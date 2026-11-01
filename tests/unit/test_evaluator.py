import json
import sys

import pytest

import ai_evaluator as evaluator

REPLY = "Sure, I can avoid tolls for you."
# criteria for sample_case: E1, N1, U1, U2, U3
ALL_IDS = ["E1", "N1", "U1", "U2", "U3"]


@pytest.fixture
def transcript_doc(sample_case) -> dict:
    return {
        "test_case": sample_case,
        "transcript": [
            {"turn": 1, "role": "user", "text": "um, take me to the airport"},
            {"turn": 1, "role": "assistant", "text": REPLY},
        ],
    }


def _judge(holds=True, quote=REPLY, ids=ALL_IDS, overrides=None) -> str:
    """Judge answer where `holds` means 'all good': expected criteria hold, none is violated."""

    def answer(i):
        if i.startswith("E"):
            return {"id": i, "holds": holds, "quote": quote}
        violated = (not holds) if isinstance(holds, bool) else holds
        return {"id": i, "violated": violated, "quote": quote if violated is True else ""}

    results = {i: answer(i) for i in ids}
    results.update(overrides or {})
    return json.dumps({"results": list(results.values())})


USER_QUOTE = "take me to the airport"
GOAL_IDS = ["G1", "G2", "G3"]


def _coverage(said=True, quote=USER_QUOTE, ids=GOAL_IDS, overrides=None) -> str:
    results = {i: {"id": i, "said": said, "quote": quote} for i in ids}
    results.update(overrides or {})
    return json.dumps({"results": list(results.values())})


def _queue(fake_llm, *judge_replies, coverage=None) -> None:
    """Every evaluation first asks the goal-coverage question, then the criteria judge."""
    for reply in judge_replies:
        fake_llm.queue(coverage or _coverage(), reply)


class TestVerdict:
    def test_pass_when_every_criterion_holds_with_a_real_quote(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge())

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "PASS"
        assert {c["status"] for c in result["criteria"]} == {"pass"}

    def test_one_failed_criterion_fails_the_case(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(overrides={"E1": {"id": "E1", "holds": False, "quote": REPLY}}))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "FAIL"
        statuses = {c["id"]: c["status"] for c in result["criteria"]}
        assert statuses["E1"] == "fail" and statuses["U1"] == "pass"

    def test_invented_quote_for_a_pass_is_unverified_and_fails(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(quote="the assistant said something it never said"))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "FAIL"
        assert {c["id"]: c["status"] for c in result["criteria"]}["E1"] == "unverified"

    def test_failing_criterion_needs_no_quote(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(overrides={"E1": {"id": "E1", "holds": False, "quote": ""}}))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert {c["id"]: c["status"] for c in result["criteria"]}["E1"] == "fail"

    def test_holds_without_quote_never_passes(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(overrides={"E1": {"id": "E1", "holds": True, "quote": ""}}))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert {c["id"]: c["status"] for c in result["criteria"]}["E1"] == "unverified"
        assert result["verdict"] == "FAIL"

    def test_criterion_missing_from_judge_answer_is_unverified(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(ids=["E1", "N1", "U1", "U2"]))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "FAIL"
        assert [c["status"] for c in result["criteria"] if c["id"] == "U3"] == ["unverified"]

    @pytest.mark.parametrize("bad", [None, "maybe", 1, ""])
    def test_non_boolean_holds_is_unverified(self, fake_llm, transcript_doc, bad):
        _queue(fake_llm, _judge(overrides={"E1": {"id": "E1", "holds": bad, "quote": REPLY}}))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "FAIL"

    def test_string_booleans_are_accepted(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(holds="true"))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert {c["id"]: c["status"] for c in result["criteria"]}["E1"] == "pass"

    def test_quote_matching_ignores_case_whitespace_and_ellipsis(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(quote='"...sure,  I can AVOID tolls..."'))

        assert evaluator.evaluate_transcript(transcript_doc)["verdict"] == "PASS"

    def test_too_short_quote_is_rejected(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge(quote="Sure"))

        assert evaluator.evaluate_transcript(transcript_doc)["verdict"] == "FAIL"

    @pytest.mark.parametrize("garbage", ["{}", '{"results": "nope"}', '{"results": [1, 2]}'])
    def test_malformed_judge_payload_fails_closed(self, fake_llm, transcript_doc, garbage):
        _queue(fake_llm, garbage)

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "FAIL"
        assert {c["status"] for c in result["criteria"]} == {"unverified"}


class TestViolationCriteria:
    def _status(self, fake_llm, transcript_doc, **n1) -> dict:
        _queue(fake_llm, _judge(overrides={"N1": {"id": "N1", **n1}}))
        result = evaluator.evaluate_transcript(transcript_doc)
        return {c["id"]: c["status"] for c in result["criteria"]}

    def test_not_violated_passes_without_any_quote(self, fake_llm, transcript_doc):
        status = self._status(fake_llm, transcript_doc, violated=False, quote="")

        assert status["N1"] == "pass"

    def test_violation_with_a_real_quote_fails(self, fake_llm, transcript_doc):
        status = self._status(fake_llm, transcript_doc, violated=True, quote=REPLY)

        assert status["N1"] == "fail"

    def test_claimed_violation_with_invented_quote_is_unverified_and_fails_the_case(
        self, fake_llm, transcript_doc
    ):
        _queue(
            fake_llm,
            _judge(overrides={"N1": {"id": "N1", "violated": True, "quote": "never said this"}}),
        )

        result = evaluator.evaluate_transcript(transcript_doc)

        assert {c["id"]: c["status"] for c in result["criteria"]}["N1"] == "unverified"
        assert result["verdict"] == "FAIL"

    @pytest.mark.parametrize("bad", [None, "maybe", 1, ""])
    def test_non_boolean_violated_is_unverified(self, fake_llm, transcript_doc, bad):
        status = self._status(fake_llm, transcript_doc, violated=bad, quote="")

        assert status["N1"] == "unverified"

    def test_a_holds_answer_to_a_violation_criterion_does_not_count(self, fake_llm, transcript_doc):
        status = self._status(fake_llm, transcript_doc, holds=True, quote=REPLY)

        assert status["N1"] == "unverified"

    def test_violation_statements_have_no_double_negatives(self, sample_case):
        statements = [c["statement"] for c in evaluator.build_criteria(sample_case)]

        assert not [
            s for s in statements if "NOT" in s or s.lower().startswith(("does not", "not "))
        ]


class TestGoalCoverage:
    def test_all_goals_said_goes_on_to_judge_the_assistant(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge())

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "PASS"
        assert [g["status"] for g in result["goal_coverage"]] == ["said"] * 3

    def test_unsaid_goal_makes_the_case_invalid_not_fail(self, fake_llm, transcript_doc):
        fake_llm.queue(_coverage(overrides={"G3": {"id": "G3", "said": False, "quote": ""}}))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "INVALID"
        assert [g["status"] for g in result["goal_coverage"]] == ["said", "said", "not_said"]
        assert result["criteria"] == []

    def test_assistant_is_not_judged_when_the_test_is_invalid(self, fake_llm, transcript_doc):
        fake_llm.queue(_coverage(said=False, quote=""))

        evaluator.evaluate_transcript(transcript_doc)

        assert len(fake_llm.calls) == 1

    def test_said_claim_with_a_quote_not_in_user_messages_is_invalid(
        self, fake_llm, transcript_doc
    ):
        fake_llm.queue(_coverage(quote="avoid the toll roads please"))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "INVALID"
        assert {g["status"] for g in result["goal_coverage"]} == {"unverified"}

    def test_quote_from_an_assistant_line_does_not_count_as_the_user_saying_it(
        self, fake_llm, transcript_doc
    ):
        fake_llm.queue(_coverage(quote=REPLY))

        assert evaluator.evaluate_transcript(transcript_doc)["verdict"] == "INVALID"

    @pytest.mark.parametrize("garbage", ["{}", '{"results": "nope"}', '{"results": [1]}'])
    def test_malformed_coverage_answer_fails_closed_as_invalid(
        self, fake_llm, transcript_doc, garbage
    ):
        fake_llm.queue(garbage)

        assert evaluator.evaluate_transcript(transcript_doc)["verdict"] == "INVALID"

    def test_coverage_prompt_shows_only_user_lines_and_the_goals(self, fake_llm, transcript_doc):
        fake_llm.queue(_coverage(), _judge())

        evaluator.evaluate_transcript(transcript_doc)
        coverage_prompt = "\n".join(m["content"] for m in fake_llm.calls[0]["messages"])

        assert "um, take me to the airport" in coverage_prompt
        assert REPLY not in coverage_prompt
        assert "G2: Avoid tolls" in coverage_prompt
        assert fake_llm.calls[0]["temperature"] == 0

    def test_rule_violations_stay_visible_on_an_invalid_case(self, fake_llm, transcript_doc):
        transcript_doc["transcript"][1]["text"] = "It is 3rd and Main, a block away."
        fake_llm.queue(_coverage(said=False, quote=""))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert result["verdict"] == "INVALID"
        assert {v["rule"] for v in result["violations"]} >= {"invented_quantity"}


class TestDeterministicGate:
    def test_rule_violation_fails_even_if_judge_approves_everything(self, fake_llm, transcript_doc):
        bad = "It is 3rd and Main, a block away."
        transcript_doc["transcript"][1]["text"] = bad
        _queue(fake_llm, _judge(quote=bad))

        result = evaluator.evaluate_transcript(transcript_doc)

        assert {c["status"] for c in result["criteria"]} == {"pass"}
        assert result["verdict"] == "FAIL"
        assert {v["rule"] for v in result["violations"]} == {
            "invented_quantity",
            "invented_intersection",
        }


class TestCriteria:
    def test_ids_kinds_and_must_not_phrasing(self, sample_case):
        criteria = evaluator.build_criteria(sample_case)

        assert [c["id"] for c in criteria] == ALL_IDS
        assert [c["kind"] for c in criteria] == ["expected"] + ["must_not"] + ["universal"] * 3
        assert criteria[1]["statement"] == "States an exact ETA it cannot know"

    def test_old_format_case_gives_actionable_error(self, sample_case):
        del sample_case["expected_behavior"]

        with pytest.raises(ValueError, match="Regenerate test cases"):
            evaluator.build_criteria(sample_case)


class TestRunMetadata:
    def test_dataset_hash_changes_when_a_case_changes_and_ignores_order(self, sample_case):
        other = {**sample_case, "test_name": "b_case"}
        base = evaluator.dataset_hash([sample_case, other])

        assert base == evaluator.dataset_hash([other, sample_case])
        assert base != evaluator.dataset_hash([{**sample_case, "must_not": ["x"]}, other])

    def test_assistant_model_can_differ_from_judge_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "judge-model")
        monkeypatch.setenv("OLLAMA_ASSISTANT_MODEL", "assistant-model")

        record = evaluator.new_run_record([])

        assert (record["judge_model"], record["assistant_model"]) == (
            "judge-model",
            "assistant-model",
        )


class TestPrompt:
    def test_includes_transcript_persona_and_criteria(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge())

        evaluator.evaluate_transcript(transcript_doc)

        prompt = fake_llm.last_prompt
        assert f"[turn 1] ASSISTANT: {REPLY}" in prompt
        assert "Rushed commuter" in prompt
        assert "E1: Follows the latest request and avoids tolls" in prompt
        assert "N1: States an exact ETA it cannot know" in prompt
        assert '"violated"' in prompt
        assert "word-for-word" in prompt

    def test_judge_is_deterministic_and_json_constrained(self, fake_llm, transcript_doc):
        _queue(fake_llm, _judge())

        evaluator.evaluate_transcript(transcript_doc)

        assert fake_llm.last_call["temperature"] == 0
        assert fake_llm.last_call["response_format"] == {"type": "json_object"}


class TestCli:
    @pytest.fixture(autouse=True)
    def isolated_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluator, "EVALUATIONS_DIR", tmp_path / "evaluations")
        monkeypatch.setattr(evaluator, "REPORTS_DIR", tmp_path / "reports")
        self.out_dir = tmp_path / "evaluations"
        self.reports_dir = tmp_path / "reports"

    def _run(self, monkeypatch, path) -> int:
        monkeypatch.setattr(sys, "argv", ["ai_evaluator.py", str(path)])
        return evaluator.main()

    def test_passing_transcript_exits_zero_and_writes_evaluation(
        self, fake_llm, monkeypatch, tmp_path, transcript_doc
    ):
        f = tmp_path / "t.json"
        f.write_text(json.dumps(transcript_doc))
        _queue(fake_llm, _judge())

        assert self._run(monkeypatch, f) == 0
        assert json.loads((self.out_dir / "t.json").read_text())["verdict"] == "PASS"

    def test_any_failing_transcript_exits_one_and_prints_why(
        self, fake_llm, monkeypatch, tmp_path, transcript_doc, capsys
    ):
        folder = tmp_path / "transcripts"
        folder.mkdir()
        (folder / "a.json").write_text(json.dumps(transcript_doc))
        (folder / "b.json").write_text(json.dumps(transcript_doc))
        _queue(
            fake_llm,
            _judge(),
            _judge(overrides={"E1": {"id": "E1", "holds": False, "quote": REPLY}}),
        )

        assert self._run(monkeypatch, folder) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out and "E1" in out

    def test_invalid_case_exits_one_and_is_counted_separately(
        self, fake_llm, monkeypatch, tmp_path, transcript_doc, capsys
    ):
        f = tmp_path / "t.json"
        f.write_text(json.dumps(transcript_doc))
        fake_llm.queue(_coverage(said=False, quote=""))

        assert self._run(monkeypatch, f) == 1

        out = capsys.readouterr().out
        assert "INVALID" in out and "Set destination" in out
        record = json.loads((self.out_dir / "runs.jsonl").read_text())
        assert (record["total"], record["passed"], record["failed"], record["invalid"]) == (
            1,
            0,
            0,
            1,
        )

    def test_empty_directory_exits_one(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        assert self._run(monkeypatch, empty) == 1

    def test_run_is_recorded_and_dashboard_is_built(
        self, fake_llm, monkeypatch, tmp_path, transcript_doc, capsys
    ):
        f = tmp_path / "t.json"
        f.write_text(json.dumps(transcript_doc))
        _queue(fake_llm, _judge(), _judge())

        self._run(monkeypatch, f)
        self._run(monkeypatch, f)

        lines = (self.out_dir / "runs.jsonl").read_text().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert (record["total"], record["passed"], record["failed"], record["invalid"]) == (
            1,
            1,
            0,
            0,
        )
        assert record["cases"][0]["test_name"] == "reroute_after_toll_change"
        assert (self.reports_dir / "index.html").exists()
        assert record["judge_model"] == record["model"] == "llama3.1:8b"
        assert record["assistant_model"] == "llama3.1:8b"
        assert record["judge_temperature"] == 0
        assert len(record["dataset_hash"]) == 12
        assert "1/1 passed" in capsys.readouterr().out
