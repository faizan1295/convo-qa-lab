import json

import pytest

import judge_benchmark as bench
from ai_evaluator import VIOLATION_KINDS, build_criteria, grade_criteria, render_transcript

BASES = json.loads(bench.BASE_FILE.read_text())["items"]


def _fake_judge(catch_if=lambda sentence_present: sentence_present, label_overrides=None):
    """Judge stand-in: flags a violation criterion only when an injected sentence is present."""
    sentences = {m.sentence.strip() for m in bench.MUTATIONS}

    def judge(case, turns):
        text = render_transcript(turns)
        injected = {s for s in sentences if s in text}
        out = []
        for c in build_criteria(case):
            if c["kind"] in VIOLATION_KINDS:
                hit = any(
                    m.statement == c["statement"] and m.sentence.strip() in injected
                    for m in bench.MUTATIONS
                )
                status = "fail" if catch_if(hit) and hit else "pass"
            else:
                status = "pass"
            out.append({**c, "status": status, "quote": "", "reason": ""})
        return out

    return judge


class TestStats:
    def test_wilson_interval_brackets_the_rate_and_shrinks_with_more_samples(self):
        small = bench.wilson(5, 10)
        large = bench.wilson(50, 100)

        assert small[0] < 0.5 < small[1]
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_zero_failures_in_fifteen_still_leaves_a_wide_upper_bound(self):
        lo, hi = bench.wilson(0, 15)

        assert lo == 0.0 and 0.15 < hi < 0.25

    def test_wilson_with_no_samples_is_the_whole_range(self):
        assert bench.wilson(0, 0) == (0.0, 1.0)

    def test_kappa_is_one_for_perfect_agreement_and_zero_for_chance_level(self):
        assert bench.cohen_kappa([(True, True), (False, False), (True, True), (False, False)]) == 1
        assert bench.cohen_kappa([(True, True), (True, False), (False, True), (False, False)]) == 0

    def test_kappa_is_undefined_without_variation(self):
        assert bench.cohen_kappa([(True, True), (True, True)]) is None
        assert bench.cohen_kappa([]) is None


class TestMutations:
    def test_sentence_is_appended_to_the_last_assistant_reply_only(self):
        turns = BASES[0]["transcript"]
        before = json.dumps(turns)

        mutated = bench.apply_mutation(turns, " It's about 4 miles from here.")

        assert mutated[-1]["text"].endswith("It's about 4 miles from here.")
        assert mutated[1]["text"] == turns[1]["text"]
        assert json.dumps(turns) == before

    def test_every_mutation_targets_a_criterion_that_exists_in_the_benchmark_case(self):
        case = bench.with_mutation_criteria(BASES[0]["test_case"])
        statements = {c["statement"] for c in build_criteria(case)}

        assert all(m.statement in statements for m in bench.MUTATIONS)

    def test_base_transcripts_are_clean_according_to_the_deterministic_rules(self):
        from behavior_checks import check_transcript

        for base in BASES:
            assert check_transcript(base["transcript"]) == [], base["test_case"]["test_name"]

    def test_mutation_sentences_really_trip_the_grounding_rules_where_expected(self):
        from behavior_checks import check_transcript

        flagged = {
            m.name
            for m in bench.MUTATIONS
            if check_transcript(bench.apply_mutation(BASES[0]["transcript"], m.sentence))
        }

        assert {"invented_distance", "invented_street"} <= flagged


class TestMutationTrials:
    def test_perfect_judge_catches_everything_with_no_false_alarms(self):
        result = bench.run_mutation_trials(BASES, _fake_judge())

        assert all(r["rate"] == 1 for r in result["mutations"].values())
        assert result["controls"]["rate"] == 0 and result["controls"]["n"] > 0

    def test_rubber_stamp_judge_has_zero_recall(self):
        result = bench.run_mutation_trials(BASES, _fake_judge(catch_if=lambda hit: False))

        assert all(r["rate"] == 0 for r in result["mutations"].values())

    def test_judge_that_flags_everything_is_exposed_by_the_false_alarm_rate(self):
        def paranoid(case, turns):
            return [
                {**c, "status": "fail" if c["kind"] in VIOLATION_KINDS else "pass"}
                for c in build_criteria(case)
            ]

        result = bench.run_mutation_trials(BASES, paranoid)

        assert result["controls"]["rate"] == 1

    def test_unverified_counts_as_flagged_but_is_reported_separately(self):
        def unsure(case, turns):
            return [{**c, "status": "unverified"} for c in build_criteria(case)]

        result = bench.run_mutation_trials(BASES[:1], unsure)

        stats = result["mutations"]["rude_tone"]
        assert stats["rate"] == 1 and stats["unverified"] == stats["n"]

    def test_repeats_multiply_the_sample_size(self):
        once = bench.run_mutation_trials(BASES[:1], _fake_judge())
        twice = bench.run_mutation_trials(BASES[:1], _fake_judge(), repeats=2)

        assert twice["controls"]["n"] == 2 * once["controls"]["n"]


class TestLabeled:
    def _item(self, human: dict, reviewed=True) -> dict:
        base = BASES[0]
        case = base["test_case"]
        labels = [
            {"id": c["id"], "kind": c["kind"], "statement": c["statement"], "human": human[c["id"]]}
            for c in build_criteria(case)
        ]
        return {
            "id": "x",
            "test_case": case,
            "transcript": base["transcript"],
            "reviewed": reviewed,
            "labels": labels,
        }

    def _judge_saying(self, statuses: dict):
        def judge(case, turns):
            return [{**c, "status": statuses[c["id"]]} for c in build_criteria(case)]

        return judge

    def test_false_pass_is_a_failure_the_judge_let_through(self):
        ids = [c["id"] for c in build_criteria(BASES[0]["test_case"])]
        human = {i: "pass" for i in ids} | {ids[0]: "fail", ids[1]: "fail"}
        judge = self._judge_saying({i: "pass" for i in ids} | {ids[0]: "fail"})

        result = bench.measure_labeled([self._item(human)], judge)

        assert (result["false_pass"]["k"], result["false_pass"]["n"]) == (1, 2)
        assert (result["failures_caught"]["k"], result["failures_caught"]["n"]) == (1, 2)
        assert result["false_alarm"]["k"] == 0

    def test_false_alarm_is_a_good_answer_the_judge_rejected(self):
        ids = [c["id"] for c in build_criteria(BASES[0]["test_case"])]
        human = {i: "pass" for i in ids}
        judge = self._judge_saying({i: "unverified" if i == ids[0] else "pass" for i in ids})

        result = bench.measure_labeled([self._item(human)], judge)

        assert result["false_alarm"]["k"] == 1
        assert result["false_alarm"]["n"] == len(ids)

    def test_unreviewed_or_incomplete_items_are_skipped(self):
        ids = [c["id"] for c in build_criteria(BASES[0]["test_case"])]
        complete = {i: "pass" for i in ids}
        partial = complete | {ids[0]: None}
        judge = self._judge_saying({i: "pass" for i in ids})

        result = bench.measure_labeled(
            [self._item(complete), self._item(complete, reviewed=False), self._item(partial)],
            judge,
        )

        assert (result["items_used"], result["items_skipped"]) == (1, 2)

    def test_draft_lists_every_criterion_unlabeled_and_unreviewed(self, tmp_path):
        base = BASES[0]
        (tmp_path / "t1.json").write_text(
            json.dumps({"test_case": base["test_case"], "transcript": base["transcript"]})
        )

        (item,) = bench.draft_items(tmp_path)

        assert item["id"] == "t1" and item["reviewed"] is False
        assert all(label["human"] is None for label in item["labels"])
        assert len(item["labels"]) == len(build_criteria(base["test_case"]))
        assert not bench._is_labeled(item)


def test_grade_criteria_is_what_the_real_judge_path_uses():
    case = bench.with_mutation_criteria(BASES[0]["test_case"])
    criteria = build_criteria(case)

    graded = grade_criteria(criteria, [], "")

    assert {c["status"] for c in graded} == {"unverified"}


@pytest.mark.parametrize("base", BASES, ids=lambda b: b["test_case"]["test_name"])
def test_base_transcripts_have_the_shape_the_judge_needs(base):
    assert base["test_case"]["expected_behavior"]
    assert all({"turn", "role", "text"} <= set(t) for t in base["transcript"])
    assert any(t["role"] == "assistant" for t in base["transcript"])


class TestShow:
    def _item(self, tmp_path):
        base = BASES[0]
        turns = json.loads(json.dumps(base["transcript"]))
        turns[1]["actions"] = [{"type": "handoff", "value": ""}]
        (tmp_path / "t1.json").write_text(
            json.dumps({"test_case": base["test_case"], "transcript": turns})
        )
        (item,) = bench.draft_items(tmp_path)
        item["labels"][0]["human"] = "fail"
        return item

    def test_item_is_readable_with_actions_and_current_labels(self, tmp_path):
        text = bench.format_item(self._item(tmp_path))

        assert "[turn 1] USER: Um, take me to the airport." in text
        assert "[ACTION handoff]" in text
        assert "[fail]" in text and "[   ?]" in text
        assert "pass = it did NOT do it" in text

    def test_unknown_item_lists_the_available_ids(self, tmp_path, monkeypatch, capsys):
        item = self._item(tmp_path)
        draft = tmp_path / "draft.json"
        draft.write_text(json.dumps({"items": [item]}))
        monkeypatch.setattr(bench, "LABELED_FILE", tmp_path / "missing.json")
        monkeypatch.setattr(bench, "DRAFT_FILE", draft)

        assert bench._cmd_show("nope") == 1
        assert "t1" in capsys.readouterr().err
        assert bench._cmd_show("t1") == 0


class TestLabelItem:
    def _item(self, tmp_path):
        item = TestShow()._item(tmp_path)
        for label in item["labels"]:
            label["human"] = None
        return item

    def test_yes_means_pass_for_expected_and_fail_for_forbidden(self, tmp_path):
        item = self._item(tmp_path)
        answers = iter(["y"] * len(item["labels"]))

        bench.label_item(item, lambda _prompt: next(answers), lambda: None)

        for label in item["labels"]:
            assert label["human"] == ("pass" if label["kind"] == "expected" else "fail")
        assert item["reviewed"] is True

    def test_no_means_fail_for_expected_and_pass_for_forbidden(self, tmp_path):
        item = self._item(tmp_path)
        answers = iter(["n"] * len(item["labels"]))

        bench.label_item(item, lambda _prompt: next(answers), lambda: None)

        for label in item["labels"]:
            assert label["human"] == ("fail" if label["kind"] == "expected" else "pass")

    def test_skip_and_quit_never_mark_reviewed(self, tmp_path):
        item = self._item(tmp_path)
        answers = iter(["s", "q"])

        bench.label_item(item, lambda _prompt: next(answers), lambda: None)

        assert item["labels"][0]["human"] is None
        assert item["reviewed"] is False

    def test_already_labeled_criteria_are_not_asked_again(self, tmp_path):
        item = self._item(tmp_path)
        asked = []
        for label in item["labels"]:
            label["human"] = "pass"

        bench.label_item(item, lambda prompt: asked.append(prompt) or "q", lambda: None)

        assert asked == []
        assert item["reviewed"] is True

    def test_invalid_answer_is_asked_again(self, tmp_path):
        item = self._item(tmp_path)
        for label in item["labels"][1:]:
            label["human"] = "pass"
        answers = iter(["pass", "x", "n"])

        bench.label_item(item, lambda _prompt: next(answers), lambda: None)

        assert item["labels"][0]["human"] == "fail"
