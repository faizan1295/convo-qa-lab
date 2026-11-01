import pytest

import ai_evaluator as evaluator
import behavior_checks as checks


def _turns(*steps) -> list[dict]:
    """(user, assistant, actions) steps -> turns. actions=None mimics the old format."""
    out = []
    for n, (user, assistant, actions) in enumerate(steps, 1):
        out.append({"turn": n, "role": "user", "text": user})
        turn = {"turn": n, "role": "assistant", "text": assistant}
        if actions is not None:
            turn["actions"] = actions
        out.append(turn)
    return out


HANDOFF = {"type": "handoff", "value": ""}


def _book(value: str) -> dict:
    return {"type": "book_appointment", "value": value}


def _rules(turns) -> list[str]:
    return [v.rule for v in checks.check_transcript(turns)]


class TestClaimedButNotPerformed:
    @pytest.mark.parametrize(
        "reply",
        [
            "Sure, I'm transferring you to an agent now.",
            "I'll connect you with a human right away.",
            "Let me transfer you.",
        ],
    )
    def test_saying_handoff_without_the_action_is_a_failed_handoff(self, reply):
        turns = _turns(("I want a human", reply, []))

        assert "handoff_claimed_not_performed" in _rules(turns)

    def test_handoff_with_the_action_is_fine(self):
        turns = _turns(("I want a human", "Transferring you to an agent now.", [HANDOFF]))

        assert _rules(turns) == []

    @pytest.mark.parametrize(
        "reply",
        [
            "I can't transfer you right now.",
            "Would you like me to connect you with a human?",
            "I'm unable to hand you over at the moment.",
        ],
    )
    def test_refusals_and_offers_are_not_claims(self, reply):
        turns = _turns(("I want a human", reply, []))

        assert "handoff_claimed_not_performed" not in _rules(turns)

    @pytest.mark.parametrize(
        "reply",
        ["I've booked your appointment.", "Your appointment is confirmed.", "You're all set."],
    )
    def test_saying_booked_without_the_action_is_flagged(self, reply):
        turns = _turns(("Tuesday at 3 pm", reply, []))

        assert "booking_claimed_not_performed" in _rules(turns)

    def test_booking_claim_with_the_action_is_fine(self):
        turns = _turns(
            ("Tuesday at 3 pm", "I've booked your appointment.", [_book("Tuesday 3 pm")])
        )

        assert _rules(turns) == []

    def test_old_transcripts_without_actions_are_not_judged_by_action_rules(self):
        turns = _turns(("I want a human", "I'm transferring you to an agent now.", None))

        assert _rules(turns) == []

    def test_handoff_action_does_not_excuse_a_booking_claim(self):
        turns = _turns(("hi", "I've booked it.", [HANDOFF]))

        assert "booking_claimed_not_performed" in _rules(turns)


class TestSilentActions:
    def test_action_with_no_spoken_text_is_flagged(self):
        turns = _turns(("I want a human", "", [HANDOFF]))

        assert "no_spoken_reply" in _rules(turns)

    def test_empty_text_without_actions_is_not_this_rules_business(self):
        turns = _turns(("hi", "", []))

        assert "no_spoken_reply" not in _rules(turns)


class TestBookingDetails:
    def test_booking_the_time_the_user_gave_is_fine(self):
        turns = _turns(("Next Tuesday at 3 pm.", "Booked.", [_book("Tuesday 3 pm")]))

        assert _rules(turns) == []

    def test_booking_a_time_the_user_never_gave_is_invented(self):
        turns = _turns(("Just pick something.", "Booked.", [_book("Friday 10 am")]))

        flagged = [v for v in checks.check_transcript(turns) if v.rule == "invented_booking_detail"]
        assert {v.evidence.split()[0] for v in flagged} == {"'Friday'", "'10'"}

    def test_booking_without_any_day_or_time_is_flagged(self):
        turns = _turns(("Book it", "Done.", [_book("soon")]))

        assert "booking_without_time" in _rules(turns)

    def test_details_from_earlier_user_turns_count(self):
        turns = _turns(
            ("Tuesday works.", "What time?", []),
            ("3 pm please.", "Booked.", [_book("Tuesday 3 pm")]),
        )

        assert _rules(turns) == []

    def test_a_digit_inside_a_longer_number_is_not_a_match(self):
        turns = _turns(("Make it 13:30.", "Booked.", [_book("3")]))

        assert "invented_booking_detail" in _rules(turns)

    def test_unknown_action_type_is_a_violation(self):
        turns = _turns(("hi", "Done.", [{"type": "delete_account", "value": ""}]))

        assert "unknown_action" in _rules(turns)


class TestOutcomes:
    def _case(self, **kw) -> dict:
        return {"test_name": "t", **kw}

    def test_expected_action_present_by_the_deadline_passes(self):
        turns = _turns(("hi", "Hello.", []), ("human please", "On it.", [HANDOFF]))

        found = checks.check_outcomes(
            turns, self._case(expected_actions=[{"type": "handoff", "by_turn": 2}])
        )

        assert found == []

    def test_expected_action_missing_is_reported(self):
        turns = _turns(("human please", "I can help with that myself.", []))

        (v,) = checks.check_outcomes(turns, self._case(expected_actions=[{"type": "handoff"}]))

        assert v.rule == "missing_action" and "handoff" in v.evidence

    def test_expected_action_too_late_counts_as_missing(self):
        turns = _turns(("a", "x", []), ("b", "y", []), ("human", "ok", [HANDOFF]))

        found = checks.check_outcomes(
            turns, self._case(expected_actions=[{"type": "handoff", "by_turn": 2}])
        )

        assert [v.rule for v in found] == ["missing_action"]

    def test_value_contains_is_case_insensitive(self):
        turns = _turns(("Tuesday 3 pm", "Booked.", [_book("TUESDAY 3 PM")]))
        spec = {"type": "book_appointment", "value_contains": "tuesday"}

        assert checks.check_outcomes(turns, self._case(expected_actions=[spec])) == []

    def test_forbidden_action_is_reported_with_its_turn(self):
        turns = _turns(("hi", "Hello.", []), ("bye", "Transferring you.", [HANDOFF]))

        (v,) = checks.check_outcomes(turns, self._case(forbidden_actions=[{"type": "handoff"}]))

        assert (v.rule, v.turn) == ("forbidden_action", 2)

    def test_forbidden_before_turn_only_blocks_early_actions(self):
        early = _turns(("book", "Booked.", [_book("Tuesday 3 pm")]), ("x", "y", []))
        late = _turns(("book", "When?", []), ("Tuesday 3 pm", "Booked.", [_book("Tuesday 3 pm")]))
        case = self._case(forbidden_actions=[{"type": "book_appointment", "before_turn": 2}])

        assert [v.rule for v in checks.check_outcomes(early, case)] == ["forbidden_action"]
        assert checks.check_outcomes(late, case) == []

    def test_case_without_outcome_fields_has_no_outcome_violations(self):
        assert checks.check_outcomes(_turns(("a", "b", [HANDOFF])), self._case()) == []


class TestPipelineIntegration:
    def test_check_case_combines_rules_phrases_and_outcomes(self):
        turns = _turns(("hi", "Sure, I'm transferring you. It is 3rd and Main.", []))
        case = {"forbidden_phrases": ["3rd and main"], "expected_actions": [{"type": "handoff"}]}

        rules = {v.rule for v in checks.check_case(case, turns)}

        assert {
            "handoff_claimed_not_performed",
            "invented_intersection",
            "forbidden_phrase",
            "missing_action",
        } <= rules

    def test_judge_sees_the_actions_the_assistant_performed(self):
        turns = _turns(("book", "Booked.", [_book("Tuesday 3 pm")]), ("x", "ok", []))

        rendered = evaluator.render_transcript(turns)

        assert "ASSISTANT: Booked. [action: book_appointment: Tuesday 3 pm]" in rendered
        assert rendered.splitlines()[-1] == "[turn 2] ASSISTANT: ok"

    def test_evaluator_fails_a_case_on_a_failed_handoff_even_if_the_judge_approves(
        self, fake_llm, sample_case
    ):
        import json

        case = {**sample_case, "user_goals": [], "expected_actions": [{"type": "handoff"}]}
        turns = _turns(("I want a human", "I can help you with that myself.", []))
        ids = [c["id"] for c in evaluator.build_criteria(case)]
        quote = "I can help you with that myself"
        fake_llm.queue(
            json.dumps(
                {
                    "results": [
                        {"id": i, "holds": True, "quote": quote}
                        if i.startswith("E")
                        else {"id": i, "violated": False, "quote": ""}
                        for i in ids
                    ]
                }
            )
        )

        result = evaluator.evaluate_transcript({"test_case": case, "transcript": turns})

        assert result["verdict"] == "FAIL"
        assert {v["rule"] for v in result["violations"]} == {"missing_action"}
