import json

import pytest

import behavior_checks as checks


def _turns(*pairs: tuple[str, str]) -> list[dict]:
    """(user, assistant) pairs -> transcript turns."""
    out = []
    for n, (user, assistant) in enumerate(pairs, 1):
        out.append({"turn": n, "role": "user", "text": user})
        out.append({"turn": n, "role": "assistant", "text": assistant})
    return out


def _rules(turns: list[dict]) -> list[str]:
    return [v.rule for v in checks.check_transcript(turns)]


class TestGrounding:
    def test_real_transcript_hallucinations_are_caught(self):
        turns = _turns(
            ("gas station nearby?", "There's one just a block away, on 3rd and Main."),
            ("which way?", "You're headed north, so I can reroute you."),
        )

        found = checks.check_transcript(turns)

        assert [(v.turn, v.rule, v.evidence) for v in found] == [
            (1, "invented_quantity", "a block"),
            (1, "invented_intersection", "3rd and Main"),
            (2, "invented_direction", "You're headed north"),
        ]

    @pytest.mark.parametrize(
        "reply",
        [
            "It is 1.2 kilometers away.",
            "About 25 minutes.",
            "Two miles ahead.",
            "Take Elm Street.",
            "It's at 42 Oak Avenue.",
            "You are heading south right now.",
            "You’re headed east.",
        ],
    )
    def test_ungrounded_specifics_are_flagged(self, reply):
        assert _rules(_turns(("where is it?", reply)))

    @pytest.mark.parametrize(
        ("user", "reply"),
        [
            ("I have 25 minutes", "Then 25 minutes is enough."),
            ("I'm on Elm Street", "Okay, staying on Elm Street."),
            ("I'm heading north", "You're headed north, got it."),
            ("it's 3rd and Main", "Routing to 3rd and Main."),
        ],
    )
    def test_specifics_the_user_provided_are_allowed(self, user, reply):
        assert _rules(_turns((user, reply))) == []

    @pytest.mark.parametrize(
        "reply",
        [
            "Where would you like to go?",
            "Sure, give me a minute.",
            "Tom and Jerry is a great show.",
            "Which direction are you headed?",
        ],
    )
    def test_clean_replies_have_no_violations(self, reply):
        assert _rules(_turns(("hi", reply))) == []

    def test_user_text_only_counts_if_said_before_the_reply(self):
        turns = [
            {"turn": 1, "role": "user", "text": "hi"},
            {"turn": 1, "role": "assistant", "text": "It is 25 minutes away."},
            {"turn": 2, "role": "user", "text": "I only have 25 minutes"},
            {"turn": 2, "role": "assistant", "text": "That works."},
        ]

        assert _rules(turns) == ["invented_quantity"]


class TestStyle:
    @pytest.mark.parametrize(
        "reply",
        ["**Route** found.", "Options:\n- toll road\n- highway", "1. Turn left", "Done `ok`"],
    )
    def test_markdown_is_flagged(self, reply):
        assert "markdown_in_voice_reply" in _rules(_turns(("hi", reply)))

    def test_emoji_is_flagged(self):
        assert "emoji_in_voice_reply" in _rules(_turns(("hi", "On my way \U0001f697")))

    def test_more_than_two_sentences_is_flagged(self):
        assert "reply_too_long" in _rules(_turns(("hi", "One. Two. Three.")))

    def test_two_sentences_and_decimals_are_fine(self):
        assert _rules(_turns(("it is 1.5 km", "Okay, 1.5 km it is. Starting now."))) == []


class TestCli:
    def _write(self, tmp_path, name, turns):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"test_case": {"test_name": name}, "transcript": turns})
        )

    def test_exit_zero_when_clean(self, tmp_path, monkeypatch, capsys):
        self._write(tmp_path, "ok", _turns(("hi", "Where to?")))
        monkeypatch.setattr("sys.argv", ["behavior_checks.py", str(tmp_path)])

        assert checks.main() == 0
        assert "PASS  ok" in capsys.readouterr().out

    def test_exit_one_and_evidence_when_violation(self, tmp_path, monkeypatch, capsys):
        self._write(tmp_path, "bad", _turns(("hi", "It is 3 miles away.")))
        monkeypatch.setattr("sys.argv", ["behavior_checks.py", str(tmp_path)])

        assert checks.main() == 1
        out = capsys.readouterr().out
        assert "FAIL  bad" in out and "invented_quantity" in out
