"""The live runner's own behaviour, with a fake page object (no browser, no LLM)."""

import json

import pytest

import e2e_assistant_runner as runner


class FakeChatPage:
    def __init__(self, replies, crash_on=None):
        self.replies = list(replies)
        self.crash_on = crash_on
        self.sent = []

    def open(self):
        pass

    def send_and_read_reply(self, message):
        self.sent.append(message)
        if self.crash_on == len(self.sent):
            raise AssertionError("reply never appeared")
        return self.replies[len(self.sent) - 1][0]

    def last_reply_actions(self):
        return self.replies[len(self.sent) - 1][1]


CASE = {
    "test_name": "scripted_case",
    "scripted_user_turns": ["first", "second", "third"],
}


@pytest.fixture(autouse=True)
def isolated_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    return tmp_path / "transcripts"


def test_scripted_turns_are_sent_verbatim_and_actions_are_recorded(isolated_transcripts):
    page = FakeChatPage(
        [("ok", []), ("Transferring you.", [{"type": "handoff", "value": ""}]), ("bye", [])]
    )

    runner.test_multi_turn_conversation(page, CASE)

    assert page.sent == ["first", "second", "third"]
    saved = json.loads((isolated_transcripts / "scripted_case.json").read_text())
    assistant = [t for t in saved["transcript"] if t["role"] == "assistant"]
    assert assistant[1]["actions"] == [{"type": "handoff", "value": ""}]
    assert all("actions" in t for t in assistant)


def test_a_crash_keeps_the_partial_transcript_as_evidence_and_reraises(isolated_transcripts):
    page = FakeChatPage([("ok", []), ("never used", [])], crash_on=2)

    with pytest.raises(AssertionError, match="reply never appeared"):
        runner.test_multi_turn_conversation(page, CASE)

    partial = json.loads((isolated_transcripts / "incomplete" / "scripted_case.json").read_text())
    assert [t["text"] for t in partial["transcript"]] == ["first", "ok", "second"]
    assert not (isolated_transcripts / "scripted_case.json").exists()


def test_action_only_reply_is_recorded_not_treated_as_a_crash(isolated_transcripts):
    page = FakeChatPage([("ok", []), ("", [{"type": "handoff", "value": ""}]), ("bye", [])])

    runner.test_multi_turn_conversation(page, CASE)

    assert (isolated_transcripts / "scripted_case.json").exists()


def test_reply_with_no_text_and_no_action_is_a_failure(isolated_transcripts):
    page = FakeChatPage([("ok", []), ("", []), ("bye", [])])

    with pytest.raises(AssertionError, match="empty reply on turn 2"):
        runner.test_multi_turn_conversation(page, CASE)

    assert (isolated_transcripts / "incomplete" / "scripted_case.json").exists()
