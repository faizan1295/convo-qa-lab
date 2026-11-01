"""Pytest + Playwright multi-turn driver for a chat assistant web app.

For every generated test case the runner:
  1. opens the chat app,
  2. asks the LLM to role-play the user persona and write a colloquial message,
  3. types it into the UI and reads the assistant's reply from the DOM,
  4. feeds the reply back to the LLM to produce the user's natural follow-up,
     and repeats for NUM_TURNS turns,
  5. writes the full transcript to transcripts/<test_name>.json.

Run:  pytest e2e_assistant_runner.py --headed
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from config import GENERATED_TESTS_DIR, GOLDEN_TESTS_DIR, TRANSCRIPTS_DIR, load_settings
from pages.chat_page import ChatPage
from user_simulator import simulate_user_message

pytestmark = pytest.mark.e2e


def load_cases() -> list[dict]:
    """Load test cases from TEST_CASES_FILE, or every file in golden_tests/ and generated_tests/."""
    override = os.getenv("TEST_CASES_FILE")
    files = (
        [Path(override)]
        if override
        else sorted(GOLDEN_TESTS_DIR.glob("*.json")) + sorted(GENERATED_TESTS_DIR.glob("*.json"))
    )
    cases: list[dict] = []
    for f in files:
        data = json.loads(f.read_text())
        for case in data["test_cases"]:
            cases.append({**case, "domain": data.get("domain", f.stem)})
    return cases


CASES = load_cases()


@pytest.mark.skipif(not CASES, reason="No test cases found. Run test_generator.py first.")
@pytest.mark.parametrize("case", CASES, ids=[c["test_name"] for c in CASES])
def test_multi_turn_conversation(chat_page: ChatPage, case: dict) -> None:
    chat_page.open()

    transcript: list[dict] = []
    try:
        _converse(chat_page, case, transcript)
    except BaseException:
        # Keep the evidence of a crashed run; the evaluator only reads transcripts/*.json.
        _save(case, transcript, TRANSCRIPTS_DIR / "incomplete")
        raise
    _save(case, transcript, TRANSCRIPTS_DIR)


def _converse(chat_page: ChatPage, case: dict, transcript: list[dict]) -> None:
    scripted = case.get("scripted_user_turns")
    for turn in range(1, (len(scripted) if scripted else load_settings().num_turns) + 1):
        user_text = scripted[turn - 1] if scripted else simulate_user_message(case, transcript)
        transcript.append({"turn": turn, "role": "user", "text": user_text})

        assistant_text = chat_page.send_and_read_reply(user_text)
        actions = chat_page.last_reply_actions()
        transcript.append(
            {"turn": turn, "role": "assistant", "text": assistant_text, "actions": actions}
        )

        print(
            f"[turn {turn}] USER: {user_text}\n[turn {turn}] ASSISTANT: {assistant_text}"
            + (f"  ACTIONS: {actions}" if actions else "")
        )
        assert assistant_text or actions, f"Assistant returned an empty reply on turn {turn}"


def _save(case: dict, transcript: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{case['test_name']}.json"
    out.write_text(json.dumps({"test_case": case, "transcript": transcript}, indent=2))
    print(f"Transcript saved to {out}")
