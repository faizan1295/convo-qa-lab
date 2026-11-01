"""Verifies the harness (ChatPage) against the real frontend + backend, LLM faked.

These tests prove the Playwright plumbing is trustworthy, so a live-run failure
points at model behaviour rather than at the test driver.
"""

from playwright.sync_api import expect

from pages.chat_page import ChatPage


def test_driver_reads_assistant_reply_from_real_ui(chat: ChatPage, fake_llm):
    fake_llm.queue("Sure, where would you like to go?")

    reply = chat.send_and_read_reply("um, like, navigate somewhere")

    assert reply == "Sure, where would you like to go?"


def test_multi_turn_sends_full_history_to_backend(chat: ChatPage, fake_llm):
    fake_llm.queue("Where to?", "Avoiding tolls, ETA 25 minutes.")

    chat.send_and_read_reply("navigate please")
    second = chat.send_and_read_reply("uh, avoid tolls")

    assert second == "Avoiding tolls, ETA 25 minutes."
    sent = fake_llm.last_call["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["content"] == "Where to?"
    expect(chat.replies).to_have_count(2)


def test_backend_failure_is_shown_as_error_not_assistant_message(chat: ChatPage, fake_llm):
    fake_llm.queue("")  # empty reply -> backend returns 502

    chat.send("hello?")

    expect(chat.page.locator(".error-message")).to_be_visible()
    expect(chat.replies).to_have_count(0)


def test_failed_turn_is_dropped_from_history(chat: ChatPage, fake_llm):
    fake_llm.queue("", "Okay, heading to the airport.")

    chat.send("hello?")
    expect(chat.page.locator(".error-message")).to_be_visible()
    chat.send_and_read_reply("airport please")

    assert [m["role"] for m in fake_llm.last_call["messages"]] == ["system", "user"]


def test_blank_input_is_not_sent(chat: ChatPage, fake_llm):
    chat.send("   ")

    expect(chat.page.locator(".user-message")).to_have_count(0)
    assert fake_llm.calls == []


def test_user_text_is_rendered_as_text_not_html(chat: ChatPage, fake_llm):
    fake_llm.queue("ok")

    chat.send_and_read_reply("<b>bold</b>")

    expect(chat.page.locator(".user-message")).to_have_text("<b>bold</b>")
    expect(chat.page.locator("#log b")).to_have_count(0)


def test_actions_are_exposed_to_the_driver_without_polluting_the_reply_text(
    chat: ChatPage, fake_llm
):
    fake_llm.queue("Connecting you to an agent. [[handoff]]")

    reply = chat.send_and_read_reply("I want a human")

    assert reply == "Connecting you to an agent."
    assert chat.last_reply_actions() == [{"type": "handoff", "value": ""}]


def test_reply_without_actions_reports_an_empty_list(chat: ChatPage, fake_llm):
    fake_llm.queue("Where to?")

    chat.send_and_read_reply("navigate")

    assert chat.last_reply_actions() == []


def test_action_only_reply_is_readable_without_waiting_for_text(chat: ChatPage, fake_llm):
    fake_llm.queue("[[handoff]]")

    reply = chat.send_and_read_reply("I want a human", timeout_ms=5_000)

    assert reply == ""
    assert chat.last_reply_actions() == [{"type": "handoff", "value": ""}]
