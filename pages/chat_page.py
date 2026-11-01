"""Page Object for the chat app under test.

All selectors come from Settings, so the same object drives the bundled demo app
or any other chat UI configured through CHAT_*_SELECTOR.
"""

from __future__ import annotations

import json
import time

from playwright.sync_api import Locator, Page, expect

from config import Settings

REPLY_TIMEOUT_MS = 30_000
STREAM_SETTLE_MS = 1_000
POLL_INTERVAL_MS = 200


class ChatPage:
    def __init__(self, page: Page, settings: Settings) -> None:
        self.page = page
        self.settings = settings
        self.input = page.locator(settings.chat_input_selector)
        self.send_button: Locator | None = (
            page.locator(settings.chat_send_selector) if settings.chat_send_selector else None
        )
        self.replies = page.locator(settings.chat_response_selector)

    def open(self, url: str | None = None) -> None:
        self.page.goto(url or self.settings.chat_app_url)
        expect(self.input).to_be_visible()

    def send(self, message: str) -> None:
        self.input.fill(message)
        if self.send_button:
            self.send_button.click()
        else:
            self.input.press("Enter")

    def send_and_read_reply(self, message: str, timeout_ms: int = REPLY_TIMEOUT_MS) -> str:
        """Send `message` and return the complete assistant reply.

        Waits for a new reply element, then until its text stops changing so
        streamed replies are captured in full.
        """
        before = self.replies.count()
        self.send(message)
        expect(self.replies.nth(before)).to_be_visible(timeout=timeout_ms)
        return self._text_once_stable(self.replies.last, timeout_ms)

    @staticmethod
    def _has_actions_marker(locator: Locator) -> bool:
        """True once the app has rendered the reply, even if it has no spoken text."""
        return locator.get_attribute("data-actions") is not None

    def last_reply_actions(self) -> list[dict]:
        """Machine-readable actions (handoff, booking, ...) the app attached to the last reply."""
        raw = self.replies.last.get_attribute("data-actions")
        return json.loads(raw) if raw else []

    def _text_once_stable(self, locator: Locator, timeout_ms: int) -> str:
        last_text, stable_since = "", time.monotonic()
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            current = locator.inner_text().strip()
            if current != last_text:
                last_text, stable_since = current, time.monotonic()
            elif (current or self._has_actions_marker(locator)) and (
                time.monotonic() - stable_since
            ) * 1000 >= STREAM_SETTLE_MS:
                return current
            self.page.wait_for_timeout(POLL_INTERVAL_MS)
        return last_text
