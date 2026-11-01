"""Playwright fixtures shared by tests/ and the live runner."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page

from config import load_settings
from pages.chat_page import ChatPage


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    return {**browser_context_args, "viewport": {"width": 1280, "height": 800}, "locale": "en-US"}


@pytest.fixture
def chat_page(page: Page) -> ChatPage:
    return ChatPage(page, load_settings())
