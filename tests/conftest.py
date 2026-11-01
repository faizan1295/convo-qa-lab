"""Shared fixtures: hermetic environment and a fake LLM client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm

CONFIG_ENV_VARS = (
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_ASSISTANT_MODEL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
    "CHAT_APP_URL",
    "CHAT_INPUT_SELECTOR",
    "CHAT_SEND_SELECTOR",
    "CHAT_RESPONSE_SELECTOR",
    "NUM_TURNS",
    "TEST_CASES_FILE",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag tests by directory so `-m unit` / `-m integration` work without decorators."""
    for item in items:
        folder = item.path.parent.name
        if folder in ("unit", "integration"):
            item.add_marker(getattr(pytest.mark, folder))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore the developer's real .env so every test starts from the defaults."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeLLM:
    """Stands in for the OpenAI client: returns queued replies, records every call."""

    def __init__(self) -> None:
        self._replies: list[str] = []
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def queue(self, *replies: str) -> None:
        self._replies.extend(replies)

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._replies:
            raise AssertionError("FakeLLM received a call but no reply was queued")
        content = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    @property
    def last_call(self) -> dict:
        return self.calls[-1]

    @property
    def last_prompt(self) -> str:
        """All message contents of the last call joined, for substring assertions."""
        return "\n".join(m["content"] for m in self.last_call["messages"])


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(llm, "_client", lambda: fake)
    return fake


@pytest.fixture
def sample_case() -> dict:
    return {
        "test_name": "reroute_after_toll_change",
        "description": "User changes mind about tolls mid-route.",
        "user_persona": "Rushed commuter, informal, interrupts themselves",
        "category": "change_of_mind",
        "user_goals": ["Set destination", "Avoid tolls", "Ask for the new ETA"],
        "expected_behavior": ["Follows the latest request and avoids tolls"],
        "must_not": ["States an exact ETA it cannot know"],
        "domain": "Car Navigation",
    }
