import pytest
from openai import AzureOpenAI, OpenAI

import llm

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def fresh_client_cache():
    llm._client.cache_clear()
    yield
    llm._client.cache_clear()


class TestComplete:
    def test_returns_stripped_content(self, fake_llm):
        fake_llm.queue("  hello there \n")

        assert llm.complete(MESSAGES) == "hello there"

    def test_uses_ollama_model_by_default(self, fake_llm):
        fake_llm.queue("ok")

        llm.complete(MESSAGES)

        assert fake_llm.last_call["model"] == "llama3.1:8b"

    def test_model_override_wins(self, fake_llm):
        fake_llm.queue("ok")

        llm.complete(MESSAGES, model="phi3")

        assert fake_llm.last_call["model"] == "phi3"

    def test_azure_uses_deployment_name_as_model(self, fake_llm, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://e.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deployment")
        fake_llm.queue("ok")

        llm.complete(MESSAGES)

        assert fake_llm.last_call["model"] == "my-deployment"

    def test_json_mode_sets_response_format(self, fake_llm):
        fake_llm.queue("{}")

        llm.complete(MESSAGES, json_mode=True)

        assert fake_llm.last_call["response_format"] == {"type": "json_object"}

    def test_plain_mode_omits_response_format(self, fake_llm):
        fake_llm.queue("ok")

        llm.complete(MESSAGES)

        assert "response_format" not in fake_llm.last_call

    def test_none_content_becomes_empty_string(self, fake_llm):
        fake_llm.queue(None)

        assert llm.complete(MESSAGES) == ""


class TestCompleteJson:
    def test_parses_valid_json(self, fake_llm):
        fake_llm.queue('{"a": 1}')

        assert llm.complete_json(MESSAGES) == {"a": 1}

    def test_retries_after_malformed_json(self, fake_llm):
        fake_llm.queue("not json", '{"ok": true}')

        assert llm.complete_json(MESSAGES) == {"ok": True}
        assert len(fake_llm.calls) == 2

    def test_raises_after_retries_exhausted(self, fake_llm):
        fake_llm.queue("bad", "bad", "bad")

        with pytest.raises(ValueError, match="valid JSON"):
            llm.complete_json(MESSAGES, retries=2)

        assert len(fake_llm.calls) == 3


class TestAssistantModel:
    def test_defaults_to_main_ollama_model(self):
        assert llm.assistant_model() == "llama3.1:8b"

    def test_uses_dedicated_assistant_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_ASSISTANT_MODEL", "phi3")

        assert llm.assistant_model() == "phi3"


class TestClientFactory:
    def test_ollama_points_at_local_openai_compatible_endpoint(self, fresh_client_cache):
        client = llm._client()

        assert type(client) is OpenAI
        assert str(client.base_url).startswith("http://localhost:11434/v1")

    def test_azure_builds_azure_client(self, fresh_client_cache, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://e.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "d")

        assert isinstance(llm._client(), AzureOpenAI)
