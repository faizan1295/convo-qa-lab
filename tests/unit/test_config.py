import pytest

from config import ConfigError, load_settings


def test_defaults_to_ollama_without_any_env():
    s = load_settings()

    assert s.llm_provider == "ollama"
    assert s.ollama_model == "llama3.1:8b"
    assert s.num_turns == 3
    assert s.chat_response_selector == ".assistant-message"


def test_assistant_model_falls_back_to_main_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")

    assert load_settings().ollama_assistant_model == "qwen2.5:14b"


def test_blank_assistant_model_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("OLLAMA_ASSISTANT_MODEL", "")

    assert load_settings().ollama_assistant_model == "qwen2.5:14b"


def test_assistant_model_can_differ(monkeypatch):
    monkeypatch.setenv("OLLAMA_ASSISTANT_MODEL", "phi3")

    assert load_settings().ollama_assistant_model == "phi3"


def test_numeric_settings_are_parsed_as_int(monkeypatch):
    monkeypatch.setenv("NUM_TURNS", "5")

    assert load_settings().num_turns == 5


def test_provider_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Ollama ")

    assert load_settings().llm_provider == "ollama"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")

    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings()


@pytest.mark.parametrize(
    "missing",
    ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"],
)
def test_azure_requires_each_credential(monkeypatch, missing):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        if name != missing:
            monkeypatch.setenv(name, "x")

    with pytest.raises(ConfigError, match=missing):
        load_settings()


def test_azure_settings_load_when_complete(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    s = load_settings()

    assert s.azure_deployment == "gpt-4o-mini"
    assert s.azure_api_version == "2024-10-21"
