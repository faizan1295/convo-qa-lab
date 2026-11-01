"""Provider-agnostic LLM wrapper shared by all components (Ollama or Azure OpenAI)."""

from __future__ import annotations

import json
from functools import lru_cache

from openai import AzureOpenAI, OpenAI

from config import load_settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    s = load_settings()
    if s.llm_provider == "azure":
        return AzureOpenAI(
            azure_endpoint=s.azure_endpoint,
            api_key=s.azure_api_key,
            api_version=s.azure_api_version,
        )
    # Ollama exposes an OpenAI-compatible API; the key is required by the SDK but ignored.
    return OpenAI(base_url=s.ollama_base_url, api_key="ollama")


def _model() -> str:
    s = load_settings()
    return s.azure_deployment if s.llm_provider == "azure" else s.ollama_model


def assistant_model() -> str:
    """Model that powers the demo assistant (may differ from the test tooling model)."""
    s = load_settings()
    return s.azure_deployment if s.llm_provider == "azure" else s.ollama_assistant_model


def complete(
    messages: list[dict],
    *,
    json_mode: bool = False,
    temperature: float = 0.7,
    model: str | None = None,
) -> str:
    """Run a chat completion on the configured backend.

    With `json_mode=True` the model is constrained to return a JSON object
    (the prompt must mention "JSON"). `model` overrides the default model.
    """
    kwargs: dict = {"temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _client().chat.completions.create(
        model=model or _model(), messages=messages, **kwargs
    )
    return (response.choices[0].message.content or "").strip()


def complete_json(messages: list[dict], *, temperature: float = 0.7, retries: int = 2) -> dict:
    """Like `complete`, but parses and returns the JSON object.

    Small local models occasionally emit malformed JSON, so retry a couple of times.
    """
    last_error: Exception | None = None
    for _ in range(retries + 1):
        raw = complete(messages, json_mode=True, temperature=temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as err:
            last_error = err
    raise ValueError(
        f"Model did not return valid JSON after {retries + 1} attempts"
    ) from last_error
