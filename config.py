"""Central configuration for Convo QA Lab.

Values come from environment variables (optionally loaded from a local `.env`).
The LLM backend is selected with LLM_PROVIDER:
  - "ollama" (default): local, free, no account needed
  - "azure": Azure OpenAI (needs the three AZURE_OPENAI_* variables)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GENERATED_TESTS_DIR = ROOT / "generated_tests"
GOLDEN_TESTS_DIR = ROOT / "golden_tests"
BENCHMARK_DIR = ROOT / "judge_benchmark"
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR") or ROOT / "transcripts")
EVALUATIONS_DIR = ROOT / "evaluations"
REPORTS_DIR = ROOT / "reports"
RUNS_FILENAME = "runs.jsonl"
REPEAT_FILENAME = "repeat_runs.json"
REPEAT_RESULTS_FILE = EVALUATIONS_DIR / REPEAT_FILENAME
JUDGE_RESULTS_FILE = BENCHMARK_DIR / "labeled_results.json"


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # LLM backend
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    ollama_assistant_model: str
    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    azure_api_version: str
    # Chat app under test
    chat_app_url: str
    chat_input_selector: str
    chat_send_selector: str
    chat_response_selector: str
    num_turns: int


def load_settings() -> Settings:
    """Read and validate settings from the environment."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in ("ollama", "azure"):
        raise ConfigError(f"LLM_PROVIDER must be 'ollama' or 'azure', got '{provider}'")

    azure = provider == "azure"
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return Settings(
        llm_provider=provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=ollama_model,
        ollama_assistant_model=os.getenv("OLLAMA_ASSISTANT_MODEL") or ollama_model,
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT") if azure else "",
        azure_api_key=_require("AZURE_OPENAI_API_KEY") if azure else "",
        azure_deployment=_require("AZURE_OPENAI_DEPLOYMENT_NAME") if azure else "",
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        chat_app_url=os.getenv("CHAT_APP_URL", "http://localhost:8000"),
        chat_input_selector=os.getenv("CHAT_INPUT_SELECTOR", "#chat-input"),
        chat_send_selector=os.getenv("CHAT_SEND_SELECTOR", "#send-btn"),
        chat_response_selector=os.getenv("CHAT_RESPONSE_SELECTOR", ".assistant-message"),
        num_turns=int(os.getenv("NUM_TURNS", "3")),
    )
