"""Boots the real demo app (FastAPI + frontend) on a free port; the LLM stays faked."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

import server
from pages.chat_page import ChatPage

STARTUP_TIMEOUT_S = 10


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(fake_llm) -> Iterator[str]:
    config = uvicorn.Config(server.app, host="127.0.0.1", port=_free_port(), log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not uvicorn_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Demo app did not start in time")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{config.port}"

    uvicorn_server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def chat(chat_page: ChatPage, live_server: str) -> ChatPage:
    chat_page.open(live_server)
    return chat_page
