"""FastAPI backend for the demo in-car voice assistant.

This is the *system under test*: a small chat service backed by an LLM. It also
serves the demo frontend, so one process is enough:

    uvicorn server:app --port 8000
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from llm import assistant_model, complete

DEMO_APP_DIR = Path(__file__).resolve().parent / "demo_app"

ASSISTANT_SYSTEM_PROMPT = """You are Navi, a voice assistant inside a car. You help with \
navigation (routes, destinations, ETA, avoiding tolls or highways) and media controls \
(play, pause, skip, volume, playlists).

Style rules:
- Reply in one or two short, natural spoken sentences. No lists, no markdown, no emojis.
- If a request is ambiguous or missing details, ask ONE clarifying question.
- Remember what the user said earlier in the conversation and use it.
- If asked about something unrelated to driving, navigation or media, politely say you \
can only help with those.

Actions: you can hand the caller to a human agent and book car service appointments. To do \
either, finish your spoken reply with ONE tag: [[handoff]] to transfer to a human, or \
[[book_appointment: <day and time>]] once the user has told you a day and a time. If the user \
asks for a human, transfer them. Never write a tag unless you are really doing that action, and \
never say you transferred or booked something without writing its tag."""

ASSISTANT_TEMPERATURE = 0.3


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    # An assistant reply may be action-only (empty text) and is replayed as history.
    content: str = Field(max_length=2000)

    @model_validator(mode="after")
    def _user_text_required(self) -> ChatMessage:
        if self.role == "user" and not self.content:
            raise ValueError("user messages must not be empty")
        return self


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class Action(BaseModel):
    type: str
    value: str = ""


class ChatResponse(BaseModel):
    reply: str
    actions: list[Action] = []


ACTION_TAG = re.compile(r"\[\[\s*([A-Za-z_]+)\s*(?::\s*([^\]]*?))?\s*\]\]")


def extract_actions(raw: str) -> tuple[str, list[Action]]:
    """Split the model output into the spoken text and the machine-readable action tags."""
    actions = [
        Action(type=m.group(1).lower(), value=(m.group(2) or "").strip())
        for m in ACTION_TAG.finditer(raw)
    ]
    return " ".join(ACTION_TAG.sub(" ", raw).split()), actions


app = FastAPI(title="Demo Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Sync endpoint on purpose: FastAPI runs it in a worker thread, so the blocking
# LLM call does not stall the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user.")

    messages = [
        {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
        *(m.model_dump() for m in request.messages),
    ]
    try:
        reply = complete(messages, temperature=ASSISTANT_TEMPERATURE, model=assistant_model())
    except openai.APIConnectionError as err:
        raise HTTPException(
            status_code=503, detail="LLM backend unreachable. Is Ollama running?"
        ) from err
    except openai.APIStatusError as err:
        raise HTTPException(
            status_code=502, detail=f"LLM backend error ({err.status_code}). Is the model pulled?"
        ) from err

    spoken, actions = extract_actions(reply)
    # An action with no spoken text is model behaviour (reported by a rule), not a backend error.
    if not spoken and not actions:
        raise HTTPException(status_code=502, detail="LLM backend returned an empty reply.")
    return ChatResponse(reply=spoken, actions=actions)


# Mounted last so it never shadows the API routes above.
app.mount("/", StaticFiles(directory=DEMO_APP_DIR, html=True), name="demo")
