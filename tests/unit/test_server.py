import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import server

USER_MSG = {"role": "user", "content": "take me home"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


class TestChat:
    def test_returns_model_reply(self, client, fake_llm):
        fake_llm.queue("Sure, heading home.")

        response = client.post("/chat", json={"messages": [USER_MSG]})

        assert response.status_code == 200
        assert response.json() == {"reply": "Sure, heading home.", "actions": []}

    def test_prepends_system_prompt_and_keeps_history_order(self, client, fake_llm):
        fake_llm.queue("ok")
        history = [
            USER_MSG,
            {"role": "assistant", "content": "Which route?"},
            {"role": "user", "content": "the fastest"},
        ]

        client.post("/chat", json={"messages": history})

        sent = fake_llm.last_call["messages"]
        assert sent[0] == {"role": "system", "content": server.ASSISTANT_SYSTEM_PROMPT}
        assert sent[1:] == history

    def test_uses_assistant_model_and_low_temperature(self, client, fake_llm, monkeypatch):
        monkeypatch.setenv("OLLAMA_ASSISTANT_MODEL", "phi3")
        fake_llm.queue("ok")

        client.post("/chat", json={"messages": [USER_MSG]})

        assert fake_llm.last_call["model"] == "phi3"
        assert fake_llm.last_call["temperature"] == server.ASSISTANT_TEMPERATURE

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"messages": []},
            {"messages": [{"role": "system", "content": "x"}]},
            {"messages": [{"role": "user", "content": ""}]},
            {"messages": [{"role": "user", "content": "x" * 2001}]},
            {"messages": [USER_MSG] * 41},
        ],
        ids=["no-field", "empty", "bad-role", "empty-content", "too-long", "too-many"],
    )
    def test_invalid_payload_is_rejected_before_calling_llm(self, client, fake_llm, body):
        response = client.post("/chat", json=body)

        assert response.status_code == 422
        assert fake_llm.calls == []

    def test_last_message_must_be_from_user(self, client, fake_llm):
        body = {"messages": [USER_MSG, {"role": "assistant", "content": "Where to?"}]}

        response = client.post("/chat", json=body)

        assert response.status_code == 422
        assert fake_llm.calls == []

    def test_empty_llm_reply_is_a_502(self, client, fake_llm):
        fake_llm.queue("")

        response = client.post("/chat", json={"messages": [USER_MSG]})

        assert response.status_code == 502


class TestUpstreamFailures:
    def _failing(self, monkeypatch, error: Exception) -> None:
        def boom(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(server, "complete", boom)

    def test_unreachable_backend_is_a_503(self, client, monkeypatch):
        request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        self._failing(monkeypatch, openai.APIConnectionError(request=request))

        response = client.post("/chat", json={"messages": [USER_MSG]})

        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]

    def test_backend_http_error_is_a_502(self, client, monkeypatch):
        request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        http_response = httpx.Response(404, request=request)
        error = openai.NotFoundError("model not found", response=http_response, body=None)
        self._failing(monkeypatch, error)

        response = client.post("/chat", json={"messages": [USER_MSG]})

        assert response.status_code == 502
        assert "404" in response.json()["detail"]


class TestStaticAndHealth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_serves_frontend_with_selectors_the_runner_relies_on(self, client):
        html = client.get("/").text

        for hook in ('id="chat-input"', 'id="send-btn"', "assistant-message"):
            assert hook in html


class TestActions:
    def _chat(self, client, fake_llm, raw):
        fake_llm.queue(raw)
        return client.post("/chat", json={"messages": [USER_MSG]}).json()

    def test_handoff_tag_becomes_an_action_and_is_not_spoken(self, client, fake_llm):
        body = self._chat(client, fake_llm, "Connecting you to an agent now. [[handoff]]")

        assert body == {
            "reply": "Connecting you to an agent now.",
            "actions": [{"type": "handoff", "value": ""}],
        }

    def test_booking_tag_keeps_its_value(self, client, fake_llm):
        body = self._chat(client, fake_llm, "Booked. [[book_appointment: Tuesday 3 pm]]")

        assert body["actions"] == [{"type": "book_appointment", "value": "Tuesday 3 pm"}]

    def test_unknown_tags_are_reported_not_hidden(self, client, fake_llm):
        body = self._chat(client, fake_llm, "Done. [[Delete_Account]]")

        assert body["actions"] == [{"type": "delete_account", "value": ""}]
        assert "[[" not in body["reply"]

    def test_action_without_spoken_text_is_returned_not_hidden_as_a_backend_error(
        self, client, fake_llm
    ):
        body = self._chat(client, fake_llm, "[[handoff]]")

        assert body == {"reply": "", "actions": [{"type": "handoff", "value": ""}]}

    def test_reply_with_neither_text_nor_action_is_still_a_backend_error(self, client, fake_llm):
        fake_llm.queue("   ")

        assert client.post("/chat", json={"messages": [USER_MSG]}).status_code == 502

    def test_system_prompt_explains_the_action_tags(self):
        assert "[[handoff]]" in server.ASSISTANT_SYSTEM_PROMPT
        assert "[[book_appointment:" in server.ASSISTANT_SYSTEM_PROMPT
