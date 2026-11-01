import user_simulator as runner


def test_opening_message_prompt_has_no_history(fake_llm, sample_case):
    fake_llm.queue("um, so I need to get to the airport")

    text = runner.simulate_user_message(sample_case, [])

    assert text == "um, so I need to get to the airport"
    assert "has not started" in fake_llm.last_prompt


def test_follow_up_prompt_contains_conversation_history(fake_llm, sample_case):
    fake_llm.queue("uh, actually skip the tolls")
    transcript = [
        {"turn": 1, "role": "user", "text": "airport please"},
        {"turn": 1, "role": "assistant", "text": "Starting route to the airport."},
    ]

    runner.simulate_user_message(sample_case, transcript)

    assert "USER: airport please" in fake_llm.last_prompt
    assert "ASSISTANT: Starting route to the airport." in fake_llm.last_prompt
    assert "next message" in fake_llm.last_prompt


def test_prompt_carries_persona_and_goals(fake_llm, sample_case):
    fake_llm.queue("hi")

    runner.simulate_user_message(sample_case, [])

    assert "Rushed commuter" in fake_llm.last_prompt
    assert "Avoid tolls" in fake_llm.last_prompt


def test_surrounding_quotes_are_stripped(fake_llm, sample_case):
    fake_llm.queue('  "like, play some jazz"  ')

    assert runner.simulate_user_message(sample_case, []) == "like, play some jazz"


def test_system_prompt_demands_filler_words(fake_llm, sample_case):
    fake_llm.queue("hi")

    runner.simulate_user_message(sample_case, [])

    system = fake_llm.last_call["messages"][0]["content"]
    assert "filler words" in system
    assert fake_llm.last_call["temperature"] >= 0.8
