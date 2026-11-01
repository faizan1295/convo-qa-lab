"""LLM role-play of the human user, used by the live runner."""

from __future__ import annotations

import json

from llm import complete

USER_SIM_SYSTEM = """You role-play a human talking to a voice/chat assistant. \
Write ONLY the user's next message, nothing else.

Rules:
- Sound spoken and colloquial: contractions, filler words ("um", "uh", "like", \
"you know", "so yeah"), the odd self-correction or trailing thought.
- Stay in character for the persona below and pursue the goals gradually, one \
or two per turn; do not dump everything at once.
- React naturally to what the assistant just said (confusion, thanks, a change \
of mind, a clarification) rather than repeating yourself.
- Keep it under 35 words. No quotation marks, no stage directions."""


def simulate_user_message(case: dict, transcript: list[dict]) -> str:
    """Generate the user's next utterance. Empty transcript => opening message."""
    if transcript:
        history = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript)
        instruction = "Conversation so far:\n" + history + "\n\nWrite the user's next message."
    else:
        instruction = "The conversation has not started. Write the user's opening message."

    context = (
        f"Domain: {case['domain']}\n"
        f"Persona: {case['user_persona']}\n"
        f"Scenario: {case['description']}\n"
        f"Goals: {json.dumps(case['user_goals'])}\n\n{instruction}"
    )
    text = complete(
        [{"role": "system", "content": USER_SIM_SYSTEM}, {"role": "user", "content": context}],
        temperature=0.9,
    )
    return text.strip().strip('"')
