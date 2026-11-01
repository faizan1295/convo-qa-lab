"""Deterministic behaviour checks on a recorded transcript. No LLM involved.

Three rule families, all cheap to audit:
  - grounding: the assistant must not state concrete specifics (addresses, distances,
    times, compass directions) that the user never provided. The demo assistant has no
    maps or sensors, so any such specific is invented.
  - style: the assistant's own spec asks for 1-2 spoken sentences, no markdown, no emoji.
  - actions: handoff and booking are machine-readable actions attached to a reply. Saying
    "I'm transferring you" without the action is a failed handoff; booking a time the user never
    gave is an invented detail. Per-case expected_actions / forbidden_actions check outcomes.

Usage:
    python behavior_checks.py transcripts/        # exit code 1 if any violation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_SENTENCES = 2
KNOWN_ACTIONS = ("handoff", "book_appointment")

_UNITS = r"(?:km|kilomet(?:er|re)s?|miles?|mi|met(?:er|re)s?|minutes?|mins?|hours?|blocks?)"
_COUNT_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|a few|a couple of|several)"
_STREET_WORDS = r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)"
_ORDINAL = r"\d+(?:st|nd|rd|th)"
_NAME = r"[A-Z][a-z]+"

QUANTITY = re.compile(
    rf"\b(?:\d+(?:\.\d+)?|{_COUNT_WORDS})\s*{_UNITS}\b|\b(?:a|one)\s+(?:block|mile|kilomet(?:er|re))\b",
    re.IGNORECASE,
)
INTERSECTION = re.compile(
    rf"\b(?:{_ORDINAL}\s+(?:and|&)\s+(?:{_ORDINAL}|{_NAME})|{_NAME}\s+(?:and|&)\s+{_ORDINAL})\b"
)
STREET = re.compile(rf"\b(?:\d{{1,5}}\s+)?{_NAME}\s+{_STREET_WORDS}\b")
DIRECTION_CLAIM = re.compile(
    r"\byou(?:'re| are)\s+(?:currently\s+|already\s+)?(?:headed|heading|driving|going|travell?ing)"
    r"\s+(north|south|east|west)",
    re.IGNORECASE,
)
MARKDOWN = re.compile(r"\*\*|__|`|^\s*(?:[-*•]|\d+[.)])\s+|^#{1,6}\s", re.MULTILINE)
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


_NEGATION = re.compile(
    r"\b(?:can't|cannot|can not|unable|not able|won't|will not|don't|do not|couldn't)\b",
    re.IGNORECASE,
)
HANDOFF_CLAIM = re.compile(
    r"\b(?:transferring|connecting|handing)\s+you\b"
    r"|\bi(?:'ll| will|'m going to| am going to)\s+(?:transfer|connect|hand)\s+you\b"
    r"|\blet me (?:transfer|connect)\b",
    re.IGNORECASE,
)
BOOKING_CLAIM = re.compile(
    r"\bi(?:'ve| have)?\s*(?:booked|scheduled)\b"
    r"|\b(?:appointment|booking)\s+(?:is |has been |was )?(?:booked|scheduled|confirmed|set)\b"
    r"|\b(?:is|are|been)\s+(?:booked|scheduled|confirmed)\b"
    r"|\byou(?:'re| are) all set\b",
    re.IGNORECASE,
)
_WHEN_TOKEN = re.compile(
    r"\d+(?::\d+)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|tomorrow|today|morning|afternoon|evening",
    re.IGNORECASE,
)

CLAIMS = (
    ("handoff_claimed_not_performed", HANDOFF_CLAIM, "handoff"),
    ("booking_claimed_not_performed", BOOKING_CLAIM, "book_appointment"),
)


@dataclass(frozen=True)
class Violation:
    rule: str
    turn: int
    evidence: str


def _normalize(text: str) -> str:
    return text.replace("’", "'")


def _contains_word(haystack: str, needle: str) -> bool:
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def _quantity_grounded(match: re.Match, user_text: str) -> bool:
    phrase = " ".join(match.group(0).lower().split())
    number = phrase.split()[0]
    return phrase in user_text or (number[0].isdigit() and _contains_word(user_text, number))


def _phrase_grounded(match: re.Match, user_text: str) -> bool:
    return match.group(0).lower() in user_text


def _direction_grounded(match: re.Match, user_text: str) -> bool:
    return _contains_word(user_text, match.group(1).lower())


GROUNDING_RULES: tuple[tuple[str, re.Pattern, Callable[[re.Match, str], bool]], ...] = (
    ("invented_quantity", QUANTITY, _quantity_grounded),
    ("invented_intersection", INTERSECTION, _phrase_grounded),
    ("invented_street", STREET, _phrase_grounded),
    ("invented_direction", DIRECTION_CLAIM, _direction_grounded),
)


def _grounding_violations(text: str, user_text: str, turn: int) -> list[Violation]:
    return [
        Violation(rule, turn, match.group(0))
        for rule, pattern, is_grounded in GROUNDING_RULES
        for match in pattern.finditer(text)
        if not is_grounded(match, user_text)
    ]


def _style_violations(text: str, turn: int) -> list[Violation]:
    violations = []
    if MARKDOWN.search(text):
        violations.append(Violation("markdown_in_voice_reply", turn, text[:60]))
    if EMOJI.search(text):
        violations.append(Violation("emoji_in_voice_reply", turn, text[:60]))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) > MAX_SENTENCES:
        violations.append(Violation("reply_too_long", turn, f"{len(sentences)} sentences"))
    return violations


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _action_violations(
    text: str, actions: list[dict], user_text: str, turn: int
) -> list[Violation]:
    """Said-vs-done mismatches, unknown actions and booking details the user never gave."""
    violations = []
    types = {a.get("type") for a in actions}
    for rule, pattern, action_type in CLAIMS:
        if action_type in types:
            continue
        for sentence in _sentences(text):
            if (
                not sentence.endswith("?")
                and pattern.search(sentence)
                and not _NEGATION.search(sentence)
            ):
                violations.append(Violation(rule, turn, sentence))
                break
    for a in actions:
        if a.get("type") not in KNOWN_ACTIONS:
            violations.append(Violation("unknown_action", turn, str(a.get("type"))))
        elif a["type"] == "book_appointment":
            tokens = _WHEN_TOKEN.findall(a.get("value", ""))
            if not tokens:
                violations.append(Violation("booking_without_time", turn, a.get("value", "")))
            for token in tokens:
                if not re.search(rf"(?<![\d:]){re.escape(token.lower())}(?![\d:])", user_text):
                    violations.append(
                        Violation("invented_booking_detail", turn, f"{token!r} in {a['value']!r}")
                    )
    return violations


def check_transcript(turns: list[dict]) -> list[Violation]:
    """Return every rule violation in the assistant's replies, in transcript order.

    Action rules only run on turns that carry an "actions" list (older transcripts do not).
    """
    violations: list[Violation] = []
    user_text = ""
    for t in turns:
        text = _normalize(t["text"])
        if t["role"] == "user":
            user_text += " " + text.lower()
            continue
        if not text.strip() and t.get("actions"):
            violations.append(
                Violation("no_spoken_reply", t["turn"], f"actions only: {t['actions']}")
            )
        violations += _grounding_violations(text, user_text, t["turn"])
        violations += _style_violations(text, t["turn"])
        if "actions" in t:
            violations += _action_violations(text, t["actions"], user_text, t["turn"])
    return violations


def _performed(turns: list[dict]) -> list[tuple[int, dict]]:
    return [(t["turn"], a) for t in turns if t["role"] == "assistant" for a in t.get("actions", [])]


def check_outcomes(turns: list[dict], case: dict) -> list[Violation]:
    """Per-case outcome validation against what the assistant actually did.

    expected_actions: [{"type", "by_turn"?, "value_contains"?}], each must happen.
    forbidden_actions: [{"type", "before_turn"?}], none may happen.
    """
    performed = _performed(turns)
    last_turn = max((t["turn"] for t in turns), default=0)
    violations = []
    for spec in case.get("expected_actions", []):
        needle = spec.get("value_contains", "").lower()
        met = any(
            a.get("type") == spec["type"]
            and turn <= spec.get("by_turn", last_turn)
            and needle in a.get("value", "").lower()
            for turn, a in performed
        )
        if not met:
            detail = spec["type"] + (f" containing {needle!r}" if needle else "")
            violations.append(
                Violation("missing_action", spec.get("by_turn", last_turn), f"expected {detail}")
            )
    for spec in case.get("forbidden_actions", []):
        for turn, a in performed:
            if a.get("type") == spec["type"] and turn < spec.get("before_turn", last_turn + 1):
                violations.append(
                    Violation(
                        "forbidden_action", turn, f"{a['type']}: {a.get('value', '')}".strip(": ")
                    )
                )
    return violations


def check_case(case: dict, turns: list[dict]) -> list[Violation]:
    """Every deterministic check for one recorded case: rules, forbidden phrases, outcomes."""
    return (
        check_transcript(turns)
        + check_forbidden_phrases(turns, case.get("forbidden_phrases", []))
        + check_outcomes(turns, case)
    )


def check_forbidden_phrases(turns: list[dict], phrases: list[str]) -> list[Violation]:
    """Case-specific tripwires: the assistant must never say any of these phrases.

    Use only distinctive phrases (a leaked system-prompt sentence, a forbidden answer) that a
    polite refusal would not repeat.
    """
    violations = []
    for t in turns:
        if t["role"] != "assistant":
            continue
        text = " ".join(_normalize(t["text"]).lower().split())
        violations += [
            Violation("forbidden_phrase", t["turn"], p)
            for p in phrases
            if " ".join(p.lower().split()) in text
        ]
    return violations


def violations_as_dicts(violations: list[Violation]) -> list[dict]:
    return [asdict(v) for v in violations]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Transcript JSON file or a directory of them")
    args = parser.parse_args()

    target = Path(args.path)
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not files:
        print(f"No transcripts found at {target}", file=sys.stderr)
        return 1

    clean = True
    for f in files:
        data = json.loads(f.read_text())
        found = check_case(data["test_case"], data["transcript"])
        name = data["test_case"]["test_name"]
        print(f"{'FAIL' if found else 'PASS'}  {name}")
        for v in found:
            print(f"      turn {v.turn}  {v.rule}: {v.evidence!r}")
        clean &= not found
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
