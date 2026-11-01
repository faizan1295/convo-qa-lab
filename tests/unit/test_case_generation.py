import json
from pathlib import Path

import pytest

import test_generator as generator


def _case(name: str) -> dict:
    return {
        "test_name": name,
        "description": "d",
        "user_persona": "p",
        "user_goals": ["u1", "u2"],
        "expected_behavior": ["asks one clarifying question"],
        "must_not": ["invent an address"],
    }


def _payload(count: int = 3) -> str:
    return json.dumps({"test_cases": [_case(f"case_{i}") for i in range(count)]})


def test_returns_requested_number_of_cases(fake_llm):
    fake_llm.queue(_payload(3))

    cases = generator.generate_test_cases("Car Navigation", count=3)

    assert [c["test_name"] for c in cases] == ["case_0", "case_1", "case_2"]


def test_default_count_covers_every_category(fake_llm):
    n = len(generator.CATEGORIES)
    fake_llm.queue(_payload(n))

    cases = generator.generate_test_cases("Car Navigation")

    assert [c["category"] for c in cases] == list(generator.CATEGORIES)


def test_category_is_assigned_by_code_not_by_the_model(fake_llm):
    payload = json.loads(_payload(1))
    payload["test_cases"][0]["category"] = "whatever_the_model_said"
    fake_llm.queue(json.dumps(payload))

    cases = generator.generate_test_cases("Car Navigation", count=1)

    assert cases[0]["category"] == "ambiguity"


def test_prompt_lists_categories_and_separates_user_from_assistant(fake_llm):
    fake_llm.queue(_payload(2))

    generator.generate_test_cases("Media Controls", count=2)

    prompt = fake_llm.last_prompt
    assert "Media Controls" in prompt
    assert "exactly 2" in prompt
    assert "1. ambiguity" in prompt and "2. change_of_mind" in prompt
    assert "user_goals" in prompt and "expected_behavior" in prompt and "must_not" in prompt
    assert 'never as "Fails to ..."' in prompt


def test_requests_json_mode(fake_llm):
    fake_llm.queue(_payload(1))

    generator.generate_test_cases("Car Navigation", count=1)

    assert fake_llm.last_call["response_format"] == {"type": "json_object"}


def test_object_persona_is_flattened_to_a_string(fake_llm):
    payload = json.loads(_payload(1))
    payload["test_cases"][0]["user_persona"] = {"mood": "frustrated", "expertise": "beginner"}
    fake_llm.queue(json.dumps(payload))

    cases = generator.generate_test_cases("Car Navigation", count=1)

    assert cases[0]["user_persona"] == "mood: frustrated, expertise: beginner"


@pytest.mark.parametrize("missing", generator.REQUIRED_FIELDS)
def test_rejects_case_missing_a_required_field(fake_llm, missing):
    broken = _case("broken")
    del broken[missing]
    fake_llm.queue(json.dumps({"test_cases": [broken]}))

    with pytest.raises(ValueError, match=missing):
        generator.generate_test_cases("Car Navigation", count=1)


@pytest.mark.parametrize("field", generator.LIST_FIELDS)
def test_rejects_empty_list_field(fake_llm, field):
    broken = {**_case("broken"), field: []}
    fake_llm.queue(json.dumps({"test_cases": [broken]}))

    with pytest.raises(ValueError, match=field):
        generator.generate_test_cases("Car Navigation", count=1)


@pytest.mark.parametrize("field", generator.LIST_FIELDS)
def test_rejects_list_field_given_as_string(fake_llm, field):
    broken = {**_case("broken"), field: "not a list"}
    fake_llm.queue(json.dumps({"test_cases": [broken]}))

    with pytest.raises(ValueError, match="must be lists"):
        generator.generate_test_cases("Car Navigation", count=1)


def test_rejects_wrong_number_of_cases(fake_llm):
    fake_llm.queue(_payload(2))

    with pytest.raises(ValueError, match="Expected 3"):
        generator.generate_test_cases("Car Navigation", count=3)


def test_rejects_response_without_test_cases_key(fake_llm):
    fake_llm.queue("{}")

    with pytest.raises(ValueError, match="Expected 3"):
        generator.generate_test_cases("Car Navigation", count=3)


def test_save_writes_slugged_file_with_domain(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "GENERATED_TESTS_DIR", tmp_path / "out")
    cases = [_case("a")]

    path = generator.save_test_cases("Car Navigation & Maps!", cases)

    assert path.endswith("car_navigation_maps.json")
    saved = json.loads(Path(path).read_text())
    assert saved == {"domain": "Car Navigation & Maps!", "test_cases": cases}


def test_prompt_demands_speakable_goals_and_consistent_expectations(fake_llm):
    fake_llm.queue(_payload(2))

    generator.generate_test_cases("Media Controls", count=2)

    prompt = fake_llm.last_prompt
    assert "literally say out loud" in prompt
    assert '"realize"' in prompt
    assert "do NOT already state" in prompt
