import json

import pytest

from askjudge import (
    DEFAULT_ASSISTANCE_POLICY,
    build_final_prompts,
    build_initial_prompts,
)


def _decode_payload(user_prompt: str) -> dict[str, str]:
    prefix = "Input payload (JSON):\n"
    assert user_prompt.startswith(prefix)
    return json.loads(user_prompt.removeprefix(prefix))


def test_initial_prompt_keeps_user_judge_contract_opaque() -> None:
    judge_prompt = 'Return JSON with "confidence" and a "yes" or "no" answer.'
    evaluation_data = 'Candidate says: "yes".'

    prompts = build_initial_prompts(
        judge_prompt=judge_prompt,
        evaluation_data=evaluation_data,
    )

    payload = _decode_payload(prompts.user_prompt)
    assert payload["judge_prompt"] == judge_prompt
    assert payload["evaluation_data"] == evaluation_data
    assert "AskJudge treats this value as opaque" in prompts.system_prompt
    assert '"judge_output"' in prompts.system_prompt


def test_initial_prompt_uses_default_assistance_policy() -> None:
    prompts = build_initial_prompts(
        judge_prompt="Judge correctness.",
        evaluation_data="An answer to inspect.",
    )

    assert DEFAULT_ASSISTANCE_POLICY in prompts.system_prompt
    assert "task difficulty" in prompts.system_prompt


def test_initial_prompt_accepts_custom_assistance_policy() -> None:
    custom_policy = "Ask only when a product requirement is missing."

    prompts = build_initial_prompts(
        judge_prompt="Judge correctness.",
        evaluation_data="An answer to inspect.",
        assistance_policy=custom_policy,
    )

    assert custom_policy in prompts.system_prompt
    assert DEFAULT_ASSISTANCE_POLICY not in prompts.system_prompt


def test_final_prompt_contains_all_context_and_disables_more_questions() -> None:
    prompts = build_final_prompts(
        judge_prompt="Return yes or no.",
        evaluation_data="Candidate output",
        initial_output="no",
        question="Is backward compatibility required?",
        human_answer="Unknown",
    )

    payload = _decode_payload(prompts.user_prompt)
    assert payload == {
        "judge_prompt": "Return yes or no.",
        "evaluation_data": "Candidate output",
        "initial_judge_output": "no",
        "question_asked": "Is backward compatibility required?",
        "human_answer": "Unknown",
    }
    assert "must not ask another question" in prompts.system_prompt
    assert "do not treat the missing information as known" in prompts.system_prompt


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (
            "judge_prompt",
            {"judge_prompt": " ", "evaluation_data": "data"},
        ),
        (
            "evaluation_data",
            {"judge_prompt": "judge", "evaluation_data": ""},
        ),
        (
            "assistance_policy",
            {
                "judge_prompt": "judge",
                "evaluation_data": "data",
                "assistance_policy": " ",
            },
        ),
    ],
)
def test_initial_prompt_rejects_empty_required_text(
    field: str,
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        build_initial_prompts(**kwargs)

