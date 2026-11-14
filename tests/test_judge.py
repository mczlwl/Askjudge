import json

import pytest

from askjudge import AskJudge, AskJudgeMetadata, EvaluationRecord
from askjudge.protocol import AskJudgeProtocolError
from tests.async_helpers import run_immediate
from tests.fakes import FakeProvider


def _response(
    output: str,
    *,
    needs_human: bool,
    question: str | None = None,
) -> str:
    return json.dumps(
        {
            "judge_output": output,
            "askjudge": {
                "needs_human": needs_human,
                "question": question,
            },
        }
    )


def test_evaluate_returns_opaque_output_without_requesting_help() -> None:
    user_output = '{"answer":"yes","confidence":0.42}'
    provider = FakeProvider(
        responses=[_response(user_output, needs_human=False)]
    )
    judge = AskJudge(provider)

    record = run_immediate(
        judge.evaluate(
            case_id="case-001",
            judge_prompt="Return answer and confidence as JSON.",
            evaluation_data="Candidate answer",
        )
    )

    assert record.output == user_output
    assert record.pending is False
    assert len(provider.calls) == 1


def test_pending_evaluation_can_be_resumed_once() -> None:
    provider = FakeProvider(
        responses=[
            _response(
                "provisional: no",
                needs_human=True,
                question="Is backward compatibility required?",
            ),
            _response("final: yes", needs_human=False),
        ]
    )
    judge = AskJudge(provider)

    initial = run_immediate(
        judge.evaluate(
            case_id="case-002",
            judge_prompt="Return yes or no.",
            evaluation_data="Candidate implementation",
        )
    )
    final = run_immediate(
        judge.resume(
            initial,
            judge_prompt="Return yes or no.",
            evaluation_data="Candidate implementation",
            human_answer="Yes, compatibility is required.",
        )
    )

    assert initial.pending is True
    assert final.pending is False
    assert final.output == "final: yes"
    assert final.initial_output == "provisional: no"
    assert final.human_answer == "Yes, compatibility is required."
    assert len(provider.calls) == 2
    assert "Yes, compatibility is required." in provider.calls[1]["user_prompt"]


def test_completed_evaluation_cannot_be_resumed() -> None:
    provider = FakeProvider(responses=[])
    judge = AskJudge(provider)
    completed = EvaluationRecord(
        case_id="case-003",
        output="yes",
        metadata=AskJudgeMetadata(needs_human=False),
    )

    with pytest.raises(ValueError, match="only a pending evaluation"):
        run_immediate(
            judge.resume(
                completed,
                judge_prompt="Return yes or no.",
                evaluation_data="Candidate implementation",
                human_answer="Extra information",
            )
        )

    assert provider.calls == []


def test_final_response_cannot_request_more_help() -> None:
    provider = FakeProvider(
        responses=[
            _response(
                "still unsure",
                needs_human=True,
                question="Can I ask another question?",
            ),
            _response(
                "still unsure",
                needs_human=True,
                question="Can I ask another question?",
            ),
        ]
    )
    judge = AskJudge(provider)
    pending = EvaluationRecord(
        case_id="case-004",
        output="provisional",
        metadata=AskJudgeMetadata(
            needs_human=True,
            question="What constraint applies?",
        ),
    )

    with pytest.raises(AskJudgeProtocolError, match="must not request"):
        run_immediate(
            judge.resume(
                pending,
                judge_prompt="Judge the candidate.",
                evaluation_data="Candidate implementation",
                human_answer="The strict constraint applies.",
            )
        )

    assert len(provider.calls) == 2


def test_protocol_error_is_retried_once_and_can_recover() -> None:
    provider = FakeProvider(
        responses=[
            "not valid JSON",
            _response("recovered", needs_human=False),
        ]
    )
    judge = AskJudge(provider)

    record = run_immediate(
        judge.evaluate(
            case_id="case-retry",
            judge_prompt="Judge.",
            evaluation_data="Candidate",
        )
    )

    assert record.output == "recovered"
    assert len(provider.calls) == 2
    assert "previous response violated" in provider.calls[1]["user_prompt"]


def test_protocol_error_is_raised_after_retry_is_exhausted() -> None:
    provider = FakeProvider(responses=["bad response", "still bad"])
    judge = AskJudge(provider)

    with pytest.raises(AskJudgeProtocolError, match="after 2 attempt"):
        run_immediate(
            judge.evaluate(
                case_id="case-invalid",
                judge_prompt="Judge.",
                evaluation_data="Candidate",
            )
        )

    assert len(provider.calls) == 2


@pytest.mark.parametrize("retries", [-1, True, 1.5])
def test_invalid_protocol_retry_configuration_is_rejected(retries: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        AskJudge(
            FakeProvider(responses=[]),
            max_protocol_retries=retries,  # type: ignore[arg-type]
        )


def test_configured_assistance_policy_reaches_initial_call() -> None:
    provider = FakeProvider(responses=[_response("yes", needs_human=False)])
    judge = AskJudge(
        provider,
        assistance_policy="Ask only when security requirements are missing.",
    )

    run_immediate(
        judge.evaluate(
            case_id="case-005",
            judge_prompt="Judge the candidate.",
            evaluation_data="Candidate implementation",
        )
    )

    assert "Ask only when security requirements are missing." in provider.calls[0][
        "system_prompt"
    ]

