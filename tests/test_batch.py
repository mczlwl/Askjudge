import json

import pytest

from askjudge import AskJudge, EvaluationItem
from tests.async_helpers import run_immediate, sequential_gather
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


@pytest.fixture(autouse=True)
def use_event_loop_free_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Windows sandbox cannot close an asyncio event loop. Production keeps
    # asyncio.gather; tests use deterministic immediate fake model responses.
    monkeypatch.setattr("askjudge.judge.asyncio.gather", sequential_gather)


def test_batch_evaluate_partitions_cases_and_preserves_order() -> None:
    provider = FakeProvider(
        responses=[
            _response("result-1", needs_human=False),
            _response(
                "result-2",
                needs_human=True,
                question="Which rubric applies?",
            ),
            _response("result-3", needs_human=False),
        ]
    )
    judge = AskJudge(provider)
    items = [
        EvaluationItem("case-001", "data-1"),
        EvaluationItem("case-002", "data-2"),
        EvaluationItem("case-003", "data-3"),
    ]

    batch = run_immediate(
        judge.batch_evaluate(
            judge_prompt="Judge each case.",
            items=items,
            concurrency=2,
        )
    )

    assert [record.case_id for record in batch.records] == [
        "case-001",
        "case-002",
        "case-003",
    ]
    assert [record.case_id for record in batch.completed] == [
        "case-001",
        "case-003",
    ]
    assert [record.case_id for record in batch.pending] == ["case-002"]
    assert len(provider.calls) == 3


def test_batch_resume_calls_model_only_for_answered_pending_cases() -> None:
    provider = FakeProvider(
        responses=[
            _response("initial-1", needs_human=False),
            _response("initial-2", needs_human=True, question="Question 2?"),
            _response("initial-3", needs_human=True, question="Question 3?"),
            _response("final-2", needs_human=False),
        ]
    )
    judge = AskJudge(provider)
    items = [
        EvaluationItem("case-001", "data-1"),
        EvaluationItem("case-002", "data-2"),
        EvaluationItem("case-003", "data-3"),
    ]
    initial = run_immediate(
        judge.batch_evaluate(
            judge_prompt="Judge each case.",
            items=items,
            concurrency=3,
        )
    )

    final = run_immediate(
        judge.batch_resume(
            initial,
            judge_prompt="Judge each case.",
            items=items,
            answers={"case-002": "Human answer 2"},
            concurrency=2,
        )
    )

    assert len(provider.calls) == 4  # N + K: 3 initial + 1 resumed.
    assert [record.output for record in final.records] == [
        "initial-1",
        "final-2",
        "initial-3",
    ]
    assert [record.case_id for record in final.pending] == ["case-003"]
    assert final.records[1].human_answer == "Human answer 2"
    assert final.records[1].initial_output == "initial-2"


def test_batch_resume_with_no_answers_makes_no_model_calls() -> None:
    provider = FakeProvider(
        responses=[
            _response("initial", needs_human=True, question="Need context?"),
        ]
    )
    judge = AskJudge(provider)
    items = [EvaluationItem("case-001", "data")]
    initial = run_immediate(
        judge.batch_evaluate(judge_prompt="Judge.", items=items)
    )

    unchanged = run_immediate(
        judge.batch_resume(
            initial,
            judge_prompt="Judge.",
            items=items,
            answers={},
        )
    )

    assert unchanged.records == initial.records
    assert len(provider.calls) == 1


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_batch_rejects_invalid_concurrency(concurrency: object) -> None:
    judge = AskJudge(FakeProvider(responses=[]))

    with pytest.raises(ValueError, match="positive integer"):
        run_immediate(
            judge.batch_evaluate(
                judge_prompt="Judge.",
                items=[],
                concurrency=concurrency,  # type: ignore[arg-type]
            )
        )


def test_batch_rejects_duplicate_case_ids_before_model_calls() -> None:
    provider = FakeProvider(responses=[])
    judge = AskJudge(provider)
    items = [
        EvaluationItem("duplicate", "first"),
        EvaluationItem("duplicate", "second"),
    ]

    with pytest.raises(ValueError, match="case_id values must be unique"):
        run_immediate(
            judge.batch_evaluate(judge_prompt="Judge.", items=items)
        )

    assert provider.calls == []


def test_batch_resume_rejects_answers_for_completed_cases() -> None:
    provider = FakeProvider(
        responses=[_response("complete", needs_human=False)]
    )
    judge = AskJudge(provider)
    items = [EvaluationItem("case-001", "data")]
    batch = run_immediate(
        judge.batch_evaluate(judge_prompt="Judge.", items=items)
    )

    with pytest.raises(ValueError, match="answers were supplied for completed"):
        run_immediate(
            judge.batch_resume(
                batch,
                judge_prompt="Judge.",
                items=items,
                answers={"case-001": "Unused answer"},
            )
        )

    assert len(provider.calls) == 1


def test_batch_records_one_failure_and_continues_other_cases() -> None:
    provider = FakeProvider(
        responses=[
            RuntimeError("provider unavailable for one case"),
            _response("second succeeded", needs_human=False),
            _response("third succeeded", needs_human=False),
        ]
    )
    judge = AskJudge(provider)
    items = [
        EvaluationItem("case-001", "data-1"),
        EvaluationItem("case-002", "data-2"),
        EvaluationItem("case-003", "data-3"),
    ]

    batch = run_immediate(
        judge.batch_evaluate(
            judge_prompt="Judge.",
            items=items,
            concurrency=2,
        )
    )

    assert len(provider.calls) == 3
    assert [record.case_id for record in batch.failed] == ["case-001"]
    assert batch.failed[0].output is None
    assert batch.failed[0].metadata is None
    assert batch.failed[0].error is not None
    assert batch.failed[0].error.kind == "RuntimeError"
    assert [record.case_id for record in batch.completed] == [
        "case-002",
        "case-003",
    ]


def test_batch_resume_failure_preserves_provisional_output() -> None:
    provider = FakeProvider(
        responses=[
            _response("provisional", needs_human=True, question="Need context?"),
            RuntimeError("final call failed"),
        ]
    )
    judge = AskJudge(provider)
    items = [EvaluationItem("case-001", "data")]
    initial = run_immediate(
        judge.batch_evaluate(judge_prompt="Judge.", items=items)
    )

    final = run_immediate(
        judge.batch_resume(
            initial,
            judge_prompt="Judge.",
            items=items,
            answers={"case-001": "Human context"},
        )
    )

    assert final.pending == []
    assert final.completed == []
    assert len(final.failed) == 1
    assert final.failed[0].output == "provisional"
    assert final.failed[0].human_answer == "Human context"
    assert final.failed[0].error is not None
    assert final.failed[0].error.kind == "RuntimeError"

