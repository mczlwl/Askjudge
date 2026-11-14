import json

import pytest

from askjudge import (
    AskJudgeMetadata,
    BatchResult,
    EvaluationError,
    EvaluationRecord,
)


def test_metadata_requires_question_when_human_help_is_needed() -> None:
    with pytest.raises(ValueError, match="question is required"):
        AskJudgeMetadata(needs_human=True)


def test_metadata_rejects_question_when_human_help_is_not_needed() -> None:
    with pytest.raises(ValueError, match="question must be None"):
        AskJudgeMetadata(needs_human=False, question="Unused question")


def test_batch_partitions_records_without_reading_user_output() -> None:
    completed = EvaluationRecord(
        case_id="case-001",
        output='{"confidence": 0.1, "answer": "no"}',
        metadata=AskJudgeMetadata(needs_human=False),
    )
    pending = EvaluationRecord(
        case_id="case-002",
        output='{"confidence": 0.99, "answer": "yes"}',
        metadata=AskJudgeMetadata(
            needs_human=True,
            question="Which product specification applies?",
        ),
    )

    batch = BatchResult(records=[completed, pending])

    assert batch.completed == [completed]
    assert batch.pending == [pending]


def test_batch_rejects_duplicate_case_ids() -> None:
    metadata = AskJudgeMetadata(needs_human=False)

    with pytest.raises(ValueError, match="case_id values must be unique"):
        BatchResult(
            records=[
                EvaluationRecord("same-id", "first", metadata),
                EvaluationRecord("same-id", "second", metadata),
            ]
        )


def test_batch_serializes_to_json() -> None:
    batch = BatchResult(
        records=[
            EvaluationRecord(
                case_id="case-001",
                output="yes",
                metadata=AskJudgeMetadata(needs_human=False),
            )
        ]
    )

    encoded = json.dumps(batch.to_dict())

    assert '"output": "yes"' in encoded
    assert '"needs_human": false' in encoded


def test_failed_record_is_separate_from_output_and_control_metadata() -> None:
    failed = EvaluationRecord(
        case_id="case-failed",
        output=None,
        metadata=None,
        error=EvaluationError(kind="RuntimeError", message="model unavailable"),
    )
    batch = BatchResult(records=[failed])

    assert batch.completed == []
    assert batch.pending == []
    assert batch.failed == [failed]
    assert failed.to_dict() == {
        "case_id": "case-failed",
        "output": None,
        "askjudge": None,
        "human_answer": None,
        "initial_output": None,
        "error": {
            "kind": "RuntimeError",
            "message": "model unavailable",
        },
    }

