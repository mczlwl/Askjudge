"""Core data models for AskJudge.

The user's judge output is intentionally kept as an opaque string. AskJudge
does not inspect fields such as scores, confidence values, or yes/no verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    """One input case for a batch evaluation."""

    case_id: str
    data: str

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not isinstance(self.data, str):
            raise TypeError("data must be a string")
        if not self.data.strip():
            raise ValueError("data must not be empty")


@dataclass(frozen=True, slots=True)
class AskJudgeMetadata:
    """Control metadata produced alongside a user's judge output."""

    needs_human: bool
    question: str | None = None

    def __post_init__(self) -> None:
        if self.needs_human:
            if self.question is None or not self.question.strip():
                raise ValueError("question is required when needs_human is true")
        elif self.question is not None:
            raise ValueError("question must be None when needs_human is false")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "needs_human": self.needs_human,
            "question": self.question,
        }


@dataclass(frozen=True, slots=True)
class EvaluationError:
    """A workflow failure kept separate from user output and control metadata."""

    kind: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "message": self.message}


@dataclass(slots=True)
class EvaluationRecord:
    """One judge result plus AskJudge's separate control metadata."""

    case_id: str
    output: str | None
    metadata: AskJudgeMetadata | None
    human_answer: str | None = None
    initial_output: str | None = None
    error: EvaluationError | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if (self.output is None) != (self.metadata is None):
            raise ValueError("output and metadata must either both be set or both be None")
        if self.error is None and self.output is None:
            raise ValueError("a successful record requires output and metadata")

    @property
    def pending(self) -> bool:
        """Whether this record is waiting for human assistance."""

        return (
            self.error is None
            and self.metadata is not None
            and self.metadata.needs_human
        )

    @property
    def failed(self) -> bool:
        """Whether the latest operation for this record failed."""

        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "output": self.output,
            "askjudge": self.metadata.to_dict() if self.metadata else None,
            "human_answer": self.human_answer,
            "initial_output": self.initial_output,
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass(slots=True)
class BatchResult:
    """A stable collection of evaluation records."""

    records: list[EvaluationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        case_ids = [record.case_id for record in self.records]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique within a batch")

    @property
    def completed(self) -> list[EvaluationRecord]:
        """Records that do not need human assistance."""

        return [
            record
            for record in self.records
            if not record.pending and not record.failed
        ]

    @property
    def pending(self) -> list[EvaluationRecord]:
        """Records that need human assistance."""

        return [record for record in self.records if record.pending]

    @property
    def failed(self) -> list[EvaluationRecord]:
        """Records whose latest model or workflow operation failed."""

        return [record for record in self.records if record.failed]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {"records": [record.to_dict() for record in self.records]}

