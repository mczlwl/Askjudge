"""Single-case AskJudge evaluation workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from typing import TypeVar

from .models import BatchResult, EvaluationError, EvaluationItem, EvaluationRecord
from .prompts import PromptBundle, build_final_prompts, build_initial_prompts
from .provider import ModelProvider
from .protocol import (
    AskJudgeProtocolError,
    ParsedJudgeResponse,
    parse_internal_response,
)


T = TypeVar("T")


class AskJudge:
    """Run one provisional judgment and, when needed, one final judgment."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        assistance_policy: str | None = None,
        max_protocol_retries: int = 1,
    ) -> None:
        if type(max_protocol_retries) is not int or max_protocol_retries < 0:
            raise ValueError("max_protocol_retries must be a non-negative integer")
        self._provider = provider
        self._assistance_policy = assistance_policy
        self._max_protocol_retries = max_protocol_retries

    async def evaluate(
        self,
        *,
        case_id: str,
        judge_prompt: str,
        evaluation_data: str,
    ) -> EvaluationRecord:
        """Produce a provisional result and decide whether help is needed."""

        prompts = build_initial_prompts(
            judge_prompt=judge_prompt,
            evaluation_data=evaluation_data,
            assistance_policy=self._assistance_policy,
        )
        parsed = await self._complete_and_parse(prompts)
        return EvaluationRecord(
            case_id=case_id,
            output=parsed.output,
            metadata=parsed.metadata,
        )

    async def resume(
        self,
        record: EvaluationRecord,
        *,
        judge_prompt: str,
        evaluation_data: str,
        human_answer: str,
    ) -> EvaluationRecord:
        """Use one human answer to produce the final judgment."""

        if not record.pending:
            raise ValueError("only a pending evaluation can be resumed")

        question = record.metadata.question
        if question is None:  # Guard for non-standard record implementations.
            raise ValueError("pending evaluation has no question")

        prompts = build_final_prompts(
            judge_prompt=judge_prompt,
            evaluation_data=evaluation_data,
            initial_output=record.output,
            question=question,
            human_answer=human_answer,
        )
        parsed = await self._complete_and_parse(prompts, final=True)

        return EvaluationRecord(
            case_id=record.case_id,
            output=parsed.output,
            metadata=parsed.metadata,
            human_answer=human_answer,
            initial_output=record.output,
        )

    async def batch_evaluate(
        self,
        *,
        judge_prompt: str,
        items: Sequence[EvaluationItem],
        concurrency: int = 10,
    ) -> BatchResult:
        """Evaluate multiple independent cases with bounded concurrency."""

        _validate_concurrency(concurrency)
        _index_items(items)

        operations = [
            partial(
                self._evaluate_safely,
                item,
                judge_prompt=judge_prompt,
            )
            for item in items
        ]
        records = await _run_limited(operations, concurrency)
        return BatchResult(records=records)

    async def batch_resume(
        self,
        batch: BatchResult,
        *,
        judge_prompt: str,
        items: Sequence[EvaluationItem],
        answers: Mapping[str, str],
        concurrency: int = 10,
    ) -> BatchResult:
        """Resume only pending cases for which a human answer was supplied."""

        _validate_concurrency(concurrency)
        items_by_id = _index_items(items)
        records_by_id = {record.case_id: record for record in batch.records}

        if set(items_by_id) != set(records_by_id):
            raise ValueError("items must contain exactly the cases in the batch")

        unknown_answer_ids = set(answers) - set(records_by_id)
        if unknown_answer_ids:
            unknown = ", ".join(sorted(unknown_answer_ids))
            raise ValueError(f"answers contain unknown case_id values: {unknown}")

        failed_answer_ids = {
            case_id
            for case_id in answers
            if records_by_id[case_id].failed
        }
        if failed_answer_ids:
            failed = ", ".join(sorted(failed_answer_ids))
            raise ValueError(f"answers were supplied for failed cases: {failed}")

        completed_answer_ids = {
            case_id
            for case_id in answers
            if not records_by_id[case_id].pending
            and not records_by_id[case_id].failed
        }
        if completed_answer_ids:
            completed = ", ".join(sorted(completed_answer_ids))
            raise ValueError(f"answers were supplied for completed cases: {completed}")

        pending_with_answers = [
            record
            for record in batch.records
            if record.pending and record.case_id in answers
        ]
        operations = [
            partial(
                self._resume_safely,
                record,
                judge_prompt=judge_prompt,
                evaluation_data=items_by_id[record.case_id].data,
                human_answer=answers[record.case_id],
            )
            for record in pending_with_answers
        ]
        resumed = await _run_limited(operations, concurrency)
        replacements = {record.case_id: record for record in resumed}

        return BatchResult(
            records=[
                replacements.get(record.case_id, record)
                for record in batch.records
            ]
        )

    async def _complete_and_parse(
        self,
        prompts: PromptBundle,
        *,
        final: bool = False,
    ) -> ParsedJudgeResponse:
        user_prompt = prompts.user_prompt
        last_error: AskJudgeProtocolError | None = None
        total_attempts = self._max_protocol_retries + 1

        for attempt in range(1, total_attempts + 1):
            raw_response = await self._provider.complete(
                system_prompt=prompts.system_prompt,
                user_prompt=user_prompt,
            )
            try:
                parsed = parse_internal_response(raw_response)
                if final and parsed.metadata.needs_human:
                    raise AskJudgeProtocolError(
                        "final response must not request additional human assistance"
                    )
                return parsed
            except AskJudgeProtocolError as exc:
                last_error = exc
                if attempt < total_attempts:
                    user_prompt = _retry_user_prompt(prompts.user_prompt, exc)

        assert last_error is not None
        raise AskJudgeProtocolError(
            f"model violated the AskJudge protocol after {total_attempts} "
            f"attempt(s): {last_error}"
        ) from last_error

    async def _evaluate_safely(
        self,
        item: EvaluationItem,
        *,
        judge_prompt: str,
    ) -> EvaluationRecord:
        try:
            return await self.evaluate(
                case_id=item.case_id,
                judge_prompt=judge_prompt,
                evaluation_data=item.data,
            )
        except Exception as exc:
            return EvaluationRecord(
                case_id=item.case_id,
                output=None,
                metadata=None,
                error=_evaluation_error(exc),
            )

    async def _resume_safely(
        self,
        record: EvaluationRecord,
        *,
        judge_prompt: str,
        evaluation_data: str,
        human_answer: str,
    ) -> EvaluationRecord:
        try:
            return await self.resume(
                record,
                judge_prompt=judge_prompt,
                evaluation_data=evaluation_data,
                human_answer=human_answer,
            )
        except Exception as exc:
            return EvaluationRecord(
                case_id=record.case_id,
                output=record.output,
                metadata=record.metadata,
                human_answer=human_answer,
                initial_output=record.output,
                error=_evaluation_error(exc),
            )


def _validate_concurrency(concurrency: int) -> None:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")


def _index_items(items: Sequence[EvaluationItem]) -> dict[str, EvaluationItem]:
    indexed = {item.case_id: item for item in items}
    if len(indexed) != len(items):
        raise ValueError("case_id values must be unique within batch items")
    return indexed


async def _run_limited(
    operations: Sequence[Callable[[], Awaitable[T]]],
    concurrency: int,
) -> list[T]:
    if not operations:
        return []

    semaphore = asyncio.Semaphore(concurrency)

    async def run(operation: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await operation()

    return list(await asyncio.gather(*(run(operation) for operation in operations)))


def _retry_user_prompt(original_prompt: str, error: Exception) -> str:
    return (
        original_prompt
        + "\n\nRetry instruction:\n"
        + "Your previous response violated the private AskJudge response "
        + f"protocol: {error}. Return only a corrected envelope."
    )


def _evaluation_error(exc: Exception) -> EvaluationError:
    return EvaluationError(kind=type(exc).__name__, message=str(exc))


