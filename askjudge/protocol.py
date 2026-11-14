"""Parsing for AskJudge's private model-response envelope."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .models import AskJudgeMetadata


class AskJudgeProtocolError(ValueError):
    """Raised when a model response violates the AskJudge protocol."""


@dataclass(frozen=True, slots=True)
class ParsedJudgeResponse:
    """A validated response with opaque judge output and control metadata."""

    output: str
    metadata: AskJudgeMetadata


def parse_internal_response(raw_response: str) -> ParsedJudgeResponse:
    """Parse and validate one private AskJudge response envelope."""

    if not isinstance(raw_response, str):
        raise AskJudgeProtocolError("model response must be a string")

    try:
        payload: Any = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise AskJudgeProtocolError("model response must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise AskJudgeProtocolError("model response must be a JSON object")

    output = payload.get("judge_output")
    if not isinstance(output, str):
        raise AskJudgeProtocolError("judge_output must be a string")

    control = payload.get("askjudge")
    if not isinstance(control, dict):
        raise AskJudgeProtocolError("askjudge must be a JSON object")

    needs_human = control.get("needs_human")
    if type(needs_human) is not bool:
        raise AskJudgeProtocolError("askjudge.needs_human must be a boolean")

    question = control.get("question")
    if question is not None and not isinstance(question, str):
        raise AskJudgeProtocolError("askjudge.question must be a string or null")

    try:
        metadata = AskJudgeMetadata(
            needs_human=needs_human,
            question=question,
        )
    except ValueError as exc:
        raise AskJudgeProtocolError(str(exc)) from exc

    return ParsedJudgeResponse(output=output, metadata=metadata)

