"""Provider-independent components for an interactive AskJudge CLI."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .judge import AskJudge
from .models import BatchResult, EvaluationItem


UNKNOWN_ANSWER = "[ASKJUDGE_UNKNOWN]"
SKIPPED_ANSWER = "[ASKJUDGE_SKIPPED]"

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class InputFormatError(ValueError):
    """Raised when a CLI input file does not follow the JSONL contract."""


def load_prompt(path: str | Path) -> str:
    """Load a non-empty UTF-8 judge prompt from disk."""

    prompt_path = Path(path)
    prompt = prompt_path.read_text(encoding="utf-8-sig")
    if not prompt.strip():
        raise InputFormatError(f"judge prompt is empty: {prompt_path}")
    return prompt


def load_items(path: str | Path) -> list[EvaluationItem]:
    """Load evaluation cases from JSONL records containing ``id`` and ``data``."""

    input_path = Path(path)
    items: list[EvaluationItem] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            payload: Any = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise InputFormatError(
                f"invalid JSON on line {line_number} of {input_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise InputFormatError(
                f"line {line_number} of {input_path} must be a JSON object"
            )

        case_id = payload.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise InputFormatError(
                f"line {line_number} of {input_path} requires a non-empty string id"
            )
        if case_id in seen_ids:
            raise InputFormatError(
                f"duplicate id {case_id!r} on line {line_number} of {input_path}"
            )
        if "data" not in payload:
            raise InputFormatError(
                f"line {line_number} of {input_path} is missing data"
            )

        data = _data_to_text(payload["data"])
        try:
            item = EvaluationItem(case_id=case_id, data=data)
        except (TypeError, ValueError) as exc:
            raise InputFormatError(
                f"invalid data on line {line_number} of {input_path}: {exc}"
            ) from exc

        items.append(item)
        seen_ids.add(case_id)

    if not items:
        raise InputFormatError(f"no evaluation cases found in {input_path}")
    return items


def write_results(path: str | Path, batch: BatchResult) -> None:
    """Write one JSON object per evaluation record without parsing its output."""

    output_path = Path(path)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False)
        for record in batch.records
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_answers(
    batch: BatchResult,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> dict[str, str]:
    """Collect one answer for every pending case in a batch."""

    answers: dict[str, str] = {}
    pending = batch.pending

    output_fn(f"AskJudge found {len(pending)} case(s) requiring assistance.")
    for index, record in enumerate(pending, start=1):
        output_fn("")
        output_fn(f"[{index}/{len(pending)}] {record.case_id}")
        output_fn(f"Question: {record.metadata.question}")
        raw_answer = input_fn("Your answer (or 'unknown'/'skip'): ")
        answers[record.case_id] = normalize_answer(raw_answer)

    return answers


def normalize_answer(answer: str) -> str:
    """Normalize CLI control answers while preserving ordinary human text."""

    stripped = answer.strip()
    lowered = stripped.casefold()
    if lowered == "unknown":
        return UNKNOWN_ANSWER
    if lowered == "skip" or not stripped:
        return SKIPPED_ANSWER
    return stripped


async def run_interactive(
    judge: AskJudge,
    *,
    judge_prompt: str,
    items: Sequence[EvaluationItem],
    concurrency: int = 10,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> BatchResult:
    """Run batch evaluation, collect needed answers, and finish the batch."""

    initial = await judge.batch_evaluate(
        judge_prompt=judge_prompt,
        items=items,
        concurrency=concurrency,
    )
    output_fn(
        f"Initial evaluation complete: {len(initial.completed)} completed, "
        f"{len(initial.pending)} requiring assistance, "
        f"{len(initial.failed)} failed."
    )

    if not initial.pending:
        return initial

    answers = collect_answers(
        initial,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    final = await judge.batch_resume(
        initial,
        judge_prompt=judge_prompt,
        items=items,
        answers=answers,
        concurrency=concurrency,
    )
    output_fn(
        f"Final evaluation complete: {len(final.completed)} completed, "
        f"{len(final.pending)} still pending, {len(final.failed)} failed."
    )
    return final


async def run_files(
    judge: AskJudge,
    *,
    prompt_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    concurrency: int = 10,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> BatchResult:
    """Run an interactive evaluation using prompt, input, and output files."""

    prompt = load_prompt(prompt_path)
    items = load_items(input_path)
    result = await run_interactive(
        judge,
        judge_prompt=prompt,
        items=items,
        concurrency=concurrency,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    write_results(output_path, result)
    return result


def _data_to_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the public ``askjudge`` command-line parser."""

    parser = argparse.ArgumentParser(
        prog="askjudge",
        description="Selective human assistance for LLM-as-a-Judge batches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="evaluate a JSONL batch and interactively answer pending questions",
    )
    run_parser.add_argument("--prompt", type=Path, required=True)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument(
        "--assistance-policy",
        type=Path,
        help="optional text file overriding the default assistance policy",
    )
    run_parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "ASKJUDGE_BASE_URL",
            "https://api.deepseek.com",
        ),
    )
    run_parser.add_argument(
        "--model",
        default=os.environ.get("ASKJUDGE_MODEL", "deepseek-flash"),
    )
    run_parser.add_argument("--concurrency", type=int, default=10)
    run_parser.add_argument("--timeout", type=float, default=90.0)
    run_parser.add_argument("--max-protocol-retries", type=int, default=1)
    run_parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="do not request provider-level JSON response mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the installed ``askjudge`` command."""

    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        return asyncio.run(_run_command(args))
    except KeyboardInterrupt:
        print("AskJudge interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"AskJudge error: {exc}", file=sys.stderr)
        return 1


async def _run_command(
    args: argparse.Namespace,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    if args.command != "run":
        raise ValueError(f"unsupported command: {args.command}")

    from .providers import OpenAICompatibleProvider

    assistance_policy = (
        load_prompt(args.assistance_policy)
        if args.assistance_policy is not None
        else None
    )
    provider = OpenAICompatibleProvider(
        api_key=_read_api_key(),
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        json_mode=not args.no_json_mode,
    )
    judge = AskJudge(
        provider,
        assistance_policy=assistance_policy,
        max_protocol_retries=args.max_protocol_retries,
    )
    result = await run_files(
        judge,
        prompt_path=args.prompt,
        input_path=args.input,
        output_path=args.output,
        concurrency=args.concurrency,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    assisted = sum(
        record.human_answer is not None
        for record in result.records
    )
    expected_calls = len(result.records) + assisted
    output_fn("")
    output_fn("AskJudge summary")
    output_fn(f"Cases: {len(result.records)}")
    output_fn(f"Human-assisted: {assisted}")
    output_fn(f"Failed: {len(result.failed)}")
    output_fn(f"Model API calls: {provider.call_count}")
    output_fn(f"Expected N + K without retries: {expected_calls}")
    output_fn(f"Results: {args.output.resolve()}")
    return 0 if not result.failed else 2


def _read_api_key() -> str:
    key = os.environ.get("ASKJUDGE_API_KEY") or os.environ.get(
        "DEEPSEEK_API_KEY"
    )
    if key:
        return key
    return getpass.getpass("Paste API key (input hidden): ")

