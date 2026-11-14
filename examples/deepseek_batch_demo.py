"""Run the 20-case AskJudge validation against DeepSeek."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
from pathlib import Path

from askjudge import AskJudge, run_files
from askjudge.providers import OpenAICompatibleProvider


EXAMPLE_DIR = Path(__file__).resolve().parent

DEMO_ASSISTANCE_POLICY = """\
Use the evaluation data field "annotation_status" as the routing rule. When it
is "needs_human", request human assistance and ask exactly one focused question
about the benchmark rule described by "missing_context". When it is "complete",
do not request human assistance. Always produce the best provisional pairwise
judgment before asking."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AskJudge with 20 cases using DeepSeek.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=EXAMPLE_DIR / "judge_prompt.md",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=EXAMPLE_DIR / "cases_20.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "results_20.jsonl",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    return getpass.getpass("Paste DEEPSEEK_API_KEY (input hidden): ")


async def _main() -> None:
    args = _arguments()
    provider = OpenAICompatibleProvider(
        api_key=_api_key(),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        base_url=os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
    )
    judge = AskJudge(
        provider,
        assistance_policy=DEMO_ASSISTANCE_POLICY,
        max_protocol_retries=1,
    )

    result = await run_files(
        judge,
        prompt_path=args.prompt,
        input_path=args.input,
        output_path=args.output,
        concurrency=args.concurrency,
    )

    assisted = sum(
        record.human_answer is not None
        for record in result.records
    )
    expected_without_retries = len(result.records) + assisted
    print("")
    print("Validation summary")
    print(f"- Cases: {len(result.records)}")
    print(f"- Human-assisted: {assisted}")
    print(f"- Failed: {len(result.failed)}")
    print(f"- Model API calls: {provider.call_count}")
    print(f"- Expected N + K without retries: {expected_without_retries}")
    print(f"- Results: {args.output.resolve()}")
    if provider.call_count > expected_without_retries:
        print(
            "- Note: extra calls indicate that at least one private response "
            "needed the configured protocol retry."
        )


if __name__ == "__main__":
    asyncio.run(_main())

