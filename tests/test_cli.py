import json
from pathlib import Path

import pytest

from askjudge import (
    AskJudge,
    AskJudgeMetadata,
    BatchResult,
    EvaluationItem,
    EvaluationRecord,
    InputFormatError,
    SKIPPED_ANSWER,
    UNKNOWN_ANSWER,
    collect_answers,
    load_items,
    load_prompt,
    normalize_answer,
    run_files,
    run_interactive,
    write_results,
)
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
    monkeypatch.setattr("askjudge.judge.asyncio.gather", sequential_gather)


def test_load_prompt_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "prompt.md"
    path.write_text("\ufeff请判断答案是否正确。", encoding="utf-8")

    assert load_prompt(path) == "请判断答案是否正确。"


def test_load_items_preserves_text_and_serializes_structured_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "text", "data": "plain text"}),
                json.dumps(
                    {"id": "object", "data": {"answer": "是", "score": 4}},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    items = load_items(path)

    assert items == [
        EvaluationItem("text", "plain text"),
        EvaluationItem("object", '{"answer":"是","score":4}'),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "invalid JSON on line 1"),
        (json.dumps([1, 2]), "must be a JSON object"),
        (json.dumps({"id": "x"}), "is missing data"),
        (
            "\n".join(
                [
                    json.dumps({"id": "same", "data": "one"}),
                    json.dumps({"id": "same", "data": "two"}),
                ]
            ),
            "duplicate id 'same' on line 2",
        ),
    ],
)
def test_load_items_reports_input_errors(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(InputFormatError, match=message):
        load_items(path)


def test_normalize_answer_handles_unknown_skip_and_ordinary_text() -> None:
    assert normalize_answer(" UNKNOWN ") == UNKNOWN_ANSWER
    assert normalize_answer("skip") == SKIPPED_ANSWER
    assert normalize_answer("   ") == SKIPPED_ANSWER
    assert normalize_answer("  Use version 2.  ") == "Use version 2."


def test_collect_answers_visits_only_pending_cases() -> None:
    batch = BatchResult(
        records=[
            EvaluationRecord(
                "complete",
                "yes",
                AskJudgeMetadata(needs_human=False),
            ),
            EvaluationRecord(
                "pending-1",
                "maybe",
                AskJudgeMetadata(needs_human=True, question="Question one?"),
            ),
            EvaluationRecord(
                "pending-2",
                "maybe",
                AskJudgeMetadata(needs_human=True, question="Question two?"),
            ),
        ]
    )
    supplied = iter(["unknown", "skip"])
    output: list[str] = []

    answers = collect_answers(
        batch,
        input_fn=lambda _: next(supplied),
        output_fn=output.append,
    )

    assert answers == {
        "pending-1": UNKNOWN_ANSWER,
        "pending-2": SKIPPED_ANSWER,
    }
    assert any("Question one?" in line for line in output)
    assert all("complete" not in line for line in output)


def test_write_results_keeps_user_output_as_opaque_string(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    user_output = '{"confidence":0.8,"answer":"yes"}'
    batch = BatchResult(
        records=[
            EvaluationRecord(
                "case-001",
                user_output,
                AskJudgeMetadata(needs_human=False),
            )
        ]
    )

    write_results(output_path, batch)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["output"] == user_output
    assert payload["askjudge"]["needs_human"] is False


def test_run_interactive_completes_only_requested_second_round() -> None:
    provider = FakeProvider(
        responses=[
            _response("direct", needs_human=False),
            _response("initial", needs_human=True, question="Need context?"),
            _response("final", needs_human=False),
        ]
    )
    judge = AskJudge(provider)
    output: list[str] = []

    result = run_immediate(
        run_interactive(
            judge,
            judge_prompt="Judge.",
            items=[
                EvaluationItem("case-001", "data-1"),
                EvaluationItem("case-002", "data-2"),
            ],
            input_fn=lambda _: "Human context",
            output_fn=output.append,
        )
    )

    assert len(provider.calls) == 3  # N + K: 2 initial + 1 resumed.
    assert [record.output for record in result.records] == ["direct", "final"]
    assert result.pending == []
    assert any("1 requiring assistance" in line for line in output)


def test_run_files_reads_evaluates_and_writes(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "results.jsonl"
    prompt_path.write_text("Judge correctness.", encoding="utf-8")
    input_path.write_text(
        json.dumps({"id": "case-001", "data": "candidate"}),
        encoding="utf-8",
    )
    provider = FakeProvider(
        responses=[_response("yes", needs_human=False)]
    )

    result = run_immediate(
        run_files(
            AskJudge(provider),
            prompt_path=prompt_path,
            input_path=input_path,
            output_path=output_path,
            output_fn=lambda _: None,
        )
    )

    assert result.records[0].output == "yes"
    assert json.loads(output_path.read_text(encoding="utf-8"))["output"] == "yes"

