import json
from pathlib import Path


EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


def test_twenty_case_fixture_has_expected_routing_split() -> None:
    cases = [
        json.loads(line)
        for line in (EXAMPLE_DIR / "cases_20.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == 20
    assert sum(
        case["data"].get("annotation_status") == "needs_human"
        for case in cases
    ) == 5

    expected_fields = {
        "benchmark",
        "question",
        "reference_answer",
        "response_a",
        "response_b",
        "annotation_status",
        "missing_context",
    }
    assert all(set(case["data"]) == expected_fields for case in cases)
    assert {case["data"]["benchmark"] for case in cases} == {
        "ml-short-answer-pairwise"
    }


def test_example_prompt_defines_one_pairwise_evaluation_contract() -> None:
    prompt = (EXAMPLE_DIR / "judge_prompt.md").read_text(encoding="utf-8")
    normalized = " ".join(prompt.split())

    assert "`confidence`" in prompt
    assert "not an AskJudge routing signal" in normalized
    assert "`winner`: `A`, `B`, or `TIE`" in prompt

