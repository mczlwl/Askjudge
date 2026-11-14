import json

import pytest

from askjudge.protocol import AskJudgeProtocolError, parse_internal_response


def test_protocol_preserves_user_output_exactly() -> None:
    output = '{"yes":false,"confidence":0.95}\nExplanation follows.'
    raw = json.dumps(
        {
            "judge_output": output,
            "askjudge": {
                "needs_human": True,
                "question": "Which rubric version applies?",
            },
        }
    )

    parsed = parse_internal_response(raw)

    assert parsed.output == output
    assert parsed.metadata.needs_human is True
    assert parsed.metadata.question == "Which rubric version applies?"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "valid JSON"),
        ("[]", "JSON object"),
        (
            json.dumps(
                {
                    "judge_output": {"answer": "yes"},
                    "askjudge": {"needs_human": False, "question": None},
                }
            ),
            "judge_output must be a string",
        ),
        (
            json.dumps({"judge_output": "yes", "askjudge": "invalid"}),
            "askjudge must be a JSON object",
        ),
        (
            json.dumps(
                {
                    "judge_output": "yes",
                    "askjudge": {"needs_human": "false", "question": None},
                }
            ),
            "needs_human must be a boolean",
        ),
        (
            json.dumps(
                {
                    "judge_output": "yes",
                    "askjudge": {"needs_human": True, "question": None},
                }
            ),
            "question is required",
        ),
        (
            json.dumps(
                {
                    "judge_output": "yes",
                    "askjudge": {
                        "needs_human": False,
                        "question": "Unexpected question",
                    },
                }
            ),
            "question must be None",
        ),
    ],
)
def test_protocol_rejects_invalid_envelopes(raw: str, message: str) -> None:
    with pytest.raises(AskJudgeProtocolError, match=message):
        parse_internal_response(raw)

