from pathlib import Path

from askjudge.cli import build_argument_parser


def test_run_command_parses_required_files_and_defaults(tmp_path: Path) -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "run",
            "--prompt",
            str(tmp_path / "prompt.md"),
            "--input",
            str(tmp_path / "input.jsonl"),
            "--output",
            str(tmp_path / "output.jsonl"),
        ]
    )

    assert args.command == "run"
    assert args.model == "deepseek-flash"
    assert args.base_url == "https://api.deepseek.com"
    assert args.concurrency == 10
    assert args.max_protocol_retries == 1
    assert args.no_json_mode is False


def test_run_command_uses_askjudge_environment_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ASKJUDGE_MODEL", "custom-model")
    monkeypatch.setenv("ASKJUDGE_BASE_URL", "https://provider.example/v1")
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "run",
            "--prompt",
            str(tmp_path / "prompt.md"),
            "--input",
            str(tmp_path / "input.jsonl"),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--no-json-mode",
        ]
    )

    assert args.model == "custom-model"
    assert args.base_url == "https://provider.example/v1"
    assert args.no_json_mode is True

