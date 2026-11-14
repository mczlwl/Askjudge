# Changelog

All notable changes to AskJudge will be documented in this file.

## 0.1.0 - Unreleased

### Added

- Single-case initial and human-assisted final evaluation.
- Batch evaluation with bounded concurrency and `N + K` model calls.
- Opaque user-defined judge output separated from AskJudge control metadata.
- Custom assistance policies and one focused human question per case.
- JSONL CLI with `unknown` and `skip` answers.
- OpenAI-compatible provider with DeepSeek configuration examples.
- Protocol validation, one configurable protocol retry, and per-case failures.
- A 20-case pairwise research-evaluation example.

