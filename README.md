# AskJudge

AskJudge adds selective human assistance to batch LLM-as-a-Judge workflows.
Every case receives an initial judgment. Clear cases finish immediately; cases
that need human context ask one focused question and receive one final judgment.
For `N` cases of which `K` need help, the normal model-call count is `N + K`.

AskJudge does not impose a score, verdict, confidence, or result schema. The
judge output requested by your prompt is preserved as an opaque string. The
library reads only its separate `needs_human` and `question` control metadata.

> AskJudge is an experimental `0.1.0` project. Validate its judgments and costs
> before using it in consequential evaluation pipelines.

## How it works

```text
N evaluation cases
        |
        v
concurrent initial judgments
        |
        +---- clear cases --------------------> completed
        |
        +---- cases needing context
                    |
                    v
             one human answer
                    |
                    v
              final judgment ----------------> completed
```

The human supplies missing context, criteria, preferences, or expertise. The
LLM still produces the final evaluation result.

## Install from source

Core library only:

```console
python -m pip install -e .
```

With the OpenAI-compatible provider and development tests:

```console
python -m pip install -e ".[openai,dev]"
```

The same provider works with services that expose a compatible Chat Completions
API. The included validation example uses DeepSeek.

## CLI quick start

### 1. Prepare a judge prompt

`judge-prompt.md` defines the evaluation criteria and the user's desired result
format. For example:

```text
Compare response A and response B for correctness and relevance.
Return JSON with winner, confidence, and reason.
```

Those fields belong to the user result; AskJudge does not interpret them.

### 2. Prepare JSONL input

Each line requires a unique string `id` and an arbitrary JSON `data` value:

```json
{"id":"case-001","data":{"question":"...","response_a":"...","response_b":"..."}}
{"id":"case-002","data":{"question":"...","response_a":"...","response_b":"..."}}
```

### 3. Configure a provider

PowerShell example for DeepSeek:

```powershell
$env:ASKJUDGE_API_KEY = Read-Host -MaskInput "API Key"
$env:ASKJUDGE_BASE_URL = "https://api.deepseek.com"
$env:ASKJUDGE_MODEL = "deepseek-flash"
```

If `ASKJUDGE_API_KEY` is absent, the CLI asks for it using hidden input. A
legacy `DEEPSEEK_API_KEY` variable is also accepted. Never put a real key in a
prompt, JSONL file, source file, screenshot, or Git commit.

### 4. Run the batch

```powershell
askjudge run `
  --prompt .\examples\judge_prompt.md `
  --input .\examples\cases_20.jsonl `
  --output .\examples\results_20.jsonl `
  --assistance-policy .\examples\assistance_policy.md `
  --concurrency 3
```

Use `python -m askjudge run ...` if the console script is not on `PATH`.

For each pending case, enter a normal answer, `unknown`, `skip`, or an empty
answer. `unknown` and `skip` still produce a final judgment, but the final prompt
explicitly forbids treating the missing information as known.

Run `askjudge run --help` for endpoint, model, timeout, concurrency, retry, and
JSON-mode options.

## Output

The output is JSONL in the same stable case order:

```json
{
  "case_id": "case-001",
  "output": "{\"winner\":\"A\",\"confidence\":0.94,\"reason\":\"...\"}",
  "askjudge": {"needs_human": false, "question": null},
  "human_answer": null,
  "initial_output": null,
  "error": null
}
```

`output` is the opaque result defined by the user's judge prompt. Human-assisted
records also contain `human_answer` and `initial_output`. Failed records contain
an `error` object instead of pretending that a judgment succeeded.

## Python API

```python
import asyncio

from askjudge import AskJudge, EvaluationItem
from askjudge.providers import OpenAICompatibleProvider


async def main() -> None:
    provider = OpenAICompatibleProvider(
        api_key="...",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )
    judge = AskJudge(provider)
    prompt = "Evaluate correctness and return my requested JSON format."
    items = [EvaluationItem("case-001", "candidate data")]

    initial = await judge.batch_evaluate(
        judge_prompt=prompt,
        items=items,
        concurrency=10,
    )

    answers = {
        record.case_id: input(f"{record.metadata.question}\nAnswer: ")
        for record in initial.pending
    }
    final = await judge.batch_resume(
        initial,
        judge_prompt=prompt,
        items=items,
        answers=answers,
        concurrency=10,
    )
    print(final.to_dict())


asyncio.run(main())
```

To bring another model SDK, implement the minimal asynchronous provider contract:

```python
class MyProvider:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...
```

## Assistance policy

The default policy allows a question when targeted human input could materially
improve reliability, including missing criteria, ambiguous preferences, domain
expertise, or task difficulty. Override it in Python with `assistance_policy=`
or in the CLI with `--assistance-policy policy.md`.

The policy changes when the judge may ask for help. It does not change the fixed
protocol invariants: one provisional result, at most one question, and no third
model round.

## Errors and retries

- Invalid AskJudge protocol output is retried once by default.
- Provider/network failures are not automatically retried to avoid hidden cost.
- A failed batch case is recorded without cancelling unrelated cases.
- The CLI exits with status `2` when the final batch contains failures.
- User-defined fields such as `confidence` never control human routing.

## Current limitations

- One human question and at most two model calls per case.
- No database, web UI, history, multi-model voting, or confidence calibration.
- Interactive CLI answers are collected after the initial batch completes.
- User judge output is returned as a string; AskJudge does not parse its schema.

## Development

```console
python -m pytest
python -m askjudge --help
```

The test suite uses fake providers and makes no paid model calls. See
[examples/VALIDATION.md](examples/VALIDATION.md) for the full 20-case validation
walkthrough.

## License

MIT. See [LICENSE](LICENSE).

