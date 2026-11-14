# AskJudge + DeepSeek validation

Run all commands from the repository root.

## 1. Create an isolated environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[deepseek,dev]"
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deepseek,dev]"
```

## 2. Supply the API key

The safest simple option is to omit the environment variable. Both example
commands will prompt for the key using hidden input.

Alternatively, set it only for the current PowerShell session:

```powershell
$env:ASKJUDGE_API_KEY = Read-Host -MaskInput "DeepSeek API Key"
$env:ASKJUDGE_MODEL = "deepseek-flash"
$env:ASKJUDGE_BASE_URL = "https://api.deepseek.com"
```

Do not put a real key in `.env.example`, source files, JSONL files, screenshots,
or Git commits.

## 3. Check credentials and model access

```powershell
python -m examples.check_deepseek
```

Expected: `DeepSeek API connection succeeded`, followed by available model IDs.
Confirm that `deepseek-flash` is present before running the paid batch test.

## 4. Run the local test suite

```powershell
python -m pytest
```

This uses fake model responses and makes no network calls.

## 5. Run the 20-case interactive batch

```powershell
askjudge run `
  --prompt .\examples\judge_prompt.md `
  --input .\examples\cases_20.jsonl `
  --output .\examples\results_20.jsonl `
  --assistance-policy .\examples\assistance_policy.md `
  --concurrency 3
```

The fixture is one homogeneous research task: pairwise judging of two model
answers in a machine-learning short-answer benchmark. It contains 15 fully
annotated samples and 5 samples whose benchmark rules require a researcher to
clarify them. Suggested answers for the five questions are:

1. Treat the missing class F1 as zero and include it in macro-F1.
2. Declare a winner only when the difference is statistically significant.
3. Use equal opportunity as the fairness definition.
4. Use `probability >= threshold` for the positive class.
5. Weight final correctness at 70% and derivation quality at 30%.

`unknown`, `skip`, or an empty answer can also be used to validate those paths.

Expected without protocol retries:

- 20 initial calls.
- 5 human-assisted final calls.
- 25 calls in total (`N + K`).
- 20 output records.
- 0 failed records.

Model routing is probabilistic. The example uses an explicit fixture routing
rule to make the expected five assisted cases as stable as possible.

## 6. Inspect the result

```powershell
Get-Content .\examples\results_20.jsonl
```

Each line should contain a user `output` string plus separate `askjudge`,
`human_answer`, `initial_output`, and `error` fields. A successful run has
`error: null` on every line.

## 7. Remove the key from the current session

```powershell
Remove-Item Env:ASKJUDGE_API_KEY
```

