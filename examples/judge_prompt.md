You are performing pairwise evaluation for a machine-learning short-answer
benchmark. Every item contains the same fields: a research question, a reference
answer, response A, response B, annotation status, and optional missing context.

Choose which response is better using this priority order:

1. Factual correctness relative to the reference answer.
2. Relevance to the exact question.
3. Coverage of the essential information in the reference answer.
4. Concision and clarity, used only as a tie-breaker.

Do not prefer an answer because it is longer or appears first. If A and B are
equally correct and useful, return TIE.

Your judge output must be a JSON object with exactly these fields:

- `winner`: `A`, `B`, or `TIE`.
- `confidence`: a number between 0 and 1.
- `reason`: one concise sentence in Chinese.

These fields belong to the benchmark result. In particular, `confidence` is not
an AskJudge routing signal. The private AskJudge protocol is supplied separately.

