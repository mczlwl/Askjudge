"""Evaluate initial routing without exposing answer keys to the judge.

The model runner reads only public cases, split IDs, and the judge prompt.
Gold labels are loaded by validation and offline scoring, never by run_model.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from askjudge import AskJudge

ROOT = Path(__file__).resolve().parent
PUBLIC_KEYS = {'task', 'rubric', 'response_a', 'response_b', 'context'}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line_number}: expected an object')
        rows.append(row)
    return rows


def index_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        case_id = row.get(key)
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f'every record requires a nonempty {key}')
        if case_id in indexed:
            raise ValueError(f'duplicate {key}: {case_id}')
        indexed[case_id] = row
    return indexed


def load_cases(root: Path = ROOT, split: str = 'dev') -> list[dict[str, Any]]:
    """Public-only loader, deliberately independent of gold.jsonl."""
    cases = read_jsonl(root / 'cases.jsonl')
    by_id = index_rows(cases, 'id')
    for case in cases:
        if set(case) != {'id', 'data'} or not isinstance(case['data'], dict):
            raise ValueError('public cases must contain only id and a data object')
        if set(case['data']) != PUBLIC_KEYS:
            raise ValueError(f'{case["id"]}: unexpected or missing public fields')
        if any(not isinstance(value, str) or not value.strip() for value in case['data'].values()):
            raise ValueError('all public fields must be nonempty strings')
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    if set(splits) != {'dev', 'test'} or any(not isinstance(v, list) for v in splits.values()):
        raise ValueError('splits.json must define dev and test ID lists')
    all_ids = [case_id for ids in splits.values() for case_id in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
        raise ValueError('splits must partition every public ID exactly once')
    if split == 'all':
        return cases
    return [by_id[case_id] for case_id in splits[split]]


def load_gold(root: Path = ROOT, split: str = 'dev') -> list[dict[str, Any]]:
    gold = read_jsonl(root / 'gold.jsonl')
    index_rows(gold, 'case_id')
    selected = {case['id'] for case in load_cases(root, split)}
    return [row for row in gold if row['case_id'] in selected]


def validate_dataset(root: Path = ROOT) -> dict[str, Any]:
    cases = load_cases(root, 'all')
    gold = read_jsonl(root / 'gold.jsonl')
    labels = index_rows(gold, 'case_id')
    if set(labels) != {case['id'] for case in cases}:
        raise ValueError('gold and public IDs must match exactly')
    split_ids = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    families = defaultdict(list)
    for row in gold:
        if type(row.get('needs_human')) is not bool:
            raise ValueError('gold needs_human must be a boolean')
        if row.get('split') not in split_ids or row['case_id'] not in split_ids[row['split']]:
            raise ValueError('gold split disagrees with public split IDs')
        if row.get('expected_winner') not in {'A', 'B', 'TIE'}:
            raise ValueError('gold winner must be A, B, or TIE')
        if not row.get('rationale'):
            raise ValueError('every reference label requires a rationale')
        if row['needs_human']:
            alternatives = row.get('counterfactuals', [])
            if (len(alternatives) != 2
                    or any(not alt.get('answer') or alt.get('winner') not in {'A', 'B', 'TIE'} for alt in alternatives)
                    or len({alt['winner'] for alt in alternatives}) != 2):
                raise ValueError('positive labels require two winner-changing contexts')
            if not row.get('human_answer') or not row.get('information_needed'):
                raise ValueError('positive labels require a missing-information explanation and answer')
            if {'answer': row['human_answer'], 'winner': row['expected_winner']} not in alternatives:
                raise ValueError('reference human answer must match a counterfactual')
        elif row.get('human_answer') is not None or row.get('information_needed') is not None or row.get('counterfactuals'):
            raise ValueError('negative labels cannot carry assistance answers')
        families[row['family']].append(row)
    paired_count = 0
    public_by_id = index_rows(cases, 'id')
    for rows in families.values():
        if len({row['split'] for row in rows}) != 1:
            raise ValueError('a family cannot cross dev/test')
        if all(row['kind'] == 'paired' for row in rows):
            if len(rows) != 2 or {row['needs_human'] for row in rows} != {True, False}:
                raise ValueError('each paired family requires one positive and one negative')
            positive = next(row for row in rows if row['needs_human'])
            negative = next(row for row in rows if not row['needs_human'])
            if (positive['expected_winner'] != negative['expected_winner']
                    or positive['category'] != negative['category']):
                raise ValueError('paired labels must agree on category and resolved winner')
            left = public_by_id[positive['case_id']]['data']
            right = public_by_id[negative['case_id']]['data']
            if any(left[key] != right[key] for key in PUBLIC_KEYS - {'context'}):
                raise ValueError('paired task, rubric, and candidate responses must be identical')
            if right['context'] != left['context'] + '\n' + positive['human_answer']:
                raise ValueError('the resolved case must add exactly the reference context')
            paired_count += 1
        elif len(rows) != 1 or rows[0]['kind'] != 'control' or rows[0]['needs_human']:
            raise ValueError('standalone controls must be negative singletons')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    expected_files = {'cases.jsonl', 'gold.jsonl', 'splits.json', 'judge_prompt.md'}
    if set(manifest['files']) != expected_files:
        raise ValueError('manifest must hash all four evaluation inputs')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'manifest checksum mismatch: {name}')
    if manifest['cases'] != len(cases) or manifest['pairs'] != paired_count:
        raise ValueError('manifest size does not match fixtures')
    return dict(cases=len(cases), pairs=paired_count,
                positives=sum(row['needs_human'] for row in gold),
                negatives=sum(not row['needs_human'] for row in gold),
                splits=dict(Counter(row['split'] for row in gold)),
                independent_human_review=manifest['independent_human_review'])


def routing_decision(row: dict[str, Any] | None) -> tuple[bool | None, str | None]:
    if row is None:
        return None, 'missing_prediction'
    if row.get('error') is not None:
        return None, 'model_or_protocol_error'
    # Final records set needs_human=false and would corrupt a routing evaluation.
    if row.get('human_answer') is not None or row.get('initial_output') is not None:
        return None, 'final_record_not_initial'
    metadata = row.get('askjudge')
    if not isinstance(metadata, dict) or type(metadata.get('needs_human')) is not bool:
        return None, 'invalid_routing_metadata'
    needs = metadata['needs_human']
    question = metadata.get('question')
    if needs and (not isinstance(question, str) or not question.strip()):
        return None, 'missing_question'
    if not needs and question is not None:
        return None, 'unexpected_question'
    return needs, None


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row['outcome'] for row in rows)
    tp, fp, tn, fn = (counts[key] for key in ('tp', 'fp', 'tn', 'fn'))
    positives = sum(row['expected'] for row in rows)
    negatives = len(rows) - positives
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, positives)  # Missing/failed positives cannot disappear.
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0)
    return dict(total=len(rows), positives=positives, negatives=negatives,
                tp=tp, fp=fp, tn=tn, fn=fn, failed=counts['error'],
                failed_positives=sum(row['expected'] for row in rows if row['outcome'] == 'error'),
                coverage=ratio(len(rows) - counts['error'], len(rows)),
                accuracy=ratio(tp + tn, len(rows)), precision=precision,
                recall=recall, f1=f1, overask_rate=ratio(fp, negatives),
                human_request_rate=ratio(tp + fp, len(rows)),
                valid_only_recall=ratio(tp, tp + fn))


def score_predictions(gold: list[dict[str, Any]], predictions: list[dict[str, Any]],
                      reviews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    labels = index_rows(gold, 'case_id')
    predicted = index_rows(predictions, 'case_id')
    unknown = set(predicted) - set(labels)
    if unknown:
        raise ValueError(f'predictions contain IDs outside the selected split: {sorted(unknown)}')
    details = []
    for case_id, label in labels.items():
        decision, error = routing_decision(predicted.get(case_id))
        outcome = ('error' if error else
                   'tp' if label['needs_human'] and decision else
                   'fn' if label['needs_human'] else 'fp' if decision else 'tn')
        metadata = (predicted.get(case_id) or {}).get('askjudge')
        details.append(dict(case_id=case_id, family=label['family'], category=label['category'],
                            expected=label['needs_human'], predicted=decision,
                            outcome=outcome, error=error,
                            question=metadata.get('question') if isinstance(metadata, dict) else None))
    grouped = defaultdict(list)
    families = defaultdict(list)
    for row in details:
        grouped[row['category']].append(row)
        if labels[row['case_id']]['kind'] == 'paired':
            families[row['family']].append(row)
    pairs = [rows for rows in families.values() if len(rows) == 2]
    pair_correct = sum(all(row['outcome'] in {'tp', 'tn'} for row in rows) for rows in pairs)
    report = dict(schema_version='0.1.0', summary=metrics(details),
                  by_category={category: metrics(rows) for category, rows in sorted(grouped.items())},
                  paired=dict(total=len(pairs), both_correct=pair_correct,
                              accuracy=ratio(pair_correct, len(pairs))), cases=details)
    asked = {row['case_id']: row for row in details if row['predicted'] is True}
    review_index = index_rows(reviews or [], 'case_id')
    if set(review_index) - set(asked):
        raise ValueError('question reviews must refer only to valid predicted human questions')
    quality_keys = ('focused', 'answerable', 'resolves_gap')
    for review in review_index.values():
        if any(type(review.get(key)) is not bool for key in quality_keys):
            raise ValueError('each question review needs boolean focused, answerable, resolves_gap')
    passes = {case_id for case_id, review in review_index.items() if all(review[key] for key in quality_keys)}
    positive_asks = {case_id for case_id, row in asked.items() if row['expected']}
    complete = positive_asks <= set(review_index)
    report['question_quality'] = dict(
        questions=len(asked), reviewed=len(review_index),
        review_coverage=ratio(len(review_index), len(asked)),
        pass_rate=ratio(len(passes), len(review_index)),
        effective_recall=(ratio(len(passes & positive_asks), report['summary']['positives']) if complete else None),
        effective_recall_complete=complete,
        note='Effective recall requires every requested positive question to be reviewed; no keyword or model proxy is used.')
    return report


def baseline_predictions(cases: list[dict[str, Any]], strategy: str, seed: int = 17) -> list[dict[str, Any]]:
    # Labels are intentionally unavailable in this function.
    rows = []
    for case in cases:
        needs = (True if strategy == 'always_ask' else False if strategy == 'never_ask'
                 else random.Random(f'{seed}:{case["id"]}').random() < 0.5)
        rows.append(dict(case_id=case['id'], output='baseline (no model judgment)',
                         askjudge=dict(needs_human=needs, question='What deciding context applies?' if needs else None),
                         human_answer=None, initial_output=None, error=None))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def percent(value: float | None) -> str:
    return 'N/A' if value is None else f'{100 * value:.1f}%'


def markdown_report(report: dict[str, Any]) -> str:
    summary = report['summary']
    lines = ['# AskJudge routing report', '',
             'Synthetic fixtures with provisional author labels; this is not a production reliability estimate.', '',
             f'Cases: {summary["total"]}; failed/missing: {summary["failed"]}; valid coverage: {percent(summary["coverage"])}.', '',
             '| Category | N | Recall | Over-asking | Accuracy | Failures |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in [('ALL', summary), *report['by_category'].items()]:
        lines.append(f'| {name} | {result["total"]} | {percent(result["recall"])} | {percent(result["overask_rate"])} | {percent(result["accuracy"])} | {result["failed"]} |')
    lines += ['', f'Precision: {percent(summary["precision"])}; F1: {summary["f1"]:.3f}.',
              f'Paired accuracy (both missing and resolved cases correct): {percent(report["paired"]["accuracy"])}.',
              '', 'Recall = TP / all required-help cases, including failed/missing positives in the denominator.',
              'Over-asking = FP / all no-help cases. Accuracy counts every failure/missing prediction as incorrect.',
              'Precision is N/A when no questions were requested. F1 is 0 when there are no true positive requests.',
              'The JSON report also includes valid-only recall and all per-case outcomes.', '',
              '## Question quality', '',
              f'Reviewed: {report["question_quality"]["reviewed"]}/{report["question_quality"]["questions"]}.',
              f'Effective recall: {percent(report["question_quality"]["effective_recall"])}.',
              'A correct request flag does not establish that its question is useful. Review questions manually.', '',
              '## Routing errors', '']
    errors = [row for row in report['cases'] if row['outcome'] not in {'tp', 'tn'}]
    lines += [f'- `{row["case_id"]}`: {row["outcome"]}' + (f' ({row["error"]})' if row['error'] else '') for row in errors]
    if not errors:
        lines.append('None.')
    return '\n'.join(lines) + '\n'


def save_report(path: Path, report: dict[str, Any]) -> None:
    if path.suffix != '.json':
        raise ValueError('report output must use the .json extension')
    if path.exists() or path.with_suffix('.md').exists():
        raise ValueError('choose a new report path to preserve prior results')
    write_json(path, report)
    path.with_suffix('.md').write_text(markdown_report(report), encoding='utf-8')


def check_run_metadata(predictions_path: Path, root: Path, split: str) -> dict[str, Any] | None:
    """Bind a runner-produced prediction file to its split and public inputs."""
    meta_path = predictions_path.with_suffix('.meta.json')
    if not meta_path.exists():
        return None  # External initial-batch files are supported too.
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('split') != split:
        raise ValueError('prediction metadata split does not match --split')
    for field, name in [('cases_sha256', 'cases.jsonl'), ('splits_sha256', 'splits.json')]:
        if metadata.get(field) != hashlib.sha256((root/name).read_bytes()).hexdigest():
            raise ValueError(f'prediction metadata does not match current {name}')
    return metadata


async def run_model(args: argparse.Namespace) -> int:
    """Run initial judgments only. No label loader or resume call occurs here."""
    from askjudge.providers import OpenAICompatibleProvider

    if args.concurrency < 1 or args.max_protocol_retries < 0 or args.timeout <= 0:
        raise ValueError('concurrency/timeout must be positive; retries must be nonnegative')
    cases = load_cases(args.dataset, args.split)
    prompt_path = args.judge_prompt or args.dataset / 'judge_prompt.md'
    prompt = prompt_path.read_text(encoding='utf-8')
    policy = args.assistance_policy.read_text(encoding='utf-8') if args.assistance_policy else None
    if not prompt.strip() or (policy is not None and not policy.strip()):
        raise ValueError('judge prompt and supplied assistance policy must not be empty')
    if args.output.exists() or args.output.with_suffix('.meta.json').exists():
        raise ValueError('choose a new output path to preserve prior runs')
    key = os.environ.get('ASKJUDGE_API_KEY') or os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('API key: ')
    provider = OpenAICompatibleProvider(api_key=key, base_url=args.base_url,
                                      model=args.model, timeout=args.timeout,
                                      json_mode=not args.no_json_mode)
    judge = AskJudge(provider, assistance_policy=policy, max_protocol_retries=args.max_protocol_retries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = dict(split=args.split, cases=len(cases), model=args.model,
                    base_url=args.base_url, concurrency=args.concurrency,
                    max_protocol_retries=args.max_protocol_retries,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    cases_sha256=hashlib.sha256((args.dataset/'cases.jsonl').read_bytes()).hexdigest(),
                    splits_sha256=hashlib.sha256((args.dataset/'splits.json').read_bytes()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    assistance_policy=policy or 'AskJudge default', status='running')
    write_json(args.output.with_suffix('.meta.json'), metadata)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0
    finished = 0
    with args.output.open('x', encoding='utf-8') as stream:
        async def evaluate(case: dict[str, Any]) -> None:
            nonlocal failures, finished
            async with semaphore:
                try:
                    record = await judge.evaluate(case_id=case['id'], judge_prompt=prompt,
                                                  evaluation_data=json.dumps(case['data'], ensure_ascii=False))
                    row = record.to_dict()
                except Exception as exc:
                    row = dict(case_id=case['id'], output=None, askjudge=None,
                               human_answer=None, initial_output=None,
                               error=dict(kind=type(exc).__name__, message=str(exc)))
                    failures += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                finished += 1
                print(f'Initial judgments: {finished}/{len(cases)}; failures: {failures}', flush=True)
        try:
            await asyncio.gather(*(evaluate(case) for case in cases))
        finally:
            metadata.update(status='complete' if finished == len(cases) else 'interrupted',
                            finished=finished, failures=failures, api_calls=provider.call_count,
                            elapsed_seconds=round(time.monotonic() - started, 3))
            write_json(args.output.with_suffix('.meta.json'), metadata)
    return 2 if failures else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', type=Path, default=ROOT)
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('validate', help='validate labels, paired controls, splits, and checksums')
    for command in ('run', 'score', 'baselines', 'review'):
        sub = commands.add_parser(command)
        sub.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
        if command in {'score', 'review'}:
            sub.add_argument('--predictions', type=Path, required=command == 'score')
        if command in {'run', 'score', 'review'}:
            sub.add_argument('--output', type=Path, required=True)
        if command == 'score':
            sub.add_argument('--question-reviews', type=Path)
        if command == 'baselines':
            sub.add_argument('--out-dir', type=Path, required=True)
            sub.add_argument('--seed', type=int, default=17)
        if command == 'run':
            sub.add_argument('--model', default=os.environ.get('ASKJUDGE_MODEL'), required=not os.environ.get('ASKJUDGE_MODEL'))
            sub.add_argument('--base-url', default=os.environ.get('ASKJUDGE_BASE_URL'), required=not os.environ.get('ASKJUDGE_BASE_URL'))
            sub.add_argument('--judge-prompt', type=Path)
            sub.add_argument('--assistance-policy', type=Path)
            sub.add_argument('--concurrency', type=int, default=3)
            sub.add_argument('--timeout', type=float, default=90)
            sub.add_argument('--max-protocol-retries', type=int, default=1)
            sub.add_argument('--no-json-mode', action='store_true')
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == 'run':
            return asyncio.run(run_model(args))
        if args.command == 'validate':
            print(json.dumps(validate_dataset(args.dataset), ensure_ascii=False, indent=2))
            return 0
        # Offline commands may inspect gold; the runner above does not.
        validate_dataset(args.dataset)
        cases = load_cases(args.dataset, args.split)
        gold = load_gold(args.dataset, args.split)
        if args.command == 'score':
            run_metadata = check_run_metadata(args.predictions, args.dataset, args.split)
            report = score_predictions(gold, read_jsonl(args.predictions),
                                       read_jsonl(args.question_reviews) if args.question_reviews else None)
            report['split'] = args.split
            report['predictions_sha256'] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
            report['dataset_version'] = json.loads((args.dataset/'manifest.json').read_text())['version']
            report['run_metadata'] = run_metadata
            report['dataset_sha256'] = hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
            save_report(args.output, report)
            print(json.dumps(report['summary'], indent=2))
        elif args.command == 'baselines':
            names = [f'{strategy}.{suffix}'
                     for strategy in ('never_ask', 'always_ask', 'random')
                     for suffix in ('predictions.jsonl', 'report.json', 'report.md')]
            if any((args.out_dir/name).exists() for name in names):
                raise ValueError('choose a new baseline directory to preserve prior results')
            for strategy in ('never_ask', 'always_ask', 'random'):
                predictions = baseline_predictions(cases, strategy, args.seed)
                write_jsonl(args.out_dir/f'{strategy}.predictions.jsonl', predictions)
                report = score_predictions(gold, predictions)
                report.update(strategy=strategy, split=args.split, seed=args.seed)
                save_report(args.out_dir/f'{strategy}.report.json', report)
                print(strategy + ': ' + json.dumps(report['summary']))
        else:
            # Without predictions: blind label review. With predictions: question-quality review.
            if args.output.exists():
                raise ValueError('choose a new review output path')
            if args.predictions:
                predicted = index_rows(read_jsonl(args.predictions), 'case_id')
                if set(predicted) - {case['id'] for case in cases}:
                    raise ValueError('predictions contain IDs outside the selected split')
                review = [dict(case_id=case['id'], data=case['data'],
                               question=predicted[case['id']]['askjudge']['question'],
                               focused=None, answerable=None, resolves_gap=None, notes='')
                          for case in cases if routing_decision(predicted.get(case['id']))[0] is True]
            else:
                review = [dict(case_id=case['id'], data=case['data'], needs_human=None,
                               rationale='', question_if_needed='', notes='') for case in cases]
            write_jsonl(args.output, review)
            print(f'Wrote {len(review)} blinded review records.')
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed initial records remain saved.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 1
