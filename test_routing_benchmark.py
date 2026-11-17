"""Offline behavioral checks; also runnable with stdlib unittest."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.routing import evaluation as ev
from tests.async_helpers import run_immediate, sequential_gather


def label(case_id, needs, family=None, kind='control'):
    return dict(case_id=case_id, needs_human=needs, family=family or case_id,
                kind=kind, category='example')


def prediction(case_id, needs):
    return dict(case_id=case_id, askjudge=dict(needs_human=needs,
                question='Which convention applies?' if needs else None), error=None)


class RoutingScoringTests(unittest.TestCase):
    def test_confusion_matrix(self):
        gold = [label('tp', True), label('fn', True), label('fp', False), label('tn', False)]
        result = ev.score_predictions(gold, [prediction('tp', True), prediction('fn', False),
                                            prediction('fp', True), prediction('tn', False)])
        summary = result['summary']
        self.assertEqual([summary[x] for x in ['tp','fn','fp','tn']], [1,1,1,1])
        self.assertEqual(summary['precision'], 0.5)
        self.assertEqual(summary['recall'], 0.5)
        self.assertEqual(summary['f1'], 0.5)
        self.assertEqual(summary['overask_rate'], 0.5)

    def test_missing_and_failed_cases_are_not_dropped(self):
        gold = [label('success', True), label('missing', True), label('failed', False)]
        result = ev.score_predictions(gold, [prediction('success', True),
                                            dict(case_id='failed', error={'kind':'TimeoutError'})])
        summary = result['summary']
        self.assertEqual(summary['recall'], 0.5)
        self.assertEqual(summary['valid_only_recall'], 1)
        self.assertEqual(summary['accuracy'], 1/3)
        self.assertEqual(summary['failed'], 2)
        self.assertEqual(summary['failed_positives'], 1)

    def test_all_failed_keeps_denominators(self):
        result = ev.score_predictions([label('p', True), label('n', False)], [])['summary']
        self.assertEqual(result['coverage'], 0)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['accuracy'], 0)
        self.assertIsNone(result['precision'])

    def test_never_ask_cannot_get_perfect_f1(self):
        gold = ev.load_gold(split='all')
        cases = ev.load_cases(split='all')
        result = ev.score_predictions(gold, ev.baseline_predictions(cases, 'never_ask'))
        self.assertEqual(result['summary']['recall'], 0)
        self.assertEqual(result['summary']['f1'], 0)
        self.assertEqual(result['summary']['accuracy'], 42/72)
        self.assertEqual(result['paired']['accuracy'], 0)

    def test_always_ask_has_overask_penalty(self):
        result = ev.score_predictions(ev.load_gold(split='all'),
                    ev.baseline_predictions(ev.load_cases(split='all'), 'always_ask'))
        self.assertEqual(result['summary']['recall'], 1)
        self.assertEqual(result['summary']['overask_rate'], 1)
        self.assertEqual(result['summary']['precision'], 30/72)
        self.assertEqual(result['paired']['accuracy'], 0)

    def test_pair_must_be_correct_in_both_directions(self):
        gold = [label('p', True, 'pair', 'paired'), label('n', False, 'pair', 'paired')]
        correct = ev.score_predictions(gold, [prediction('p', True), prediction('n', False)])
        wrong = ev.score_predictions(gold, [prediction('p', False), prediction('n', True)])
        self.assertEqual(correct['paired']['accuracy'], 1)
        self.assertEqual(wrong['paired']['accuracy'], 0)

    def test_duplicate_or_foreign_ids_rejected(self):
        gold = [label('p', True)]
        for rows in [[prediction('p', True)]*2, [prediction('foreign', False)]]:
            with self.assertRaises(ValueError):
                ev.score_predictions(gold, rows)

    def test_string_booleans_and_question_protocol_rejected(self):
        for metadata in [dict(needs_human='false', question=None),
                         dict(needs_human=1, question='Why?'),
                         dict(needs_human=True, question=''),
                         dict(needs_human=False, question='Why?')]:
            decision, error = ev.routing_decision(dict(askjudge=metadata))
            self.assertIsNone(decision)
            self.assertIsNotNone(error)

    def test_final_records_are_rejected(self):
        row = prediction('p', False)
        row['human_answer'] = 'Use metric units.'
        self.assertEqual(ev.routing_decision(row)[1], 'final_record_not_initial')

    def test_output_confidence_is_not_a_routing_signal(self):
        row = prediction('p', False)
        row['output'] = '{"needs_human":true,"confidence":0.001}'
        self.assertEqual(ev.routing_decision(row), (False, None))

    def test_reports_cannot_overwrite_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.json'
            path.write_text('preserve', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'preserve'):
                ev.save_report(path, {})
            self.assertEqual(path.read_text(), 'preserve')

    def test_report_extension_prevents_json_markdown_collision(self):
        with self.assertRaisesRegex(ValueError, '.json'):
            ev.save_report(Path('report.md'), {})

    def test_random_baseline_is_reproducible_and_order_independent(self):
        cases = ev.load_cases()
        first = ev.index_rows(ev.baseline_predictions(cases, 'random', 42), 'case_id')
        second = ev.index_rows(ev.baseline_predictions(list(reversed(cases)), 'random', 42), 'case_id')
        self.assertEqual(first, second)

    def test_question_quality_requires_complete_reviews(self):
        gold = [label('p', True), label('q', True), label('n', False)]
        predicted = [prediction(row['case_id'], True) for row in gold]
        review = dict(case_id='p', focused=True, answerable=True, resolves_gap=True)
        partial = ev.score_predictions(gold, predicted, [review])['question_quality']
        self.assertIsNone(partial['effective_recall'])
        full = ev.score_predictions(gold, predicted, [review,
               dict(case_id='q', focused=True, answerable=True, resolves_gap=False)])['question_quality']
        self.assertEqual(full['effective_recall'], 0.5)
        self.assertTrue(full['effective_recall_complete'])

    def test_unreviewed_questions_do_not_get_automatic_quality_credit(self):
        result = ev.score_predictions([label('p', True)], [prediction('p', True)])
        self.assertIsNone(result['question_quality']['effective_recall'])
        self.assertIsNone(result['question_quality']['pass_rate'])

    def test_invalid_question_reviews_rejected(self):
        gold = [label('p', True), label('n', False)]
        rows = [prediction('p', True), prediction('n', False)]
        for reviews in [[dict(case_id='p', focused='yes', answerable=True, resolves_gap=True)],
                        [dict(case_id='n', focused=True, answerable=True, resolves_gap=True)]]:
            with self.assertRaises(ValueError):
                ev.score_predictions(gold, rows, reviews)


class RoutingDatasetTests(unittest.TestCase):
    def test_dataset_contract_and_split_sizes(self):
        result = ev.validate_dataset()
        self.assertEqual(result['cases'], 72)
        self.assertEqual(result['pairs'], 30)
        self.assertEqual(result['positives'], 30)
        self.assertEqual(result['splits'], {'dev':30, 'test':42})
        self.assertFalse(result['independent_human_review'])

    def test_public_payload_has_no_gold_or_family_fields(self):
        for row in ev.load_cases(split='all'):
            self.assertEqual(set(row), {'id','data'})
            self.assertEqual(set(row['data']), ev.PUBLIC_KEYS)
            self.assertNotIn('needs_human', row['data'])

    def test_all_positive_labels_have_winner_changing_evidence(self):
        for row in ev.load_gold(split='all'):
            if row['needs_human']:
                self.assertEqual(len({x['winner'] for x in row['counterfactuals']}), 2)

    def test_no_family_crosses_dev_and_test(self):
        dev = {x['family'] for x in ev.load_gold(split='dev')}
        test = {x['family'] for x in ev.load_gold(split='test')}
        self.assertFalse(dev & test)

    def test_public_loader_works_without_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['cases.jsonl','splits.json']:
                shutil.copyfile(ev.ROOT/name, root/name)
            self.assertEqual(len(ev.load_cases(root)), 30)

    def test_leaked_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['cases.jsonl','splits.json']:
                shutil.copyfile(ev.ROOT/name, root/name)
            cases = ev.read_jsonl(root/'cases.jsonl')
            cases[0]['data']['needs_human'] = True
            ev.write_jsonl(root/'cases.jsonl', cases)
            with self.assertRaisesRegex(ValueError, 'public fields'):
                ev.load_cases(root)

    def test_checksum_detects_tampered_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['cases.jsonl','gold.jsonl','splits.json','judge_prompt.md','manifest.json']:
                shutil.copyfile(ev.ROOT/name, root/name)
            (root/'judge_prompt.md').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                ev.validate_dataset(root)

    def test_scoring_rejects_metadata_from_other_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.jsonl'
            ev.write_json(path.with_suffix('.meta.json'), dict(split='test'))
            with self.assertRaisesRegex(ValueError, 'split'):
                ev.check_run_metadata(path, ev.ROOT, 'dev')

    def test_scoring_rejects_metadata_from_other_public_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.jsonl'
            ev.write_json(path.with_suffix('.meta.json'), dict(split='dev', cases_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'cases.jsonl'):
                ev.check_run_metadata(path, ev.ROOT, 'dev')

    def test_blind_review_exports_no_reference_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'review.jsonl'
            self.assertEqual(ev.main(['review','--split','dev','--output',str(out)]), 0)
            rows = ev.read_jsonl(out)
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row['needs_human'] is None for row in rows))
            self.assertTrue(all('human_answer' not in row and 'expected_winner' not in row for row in rows))

    def test_actual_runner_never_reads_gold_or_calls_final_round(self):
        calls = []
        class SpyProvider:
            def __init__(self, **kwargs):
                self.call_count = 0
            async def complete(self, *, system_prompt, user_prompt):
                calls.append((system_prompt, user_prompt))
                self.call_count += 1
                return json.dumps(dict(judge_output='anything', askjudge=dict(
                    needs_human=True, question='Which policy applies?')))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['cases.jsonl','splits.json','judge_prompt.md']:
                shutil.copyfile(ev.ROOT/name, root/name)
            out = root/'predictions.jsonl'
            args = ev.parser().parse_args(['--dataset',str(root),'run','--split','dev',
                                          '--output',str(out),'--model','fake',
                                          '--base-url','https://example.invalid'])
            with patch('askjudge.providers.OpenAICompatibleProvider', SpyProvider), \
                 patch.dict('os.environ', {'ASKJUDGE_API_KEY':'fake-key'}), \
                 patch.object(ev.asyncio, 'gather', sequential_gather):
                self.assertEqual(run_immediate(ev.run_model(args)), 0)
            self.assertEqual(len(calls), 30)
            for system_prompt, user_prompt in calls:
                payload = json.loads(user_prompt.split('\n',1)[1])
                self.assertEqual(set(payload), {'judge_prompt','evaluation_data'})
                self.assertEqual(set(json.loads(payload['evaluation_data'])), ev.PUBLIC_KEYS)
                self.assertNotIn('human_answer', payload)
                self.assertNotIn('initial_judge_output', payload)
                self.assertIn('initial judge', system_prompt)
            self.assertTrue(all(row['human_answer'] is None for row in ev.read_jsonl(out)))
            self.assertEqual(json.loads(out.with_suffix('.meta.json').read_text())['api_calls'], 30)

    def test_runner_records_failure_and_keeps_other_cases(self):
        class FailingProvider:
            def __init__(self, **kwargs):
                self.call_count = 0
            async def complete(self, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    raise TimeoutError('simulated')
                return json.dumps(dict(judge_output='anything', askjudge=dict(needs_human=False, question=None)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'predictions.jsonl'
            args = ev.parser().parse_args(['run','--output',str(out),'--model','fake',
                                          '--base-url','https://example.invalid'])
            with patch('askjudge.providers.OpenAICompatibleProvider', FailingProvider), \
                 patch.dict('os.environ', {'ASKJUDGE_API_KEY':'fake-key'}), \
                 patch.object(ev.asyncio, 'gather', sequential_gather):
                self.assertEqual(run_immediate(ev.run_model(args)), 2)
            rows = ev.read_jsonl(out)
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(row['error'] is not None for row in rows), 1)
            report = ev.score_predictions(ev.load_gold(), rows)
            self.assertEqual(report['summary']['failed'], 1)


if __name__ == '__main__':
    unittest.main()
