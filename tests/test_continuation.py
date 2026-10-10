"""Failure boundaries, lineage, and cost accounting without model execution."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.common.artifact_io import DataContractError, atomic_write_json, load_json
from utils.common.execution import checked_process, exclusive_run
from utils.common.paths import RunLayout
from utils.test.fixtures import load_pipeline_config
from utils.rag import evaluator, runner
from utils.rag.journal import evaluate_ordered
from utils.verifier.verifier import _EventLog, _live_response
from utils.evaluation.reporting import response_statistics, training_statistics, Evidence, _write
from stages.comparison_matrix import require_matrix_progress
from utils.encoder_comparison.config import load_comparison_config

ROOT=Path(__file__).resolve().parents[1]


class ContinuationTests(unittest.TestCase):
    def test_exclusive_os_lock_and_release(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'locked')
            layout.create()
            with exclusive_run(layout):
                with self.assertRaises(DataContractError):
                    with exclusive_run(layout): pass
            with exclusive_run(layout): pass

    def test_failed_dispatch_is_charged_and_cannot_reset_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'budget')
            layout.create()
            command=mock.Mock(side_effect=subprocess.CalledProcessError(1,['fixture']))
            with mock.patch('utils.common.execution.time.monotonic',side_effect=[0,10]):
                with self.assertRaises(subprocess.CalledProcessError):
                    checked_process(layout,['fixture'],label='comparison-full',limit_seconds=10,command_runner=command)
            with self.assertRaisesRegex(DataContractError,'exhausted'):
                checked_process(layout,['fixture'],label='comparison-full',limit_seconds=10,command_runner=command)
            self.assertEqual(command.call_count,1)
            self.assertEqual(load_json(layout.resolve('logs/invocations/comparison-full/0001-finish.json'))['status'],'failed')

    def test_verifier_preserves_original_timing_and_terminal_budget(self):
        # The already validated parser is isolated so this test targets durable attempt behavior.
        config=load_pipeline_config(ROOT,'resources/configs/pipeline.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            directory=Path(temporary)
            request={'request_sha256':'a'*64,'cache_key':'b'*64,'candidate_id':'fixture',
                     'payload':{},'_candidate':object(),'_sentence':object(),
                     '_journal_path':directory/'attempts.json'}
            log=_EventLog(directory/'events.jsonl','VER-SIMPLE','live')
            transport=mock.Mock(side_effect=TimeoutError)
            with mock.patch('utils.verifier.verifier._attempt_result',return_value={'response_status':'timeout'}):
                first=_live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
                raw=(directory/'attempts.json').read_bytes()
                second=_live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
            self.assertEqual(transport.call_count,3)
            self.assertEqual(first,second)
            self.assertFalse(second['cache_hit'])
            self.assertEqual((directory/'attempts.json').read_bytes(),raw)
            self.assertEqual(len(log.events),3)
            atomic_write_json(directory/'attempts.json',{'binding':{'request_sha256':'foreign','cache_key':'b'*64},'attempts':[]})
            with self.assertRaisesRegex(DataContractError,'another request'):
                _live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)

    def test_verifier_continues_after_saved_attempt_and_missing_event(self):
        config=load_pipeline_config(ROOT,'resources/configs/pipeline.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            directory=Path(temporary)
            request={'request_sha256':'a'*64,'cache_key':'b'*64,'candidate_id':'fixture',
                     'payload':{},'_candidate':object(),'_sentence':object(),
                     '_journal_path':directory/'attempts.json'}
            log=_EventLog(directory/'events.jsonl','VER-SIMPLE','live')
            original_append=log.append
            transport=mock.Mock(side_effect=TimeoutError)
            with mock.patch('utils.verifier.verifier._attempt_result',return_value={'response_status':'timeout'}):
                with mock.patch.object(log,'append',side_effect=RuntimeError('event receipt interrupted')):
                    with self.assertRaises(RuntimeError):
                        _live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
                self.assertEqual(len(load_json(directory/'attempts.json')['attempts']),1)
                _live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
            self.assertEqual(transport.call_count,3)
            self.assertEqual([r['attempt'] for r in log.events],[1,2,3])

    def test_reports_are_deterministic_and_reject_changed_measurements(self):
        measured=[{'cache_hit':False,'attempts':[{'elapsed_seconds':1.25,'transport_status':'response',
            'http_status':200,'raw_response':{'prompt_eval_count':10,'eval_count':2}}]},
            {'cache_hit':False,'attempts':[{'elapsed_seconds':.75,'transport_status':'timeout','http_status':None},
                {'elapsed_seconds':2.,'transport_status':'response','http_status':200,'raw_response':{'prompt_eval_count':20,'eval_count':3}}]}]
        result=response_statistics(measured)
        self.assertEqual(result['uncached_attempts'],3)
        self.assertEqual(result['recorded_request_seconds'],4.)
        self.assertEqual(result['retry_attempts'],1)
        self.assertEqual(result['tokens']['prompt_eval_count']['sum'],30)
        self.assertEqual(result['tokens']['prompt_eval_count']['attempts_with_measurement'],2)
        self.assertIsNone(training_statistics([],list(range(42,50)))['recorded_wall_hours'])
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'report')
            layout.create()
            evidence=Evidence(layout)
            first=_write(evidence,'reports',result)
            self.assertEqual(_write(evidence,'reports',result),first)
            with self.assertRaisesRegex(DataContractError,'differs'):
                _write(evidence,'reports',{**result,'uncached_attempts':4})

    def test_matrix_requires_every_smoke_before_any_full_training(self):
        config = load_comparison_config(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            root = Path(temporary)
            layout = RunLayout(root, 'matrix-arm')
            members = {arm: f'arm-{i}' for i, arm in enumerate(config.arms)}
            for run_id in list(members.values())[:-1]:
                member = RunLayout(root, run_id)
                member.create()
                atomic_write_json(member.resolve('manifests/encoder-comparison-smoke.json'), {})
            with (mock.patch('stages.comparison_matrix.selected_matrix',return_value=(layout,{'members':members,'training_seeds':list(range(42,50))})),
                  mock.patch('stages.encoder_comparison.run_smoke') as smoke):
                with self.assertRaisesRegex(DataContractError,'all four'):
                    require_matrix_progress(layout,config.select_arm(next(iter(config.arms))),boundary='training')
                self.assertLess(smoke.call_count,4)

    def test_matrix_requires_all_development_records_before_test(self):
        config = load_comparison_config(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            root = Path(temporary)
            layout = RunLayout(root, 'matrix-arm')
            members = {arm: f'arm-{i}' for i, arm in enumerate(config.arms)}
            for run_id in members.values():
                member = RunLayout(root, run_id)
                member.create()
                atomic_write_json(member.resolve('manifests/encoder-comparison-smoke.json'), {})
            with (mock.patch('stages.comparison_matrix.selected_matrix',return_value=(layout,{'members':members,'training_seeds':list(range(42,50))})),
                  mock.patch('stages.encoder_comparison.run_smoke')):
                with self.assertRaisesRegex(DataContractError,'all 32'):
                    require_matrix_progress(layout,config.select_arm(next(iter(config.arms))),boundary='test')


class RagOccurrencesTests(unittest.TestCase):
    def _arguments(self,directory):
        record={'doc_id':'fixture','sentence':'Fixture head part of tail.',
                'gold_triples':[{'head_text':'head','relation':'part-of','tail_text':'tail'}],
                'predicted_triples':[]}
        model={'name':'qwen3:32b','tag_digest':'a'*64,'blob_sha256':'b'*64,
               'details':{'family':'qwen3','parameter_size':'32.8B','quantization_level':'Q4_K_M'}}
        tags=runner.canonical_json_bytes({'models':[{'name':model['name'],'digest':model['tag_digest']}]})
        show=runner.canonical_json_bytes({'details':model['details'],'modelfile':'FROM sha256-'+model['blob_sha256']})
        capture=({**model,'tags_response_sha256':runner.sha256_bytes(tags),'show_response_sha256':runner.sha256_bytes(show)},tags,show)
        args={'records':[record],'kg':{'nodes':[],'edges':[]},'questions':evaluator.generate_questions([record]),
              'condition':'gold','identity':{'run_id':'fixture'},'child_dir':directory,
              'status':{'stages':{}},'status_path':directory/'run-status.json',
              'ollama_url':'http://localhost:11434','model_identity':model}
        return args,capture

    def test_rejected_occurrence_retries_while_successes_keep_their_answers(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            args,capture=self._arguments(Path(temporary))
            with (mock.patch.object(evaluator,'call_llm',side_effect=['ERROR: fixture',*['tail']*4]) as call,
                    mock.patch.object(runner,'fetch_matching_model',return_value=capture)):
                first=evaluate_ordered(**args)
                self.assertEqual(call.call_count,5)
                self.assertFalse(first['results']['llm_only'][0]['correct'])
            original=sorted((args['child_dir']/'calls/gold').rglob('*-answer.json'))
            hashes={p:runner.sha256_file(p) for p in original}
            with (mock.patch.object(evaluator,'call_llm',return_value='tail') as call,
                    mock.patch.object(runner,'fetch_matching_model',return_value=capture)):
                second=evaluate_ordered(**args)
                self.assertEqual(call.call_count,1)
            self.assertEqual({p:runner.sha256_file(p) for p in original},hashes)
            self.assertTrue(all(rows[0]['correct'] for rows in second['results'].values()))
            with (mock.patch.object(evaluator,'call_llm') as call,
                    mock.patch.object(runner,'fetch_matching_model') as fetch):
                evaluate_ordered(**args)
                call.assert_not_called();fetch.assert_not_called()

    def test_missing_contemporaneous_model_proof_never_requeries_or_reruns(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            args,capture=self._arguments(Path(temporary))
            with (mock.patch.object(evaluator,'call_llm',return_value='tail') as call,
                    mock.patch.object(runner,'fetch_matching_model',side_effect=runner.PhaseEError('interrupted'))):
                with self.assertRaises(runner.PhaseEError): evaluate_ordered(**args)
            with (mock.patch.object(evaluator,'call_llm') as call,
                    mock.patch.object(runner,'fetch_matching_model') as fetch):
                with self.assertRaisesRegex(runner.ModelEvidenceError,'contemporaneous'):
                    evaluate_ordered(**args)
                call.assert_not_called();fetch.assert_not_called()
