"""Failure boundaries, lineage, and cost accounting without model execution."""
from __future__ import annotations

import json
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.common.artifact_io import DataContractError, atomic_write_json, atomic_write_jsonl, load_json
from utils.common.execution import checked_process, exclusive_run
from utils.common.paths import RunLayout
from utils.test.fixtures import load_pipeline_config
from utils.rag import evaluator, runner
from utils.rag.journal import evaluate_ordered
from utils.verifier.verifier import _EventLog, _live_response, HttpResult, run_verifier
from tests.test_verifier import _sentence, _candidate
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

    def test_verifier_rejects_changed_or_deleted_attempts_before_calling(self):
        config=load_pipeline_config(ROOT,'resources/configs/pipeline.json')
        for mutation in ('timing', 'delete'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
                directory=Path(temporary)
                request={'request_sha256':'a'*64,'cache_key':'b'*64,'candidate_id':'fixture',
                         'payload':{},'_candidate':object(),'_sentence':object(),
                         '_journal_path':directory/'attempts.json'}
                log=_EventLog(directory/'events.jsonl','VER-SIMPLE','live')
                transport=mock.Mock(side_effect=TimeoutError)
                with mock.patch('utils.verifier.verifier._attempt_result',return_value={'response_status':'timeout'}):
                    _live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
                    if mutation == 'timing':
                        saved=load_json(request['_journal_path'])
                        saved['attempts'][0]['elapsed_seconds'] += 1
                        atomic_write_json(request['_journal_path'],saved)
                    else:
                        request['_journal_path'].unlink()
                    transport.reset_mock()
                    with self.assertRaisesRegex(DataContractError,'evidence is missing|changed after'):
                        _live_response(request,config,'simple','http://localhost:11434',transport,log,lambda _:None)
                    transport.assert_not_called()

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

    def test_completed_rag_replay_preserves_exact_output_and_condition_timing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            args,capture=self._arguments(Path(temporary))
            with (mock.patch.object(evaluator,'call_llm',return_value='tail'),
                  mock.patch.object(runner,'fetch_matching_model',return_value=capture),
                  mock.patch('utils.rag.journal.time.time',side_effect=range(100,120))):
                first=evaluate_ordered(**args)
            recorded=args['status_path'].read_bytes()
            with (mock.patch.object(evaluator,'call_llm') as call,
                  mock.patch.object(runner,'fetch_matching_model') as fetch,
                  mock.patch('utils.rag.journal.time.time',return_value=5000)):
                replay=evaluate_ordered(**args,completed_only=True)
            self.assertGreater(first['metadata']['time_seconds'],0)
            self.assertEqual(runner.scientific_json_bytes(replay),runner.scientific_json_bytes(first))
            self.assertEqual(args['status_path'].read_bytes(),recorded)
            call.assert_not_called();fetch.assert_not_called()

    def test_saved_rag_evidence_is_preflighted_before_retrying_an_earlier_failure(self):
        for mutation in ('delete_occurrence','delete_proof','change_answer','delete_plan'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
                args,capture=self._arguments(Path(temporary))
                with (mock.patch.object(evaluator,'call_llm',side_effect=['ERROR: fixture',*['tail']*4]),
                      mock.patch.object(runner,'fetch_matching_model',return_value=capture)):
                    evaluate_ordered(**args)
                directory=args['child_dir']/'calls/gold'
                answer=directory/'000004/001-answer.json'
                proof=directory/'000004/001-model.json'
                if mutation == 'delete_occurrence':
                    answer.unlink();proof.unlink()
                elif mutation == 'delete_proof':
                    proof.unlink()
                elif mutation == 'change_answer':
                    saved=runner.load_json(answer)
                    saved['answer']='changed'
                    answer.write_bytes(runner.canonical_json_bytes(saved))
                else:
                    (directory/'requests.json').unlink()
                with (mock.patch.object(evaluator,'call_llm') as call,
                      mock.patch.object(runner,'fetch_matching_model') as fetch):
                    with self.assertRaises(runner.PhaseEError):
                        evaluate_ordered(**args)
                    call.assert_not_called();fetch.assert_not_called()

    def test_completed_rag_validation_never_retries_rejected_occurrences(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            args,capture=self._arguments(Path(temporary))
            with (mock.patch.object(evaluator,'call_llm',side_effect=['ERROR: fixture',*['tail']*4]),
                  mock.patch.object(runner,'fetch_matching_model',return_value=capture)):
                evaluate_ordered(**args)
            with (mock.patch.object(evaluator,'call_llm') as call,
                  mock.patch.object(runner,'fetch_matching_model') as fetch):
                with self.assertRaisesRegex(runner.PhaseEError,'successful occurrence evidence'):
                    evaluate_ordered(**args,completed_only=True)
                call.assert_not_called();fetch.assert_not_called()

    def test_validated_stage_output_is_reused_without_quarantine(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            directory=Path(temporary)
            path=directory/'results.json'
            path.write_bytes(b'{"fixture": true}')
            original=path.read_bytes()
            status={'stages':{'rag_gold':{'attempts':[], 'status':'output-validated',
                                        'output_sha256':runner.sha256_file(path)}}}
            execute=mock.Mock()
            runner.run_stage('rag_gold',execute,path,runner.load_json,directory,status,directory/'status.json')
            execute.assert_not_called()
            self.assertEqual(path.read_bytes(),original)
            self.assertFalse((directory/'recovery').exists())
            path.write_bytes(b'{"fixture": false}')
            with self.assertRaisesRegex(runner.PhaseEError,'no longer matches'):
                runner.run_stage('rag_gold',execute,path,runner.load_json,directory,status,directory/'status.json')
            execute.assert_not_called()


class VerifierLifecycleTests(unittest.TestCase):
    def _fixture(self, directory):
        layout=RunLayout(directory,'verifier-resume')
        layout.create()
        config=load_pipeline_config(ROOT,'resources/configs/pipeline.json')
        atomic_write_json(layout.resolve('manifests/00-checkout-manifest.json'),{'source':{'commit':'3'*40}})
        inputs={
            'sentences_path':('data-prepared/test.jsonl',[_sentence()]),
            'candidates_path':('predictions/test/candidates.jsonl',[_candidate()]),
            'warmup_sentences_path':('data-prepared/development.jsonl',[_sentence('warm',split='development')]),
            'warmup_candidates_path':('predictions/dev/warmup.jsonl',[_candidate('warm',split='development')]),
        }
        arguments={'layout':layout,'config':config,'mode':'simple','execution_mode':'live','sleep':lambda _:None}
        for name,(relative,rows) in inputs.items():
            arguments[name]=layout.resolve(relative)
            atomic_write_jsonl(arguments[name],rows)
        blob=layout.resolve('inputs/ollama/blob')
        blob.parent.mkdir(parents=True,exist_ok=True)
        blob.write_bytes(b'fixture')
        arguments['model_blob_path']=blob
        environment={'python':{},'platform':{},'hardware':{},'software':{'packages':{}},
                     'ollama':{'version_command':{}},'model':{}}
        evidence={'tag_digest':config.value['verifier']['registry_manifest_sha256'],
                  'blob_sha256':config.value['verifier']['model_blob_sha256']}

        def verify_model(*_args, artifact_base, **_kwargs):
            atomic_write_json(layout.resolve(f'{artifact_base}/model/tags.json'),{'fixture':'tags'})
            atomic_write_json(layout.resolve(f'{artifact_base}/model/show.json'),{'fixture':'show'})
            layout.resolve(f'{artifact_base}/model/ollama-modelfile.txt').write_bytes(b'FROM fixture\n')
            return evidence

        body={'message':{'content':'{"action":"KEEP","reason_code":"SUPPORTED"}'},'done_reason':'stop'}
        transport=mock.Mock(return_value=HttpResult(200,body,'a'*64))
        arguments['transport']=transport
        return layout,arguments,environment,verify_model,transport

    def _interrupt_final_manifest(self, arguments):
        def interrupted(path, value):
            if path.name == 'verifier-simple-live.json':
                raise RuntimeError('fixture final-manifest interruption')
            atomic_write_json(path,value)
        with mock.patch('utils.verifier.verifier.atomic_write_json',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'final-manifest interruption'):
                run_verifier(**arguments)

    def test_verifier_finalization_preserves_warmup_responses_verdicts_and_environment(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout,args,environment,verify_model,transport=self._fixture(Path(temporary))
            with (mock.patch('utils.verifier.verifier._environment_manifest',side_effect=lambda *_:copy.deepcopy(environment)),
                  mock.patch('utils.verifier.verifier._verify_live_model',side_effect=verify_model)):
                self._interrupt_final_manifest(args)
                paths=[layout.resolve(f'verifier/simple/{name}') for name in
                       ('warmup-request.jsonl','warmup-response.jsonl','responses.jsonl','verdicts.jsonl','environment-manifest.json')]
                saved={path:path.read_bytes() for path in paths}
                transport.reset_mock()
                result=run_verifier(**args)
            transport.assert_not_called()
            self.assertEqual(result['status'],'completed')
            self.assertEqual({path:path.read_bytes() for path in paths},saved)
            self.assertFalse(json.loads(paths[2].read_text())['cache_hit'])

    def test_verifier_corrupt_partial_artifacts_fail_before_any_model_query(self):
        for mutation in ('journal','missing_journal','verdict','environment','model_proof','response'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
                layout,args,environment,verify_model,transport=self._fixture(Path(temporary))
                with (mock.patch('utils.verifier.verifier._environment_manifest',side_effect=lambda *_:copy.deepcopy(environment)),
                      mock.patch('utils.verifier.verifier._verify_live_model',side_effect=verify_model) as discovery):
                    self._interrupt_final_manifest(args)
                    if mutation in {'journal','missing_journal'}:
                        path=next(layout.resolve('verifier/simple/journal/test').glob('*.json'))
                        if mutation == 'missing_journal':
                            path.unlink()
                        else:
                            saved=load_json(path)
                            saved['attempts'][0]['elapsed_seconds'] += 1
                            atomic_write_json(path,saved)
                    elif mutation == 'verdict':
                        path=layout.resolve('verifier/simple/verdicts.jsonl')
                        verdict=json.loads(path.read_text())
                        verdict['action']='DROP'
                        atomic_write_jsonl(path,[verdict])
                    elif mutation == 'environment':
                        path=layout.resolve('verifier/simple/environment-manifest.json')
                        saved=load_json(path)
                        saved['fixture']='changed'
                        atomic_write_json(path,saved)
                    elif mutation == 'model_proof':
                        layout.resolve('verifier/simple/model/tags.json').unlink()
                    else:
                        path=layout.resolve('verifier/simple/responses.jsonl')
                        saved=json.loads(path.read_text())
                        saved['attempts'][0]['elapsed_seconds'] += 1
                        atomic_write_jsonl(path,[saved])
                    transport.reset_mock();discovery.reset_mock()
                    with self.assertRaises(DataContractError):
                        run_verifier(**args)
                    transport.assert_not_called();discovery.assert_not_called()
