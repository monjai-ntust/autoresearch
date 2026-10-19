"""Automatic run-report products using small, explicitly synthetic evidence."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.common.artifact_io import DataContractError, atomic_write_json, atomic_write_jsonl, load_json, sha256_file
from utils.common.paths import RunLayout
from utils.test.fixtures import load_pipeline_config
from utils.evaluation.reporting import Evidence, _invocation_cost, _write, report_canonical, report_comparison, report_matrix, report_rag, training_statistics

ROOT=Path(__file__).resolve().parents[1]


def invocation_fixture(layout, label, ordinal, *, seconds=None, status='completed'):
    path=layout.resolve(f'logs/invocations/{label}/{ordinal:04d}-start.json')
    atomic_write_json(path,{'run_id':layout.run_id,'label':label,'started_unix':1.,
                            'previous_charged_seconds':999999.,'command':['fixture']})
    if seconds is not None:
        atomic_write_json(path.with_name(path.name.replace('-start','-finish')),
                          {'start_sha256':sha256_file(path),'status':status,'elapsed_seconds':seconds})
    return path


class AutomaticReportingTests(unittest.TestCase):
    def test_comparison_report_rejects_semantically_foreign_authenticated_records(self):
        from tests.test_pipeline_core import _candidate_record, _span, _triple
        triple=_triple(_span(0,0,'Object','head'),'part-of',_span(1,1,'Object','tail'))
        for mutation in ('selection','producer_seed','candidate_seed','candidate_split','candidate_example','candidate_source'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
                layout=RunLayout(Path(temporary),'fixture-comparison-report')
                layout.create()
                selected={'run_id':layout.run_id,'fixture_arm':'a'}
                selection=SimpleNamespace(manifest=lambda run_id: selected,
                                          config=SimpleNamespace(value={'training_seeds':[42]}))
                atomic_write_json(layout.resolve('manifests/00-checkout-manifest.json'),{'run_id':layout.run_id})
                atomic_write_json(layout.resolve('manifests/encoder-comparison-selection.json'),
                                  {**selected,'fixture_arm':'b'} if mutation=='selection' else selected)
                split='data-prepared/split-manifest.json'
                atomic_write_json(layout.resolve(split),{'fixture':True})
                atomic_write_jsonl(layout.resolve('data-prepared/test-gold.jsonl'),[{
                    'protocol_id':'B04-PATH-A-1.3','split_id':'CODE-SPLIT-1:test',
                    'example_id':'example','source_document_id':'document','gold_triples':[triple],
                    'input_hashes':{'fixture':'0'*64}}])
                candidate=_candidate_record(43 if mutation=='candidate_seed' else 42,
                    'foreign' if mutation=='candidate_example' else 'example',
                    'foreign' if mutation=='candidate_source' else 'document',triple,.9)
                if mutation=='candidate_split': candidate['split_id']='CODE-SPLIT-1:development'
                relative='predictions/test/seed-42-candidates.jsonl'
                atomic_write_jsonl(layout.resolve(relative),[candidate])
                atomic_write_json(layout.resolve('manifests/model-generate-candidates-replay-seed-42-test.json'),{
                    'run_id':layout.run_id,'training_seed':43 if mutation=='producer_seed' else 42,
                    'status':'completed','checkpoint':{'split_manifest_sha256':sha256_file(layout.resolve(split))},
                    'candidates_output':relative,'output_hashes':{relative:sha256_file(layout.resolve(relative))}})
                with patch('utils.evaluation.reporting._corpus',return_value={}), self.assertRaises(DataContractError):
                    report_comparison(layout,selection)
                self.assertFalse(layout.resolve('reports/comparison').exists())

    def test_canonical_report_reads_authenticated_producers_and_keeps_coverage(self):
        config=load_pipeline_config(ROOT,'resources/configs/pipeline.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-reports')
            layout.create()
            atomic_write_json(layout.resolve('manifests/00-checkout-manifest.json'),{'run_id':layout.run_id,'source':{'commit':'a'*40}})
            cache='manifests/huggingface-model-cache.json'
            atomic_write_json(layout.resolve(cache),{'fixture':True})
            atomic_write_json(layout.resolve('data-prepared/data-manifest.json'),{
                'counts':{'sentences':1},'alignment':{'unresolved':0}})
            atomic_write_json(layout.resolve('data-prepared/distributions.json'),{'test':{'sentences':1}})
            from utils.preparation.materialization import _tree_inventory
            atomic_write_json(layout.resolve('manifests/03-data-preparation-manifest.json'),{
                'run_id':layout.run_id,'dataset_tree_sha256':_tree_inventory(layout.resolve('data-prepared'))[1]})
            for seed in config.value['training_seeds']:
                summary=f'checkpoints/seed-{seed}/training-summary.json'
                atomic_write_json(layout.resolve(summary),{'status':'completed','seed':seed,'elapsed_seconds':3600.,'resumed_from':None,'environment':{'fixture':True}})
                atomic_write_json(layout.resolve(f'checkpoints/seed-{seed}/checkpoint-manifest.json'),{
                    'model_cache_manifest':cache,'model_cache_manifest_sha256':sha256_file(layout.resolve(cache))})
                atomic_write_json(layout.resolve(f'manifests/model-train-live-seed-{seed}.json'),{
                    'run_id':layout.run_id,'outputs':{'training_summary':summary},
                    'output_hashes':{summary:sha256_file(layout.resolve(summary))}})
            metrics='metrics/metrics.json'
            atomic_write_json(layout.resolve(metrics),{'selected_confidence_threshold':.25,
                'observed_training_seeds':[42,43], 'conditions':{'VER-RAW':{'per_seed':{'42':{
                'end_to_end_strict_triple':{'counts':{'tp':1,'fp':1,'fn':1},'f1':{'value':.5}}},
                '43':{'end_to_end_strict_triple':{'counts':{'tp':1,'fp':1,'fn':1},'f1':{'value':.5}}}}}},'statistics':{'status':'fixture-only'}})
            candidates=[]
            outcomes=[]
            sentences=[]
            gold=['fixture-example',0,0,'Object','part-of',1,1,'Object']
            extra=['fixture-example',0,0,'Object','necessity',1,1,'Object']
            missing=['fixture-example',0,0,'Object','selection',1,1,'Object']
            for seed in (42,43):
                for ordinal,key in enumerate((gold,extra)):
                    candidate={'training_seed':seed,'candidate_id':f'{seed}-{ordinal}',
                               'triple_confidence':.8 if ordinal==0 else .2}
                    candidates.append(candidate)
                    for condition in ('VER-RAW','VER-CONFIDENCE','VER-SIMPLE','VER-CORRECTIVE'):
                        outcomes.append({**candidate,'example_id':'fixture-example',
                            'condition_id':condition,'final_emitted_key':key if condition=='VER-RAW' or ordinal==0 else None})
                sentences.append({'training_seed':seed,'example_id':'fixture-example',
                                  'condition_id':'VER-RAW','gold_strict_keys':[gold,missing]})
            sources={'predictions/test/candidates.jsonl':candidates,
                     'outcomes/candidate-outcomes.jsonl':outcomes,'outcomes/sentence-outcomes.jsonl':sentences}
            for relative,rows in sources.items():
                atomic_write_jsonl(layout.resolve(relative),rows)
            threshold='predictions/dev/threshold-selection.json'
            atomic_write_json(layout.resolve(threshold),{'selected_threshold':.25,'used_test_labels':False})
            atomic_write_json(layout.resolve('manifests/score-manifest.json'),{'run_id':layout.run_id,
                'inputs':{relative:sha256_file(layout.resolve(relative)) for relative in
                          ('predictions/test/candidates.jsonl',threshold)},
                'outputs':{relative:sha256_file(layout.resolve(relative)) for relative in
                           (metrics,'outcomes/candidate-outcomes.jsonl','outcomes/sentence-outcomes.jsonl')}})
            for mode in ('simple', 'corrective'):
                base=f'verifier/{mode}'
                atomic_write_jsonl(layout.resolve(f'{base}/responses.jsonl'), [{
                    'candidate_id':'fixture-candidate', 'cache_hit':False,
                    'attempts':[{'elapsed_seconds':2., 'transport_status':'response', 'http_status':200,
                                 'raw_response':{'prompt_eval_count':5}}]}])
                atomic_write_jsonl(layout.resolve(f'{base}/warmup-response.jsonl'), [{
                    'candidate_id':'fixture-warmup', 'cache_hit':False,
                    'attempts':[{'elapsed_seconds':.25, 'transport_status':'response', 'http_status':200,
                                 'raw_response':{'prompt_eval_count':2, 'eval_count':1}}]}])
                atomic_write_json(layout.resolve(f'{base}/environment-manifest.json'), {'fixture':True})
                atomic_write_json(layout.resolve(f'manifests/verifier-{mode}-live.json'), {
                    'run_id':layout.run_id, 'outputs':{f'{base}/{name}':sha256_file(layout.resolve(f'{base}/{name}'))
                        for name in ('responses.jsonl', 'warmup-response.jsonl', 'environment-manifest.json')}})
            invocation_fixture(layout,'canonical-train-seed-42',1,seconds=360.,status='failed')
            invocation_fixture(layout,'canonical-train-seed-42',2,seconds=3600.)
            result=report_canonical(layout,config)
            self.assertEqual(result['observations']['training']['recorded_wall_hours'],8.)
            invocations=result['observations']['training']['invocations']
            self.assertEqual(invocations['recorded_wall_hours'],1.1)
            self.assertEqual(invocations['seeds_with_measurement'],1)
            self.assertEqual(invocations['per_seed']['42']['failed_invocations'],1)
            self.assertIsNone(invocations['per_seed']['43']['recorded_seconds'])
            self.assertEqual(result['observations']['descriptive_best_seed']['VER-RAW']['seed'],42)
            self.assertEqual(result['observations']['paired_statistics']['status'],'fixture-only')
            self.assertEqual(result['observations']['descriptive_best_seed']['VER-RAW']['tied_seeds'],[42,43])
            self.assertEqual(result['observations']['emitted_unique_triples']['VER-RAW']['42'],2)
            self.assertEqual(result['observations']['corpus']['split_distributions']['test']['sentences'],1)
            cost=result['observations']['verifier']
            self.assertEqual(cost['verifier/simple/responses.jsonl']['recorded_request_seconds'],2.)
            self.assertEqual(cost['verifier/simple/warmup-response.jsonl']['recorded_request_seconds'],.25)
            self.assertIsNone(cost['verifier/simple/responses.jsonl']['tokens']['eval_count']['sum'])
            self.assertIn('publication_seed_coverage_complete',result['observations']['coverage'])
            self.assertEqual(result['observations']['threshold_selection']['selected_threshold'],.25)
            self.assertEqual(result['observations']['saved_record_supplement']['joint_selected_threshold']['pooled']['counts'],
                             {'tp':2,'fp':0,'fn':2})
            raw=layout.resolve('metrics/paper-report/paper-statistics.json').read_bytes()
            report_canonical(layout,config)
            self.assertEqual(layout.resolve('metrics/paper-report/paper-statistics.json').read_bytes(),raw)
            manifest_path=layout.resolve('manifests/verifier-simple-live.json')
            manifest=load_json(manifest_path)
            del manifest['outputs']['verifier/simple/responses.jsonl']
            atomic_write_json(manifest_path,manifest)
            with self.assertRaisesRegex(DataContractError,'lacks authenticated simple'):
                report_canonical(layout,config)

    def test_unavailable_training_measurements_are_not_counted_as_seconds(self):
        result=training_statistics([
            {'seed':42,'elapsed_seconds':3600.}, {'seed':43,'elapsed_seconds':True},
            {'seed':44,'elapsed_seconds':-1}, {'seed':45,'elapsed_seconds':float('nan')},
            {'seed':46},
        ],[42,43,44,45,46])
        self.assertEqual(result['recorded_wall_hours'],1.)
        self.assertEqual(result['seeds_with_measurement'],1)
        self.assertEqual([row['recorded_seconds'] for row in result['per_seed']],[3600.,None,None,None,None])

    def test_matrix_reports_original_failed_and_completed_invocation_costs_separately(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-matrix-report')
            layout.create()
            member=RunLayout(Path(temporary),'fixture-member')
            member.create()
            measurement=member.resolve('metrics/measured.json')
            atomic_write_json(measurement,{'fixture':True})
            reader=Evidence(member)
            reader.file('metrics/measured.json')
            pairing={}
            for relative in ('data-prepared/split-manifest.json','data-prepared/test-gold.jsonl'):
                atomic_write_json(member.resolve(relative),{'fixture':True})
                reader.file(relative)
                pairing[relative]=sha256_file(member.resolve(relative))
            _write(reader,'reports/comparison',{'arm':{'fixture':True},'pairing':pairing,
                'quality':{'mean_f1':.5}, 'training':{'recorded_wall_hours':1.,
                'per_seed':[{'seed':42,'recorded_seconds':3600.}], 'environments':{'42':{'fixture':True}}}})
            pipeline=ROOT/'resources/configs/pipeline.json'
            config=SimpleNamespace(value=load_json(ROOT/'resources/configs/encoder-comparison.json'),
                                   sha256='fixture-config',pipeline=SimpleNamespace(path=pipeline),
                                   arms={'bert-base-common':None},
                                   select_arm=lambda arm:SimpleNamespace(manifest=lambda run_id:{'fixture':True}))
            identity={'matrix_id':layout.run_id,'members':{'bert-base-common':member.run_id},
                      'arms':{'bert-base-common':{'fixture':True}},'config_sha256':config.sha256,
                      'pipeline_config_sha256':sha256_file(pipeline),
                      'training_seeds':config.value['training_seeds']}
            atomic_write_json(layout.resolve('manifests/comparison-matrix.json'),identity)
            invocation_fixture(layout,'comparison-smoke',1,seconds=90.)
            invocation_fixture(layout,'comparison-full',1,seconds=1800.,status='failed')
            invocation_fixture(layout,'comparison-full',2)
            invocation_fixture(layout,'comparison-full',3,seconds=3600.)
            report=report_matrix(layout,config,identity)
            costs=report['observations']['recorded_execution']
            self.assertEqual(costs['smoke']['recorded_seconds'],90.)
            full=costs['full']
            self.assertEqual(full['recorded_seconds'],5400.)
            self.assertEqual(full['recorded_wall_hours'],1.5)
            self.assertEqual(full['retained_starts'],3)
            self.assertEqual(full['invocations_with_measurement'],2)
            self.assertEqual(full['failed_invocations'],1)
            self.assertEqual(full['completed_invocations'],1)
            self.assertEqual(full['invocations_without_finish'],1)
            self.assertIsNone(full['per_invocation'][1]['recorded_seconds'])
            self.assertEqual(report['observations']['members']['bert-base-common']['training']['recorded_wall_hours'],1.)
            self.assertEqual(report['observations']['training_cost']['seconds_sum'],3600.)
            self.assertEqual(report['observations']['paired_statistics']['status'],'withheld_incomplete_declared_family')
            self.assertIn('logs/invocations/comparison-full/0003-finish.json',report['inputs'])
            hashes={path.name:sha256_file(path) for path in layout.resolve('reports/matrix').iterdir()}
            self.assertEqual(report_matrix(layout,config,identity),report)
            self.assertEqual({path.name:sha256_file(path) for path in layout.resolve('reports/matrix').iterdir()},hashes)
            wrong_identity={**identity,'arms':{'bert-base-common':{'fixture':'another-arm'}}}
            atomic_write_json(layout.resolve('manifests/comparison-matrix.json'),wrong_identity)
            with self.assertRaisesRegex(DataContractError,'arm identity differs'):
                report_matrix(layout,config,wrong_identity)
            atomic_write_json(layout.resolve('manifests/comparison-matrix.json'),identity)
            wrong_config=SimpleNamespace(**{**vars(config),'sha256':'another-config'})
            with self.assertRaisesRegex(DataContractError,'configuration'):
                report_matrix(layout,wrong_config,identity)
            self.assertEqual({path.name:sha256_file(path) for path in layout.resolve('reports/matrix').iterdir()},hashes)
            atomic_write_json(measurement,{'fixture':'tampered'})
            with self.assertRaisesRegex(DataContractError,'evidence changed'):
                report_matrix(layout,config,identity)

    def test_invocation_reader_rejects_foreign_broken_or_malformed_measurements(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-invocation-report')
            layout.create()
            for label, change, message in (
                ('foreign',{'run_id':'another-run'},'another run'),
                ('unbound',{'command':['changed']},'hash differs'),
            ):
                with self.subTest(label=label):
                    start=invocation_fixture(layout,label,1,seconds=1.)
                    atomic_write_json(start,{**load_json(start),**change})
                    with self.assertRaisesRegex(DataContractError,message):
                        _invocation_cost(Evidence(layout),label)
            for index, seconds in enumerate((True,-1,float('nan'))):
                label=f'invalid-{index}'
                with self.subTest(label=label):
                    start=invocation_fixture(layout,label,1,seconds=1.)
                    finish=start.with_name(start.name.replace('-start','-finish'))
                    # json permits NaN for this malformed-input rejection fixture only.
                    import json
                    finish.write_text(json.dumps({**load_json(finish),'elapsed_seconds':seconds}),encoding='utf-8')
                    with self.assertRaises(DataContractError):
                        _invocation_cost(Evidence(layout),label)
            invocation_fixture(layout,'gap',2,seconds=1.)
            with self.assertRaisesRegex(DataContractError,'not contiguous'):
                _invocation_cost(Evidence(layout),'gap')
            atomic_write_json(layout.resolve('logs/invocations/orphan/0001-finish.json'),{'elapsed_seconds':1.})
            with self.assertRaisesRegex(DataContractError,'no start'):
                _invocation_cost(Evidence(layout),'orphan')

    def test_changed_report_receipt_blocks_writes_before_partial_finalization(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-report-receipt')
            layout.create()
            receipt=layout.resolve('reports/manifest.json')
            atomic_write_json(receipt,{'run_id':'foreign-run','inputs':{},'outputs':{}})
            original=receipt.read_bytes()
            with self.assertRaisesRegex(DataContractError,'receipt differs'):
                _write(Evidence(layout),'reports',{'fixture':True})
            self.assertEqual(receipt.read_bytes(),original)
            self.assertEqual(sorted(p.name for p in receipt.parent.iterdir()),['manifest.json'])

    def test_changed_input_blocks_report_finalization(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-report-input')
            layout.create()
            path=layout.resolve('metrics/fixture.json')
            atomic_write_json(path,{'measured':1})
            evidence=Evidence(layout)
            evidence.file('metrics/fixture.json')
            atomic_write_json(path,{'measured':2})
            with self.assertRaisesRegex(DataContractError,'evidence changed'):
                _write(evidence,'reports',{'measured':1})
            self.assertFalse(layout.resolve('reports').exists())

    def test_rag_report_preserves_values_and_exposes_diagnostic_oracle_limits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout=RunLayout(Path(temporary),'fixture-rag-report')
            layout.create()
            child=layout.resolve('table2-code-all')
            atomic_write_json(child/'environment.json',{'run_id':layout.run_id,'fixture':True})
            from utils.rag.evaluator import MODES
            for condition in ('confidence','corrective','gold'):
                atomic_write_json(child/'rag-results'/f'{condition}.json',{
                    'metadata':{'n_questions':2,'time_seconds':1.25},
                    'correct_counts':{mode:1 for mode in MODES},'accuracy':{mode:.5 for mode in MODES},
                    'results':{mode:[{'q':'first','gold':'one','pred':'one','correct':True},
                                     {'q':'second','gold':'one','pred':'two','correct':False}] for mode in MODES}})
                atomic_write_json(child/'part-of-results'/f'{condition}.json',{
                    'results':{mode:[{'q':'first','gold':'one','pred':'one','correct':True}] for mode in MODES}})
            hashes={p.relative_to(child).as_posix():sha256_file(p) for p in child.rglob('*') if p.is_file()}
            atomic_write_json(child/'artifact-hashes.json',{'run_id':layout.run_id,'artifacts':hashes})
            summary={'questions':{'combined':2,'part_of':1}}
            document=report_rag(layout,summary,{},child)
            observations=document['observations']
            self.assertIn('oracle',observations['claim_limits']['gold_condition'])
            self.assertIn('diagnostic',observations['claim_limits']['result_status'])
            self.assertEqual(observations['conditions']['gold']['full']['llm_only'],{
                'questions':2,'hits':1,'answer_hit_rate':.5})
            self.assertEqual(observations['conditions']['gold']['part_of']['llm_only'],{
                'questions':1,'hits':1,'answer_hit_rate':1.})
            self.assertEqual(observations['conditions']['gold']['recorded_condition_seconds'],1.25)
            self.assertEqual(report_rag(layout,summary,{},child),document)
            self.assertEqual({name:sha256_file(child/name) for name in hashes},hashes)
            del hashes['environment.json']
            atomic_write_json(child/'artifact-hashes.json',{'run_id':layout.run_id,'artifacts':hashes})
            with self.assertRaisesRegex(DataContractError,'lacks authenticated'):
                report_rag(layout,summary,{},child)
