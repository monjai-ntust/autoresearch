"""Automatic run-report products using small, explicitly synthetic evidence."""
from pathlib import Path
import tempfile
import unittest

from utils.common.artifact_io import atomic_write_json, sha256_file
from utils.common.paths import RunLayout
from utils.test.fixtures import load_pipeline_config
from utils.evaluation.reporting import report_canonical

ROOT=Path(__file__).resolve().parents[1]


class AutomaticReportingTests(unittest.TestCase):
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
            atomic_write_json(layout.resolve(metrics),{'conditions':{'VER-RAW':{'per_seed':{'42':{
                'end_to_end_strict_triple':{'counts':{'tp':1,'fp':1,'fn':1},'f1':{'value':.5}}},
                '43':{'end_to_end_strict_triple':{'counts':{'tp':1,'fp':1,'fn':1},'f1':{'value':.5}}}}}},'statistics':{'status':'fixture-only'}})
            atomic_write_json(layout.resolve('manifests/score-manifest.json'),{'run_id':layout.run_id,
                'inputs':{},'outputs':{metrics:sha256_file(layout.resolve(metrics))}})
            result=report_canonical(layout,config)
            self.assertEqual(result['observations']['training']['recorded_wall_hours'],8.)
            self.assertEqual(result['observations']['descriptive_best_seed']['VER-RAW']['seed'],42)
            self.assertEqual(result['observations']['paired_statistics']['status'],'fixture-only')
            self.assertEqual(result['observations']['descriptive_best_seed']['VER-RAW']['tied_seeds'],[42,43])
            self.assertEqual(result['observations']['emitted_unique_triples']['VER-RAW']['42'],2)
            self.assertEqual(result['observations']['corpus']['split_distributions']['test']['sentences'],1)
            raw=layout.resolve('metrics/paper-report/paper-statistics.json').read_bytes()
            report_canonical(layout,config)
            self.assertEqual(layout.resolve('metrics/paper-report/paper-statistics.json').read_bytes(),raw)
