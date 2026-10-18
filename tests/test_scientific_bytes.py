"""Baseline scientific characterization needs no parent or old Git objects."""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class ScientificBytesTests(unittest.TestCase):
    def test_scientific_definitions_and_training_loops_are_unchanged(self):
        contract=json.loads((ROOT/'tests/fixtures/scientific-contract.json').read_text(encoding='utf-8'))
        for record in contract['relocated_scientific_definitions']:
            with self.subTest(path=record['path'],name=record['name']):
                tree=ast.parse((ROOT/record['path']).read_bytes())
                node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==record['name'])
                self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),record['ast_sha256'])
        for record in contract['training_loops']:
            tree=ast.parse((ROOT/record['path']).read_bytes())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==record['function'])
            actual=[hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in node.body if isinstance(n,ast.While)]
            self.assertTrue(actual)
            self.assertEqual(actual,record['while_sha256'])
    def test_worker_values_match_independently_parsed_baseline_options(self):
        from utils.common.config import load_pipeline_config
        from utils.encoder_comparison.config import load_comparison_config
        from utils.encoder.model import canonical_trainer_arguments
        from stages.encoder_comparison import trainer_arguments
        from utils.common.artifact_io import canonical_json_bytes
        class Layout:
            source_root = ROOT
            def resolve(self, path, must_exist=False): return ROOT/'output/option-fixture'/path
        def normalize(value):
            return {k: v.replace(str(ROOT/'output/option-fixture'), '<run>').replace('\\', '/')
                    if isinstance(v, str) else v for k, v in vars(value).items()}
        config = load_comparison_config(ROOT)
        actual = {'canonical': normalize(canonical_trainer_arguments(Layout(), load_pipeline_config(ROOT, 'resources/configs/pipeline.json'), 42)),
                  'comparison': {arm: normalize(trainer_arguments(Layout(), config.select_arm(arm), 42, mode='train')) for arm in config.arms}}
        expected = json.loads((ROOT/'tests/fixtures/worker-options.json').read_text(encoding='utf-8'))
        self.assertEqual(canonical_json_bytes(actual), canonical_json_bytes(expected))
    def test_unchanged_scientific_modules_and_prompt_schema_bytes(self):
        contract=json.loads((ROOT/'tests/fixtures/scientific-contract.json').read_text(encoding='utf-8'))
        for relative,expected in contract['files'].items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),expected)

    def test_exact_evaluator_graph_projection_and_statistics_bytes(self):
        # Fix only Python's hash seed for the characterized set iteration order.
        program='''import json,sys,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,'tests')
from test_artifact_contracts import _build_graphs
from utils.common.artifact_io import canonical_json_bytes
from utils.rag import evaluator
from utils.rag.graph import project_graph
from utils.evaluation.statistics import exact_wilcoxon_signed_rank,holm_adjust
gold=json.loads(Path('tests/fixtures/scientific-goldens.json').read_text(encoding='utf-8'))
prompts=[]
def call(url,model,prompt):
    prompts.append(prompt); return gold['questions'][0]['gold_answer']
with mock.patch.object(evaluator.time,'time',side_effect=[100.0,103.25]):
    result=evaluator.evaluate(gold['records'],gold['graph'],kg_identity='source-characterization',ollama_url='http://localhost:11434',ollama_model='qwen3:32b',llm_call=call)
graphs=_build_graphs('source-characterization')
observed={'questions':evaluator.generate_questions(gold['records']),'prompts':prompts,'result':result,'canonical_graphs':graphs,'projections':{k:project_graph(v) for k,v in graphs.items()},'statistics':{'wilcoxon':exact_wilcoxon_signed_rank([0.1,0.1,-0.1,0,0.2,0.3,-0.2,0.1]),'holm':holm_adjust({'first':0.02,'second':0.08,'third':0.01})}}
expected={k:gold[k] for k in observed}
expected['statistics']={k:gold['statistics'][k] for k in observed['statistics']}
check=unittest.TestCase()
for section in observed:
    check.assertEqual(canonical_json_bytes(observed[section]),canonical_json_bytes(expected[section]),msg=section)
'''
        subprocess.run([sys.executable,'-B','-c',program],cwd=ROOT,check=True,
                       env={**os.environ,'PYTHONHASHSEED':'0'})

    def test_paired_t_matches_baseline_with_bounded_libm_roundoff(self):
        from utils.common.artifact_io import canonical_json_bytes
        from utils.evaluation.statistics import paired_t_test
        expected=json.loads((ROOT/'tests/fixtures/scientific-goldens.json').read_text(encoding='utf-8'))['statistics']['t']
        actual=paired_t_test([0.1,0.1,-0.1,0,0.2,0.3,-0.2,0.1])
        # The frozen Windows result uses sqrt/lgamma/log/log1p/exp. Their last
        # bits vary with libm; keep a 32-ULP bound only for these three values.
        # Algorithm source bytes and every other field remain exact. Neither
        # the fixture nor production values are rounded or rewritten.
        for field in ('sample_standard_deviation','statistic','p_value'):
            with self.subTest(field=field):
                self.assertAlmostEqual(actual[field],expected[field],
                                       delta=32*math.ulp(expected[field]))
        exact_fields={k:v for k,v in expected.items()
                      if k not in ('sample_standard_deviation','statistic','p_value')}
        self.assertEqual(set(actual),set(expected))
        self.assertEqual(canonical_json_bytes({k:actual[k] for k in exact_fields}),
                         canonical_json_bytes(exact_fields))
