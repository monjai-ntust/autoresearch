"""Scientific output regressions without source-hash or approval inventories."""
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScientificOutputTests(unittest.TestCase):
    def test_exact_evaluator_graph_projection_and_statistics_bytes(self):
        program = '''import json,sys,unittest
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
        subprocess.run([sys.executable, '-I', '-B', '-c',
                        f'import sys;sys.path.insert(0,{str(ROOT)!r});' + program],
                       cwd=ROOT, check=True)

    def test_paired_t_matches_expected_values_with_bounded_libm_roundoff(self):
        from utils.common.artifact_io import canonical_json_bytes
        from utils.evaluation.statistics import paired_t_test
        expected = json.loads((ROOT/'tests/fixtures/scientific-goldens.json').read_text(encoding='utf-8'))['statistics']['t']
        actual = paired_t_test([0.1, 0.1, -0.1, 0, 0.2, 0.3, -0.2, 0.1])
        # libm can differ in the final bits; only these floating fields use a bound.
        floating = ('sample_standard_deviation', 'statistic', 'p_value')
        for field in floating:
            with self.subTest(field=field):
                self.assertAlmostEqual(actual[field], expected[field],
                                       delta=32*math.ulp(expected[field]))
        exact = {key: value for key, value in expected.items() if key not in floating}
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(canonical_json_bytes({key: actual[key] for key in exact}),
                         canonical_json_bytes(exact))
