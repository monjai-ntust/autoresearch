"""Saved-record statistics, exact numerical characterization and refusal cases."""
import copy
import json
from pathlib import Path
import unittest

from utils.common.artifact_io import DataContractError
from utils.evaluation.comparison_statistics import comparison_statistics
from utils.evaluation.reporting import canonical_record_statistics, rag_record_statistics
from utils.rag.evaluator import MODES


class FrozenComparisonStatisticsTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).parent/'fixtures/paper-table1-statistics.json').read_text(encoding='utf-8'))

    def test_all_four_archived_contrasts_and_seed_intervals_match_exactly(self):
        fixture=self.fixture()
        result=comparison_statistics(fixture['members'],fixture['protocol'],fixture['seeds'])
        self.assertEqual(result['status'],'defined')
        for actual,expected in zip(result['contrasts'],fixture['expected_contrasts']):
            self.assertEqual({key:actual[key] for key in expected},expected)
        self.assertEqual({arm:row['seed_bootstrap_mean_f1_95ci'] for arm,row in result['per_arm'].items()},
                         fixture['expected_per_arm'])
        self.assertEqual(result['contrasts'][-1]['seeds_improved'],6)
        self.assertEqual(result['contrasts'][-1]['seeds_worsened'],2)
        self.assertIsNone(result['limits']['document_cluster_uncertainty'])
        self.assertEqual(comparison_statistics(fixture['members'],fixture['protocol'],fixture['seeds']),result)

    def test_incomplete_seed_or_arm_never_shrinks_the_planned_family(self):
        for mutation in ('arm','seed','foreign_seed','undefined','nan'):
            fixture=self.fixture()
            members=fixture['members']
            arm=next(iter(members))
            if mutation=='arm': del members[arm]
            elif mutation=='seed': del members[arm]['quality']['per_seed']['42']
            elif mutation=='foreign_seed': members[arm]['quality']['per_seed']['50']=members[arm]['quality']['per_seed'].pop('42')
            else: members[arm]['quality']['per_seed']['42']['f1']['value']=None if mutation=='undefined' else float('nan')
            with self.subTest(mutation=mutation):
                result=comparison_statistics(members,fixture['protocol'],fixture['seeds'])
                self.assertEqual(result['status'],'withheld_incomplete_declared_family')
                self.assertEqual(result['contrasts'],[])


def canonical_fixture():
    candidates=[]
    outcomes=[]
    sentences=[]
    good=['sentence-a',0,0,'Object','part-of',1,1,'Object']
    second=['sentence-a',0,0,'Object','necessity',1,1,'Quality']
    wrong=['sentence-a',0,0,'Object','selection',1,1,'Object']
    other=['sentence-a',0,0,'Object','equal',1,1,'Value']
    missed=['sentence-b',0,0,'Object','part-of',1,1,'Object']
    for seed in (42,43):
        for ordinal,(key,confidence) in enumerate(((good,.9),(wrong,.7),(other,.2))):
            candidate={'training_seed':seed,'candidate_id':f'{seed}-{ordinal}','triple_confidence':confidence}
            candidates.append(candidate)
            for condition in ('VER-RAW','VER-CONFIDENCE','VER-SIMPLE','VER-CORRECTIVE'):
                emitted=(key if condition=='VER-RAW' else
                         key if condition=='VER-CONFIDENCE' and confidence>=.25 else
                         key if condition=='VER-SIMPLE' and ordinal!=1 else
                         good if condition=='VER-CORRECTIVE' and ordinal==0 else
                         second if condition=='VER-CORRECTIVE' else None)
                outcomes.append({**candidate,'condition_id':condition,'example_id':'sentence-a','final_emitted_key':emitted})
        for example,keys in (('sentence-a',[good,second]),('sentence-b',[missed])):
            sentences.append({'training_seed':seed,'example_id':example,'condition_id':'VER-RAW','gold_strict_keys':keys})
    return candidates,outcomes,sentences


class SavedRecordStatisticsTests(unittest.TestCase):
    def test_joint_and_sensitivity_keep_gold_denominators_and_deduplicate_rewrites(self):
        inputs=canonical_fixture()
        unchanged=copy.deepcopy(inputs)
        result=canonical_record_statistics(*inputs,.25,[42,43])
        self.assertEqual(result['joint_selected_threshold']['pooled']['counts'],{'tp':2,'fp':0,'fn':4})
        self.assertEqual(result['threshold_sensitivity']['VER-RAW']['pooled']['counts'],{'tp':2,'fp':2,'fn':4})
        self.assertEqual(result['threshold_sensitivity']['VER-SIMPLE']['pooled']['counts'],{'tp':2,'fp':0,'fn':4})
        self.assertEqual(result['threshold_sensitivity']['VER-CORRECTIVE']['pooled']['counts'],{'tp':4,'fp':0,'fn':2})
        self.assertEqual(result['threshold_sensitivity']['VER-CORRECTIVE']['emitted_unique_triples'],4)
        self.assertEqual(inputs,unchanged)

    def test_missing_duplicate_or_foreign_original_records_fail(self):
        for mutation in ('missing_outcome','duplicate_outcome','foreign_seed','missing_gold','nan_threshold'):
            candidates,outcomes,sentences=canonical_fixture()
            threshold=.25
            if mutation=='missing_outcome': outcomes.pop()
            elif mutation=='duplicate_outcome': outcomes.append(outcomes[0])
            elif mutation=='foreign_seed': candidates[0]['training_seed']=50
            elif mutation=='missing_gold': sentences.pop()
            else: threshold=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(DataContractError):
                canonical_record_statistics(candidates,outcomes,sentences,threshold,[42,43])

    def rag_fixture(self):
        rows=[{'q':f'question-{i}','gold':'answer','pred':'answer' if hit else 'miss','correct':hit}
              for i,hit in enumerate((True,True,False,False))]
        hybrid=copy.deepcopy(rows)
        for row,hit in zip(hybrid,(True,False,True,False)):
            row.update(pred='answer' if hit else 'miss',correct=hit)
        return {'metadata':{'n_questions':4},'results':{mode:copy.deepcopy(hybrid if mode=='hybrid' else rows) for mode in MODES},
                'correct_counts':{mode:2 for mode in MODES},'accuracy':{mode:.5 for mode in MODES}}

    def test_rag_transitions_are_paired_by_original_question_and_reference(self):
        result=rag_record_statistics(self.rag_fixture())
        transitions=result['transitions_1hop_to_hybrid']
        self.assertEqual({key:transitions[key] for key in ('lost_hits','gained_hits','net_hit_change','both_hit','neither_hit')},
                         {'lost_hits':1,'gained_hits':1,'net_hit_change':0,'both_hit':1,'neither_hit':1})
        for mutation in ('target','missing_mode','incorrect_count','incorrect_flag'):
            fixture=self.rag_fixture()
            if mutation=='target': fixture['results']['text_retrieval'][0]['gold']='changed'
            elif mutation=='missing_mode': del fixture['results']['llm_only']
            elif mutation=='incorrect_count': fixture['correct_counts']['hybrid']=3
            else: fixture['results']['hybrid'][0]['correct']=False
            with self.subTest(mutation=mutation),self.assertRaises(DataContractError):
                rag_record_statistics(fixture)


if __name__=='__main__':
    unittest.main()
