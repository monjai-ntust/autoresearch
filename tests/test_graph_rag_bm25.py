import hashlib
import unittest

from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import Chunk, Provenance, QueryView
from graph_rag_eval.retrieval.bm25 import BM25Retriever
from graph_rag_eval.retrieval.dense import FrozenTransformerDenseRetriever


def chunk(identifier, text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    provenance = Provenance(
        "bm25-test", "fixture", "source", "1", digest, document_id=f"d-{identifier}"
    )
    return Chunk(
        "bm25-test", identifier, f"d-{identifier}", text, 0, len(text), 0,
        len(text.split()), "fixture", digest, provenance
    )


def query(text):
    return QueryView("bm25-test", "q", text, "question_only")


class GraphRagBM25Tests(unittest.TestCase):
    budget = Budget(10, 1000)

    def test_term_frequency_saturates_and_diverges_from_unique_overlap(self):
        retriever = BM25Retriever(
            (chunk("a", "rare rare rare filler"), chunk("b", "rare filler"))
        )
        result = retriever.retrieve(query("rare"), self.budget)
        scores = {item.evidence_id: item.score for item in result.items}
        self.assertGreater(scores["a"], scores["b"])
        self.assertLess(scores["a"] / scores["b"], 3.0)

    def test_document_length_normalization_prefers_shorter_equal_tf_document(self):
        retriever = BM25Retriever(
            (
                chunk("a", "anchor filler filler filler filler filler"),
                chunk("b", "anchor filler"),
            )
        )
        result = retriever.retrieve(query("anchor"), self.budget)
        self.assertEqual(result.items[0].evidence_id, "b")

    def test_idf_and_query_terms_affect_ranking(self):
        retriever = BM25Retriever(
            (
                chunk("a", "common rare"),
                chunk("b", "common common"),
                chunk("c", "common other"),
            )
        )
        rare = retriever.retrieve(query("rare"), self.budget)
        common = retriever.retrieve(query("common"), self.budget)
        self.assertEqual(rare.items[0].evidence_id, "a")
        self.assertNotEqual(rare.items[0].score, common.items[0].score)

    def test_exact_ties_use_stable_evidence_id(self):
        retriever = BM25Retriever((chunk("z", "same words"), chunk("a", "same words")))
        result = retriever.retrieve(query("same words"), self.budget)
        self.assertEqual([item.evidence_id for item in result.items], ["a", "z"])

    def test_frozen_dense_interface_requires_immutable_revision_without_loading(self):
        rows = (chunk("a", "one"),)
        with self.assertRaisesRegex(ValueError, "immutable"):
            FrozenTransformerDenseRetriever(
                rows, model_id="example/model", revision="main", cache_dir="output/cache"
            )
        retriever = FrozenTransformerDenseRetriever(
            rows,
            model_id="example/model",
            revision="0123456789abcdef",
            cache_dir="output/cache",
        )
        self.assertEqual(len(retriever.index_fingerprint), 64)


if __name__ == "__main__":
    unittest.main()
