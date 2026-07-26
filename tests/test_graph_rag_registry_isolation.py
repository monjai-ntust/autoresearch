import ast
import unittest
from pathlib import Path

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.identity import identity_namespace
from graph_rag_eval.registry import load_adapter


ROOT = Path(__file__).resolve().parents[1]


class ThirdAdapter(SyntheticAdapter):
    adapter_id = "third-test-adapter"


class GraphRagRegistryIsolationTests(unittest.TestCase):
    def test_temporary_third_adapter_loads_without_core_change(self):
        adapter = load_adapter(
            "test_graph_rag_registry_isolation:ThirdAdapter",
            {"dataset_id": "third", "variant": "extension"},
        )
        bundle = adapter.load()
        self.assertEqual(bundle.descriptor.adapter_id, "third-test-adapter")
        self.assertEqual(bundle.descriptor.dataset_id, "third")

    def test_two_adapters_with_colliding_readable_ids_remain_content_isolated(self):
        first = SyntheticAdapter(dataset_id="same", variant="one").load()
        second = SyntheticAdapter(dataset_id="same", variant="two").load()
        first_id = identity_namespace(
            "same", first.descriptor.snapshot_id, first.descriptor.adapter_id,
            first.descriptor.adapter_version, first.descriptor.canonical_content_sha256, {}
        )
        second_id = identity_namespace(
            "same", second.descriptor.snapshot_id, second.descriptor.adapter_id,
            second.descriptor.adapter_version, second.descriptor.canonical_content_sha256, {}
        )
        self.assertNotEqual(first_id, second_id)

    def test_core_has_no_dataset_import_or_dataset_name_branch(self):
        adapter_root = ROOT / "graph_rag_eval" / "adapters"
        forbidden_names = {"code_accord", "scierc", "cuad"}
        for path in (ROOT / "graph_rag_eval").rglob("*.py"):
            if adapter_root == path.parent or adapter_root in path.parents:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            self.assertFalse(
                any(name == "graph_rag_eval.adapters" or name.startswith("data.") for name in imports),
                path,
            )
            strings = {
                node.value.casefold()
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertFalse(forbidden_names & strings, path)


if __name__ == "__main__":
    unittest.main()
