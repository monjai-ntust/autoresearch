import json
import tempfile
import unittest
from pathlib import Path

from graph_rag_eval.identity import (
    fingerprint,
    identity_namespace,
    tree_sha256,
    verify_cache,
)


ROOT = Path(__file__).resolve().parents[1]


class GraphRagIdentityTests(unittest.TestCase):
    def test_canonical_fingerprint_is_order_stable_and_content_sensitive(self):
        left = fingerprint("test", {"b": 2, "a": 1})
        right = fingerprint("test", {"a": 1, "b": 2})
        self.assertEqual(left, right)
        self.assertNotEqual(left, fingerprint("test", {"a": 1, "b": 3}))

    def test_dataset_identity_prevents_same_local_id_collisions(self):
        left = identity_namespace("same", "s1", "a", "1", "0" * 64, {"x": 1})
        right = identity_namespace("same", "s1", "a", "1", "1" * 64, {"x": 1})
        self.assertNotEqual(left, right)

    def test_cache_hit_requires_manifest_and_verified_content_tree(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            cache = Path(temporary)
            content = cache / "content"
            content.mkdir()
            (content / "value.txt").write_text("fixed", encoding="utf-8")
            expected = fingerprint("cache", "fixed")
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "fingerprint": expected,
                        "content_tree_sha256": tree_sha256(content),
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(verify_cache(cache, expected))
            (content / "value.txt").write_text("tampered", encoding="utf-8")
            self.assertFalse(verify_cache(cache, expected))


if __name__ == "__main__":
    unittest.main()
