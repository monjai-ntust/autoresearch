"""Minimal producer tokens remain frozen without probing execution settings."""
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from utils.common import process_identity as identity


ROOT = Path(__file__).resolve().parents[1]


class ProcessIdentityTests(unittest.TestCase):
    def test_token_has_only_original_process_identity_and_is_frozen(self):
        with mock.patch.object(identity, "_IDENTITY", None), \
             mock.patch.object(identity.os, "getpid", return_value=101), \
             mock.patch.object(identity.time, "time_ns", return_value=1111) as clock:
            first = identity.process_identity()
            self.assertEqual(first, {"process_id": "101-1111", "pid": 101, "observed_unix_ns": 1111})
            first["process_id"] = "mutated caller copy"
            first["pid"] = 0
            self.assertEqual(identity.process_identity(),
                             {"process_id": "101-1111", "pid": 101, "observed_unix_ns": 1111})
            clock.assert_called_once_with()

    def test_forked_pid_receives_its_own_original_token(self):
        with mock.patch.object(identity, "_IDENTITY", None), \
             mock.patch.object(identity.os, "getpid", return_value=101) as pid, \
             mock.patch.object(identity.time, "time_ns", side_effect=(1111, 2222)) as clock:
            parent = identity.process_identity()
            pid.return_value = 202
            child = identity.process_identity()
            self.assertEqual(parent["process_id"], "101-1111")
            self.assertEqual(child, {"process_id": "202-2222", "pid": 202, "observed_unix_ns": 2222})
            self.assertEqual(identity.process_identity(), child)
            self.assertEqual(clock.call_count, 2)

    def test_token_does_not_change_rng_hashes_or_read_environment(self):
        before_rng = random.getstate()
        before_hash = hash("unchanged scientific string")
        with mock.patch.object(identity, "_IDENTITY", None), \
             mock.patch.object(identity.os, "environ", None):
            token = identity.process_identity()
        self.assertEqual(set(token), {"process_id", "pid", "observed_unix_ns"})
        self.assertEqual(random.getstate(), before_rng)
        self.assertEqual(hash("unchanged scientific string"), before_hash)

    def test_independent_isolated_processes_need_no_startup_capture_or_output(self):
        program = f"""import json,sys
sys.path.insert(0,{str(ROOT)!r})
from utils.common.process_identity import process_identity
first=process_identity()
assert first==process_identity()
print(json.dumps(first))
"""
        results = []
        for _ in range(2):
            completed = subprocess.run([sys.executable, "-I", "-B", "-c", program],
                cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
            token = json.loads(completed.stdout)
            self.assertEqual(set(token), {"process_id", "pid", "observed_unix_ns"})
            self.assertEqual(token["process_id"], f"{token['pid']}-{token['observed_unix_ns']}")
            self.assertGreater(token["pid"], 0)
            self.assertGreater(token["observed_unix_ns"], 0)
            results.append(token)
        self.assertNotEqual(results[0]["process_id"], results[1]["process_id"])


if __name__ == "__main__":
    unittest.main()
