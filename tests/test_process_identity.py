"""Process-start hash evidence remains honest across isolation and mutation."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from utils.common import process_identity as identity


ROOT = Path(__file__).resolve().parents[1]


class ProcessIdentityTests(unittest.TestCase):
    def _subprocess(self, seed, *, flags=(), before=None, after=None):
        environment = dict(os.environ)
        environment.pop("PYTHONHASHSEED", None)
        if seed is not None:
            environment["PYTHONHASHSEED"] = seed
        program = f"""import json,os,random,sys
sys.path.insert(0,{str(ROOT)!r})
original_hash=hash('unchanged scientific hash behavior')
original_random=random.getstate()
before={before!r}
if before is not None: os.environ['PYTHONHASHSEED']=before
from utils.common.process_identity import capture_process_start,process_identity
captured=capture_process_start('fixture.py')
after={after!r}
if after is not None: os.environ['PYTHONHASHSEED']=after
assert process_identity()==captured
assert hash('unchanged scientific hash behavior')==original_hash
assert random.getstate()==original_random
print(json.dumps(captured))
"""
        result = subprocess.run([sys.executable, *flags, "-B", "-c", program],
            cwd=ROOT, env=environment, check=True, capture_output=True,
            text=True, encoding="utf-8")
        return json.loads(result.stdout)

    def test_subprocess_startup_settings_and_interpreter_isolation(self):
        for seed, kind in ((None, "unset"), ("random", "random"), ("0", "zero"),
                           ("42", "numeric"), ("4294967295", "numeric")):
            for flags in ((), ("-I",), ("-E",)):
                with self.subTest(seed=seed, flags=flags):
                    record = self._subprocess(seed, flags=flags, after="99")
                    observed = record["python_hash_seed"]
                    self.assertEqual(observed["environment_at_capture"],
                                     {"kind": kind, "value": seed})
                    self.assertEqual(record["capture_scope"], "entrypoint")
                    self.assertEqual(record["entrypoint"], "fixture.py")
                    self.assertTrue(record["process_id"].startswith(f"{record['pid']}-"))
                    if sys.platform == "linux":
                        self.assertEqual(observed["startup_setting"]["kind"], kind)
                        self.assertEqual(observed["startup_setting"]["value"], seed)
                    else:
                        self.assertEqual(observed["startup_setting"]["kind"], "unavailable")
                    effective = observed["effective_setting"]
                    if flags:
                        self.assertEqual(effective["kind"], "random")
                        self.assertTrue(record["python"]["ignore_environment"])
                    elif seed == "0":
                        self.assertEqual(effective["kind"], "zero")
                    elif sys.platform == "linux":
                        self.assertEqual(effective["kind"],
                                         "numeric" if kind == "numeric" else "random")
                    else:
                        self.assertEqual(effective["kind"], "unavailable")

    def test_environment_changed_before_capture_is_not_relabelled_startup(self):
        record = self._subprocess("42", before="456", after="789")
        seed = record["python_hash_seed"]
        self.assertEqual(seed["environment_at_capture"], {"kind": "numeric", "value": "456"})
        if sys.platform == "linux":
            self.assertEqual(seed["startup_setting"]["value"], "42")
            self.assertEqual(seed["effective_setting"]["value"], "42")
        else:
            self.assertEqual(seed["startup_setting"]["kind"], "unavailable")
            self.assertEqual(seed["effective_setting"]["kind"], "unavailable")

    def test_linux_initial_environment_only_extracts_seed(self):
        fixtures = ((b"PRIVATE=do-not-retain\0", "unset", None),
                    (b"PRIVATE=do-not-retain\0PYTHONHASHSEED=random\0", "random", "random"),
                    (b"PYTHONHASHSEED=0\0", "zero", "0"),
                    (b"PYTHONHASHSEED=123\0", "numeric", "123"))
        with mock.patch.object(identity.sys, "platform", "linux"):
            for raw, kind, value in fixtures:
                with self.subTest(kind=kind), mock.patch.object(identity.Path, "read_bytes", return_value=raw):
                    result = identity._initial_setting()
                    self.assertEqual(result, {"kind": kind, "value": value,
                        "evidence": "linux_proc_initial_exec_environment"})
                    self.assertNotIn("do-not-retain", json.dumps(result))

    def test_initial_environment_unavailable_or_ambiguous_stays_unavailable(self):
        with mock.patch.object(identity.sys, "platform", "linux"):
            for raw in (b"PYTHONHASHSEED=1\0PYTHONHASHSEED=2\0",
                        b"PYTHONHASHSEED=\xff\0", b"PYTHONHASHSEED=123"):
                with self.subTest(raw=raw), mock.patch.object(identity.Path, "read_bytes", return_value=raw):
                    result = identity._initial_setting()
                    self.assertEqual(result["kind"], "unavailable")
                    self.assertIsNone(result["value"])
            with mock.patch.object(identity.Path, "read_bytes", side_effect=PermissionError):
                self.assertEqual(identity._initial_setting()["kind"], "unavailable")
        with mock.patch.object(identity.sys, "platform", "win32"):
            with mock.patch.object(identity.Path, "read_bytes") as read:
                self.assertEqual(identity._initial_setting()["kind"], "unavailable")
            read.assert_not_called()

    def test_capture_is_frozen_and_forked_pid_has_its_own_identity(self):
        with mock.patch.object(identity, "_IDENTITY", None), \
             mock.patch.object(identity.os, "getpid", return_value=101) as pid, \
             mock.patch.object(identity.time, "time_ns", side_effect=(1111, 2222)):
            with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "random"}):
                first = identity.capture_process_start("rag.py")
            first["python_hash_seed"]["environment_at_capture"]["value"] = "modified-copy"
            second = identity.capture_process_start("different.py")
            self.assertEqual(second["entrypoint"], "rag.py")
            self.assertEqual(second["python_hash_seed"]["environment_at_capture"]["value"], "random")
            self.assertEqual(second["process_id"], "101-1111")
            pid.return_value = 202
            child = identity.process_identity()
            self.assertEqual(child["process_id"], "202-2222")
            self.assertEqual(child["capture_scope"], "late_import")

    def test_late_access_never_claims_an_entrypoint_capture(self):
        with mock.patch.object(identity, "_IDENTITY", None):
            record = identity.process_identity()
            self.assertEqual(record["capture_scope"], "late_import")
            self.assertIsNone(record["entrypoint"])
            self.assertIn("not_original_startup_proof", record["python_hash_seed"]["environment_evidence"])


if __name__ == "__main__":
    unittest.main()
