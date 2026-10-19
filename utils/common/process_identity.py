"""A minimal immutable token for the original producing process."""
from __future__ import annotations

import os
import time


_IDENTITY = None


def process_identity():
    """Return the same token per PID; a fork receives its own independent token."""
    global _IDENTITY
    pid = os.getpid()
    if _IDENTITY is None or _IDENTITY["pid"] != pid:
        observed_ns = time.time_ns()
        _IDENTITY = {"process_id": f"{pid}-{observed_ns}",
                     "pid": pid, "observed_unix_ns": observed_ns}
    return _IDENTITY.copy()
