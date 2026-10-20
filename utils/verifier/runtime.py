"""Actual Ollama generation identity shared by one fresh run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from utils.common.artifact_io import (
    DataContractError, atomic_write_json, load_json, sha256_file,
)


RUNTIME_PATH = "runtime/ollama.json"


def _comparison_identity(identity: Any) -> Any:
    """Ignore parameter-name display order, retaining each name's ordered values."""
    if not isinstance(identity, dict):
        return identity
    defaults = identity.get("generation_defaults")
    if not isinstance(defaults, dict) or not isinstance(defaults.get("parameters"), str):
        return identity
    groups: dict[str, list[str]] = {}
    for line in defaults["parameters"].splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[a-z][a-z0-9_]*", parts[0]) is None:
            # Unknown display syntax must still compare exactly.
            return identity
        name, value = parts
        groups.setdefault(name, []).append(value)
    return {**identity, "generation_defaults": {**defaults, "parameters": tuple(
        (name, tuple(values)) for name, values in sorted(groups.items())
    )}}


def runtime_matches(recorded: Any, observed: Any) -> bool:
    """Compare runtime settings without changing raw hash-bound observations."""
    return _comparison_identity(recorded) == _comparison_identity(observed)


def runtime_identity(origin: str, version: dict[str, Any],
                     show: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Bind server version and actual model/request defaults without inventory."""
    server_version = version.get("version")
    if not isinstance(server_version, str) or not server_version.strip():
        raise DataContractError("Ollama /api/version lacks the actual server version")
    fields = {name: show[name] for name in ("template", "parameters", "system") if name in show}
    info = show.get("model_info")
    if isinstance(info, dict):
        architecture = info.get("general.architecture")
        fields["model_info"] = {key: value for key, value in info.items()
            if key == "general.architecture"
            or (isinstance(architecture, str) and key.startswith(architecture + "."))}
    return {
        "origin": origin,
        "server_version": server_version,
        "model": {key: model[key] for key in ("name", "tag_digest", "blob_sha256", "details")},
        "generation_defaults": fields,
    }


def bind_runtime(layout, identity: dict[str, Any]) -> dict[str, str]:
    """Create once and reject later server/default drift under the same run ID."""
    path = layout.resolve(RUNTIME_PATH)
    if path != layout.run_root / RUNTIME_PATH or path.is_symlink():
        raise DataContractError("same-run Ollama runtime must be one physical artifact")
    if path.exists():
        document = load_json(path)
        if (not isinstance(document, dict) or document.get("run_id") != layout.run_id
                or not runtime_matches(document.get("identity"), identity)):
            raise DataContractError("same-run Ollama server/model runtime drift")
    else:
        atomic_write_json(path, {"run_id": layout.run_id, "identity": identity})
    return {"path": RUNTIME_PATH, "sha256": sha256_file(path)}


def load_runtime(run_dir: Path, reference: Any) -> dict[str, Any]:
    """Load only the fixed same-run runtime artifact named by a hashed reference."""
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or reference.get("path") != RUNTIME_PATH):
        raise DataContractError("invalid same-run Ollama runtime reference")
    path = run_dir / RUNTIME_PATH
    if (not path.is_file() or path.is_symlink()
            or path.resolve() != path or sha256_file(path) != reference.get("sha256")):
        raise DataContractError("same-run Ollama runtime evidence changed or is missing")
    document = load_json(path)
    if (not isinstance(document, dict) or document.get("run_id") != run_dir.name
            or not isinstance(document.get("identity"), dict)):
        raise DataContractError("same-run Ollama runtime evidence is malformed or foreign")
    return document["identity"]
