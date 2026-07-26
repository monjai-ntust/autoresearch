"""Output containment and deterministic artifact writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import re

from graph_rag_eval.contracts import canonical_data
from graph_rag_eval.identity import file_sha256


class OutputContainmentError(ValueError):
    pass


class SchemaValidationError(ValueError):
    pass


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset used by the tracked Graph RAG contracts."""

    value = canonical_data(value)
    if "anyOf" in schema:
        errors = []
        for branch in schema["anyOf"]:
            try:
                validate_schema(value, branch, path)
                break
            except SchemaValidationError as error:
                errors.append(str(error))
        else:
            raise SchemaValidationError(f"{path} does not satisfy anyOf: {errors}")
    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        checks = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }
        if not any(checks[item](value) for item in expected_types):
            raise SchemaValidationError(f"{path} has invalid type; expected {expected_types}")
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path} does not equal required constant")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} is not in the allowed enumeration")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{path} is shorter than minLength")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise SchemaValidationError(f"{path} does not match required pattern")
    if isinstance(value, (int, float)) and "minimum" in schema:
        if value < schema["minimum"]:
            raise SchemaValidationError(f"{path} is below minimum")
    if isinstance(value, list):
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                raise SchemaValidationError(f"{path} array items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        missing = sorted(required - set(value))
        if missing:
            raise SchemaValidationError(f"{path} is missing required keys: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path} has unknown keys: {extras}")
        for key, child in properties.items():
            if key in value:
                validate_schema(value[key], child, f"{path}.{key}")


def validate_run_id(run_id: str) -> str:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    return run_id


def confined_path(source_root: Path, run_id: str) -> Path:
    source_root = source_root.resolve()
    output_root = (source_root / "output").resolve()
    candidate = (output_root / validate_run_id(run_id) / "graph-rag").resolve()
    if output_root not in candidate.parents:
        raise OutputContainmentError("run output escapes source output root")
    return candidate


def resolve_within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root.resolve() != candidate and root.resolve() not in candidate.parents:
        raise OutputContainmentError(f"artifact path escapes run root: {relative}")
    return candidate


def write_json(root: Path, relative: str, value: Any) -> dict[str, Any]:
    path = resolve_within(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def write_jsonl(root: Path, relative: str, values: Iterable[Any]) -> dict[str, Any]:
    path = resolve_within(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in values
    ]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8", newline="\n")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "records": len(lines),
    }


def write_text(root: Path, relative: str, text: str) -> dict[str, Any]:
    path = resolve_within(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
