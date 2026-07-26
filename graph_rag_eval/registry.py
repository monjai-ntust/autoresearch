"""Configuration-driven adapter loading without core dataset branches."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class AdapterRegistryError(ValueError):
    pass


def load_adapter(import_path: str, options: dict[str, Any] | None = None):
    if not isinstance(import_path, str) or ":" not in import_path:
        raise AdapterRegistryError("adapter import path must be 'module:Class'")
    module_name, object_name = import_path.split(":", 1)
    module = import_module(module_name)
    adapter_type = getattr(module, object_name, None)
    if adapter_type is None:
        raise AdapterRegistryError(f"adapter object not found: {import_path}")
    adapter = adapter_type(**dict(options or {}))
    required = ("adapter_id", "adapter_version", "validate", "load")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise AdapterRegistryError(f"adapter is missing protocol members: {missing}")
    return adapter
