"""Configuration-driven component loading without core dataset branches."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class AdapterRegistryError(ValueError):
    pass


def load_object(import_path: str, options: dict[str, Any] | None = None):
    """Instantiate `module:Class` with keyword options taken from configuration."""

    if not isinstance(import_path, str) or ":" not in import_path:
        raise AdapterRegistryError("import path must be 'module:Class'")
    module_name, object_name = import_path.split(":", 1)
    module = import_module(module_name)
    factory = getattr(module, object_name, None)
    if factory is None:
        raise AdapterRegistryError(f"object not found: {import_path}")
    return factory(**dict(options or {}))


def require_members(instance: Any, members: tuple[str, ...], label: str):
    missing = [name for name in members if not hasattr(instance, name)]
    if missing:
        raise AdapterRegistryError(f"{label} is missing protocol members: {missing}")
    return instance


def load_adapter(import_path: str, options: dict[str, Any] | None = None):
    return require_members(
        load_object(import_path, options),
        ("adapter_id", "adapter_version", "validate", "load", "generated_graph"),
        "adapter",
    )


def load_generator(import_path: str, options: dict[str, Any] | None = None):
    return require_members(
        load_object(import_path, options),
        ("generator_id", "build_prompt", "generate"),
        "generator",
    )
