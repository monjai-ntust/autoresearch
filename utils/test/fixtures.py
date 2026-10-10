"""Explicit fake model identity for source-grounded offline contract tests."""
from utils.common.config import load_pipeline_config as _load_config


def load_pipeline_config(*args, **kwargs):
    config = _load_config(*args, **kwargs)
    config.value['verifier']['registry_manifest_sha256'] = 'a' * 64
    config.value['verifier']['model_blob_sha256'] = 'b' * 64
    return config
