"""One authenticated numerical training environment per physical run."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from utils.common.artifact_io import DataContractError, load_json, sha256_file
from utils.common.execution import bind_json


_configured_torch = None


def configure_numerics():
    """Establish the fresh-run numerical policy before CUDA initialization.

    This is a prospective execution policy, not a claim that archived runs used
    deterministic algorithms. Unsupported deterministic operations raise their
    original Torch error; no warning-only mode or fallback is enabled.
    """
    import torch
    global _configured_torch
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace not in (None, ':4096:8'):
        raise DataContractError(
            'strict encoder determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8; '
            'remove the incompatible external setting before starting the worker')
    for name in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        if os.environ.get(name) not in (None, ''):
            raise DataContractError(
                f'{name} overrides the recorded encoder precision policy; '
                'remove it before starting the worker')
    if torch.cuda.is_initialized():
        if _configured_torch is not torch:
            raise DataContractError('configure encoder numerical policy before CUDA initialization')
        if (workspace != ':4096:8'
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.backends.cudnn.deterministic
                or torch.backends.cudnn.benchmark
                or torch.get_float32_matmul_precision() != 'highest'
                or not torch.backends.cudnn.allow_tf32):
            raise DataContractError('encoder numerical policy changed after CUDA initialization')
        return
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Preserve the effective Torch 2.9 defaults rather than disabling TF32
    # everywhere: FP32 matmul is IEEE; cuDNN's TF32 policy remains enabled.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = True
    _configured_torch = torch


def continue_training(loader, later_epochs):
    """Resume a partial epoch without an extra DataLoader CPU RNG draw.

    A zero-worker DataLoader draws its otherwise unused base seed when an
    iterator is constructed. A fresh epoch must retain that original draw. A
    partial resumed epoch already drew it before its checkpoint, so construct
    only that replacement iterator with the saved CPU RNG restored afterward.
    The sampler's private generator, order and cursor are left untouched.
    """
    import torch
    sampler = loader.sampler
    if 0 < sampler.cursor < len(sampler.order):
        if loader.num_workers != 0:
            raise DataContractError('exact partial-epoch continuation requires num_workers=0')
        state = torch.get_rng_state()
        try:
            iterator = iter(loader)
        finally:
            torch.set_rng_state(state)
        yield from iterator
    yield from later_epochs


def _driver_version():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return values or None


def record_model_runtime(prepared_dir, model):
    """Bind the loaded attention implementation and parameter dtypes once/run."""
    backbone = getattr(getattr(model, 'backbone', None), 'bert', None)
    config = getattr(backbone, 'config', None)
    value = {
        'backbone_class': (
            f'{type(backbone).__module__}.{type(backbone).__qualname__}'
            if backbone is not None else None),
        'attention_implementation': getattr(config, '_attn_implementation', None),
        'parameter_dtypes': sorted({str(parameter.dtype) for parameter in model.parameters()}),
    }
    path = Path(prepared_dir).resolve().parent / 'manifests/encoder-model-runtime.json'
    bind_json(path, value)


def restart_identity(prepared_dir):
    """Bind saved state to the actual run, source, inputs, recipe, and runtime."""
    from utils.common.paths import RunLayout
    root = Path(prepared_dir).absolute().parent
    layout = RunLayout(root.parent.parent, root.name)
    layout.require_existing()
    paths = [
        'manifests/00-checkout-manifest.json',
        'manifests/02-input-acquisition-manifest.json',
        'manifests/huggingface-model-cache.json',
        'manifests/03-data-preparation-manifest.json',
        'data-prepared/split-manifest.json',
        'data-prepared/train.jsonl',
        'data-prepared/development.jsonl',
        'manifests/encoder-runtime.json',
        'manifests/encoder-model-runtime.json',
    ]
    for relative in ('manifests/encoder-comparison-selection.json',
                     'manifests/comparison-matrix-link.json'):
        if layout.resolve(relative).exists():
            paths.append(relative)
    return {'run_id': layout.run_id, 'inputs': {
        relative: sha256_file(layout.resolve(relative, must_exist=True))
        for relative in paths}}


def record_runtime(prepared_dir):
    import torch
    configure_numerics()
    root = Path(prepared_dir).resolve().parent
    relative = 'manifests/encoder-runtime.json'
    path = root/relative
    environment = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'machine': platform.machine(), 'cpu_count': os.cpu_count(),
        'packages': {name: version(name) for name in (
            'torch', 'transformers', 'tokenizers', 'numpy', 'safetensors', 'sentencepiece')},
        'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        'cuda_device_memory_bytes': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory if torch.cuda.is_available() else None,
        'cuda_compute_capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        'nvidia_driver_versions': _driver_version() if torch.cuda.is_available() else None,
        'numerical_environment_variables': {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'TORCH_CUDNN_V8_API_DISABLED',
            'TORCH_BLAS_PREFER_CUBLASLT')},
        'total_memory_bytes': None}
    try:
        environment['total_memory_bytes'] = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        pass
    bind_json(path, environment)
    return {'path': relative, 'sha256': sha256_file(path)}


def training_environment(run_root, summary):
    ref = summary.get('runtime_environment')
    if ref is None:
        return summary.get('environment')  # Read-only archived upstream compatibility.
    from utils.common.paths import RunLayout
    root = Path(run_root).resolve()
    layout = RunLayout(root.parent.parent, root.name)
    path = layout.resolve(ref['path'], must_exist=True)
    if sha256_file(path) != ref['sha256']:
        raise DataContractError('training numerical environment changed')
    return load_json(path)
