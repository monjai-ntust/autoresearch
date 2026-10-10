"""One authenticated numerical training environment per physical run."""
from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from utils.common.artifact_io import DataContractError, load_json, sha256_file
from utils.common.execution import bind_json


def record_runtime(prepared_dir):
    import torch
    root = Path(prepared_dir).resolve().parent
    relative = 'manifests/encoder-runtime.json'
    path = root/relative
    environment = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'machine': platform.machine(), 'cpu_count': os.cpu_count(),
        'packages': {name: version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy')},
        'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        'cuda_device_memory_bytes': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory if torch.cuda.is_available() else None,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'numerical_environment_variables': {key: os.environ.get(key) for key in (
            'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'total_memory_bytes': None, 'peak_resident_memory_bytes': None}
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
