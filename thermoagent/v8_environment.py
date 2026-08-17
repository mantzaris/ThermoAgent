"""Capture the local V8 execution environment without credentials."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from typing import Any, Dict

from .v5_experiments import atomic_json, git_metadata


PACKAGES = (
    "numpy", "pandas", "scipy", "networkx", "matplotlib", "PyYAML", "pytest",
)


def capture_v8_environment(repository: Path, results_root: Path) -> Dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    value = {
        "execution_host": "local_cpu_after_existing_runpod_endpoint_refused_connection",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not_reported",
        "logical_cpu_count": os.cpu_count(),
        "packages": versions,
        "git": git_metadata(repository),
        "gpu_used_for_v8": False,
        "runpod_endpoint_status_at_start": "connection_refused",
        "replacement_pod_created": False,
        "qwen_used_for_v8": False,
        "qwen_reference_model": "Qwen/Qwen2.5-7B-Instruct",
        "qwen_reference_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "credentials_recorded": False,
    }
    atomic_json(
        results_root / "reproducibility" / "environment" / "environment_manifest.json",
        value,
    )
    return value
