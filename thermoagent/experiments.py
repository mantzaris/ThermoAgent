"""Reproducible experiment matrices, manifests, calibration, and PPO training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from . import __version__
from .environment import RNG_STREAM_OFFSETS, ScenarioConfig, derived_rng_seed
from .events import sha256_file
from .mechanics import MacrostateCalibration, free_energy_gap, occupancy_distribution
from .planners import MockPlanner, TransformersPlanner
from .policy import CoordinationPolicy, PPOConfig
from .runner import EpisodeRunner, calibration_from_json, write_episode
from .types import Method


MODEL_REQUIRED_METHODS = {
    Method.CENTRAL_LLM.value,
    Method.NO_COMM.value,
    Method.FIXED_COMM.value,
    Method.LEARNED_NO_ENTROPY.value,
    Method.THERMO.value,
    Method.RANDOM_GATE.value,
    Method.ENTROPY_LLM_ONLY.value,
    Method.NO_EPISODIC_MEMORY.value,
    Method.GLOBAL_ORACLE.value,
    Method.SHUFFLED_ENTROPY.value,
    Method.FIXED_ALWAYS_ON.value,
    Method.PERIODIC_COMMUNICATION.value,
    Method.RANDOM_BUDGET_MATCHED.value,
    Method.KPI_CUSUM_TRIGGER.value,
    Method.DOET_RULE.value,
    Method.DOET_RL.value,
    Method.GLOBAL_ENTROPY_TRIGGER_ORACLE.value,
    Method.DISRUPTION_LABEL_ORACLE.value,
}

LEARNED_METHODS = {
    Method.LEARNED_NO_ENTROPY.value,
    Method.THERMO.value,
    Method.NO_EPISODIC_MEMORY.value,
    Method.GLOBAL_ORACLE.value,
    Method.SHUFFLED_ENTROPY.value,
    Method.DOET_RL.value,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must contain a mapping")
    return value


def _load_training_trigger_settings(
    trigger_config_path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Mapping[str, Any]], Optional[Path]]:
    """Resolve the validation-selected trigger exactly as evaluation does.

    The selected-trigger artifact references nominal normalizers by repository-
    relative path. Training must not silently substitute the generic
    ``TriggerConfig`` center and scale, because that changes the trigger modes
    and therefore the DOET-RL actor's execution-time action masks.
    """

    trigger_record: Dict[str, Any] = {}
    if trigger_config_path is not None:
        trigger_record = load_yaml(trigger_config_path)
        if "trigger" in trigger_record and "parameters" not in trigger_record:
            trigger_record = dict(trigger_record["trigger"])
    trigger_parameters = dict(trigger_record.get("parameters", {}))
    trigger_normalizers = trigger_record.get("normalizers")
    normalizers_path: Optional[Path] = None
    configured_path = trigger_record.get("normalizers_path")
    if configured_path:
        configured = Path(str(configured_path))
        candidates = [configured] if configured.is_absolute() else [
            Path.cwd() / configured,
            *(
                parent / configured
                for parent in trigger_config_path.resolve().parents
            ),
        ]
        normalizers_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        ).resolve()
        if not normalizers_path.is_file():
            raise FileNotFoundError(
                "selected trigger normalizers are missing: %s"
                % normalizers_path
            )
        normalizer_record = load_yaml(normalizers_path)
        normalizer_key = str(trigger_record.get(
            "normalizers_key", "normalizers"
        ))
        if normalizer_key not in normalizer_record:
            raise ValueError(
                "selected trigger calibration %s has no %s field"
                % (normalizers_path, normalizer_key)
            )
        referenced_normalizers = normalizer_record[normalizer_key]
        if (
            trigger_normalizers is not None
            and trigger_normalizers != referenced_normalizers
        ):
            raise ValueError(
                "embedded selected trigger normalizers differ from %s[%s]"
                % (normalizers_path, normalizer_key)
            )
        trigger_normalizers = referenced_normalizers
    return (
        trigger_record,
        trigger_parameters,
        trigger_normalizers,
        normalizers_path,
    )


def _resolved_trigger_settings(
    config: Mapping[str, Any],
    scenario_values: Mapping[str, Any],
    root: Path,
) -> Dict[str, Any]:
    base = dict(config.get("trigger", {}))
    override = dict(scenario_values.get("trigger", {}))
    parameters = dict(base.get("parameters", {}))
    parameters.update(override.get("parameters", {}))
    settings = {**base, **override, "parameters": parameters}
    normalizers = settings.get("normalizers")
    normalizers_path = settings.get("normalizers_path")
    if normalizers is None and normalizers_path:
        path = Path(str(normalizers_path))
        if not path.is_absolute():
            path = root / path
        record = load_yaml(path)
        key = str(settings.get("normalizers_key", "normalizers"))
        if key not in record:
            raise ValueError(
                "trigger calibration %s has no %s field" % (path, key)
            )
        normalizers = record[key]
    settings["normalizers"] = normalizers
    return settings


@lru_cache(maxsize=8)
def source_checksum(root: Path) -> str:
    """Hash immutable run source once per process, then reuse in manifests."""

    root = root.resolve()
    digest = hashlib.sha256()
    included = []
    for base in ("thermoagent", "configs", "scripts", "tests"):
        directory = root / base
        if not directory.exists():
            continue
        included.extend(
            path for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        )
    included.extend(path for path in (root / "pyproject.toml", root / "requirements-runpod.txt") if path.exists())
    for path in sorted(included):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def dependency_versions() -> Dict[str, str]:
    names = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "networkx", "pydantic", "torch", "transformers", "accelerate", "bitsandbytes", "pymupdf"]
    versions: Dict[str, str] = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except (ImportError, OSError):
            versions[name] = "not-installed"
    return versions


@lru_cache(maxsize=1)
def hardware_summary() -> Dict[str, Any]:
    """Capture stable hardware once without an unbounded CUDA driver call."""
    summary: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        summary.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        })
    except ImportError:
        summary["torch"] = "not-installed"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        row = completed.stdout.strip().splitlines()[0]
        gpu, memory_mib, driver = [value.strip() for value in row.split(",", 2)]
        summary.update({
            "cuda_available": True,
            "gpu": gpu,
            "gpu_memory_bytes": int(float(memory_mib) * 1024 * 1024),
            "driver_version": driver,
            "gpu_query": "bounded nvidia-smi (10-second timeout)",
        })
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        summary.update({
            "cuda_available": None,
            "gpu": None,
            "gpu_memory_bytes": None,
            "gpu_query": "unavailable: %s" % type(error).__name__,
        })
    return summary


def capture_reproducibility(root: Path, results_root: Path) -> Dict[str, Any]:
    """Capture dependency and hardware facts without reading environment secrets."""

    output = results_root / "reproducibility"
    output.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=True,
    ).stdout
    (output / "pip-freeze.txt").write_text("\n".join(sorted(freeze.splitlines())) + "\n", encoding="utf-8")
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit",
            "--format=csv,noheader,nounits",
        ],
        "cpu": ["lscpu", "--json"],
        "memory": ["free", "--bytes"],
        "workspace_disk": ["df", "--block-size=1", "/workspace"],
    }
    command_output: Dict[str, Any] = {}
    for name, command in commands.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            command_output[name] = completed.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            command_output[name] = "unavailable: %s" % type(error).__name__
    record = {
        "captured_at": utc_now(),
        "hardware": hardware_summary(),
        "dependencies": dependency_versions(),
        "git_provenance": git_provenance(root),
        "source_checksum": source_checksum(root),
        "system_commands": command_output,
        "security_note": "No environment variables, credentials, SSH configuration, or tokens were read.",
    }
    (output / "hardware_and_source.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


@lru_cache(maxsize=8)
def git_provenance(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    if not (root / ".git").exists():
        return {
            "commit": "not-present-on-execution-copy",
            "branch": "not-present-on-execution-copy",
            "dirty": None,
        }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=False).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(root), capture_output=True, text=True, check=False,
    ).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, check=False).stdout
    return {
        "commit": commit or "unknown",
        "branch": branch or "detached-or-unknown",
        "dirty": bool(status.strip()),
    }


def capture_source_provenance(root: Path, output: Path) -> Dict[str, Any]:
    """Capture non-secret local Git/source identity for a filtered deployment.

    The normal RunPod synchronization deliberately excludes ``.git``.  This
    small record lets the execution copy report the exact originating branch
    and commit without copying Git metadata or credentials.  The independent
    source checksum remains the authoritative byte-level identity.
    """

    root = root.resolve()
    record = {
        **git_provenance(root),
        "source_checksum": source_checksum(root),
        "captured_at": utc_now(),
        "deployment": "filtered rsync; Git metadata and credentials excluded",
        "security_note": (
            "No environment variables, remotes, credentials, SSH configuration, "
            "or tokens were read."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def expand_matrix(config: Mapping[str, Any]) -> List[Tuple[str, int, int, str, Dict[str, Any]]]:
    matrix: List[Tuple[str, int, int, str, Dict[str, Any]]] = []
    balanced_counters: Dict[Tuple[str, str], int] = {}
    applications = config["applications"]
    for app_name, app_values in applications.items():
        sizes = app_values["agent_counts"] if "agent_counts" in app_values else [app_values["n_agents"]]
        for n_agents in sizes:
            for scenario_name, scenario in config["scenarios"].items():
                if scenario.get("applications") and app_name not in scenario["applications"]:
                    continue
                counts_by_application = scenario.get("agent_counts_by_application", {})
                allowed_counts = counts_by_application.get(app_name, scenario.get("agent_counts"))
                if allowed_counts and int(n_agents) not in [int(value) for value in allowed_counts]:
                    continue
                scenario_methods = scenario.get("methods", config["methods"])
                for method in scenario_methods:
                    method_seeds = scenario.get("method_seeds", {})
                    evaluation_seeds = method_seeds.get(
                        str(method), scenario.get("seeds", config["seeds"])
                    )
                    method_variants = config.get("method_variants", {}).get(
                        str(method), [{"name": "base"}]
                    )
                    rl_seeds = (
                        config.get("rl_seeds", [config.get("rl_seed", 0)])
                        if str(method) in LEARNED_METHODS else [0]
                    )
                    for variant in method_variants:
                        variant_name = str(variant.get("name", "base"))
                        for seed in evaluation_seeds:
                            panel_rl_seeds = list(rl_seeds)
                            if (
                                str(method) in LEARNED_METHODS
                                and bool(config.get("balanced_rl_assignment", False))
                            ):
                                counter_key = (str(method), variant_name)
                                index = balanced_counters.get(counter_key, 0)
                                balanced_counters[counter_key] = index + 1
                                index %= len(panel_rl_seeds)
                                panel_rl_seeds = [panel_rl_seeds[index]]
                            for rl_seed in panel_rl_seeds:
                                matrix.append((
                                    app_name,
                                    int(n_agents),
                                    int(seed),
                                    str(method),
                                    {
                                        "name": scenario_name,
                                        "_rl_seed": int(rl_seed),
                                        "_method_variant": variant_name,
                                        **scenario,
                                        **{
                                            key: value for key, value in variant.items()
                                            if key != "name"
                                        },
                                    },
                                ))
    return matrix


def _policy_for_method(
    method: str,
    checkpoints: Mapping[str, str],
    cache: Optional[Dict[str, CoordinationPolicy]] = None,
    rl_seed: int = 0,
) -> Optional[CoordinationPolicy]:
    if method not in LEARNED_METHODS:
        return None
    if method == Method.LEARNED_NO_ENTROPY.value:
        key = "no_entropy"
    elif method == Method.DOET_RL.value:
        key = "doet_rl"
    else:
        key = "thermo"
    if key not in checkpoints:
        raise ValueError("method %s requires checkpoints.%s" % (method, key))
    configured = checkpoints[key]
    if isinstance(configured, Mapping):
        seed_key = str(int(rl_seed))
        if seed_key not in configured and int(rl_seed) not in configured:
            raise ValueError(
                "method %s has no checkpoint for RL seed %s" % (method, rl_seed)
            )
        configured = configured.get(seed_key, configured.get(int(rl_seed)))
    cache_key = "%s:%s:%s" % (key, rl_seed, configured)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    policy = CoordinationPolicy.load(Path(str(configured)))
    if cache is not None:
        cache[cache_key] = policy
    return policy


def _planner_for_method(method: str, shared_planner: Optional[Any]) -> Any:
    return shared_planner if method in MODEL_REQUIRED_METHODS and shared_planner is not None else MockPlanner()


def _checkpoint_for_method(
    method: str,
    checkpoints: Mapping[str, Any],
    rl_seed: int,
) -> Optional[str]:
    if method not in LEARNED_METHODS:
        return None
    if method == Method.LEARNED_NO_ENTROPY.value:
        key = "no_entropy"
    elif method == Method.DOET_RL.value:
        key = "doet_rl"
    else:
        key = "thermo"
    configured = checkpoints.get(key)
    if isinstance(configured, Mapping):
        configured = configured.get(str(int(rl_seed)), configured.get(int(rl_seed)))
    return str(configured) if configured is not None else None


def _recover_published_staging(
    staging_root: Path,
    output_dir: Path,
    run_id: str,
    manifest: Mapping[str, Any],
) -> bool:
    """Finish a manifest-backed atomic publish without rerunning an episode."""

    expected = dict(manifest.get("output_checksums", {}))
    matches: List[Path] = []
    for candidate in sorted(staging_root.glob(run_id + ".partial-*")):
        episode = candidate / "episode.json"
        events = candidate / "events.jsonl.gz"
        if (
            episode.exists()
            and events.exists()
            and sha256_file(episode) == expected.get("episode.json")
            and sha256_file(events) == expected.get("events.jsonl.gz")
        ):
            matches.append(candidate)
    if len(matches) != 1 or output_dir.exists():
        return False
    matches[0].replace(output_dir)
    return True


def _published_output_matches(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> bool:
    """Verify both immutable episode files before accepting a resumed row."""

    expected = dict(manifest.get("output_checksums", {}))
    required = ("episode.json", "events.jsonl.gz")
    if any(not expected.get(name) for name in required):
        return False
    return all(
        (output_dir / name).is_file()
        and sha256_file(output_dir / name) == expected[name]
        for name in required
    )


def _resumed_manifest_matches_execution(
    manifest: Mapping[str, Any],
    expected_source_checksum: str,
    scenario: ScenarioConfig,
    method: str,
    rl_seed: int,
    run_config: Mapping[str, Any],
) -> bool:
    """Reject a run-ID collision from a different frozen execution contract."""

    uses_model = method in MODEL_REQUIRED_METHODS
    model = dict(run_config.get("model", {}))
    expected_protocol = run_config.get("protocol_checksum")
    checks = (
        manifest.get("completion_status") == "complete",
        manifest.get("source", {}).get("checksum")
        == expected_source_checksum,
        manifest.get("application") == scenario.application,
        manifest.get("method") == method,
        manifest.get("configuration") == asdict(scenario),
        int(manifest.get("environment_seed", -1)) == scenario.seed,
        int(manifest.get("llm_seed", -1))
        == int(run_config.get("llm_seed", 0)),
        int(manifest.get("rl_seed", -1)) == int(rl_seed),
        manifest.get("topology_identifier") == scenario.topology,
        manifest.get("model_identifier")
        == (model.get("identifier") if uses_model else "none"),
        manifest.get("model_revision")
        == (model.get("revision") if uses_model else "none"),
        manifest.get("protocol_checksum") == expected_protocol,
        manifest.get("trigger_parameters")
        == run_config.get("resolved_trigger", {}).get("parameters"),
    )
    return all(checks)


def _episode_manifest(
    root: Path,
    result: Any,
    scenario: ScenarioConfig,
    config: Mapping[str, Any],
    output_checksums: Mapping[str, str],
    started_at: str,
    ended_at: str,
    topology_checksum: Optional[str] = None,
    rl_seed: int = 0,
) -> Dict[str, Any]:
    model = dict(config.get("model", {}))
    uses_model = result.method in MODEL_REQUIRED_METHODS
    configured_checkpoints = dict(config.get("checkpoints", {}))
    checkpoint = _checkpoint_for_method(
        result.method, configured_checkpoints, rl_seed
    )
    return {
        "run_id": result.run_id,
        "source": {
            **git_provenance(root),
            **dict(config.get("source_provenance", {})),
            "checksum": source_checksum(root),
            "thermoagent_version": __version__,
        },
        "configuration": asdict(scenario),
        "experiment_configuration": dict(config),
        "application": result.application,
        "method": result.method,
        "method_uses_language_model": uses_model,
        "model_identifier": model.get("identifier") if uses_model else "none",
        "model_revision": model.get("revision") if uses_model else "none",
        "precision": model.get("precision") if uses_model else "none",
        "planner_revision": result.planner_metrics.get("planner_revision"),
        "prompt_template_revision": (
            config.get("prompt_template_revision", "planner-json-v2")
            if uses_model else "not-applicable"
        ),
        "agentic_metric_revision": config.get(
            "agentic_metric_revision", "agentic-metrics-v1"
        ),
        "central_baseline_revision": config.get(
            "central_baseline_revision", "central-controls-v1"
        ),
        "decoding": model.get("decoding", {"do_sample": False}) if uses_model else None,
        "max_input_tokens": int(model.get("max_input_tokens", 2560)) if uses_model else None,
        "max_new_tokens": int(model.get("max_new_tokens", 128)) if uses_model else None,
        "environment_seed": scenario.seed,
        "environment_rng_streams": {
            stream: derived_rng_seed(scenario.seed, stream)
            for stream in sorted(RNG_STREAM_OFFSETS)
        },
        "agent_rng_rule": "environment_seed * 100 + stable agent index",
        "llm_seed": int(config.get("llm_seed", 0)),
        "rl_seed": int(rl_seed),
        "topology_identifier": scenario.topology,
        "topology_checksum": topology_checksum or hashlib.sha256((scenario.topology + ":" + str(scenario.n_agents)).encode()).hexdigest(),
        "dependencies": dependency_versions(),
        "hardware": hardware_summary(),
        "start_timestamp": started_at,
        "end_timestamp": ended_at,
        "wall_clock_seconds": result.wall_clock_seconds,
        "single_gpu_hours": (
            result.wall_clock_seconds / 3600.0
            if uses_model and result.planner_metrics["llm_calls"] > 0 else 0.0
        ),
        "approximate_gpu_cost_usd": (
            result.wall_clock_seconds / 3600.0
            * float(config.get("hourly_gpu_rate_usd", 0.34))
            if uses_model and result.planner_metrics["llm_calls"] > 0 else 0.0
        ),
        "environment_steps": scenario.horizon,
        "llm_calls": result.planner_metrics["llm_calls"],
        "prompt_tokens": result.planner_metrics["prompt_tokens"],
        "generated_tokens": result.planner_metrics["generated_tokens"],
        "tool_calls": result.metrics["tool_calls"],
        "messages": result.metrics["total_communication_messages"],
        "operational_messages": result.metrics["messages"],
        "entropy_sketch_messages": result.metrics["monitor_sketch_messages"],
        "structured_bytes": result.metrics["total_communication_bytes"],
        "operational_message_bytes": result.metrics["message_bytes"],
        "entropy_sketch_bytes": result.metrics["monitor_sketch_bytes"],
        "communication_active_decision_epochs": result.metrics.get(
            "communication_active_decision_epochs", 0
        ),
        "trigger_type": config.get("resolved_trigger", {}).get(
            "parameters", {}
        ).get("trigger_type"),
        "trigger_parameters": config.get("resolved_trigger", {}).get(
            "parameters"
        ),
        "communication_mode": result.method,
        "protocol_checksum": config.get("protocol_checksum"),
        "checkpoint_selection": checkpoint,
        "checkpoint_sha256": (
            sha256_file(Path(checkpoint))
            if checkpoint and Path(checkpoint).is_file() else None
        ),
        "output_checksums": dict(output_checksums),
        "completion_status": result.completion_status,
    }


def run_matrix(config_path: Path, root: Path, results_root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sweep_started_at = utc_now()
    sweep_started = time.perf_counter()
    config = load_yaml(config_path)
    provenance_value = config.get("source_provenance_path")
    if provenance_value:
        provenance_path = Path(str(provenance_value))
        if not provenance_path.is_absolute():
            provenance_path = root / provenance_path
        if not provenance_path.is_file():
            raise FileNotFoundError(
                "filtered-deployment source provenance is missing: %s"
                % provenance_path
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("source_checksum") != source_checksum(root):
            raise RuntimeError(
                "execution source differs from captured deployment provenance"
            )
        config["source_provenance"] = {
            "commit": provenance.get("commit", "unknown"),
            "branch": provenance.get("branch", "unknown"),
            "dirty": provenance.get("dirty"),
            "deployment_provenance_checksum": sha256_file(provenance_path),
        }
    freeze_value = config.get("protocol_freeze_path")
    if freeze_value:
        freeze_path = Path(str(freeze_value))
        if not freeze_path.is_absolute():
            freeze_path = root / freeze_path
        verify_protocol(root, freeze_path)
        config["protocol_checksum"] = sha256_file(freeze_path)
    stage = str(config["stage"])
    calibration_path = Path(config["calibration"]) if config.get("calibration") else None
    calibration = calibration_from_json(calibration_path)
    monitor_settings = {"formulation": "pooled", "window": 3}
    if calibration_path is not None:
        calibration_record = json.loads(calibration_path.read_text(encoding="utf-8"))
        monitor_settings.update(calibration_record.get("monitor_selection", {}))
    checkpoints = config.get("checkpoints", {})
    matrix = expand_matrix(config)
    if limit is not None:
        matrix = matrix[:limit]
    model_config = config.get("model", {})
    needs_model = any(method in MODEL_REQUIRED_METHODS for _, _, _, method, _ in matrix)
    planner: Optional[Any] = None
    planner_backend = str(config.get("planner_backend", "transformers"))
    if planner_backend not in ("transformers", "mock"):
        raise ValueError("planner_backend must be transformers or mock")
    if needs_model and planner_backend == "transformers":
        planner = TransformersPlanner(
            model_id=model_config["identifier"],
            revision=model_config["revision"],
            max_new_tokens=int(model_config.get("max_new_tokens", 128)),
            max_input_tokens=int(model_config.get("max_input_tokens", 2560)),
            load_in_4bit=bool(model_config.get("load_in_4bit", True)),
            seed=int(config.get("llm_seed", 0)),
        )
    summaries: List[Dict[str, Any]] = []
    policy_cache: Dict[str, CoordinationPolicy] = {}
    summary_dir = results_root / stage
    raw_root = results_root / "raw" / stage
    staging_root = results_root / ".staging" / stage
    manifest_root = results_root / "manifests"
    log_root = results_root / "logs" / "evaluation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    for index, (application, n_agents, seed, method, scenario_values) in enumerate(matrix):
        rl_seed = int(scenario_values.get("_rl_seed", config.get("rl_seed", 0)))
        scenario = ScenarioConfig(
            application=application,
            seed=seed,
            horizon=int(scenario_values.get("horizon", config.get("horizon", 20))),
            n_agents=n_agents,
            private_information=float(scenario_values.get("private_information", 0.5)),
            objective_misalignment=float(scenario_values.get("objective_misalignment", 0.5)),
            communication=str(scenario_values.get("communication", "reliable")),
            disruption=str(scenario_values.get("disruption", "moderate")),
            decision_interval=int(scenario_values.get("decision_interval", config.get("decision_interval", 4))),
            communication_budget=int(scenario_values.get("communication_budget", config.get("communication_budget", 12))),
            random_gate_probability=float(scenario_values.get("random_gate_probability", config.get("random_gate_probability", 0.5))),
            topology=str(scenario_values.get("topology", "ring_plus_hubs")),
        )
        trigger_settings = _resolved_trigger_settings(
            config, scenario_values, root
        )
        run_config = {
            **config,
            "resolved_scenario_name": scenario_values["name"],
            "resolved_trigger": trigger_settings,
            "resolved_rl_seed": rl_seed,
        }
        run_id = "%s-%s-%s-n%02d-s%03d" % (stage, application, method, n_agents, seed)
        if len(config["scenarios"]) > 1:
            run_id = "%s-%s" % (run_id, scenario_values["name"])
        if method in LEARNED_METHODS:
            run_id = "%s-r%04d" % (run_id, rl_seed)
        method_variant = str(scenario_values.get("_method_variant", "base"))
        if method_variant != "base":
            run_id = "%s-v%s" % (run_id, method_variant)
        output_dir = raw_root / run_id
        episode_path = output_dir / "episode.json"
        existing_manifest_path = manifest_root / (run_id + ".json")
        if existing_manifest_path.exists() and not episode_path.exists():
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            if existing_manifest.get("completion_status") == "complete":
                _recover_published_staging(
                    staging_root, output_dir, run_id, existing_manifest
                )
            if episode_path.exists():
                # The normal resume branch below now validates and records it.
                pass
            if existing_manifest.get("completion_status") == "failed":
                summaries.append({
                    "run_id": run_id,
                    "application": application,
                    "method": method,
                    "method_variant": method_variant,
                    "scenario_name": scenario_values["name"],
                    "seed": seed,
                    "rl_training_seed": rl_seed,
                    "n_agents": n_agents,
                    "status": "failed",
                    "error": existing_manifest.get("error", "recorded failure"),
                    "resumed": True,
                    "manifest": str(existing_manifest_path.relative_to(results_root)),
                })
                continue
            if not episode_path.exists():
                summaries.append({
                    "run_id": run_id,
                    "application": application,
                    "method": method,
                    "method_variant": method_variant,
                    "scenario_name": scenario_values["name"],
                    "seed": seed,
                    "rl_training_seed": rl_seed,
                    "n_agents": n_agents,
                    "status": "failed",
                    "error": "complete manifest lacks one checksum-matching staged output; retained without rerun",
                    "wall_clock_seconds": existing_manifest.get("wall_clock_seconds", 0.0),
                    "resumed": True,
                    "manifest": str(existing_manifest_path.relative_to(results_root)),
                })
                continue
        if episode_path.exists():
            try:
                existing = json.loads(
                    episode_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                summaries.append({
                    "run_id": run_id,
                    "application": application,
                    "method": method,
                    "method_variant": method_variant,
                    "scenario_name": scenario_values["name"],
                    "seed": seed,
                    "rl_training_seed": rl_seed,
                    "n_agents": n_agents,
                    "status": "failed",
                    "error": (
                        "published episode is unreadable (%s); retained "
                        "without rerun" % type(error).__name__
                    ),
                    "wall_clock_seconds": 0.0,
                    "resumed": True,
                    "manifest": (
                        str(existing_manifest_path.relative_to(results_root))
                        if existing_manifest_path.exists() else ""
                    ),
                })
                continue
            if (
                existing.get("completion_status") == "complete"
                and not existing_manifest_path.exists()
            ):
                summaries.append({
                    "run_id": run_id,
                    "application": application,
                    "method": method,
                    "method_variant": method_variant,
                    "scenario_name": scenario_values["name"],
                    "seed": seed,
                    "rl_training_seed": rl_seed,
                    "n_agents": n_agents,
                    "status": "failed",
                    "error": "complete episode lacks its required manifest; retained without rerun",
                    "wall_clock_seconds": existing.get("wall_clock_seconds", 0.0),
                    "resumed": True,
                })
                continue
            if existing.get("completion_status") == "complete":
                existing_manifest = json.loads(
                    existing_manifest_path.read_text(encoding="utf-8")
                )
                if not _resumed_manifest_matches_execution(
                    existing_manifest,
                    source_checksum(root),
                    scenario,
                    method,
                    rl_seed,
                    run_config,
                ):
                    summaries.append({
                        "run_id": run_id,
                        "application": application,
                        "method": method,
                        "method_variant": method_variant,
                        "scenario_name": scenario_values["name"],
                        "seed": seed,
                        "rl_training_seed": rl_seed,
                        "n_agents": n_agents,
                        "status": "failed",
                        "error": (
                            "published manifest does not match the current "
                            "frozen execution contract; retained without rerun"
                        ),
                        "wall_clock_seconds": existing.get(
                            "wall_clock_seconds", 0.0
                        ),
                        "resumed": True,
                        "manifest": str(
                            existing_manifest_path.relative_to(results_root)
                        ),
                    })
                    continue
                if not _published_output_matches(output_dir, existing_manifest):
                    summaries.append({
                        "run_id": run_id,
                        "application": application,
                        "method": method,
                        "method_variant": method_variant,
                        "scenario_name": scenario_values["name"],
                        "seed": seed,
                        "rl_training_seed": rl_seed,
                        "n_agents": n_agents,
                        "status": "failed",
                        "error": (
                            "published episode checksum mismatch; retained "
                            "without rerun"
                        ),
                        "wall_clock_seconds": existing.get(
                            "wall_clock_seconds", 0.0
                        ),
                        "resumed": True,
                        "manifest": str(
                            existing_manifest_path.relative_to(results_root)
                        ),
                    })
                    continue
                summaries.append({
                    "run_id": run_id,
                    "application": application,
                    "method": method,
                    "method_variant": method_variant,
                    "scenario": existing["scenario"],
                    "scenario_name": scenario_values["name"],
                    "seed": seed,
                    "rl_training_seed": rl_seed,
                    "n_agents": n_agents,
                    **existing["metrics"],
                    **existing.get("agent_metrics", {}),
                    **existing.get("planner_metrics", {}),
                    "wall_clock_seconds": existing.get("wall_clock_seconds", 0.0),
                    "status": "complete",
                    "resumed": True,
                    "manifest": str((manifest_root / (run_id + ".json")).relative_to(results_root)),
                })
                continue
        policy = _policy_for_method(
            method, checkpoints, policy_cache, rl_seed=rl_seed
        )
        episode_planner = _planner_for_method(method, planner)
        runner = EpisodeRunner(
            scenario, method, planner=episode_planner, policy=policy,
            calibration=calibration,
            monitor_window=int(monitor_settings["window"]),
            monitor_formulation=str(monitor_settings["formulation"]),
            trigger_config=dict(trigger_settings.get("parameters", {})) or None,
            trigger_normalizers=trigger_settings.get("normalizers"),
            periodic_interval=scenario_values.get(
                "periodic_interval", config.get("periodic_interval")
            ),
            fixed_broadcast_fanout=int(scenario_values.get(
                "fixed_broadcast_fanout", config.get("fixed_broadcast_fanout", 3)
            )),
        )
        started_at = utc_now()
        log_path = log_root / (run_id + ".json")
        episode_started = time.perf_counter()
        try:
            result = runner.run(run_id)
            # Publish an episode directory only after its manifest exists. If a
            # process dies in the narrow write window, the hidden staging copy
            # is retained for diagnosis and replay never mistakes it for a
            # complete experimental unit.
            staging_dir = staging_root / (
                "%s.partial-%s" % (
                    run_id,
                    started_at.replace(":", "").replace("+", "_").replace(".", ""),
                )
            )
            checksums = write_episode(result, runner.env.ledger, staging_dir)
            ended_at = utc_now()
            topology_payload = json.dumps({
                "physical_edges": sorted([list(edge) for edge in runner.env.initial_physical_edges]),
                "communication_edges": sorted([list(edge) for edge in runner.env.initial_communication_edges]),
            }, sort_keys=True, separators=(",", ":"))
            topology_checksum = hashlib.sha256(topology_payload.encode("utf-8")).hexdigest()
            manifest = _episode_manifest(
                root, result, scenario, run_config, checksums, started_at, ended_at,
                topology_checksum=topology_checksum,
                rl_seed=rl_seed,
            )
            manifest_path = manifest_root / (run_id + ".json")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if output_dir.exists():
                raise FileExistsError(
                    "refusing to replace an existing incomplete output directory: %s"
                    % output_dir
                )
            staging_dir.replace(output_dir)
            row = {
                "run_id": run_id, "application": application, "method": method,
                "method_variant": method_variant, "scenario": result.scenario,
                "scenario_name": scenario_values["name"], "seed": seed,
                "rl_training_seed": rl_seed, "n_agents": n_agents,
                **result.metrics, **result.agent_metrics, **result.planner_metrics,
                "wall_clock_seconds": result.wall_clock_seconds, "status": result.completion_status,
                "manifest": str(manifest_path.relative_to(results_root)),
            }
            summaries.append(row)
            log_path.write_text(json.dumps({"status": "complete", "index": index, "total": len(matrix), "run_id": run_id, "ended_at": ended_at}) + "\n", encoding="utf-8")
        except Exception as error:
            ended_at = utc_now()
            row = {"run_id": run_id, "application": application, "method": method, "method_variant": method_variant, "scenario_name": scenario_values["name"], "seed": seed, "rl_training_seed": rl_seed, "n_agents": n_agents, "status": "failed", "error": "%s: %s" % (type(error).__name__, str(error)), "wall_clock_seconds": time.perf_counter() - episode_started}
            summaries.append(row)
            log_path.write_text(json.dumps({**row, "ended_at": ended_at}) + "\n", encoding="utf-8")
            checkpoint_selection = _checkpoint_for_method(
                method, checkpoints, rl_seed
            )
            topology_payload = json.dumps({
                "physical_edges": sorted([
                    list(edge) for edge in runner.env.initial_physical_edges
                ]),
                "communication_edges": sorted([
                    list(edge) for edge in runner.env.initial_communication_edges
                ]),
            }, sort_keys=True, separators=(",", ":"))
            uses_model = method in MODEL_REQUIRED_METHODS
            failure_manifest = {
                "run_id": run_id,
                "source": {**git_provenance(root), **dict(config.get("source_provenance", {})), "checksum": source_checksum(root)},
                "configuration": asdict(scenario),
                "experiment_configuration": dict(run_config),
                "application": application,
                "method": method,
                "method_uses_language_model": uses_model,
                "model_identifier": (
                    model_config.get("identifier")
                    if uses_model else "none"
                ),
                "model_revision": (
                    model_config.get("revision")
                    if uses_model else "none"
                ),
                "precision": model_config.get("precision") if uses_model else "none",
                "planner_revision": getattr(runner.planner, "revision", "unknown"),
                "prompt_template_revision": (
                    config.get("prompt_template_revision", "planner-json-v2")
                    if uses_model else "not-applicable"
                ),
                "max_input_tokens": (
                    int(model_config.get("max_input_tokens", 2560))
                    if uses_model else None
                ),
                "max_new_tokens": (
                    int(model_config.get("max_new_tokens", 128))
                    if uses_model else None
                ),
                "decoding": (
                    model_config.get("decoding", {"do_sample": False})
                    if uses_model else None
                ),
                "agentic_metric_revision": config.get(
                    "agentic_metric_revision", "agentic-metrics-v1"
                ),
                "central_baseline_revision": config.get(
                    "central_baseline_revision", "central-controls-v1"
                ),
                "environment_seed": seed,
                "environment_rng_streams": {
                    stream: derived_rng_seed(seed, stream)
                    for stream in sorted(RNG_STREAM_OFFSETS)
                },
                "agent_rng_rule": "environment_seed * 100 + stable agent index",
                "llm_seed": int(config.get("llm_seed", 0)),
                "rl_seed": int(rl_seed),
                "topology_identifier": scenario.topology,
                "topology_checksum": hashlib.sha256(
                    topology_payload.encode("utf-8")
                ).hexdigest(),
                "dependencies": dependency_versions(),
                "hardware": hardware_summary(),
                "start_timestamp": started_at,
                "end_timestamp": ended_at,
                "wall_clock_seconds": row["wall_clock_seconds"],
                "single_gpu_hours": (
                    row["wall_clock_seconds"] / 3600.0
                    if uses_model and runner.llm_calls > 0 else 0.0
                ),
                "approximate_gpu_cost_usd": (
                    row["wall_clock_seconds"] / 3600.0
                    * float(config.get("hourly_gpu_rate_usd", 0.34))
                    if uses_model and runner.llm_calls > 0 else 0.0
                ),
                "environment_steps": int(
                    len(runner.macro_features) / max(len(runner.env.agent_ids), 1)
                ),
                "llm_calls": runner.llm_calls,
                "prompt_tokens": runner.prompt_tokens,
                "generated_tokens": runner.generated_tokens,
                "tool_calls": runner.env.tool_calls,
                "messages": (
                    runner.env.message_attempts + runner.monitor_sketch_messages
                ),
                "operational_messages": runner.env.message_attempts,
                "entropy_sketch_messages": runner.monitor_sketch_messages,
                "structured_bytes": (
                    runner.env.message_bytes + runner.monitor_sketch_bytes
                ),
                "operational_message_bytes": runner.env.message_bytes,
                "entropy_sketch_bytes": runner.monitor_sketch_bytes,
                "communication_active_decision_epochs": (
                    runner.communication_active_decision_epochs
                ),
                "trigger_type": trigger_settings.get(
                    "parameters", {}
                ).get("trigger_type"),
                "trigger_parameters": trigger_settings.get("parameters"),
                "communication_mode": method,
                "protocol_checksum": run_config.get("protocol_checksum"),
                "completion_status": "failed",
                "error": row["error"],
                "checkpoint_selection": checkpoint_selection,
                "checkpoint_sha256": (
                    sha256_file(Path(checkpoint_selection))
                    if checkpoint_selection
                    and Path(checkpoint_selection).is_file()
                    else None
                ),
                "output_checksums": {},
            }
            (manifest_root / (run_id + ".json")).write_text(
                json.dumps(failure_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    summary_path = summary_dir / "episodes.csv"
    history_dir = summary_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        history_path = history_dir / ("episodes-%s.csv" % sha256_file(summary_path)[:12])
        if not history_path.exists():
            shutil.copy2(str(summary_path), str(history_path))
    keys = sorted({key for row in summaries for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=keys, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    jsonl_path = summary_dir / "episodes.jsonl"
    if jsonl_path.exists():
        history_path = history_dir / ("episodes-%s.jsonl" % sha256_file(jsonl_path)[:12])
        if not history_path.exists():
            shutil.copy2(str(jsonl_path), str(history_path))
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries), encoding="utf-8")
    sweep_manifest = {
        "stage": stage,
        "config_path": str(config_path),
        "config_checksum": sha256_file(config_path),
        "source_checksum": source_checksum(root),
        "planner_backend": planner_backend,
        "model_loaded": bool(needs_model and planner_backend == "transformers"),
        "start_timestamp": sweep_started_at,
        "end_timestamp": utc_now(),
        "wall_clock_seconds_including_model_load": time.perf_counter() - sweep_started,
        "cumulative_episode_single_gpu_hours": sum(
            float(row.get("wall_clock_seconds", 0.0) or 0.0) / 3600.0
            for row in summaries
            if int(row.get("llm_calls", 0) or 0) > 0
        ),
        "episodes_planned": len(matrix),
        "episodes_complete": sum(row.get("status") == "complete" for row in summaries),
        "episodes_failed": sum(row.get("status") == "failed" for row in summaries),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in summaries),
        "generated_tokens": sum(int(row.get("generated_tokens", 0) or 0) for row in summaries),
        "llm_calls": sum(int(row.get("llm_calls", 0) or 0) for row in summaries),
    }
    sweep_path = manifest_root / (stage + "_sweep.json")
    if sweep_path.exists():
        safe_timestamp = sweep_started_at.replace(":", "").replace("+", "_")
        sweep_path = manifest_root / (stage + "_sweep_" + safe_timestamp + ".json")
    sweep_path.write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summaries


def calibrate_nominal(output_path: Path, seeds: Sequence[int], horizon: int = 24) -> Dict[str, Any]:
    features: List[List[float]] = []
    features_by_role: Dict[str, List[List[float]]] = {}
    entropies: List[float] = []
    for application, n_agents in (("commercial", 8), ("humanitarian", 8)):
        for seed in seeds:
            config = ScenarioConfig(application=application, seed=int(seed), horizon=horizon, n_agents=n_agents, disruption="nominal", communication="reliable")
            runner = EpisodeRunner(config, Method.SCRIPTED.value)
            result = runner.run()
            features.extend(runner.macro_features)
            agent_ids = runner.env.agent_ids
            for index, feature in enumerate(runner.macro_features):
                role = runner.env.agents[agent_ids[index % len(agent_ids)]].identity.role
                features_by_role.setdefault(role, []).append(feature)
            entropies.extend(row["exact_entropy"] for row in result.time_series)
    calibration = MacrostateCalibration.fit(features, alpha=0.1)
    nominal_states = [calibration.encode(feature) for feature in features]
    nominal_p = occupancy_distribution(nominal_states, calibration.alpha)
    temperature_grid = np.geomspace(0.05, 2.0, 120)
    objectives: List[float] = []
    for temperature in temperature_grid:
        calibration.temperature = float(temperature)
        q = calibration.healthy_reference()
        objectives.append(free_energy_gap(nominal_p, q, 1.0))  # KL; unit T avoids rescaling the objective.
    best_index = int(np.argmin(objectives))
    calibration.temperature = float(temperature_grid[best_index])
    # Local surprisal uses a nominal role-conditioned reference. Sparse roles
    # are shrunk halfway toward the Gibbs healthy ensemble fixed above.
    global_reference = calibration.healthy_reference()
    role_shrinkage = 0.5
    calibration.role_references = {}
    for role, role_features in sorted(features_by_role.items()):
        role_states = [calibration.encode(feature) for feature in role_features]
        empirical = occupancy_distribution(role_states, calibration.alpha)
        reference = role_shrinkage * empirical + (1.0 - role_shrinkage) * global_reference
        calibration.role_references[role] = reference.tolist()
    value = {
        "calibration_source": "nominal scripted training episodes only",
        "seeds": list(seeds),
        "horizon": horizon,
        "n_feature_rows": len(features),
        "thresholds": calibration.thresholds.tolist(),
        "alpha": calibration.alpha,
        "temperature": calibration.temperature,
        "temperature_calibration": {
            "method": "nominal occupancy KL minimization over fixed log grid",
            "grid_min": float(temperature_grid.min()),
            "grid_max": float(temperature_grid.max()),
            "grid_points": len(temperature_grid),
            "nominal_kl_at_selected_temperature": float(objectives[best_index]),
        },
        "energy_weights": list(calibration.energy_weights),
        "role_references": calibration.role_references,
        "role_reference_calibration": {
            "source": "same nominal scripted training episodes only",
            "laplace_alpha": calibration.alpha,
            "empirical_weight": role_shrinkage,
            "gibbs_shrinkage_weight": 1.0 - role_shrinkage,
            "rows_by_role": {
                role: len(role_features)
                for role, role_features in sorted(features_by_role.items())
            },
        },
        "nominal_entropy_mean_before_refit": float(np.mean(entropies)),
        "generated_at": utc_now(),
        "monitor_selection": {"formulation": "pooled", "window": 3, "status": "provisional before monitor pilot"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def select_monitor_formulation(
    calibration_path: Path,
    output_path: Path,
    seeds: Sequence[int],
    horizon: int = 18,
) -> Dict[str, Any]:
    """Select from requested estimators using only nominal/moderate pilot data."""
    from .mechanics import RollingMacrostateMonitor

    calibration = calibration_from_json(calibration_path)
    monitor_settings = {"formulation": "pooled", "window": 3}
    if calibration_path is not None:
        monitor_settings.update(json.loads(calibration_path.read_text(encoding="utf-8")).get("monitor_selection", {}))
    datasets: Dict[str, List[Tuple[List[Dict[str, int]], Dict[str, str], int]]] = {"nominal": [], "moderate": []}
    for disruption in datasets:
        for application, n_agents in (("commercial", 8), ("humanitarian", 8)):
            for seed in seeds:
                scenario = ScenarioConfig(
                    application=application, seed=int(seed), horizon=horizon, n_agents=n_agents,
                    communication="reliable", disruption=disruption, decision_interval=4,
                )
                runner = EpisodeRunner(scenario, Method.SCRIPTED.value, calibration=calibration)
                runner.run()
                ids = runner.env.agent_ids
                roles = {agent_id: runner.env.agents[agent_id].identity.role for agent_id in ids}
                frames: List[Dict[str, int]] = []
                width = len(ids)
                for start in range(0, len(runner.macro_features), width):
                    features = runner.macro_features[start : start + width]
                    frames.append({agent_id: calibration.encode(feature) for agent_id, feature in zip(ids, features)})
                datasets[disruption].append((frames, roles, max(2, horizon // 3)))

    candidates = [("pooled", 1), ("pooled", 3), ("role_conditioned", 1), ("role_conditioned", 3)]
    comparisons: List[Dict[str, Any]] = []
    for formulation, window in candidates:
        nominal_values: List[float] = []
        disrupted_pre: List[float] = []
        disrupted_post: List[float] = []
        nominal_free: List[float] = []
        post_free: List[float] = []
        for frames, roles, disruption_step in datasets["nominal"]:
            monitor = RollingMacrostateMonitor(calibration, window=window, formulation=formulation)
            for frame in frames:
                values = monitor.update(frame, roles)
                nominal_values.append(float(values["entropy"]))
                nominal_free.append(float(values["free_energy"]))
        for frames, roles, disruption_step in datasets["moderate"]:
            monitor = RollingMacrostateMonitor(calibration, window=window, formulation=formulation)
            for index, frame in enumerate(frames):
                values = monitor.update(frame, roles)
                target = disrupted_post if index >= disruption_step else disrupted_pre
                target.append(float(values["entropy"]))
                if index >= disruption_step:
                    post_free.append(float(values["free_energy"]))
        nominal_mean = float(np.mean(nominal_values))
        nominal_std = float(np.std(nominal_values))
        entropy_shift = float(np.mean(disrupted_post) - np.mean(disrupted_pre))
        free_shift = float(np.mean(post_free) - np.mean(nominal_free))
        stability_cv = nominal_std / max(abs(nominal_mean), 1e-6)
        sensitivity_score = abs(entropy_shift) / max(nominal_std, 0.02) + abs(free_shift) / max(float(np.std(nominal_free)), 0.02)
        score = sensitivity_score - max(0.0, stability_cv - 0.35) * 2.0
        comparisons.append({
            "formulation": formulation, "window": window,
            "nominal_entropy_mean": nominal_mean, "nominal_entropy_std": nominal_std,
            "nominal_stability_cv": stability_cv, "entropy_shift_after_moderate_disruption": entropy_shift,
            "free_energy_shift_after_moderate_disruption": free_shift,
            "sensitivity_score": sensitivity_score, "selection_score": score,
        })
    eligible = [row for row in comparisons if row["nominal_stability_cv"] <= 0.5] or comparisons
    selected = max(eligible, key=lambda row: row["selection_score"])
    record = {
        "status": "selected before method comparison and main evaluation",
        "seeds": list(seeds), "horizon": horizon,
        "selection_rule": "maximize absolute disruption sensitivity with nominal CV penalty; no treatment outcomes used",
        "selected": {"formulation": selected["formulation"], "window": selected["window"]},
        "comparisons": comparisons,
        "generated_at": utc_now(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calibration_record = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_record["monitor_selection"] = {
        **record["selected"], "status": record["status"],
        "comparison_artifact": str(output_path),
    }
    calibration_path.write_text(json.dumps(calibration_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def train_policy(
    output_path: Path,
    variant: str,
    episodes: int,
    seed: int,
    calibration_path: Optional[Path],
    log_path: Path,
    trigger_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if variant not in ("thermo", "no_entropy", "doet_rl"):
        raise ValueError("variant must be thermo, no_entropy, or doet_rl")
    calibration = calibration_from_json(calibration_path)
    monitor_settings = {"formulation": "pooled", "window": 3}
    if calibration_path is not None:
        monitor_settings.update(
            json.loads(calibration_path.read_text(encoding="utf-8")).get("monitor_selection", {})
        )
    policy = CoordinationPolicy(PPOConfig(), seed=seed)
    rng = np.random.RandomState(seed)
    pending: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    method = {
        "thermo": Method.THERMO.value,
        "no_entropy": Method.LEARNED_NO_ENTROPY.value,
        "doet_rl": Method.DOET_RL.value,
    }[variant]
    (
        trigger_record,
        trigger_parameters,
        trigger_normalizers,
        trigger_normalizers_path,
    ) = _load_training_trigger_settings(trigger_config_path)
    scenarios = [
        ("nominal", "reliable", 0.0, 0.0),
        ("moderate", "reliable", 0.5, 0.5),
        ("correlated", "intermittent", 1.0, 0.8),
        ("compound", "partition", 1.0, 1.0),
    ]
    started = time.perf_counter()
    # Pilot v1 showed that direct PPO from a near-uniform actor produced a
    # brittle deterministic argmax concentrated on options 0 and 7. Initialize
    # both variants from the same scripted, local-observation demonstrations;
    # PPO then refines each actor under its own execution features. No final
    # evaluation seed or outcome is used here.
    demonstrations: List[Dict[str, Any]] = []
    demonstration_rng = np.random.RandomState(seed + 17)
    demonstration_episodes = 32
    for episode in range(demonstration_episodes):
        application = "commercial" if episode % 2 == 0 else "humanitarian"
        disruption, communication, privacy, objectives = scenarios[episode % len(scenarios)]
        demonstration_config = ScenarioConfig(
            application=application,
            seed=int(demonstration_rng.randint(1, 1_000_000)),
            horizon=16,
            n_agents=8,
            private_information=privacy,
            objective_misalignment=objectives,
            communication=communication,
            disruption=disruption,
            decision_interval=4,
        )
        demonstration_runner = EpisodeRunner(
            demonstration_config,
            Method.SCRIPTED.value,
            planner=MockPlanner(),
            calibration=calibration,
            monitor_window=int(monitor_settings["window"]),
            monitor_formulation=str(monitor_settings["formulation"]),
        )
        demonstrations.extend(demonstration_runner.run(
            "demonstration-%05d" % episode
        ).trajectory)
    demonstrations_by_option: Dict[int, List[Dict[str, Any]]] = {}
    for row in demonstrations:
        demonstrations_by_option.setdefault(int(row["action"]), []).append(row)
    balanced_demonstrations: List[Dict[str, Any]] = []
    per_option_rows = 256
    for option in sorted(demonstrations_by_option):
        candidates = demonstrations_by_option[option]
        selected = demonstration_rng.choice(
            len(candidates), size=per_option_rows,
            replace=len(candidates) < per_option_rows,
        )
        balanced_demonstrations.extend(candidates[int(index)] for index in selected)
    imitation = policy.behavior_clone(balanced_demonstrations, epochs=12)
    for episode in range(episodes):
        application = "commercial" if episode % 2 == 0 else "humanitarian"
        n_agents = 8
        disruption, communication, privacy, objectives = scenarios[episode % len(scenarios)]
        config = ScenarioConfig(
            application=application, seed=int(rng.randint(1, 1_000_000)), horizon=16, n_agents=n_agents,
            private_information=privacy, objective_misalignment=objectives,
            communication=communication, disruption=disruption, decision_interval=4,
        )
        runner = EpisodeRunner(
            config, method, planner=MockPlanner(), policy=policy,
            calibration=calibration, deterministic_policy=False,
            monitor_window=int(monitor_settings["window"]),
            monitor_formulation=str(monitor_settings["formulation"]),
            trigger_config=trigger_parameters or None,
            trigger_normalizers=trigger_normalizers,
        )
        result = runner.run("train-%s-%05d" % (variant, episode))
        for trajectory_row in result.trajectory:
            trajectory_row["trajectory_id"] = "%s:%05d:%s" % (
                variant, episode, trajectory_row["agent_id"]
            )
        pending.extend(result.trajectory)
        row: Dict[str, Any] = {
            "episode": episode, "application": application, "scenario": result.scenario,
            "primary_outcome": result.metrics["primary_outcome"], "reward_sum": sum(t["reward"] for t in result.trajectory),
            "messages": result.metrics["messages"],
            "total_communication_messages": result.metrics["total_communication_messages"],
            "trigger_activations": result.metrics.get("trigger_activations", 0),
            "failed_actions": result.agent_metrics["failed_actions"],
        }
        if len(pending) >= 512 or episode == episodes - 1:
            losses = policy.update(pending)
            row.update({"ppo_" + key: value for key, value in losses.items()})
            anchor = policy.behavior_clone(balanced_demonstrations, epochs=1)
            row.update({
                "imitation_anchor_loss": anchor["loss"],
                "imitation_anchor_accuracy": anchor["accuracy"],
            })
            row["ppo_rows"] = len(pending)
            pending = []
        history.append(row)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in history), encoding="utf-8")
    metadata = {
        "variant": variant,
        "training_seed": seed,
        "episodes": episodes,
        "planner": "deterministic mock-v2",
        "training_method": "PPO, staged planner training",
        "trigger_configuration": trigger_record if variant == "doet_rl" else None,
        "trigger_normalizer_source": (
            {
                "path": str(trigger_normalizers_path),
                "sha256": sha256_file(trigger_normalizers_path),
                "key": str(trigger_record.get(
                    "normalizers_key", "normalizers"
                )),
            }
            if variant == "doet_rl" and trigger_normalizers_path is not None
            else (
                {"embedded": True}
                if variant == "doet_rl" and trigger_normalizers is not None
                else None
            )
        ),
        "offline_initialization": {
            "method": "behavior cloning from scripted local-observation trajectories",
            "demonstration_episodes": demonstration_episodes,
            "demonstration_rows": int(imitation["rows"]),
            "demonstration_options": sorted(demonstrations_by_option),
            "balanced_rows_per_option": per_option_rows,
            "epochs": int(imitation["epochs"]),
            "final_minibatch_accuracy_mean": imitation["accuracy"],
            "loss_mean": imitation["loss"],
            "seed": seed + 17,
        },
        "execution_features": (
            "24 local features; distributed entropy/free-energy fields present"
            if variant in ("thermo", "doet_rl") else
            "24 local features; entropy/free-energy fields zeroed"
        ),
        "wall_clock_seconds": time.perf_counter() - started,
        "final_window_primary_mean": float(np.mean([row["primary_outcome"] for row in history[-10:]])),
        "generated_at": utc_now(),
    }
    policy.save(output_path, metadata)
    metadata["checkpoint_sha256"] = sha256_file(output_path)
    return metadata


def freeze_protocol(root: Path, output: Path, files: Sequence[Path]) -> Dict[str, Any]:
    if output.exists():
        raise FileExistsError("protocol freeze already exists: %s" % output)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("cannot freeze missing files: %s" % ", ".join(missing))
    records = {str(path): sha256_file(path) for path in files}
    record = {
        "status": "frozen",
        "frozen_at": utc_now(),
        "source_checksum": source_checksum(root),
        "git_provenance": git_provenance(root),
        "files": records,
        "rule": "No listed code/config/prompt/calibration/checkpoint may change after the first main episode. Any violation invalidates and restarts main evaluation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def verify_protocol(root: Path, freeze_path: Path) -> Dict[str, Any]:
    """Fail closed if any frozen execution artifact has changed or disappeared."""

    root = root.resolve()
    if not freeze_path.exists():
        raise FileNotFoundError("protocol freeze does not exist: %s" % freeze_path)
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if record.get("status") != "frozen" or not isinstance(record.get("files"), dict):
        raise ValueError("invalid protocol freeze record: %s" % freeze_path)
    mismatches: Dict[str, Dict[str, Optional[str]]] = {}
    for raw_path, expected in record["files"].items():
        path = Path(raw_path)
        candidate = path if path.is_absolute() else root / path
        observed = sha256_file(candidate) if candidate.is_file() else None
        if observed != expected:
            mismatches[raw_path] = {"expected": expected, "observed": observed}
    observed_source = source_checksum(root)
    if observed_source != record.get("source_checksum"):
        mismatches["<source_checksum>"] = {
            "expected": record.get("source_checksum"),
            "observed": observed_source,
        }
    if mismatches:
        raise RuntimeError(
            "frozen protocol verification failed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "verified",
        "freeze": str(freeze_path),
        "frozen_at": record.get("frozen_at"),
        "source_checksum": observed_source,
        "files_verified": len(record["files"]),
    }
