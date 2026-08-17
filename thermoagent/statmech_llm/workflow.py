"""External-artifact workflow for the compact V10 study."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from thermoagent.statmech.exact import entropy_production_rate, exact_transition_matrix, stationary_distribution
from thermoagent.statmech.model import ModelParameters, MultiplexModel, topology_adjacency

from .estimators import (
    exact_cycle_entropy_production,
    known_three_state_cycle,
    stationary_chain_sample,
    transition_counts,
    transition_pair_irreversibility,
)
from .theory import directed_family, exact_family_point, finite_difference_kernel_derivative, perturbative_entropy_production
from .trajectories import simulate_stationary_pathwise


V9_COMMIT = "8e8315d25684a1c582c6a7b46fbb5786bc3f0557"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_root() -> Path:
    configured = os.environ.get("THERMO_V10_ARTIFACT_ROOT")
    if configured:
        return Path(configured).resolve()
    preferred = Path("/workspace/ThermoAgent-v10-artifacts")
    if preferred.parent.exists() and os.access(str(preferred.parent), os.W_OK):
        return preferred
    return Path("/tmp/ThermoAgent-v10-artifacts")


def scratch_root() -> Path:
    configured = os.environ.get("THERMO_V10_SCRATCH_ROOT")
    return Path(configured).resolve() if configured else Path("/tmp/ThermoAgent-v10-scratch")


def clean_export_root() -> Path:
    configured = os.environ.get("THERMO_V10_EXPORT_ROOT")
    if configured:
        return Path(configured).resolve()
    preferred = Path("/workspace/ThermoAgent-JSTAT-v10-clean-export")
    if preferred.parent.exists() and os.access(str(preferred.parent), os.W_OK):
        return preferred
    return Path("/tmp/ThermoAgent-JSTAT-v10-clean-export")


def load_yaml(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_checksum(repository: Path) -> str:
    """Hash the frozen scientific implementation, tests, and protocol only."""

    paths: List[Path] = []
    for pattern in (
        "thermoagent/statmech_llm/*.py",
        "tests/statmech_v10/*.py",
        "configs/statmech_v10/*.yaml",
        "scripts/run-statmech-v10-*.sh",
    ):
        paths.extend(repository.glob(pattern))
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(repository))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(destination))


def _atomic_csv(rows: Sequence[Mapping[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, lineterminator="\n")
    os.replace(str(temporary), str(destination))


@contextmanager
def stage_lock(name: str) -> Iterator[None]:
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / (".%s.lock" % name)
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, ("pid=%d started=%s\n" % (os.getpid(), utc_now())).encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def exact_topology(n_agents: int, family: str, seed: int) -> np.ndarray:
    if family == "path":
        graph = nx.path_graph(int(n_agents))
    elif family == "star":
        graph = nx.star_graph(int(n_agents) - 1)
    elif family == "complete":
        graph = nx.complete_graph(int(n_agents))
    else:
        return topology_adjacency(int(n_agents), family, int(seed))
    return nx.to_numpy_array(graph, dtype=float)


def audit_v9(repository: Path) -> List[Dict[str, object]]:
    """Evidence-based audit without modifying the frozen V9 namespace."""

    rows = [
        {
            "item": "remote provenance",
            "finding": V9_COMMIT,
            "evidence": "verified origin/statistical-mechanics-agentic-systems-v9 before branching",
            "v10_treatment": "record correct committed/pushed provenance only in V10",
        },
        {
            "item": "exact update convention",
            "finding": "one of 2N belief/action variables chosen uniformly; self transitions retained",
            "evidence": "thermoagent/statmech/exact.py:exact_transition_matrix",
            "v10_treatment": "report EPR per attempted variable update and explicit sweep conversions",
        },
        {
            "item": "exact EPR normalization",
            "finding": "Schnakenberg discrete-time stationary EPR per attempted variable update",
            "evidence": "thermoagent/statmech/exact.py:entropy_production_rate",
            "v10_treatment": "separate per-update, per-agent-sweep, and per-sweep units",
        },
        {
            "item": "large-system statistic",
            "finding": "stationary mean local log-rate ratio; not a dense exact current calculation",
            "evidence": "thermoagent/statmech/simulate.py:simulate_stationary",
            "v10_treatment": "call pathwise stationary irreversibility and validate against exact kernels",
        },
        {
            "item": "nonreciprocity family",
            "finding": "pair weights 1+alpha and 1-alpha; support and pairwise total weight fixed",
            "evidence": "thermoagent/statmech/simulate.py:directed_communication",
            "v10_treatment": "express explicitly as A_s + alpha A_a and report spectral diagnostics",
        },
        {
            "item": "V9 quadratic evidence",
            "finding": "descriptive exact N=3 pattern; no perturbative coefficient was derived",
            "evidence": "results/statmech_agentic_v9/tables/entropy_production_summary.csv",
            "v10_treatment": "derive C from the stationary response and compare pointwise",
        },
        {
            "item": "LLM realization",
            "finding": "absent from V9 primary model",
            "evidence": "thermoagent/statmech and paper/jstat_v9",
            "v10_treatment": "new independent-context Qwen architecture; do not relabel stochastic spins as LLMs",
        },
    ]
    destination = artifact_root() / "audit" / "v9_audit.csv"
    _atomic_csv(rows, destination)
    return rows


def _reference_model(settings: Mapping[str, object], topology_seed: int = 0) -> Tuple[MultiplexModel, np.ndarray]:
    n_agents = int(settings["n_agents"])
    adjacency = exact_topology(n_agents, str(settings["topology"]), int(topology_seed))
    parameters = ModelParameters(
        float(settings["belief_coupling"]),
        float(settings["action_coupling"]),
        float(settings["belief_action_coupling"]),
        float(settings["temperature"]),
    )
    private = np.asarray(settings.get("private_fields", [0.0] * n_agents), dtype=float)
    return MultiplexModel(adjacency, adjacency, parameters, private_fields=private), adjacency


def run_development(repository: Path) -> Dict[str, object]:
    configuration = load_yaml(repository / "configs/statmech_v10/development.yaml")
    started = time.perf_counter()
    with stage_lock("development"):
        audit_rows = audit_v9(repository)
        settings = configuration["analytical_reference"]
        model, base = _reference_model(settings)
        exact_rows: List[Dict[str, object]] = []
        maximum_derivative_error = 0.0
        for orientation_seed in settings["orientation_seeds"]:
            family = directed_family(base, int(orientation_seed))
            perturbation = perturbative_entropy_production(model, family.antisymmetric)
            numerical = finite_difference_kernel_derivative(model, family.antisymmetric, 2e-6)
            maximum_derivative_error = max(
                maximum_derivative_error,
                float(np.max(np.abs(numerical - perturbation.kernel_derivative))),
            )
            for alpha in settings["alphas"]:
                point = exact_family_point(model, family.antisymmetric, float(alpha))
                exact_rows.append(
                    {
                        "orientation_seed": int(orientation_seed),
                        "coefficient_prediction": perturbation.coefficient_per_update,
                        **point,
                    }
                )
        trajectory_rows: List[Dict[str, object]] = []
        profile = configuration["trajectory_profile"]
        for n_agents in profile["agent_counts"]:
            adjacency = topology_adjacency(int(n_agents), "ring", 10200 + int(n_agents))
            family = directed_family(adjacency, 10300 + int(n_agents))
            parameters = ModelParameters(0.70, 0.30, 0.40, float(profile["temperature"]))
            for seed in profile["seeds"]:
                began = time.perf_counter()
                metrics, _ = simulate_stationary_pathwise(
                    family.at(float(profile["alpha"])),
                    adjacency,
                    parameters,
                    int(seed) + int(n_agents),
                    int(profile["burn_in_sweeps"]),
                    int(profile["sample_sweeps"]),
                )
                trajectory_rows.append(
                    {
                        "n_agents": int(n_agents),
                        "seed": int(seed),
                        "runtime_seconds": time.perf_counter() - began,
                        **metrics,
                    }
                )
        output = artifact_root() / "development"
        _atomic_csv(exact_rows, output / "exact_pilot.csv")
        _atomic_csv(trajectory_rows, output / "trajectory_profile.csv")
        summary = {
            "protocol_version": configuration["protocol_version"],
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "audit_rows": len(audit_rows),
            "exact_rows": len(exact_rows),
            "trajectory_rows": len(trajectory_rows),
            "maximum_kernel_derivative_error": maximum_derivative_error,
            "maximum_stationarity_residual": max(float(row["stationarity_residual"]) for row in exact_rows),
            "artifact_root": str(artifact_root()),
            "formal_outcomes_not_examined": True,
        }
        _atomic_json(summary, output / "summary.json")
        return summary


def freeze(repository: Path) -> Dict[str, object]:
    protocol = repository / "configs/statmech_v10/protocol.yaml"
    if not protocol.exists():
        raise FileNotFoundError("formal protocol does not exist")
    manifest = {
        "protocol_version": load_yaml(protocol)["protocol_version"],
        "v9_parent_commit": V9_COMMIT,
        "protocol_sha256": sha256_file(protocol),
        "scientific_source_sha256": source_checksum(repository),
        "frozen_at": utc_now(),
        "git_commit": None,
        "reason_git_commit_absent": "User required all V10 work to remain uncommitted.",
        "llm_stage_status": "designed but not unlocked until existing authorized RunPod is reachable and pilot passes",
    }
    _atomic_json(manifest, artifact_root() / "formal" / "freeze_manifest.json")
    return manifest


def environment_manifest() -> Dict[str, object]:
    versions: Dict[str, object] = {}
    for module_name in ("numpy", "scipy", "pandas", "matplotlib", "networkx", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "available")
        except ImportError:
            versions[module_name] = None
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": versions,
        "artifact_root": str(artifact_root()),
        "gpu_execution": "local environment has no GPU; Qwen pilots used the existing RunPod through its working SSH proxy",
    }


def _write_failure(stage: str, error: BaseException) -> None:
    _atomic_json(
        {
            "stage": stage,
            "failed_at": utc_now(),
            "exception_type": type(error).__name__,
            "message": str(error),
        },
        artifact_root() / "failures" / (stage + ".json"),
    )


def _formal_model(
    adjacency: np.ndarray,
    temperature: float,
    belief_action_coupling: float,
    private_fields: np.ndarray,
) -> MultiplexModel:
    parameters = ModelParameters(
        belief_coupling=0.70,
        action_coupling=0.30,
        belief_action_coupling=float(belief_action_coupling),
        temperature=float(temperature),
    )
    return MultiplexModel(
        adjacency,
        adjacency,
        parameters,
        private_fields=np.asarray(private_fields, dtype=float),
    )


def _formal_exact_primary(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["exact_primary"]
    model_settings = protocol["model"]
    n_agents = int(settings["n_agents"])
    base = exact_topology(n_agents, str(settings["topology"]), 40001)
    model = _formal_model(
        base,
        float(settings["temperature"]),
        float(model_settings["belief_action_coupling"]),
        np.asarray(model_settings["equilibrium_private_fields"], dtype=float),
    )
    rows: List[Dict[str, object]] = []
    orientation_signatures = set()
    for orientation_seed in settings["orientation_seeds"]:
        family = directed_family(base, int(orientation_seed))
        signature = tuple(family.antisymmetric.ravel().tolist())
        if signature in orientation_signatures:
            raise RuntimeError("exact-primary orientation seeds are not unique")
        orientation_signatures.add(signature)
        perturbation = perturbative_entropy_production(model, family.antisymmetric)
        numerical_derivative = finite_difference_kernel_derivative(model, family.antisymmetric, 2e-6)
        diagnostics = family.diagnostics()
        for alpha in settings["alphas"]:
            point = exact_family_point(model, family.antisymmetric, float(alpha))
            rows.append(
                {
                    "n_agents": n_agents,
                    "topology": str(settings["topology"]),
                    "temperature": float(settings["temperature"]),
                    "belief_action_coupling": float(model_settings["belief_action_coupling"]),
                    "orientation_seed": int(orientation_seed),
                    "coefficient_prediction_per_update": perturbation.coefficient_per_update,
                    "belief_coefficient_prediction": perturbation.belief_coefficient_per_update,
                    "action_coefficient_prediction": perturbation.action_coefficient_per_update,
                    "kernel_derivative_maximum_error": float(
                        np.max(np.abs(numerical_derivative - perturbation.kernel_derivative))
                    ),
                    "stationary_response_residual": perturbation.stationary_response_residual,
                    **diagnostics,
                    **point,
                }
            )
    return rows


def _formal_coefficient_grid(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["coefficient_grid"]
    rows: List[Dict[str, object]] = []
    n_agents = int(settings["n_agents"])
    private = np.linspace(0.20, -0.10, n_agents)
    for topology_index, topology in enumerate(settings["topologies"]):
        base = exact_topology(n_agents, str(topology), 41100 + topology_index)
        graph = nx.from_numpy_array(base)
        graph_metrics = {
            "mean_degree": float(np.mean(np.sum(base > 0.0, axis=1))),
            "degree_variance": float(np.var(np.sum(base > 0.0, axis=1))),
            "clustering": float(nx.average_clustering(graph)),
            "average_shortest_path": float(nx.average_shortest_path_length(graph)),
        }
        for temperature in settings["temperatures"]:
            for coupling in settings["belief_action_couplings"]:
                model = _formal_model(base, float(temperature), float(coupling), private)
                for orientation_seed in settings["orientation_seeds"]:
                    family = directed_family(base, int(orientation_seed) + 100 * topology_index)
                    perturbation = perturbative_entropy_production(model, family.antisymmetric)
                    rows.append(
                        {
                            "n_agents": n_agents,
                            "topology": str(topology),
                            "temperature": float(temperature),
                            "belief_action_coupling": float(coupling),
                            "orientation_seed": int(orientation_seed),
                            "coefficient_per_update": perturbation.coefficient_per_update,
                            "belief_coefficient_per_update": perturbation.belief_coefficient_per_update,
                            "action_coefficient_per_update": perturbation.action_coefficient_per_update,
                            "stationary_response_residual": perturbation.stationary_response_residual,
                            **graph_metrics,
                            **family.diagnostics(),
                        }
                    )
    return rows


def _formal_exact_size(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["exact_size_grid"]
    rows: List[Dict[str, object]] = []
    for n_agents in settings["agent_counts"]:
        base = exact_topology(int(n_agents), str(settings["topology"]), 42100 + int(n_agents))
        private = np.linspace(0.15, -0.10, int(n_agents))
        for temperature in settings["temperatures"]:
            model = _formal_model(
                base,
                float(temperature),
                float(settings["belief_action_coupling"]),
                private,
            )
            for orientation_seed in settings["orientation_seeds"]:
                family = directed_family(base, int(orientation_seed) + 10 * int(n_agents))
                perturbation = perturbative_entropy_production(model, family.antisymmetric)
                for alpha in settings["alphas"]:
                    point = exact_family_point(model, family.antisymmetric, float(alpha))
                    rows.append(
                        {
                            "n_agents": int(n_agents),
                            "temperature": float(temperature),
                            "orientation_seed": int(orientation_seed),
                            "coefficient_prediction_per_update": perturbation.coefficient_per_update,
                            **point,
                        }
                    )
    return rows


def _formal_trajectory_grid(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["trajectory_grid"]
    rows: List[Dict[str, object]] = []
    topology_codes = {name: index for index, name in enumerate(settings["topologies"])}
    for n_agents in settings["agent_counts"]:
        eligible_topologies = list(settings["topologies"])
        if int(n_agents) > int(settings["full_topology_max_agents"]):
            eligible_topologies = ["ring"]
        for topology in eligible_topologies:
            topology_index = topology_codes[str(topology)]
            for seed in settings["seeds"]:
                graph_seed = int(seed) + 1000 * int(n_agents) + 100000 * topology_index
                base = topology_adjacency(int(n_agents), str(topology), graph_seed)
                family = directed_family(base, graph_seed + 71)
                graph = nx.from_numpy_array(base)
                private_rng = np.random.default_rng(graph_seed + 83)
                private = private_rng.normal(0.0, 0.12, int(n_agents))
                graph_metrics = {
                    "mean_degree": float(np.mean(np.sum(base > 0.0, axis=1))),
                    "degree_variance": float(np.var(np.sum(base > 0.0, axis=1))),
                    "clustering": float(nx.average_clustering(graph)),
                    "graph_seed": graph_seed,
                }
                for temperature in settings["temperatures"]:
                    parameters = ModelParameters(0.70, 0.30, 0.40, float(temperature))
                    for alpha in settings["alphas"]:
                        simulation_seed = graph_seed + int(float(temperature) * 1000) + int(float(alpha) * 10000)
                        began = time.perf_counter()
                        metrics, blocks = simulate_stationary_pathwise(
                            family.at(float(alpha)),
                            base,
                            parameters,
                            simulation_seed,
                            int(settings["burn_in_sweeps"]),
                            int(settings["sample_sweeps"]),
                            int(settings["block_sweeps"]),
                            private_fields=private,
                        )
                        rows.append(
                            {
                                "n_agents": int(n_agents),
                                "topology": str(topology),
                                "temperature": float(temperature),
                                "alpha": float(alpha),
                                "replicate_seed": int(seed),
                                "simulation_seed": simulation_seed,
                                "runtime_seconds": time.perf_counter() - began,
                                "block_mean": float(np.mean(blocks)),
                                "block_standard_error": float(np.std(blocks, ddof=1) / np.sqrt(blocks.size))
                                if blocks.size > 1
                                else 0.0,
                                **graph_metrics,
                                **family.diagnostics(),
                                **metrics,
                            }
                        )
    return rows


def _formal_estimator_validation(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["synthetic_estimator_validation"]
    clockwise = float(settings["clockwise"])
    counterclockwise = float(settings["counterclockwise"])
    kernel = known_three_state_cycle(clockwise, counterclockwise, float(settings["stay"]))
    exact = exact_cycle_entropy_production(clockwise, counterclockwise)
    rows: List[Dict[str, object]] = []
    for steps in settings["trajectory_steps"]:
        for seed in settings["seeds"]:
            trajectory = stationary_chain_sample(kernel, int(steps), int(seed) + int(steps))
            counts = transition_counts(trajectory, 3)
            for pseudocount in settings["pseudocounts"]:
                estimate = transition_pair_irreversibility(counts, float(pseudocount))
                rows.append(
                    {
                        "steps": int(steps),
                        "seed": int(seed),
                        "pseudocount": float(pseudocount),
                        "exact_epr_per_transition": exact,
                        "estimated_irreversibility_per_transition": estimate.estimate_per_transition,
                        "absolute_error": abs(estimate.estimate_per_transition - exact),
                    }
                )
    reciprocal = known_three_state_cycle(0.40, 0.40, 0.20)
    for seed in settings["seeds"]:
        trajectory = stationary_chain_sample(reciprocal, 100000, int(seed) + 990000)
        estimate = transition_pair_irreversibility(transition_counts(trajectory, 3), 0.5)
        rows.append(
            {
                "steps": 100000,
                "seed": int(seed),
                "pseudocount": 0.5,
                "exact_epr_per_transition": 0.0,
                "estimated_irreversibility_per_transition": estimate.estimate_per_transition,
                "absolute_error": estimate.estimate_per_transition,
            }
        )
    return rows


def run_formal(repository: Path) -> Dict[str, object]:
    """Run the frozen CPU formal study once, with phase-level resumption."""

    protocol_path = repository / "configs/statmech_v10/protocol.yaml"
    protocol = load_yaml(protocol_path)
    freeze_path = artifact_root() / "formal" / "freeze_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("freeze manifest must exist before formal execution")
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze_manifest["protocol_sha256"] != sha256_file(protocol_path):
        raise RuntimeError("protocol changed after freeze")
    if freeze_manifest["scientific_source_sha256"] != source_checksum(repository):
        raise RuntimeError("scientific source changed after freeze")
    output = artifact_root() / "formal"
    completion = output / "completion_manifest.json"
    expected_files = {
        "exact_primary": output / "exact_primary.csv",
        "coefficient_grid": output / "coefficient_grid.csv",
        "exact_size_grid": output / "exact_size_grid.csv",
        "trajectory_grid": output / "trajectory_grid.csv",
        "estimator_validation": output / "estimator_validation.csv",
    }
    if completion.exists() and all(path.exists() for path in expected_files.values()):
        return json.loads(completion.read_text(encoding="utf-8"))
    started_at = utc_now()
    began = time.perf_counter()
    with stage_lock("formal"):
        _atomic_json(
            {"state": "running", "started_at": started_at, "protocol_sha256": sha256_file(protocol_path)},
            output / "status.json",
        )
        try:
            runners = {
                "exact_primary": _formal_exact_primary,
                "coefficient_grid": _formal_coefficient_grid,
                "exact_size_grid": _formal_exact_size,
                "trajectory_grid": _formal_trajectory_grid,
                "estimator_validation": _formal_estimator_validation,
            }
            phase_counts: Dict[str, int] = {}
            for phase, destination in expected_files.items():
                if destination.exists():
                    phase_counts[phase] = int(len(pd.read_csv(destination)))
                    continue
                rows = runners[phase](protocol)
                _atomic_csv(rows, destination)
                phase_counts[phase] = len(rows)
                _atomic_json(
                    {
                        "state": "running",
                        "last_completed_phase": phase,
                        "phase_counts": phase_counts,
                        "elapsed_seconds": time.perf_counter() - began,
                    },
                    output / "status.json",
                )
            manifest = {
                "state": "complete",
                "started_at": started_at,
                "completed_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - began,
                "protocol_sha256": sha256_file(protocol_path),
                "scientific_source_sha256": source_checksum(repository),
                "phase_counts": phase_counts,
                "file_sha256": {name: sha256_file(path) for name, path in expected_files.items()},
                "llm_execution": "not run",
            }
            _atomic_json(manifest, completion)
            _atomic_json(manifest, output / "status.json")
            _atomic_json(environment_manifest(), output / "environment.json")
            return manifest
        except BaseException as error:
            _write_failure("formal", error)
            raise


def _bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap requires a nonempty vector")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, data.size, size=(int(replicates), data.size))
    means = np.mean(data[draws], axis=1)
    return float(np.mean(data)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summary_rows(
    frame: pd.DataFrame,
    groups: Sequence[str],
    value: str,
    replicates: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    grouped = frame.groupby(list(groups), sort=True, dropna=False)
    for index, (keys, part) in enumerate(grouped):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        mean, low, high = _bootstrap_mean(part[value].to_numpy(float), replicates, seed + index)
        row = {name: key for name, key in zip(groups, key_tuple)}
        row.update(
            {
                "independent_n": int(len(part)),
                "%s_mean" % value: mean,
                "%s_ci_low" % value: low,
                "%s_ci_high" % value: high,
            }
        )
        rows.append(row)
    return rows


def _fit_curve_models(primary: pd.DataFrame, maximum_alpha: float) -> List[Dict[str, object]]:
    data = primary[(primary["alpha"] > 0.0) & (primary["alpha"] <= maximum_alpha)].copy()
    models = {
        "linear_zero_intercept": lambda alpha: np.column_stack([alpha]),
        "quadratic_zero_intercept": lambda alpha: np.column_stack([alpha ** 2]),
        "linear_plus_quadratic_zero_intercept": lambda alpha: np.column_stack([alpha, alpha ** 2]),
    }
    rows: List[Dict[str, object]] = []
    seeds = sorted(data["orientation_seed"].unique())
    for model_name, design_function in models.items():
        errors = []
        coefficients = []
        for held_out in seeds:
            training = data[data["orientation_seed"] != held_out]
            testing = data[data["orientation_seed"] == held_out]
            design = design_function(training["alpha"].to_numpy(float))
            coefficient, _, _, _ = np.linalg.lstsq(design, training["total_per_update"].to_numpy(float), rcond=None)
            prediction = design_function(testing["alpha"].to_numpy(float)).dot(coefficient)
            errors.extend((prediction - testing["total_per_update"].to_numpy(float)).tolist())
            coefficients.append(coefficient)
        full_design = design_function(data["alpha"].to_numpy(float))
        full_coefficient, _, _, _ = np.linalg.lstsq(
            full_design, data["total_per_update"].to_numpy(float), rcond=None
        )
        coefficient_values = [float(value) for value in full_coefficient]
        rows.append(
            {
                "model": model_name,
                "low_alpha_maximum": float(maximum_alpha),
                "independent_orientations": len(seeds),
                "held_out_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
                "coefficient_linear": coefficient_values[0]
                if model_name != "quadratic_zero_intercept"
                else 0.0,
                "coefficient_quadratic": coefficient_values[-1]
                if model_name != "linear_zero_intercept"
                else 0.0,
            }
        )
    return rows


def _current_source_data(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["exact_primary"]
    model_settings = protocol["model"]
    n_agents = int(settings["n_agents"])
    base = exact_topology(n_agents, str(settings["topology"]), 40001)
    reference = _formal_model(
        base,
        float(settings["temperature"]),
        float(model_settings["belief_action_coupling"]),
        np.asarray(model_settings["equilibrium_private_fields"], dtype=float),
    )
    family = directed_family(base, int(settings["orientation_seeds"][0]))
    rows: List[Dict[str, object]] = []
    for alpha in (0.0, 0.5):
        model = MultiplexModel(
            family.at(alpha),
            reference.dependency,
            reference.parameters,
            reference.private_fields,
            reference.task_fields,
        )
        kernel = exact_transition_matrix(model)
        stationary = stationary_distribution(kernel)
        flux = stationary[:, None] * kernel
        current = flux - flux.T
        candidates = []
        for source in range(kernel.shape[0]):
            for destination in range(source + 1, kernel.shape[0]):
                if abs(current[source, destination]) > 1e-16:
                    candidates.append((abs(current[source, destination]), source, destination))
        for rank, (_, source, destination) in enumerate(sorted(candidates, reverse=True)[:18], start=1):
            rows.append(
                {
                    "alpha": alpha,
                    "rank": rank,
                    "source_state": source,
                    "destination_state": destination,
                    "current": float(current[source, destination]),
                    "absolute_current": float(abs(current[source, destination])),
                }
            )
    if not any(row["alpha"] == 0.0 for row in rows):
        rows.append(
            {
                "alpha": 0.0,
                "rank": 1,
                "source_state": 0,
                "destination_state": 0,
                "current": 0.0,
                "absolute_current": 0.0,
            }
        )
    return rows


def analyze(repository: Path) -> Dict[str, object]:
    """Regenerate every compact V10 table from external formal outputs."""

    protocol = load_yaml(repository / "configs/statmech_v10/protocol.yaml")
    formal = artifact_root() / "formal"
    completion_path = formal / "completion_manifest.json"
    if not completion_path.exists():
        raise RuntimeError("formal CPU study is incomplete")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for name, expected in completion["file_sha256"].items():
        if sha256_file(formal / (name + ".csv")) != expected:
            raise RuntimeError("formal artifact checksum mismatch: %s" % name)
    primary = pd.read_csv(formal / "exact_primary.csv")
    coefficients = pd.read_csv(formal / "coefficient_grid.csv")
    exact_size = pd.read_csv(formal / "exact_size_grid.csv")
    trajectories = pd.read_csv(formal / "trajectory_grid.csv")
    estimators = pd.read_csv(formal / "estimator_validation.csv")
    analysis_settings = protocol["analysis"]
    bootstrap_replicates = int(analysis_settings["bootstrap_replicates"])
    bootstrap_seed = int(analysis_settings["bootstrap_seed"])
    results_root = repository / "results/llm_agent_entropy_v10"
    tables = results_root / "tables"
    source_data = results_root / "figures/source_data"

    primary_summary = _summary_rows(
        primary,
        ["alpha"],
        "total_per_update",
        bootstrap_replicates,
        bootstrap_seed,
    )
    # Add the orientation-averaged perturbative prediction to every alpha row.
    predicted_mean, predicted_low, predicted_high = _bootstrap_mean(
        primary.drop_duplicates("orientation_seed")["coefficient_prediction_per_update"].to_numpy(float),
        bootstrap_replicates,
        bootstrap_seed + 101,
    )
    for row in primary_summary:
        row["coefficient_prediction_mean"] = predicted_mean
        row["coefficient_prediction_ci_low"] = predicted_low
        row["coefficient_prediction_ci_high"] = predicted_high
        row["quadratic_prediction_mean"] = predicted_mean * float(row["alpha"]) ** 2
    coefficient_summary = _summary_rows(
        coefficients,
        ["topology", "temperature", "belief_action_coupling"],
        "coefficient_per_update",
        bootstrap_replicates,
        bootstrap_seed + 200,
    )
    exact_size_summary = _summary_rows(
        exact_size,
        ["n_agents", "temperature", "alpha"],
        "total_per_update",
        bootstrap_replicates,
        bootstrap_seed + 300,
    )
    trajectory_summary = _summary_rows(
        trajectories,
        ["n_agents", "topology", "temperature", "alpha"],
        "pathwise_irreversibility_per_update",
        bootstrap_replicates,
        bootstrap_seed + 400,
    )
    curve_models = _fit_curve_models(primary, float(protocol["exact_primary"]["perturbative_comparison_max_alpha"]))
    estimator_summary = _summary_rows(
        estimators,
        ["steps", "pseudocount", "exact_epr_per_transition"],
        "estimated_irreversibility_per_transition",
        bootstrap_replicates,
        bootstrap_seed + 500,
    )
    low_alpha = primary[
        (primary["alpha"] > 0.0)
        & (primary["alpha"] <= float(protocol["exact_primary"]["perturbative_comparison_max_alpha"]))
    ].copy()
    low_alpha["relative_coefficient_error"] = np.abs(
        low_alpha["total_per_update"] / low_alpha["alpha"] ** 2
        - low_alpha["coefficient_prediction_per_update"]
    ) / low_alpha["coefficient_prediction_per_update"]
    maximum_relative_error = float(low_alpha["relative_coefficient_error"].max())
    reciprocal_max = float(primary[primary["alpha"] == 0.0]["total_per_update"].abs().max())
    coefficient_range = [
        float(coefficients["coefficient_per_update"].min()),
        float(coefficients["coefficient_per_update"].max()),
    ]
    spectral_correlation = float(
        np.corrcoef(
            coefficients["coefficient_per_update"].to_numpy(float),
            coefficients["antisymmetric_spectral_norm"].to_numpy(float),
        )[0, 1]
    )
    trajectory_runtime = float(trajectories["runtime_seconds"].sum())
    estimator_100k = estimators[
        (estimators["steps"] == 100000)
        & (estimators["pseudocount"] == 0.5)
        & (estimators["exact_epr_per_transition"] > 0.0)
    ]
    estimator_error = float(estimator_100k["absolute_error"].mean())
    current_rows = _current_source_data(protocol)
    audit = pd.read_csv(artifact_root() / "audit/v9_audit.csv").to_dict(orient="records")

    literature_rows = [
        {
            "work": "Glauber (1963)",
            "closest_feature": "single-spin stochastic Ising dynamics",
            "persistent_agents": "no",
            "belief_action_coupling": "no",
            "directed_messages": "no",
            "entropy_production": "no",
            "actual_llm_agents": "no",
            "v10_difference": "two coupled local variables and a controlled nonreciprocal extension",
        },
        {
            "work": "Schnakenberg (1976)",
            "closest_feature": "network currents and entropy production",
            "persistent_agents": "no",
            "belief_action_coupling": "no",
            "directed_messages": "generic transitions",
            "entropy_production": "yes",
            "actual_llm_agents": "no",
            "v10_difference": "applies current theory to a decentralized belief-action kernel",
        },
        {
            "work": "Aguilera et al. (2021)",
            "closest_feature": "asymmetric kinetic Ising networks",
            "persistent_agents": "binary units",
            "belief_action_coupling": "no separate layers",
            "directed_messages": "asymmetric couplings",
            "entropy_production": "yes",
            "actual_llm_agents": "no",
            "v10_difference": "near-reciprocal coefficient and empirical LLM-agent mapping",
        },
        {
            "work": "Fruchart et al. (2021)",
            "closest_feature": "nonreciprocal collective dynamics",
            "persistent_agents": "generic fields/robots",
            "belief_action_coupling": "no",
            "directed_messages": "nonreciprocal interactions",
            "entropy_production": "not primary",
            "actual_llm_agents": "no",
            "v10_difference": "finite Markov currents near a reversible reference",
        },
        {
            "work": "Di Carlo (2025)",
            "closest_feature": "quadratic near-reciprocal EPR in a metric kinetic Ising model",
            "persistent_agents": "binary spins",
            "belief_action_coupling": "no separate layers",
            "directed_messages": "nonreciprocal nearest-neighbor coupling",
            "entropy_production": "yes",
            "actual_llm_agents": "no",
            "v10_difference": "general finite-kernel coefficient with stationary-response and layer decomposition",
        },
        {
            "work": "Roldan and Parrondo (2012)",
            "closest_feature": "time-reversal KL from partial stationary trajectories",
            "persistent_agents": "no",
            "belief_action_coupling": "no",
            "directed_messages": "no",
            "entropy_production": "coarse-grained lower bound",
            "actual_llm_agents": "no",
            "v10_difference": "applies qualified coarse irreversibility estimators to LLM belief-action sequences",
        },
        {
            "work": "De Nobili (2026 preprint)",
            "closest_feature": "LLM population with neighbor-conditioned binary choices",
            "persistent_agents": "limited",
            "belief_action_coupling": "no distinct committed action",
            "directed_messages": "neighbor states in prompts",
            "entropy_production": "no",
            "actual_llm_agents": "yes",
            "v10_difference": "private memory, separate belief/action, typed actions, and time-reversal analysis",
        },
    ]
    _atomic_csv(primary_summary, tables / "quadratic_onset.csv")
    _atomic_csv(curve_models, tables / "curve_model_comparison.csv")
    _atomic_csv(coefficient_summary, tables / "coefficient_topology_temperature.csv")
    _atomic_csv(exact_size_summary, tables / "exact_size_scaling.csv")
    _atomic_csv(trajectory_summary, tables / "trajectory_scaling.csv")
    _atomic_csv(estimator_summary, tables / "estimator_validation.csv")
    _atomic_csv(audit, tables / "v9_audit.csv")
    _atomic_csv(literature_rows, tables / "literature_comparison.csv")
    _atomic_csv(current_rows, source_data / "figure_05_probability_currents.csv")
    for filename, rows in (
        ("figure_02_quadratic_onset.csv", primary_summary),
        ("figure_03_coefficient.csv", coefficient_summary),
        ("figure_04_size_scaling.csv", trajectory_summary),
        ("figure_06_temperature_nonreciprocity.csv", trajectory_summary),
    ):
        _atomic_csv(rows, source_data / filename)
    principal = {
        "protocol_version": protocol["protocol_version"],
        "v9_parent_commit": V9_COMMIT,
        "formal_source_sha256": completion["scientific_source_sha256"],
        "protocol_sha256": completion["protocol_sha256"],
        "exact_reciprocal_maximum_epr_per_update": reciprocal_max,
        "perturbative_maximum_relative_error_alpha_le_0_02": maximum_relative_error,
        "perturbative_mean_coefficient_per_update": predicted_mean,
        "perturbative_coefficient_ci": [predicted_low, predicted_high],
        "coefficient_range_over_grid": coefficient_range,
        "coefficient_spectral_norm_correlation_exploratory": spectral_correlation,
        "synthetic_estimator_mean_absolute_error_100k": estimator_error,
        "formal_row_counts": completion["phase_counts"],
        "trajectory_cpu_seconds": trajectory_runtime,
        "llm_stage": {
            "status": "not_run",
            "reason": "existing authorized RunPod SSH endpoint refused the connection",
            "qwen_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "gpu_hours": 0.0,
            "estimated_incremental_cost_usd": 0.0,
        },
        "claims": {
            "H1_reversible_reference": reciprocal_max
            <= float(analysis_settings["exact_reciprocal_tolerance"]),
            "H2_quadratic_analytical_onset": maximum_relative_error
            <= float(analysis_settings["coefficient_relative_error_tolerance"]),
            "H3_topological_dependence": "supported numerically; descriptive/exploratory",
            "H4_temperature_dependence": "estimated numerically",
            "H5_llm_local_policy": "not tested",
            "H6_llm_time_reversal_asymmetry": "not tested",
            "H7_llm_replication": "not tested",
        },
        "generated_at": utc_now(),
    }
    _atomic_json(principal, tables / "principal_results.json")
    return principal
