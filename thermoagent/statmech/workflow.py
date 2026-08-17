"""External-artifact workflow for the compact V9 statistical-mechanics study."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .driven import run_application_mapping
from .exact import (
    detailed_balance_residual,
    distribution_distances,
    empirical_distribution,
    entropy_production_rate,
    exact_transition_matrix,
    gibbs_distribution,
    stationary_distribution,
    verify_free_energy_identity,
)
from .model import Microstate, ModelParameters, MultiplexModel, topology_adjacency
from .simulate import (
    directed_communication,
    make_model_layers,
    run_parameter_cell,
    simulate_field_hysteresis,
    simulate_stationary,
)


def artifact_root() -> Path:
    configured = os.environ.get("THERMO_V9_ARTIFACT_ROOT")
    if configured:
        return Path(configured).resolve()
    preferred = Path("/workspace/ThermoAgent-v9-artifacts")
    if preferred.parent.exists() and os.access(str(preferred.parent), os.W_OK):
        return preferred
    return Path("/tmp/ThermoAgent-v9-artifacts")


def load_protocol(repository: Path) -> Dict[str, object]:
    with (repository / "configs/statmech_v9/formal.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _atomic_csv(rows: Sequence[Dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # pandas 1.3 uses the pre-2.0 spelling; the explicit value prevents CRLF.
    pd.DataFrame(rows).to_csv(temporary, index=False, line_terminator="\n")
    os.replace(str(temporary), str(destination))


def _atomic_json(payload: Dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(destination))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_checksum(repository: Path) -> str:
    paths: List[Path] = []
    for pattern in (
        "thermoagent/statmech/*.py",
        "tests/statmech_v9/*.py",
        "configs/statmech_v9/*.yaml",
        "scripts/run-statmech-v9-*.sh",
    ):
        paths.extend(repository.glob(pattern))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_manifest(repository: Path) -> Dict[str, object]:
    protocol_path = repository / "configs/statmech_v9/formal.yaml"
    payload = {
        "protocol_version": load_protocol(repository)["protocol_version"],
        "parent_commit": "b86f97fa0940f11cb366c809e0e46fa888dfaba1",
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": source_checksum(repository),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_intentionally_absent": True,
        "reason": "User required an uncommitted human-review worktree.",
    }
    _atomic_json(payload, artifact_root() / "formal" / "freeze_manifest.json")
    return payload


def run_pilot(repository: Path) -> Dict[str, object]:
    output = artifact_root() / "pilot"
    parameters = ModelParameters()
    rows: List[Dict[str, object]] = []
    for n_agents in (32, 64):
        for temperature in (1.4, 1.8, 2.2, 2.8):
            for availability in (0.5, 1.0):
                row = run_parameter_cell(
                    n_agents,
                    "regular",
                    81000 + n_agents + int(temperature * 100) + int(availability * 10),
                    82000 + n_agents + int(temperature * 100) + int(availability * 10),
                    temperature,
                    availability,
                    0.0,
                    parameters,
                    80,
                    180,
                    2,
                )
                rows.append(row)
    _atomic_csv(rows, output / "pilot_cells.csv")
    summary = {
        "cells": len(rows),
        "artifact_root": str(artifact_root()),
        "magnetization_range": [
            float(min(row["mean_abs_magnetization"] for row in rows)),
            float(max(row["mean_abs_magnetization"] for row in rows)),
        ],
        "maximum_autocorrelation_time": float(max(row["integrated_autocorrelation_time"] for row in rows)),
        "purpose": "range, mixing, and runtime selection only",
    }
    _atomic_json(summary, output / "pilot_summary.json")
    return summary


def _exact_rows(protocol: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    settings = protocol["exact_validation"]
    n_agents = int(settings["agent_count"])
    adjacency = topology_adjacency(n_agents, "ring", 290001)
    validation_rows: List[Dict[str, object]] = []
    probability_rows: List[Dict[str, object]] = []
    landscape_rows: List[Dict[str, object]] = []
    for temperature in settings["temperatures"]:
        parameters = ModelParameters(0.45, 0.45, 0.60, float(temperature))
        model = MultiplexModel(adjacency, adjacency, parameters)
        equilibrium, energies = gibbs_distribution(model)
        kernel = exact_transition_matrix(model)
        empirical = empirical_distribution(
            model,
            seed=290100 + int(float(temperature) * 100),
            burn_in_steps=int(settings["burn_in_steps"]),
            sample_steps=int(settings["sample_steps"]),
        )
        distances = distribution_distances(empirical, equilibrium)
        balance = detailed_balance_residual(equilibrium, kernel)
        trial = 0.8 * equilibrium + 0.2 / equilibrium.size
        identity = verify_free_energy_identity(trial, equilibrium, energies, float(temperature))
        validation_rows.append(
            {
                "n_agents": n_agents,
                "temperature": float(temperature),
                "total_variation": distances["total_variation"],
                "kl_empirical_reference": distances["kl_empirical_reference"],
                "detailed_balance_maximum": balance["maximum"],
                "detailed_balance_l1": balance["l1"],
                "equilibrium_entropy_production": entropy_production_rate(equilibrium, kernel),
                "free_energy_identity_residual": identity["absolute_residual"],
            }
        )
        if abs(float(temperature) - 1.7) < 1e-9:
            for state_index, (energy, exact_probability, observed_probability) in enumerate(
                zip(energies, equilibrium, empirical)
            ):
                probability_rows.append(
                    {
                        "state_index": state_index,
                        "energy": float(energy),
                        "gibbs_probability": float(exact_probability),
                        "empirical_probability": float(observed_probability),
                    }
                )
            # Exact equilibrium Landau-like constrained free energy by total order.
            order_values = []
            for state_index in range(equilibrium.size):
                bits = ((state_index >> np.arange(2 * n_agents)) & 1)
                order_values.append(float(np.mean(2 * bits - 1)))
            for order in sorted(set(order_values)):
                mass = float(np.sum(equilibrium[np.isclose(order_values, order)]))
                landscape_rows.append(
                    {
                        "order_parameter": order,
                        "probability": mass,
                        "constrained_free_energy": float(-temperature * np.log(max(mass, 1e-300))),
                    }
                )
    epr_rows: List[Dict[str, object]] = []
    base = topology_adjacency(n_agents, "ring", 290001)
    for asymmetry in settings["asymmetries"]:
        for orientation_seed in settings["orientation_seeds"]:
            directed = directed_communication(base, float(asymmetry), int(orientation_seed))
            parameters = ModelParameters(0.70, 0.30, 0.40, 1.35)
            model = MultiplexModel(
                directed,
                base,
                parameters,
                private_fields=np.array([0.20, -0.10, 0.00]),
            )
            kernel = exact_transition_matrix(model)
            stationary = stationary_distribution(kernel)
            epr_rows.append(
                {
                    "asymmetry": float(asymmetry),
                    "orientation_seed": int(orientation_seed),
                    "entropy_production_rate": entropy_production_rate(stationary, kernel),
                    "stationarity_residual": float(np.max(np.abs(stationary.dot(kernel) - stationary))),
                }
            )
    return validation_rows, probability_rows, epr_rows, landscape_rows


def _run_finite_size(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    settings = protocol["finite_size"]
    parameters = ModelParameters()
    rows: List[Dict[str, object]] = []
    for n_agents in settings["agent_counts"]:
        for temperature in settings["temperatures"]:
            for replicate, seed in enumerate(settings["seeds"]):
                rows.append(
                    run_parameter_cell(
                        int(n_agents),
                        str(settings["topology"]),
                        int(seed) + int(n_agents) * 100 + replicate,
                        int(seed) + int(float(temperature) * 1000),
                        float(temperature),
                        1.0,
                        0.0,
                        parameters,
                        int(settings["burn_in_sweeps"]),
                        int(settings["sample_sweeps"]),
                        int(settings["sample_interval"]),
                    )
                )
    return rows


def _run_phase_grid(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    settings = protocol["phase_grid"]
    parameters = ModelParameters()
    rows: List[Dict[str, object]] = []
    for n_agents in settings["agent_counts"]:
        for topology in settings["topologies"]:
            for temperature in settings["temperatures"]:
                for availability in settings["communication_availability"]:
                    for fragmentation in settings["fragmentation"]:
                        for replicate, seed in enumerate(settings["seeds"]):
                            graph_seed = int(seed) + int(n_agents) * 100 + replicate
                            simulation_seed = (
                                int(seed)
                                + int(float(temperature) * 1000)
                                + int(float(availability) * 100)
                                + int(float(fragmentation) * 10000)
                            )
                            rows.append(
                                run_parameter_cell(
                                    int(n_agents),
                                    str(topology),
                                    graph_seed,
                                    simulation_seed,
                                    float(temperature),
                                    float(availability),
                                    float(fragmentation),
                                    parameters,
                                    int(settings["burn_in_sweeps"]),
                                    int(settings["sample_sweeps"]),
                                    int(settings["sample_interval"]),
                                )
                            )
    return rows


def _run_relaxation(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    settings = protocol["relaxation"]
    rows: List[Dict[str, object]] = []
    for n_agents in settings["agent_counts"]:
        for temperature in settings["temperatures"]:
            for replicate, seed in enumerate(settings["seeds"]):
                communication, dependency = make_model_layers(
                    int(n_agents), "lattice", int(seed) + int(n_agents) + replicate, 1.0
                )
                parameters = ModelParameters(temperature=float(temperature))
                _, trajectory = simulate_stationary(
                    communication,
                    dependency,
                    parameters,
                    int(seed) + int(float(temperature) * 1000),
                    burn_in_sweeps=0,
                    sample_sweeps=int(settings["sample_sweeps"]),
                    sample_interval=int(settings["sample_interval"]),
                    initial_state="ordered_plus",
                    keep_trajectory=True,
                )
                magnetization = np.abs(trajectory["magnetization"])
                below = np.flatnonzero(magnetization <= np.exp(-1.0))
                relaxation_time = float(below[0] + 1) if below.size else float(settings["sample_sweeps"])
                rows.append(
                    {
                        "n_agents": int(n_agents),
                        "temperature": float(temperature),
                        "seed": int(seed),
                        "relaxation_time": relaxation_time,
                        "censored": int(below.size == 0),
                        "final_abs_magnetization": float(magnetization[-1]),
                    }
                )
    return rows


def _run_applications(protocol: Dict[str, object]) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    settings = protocol["applications"]
    rows: List[Dict[str, object]] = []
    snapshot_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    conservation_rows: List[Dict[str, object]] = []
    for application in ("humanitarian", "utility"):
        for seed in settings["seeds"]:
            trajectory, diagnostic = run_application_mapping(
                application,
                int(settings["agent_count"]),
                int(seed),
                horizon=int(settings["horizon"]),
            )
            for row in trajectory:
                row["application"] = application
                row["seed"] = int(seed)
                rows.append(row)
            conservation_rows.append(
                {
                    "application": application,
                    "seed": int(seed),
                    "initial_workload": float(diagnostic["initial_workload"][0]),
                    "exogenous_injection": float(diagnostic["exogenous_injection"][0]),
                    "endogenous_cascade": float(diagnostic["endogenous_cascade"][0]),
                    "completed_work": float(diagnostic["completed_work"][0]),
                    "final_workload": float(diagnostic["final_workload"][0]),
                    "workload_residual": float(diagnostic["workload_residual"][0]),
                    "initial_resources": float(diagnostic["initial_resources"][0]),
                    "consumed_resources": float(diagnostic["consumed_resources"][0]),
                    "final_resources": float(diagnostic["final_resources"][0]),
                    "resource_residual": float(diagnostic["resource_residual"][0]),
                }
            )
            if int(seed) == int(settings["seeds"][0]):
                for index in range(int(settings["agent_count"])):
                    snapshot_rows.append(
                        {
                            "application": application,
                            "time": int(diagnostic["time"][0]),
                            "node": index,
                            "belief": int(diagnostic["beliefs"][index]),
                            "action": int(diagnostic["actions"][index]),
                            "workload": float(diagnostic["workload"][index]),
                            "private_field": float(diagnostic["private_fields"][index]),
                        }
                    )
                communication = diagnostic["communication"]
                dependency = diagnostic["dependency"]
                for left in range(communication.shape[0]):
                    for right in range(left + 1, communication.shape[1]):
                        if communication[left, right] or communication[right, left]:
                            edge_rows.append(
                                {
                                    "application": application,
                                    "layer": "communication",
                                    "source": left,
                                    "target": right,
                                    "weight_forward": float(communication[left, right]),
                                    "weight_reverse": float(communication[right, left]),
                                }
                            )
                        if dependency[left, right] or dependency[right, left]:
                            edge_rows.append(
                                {
                                    "application": application,
                                    "layer": "dependency",
                                    "source": left,
                                    "target": right,
                                    "weight_forward": float(dependency[left, right]),
                                    "weight_reverse": float(dependency[right, left]),
                                }
                            )
    return rows, snapshot_rows, edge_rows, conservation_rows


def _run_hysteresis(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    settings = protocol["hysteresis"]
    rows: List[Dict[str, object]] = []
    n_agents = int(settings["agent_count"])
    for seed in settings["seeds"]:
        communication, dependency = make_model_layers(n_agents, "lattice", int(seed), 1.0)
        parameters = ModelParameters(temperature=float(settings["temperature"]))
        trajectory = simulate_field_hysteresis(
            communication,
            dependency,
            parameters,
            [float(value) for value in settings["field_values"]],
            int(settings["sweeps_per_field"]),
            int(seed),
        )
        for row in trajectory:
            row["seed"] = int(seed)
            row["n_agents"] = n_agents
            row["temperature"] = float(settings["temperature"])
            rows.append(row)
    return rows


def run_formal(repository: Path) -> Dict[str, object]:
    protocol = load_protocol(repository)
    output = artifact_root() / "formal"
    if (output / "formal_manifest.json").exists():
        raise FileExistsError(
            "a completed formal manifest already exists; choose a fresh "
            "THERMO_V9_ARTIFACT_ROOT instead of overwriting frozen evidence"
        )
    freeze = freeze_manifest(repository)
    started = datetime.now(timezone.utc)
    status_path = output / "supervisor_status.json"
    _atomic_json({"state": "running", "stage": "exact", "started_at": started.isoformat()}, status_path)
    validation, probabilities, epr, landscape = _exact_rows(protocol)
    _atomic_csv(validation, output / "exact_validation.csv")
    _atomic_csv(probabilities, output / "exact_probabilities.csv")
    _atomic_csv(epr, output / "exact_entropy_production.csv")
    _atomic_csv(landscape, output / "exact_free_energy_landscape.csv")
    _atomic_json({"state": "running", "stage": "finite_size", "exact_rows": len(validation) + len(epr)}, status_path)
    finite_size = _run_finite_size(protocol)
    _atomic_csv(finite_size, output / "finite_size.csv")
    _atomic_json({"state": "running", "stage": "phase_grid", "finite_size_cells": len(finite_size)}, status_path)
    phase_grid = _run_phase_grid(protocol)
    _atomic_csv(phase_grid, output / "phase_grid.csv")
    _atomic_json({"state": "running", "stage": "relaxation", "phase_grid_cells": len(phase_grid)}, status_path)
    relaxation = _run_relaxation(protocol)
    _atomic_csv(relaxation, output / "relaxation.csv")
    hysteresis = _run_hysteresis(protocol)
    _atomic_csv(hysteresis, output / "hysteresis.csv")
    _atomic_json({"state": "running", "stage": "applications", "relaxation_cells": len(relaxation)}, status_path)
    applications, application_snapshots, application_edges, application_conservation = _run_applications(protocol)
    _atomic_csv(applications, output / "applications.csv")
    _atomic_csv(application_snapshots, output / "application_snapshots.csv")
    _atomic_csv(application_edges, output / "application_edges.csv")
    _atomic_csv(application_conservation, output / "application_conservation.csv")
    files = sorted(output.glob("*.csv"))
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "freeze_manifest": freeze,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": {path.name: int(len(pd.read_csv(path))) for path in files},
        "sha256": {path.name: sha256_file(path) for path in files},
        "external_artifact_root": str(artifact_root()),
        "raw_per_episode_files": 0,
    }
    _atomic_json(manifest, output / "formal_manifest.json")
    _atomic_json({"state": "complete", "stage": "complete", "completed_at": manifest["completed_at"], "row_counts": manifest["row_counts"]}, status_path)
    return manifest
