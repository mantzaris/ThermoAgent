"""Compact analysis, integrity summaries, and clean-export construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .workflow import artifact_root, load_protocol, sha256_file, source_checksum


RESULTS_RELATIVE = Path("results/statmech_agentic_v9")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, line_terminator="\n", float_format="%.10g")
    os.replace(str(temporary), str(path))


def _write_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _normal_interval(values: np.ndarray) -> Sequence[float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def _group_summary(frame: pd.DataFrame, groups: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in frame.groupby(list(groups), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(groups, keys))
        row["independent_n"] = int(len(group))
        for metric in metrics:
            mean, lower, upper = _normal_interval(group[metric].to_numpy(float))
            row[metric + "_mean"] = mean
            row[metric + "_ci_low"] = lower
            row[metric + "_ci_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    means = np.mean(values[draws], axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "independent_n": int(values.size),
    }


def _binder_crossings(summary: pd.DataFrame) -> pd.DataFrame:
    sizes = sorted(summary["n_agents"].unique())
    rows: List[Dict[str, object]] = []
    for smaller, larger in zip(sizes[:-1], sizes[1:]):
        left = summary[summary["n_agents"] == smaller].sort_values("temperature")
        right = summary[summary["n_agents"] == larger].sort_values("temperature")
        temperatures = left["temperature"].to_numpy(float)
        difference = left["binder_cumulant_mean"].to_numpy(float) - right["binder_cumulant_mean"].to_numpy(float)
        crossings = []
        for index in range(len(temperatures) - 1):
            if difference[index] == 0.0:
                crossings.append(float(temperatures[index]))
            elif difference[index] * difference[index + 1] < 0.0:
                fraction = abs(difference[index]) / (abs(difference[index]) + abs(difference[index + 1]))
                crossings.append(float(temperatures[index] + fraction * (temperatures[index + 1] - temperatures[index])))
        rows.append(
            {
                "smaller_n": int(smaller),
                "larger_n": int(larger),
                "crossing_count": len(crossings),
                "first_crossing_temperature": crossings[0] if crossings else np.nan,
                "all_crossings": ";".join("%.6f" % value for value in crossings),
            }
        )
    return pd.DataFrame(rows)


def analyze_formal(repository: Path) -> Dict[str, object]:
    external = artifact_root() / "formal"
    results = repository / RESULTS_RELATIVE
    tables = results / "tables"
    source_data = results / "figures/source_data"
    reproducibility = results / "reproducibility"
    required = [
        "formal_manifest.json",
        "exact_validation.csv",
        "exact_probabilities.csv",
        "exact_entropy_production.csv",
        "exact_free_energy_landscape.csv",
        "finite_size.csv",
        "phase_grid.csv",
        "relaxation.csv",
        "hysteresis.csv",
        "applications.csv",
        "application_snapshots.csv",
        "application_edges.csv",
        "application_conservation.csv",
    ]
    missing = [name for name in required if not (external / name).exists()]
    if missing:
        raise FileNotFoundError("missing external formal artifacts: %s" % missing)
    protocol = load_protocol(repository)
    exact = pd.read_csv(external / "exact_validation.csv")
    probabilities = pd.read_csv(external / "exact_probabilities.csv")
    epr = pd.read_csv(external / "exact_entropy_production.csv")
    landscape = pd.read_csv(external / "exact_free_energy_landscape.csv")
    finite = pd.read_csv(external / "finite_size.csv")
    phase = pd.read_csv(external / "phase_grid.csv")
    relaxation = pd.read_csv(external / "relaxation.csv")
    hysteresis = pd.read_csv(external / "hysteresis.csv")
    applications = pd.read_csv(external / "applications.csv")
    application_snapshots = pd.read_csv(external / "application_snapshots.csv")
    application_edges = pd.read_csv(external / "application_edges.csv")
    application_conservation = pd.read_csv(external / "application_conservation.csv")

    finite_metrics = [
        "mean_abs_magnetization",
        "susceptibility_per_agent",
        "heat_capacity_per_agent",
        "binder_cumulant",
        "integrated_autocorrelation_time",
        "energy_per_agent",
        "macrostate_shannon",
        "macrostate_tsallis_q_0_5",
        "macrostate_tsallis_q_2_0",
    ]
    finite_summary = _group_summary(finite, ["n_agents", "temperature"], finite_metrics)
    phase_summary = _group_summary(
        phase,
        ["n_agents", "topology", "temperature", "communication_availability", "fragmentation"],
        [
            "mean_abs_magnetization",
            "belief_action_consistency",
            "susceptibility_per_agent",
            "heat_capacity_per_agent",
            "integrated_autocorrelation_time",
            "activity",
            "macrostate_shannon",
        ],
    )
    epr_summary = _group_summary(epr, ["asymmetry"], ["entropy_production_rate"])
    relaxation_summary = _group_summary(
        relaxation, ["n_agents", "temperature"], ["relaxation_time", "final_abs_magnetization"]
    )
    application_summary = _group_summary(
        applications,
        ["application", "time"],
        [
            "belief_order",
            "action_order",
            "belief_action_consistency",
            "belief_entropy",
            "action_entropy",
            "entropy_flow_per_update",
            "workload_density",
            "service_loss",
            "cascade_depth",
            "activity",
        ],
    )
    hysteresis_summary = _group_summary(hysteresis, ["branch_code", "field"], ["magnetization"])
    crossings = _binder_crossings(finite_summary)
    susceptibility_peaks = (
        finite_summary.sort_values("susceptibility_per_agent_mean", ascending=False)
        .groupby("n_agents", as_index=False)
        .first()
        .sort_values("n_agents")
    )
    susceptibility_fit = np.polyfit(
        np.log(susceptibility_peaks["n_agents"].to_numpy(float)),
        np.log(np.maximum(susceptibility_peaks["susceptibility_per_agent_mean"].to_numpy(float), 1e-12)),
        1,
    )
    correlation_matrix = finite[
        ["macrostate_shannon", "macrostate_tsallis_q_0_5", "macrostate_tsallis_q_2_0", "macrostate_gini_simpson"]
    ].corr()
    correlation_rows = []
    for left in correlation_matrix.columns:
        for right in correlation_matrix.columns:
            correlation_rows.append({"measure_a": left, "measure_b": right, "pearson_correlation": correlation_matrix.loc[left, right]})
    entropy_correlations = pd.DataFrame(correlation_rows)

    _write_csv(exact, tables / "exact_validation.csv")
    _write_csv(finite_summary, tables / "finite_size_summary.csv")
    _write_csv(phase_summary, tables / "phase_summary.csv")
    _write_csv(epr_summary, tables / "entropy_production_summary.csv")
    _write_csv(relaxation_summary, tables / "relaxation_summary.csv")
    _write_csv(hysteresis_summary, tables / "hysteresis_summary.csv")
    _write_csv(application_summary, tables / "application_mapping_summary.csv")
    _write_csv(application_conservation, tables / "application_conservation.csv")
    _write_csv(crossings, tables / "binder_crossing_candidates.csv")
    _write_csv(susceptibility_peaks, tables / "susceptibility_peaks.csv")
    _write_csv(entropy_correlations, tables / "generalized_entropy_correlations.csv")
    _write_csv(probabilities, source_data / "figure_02_equilibrium_validation.csv")
    _write_csv(landscape, source_data / "figure_07_free_energy_landscape.csv")
    _write_csv(application_snapshots, source_data / "figure_09_application_nodes.csv")
    _write_csv(application_edges, source_data / "figure_09_application_edges.csv")

    thresholds = protocol["stage_rules"]
    exact_pass = bool(
        exact["detailed_balance_maximum"].max() <= float(thresholds["exact_detailed_balance_maximum"])
        and exact["total_variation"].max() <= float(thresholds["gibbs_empirical_total_variation_maximum"])
        and exact["free_energy_identity_residual"].max() <= float(thresholds["free_energy_identity_residual_maximum"])
        and exact["equilibrium_entropy_production"].abs().max()
        <= float(thresholds["entropy_production_equilibrium_absolute_maximum"])
    )
    epr_high = epr[epr["asymmetry"] >= 0.5]["entropy_production_rate"].to_numpy(float)
    epr_effect = _bootstrap_mean(epr_high, int(protocol["analysis"]["bootstrap_seed"]), int(protocol["analysis"]["bootstrap_replicates"]))
    crossover_values = crossings["first_crossing_temperature"].dropna().to_numpy(float)
    principal = {
        "protocol_version": protocol["protocol_version"],
        "evidence_stage": "frozen numerical evaluation; no external confirmatory holdout",
        "exact_equilibrium_gate_passed": exact_pass,
        "maximum_detailed_balance_residual": float(exact["detailed_balance_maximum"].max()),
        "maximum_empirical_gibbs_total_variation": float(exact["total_variation"].max()),
        "maximum_free_energy_identity_residual": float(exact["free_energy_identity_residual"].max()),
        "maximum_equilibrium_entropy_production": float(exact["equilibrium_entropy_production"].abs().max()),
        "binder_crossing_candidate_median": float(np.median(crossover_values)) if crossover_values.size else None,
        "binder_pair_count": int(len(crossings)),
        "binder_pairs_with_crossing": int(crossings["first_crossing_temperature"].notna().sum()),
        "susceptibility_peak_scaling_exponent": float(susceptibility_fit[0]),
        "high_asymmetry_entropy_production": epr_effect,
        "minimum_generalized_entropy_correlation_with_shannon": float(
            entropy_correlations[
                (entropy_correlations["measure_a"] == "macrostate_shannon")
                & (entropy_correlations["measure_b"] != "macrostate_shannon")
            ]["pearson_correlation"].min()
        ),
        "maximum_application_conservation_residual": float(
            max(
                application_conservation["workload_residual"].abs().max(),
                application_conservation["resource_residual"].abs().max(),
            )
        ),
        "formal_cell_counts": {
            "exact_validation": int(len(exact)),
            "directed_exact": int(len(epr)),
            "finite_size": int(len(finite)),
            "phase_grid": int(len(phase)),
            "relaxation": int(len(relaxation)),
            "hysteresis": int(len(hysteresis)),
            "application_time_rows": int(len(applications)),
            "application_trajectories": int(applications.groupby(["application", "seed"]).ngroups),
            "application_conservation_records": int(len(application_conservation)),
        },
        "independent_seed_counts": {
            "finite_size_per_cell": int(finite.groupby(["n_agents", "temperature"]).size().min()),
            "phase_grid_per_cell": int(phase.groupby(["n_agents", "topology", "temperature", "communication_availability", "fragmentation"]).size().min()),
            "applications_per_mapping": int(applications.groupby("application")["seed"].nunique().min()),
        },
    }
    _write_json(principal, tables / "principal_results.json")

    external_manifest = json.loads((external / "formal_manifest.json").read_text(encoding="utf-8"))
    reproducibility_payload = {
        "parent_v8_commit": "b86f97fa0940f11cb366c809e0e46fa888dfaba1",
        "branch": "statistical-mechanics-agentic-systems-v9",
        "uncommitted_by_requirement": True,
        "protocol_sha256": external_manifest["freeze_manifest"]["protocol_sha256"],
        "formal_source_sha256": external_manifest["freeze_manifest"]["source_sha256"],
        "current_source_sha256": source_checksum(repository),
        "external_artifact_root": str(artifact_root()),
        "external_artifact_sha256": external_manifest["sha256"],
        "raw_artifacts_in_repository": False,
        "analysis": principal,
    }
    _write_json(reproducibility_payload, reproducibility / "summary.json")
    return principal


def build_clean_export(repository: Path) -> Dict[str, object]:
    destination = Path(os.environ.get("THERMO_V9_EXPORT_ROOT", "/tmp/ThermoAgent-JSTAT-clean-export"))
    destination.mkdir(parents=True, exist_ok=True)
    selected = [
        Path("pyproject.toml"),
        Path("LICENSE"),
        Path("thermoagent/statmech"),
        Path("configs/statmech_v9"),
        Path("tests/statmech_v9"),
        Path("results/statmech_agentic_v9"),
        Path("paper/jstat_v9"),
        Path("notes/v9_research_log.md"),
    ]
    selected.extend(
        sorted(path.relative_to(repository) for path in (repository / "scripts").glob("run-statmech-v9-*.sh"))
    )
    copied: List[Path] = []
    excluded_suffixes = {
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".png",
        ".pyc",
    }
    for relative in selected:
        source = repository / relative
        if not source.exists():
            continue
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for path in source.rglob("*"):
                if (
                    path.is_dir()
                    or "__pycache__" in path.parts
                    or path.suffix in excluded_suffixes
                    or path.name.endswith(".synctex.gz")
                ):
                    continue
                output = target / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(path), str(output))
                copied.append(output)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
            copied.append(target)
    inventory = []
    for path in sorted(copied):
        inventory.append(
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _write_json(
        {
            "not_a_git_repository": True,
            "source_parent": str(repository),
            "file_count": len(inventory),
            "total_bytes": int(sum(row["bytes"] for row in inventory)),
            "files": inventory,
            "proposed_layout": [str(path) for path in selected],
        },
        destination / "EXPORT_INVENTORY.json",
    )
    return {
        "destination": str(destination),
        "file_count": len(inventory) + 1,
        "total_bytes": int(sum(row["bytes"] for row in inventory) + (destination / "EXPORT_INVENTORY.json").stat().st_size),
    }
