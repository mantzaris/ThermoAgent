"""Out-of-sample kinetic reference under the V14/V15 quench schedule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from thermoagent.statmech_llm.discovery.graphs import select_recipient
from thermoagent.statmech_llm.discovery.simulation import generate_update_tape
from thermoagent.statmech_llm.replication.observables import instantaneous_state, integrated_correlation_time, plugin_entropy
from thermoagent.statmech_llm.replication.simulation import make_replication_agents, phase_for_update
from thermoagent.statmech_llm.replication.simulation import build_reciprocal_graph
from thermoagent.statmech_llm.replication.surrogate import fit_kinetic_surrogate
from thermoagent.statmech_llm.corrected_quench.experiment import formal_panel_design as v14_design
from thermoagent.statmech_llm.corrected_quench.experiment import graph_for_panel as v14_graph
from thermoagent.statmech_llm.corrected_quench.experiment import panel_seed as v14_panel_seed
from thermoagent.statmech_llm.corrected_quench.observables import recovery_time, signed_polygon_area

from .workflow import atomic_csv, atomic_json, load_yaml, utc_now


SHARED_RESPONSE_FEATURES = (
    "belief_magnetization",
    "action_magnetization",
    "belief_action_overlap",
    "reference_energy_per_agent",
    "configuration_entropy",
    "belief_susceptibility",
)


def fit_reference_coefficients(repository: Path) -> Dict[str, object]:
    frame = pd.read_csv(
        Path(repository) / "results/JSTAT/stages/replication/tables/microscopic_response.csv"
    )
    return fit_kinetic_surrogate(frame)


def _coefficients(parameters: Mapping[str, object], temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    key = "%.2f" % float(temperature)
    fit = parameters["fits_by_decoding_noise"][key]  # type: ignore[index]
    return (
        np.asarray(fit["belief_coefficients"], dtype=float),  # type: ignore[index]
        np.asarray(fit["action_coefficients"], dtype=float),  # type: ignore[index]
    )


def simulate_kinetic_quench(
    graph,
    panel_seed: int,
    sweeps: int,
    coupling: float,
    temperature: float,
    disruption: str,
    periods_sweeps: Sequence[int],
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    belief_beta, action_beta = _coefficients(parameters, temperature)
    agents = make_replication_agents(graph.n_agents, int(panel_seed), "disordered")
    fields = np.asarray([agent.private_field for agent in agents], dtype=int)
    beliefs = np.asarray([agent.belief for agent in agents], dtype=int)
    actions = np.asarray([agent.action for agent in agents], dtype=int)
    inboxes: List[List[int]] = [[] for _ in range(graph.n_agents)]
    tape = generate_update_tape(graph.n_agents, graph.n_agents * int(sweeps), int(panel_seed) + 29009)
    update_rows: List[Dict[str, float]] = []
    for update, item in enumerate(tape):
        phase = phase_for_update(update, graph.n_agents, periods_sweeps)
        active_fields = -fields if disruption == "field_reversal" and phase == "disruption" else fields
        agent = int(item.scheduled_agent)
        neighbor = float(np.mean(inboxes[agent])) if inboxes[agent] else 0.0
        inboxes[agent].clear()
        rng = np.random.default_rng(int(item.inference_seed))
        linear_b = float(
            belief_beta.dot(
                [1.0, active_fields[agent], float(coupling) * neighbor, beliefs[agent], actions[agent]]
            )
        )
        p_b = 1.0 / (1.0 + np.exp(-np.clip(linear_b, -35.0, 35.0)))
        beliefs[agent] = 1 if rng.random() < p_b else -1
        linear_a = float(action_beta.dot([1.0, beliefs[agent], actions[agent]]))
        p_a = 1.0 / (1.0 + np.exp(-np.clip(linear_a, -35.0, 35.0)))
        actions[agent] = 1 if rng.random() < p_a else -1
        recipient = select_recipient(graph.weights, agent, float(item.recipient_uniform))
        inboxes[recipient].append(int(beliefs[agent]))
        state = instantaneous_state(
            beliefs, actions, graph.adjacency, graph.symmetric, active_fields
        )
        update_rows.append(
            {
                "update": update,
                "sweep": int(update // graph.n_agents + 1),
                "phase": phase,
                **{key: float(value) for key, value in state.items()},
                "state_code": int(
                    sum((beliefs[index] > 0) << index for index in range(graph.n_agents))
                    + sum(
                        (actions[index] > 0) << (graph.n_agents + index)
                        for index in range(graph.n_agents)
                    )
                ),
            }
        )
    updates = pd.DataFrame(update_rows)
    rows: List[Dict[str, object]] = []
    window_updates = 5 * graph.n_agents
    for sweep in range(1, int(sweeps) + 1):
        end = sweep * graph.n_agents
        start = max(0, end - window_updates)
        window = updates.iloc[start:end]
        final = window.iloc[-1]
        magnetization = window["belief_magnetization"].to_numpy(float)
        rows.append(
            {
                "sweep": sweep,
                "phase": final["phase"],
                "belief_magnetization": float(final["belief_magnetization"]),
                "action_magnetization": float(final["action_magnetization"]),
                "belief_action_overlap": float(final["belief_action_overlap"]),
                "reference_energy_per_agent": float(final["reference_energy_per_agent"]),
                "configuration_entropy": plugin_entropy(window["state_code"].astype(int).tolist()),
                "belief_susceptibility": float(
                    graph.n_agents * np.var(magnetization, ddof=1)
                )
                if len(magnetization) > 1
                else 0.0,
                "integrated_correlation_time": integrated_correlation_time(magnetization),
            }
        )
    return pd.DataFrame(rows)


def _shared_response_distance(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    output["shared_response_distance"] = np.nan
    thresholds: List[Dict[str, object]] = []
    for source in sorted(output["source"].unique()):
        source_frame = output[output["source"] == source]
        for cluster in sorted(source_frame["cluster_id"].unique()):
            train = source_frame[
                (source_frame["cluster_id"] != cluster)
                & (source_frame["disruption"] == "nominal")
                & (source_frame["phase"] == "baseline")
            ]
            test = source_frame[source_frame["cluster_id"] == cluster]
            center = train[list(SHARED_RESPONSE_FEATURES)].mean().to_numpy(float)
            scale = train[list(SHARED_RESPONSE_FEATURES)].std(ddof=1).to_numpy(float)
            scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
            train_distance = np.linalg.norm(
                (train[list(SHARED_RESPONSE_FEATURES)].to_numpy(float) - center) / scale,
                axis=1,
            )
            test_distance = np.linalg.norm(
                (test[list(SHARED_RESPONSE_FEATURES)].to_numpy(float) - center) / scale,
                axis=1,
            )
            output.loc[test.index, "shared_response_distance"] = test_distance
            thresholds.append(
                {
                    "source": source,
                    "cluster_id": cluster,
                    "nominal_threshold_95": float(np.quantile(train_distance, 0.95)),
                }
            )
    return output, pd.DataFrame(thresholds)


def corrected_quench_out_of_sample_comparison(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    protocol = load_yaml(repository / "configs/statmech_llm/corrected_quench/protocol.yaml")
    parameters = fit_reference_coefficients(repository)
    direct = pd.read_csv(
        repository / "results/JSTAT/stages/corrected_quench/tables/macrostate_trajectories.csv"
    )
    direct = direct[direct["disruption"].isin(["nominal", "field_reversal"])].copy()
    direct["source"] = "Direct Qwen"
    rows: List[pd.DataFrame] = [direct]
    for panel in v14_design(protocol):
        if panel["disruption"] not in ("nominal", "field_reversal"):
            continue
        simulated = simulate_kinetic_quench(
            v14_graph(panel),
            v14_panel_seed(panel),
            int(panel["sweeps"]),
            float(panel["coupling_strength"]),
            float(panel["sampling_temperature"]),
            str(panel["disruption"]),
            panel["periods_sweeps"],
            parameters,
        )
        simulated["source"] = "Kinetic surrogate"
        simulated["cluster_id"] = panel["cluster_id"]
        simulated["panel_id"] = "surrogate_%s" % panel["panel_id"]
        simulated["disruption"] = panel["disruption"]
        rows.append(simulated)
    combined = pd.concat(rows, ignore_index=True)
    combined, thresholds = _shared_response_distance(combined)
    summaries: List[Dict[str, object]] = []
    threshold_lookup = {
        (str(row.source), str(row.cluster_id)): float(row.nominal_threshold_95)
        for row in thresholds.itertuples()
    }
    for (source, cluster, disruption), group in combined.groupby(
        ["source", "cluster_id", "disruption"]
    ):
        group = group.sort_values("sweep")
        recovery = group[group["phase"] == "recovery"]
        threshold = threshold_lookup[(str(source), str(cluster))]
        summaries.append(
            {
                "source": source,
                "cluster_id": cluster,
                "disruption": disruption,
                "peak_shared_response": float(group["shared_response_distance"].max()),
                "peak_sweep": int(
                    group.loc[group["shared_response_distance"].idxmax(), "sweep"]
                ),
                "recovery_time_sweeps": recovery_time(
                    recovery["shared_response_distance"], threshold, 2
                ),
                "energy_entropy_route_area": signed_polygon_area(
                    group["reference_energy_per_agent"], group["configuration_entropy"]
                ),
                "final_shared_response": float(group["shared_response_distance"].tail(5).mean()),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    result = repository / "results/JSTAT/stages/cross_model/tables"
    result.mkdir(parents=True, exist_ok=True)
    atomic_csv(combined, result / "corrected_quench_direct_surrogate_trajectories.csv")
    atomic_csv(summary_frame, result / "corrected_quench_direct_surrogate_summary.csv")
    atomic_csv(thresholds, result / "corrected_quench_direct_surrogate_thresholds.csv")
    metadata = {
        "generated_at": utc_now(),
        "coefficient_source": "immutable V13 microscopic_response.csv",
        "refit_to_quench": False,
        "fit": parameters,
        "direct_clusters": int(direct["cluster_id"].nunique()),
        "interpretation": "exploratory out-of-sample effective-model comparison",
    }
    atomic_json(metadata, repository / "results/JSTAT/stages/cross_model/statistics/surrogate_fit.json")
    return metadata


def dense_surrogate_size_quench(repository: Path) -> Dict[str, object]:
    """Run the frozen inexpensive N={8,16,32,64} effective-model sensitivity."""

    repository = Path(repository).resolve()
    protocol = load_yaml(repository / "configs/statmech_llm/cross_model/protocol.yaml")
    parameters = fit_reference_coefficients(repository)
    sizes = [int(value) for value in protocol["surrogate"]["cpu_sizes"]]  # type: ignore[index]
    replicates = int(protocol["surrogate"]["replicates_per_size"])  # type: ignore[index]
    rows: List[pd.DataFrame] = []
    for n_agents in sizes:
        for replicate in range(replicates):
            seed = 15170000 + 10000 * n_agents + 101 * replicate
            graph = build_reciprocal_graph(n_agents, "modular", seed + 17)
            for disruption in ("nominal", "field_reversal"):
                frame = simulate_kinetic_quench(
                    graph,
                    seed,
                    45,
                    0.8,
                    0.5,
                    disruption,
                    [15, 15, 15],
                    parameters,
                )
                frame["n_agents"] = n_agents
                frame["replicate"] = replicate
                frame["disruption"] = disruption
                rows.append(frame)
    trajectories = pd.concat(rows, ignore_index=True)
    summary = (
        trajectories.groupby(["n_agents", "disruption", "sweep", "phase"], as_index=False)
        .agg(
            belief_magnetization_mean=("belief_magnetization", "mean"),
            belief_magnetization_sd=("belief_magnetization", "std"),
            action_magnetization_mean=("action_magnetization", "mean"),
            reference_energy_mean=("reference_energy_per_agent", "mean"),
            configuration_entropy_mean=("configuration_entropy", "mean"),
            susceptibility_mean=("belief_susceptibility", "mean"),
        )
    )
    result = repository / "results/JSTAT/stages/cross_model/tables"
    atomic_csv(summary, result / "surrogate_size_quench.csv")
    metadata = {
        "generated_at": utc_now(),
        "sizes": sizes,
        "replicates_per_size": replicates,
        "trajectories": int(len(sizes) * replicates * 2),
        "direct_llm_interpretation": False,
        "coefficient_source": "immutable V13 Qwen microscopic response",
    }
    atomic_json(metadata, repository / "results/JSTAT/stages/cross_model/statistics/surrogate_size_summary.json")
    return metadata


__all__ = [
    "SHARED_RESPONSE_FEATURES",
    "fit_reference_coefficients",
    "dense_surrogate_size_quench",
    "simulate_kinetic_quench",
    "corrected_quench_out_of_sample_comparison",
]
