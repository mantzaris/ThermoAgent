"""Conditional V11 formal decentralized LLM-network execution and analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .network import (
    DecentralizedEvidenceNetwork,
    coarse_macrostate,
    generate_step_tape,
    make_network_agents,
    oriented_edge_signs,
    undirected_skeleton,
)
from .qwen import QwenEvidenceProvider
from .statistics import (
    block_time_reversal_kl,
    conditional_mutual_information_history,
    entropy_production_per_update,
    normalize_transition_counts,
    paired_cluster_bootstrap,
    stationary_distribution,
    trajectory_transition_counts,
)
from .workflow import artifact_root, atomic_csv, atomic_json, load_yaml, stage_lock, utc_now


def _require_formal_unlock(repository: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    if os.environ.get("THERMO_V11_ENABLE_QWEN") != "1":
        raise RuntimeError("Qwen execution is not explicitly enabled")
    qualification_path = artifact_root() / "qualification/analysis.json"
    if not qualification_path.exists():
        raise RuntimeError("qualification analysis is absent")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not qualification.get("formal_network_unlocked", False):
        raise RuntimeError("qualification gate did not unlock the formal network")
    protocol_path = Path(repository) / "configs/statmech_v11/formal_frozen.yaml"
    if not protocol_path.exists():
        raise RuntimeError("formal protocol is not frozen")
    protocol = load_yaml(protocol_path)
    if protocol.get("status") != "frozen_before_formal_generation":
        raise RuntimeError("formal protocol status is not frozen")
    return qualification, protocol


def formal_panel_design(settings: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for application, domain in (("humanitarian", "route_viability"), ("utility", "repair_hypothesis")):
        for n_agents in settings["agent_counts"]:  # type: ignore[index]
            for topology in settings["topologies"]:  # type: ignore[index]
                for orientation in settings["orientation_seeds"]:  # type: ignore[index]
                    for environment_seed in settings["environment_seeds"]:  # type: ignore[index]
                        base = "%s_n%s_%s_o%s_e%s" % (
                            application,
                            n_agents,
                            topology,
                            orientation,
                            environment_seed,
                        )
                        for alpha in settings["nonreciprocity_levels"]:  # type: ignore[index]
                            rows.append(
                                {
                                    "panel_id": base + "_a%s" % str(alpha).replace(".", "p"),
                                    "matched_cluster": base,
                                    "application": application,
                                    "domain": domain,
                                    "n_agents": int(n_agents),
                                    "topology": str(topology),
                                    "orientation_seed": int(orientation),
                                    "environment_seed": int(environment_seed),
                                    "alpha": float(alpha),
                                    "turns": int(settings["trajectory_turns"]),
                                    "control": "unaltered",
                                    "prompt_mode": "formal",
                                    "panel_family": "primary",
                                }
                            )
    control_settings = settings["control_design"]  # type: ignore[index]
    for application, domain in (("humanitarian", "route_viability"), ("utility", "repair_hypothesis")):
        for orientation in control_settings["orientation_seeds"]:
            for environment_seed in control_settings["environment_seeds"]:
                base = "%s_control_o%s_e%s" % (application, orientation, environment_seed)
                for control in control_settings["controls"]:
                    prompt_mode = "formal"
                    if control == "reported_probability_removed":
                        prompt_mode = "formal_no_reported_probability"
                    elif control == "commitment_removed":
                        prompt_mode = "formal_no_commitment"
                    transport_control = "unaltered" if control in ("reported_probability_removed", "commitment_removed") else control
                    rows.append(
                        {
                            "panel_id": base + "_" + str(control),
                            "matched_cluster": base,
                            "application": application,
                            "domain": domain,
                            "n_agents": int(control_settings["n_agents"]),
                            "topology": str(control_settings["topology"]),
                            "orientation_seed": int(orientation),
                            "environment_seed": int(environment_seed),
                            "alpha": float(control_settings["alpha"]),
                            "turns": int(control_settings["turns"]),
                            "control": str(transport_control),
                            "analysis_control_label": str(control),
                            "prompt_mode": prompt_mode,
                            "panel_family": "control",
                        }
                    )
    return rows


def expected_formal_decisions(settings: Mapping[str, object]) -> int:
    return int(sum(int(panel["turns"]) for panel in formal_panel_design(settings)))


def run_formal_network(repository: Path) -> Dict[str, object]:
    _qualification, protocol = _require_formal_unlock(repository)
    settings = protocol["formal"]  # type: ignore[index]
    output = artifact_root() / "formal"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "trajectory_rows.csv"
    completed_path = output / "completed_panels.json"
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    all_rows: List[Dict[str, object]] = []
    completed: List[str] = []
    if rows_path.exists():
        all_rows = pd.read_csv(rows_path).to_dict(orient="records")
    if completed_path.exists():
        completed = list(json.loads(completed_path.read_text(encoding="utf-8"))["completed_panels"])
    provider = QwenEvidenceProvider(
        artifact_root() / "raw/formal",
        repository,
        float(settings["inference_sampling_temperature"]),
        float(settings["top_p"]),
        int(settings["maximum_new_tokens"]),
    )
    panels = formal_panel_design(settings)
    with stage_lock("formal"):
        for panel in panels:
            panel_id = str(panel["panel_id"])
            if panel_id in completed:
                continue
            seed = int(settings["seed_base"]) + int(panel["environment_seed"]) + 100 * int(panel["orientation_seed"])
            latent_state = "right" if seed % 2 else "left"
            skeleton = undirected_skeleton(int(panel["n_agents"]), str(panel["topology"]), seed)
            orientation = oriented_edge_signs(skeleton, int(panel["orientation_seed"]) + 9100)
            agents = make_network_agents(
                int(panel["n_agents"]),
                str(panel["domain"]),
                latent_state,
                float(settings["private_signal_reliability_levels"][1]),
                seed,
            )
            network = DecentralizedEvidenceNetwork(agents, skeleton, orientation, latent_state)
            tapes = generate_step_tape(int(panel["n_agents"]), int(panel["turns"]), seed + 200000)
            panel_rows: List[Dict[str, object]] = []
            for turn, tape in enumerate(tapes):
                reliability_levels = [float(value) for value in settings["private_signal_reliability_levels"]]
                reliability_index = min(
                    int(float(tape.reliability_uniform) * len(reliability_levels)), len(reliability_levels) - 1
                )
                local_reliability = reliability_levels[reliability_index]
                try:
                    row = network.offered_step(
                        provider,
                        tape,
                        float(panel["alpha"]),
                        local_reliability,
                        str(panel["domain"]),
                        turn,
                        str(panel["control"]),
                        str(panel["prompt_mode"]),
                    )
                    row["valid_after_repair"] = 1
                except ValueError:
                    row = {
                        "turn": turn,
                        "valid_after_repair": 0,
                        "message_attempted": 0,
                        "message_transmitted": 0,
                        "message_wire_bytes": 0,
                        "prompt_tokens": 0,
                        "generated_tokens": 0,
                        "latency_seconds": 0.0,
                        # Invalid output is a failed attempted update.  The
                        # scheduler never substitutes an action, so the state
                        # is retained as an explicit self transition.
                        "coarse_macrostate": coarse_macrostate(network.agents),
                        "belief_macrostate": int(sum(item.belief_choice == "right" for item in network.agents)),
                        "action_macrostate": int(
                            sum(item.action == "select_right" for item in network.agents)
                            - sum(item.action == "select_left" for item in network.agents)
                            + len(network.agents)
                        ),
                        "service_after": network.service_deficit,
                        "causal_service_change": 0.0,
                    }
                row.update({key: value for key, value in panel.items() if key not in row})
                panel_rows.append(row)
            all_rows.extend(panel_rows)
            completed.append(panel_id)
            atomic_csv(all_rows, rows_path)
            atomic_json({"completed_panels": completed, "updated_at": utc_now()}, completed_path)
        summary = {
            "completed_at": utc_now(),
            "panels": len(panels),
            "decision_requests": sum(int(panel["turns"]) for panel in panels),
            "valid_decisions": int(sum(int(row["valid_after_repair"]) for row in all_rows)),
            "provider_accounting": provider.accounting,
            "environment": provider.environment_manifest(),
        }
        atomic_json(summary, summary_path)
    return summary


def _trajectory_metrics(states: np.ndarray, shuffle_replicates: int, rng: np.random.Generator) -> Dict[str, float]:
    if states.size < 8:
        return {"block_kl": float("nan"), "shuffle_floor": float("nan"), "adjusted_block_kl": float("nan")}
    observed = block_time_reversal_kl(states, 3, 0.5)
    null = np.asarray(
        [block_time_reversal_kl(states[rng.permutation(states.size)], 3, 0.5) for _ in range(int(shuffle_replicates))]
    )
    unique = {int(value): index for index, value in enumerate(sorted(set(states.tolist())))}
    encoded = np.asarray([unique[int(value)] for value in states], dtype=int)
    counts = trajectory_transition_counts(encoded, len(unique))
    kernel = normalize_transition_counts(counts, 0.5)
    epr = entropy_production_per_update(stationary_distribution(kernel), kernel)
    return {
        "block_kl": float(observed),
        "shuffle_floor": float(np.mean(null)),
        "shuffle_floor_95": float(np.quantile(null, 0.95)),
        "adjusted_block_kl": float(observed - np.mean(null)),
        "coarse_kernel_epr_per_update": epr,
        "markov_cmi_history_1": conditional_mutual_information_history(states, 1, 0.1),
        "markov_cmi_history_2": conditional_mutual_information_history(states, 2, 0.1),
    }


def _curve_model_comparison(primary: pd.DataFrame) -> List[Dict[str, object]]:
    """Compare frozen low-alpha response forms by leave-one-cluster-out error."""

    model_columns = {
        "linear": lambda alpha: np.column_stack([alpha]),
        "quadratic": lambda alpha: np.column_stack([alpha ** 2]),
        "linear_plus_quadratic": lambda alpha: np.column_stack([alpha, alpha ** 2]),
    }
    prepared: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cluster, group in primary.groupby("matched_cluster", sort=True):
        ordered = group.sort_values("alpha")
        baseline = float(ordered.loc[np.isclose(ordered["alpha"], ordered["alpha"].min()), "adjusted_block_kl"].iloc[0])
        prepared[str(cluster)] = (
            ordered["alpha"].to_numpy(float),
            ordered["adjusted_block_kl"].to_numpy(float) - baseline,
        )
    if len(prepared) < 2:
        raise ValueError("curve comparison requires at least two independent clusters")
    output: List[Dict[str, object]] = []
    for model, design_function in model_columns.items():
        all_alpha = np.concatenate([value[0] for value in prepared.values()])
        all_y = np.concatenate([value[1] for value in prepared.values()])
        coefficients = np.linalg.lstsq(design_function(all_alpha), all_y, rcond=None)[0]
        held_errors: List[float] = []
        for held_cluster, (held_alpha, held_y) in prepared.items():
            training = [value for key, value in prepared.items() if key != held_cluster]
            train_alpha = np.concatenate([value[0] for value in training])
            train_y = np.concatenate([value[1] for value in training])
            fitted = np.linalg.lstsq(design_function(train_alpha), train_y, rcond=None)[0]
            held_errors.extend((held_y - design_function(held_alpha).dot(fitted)).tolist())
        row: Dict[str, object] = {
            "model": model,
            "independent_clusters": len(prepared),
            "observations": int(all_y.size),
            "leave_cluster_out_rmse": float(np.sqrt(np.mean(np.square(held_errors)))),
            "full_fit_rmse": float(np.sqrt(np.mean(np.square(all_y - design_function(all_alpha).dot(coefficients))))),
            "linear_coefficient": float(coefficients[0]) if model in ("linear", "linear_plus_quadratic") else 0.0,
            "quadratic_coefficient": float(coefficients[-1]) if model in ("quadratic", "linear_plus_quadratic") else 0.0,
        }
        output.append(row)
    return output


def analyze_formal_network(repository: Path) -> Dict[str, object]:
    _qualification, protocol = _require_formal_unlock(repository)
    settings = protocol["analysis"]  # type: ignore[index]
    output = artifact_root() / "formal"
    rows = pd.read_csv(output / "trajectory_rows.csv")
    rng = np.random.default_rng(int(settings["analysis_seed"]))
    panel_rows: List[Dict[str, object]] = []
    for panel_id, group in rows.groupby("panel_id", sort=True):
        ordered_all = group.sort_values("turn")
        burn_in = int(protocol["formal"]["burn_in_turns"])  # type: ignore[index]
        ordered = ordered_all[ordered_all["turn"] >= burn_in].copy()
        if len(ordered) < 8:
            raise RuntimeError("formal panel has too few post-burn-in attempted updates: %s" % panel_id)
        valid_ordered = ordered[ordered["valid_after_repair"] == 1].copy()
        metric = _trajectory_metrics(
            ordered["coarse_macrostate"].to_numpy(int), int(settings["shuffle_replicates_per_panel"]), rng
        )
        belief_metric = _trajectory_metrics(
            ordered["belief_macrostate"].to_numpy(int), int(settings["shuffle_replicates_per_panel"]), rng
        )
        action_metric = _trajectory_metrics(
            ordered["action_macrostate"].to_numpy(int), int(settings["shuffle_replicates_per_panel"]), rng
        )
        first = ordered.iloc[0]
        analysis_label = first.get("analysis_control_label", first["control"])
        if pd.isna(analysis_label):
            analysis_label = first["control"]
        panel_rows.append(
            {
                "panel_id": panel_id,
                "matched_cluster": first["matched_cluster"],
                "application": first["application"],
                "n_agents": int(first["n_agents"]),
                "topology": first["topology"],
                "alpha": float(first["alpha"]),
                "control": str(analysis_label),
                "panel_family": str(first["panel_family"]),
                "turns": int(len(ordered)),
                "attempted_turns": int(len(group)),
                "burn_in_turns": burn_in,
                "valid_post_burn_in_turns": int(len(valid_ordered)),
                "invalid_post_burn_in_turns": int(len(ordered) - len(valid_ordered)),
                **metric,
                "coarse_kernel_epr_per_sweep": float(metric["coarse_kernel_epr_per_update"] * int(first["n_agents"])),
                "belief_adjusted_block_kl": belief_metric["adjusted_block_kl"],
                "action_adjusted_block_kl": action_metric["adjusted_block_kl"],
                "messages_transmitted": int(ordered["message_transmitted"].fillna(0).sum()),
                "message_wire_bytes": int(ordered["message_wire_bytes"].fillna(0).sum()),
                "prompt_tokens": int(ordered["prompt_tokens"].fillna(0).sum()),
                "generated_tokens": int(ordered["generated_tokens"].fillna(0).sum()),
                "latency_seconds": float(ordered["latency_seconds"].fillna(0).sum()),
                "final_service_deficit": float(ordered["service_after"].iloc[-1]),
                "beneficial_actions": int(np.sum(valid_ordered["causal_service_change"].to_numpy(float) < 0.0)),
                "neutral_actions": int(np.sum(np.isclose(valid_ordered["causal_service_change"].to_numpy(float), 0.0))),
                "harmful_actions": int(np.sum(valid_ordered["causal_service_change"].to_numpy(float) > 0.0)),
                "privacy_mutations": int(ordered.get("unrelated_peer_private_mutations", pd.Series(dtype=float)).sum()),
            }
        )
    panel = pd.DataFrame(panel_rows)
    primary = panel[panel["panel_family"] == "primary"].copy()
    alpha_values = sorted(primary["alpha"].unique())
    baseline_alpha = min(alpha_values)
    comparison_alpha = max(alpha_values)
    effects: Dict[str, Sequence[float]] = {}
    paired_rows: List[Dict[str, object]] = []
    for cluster, group in primary.groupby("matched_cluster"):
        indexed = group.set_index("alpha")
        if baseline_alpha not in indexed.index or comparison_alpha not in indexed.index:
            continue
        difference = float(indexed.loc[comparison_alpha, "adjusted_block_kl"] - indexed.loc[baseline_alpha, "adjusted_block_kl"])
        effects[str(cluster)] = [difference]
        paired_rows.append({"matched_cluster": cluster, "alpha": comparison_alpha, "adjusted_irreversibility_difference": difference})
    h3 = paired_cluster_bootstrap(effects, int(settings["cluster_bootstrap_replicates"]), int(settings["analysis_seed"]) + 1)
    h3_by_application: Dict[str, Dict[str, float]] = {}
    for application in sorted(primary["application"].unique()):
        application_effects = {
            str(row["matched_cluster"]): [float(row["adjusted_irreversibility_difference"])]
            for row in paired_rows
            if str(row["matched_cluster"]).startswith(str(application) + "_")
        }
        h3_by_application[str(application)] = paired_cluster_bootstrap(
            application_effects,
            int(settings["cluster_bootstrap_replicates"]),
            int(settings["analysis_seed"]) + 100 + len(h3_by_application),
        )
    weak_alpha_values = [float(value) for value in settings["weak_alpha_range"]]  # type: ignore[index]
    weak_primary = primary[primary["alpha"].isin(weak_alpha_values)].copy()
    slopes: Dict[str, Sequence[float]] = {}
    for cluster, group in weak_primary.groupby("matched_cluster"):
        indexed = group.set_index("alpha")
        if baseline_alpha not in indexed.index:
            continue
        x = group["alpha"].to_numpy(float) ** 2
        y = group["adjusted_block_kl"].to_numpy(float) - float(indexed.loc[baseline_alpha, "adjusted_block_kl"])
        if np.dot(x, x) > 0.0:
            slopes[str(cluster)] = [float(np.dot(x, y) / np.dot(x, x))]
    h4 = paired_cluster_bootstrap(slopes, int(settings["cluster_bootstrap_replicates"]), int(settings["analysis_seed"]) + 2)
    curve_models = _curve_model_comparison(weak_primary)

    message_ratio_effects: Dict[str, Sequence[float]] = {}
    byte_ratio_effects: Dict[str, Sequence[float]] = {}
    for cluster, group in primary.groupby("matched_cluster"):
        indexed = group.set_index("alpha")
        if baseline_alpha not in indexed.index or comparison_alpha not in indexed.index:
            continue
        baseline_messages = float(indexed.loc[baseline_alpha, "messages_transmitted"])
        baseline_bytes = float(indexed.loc[baseline_alpha, "message_wire_bytes"])
        message_ratio_effects[str(cluster)] = [
            float(indexed.loc[comparison_alpha, "messages_transmitted"] / max(baseline_messages, 1.0) - 1.0)
        ]
        byte_ratio_effects[str(cluster)] = [
            float(indexed.loc[comparison_alpha, "message_wire_bytes"] / max(baseline_bytes, 1.0) - 1.0)
        ]
    message_ratio = paired_cluster_bootstrap(
        message_ratio_effects,
        int(settings["cluster_bootstrap_replicates"]),
        int(settings["analysis_seed"]) + 3,
    )
    byte_ratio = paired_cluster_bootstrap(
        byte_ratio_effects,
        int(settings["cluster_bootstrap_replicates"]),
        int(settings["analysis_seed"]) + 4,
    )
    control_frame = panel[panel["panel_family"] == "control"].copy()
    control_effects: List[Dict[str, object]] = []
    for control in sorted(value for value in control_frame["control"].unique() if value != "unaltered"):
        effects_by_cluster: Dict[str, Sequence[float]] = {}
        for cluster, group in control_frame.groupby("matched_cluster"):
            indexed = group.set_index("control")
            if "unaltered" in indexed.index and control in indexed.index:
                effects_by_cluster[str(cluster)] = [
                    float(indexed.loc[control, "adjusted_block_kl"] - indexed.loc["unaltered", "adjusted_block_kl"])
                ]
        if effects_by_cluster:
            effect = paired_cluster_bootstrap(
                effects_by_cluster,
                int(settings["cluster_bootstrap_replicates"]),
                int(settings["analysis_seed"]) + 10 + len(control_effects),
            )
            control_effects.append({"control": control, **effect})
    atomic_csv(panel_rows, output / "panel_metrics.csv")
    atomic_csv(paired_rows, output / "paired_primary_effects.csv")
    if control_effects:
        atomic_csv(control_effects, output / "control_effects.csv")
    atomic_csv(curve_models, output / "quadratic_model_comparison.csv")
    h3_pass = bool(
        float(h3["ci_low"]) > 0.0
        and all(float(value["ci_low"]) > 0.0 for value in h3_by_application.values())
    )
    quadratic_row = next(row for row in curve_models if row["model"] == "quadratic")
    linear_row = next(row for row in curve_models if row["model"] == "linear")
    h4_pass = bool(float(h4["ci_low"]) > 0.0 and float(quadratic_row["leave_cluster_out_rmse"]) <= float(linear_row["leave_cluster_out_rmse"]))
    traffic_limit = float(settings["maximum_primary_traffic_ratio_difference"])
    traffic_matched = bool(
        max(abs(float(message_ratio["ci_low"])), abs(float(message_ratio["ci_high"]))) <= traffic_limit
        and max(abs(float(byte_ratio["ci_low"])), abs(float(byte_ratio["ci_high"]))) <= traffic_limit
    )
    report = {
        "analyzed_at": utc_now(),
        "panels": int(len(panel)),
        "primary_panels": int(len(primary)),
        "independent_primary_clusters": int(primary["matched_cluster"].nunique()),
        "H3_nonreciprocal_minus_reciprocal_adjusted_irreversibility": h3,
        "H3_by_application": h3_by_application,
        "H4_quadratic_coefficient": h4,
        "H4_curve_model_comparison": curve_models,
        "H3_frozen_success_rule_passed": h3_pass,
        "H4_frozen_success_rule_passed": h4_pass,
        "alpha_0p75_vs_0_message_ratio_minus_one": message_ratio,
        "alpha_0p75_vs_0_wire_byte_ratio_minus_one": byte_ratio,
        "primary_traffic_matched_within_frozen_tolerance": traffic_matched,
        "control_minus_unaltered_irreversibility": control_effects,
        "reciprocal_noise_floor_mean": float(primary[np.isclose(primary["alpha"], baseline_alpha)]["shuffle_floor"].mean()),
        "markov_cmi_history_1_mean": float(primary["markov_cmi_history_1"].mean()),
        "markov_cmi_history_2_mean": float(primary["markov_cmi_history_2"].mean()),
        "first_order_markov_adequate": bool(
            float(primary["markov_cmi_history_2"].mean()) <= float(settings["maximum_markov_history_cmi"])
        ),
        "metric_interpretation": "coarse-grained block time-reversal KL lower bound; not exact full-process entropy production",
        "total_attempted_updates": int(len(rows)),
        "total_valid_updates": int(rows["valid_after_repair"].sum()),
        "total_messages": int(rows["message_transmitted"].fillna(0).sum()),
        "total_wire_bytes": int(rows["message_wire_bytes"].fillna(0).sum()),
        "total_prompt_tokens": int(rows["prompt_tokens"].fillna(0).sum()),
        "total_generated_tokens": int(rows["generated_tokens"].fillna(0).sum()),
        "total_latency_seconds": float(rows["latency_seconds"].fillna(0).sum()),
        "privacy_mutations": int(rows.get("unrelated_peer_private_mutations", pd.Series(dtype=float)).fillna(0).sum()),
    }
    atomic_json(report, output / "analysis.json")
    return report
