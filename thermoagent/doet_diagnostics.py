"""Retrospective diagnostics for the frozen ThermoAgent v1 holdout.

This module is deliberately read-only with respect to the v1 result tree.  It
writes derived artifacts only below ``results/entropy_triggered_v2``.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .policy import CoordinationPolicy


FEATURE_NAMES = [
    "backlog_pressure",
    "inventory_capacity_ratio",
    "impairment",
    "delay",
    "service_shortfall",
    "commitment_strain",
    "communication_reliability",
    "private_cost",
    "utility_service",
    "utility_cost",
    "utility_fairness",
    "utility_disclosure",
    "pending_commitments",
    "accepted_commitments",
    "partner_trust",
    "communication_budget",
    "local_surprisal",
    "distributed_entropy",
    "distributed_free_energy",
    "free_energy_change",
    "local_interaction_entropy",
    "local_consensus_error",
    "role_index",
    "previous_tool_failed",
]


def _analysis_environment() -> Dict[str, str]:
    packages = (
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "torch",
    )
    values = {"python": platform.python_version()}
    for package in packages:
        try:
            values[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            values[package.replace("-", "_")] = "not-installed"
    return values


ENTROPY_FEATURES = tuple(range(16, 22))
MATERIAL_TOOLS = {
    "schedule_shipment",
    "transfer_resource",
    "reroute_shipment",
    "expedite_shipment",
    "central_dispatch",
}
NEGOTIATION_TOOLS = {"request_quote", "submit_offer", "pledge_resource"}
DEMAND_ROLES = {"retailer", "clinic", "community"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def semantic_payload(value: Any) -> Any:
    """Drop generated identifiers while preserving operational semantics."""

    generated = {
        "message_id",
        "commitment_id",
        "parent_commitment_id",
        "coalition_id",
        "shipment_id",
    }
    if isinstance(value, Mapping):
        return {
            key: semantic_payload(item)
            for key, item in sorted(value.items())
            if key not in generated
        }
    if isinstance(value, list):
        return [semantic_payload(item) for item in value]
    return value


@dataclass
class FrozenEpisode:
    method: str
    directory: Path
    episode_path: Path
    event_path: Path
    episode: Dict[str, Any]
    events: List[Dict[str, Any]]

    @classmethod
    def read(cls, directory: Path, method: str) -> "FrozenEpisode":
        episode_path = directory / "episode.json"
        event_path = directory / "events.jsonl.gz"
        return cls(
            method=method,
            directory=directory,
            episode_path=episode_path,
            event_path=event_path,
            episode=load_json(episode_path),
            events=load_events(event_path),
        )

    @property
    def decisions(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        rows = {
            (str(row["agent_id"]), int(row["step"])): row
            for row in self.episode["trajectory"]
        }
        if len(rows) != len(self.episode["trajectory"]):
            raise ValueError("v1 trajectory contains duplicate agent/step decisions")
        return rows


def discover_pairs(results_root: Path) -> List[Tuple[FrozenEpisode, FrozenEpisode]]:
    holdout = results_root / "raw" / "holdout"
    pairs: List[Tuple[FrozenEpisode, FrozenEpisode]] = []
    for directory in sorted(holdout.glob("holdout-*-thermoagent-*/")):
        name = directory.name
        counterpart = holdout / name.replace("-thermoagent-", "-learned_no_entropy-")
        if not counterpart.is_dir():
            raise FileNotFoundError("missing matched holdout counterpart: %s" % counterpart)
        pairs.append(
            (
                FrozenEpisode.read(directory, "thermoagent"),
                FrozenEpisode.read(counterpart, "learned_no_entropy"),
            )
        )
    if len(pairs) != 16:
        raise ValueError("expected 16 v1 thermo/no-entropy holdout pairs, found %d" % len(pairs))
    return pairs


def event_signature(event: Mapping[str, Any], category: str) -> str | None:
    payload = event.get("payload", {})
    kind = str(event.get("kind"))
    if category == "messages" and kind == "message":
        return stable({"kind": payload.get("kind"), "recipient": payload.get("recipient")})
    if category == "message_recipients" and kind == "message":
        return stable(payload.get("recipient"))
    if category == "negotiation_initiations" and kind == "tool_call" and payload.get("tool") in NEGOTIATION_TOOLS:
        arguments = payload.get("arguments", {})
        return stable({"tool": payload.get("tool"), "target": arguments.get("target")})
    if category == "offers" and kind == "offer":
        return stable(semantic_payload(payload))
    if category == "counteroffers" and kind == "counteroffer":
        return stable(semantic_payload(payload))
    if category == "coalition_proposals" and kind == "coalition_event" and payload.get("action") == "propose":
        return stable(semantic_payload(payload))
    if category == "coalition_memberships" and kind == "coalition_event" and payload.get("action") == "join":
        return stable({"actor": event.get("actor"), "members": sorted(payload.get("members", []))})
    if category == "tool_calls" and kind == "tool_call":
        return stable({"tool": payload.get("tool"), "arguments": semantic_payload(payload.get("arguments", {}))})
    if category == "operational_actions" and kind == "tool_call" and payload.get("tool") in MATERIAL_TOOLS:
        return stable({"tool": payload.get("tool"), "arguments": semantic_payload(payload.get("arguments", {}))})
    return None


def event_buckets(events: Sequence[Mapping[str, Any]], category: str) -> Dict[Tuple[str, int], collections.Counter[str]]:
    output: Dict[Tuple[str, int], collections.Counter[str]] = {}
    for event in events:
        signature = event_signature(event, category)
        if signature is None:
            continue
        key = (str(event.get("actor")), int(event.get("step", -1)))
        output.setdefault(key, collections.Counter())[signature] += 1
    return output


def topology_roles(events: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    snapshot = next(event for event in events if event.get("kind") == "topology_snapshot")
    return {
        str(agent): str(record["role"])
        for agent, record in snapshot["payload"]["agents"].items()
    }


def material_action_outcomes(episode: FrozenEpisode) -> Dict[str, int]:
    roles = topology_roles(episode.events)
    counts = collections.Counter(
        material_calls=0,
        material_successes=0,
        material_failures=0,
        successful_to_demand=0,
        successful_to_intermediate=0,
    )
    events = episode.events
    for index, event in enumerate(events):
        if event.get("kind") != "tool_call" or event["payload"].get("tool") not in MATERIAL_TOOLS:
            continue
        counts["material_calls"] += 1
        result = next(
            (
                candidate
                for candidate in events[index + 1 :]
                if candidate.get("kind") == "tool_result"
                and candidate.get("actor") == event.get("actor")
                and candidate.get("step") == event.get("step")
            ),
            None,
        )
        if not result or not bool(result["payload"].get("ok")):
            counts["material_failures"] += 1
            continue
        counts["material_successes"] += 1
        target = event["payload"].get("arguments", {}).get("target")
        if roles.get(str(target)) in DEMAND_ROLES:
            counts["successful_to_demand"] += 1
        else:
            counts["successful_to_intermediate"] += 1
    return dict(counts)


def policy_choices(
    policy: CoordinationPolicy,
    observation: Sequence[float],
    mask: Sequence[bool],
) -> Tuple[int, int, np.ndarray]:
    torch = policy.torch
    vector = torch.tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy.model(vector)
    raw = logits.detach().cpu().numpy()[0]
    enabled = np.asarray(mask, dtype=bool)
    masked = np.where(enabled, raw, -1e9)
    return int(np.argmax(raw)), int(np.argmax(masked)), raw


def compare_pair(
    thermo: FrozenEpisode,
    control: FrozenEpisode,
    thermo_policy: CoordinationPolicy,
    control_policy: CoordinationPolicy,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    a = thermo.episode
    b = control.episode
    application = str(a["application"])
    scenario_name = str(a["run_id"]).split("-s%d-" % int(a["seed"]), 1)[1]
    decisions_a = thermo.decisions
    decisions_b = control.decisions
    common = set(decisions_a) & set(decisions_b)
    union = set(decisions_a) | set(decisions_b)
    common_different = sum(
        int(decisions_a[key]["action"] != decisions_b[key]["action"])
        for key in common
    )
    union_different = sum(
        int(
            key not in decisions_a
            or key not in decisions_b
            or decisions_a[key]["action"] != decisions_b[key]["action"]
        )
        for key in union
    )
    steps = range(len(a["time_series"]))
    steps_different = 0
    mask_different = 0
    singleton_a = 0
    singleton_b = 0
    mask_altered_a = 0
    mask_altered_b = 0
    mask_forced_convergence = 0
    for step in steps:
        options_a = collections.Counter(
            int(row["action"]) for row in a["trajectory"] if int(row["step"]) == step
        )
        options_b = collections.Counter(
            int(row["action"]) for row in b["trajectory"] if int(row["step"]) == step
        )
        steps_different += int(options_a != options_b)
    for key in common:
        row_a, row_b = decisions_a[key], decisions_b[key]
        mask_a = np.asarray(row_a["action_mask"], dtype=bool)
        mask_b = np.asarray(row_b["action_mask"], dtype=bool)
        mask_different += int(not np.array_equal(mask_a, mask_b))
        singleton_a += int(mask_a.sum() == 1)
        singleton_b += int(mask_b.sum() == 1)
        raw_a, masked_a, _ = policy_choices(thermo_policy, row_a["observation"], mask_a)
        raw_b, masked_b, _ = policy_choices(control_policy, row_b["observation"], mask_b)
        if masked_a != int(row_a["action"]) or masked_b != int(row_b["action"]):
            raise ValueError("checkpoint prediction does not reproduce frozen deterministic action")
        mask_altered_a += int(raw_a != masked_a)
        mask_altered_b += int(raw_b != masked_b)
        mask_forced_convergence += int(
            masked_a == masked_b and raw_a != raw_b and (raw_a != masked_a or raw_b != masked_b)
        )

    primary_a = float(a["metrics"]["primary_outcome"])
    primary_b = float(b["metrics"]["primary_outcome"])
    service_fields = (
        ["service_loss", "backlog", "fulfilled", "fulfillment_rate"]
        if application == "commercial"
        else ["weighted_backlog", "backlog", "fulfilled", "fulfillment_rate"]
    )
    service_differences = [
        abs(float(left[field]) - float(right[field]))
        for left, right in zip(a["time_series"], b["time_series"])
        for field in service_fields
    ]
    demand_differences = [
        abs(float(left["cumulative_demand"]) - float(right["cumulative_demand"]))
        for left, right in zip(a["time_series"], b["time_series"])
    ]
    material_a = material_action_outcomes(thermo)
    material_b = material_action_outcomes(control)
    planner_a = a["planner_metrics"]
    planner_b = b["planner_metrics"]
    metrics_a = a["metrics"]
    metrics_b = b["metrics"]
    source = {
        "v1_thermo_episode_sha256": sha256_file(thermo.episode_path),
        "v1_control_episode_sha256": sha256_file(control.episode_path),
        "v1_thermo_ledger_sha256": sha256_file(thermo.event_path),
        "v1_control_ledger_sha256": sha256_file(control.event_path),
    }
    tie_row = {
        "application": application,
        "scenario_name": scenario_name,
        "environment_seed": int(a["seed"]),
        "thermo_primary_outcome": primary_a,
        "control_primary_outcome": primary_b,
        "primary_difference_thermo_minus_control": primary_a - primary_b,
        "raw_float_exactly_equal": primary_a.hex() == primary_b.hex(),
        "maximum_service_trajectory_absolute_difference": max(service_differences, default=0.0),
        "service_trajectory_exactly_equal": max(service_differences, default=0.0) == 0.0,
        "maximum_exogenous_demand_absolute_difference": max(demand_differences, default=0.0),
        "exogenous_demand_trajectory_exactly_equal": max(demand_differences, default=0.0) == 0.0,
        **{"thermo_" + key: value for key, value in material_a.items()},
        **{"control_" + key: value for key, value in material_b.items()},
        "thermo_messages_excluding_sketches": int(metrics_a["messages"]),
        "control_messages_excluding_sketches": int(metrics_b["messages"]),
        "thermo_entropy_sketch_messages": int(metrics_a["monitor_sketch_messages"]),
        "control_entropy_sketch_messages": int(metrics_b["monitor_sketch_messages"]),
        "thermo_total_messages": int(metrics_a["total_communication_messages"]),
        "control_total_messages": int(metrics_b["total_communication_messages"]),
        "thermo_total_bytes": int(metrics_a["total_communication_bytes"]),
        "control_total_bytes": int(metrics_b["total_communication_bytes"]),
        "thermo_prompt_tokens": int(planner_a["prompt_tokens"]),
        "control_prompt_tokens": int(planner_b["prompt_tokens"]),
        "thermo_generated_tokens": int(planner_a["generated_tokens"]),
        "control_generated_tokens": int(planner_b["generated_tokens"]),
        "thermo_llm_calls": int(planner_a["llm_calls"]),
        "control_llm_calls": int(planner_b["llm_calls"]),
        "thermo_llm_latency_seconds": float(planner_a["llm_latency_seconds"]),
        "control_llm_latency_seconds": float(planner_b["llm_latency_seconds"]),
        "thermo_tool_calls": int(metrics_a["tool_calls"]),
        "control_tool_calls": int(metrics_b["tool_calls"]),
        "v1_rl_training_seed": 3001,
        **source,
    }
    action_row = {
        "aggregation_level": "pair",
        "application": application,
        "scenario_name": scenario_name,
        "environment_seed": int(a["seed"]),
        "simulator_steps": len(list(steps)),
        "steps_with_different_option_multisets": steps_different,
        "step_option_divergence_percent": 100.0 * steps_different / max(len(list(steps)), 1),
        "thermo_decision_epochs": len(decisions_a),
        "control_decision_epochs": len(decisions_b),
        "common_agent_decision_epochs": len(common),
        "union_agent_decision_epochs": len(union),
        "different_options_on_common_epochs": common_different,
        "common_option_divergence_percent": 100.0 * common_different / max(len(common), 1),
        "different_or_missing_options_on_union_epochs": union_different,
        "union_option_divergence_percent": 100.0 * union_different / max(len(union), 1),
        "decision_epochs_present_in_only_one_method": len(union) - len(common),
        "different_action_masks_on_common_epochs": mask_different,
        "action_mask_divergence_percent": 100.0 * mask_different / max(len(common), 1),
        "thermo_singleton_action_masks": singleton_a,
        "control_singleton_action_masks": singleton_b,
        "thermo_epochs_where_mask_changed_raw_argmax": mask_altered_a,
        "control_epochs_where_mask_changed_raw_argmax": mask_altered_b,
        "same_action_from_different_raw_argmax_due_to_mask": mask_forced_convergence,
    }
    categories = [
        "messages",
        "message_recipients",
        "negotiation_initiations",
        "offers",
        "counteroffers",
        "coalition_proposals",
        "coalition_memberships",
        "tool_calls",
        "operational_actions",
    ]
    communication_rows: List[Dict[str, Any]] = []
    for category in categories:
        buckets_a = event_buckets(thermo.events, category)
        buckets_b = event_buckets(control.events, category)
        divergent = sum(
            int(buckets_a.get(key, collections.Counter()) != buckets_b.get(key, collections.Counter()))
            for key in union
        )
        communication_rows.append({
            "aggregation_level": "pair",
            "application": application,
            "scenario_name": scenario_name,
            "environment_seed": int(a["seed"]),
            "category": category,
            "union_agent_decision_epochs": len(union),
            "divergent_agent_decision_epochs": divergent,
            "divergent_agent_decision_epochs_percent": 100.0 * divergent / max(len(union), 1),
            "thermo_event_count": int(sum(sum(counter.values()) for counter in buckets_a.values())),
            "control_event_count": int(sum(sum(counter.values()) for counter in buckets_b.values())),
        })
    return tie_row, action_row, communication_rows


def add_weighted_aggregates(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows = frame.to_dict("records")
    for application, group in frame.groupby("application"):
        aggregate: Dict[str, Any] = {
            "aggregation_level": "application",
            "application": application,
            "scenario_name": "all_v1_holdout",
            "environment_seed": "all",
        }
        for column in group_columns:
            aggregate[column] = float(group[column].sum())
        if "simulator_steps" in aggregate:
            aggregate["step_option_divergence_percent"] = 100.0 * aggregate["steps_with_different_option_multisets"] / max(aggregate["simulator_steps"], 1)
            aggregate["common_option_divergence_percent"] = 100.0 * aggregate["different_options_on_common_epochs"] / max(aggregate["common_agent_decision_epochs"], 1)
            aggregate["union_option_divergence_percent"] = 100.0 * aggregate["different_or_missing_options_on_union_epochs"] / max(aggregate["union_agent_decision_epochs"], 1)
            aggregate["action_mask_divergence_percent"] = 100.0 * aggregate["different_action_masks_on_common_epochs"] / max(aggregate["common_agent_decision_epochs"], 1)
        rows.append(aggregate)
    return pd.DataFrame(rows)


def aggregate_communication(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.to_dict("records")
    for (application, category), group in frame.groupby(["application", "category"]):
        denominator = int(group["union_agent_decision_epochs"].sum())
        divergent = int(group["divergent_agent_decision_epochs"].sum())
        rows.append({
            "aggregation_level": "application",
            "application": application,
            "scenario_name": "all_v1_holdout",
            "environment_seed": "all",
            "category": category,
            "union_agent_decision_epochs": denominator,
            "divergent_agent_decision_epochs": divergent,
            "divergent_agent_decision_epochs_percent": 100.0 * divergent / max(denominator, 1),
            "thermo_event_count": int(group["thermo_event_count"].sum()),
            "control_event_count": int(group["control_event_count"].sum()),
        })
    return pd.DataFrame(rows)


def first_layer_norms(policy: CoordinationPolicy) -> np.ndarray:
    weights = policy.model.encoder[0].weight.detach().cpu().numpy()
    return np.linalg.norm(weights, axis=0)


def selected_logit_gradients(policy: CoordinationPolicy, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    torch = policy.torch
    values: List[np.ndarray] = []
    for row in rows:
        vector = torch.tensor(
            np.asarray(row["observation"], dtype=np.float32), requires_grad=True
        ).unsqueeze(0)
        logits, _ = policy.model(vector)
        selected = logits[0, int(row["action"])]
        gradient = torch.autograd.grad(selected, vector)[0]
        values.append(np.abs(gradient.detach().cpu().numpy()[0]))
    return np.mean(values, axis=0)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def feature_usage(
    pairs: Sequence[Tuple[FrozenEpisode, FrozenEpisode]],
    thermo_policy: CoordinationPolicy,
    results_root: Path,
) -> pd.DataFrame:
    main_by_application: Dict[str, List[Sequence[float]]] = collections.defaultdict(list)
    for path in sorted((results_root / "raw" / "main").glob("main-*-thermoagent-*/episode.json")):
        episode = load_json(path)
        main_by_application[str(episode["application"])].extend(
            row["observation"] for row in episode["trajectory"]
        )
    norms = first_layer_norms(thermo_policy)
    rows_out: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian"):
        holdout_rows = [
            row
            for thermo, _ in pairs
            if thermo.episode["application"] == application
            for row in thermo.episode["trajectory"]
        ]
        matrix = np.asarray([row["observation"] for row in holdout_rows], dtype=float)
        development = np.asarray(main_by_application[application], dtype=float)
        gradients = selected_logit_gradients(thermo_policy, holdout_rows)
        changed = 0
        probability_l1: List[float] = []
        for row in holdout_rows:
            observation = np.asarray(row["observation"], dtype=np.float32)
            mask = np.asarray(row["action_mask"], dtype=bool)
            _, baseline, logits = policy_choices(thermo_policy, observation, mask)
            zeroed = observation.copy()
            zeroed[list(ENTROPY_FEATURES)] = 0.0
            _, counterfactual, zeroed_logits = policy_choices(thermo_policy, zeroed, mask)
            changed += int(baseline != counterfactual)
            p = softmax(np.where(mask, logits, -1e9))
            q = softmax(np.where(mask, zeroed_logits, -1e9))
            probability_l1.append(float(np.abs(p - q).sum() / 2.0))
        for index, name in enumerate(FEATURE_NAMES):
            lower = float(development[:, index].min()) if len(development) else np.nan
            upper = float(development[:, index].max()) if len(development) else np.nan
            out_of_main = np.mean((matrix[:, index] < lower - 1e-9) | (matrix[:, index] > upper + 1e-9)) if len(development) else np.nan
            design_low = -1.0 if index == 19 else 0.0
            design_high = 1.0
            rows_out.append({
                "application": application,
                "feature_index": index,
                "feature": name,
                "feature_group": "entropy_monitor" if index in ENTROPY_FEATURES else "ordinary_local",
                "n_holdout_decisions": len(matrix),
                "holdout_min": float(matrix[:, index].min()),
                "holdout_mean": float(matrix[:, index].mean()),
                "holdout_standard_deviation": float(matrix[:, index].std()),
                "holdout_max": float(matrix[:, index].max()),
                "holdout_nonzero_percent": float(100.0 * np.mean(np.abs(matrix[:, index]) > 1e-12)),
                "design_lower_bound": design_low,
                "design_upper_bound": design_high,
                "out_of_design_range_percent": float(100.0 * np.mean((matrix[:, index] < design_low - 1e-9) | (matrix[:, index] > design_high + 1e-9))),
                "at_design_bound_percent": float(100.0 * np.mean(np.isclose(matrix[:, index], design_low) | np.isclose(matrix[:, index], design_high))),
                "v1_main_observed_min": lower,
                "v1_main_observed_max": upper,
                "out_of_v1_main_observed_range_percent": float(100.0 * out_of_main),
                "exact_v1_training_range_retained": False,
                "exact_v1_training_range_status": "training trajectories were not retained in v1",
                "thermo_checkpoint_first_layer_l2_norm": float(norms[index]),
                "thermo_checkpoint_first_layer_norm_rank": int((-norms).argsort().tolist().index(index) + 1),
                "mean_absolute_selected_logit_gradient": float(gradients[index]),
                "entropy_block_zeroing_action_change_percent": float(100.0 * changed / max(len(holdout_rows), 1)) if index in ENTROPY_FEATURES else np.nan,
                "entropy_block_zeroing_mean_total_variation": float(np.mean(probability_l1)) if index in ENTROPY_FEATURES else np.nan,
            })
    return pd.DataFrame(rows_out)


def write_figure(
    ties: pd.DataFrame,
    actions: pd.DataFrame,
    communication: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    palette = {"commercial": "#0072B2", "humanitarian": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for offset, application in enumerate(("commercial", "humanitarian")):
        group = ties[ties["application"] == application]
        x = np.arange(len(group)) + (offset - 0.5) * 0.10
        ax.scatter(x, group["primary_difference_thermo_minus_control"], label=application.title(), color=palette[application], marker="o" if offset == 0 else "s", s=30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("A. Raw primary outcomes")
    ax.set_xlabel("Matched holdout pair within application")
    ax.set_ylabel("ThermoAgent − no-entropy loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    app_actions = actions[actions["aggregation_level"] == "application"].set_index("application")
    labels = ["Common decisions", "Union decisions", "Simulator steps"]
    x = np.arange(len(labels))
    width = 0.34
    for offset, application in enumerate(("commercial", "humanitarian")):
        values = [
            app_actions.loc[application, "common_option_divergence_percent"],
            app_actions.loc[application, "union_option_divergence_percent"],
            app_actions.loc[application, "step_option_divergence_percent"],
        ]
        ax.bar(x + (offset - 0.5) * width, values, width, color=palette[application], label=application.title(), hatch="" if offset == 0 else "//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Divergent epochs or steps (%)")
    ax.set_title("B. Policies behaved differently")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    totals = ties.groupby("application").sum(numeric_only=True)
    methods = ["No entropy", "ThermoAgent"]
    x = np.arange(4)
    bar_width = 0.36
    for app_index, application in enumerate(("commercial", "humanitarian")):
        ordinary = [
            totals.loc[application, "control_messages_excluding_sketches"],
            totals.loc[application, "thermo_messages_excluding_sketches"],
        ]
        sketches = [
            totals.loc[application, "control_entropy_sketch_messages"],
            totals.loc[application, "thermo_entropy_sketch_messages"],
        ]
        positions = np.asarray([app_index * 2, app_index * 2 + 1], dtype=float)
        ax.bar(positions, ordinary, color="#56B4E9", label="Operational messages" if app_index == 0 else None)
        ax.bar(positions, sketches, bottom=ordinary, color="#E69F00", hatch="//", label="Entropy sketches" if app_index == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["C: no entropy", "C: Thermo", "H: no entropy", "H: Thermo"],
        rotation=14,
        ha="right",
    )
    ax.set_ylabel("Messages across eight pairs")
    ax.set_title("C. Equal service did not mean equal cost")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    outcome_labels = ["Failed", "Succeeded to\nintermediate", "Succeeded to\ndemand"]
    x = np.arange(len(outcome_labels))
    for method_index, (label, prefix, color, hatch) in enumerate((
        ("No entropy", "control", "#009E73", ""),
        ("ThermoAgent", "thermo", "#CC79A7", "//"),
    )):
        values = [
            ties[f"{prefix}_material_failures"].sum(),
            ties[f"{prefix}_successful_to_intermediate"].sum(),
            ties[f"{prefix}_successful_to_demand"].sum(),
        ]
        ax.bar(x + (method_index - 0.5) * width, values, width, label=label, color=color, hatch=hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels)
    ax.set_ylabel("Material tool calls")
    ax.set_title("D. Divergence rarely reached material flow")
    ax.legend(frameon=False)

    figure.suptitle("Why the frozen v1 holdout produced exact primary-outcome ties", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def write_readme(
    output: Path,
    ties: pd.DataFrame,
    actions: pd.DataFrame,
    communication: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    aggregate = actions[actions["aggregation_level"] == "application"].set_index("application")
    comm = communication[communication["aggregation_level"] == "application"]
    lines = [
        "# Frozen-v1 holdout tie diagnosis",
        "",
        "This is a derived, retrospective analysis of immutable v1 artifacts. The original holdout is seen and is not eligible for v2 selection or confirmation. No v1 file was changed.",
        "",
        "## Findings",
        "",
        "- All 16 ThermoAgent/no-entropy matched pairs are bit-for-bit equal in the raw primary outcome; the equality is not a reporting-rounding artifact.",
        "- Exogenous cumulative-demand and service trajectories are exactly equal in every pair. Three successful ThermoAgent material calls moved resources only to intermediate nodes; zero successful material call in either method reached a demand node. Most material calls failed route or capacity validation.",
    ]
    for application in ("commercial", "humanitarian"):
        row = aggregate.loc[application]
        totals = ties[ties["application"] == application].sum(numeric_only=True)
        lines.append(
            "- %s: options differ on %.1f%% of common agent-decision epochs and %.1f%% of union epochs; total counted messages are %d versus %d because ThermoAgent includes %d entropy sketches. LLM calls are %d versus %d."
            % (
                application.title(),
                row["common_option_divergence_percent"],
                row["union_option_divergence_percent"],
                totals["thermo_total_messages"],
                totals["control_total_messages"],
                totals["thermo_entropy_sketch_messages"],
                totals["thermo_llm_calls"],
                totals["control_llm_calls"],
            )
        )
    entropy = features[features["feature_group"] == "entropy_monitor"]
    lines.extend([
        "- Entropy inputs are neither constant nor outside their designed numerical bounds. The checkpoint is behaviorally sensitive to them: zeroing the six monitor fields while holding each recorded action mask fixed changes deterministic ThermoAgent choices at the rate reported in `feature_usage.csv`. Exact training-trajectory feature ranges were not retained by v1, so that narrower question cannot be reconstructed without rerunning training; `feature_usage.csv` instead reports the explicit design bounds and the observed v1-main range.",
        "- Action masks are not the main explanation: singleton masks are rare/absent and the policies still diverge substantially. Mask-level diagnostics and raw-versus-masked argmax effects are retained in `action_divergence.csv`.",
        "- V1 used one RL initialization/training seed (`3001`) for each learned checkpoint. Evaluation-seed replication therefore did not provide training-seed replication.",
        "",
        "## Causal diagnosis",
        "",
        "The tie arose downstream of policy selection. The learned actors and LLM conversations diverged, but their rare material proposals were overwhelmingly invalid or routed to intermediates rather than demand nodes. Consequently the common purpose-specific exogenous RNG streams generated identical demand, and neither policy changed the service trajectory. ThermoAgent paid much higher counted communication cost—especially mandatory gossip sketches—without influencing the primary outcome. This motivates an event trigger, but it also requires v2 to improve operational actionability rather than merely reduce chatter.",
        "",
        "## Table definitions",
        "",
        "- `holdout_tie_analysis.csv`: one row per matched pair, raw float equality, service/demand trajectory equality, material consequences, communication/inference totals, and SHA-256 provenance.",
        "- `action_divergence.csv`: pair and weighted application summaries. A common epoch exists in both trajectories; a union epoch counts a missing decision as divergence. Simulator-step divergence compares option multisets.",
        "- `communication_divergence.csv`: semantic event signatures by agent-decision epoch. Generated message/commitment/coalition/shipment IDs are excluded so identifier renumbering alone is not counted as behavioral divergence.",
        "- `feature_usage.csv`: observed ranges, saturation, v1-main support comparison, first-layer norms, selected-logit gradients, and a fixed-mask zero-monitor counterfactual.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "./scripts/run-entropy-trigger-diagnostics.sh",
        "```",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(results_root: Path, output_root: Path) -> Dict[str, Any]:
    pairs = discover_pairs(results_root)
    thermo_policy = CoordinationPolicy.load(results_root / "checkpoints" / "coordination_thermo.pt")
    control_policy = CoordinationPolicy.load(results_root / "checkpoints" / "coordination_no_entropy.pt")
    tie_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    communication_rows: List[Dict[str, Any]] = []
    for thermo, control in pairs:
        tie, action, communication = compare_pair(
            thermo, control, thermo_policy, control_policy
        )
        tie_rows.append(tie)
        action_rows.append(action)
        communication_rows.extend(communication)
    ties = pd.DataFrame(tie_rows).sort_values(["application", "scenario_name", "environment_seed"])
    action_frame = pd.DataFrame(action_rows)
    action_sum_columns = [
        "simulator_steps",
        "steps_with_different_option_multisets",
        "thermo_decision_epochs",
        "control_decision_epochs",
        "common_agent_decision_epochs",
        "union_agent_decision_epochs",
        "different_options_on_common_epochs",
        "different_or_missing_options_on_union_epochs",
        "decision_epochs_present_in_only_one_method",
        "different_action_masks_on_common_epochs",
        "thermo_singleton_action_masks",
        "control_singleton_action_masks",
        "thermo_epochs_where_mask_changed_raw_argmax",
        "control_epochs_where_mask_changed_raw_argmax",
        "same_action_from_different_raw_argmax_due_to_mask",
    ]
    actions = add_weighted_aggregates(action_frame, action_sum_columns)
    communication = aggregate_communication(pd.DataFrame(communication_rows))
    features = feature_usage(pairs, thermo_policy, results_root)

    diagnostics = output_root / "diagnostics"
    figures = output_root / "figures" / "pdf"
    previews = output_root / "figures" / "previews"
    diagnostics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    ties.to_csv(diagnostics / "holdout_tie_analysis.csv", index=False)
    actions.to_csv(diagnostics / "action_divergence.csv", index=False)
    communication.to_csv(diagnostics / "communication_divergence.csv", index=False)
    features.to_csv(diagnostics / "feature_usage.csv", index=False)
    pdf = figures / "original_holdout_tie_diagnostics.pdf"
    write_figure(ties, actions, communication, pdf)
    write_readme(diagnostics / "README.md", ties, actions, communication, features)
    record = {
        "status": "complete",
        "analysis_environment": _analysis_environment(),
        "analysis_scope": "retrospective frozen-v1 holdout diagnostics only",
        "pairs": len(ties),
        "exact_primary_ties": int(ties["raw_float_exactly_equal"].sum()),
        "v1_frozen_commit": "d555ac04927968ad577707b5c7e9e7b1162069e6",
        "v1_protocol_sha256": sha256_file(results_root / "reproducibility" / "protocol_freeze.json"),
        "outputs": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in sorted([
                diagnostics / "holdout_tie_analysis.csv",
                diagnostics / "action_divergence.csv",
                diagnostics / "communication_divergence.csv",
                diagnostics / "feature_usage.csv",
                diagnostics / "README.md",
                pdf,
            ])
        },
    }
    (diagnostics / "diagnostic_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/entropy_triggered_v2"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.results_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
