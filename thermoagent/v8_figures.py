"""Focused, source-data-backed publication figures for V8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

from .v5_experiments import write_csv
from .v7_experiments import make_environment
from .v8_io import read_csv_gzip


COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#222222", "gray": "#777777",
}
APP_COLORS = {"humanitarian": COLORS["blue"], "utility_restoration": COLORS["orange"]}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.labelsize": 11.0, "axes.titlesize": 11.0,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.8, "lines.markersize": 6,
    })


def _save(fig: Any, results_root: Path, name: str) -> None:
    pdf = results_root / "figures" / "pdf" / (name + ".pdf")
    png = results_root / "figures" / "png" / (name + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _source(results_root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(results_root / "figures" / "source_data" / (name + ".csv"), rows)


def _evidence_stage(results_root: Path) -> str:
    if (results_root / "statistics" / "v8_pilot_no_go_summary.json").exists():
        return "hysteresis_repair_pilot_v3"
    for stage in ("holdout", "validation", "development_agent", "development"):
        if (results_root / stage / "episode_summary.csv").exists():
            return stage
    if (results_root / "hysteresis_repair_pilot_v3" / "episode_summary.csv").exists():
        return "hysteresis_repair_pilot_v3"
    raise RuntimeError("no V8 evidence stage is available for figures")


def _display_trigger(results_root: Path) -> str:
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    if protocol_path.exists():
        return str(json.loads(protocol_path.read_text())["primary_trigger"])
    no_go = results_root / "statistics" / "v8_pilot_no_go_summary.json"
    if no_go.exists():
        return str(json.loads(no_go.read_text())["diagnostic_candidate_not_frozen_primary"])
    return "generalized_0125_u8"


def _frame(results_root: Path, stage: str) -> pd.DataFrame:
    frame = pd.read_csv(results_root / stage / "episode_summary.csv")
    registry = pd.read_csv(results_root / stage / "candidate_registry.csv")
    return frame.merge(
        registry[["candidate_name", "configuration_digest", "encoding"]],
        left_on=["trigger_configuration_digest", "encoding"],
        right_on=["configuration_digest", "encoding"], how="left",
    )


def architecture_figure(results_root: Path) -> None:
    name = "v8_belief_monitoring_architecture"
    nodes = [
        (0.11, 0.76, "Private observation\nand belief", "agent_private", COLORS["blue"]),
        (0.37, 0.76, "Local generalized-\ninformation trigger", "agent_private", COLORS["green"]),
        (0.63, 0.76, "Deterministic binary\nbelief sketch", "wire", COLORS["purple"]),
        (0.89, 0.76, "Lossy ad-hoc\nnetwork", "wire", COLORS["orange"]),
        (0.78, 0.36, "Recipient-local\ndistributed estimate", "agent_private", COLORS["green"]),
        (0.48, 0.36, "Independent decentralized\naction policy", "agent_private", COLORS["blue"]),
        (0.18, 0.36, "Dynamic service and\nsafety outcome", "environment", COLORS["red"]),
        (0.48, 0.07, "Evaluator-only global truth\n(scoring; never deployed)", "evaluator", COLORS["gray"]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (4, 7), (6, 7)]
    rows = [
        {"record_type": "node", "id": index, "x": x, "y": y,
         "label": label.replace("\n", " "), "boundary": boundary}
        for index, (x, y, label, boundary, _) in enumerate(nodes)
    ] + [
        {"record_type": "edge", "id": "%d-%d" % edge, "source": edge[0], "target": edge[1]}
        for edge in edges
    ]
    _source(results_root, name, rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    patches = []
    for x, y, label, _, color in nodes:
        patch = FancyBboxPatch(
            (x - 0.105, y - 0.065), 0.21, 0.13,
            boxstyle="round,pad=0.012", facecolor="white",
            edgecolor=color, linewidth=2.0,
        )
        patches.append(patch); ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.2)
    for source, target in edges:
        evaluator_edge = target == 7
        connection = "arc3,rad=-0.28" if (source, target) == (6, 0) else "arc3,rad=0"
        arrow = FancyArrowPatch(
            nodes[source][:2], nodes[target][:2], patchA=patches[source],
            patchB=patches[target], arrowstyle="->", mutation_scale=12,
            color="#666666", linewidth=1.4,
            linestyle="--" if evaluator_edge else "-",
            connectionstyle=connection,
        )
        ax.add_patch(arrow)
    ax.text(0.01, 0.95, "Deployable local-to-network path", color=COLORS["green"], weight="bold")
    ax.text(0.67, 0.08, "offline scoring only", color=COLORS["gray"], fontsize=9)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.03, 1.0)); ax.axis("off")
    _save(fig, results_root, name)


def application_networks(results_root: Path) -> None:
    name = "v8_application_network_snapshots"
    specifications = [
        ("humanitarian", "small_world", 887101),
        ("utility_restoration", "grid", 887151),
    ]
    source_rows: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, (application, topology, seed) in zip(axes, specifications):
        environment = make_environment(
            application, "medium", "high", "high", "high", topology,
            seed, "private_fragmented", sketch_policy="none",
        )
        for step in range(environment.disruption_step + 1):
            environment.advance_domain(step)
        environment.deliver_private_observations(environment.disruption_step)
        graph = environment.operational_graph if application == "humanitarian" else environment.physical_graph
        position = nx.spring_layout(graph, seed=88)
        uncertainties = []
        for node in graph.nodes:
            asset = graph.nodes[node].get("asset_id")
            scoped = [
                agent for agent in environment.agents.values()
                if asset in agent.private_beliefs
            ]
            if scoped:
                belief = np.mean([agent.private_beliefs[asset] for agent in scoped], axis=0)
                uncertainty = -float(np.sum(belief * np.log(np.maximum(belief, 1e-12)))) / np.log(len(belief))
            else:
                uncertainty = 0.0
            uncertainties.append(uncertainty)
            source_rows.append({
                "application": application, "record_type": "node", "node": node,
                "x": float(position[node][0]), "y": float(position[node][1]),
                "role": graph.nodes[node].get("role", "node"),
                "uncertainty": uncertainty,
            })
        for first, second, data in graph.edges(data=True):
            available = bool(data.get("route_available", data.get("service_available", True)))
            source_rows.append({
                "application": application, "record_type": "edge",
                "source": first, "target": second, "available": available,
            })
            ax.plot(
                [position[first][0], position[second][0]],
                [position[first][1], position[second][1]],
                color="#BBBBBB" if available else COLORS["red"],
                ls="-" if available else "--", lw=1.0 if available else 2.0,
                zorder=1,
            )
        scatter = ax.scatter(
            [position[node][0] for node in graph.nodes],
            [position[node][1] for node in graph.nodes],
            c=uncertainties, cmap="viridis", vmin=0, vmax=1,
            s=48, edgecolors="white", linewidths=0.5, zorder=2,
        )
        ax.set_title("Humanitarian logistics" if application == "humanitarian" else "Utility restoration")
        ax.axis("off")
    fig.colorbar(scatter, ax=axes, shrink=0.72, label="local belief uncertainty")
    axes[0].legend(handles=[
        Line2D([0], [0], color="#BBBBBB", lw=1.5, label="available edge"),
        Line2D([0], [0], color=COLORS["red"], lw=2, ls="--", label="disrupted edge"),
    ], frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Illustrative simulated disruption states (not effect estimates)", fontsize=10)
    _source(results_root, name, source_rows)
    _save(fig, results_root, name)


def communication_frontier(results_root: Path) -> None:
    name = "v8_communication_estimation_frontier"
    stage = (
        "ablations" if (results_root / "ablations" / "episode_summary.csv").exists()
        else "hysteresis_repair_pilot_v3"
        if (results_root / "statistics" / "v8_pilot_no_go_summary.json").exists()
        else "development"
    )
    frame = _frame(results_root, stage)
    rows: List[Dict[str, Any]] = []
    for (application, candidate), values in frame.groupby(["application", "candidate_name"]):
        x = values.sketch_on_wire_bytes.to_numpy(dtype=float)
        y = values.primary_distributed_state_error.to_numpy(dtype=float)
        rows.append({
            "stage": stage, "application": application, "candidate": candidate,
            "independent_panels": int(values.environment_seed.nunique()),
            "mean_wire_bytes": float(x.mean()), "wire_bytes_se": float(x.std(ddof=1) / np.sqrt(len(x))),
            "mean_primary_error": float(y.mean()), "primary_error_se": float(y.std(ddof=1) / np.sqrt(len(y))),
        })
    data = pd.DataFrame(rows)
    protocol = (
        json.loads((results_root / "protocol" / "v8_frozen_protocol.json").read_text())
        if (results_root / "protocol" / "v8_frozen_protocol.json").exists()
        else {}
    )
    annotated = {
        "always_on_u8", "none_u8",
        protocol.get("primary_trigger", _display_trigger(results_root)),
        protocol.get("strongest_nonentropic_comparator", "kpi_012_u8"),
    }
    data["nondominated"] = False
    for application, values in data.groupby("application"):
        for index, row in values.iterrows():
            dominated = ((values.mean_wire_bytes <= row.mean_wire_bytes)
                         & (values.mean_primary_error <= row.mean_primary_error)
                         & ((values.mean_wire_bytes < row.mean_wire_bytes)
                            | (values.mean_primary_error < row.mean_primary_error))).any()
            data.loc[index, "nondominated"] = not bool(dominated)
    _source(results_root, name, data.to_dict("records"))
    candidate_order = [
        "none_u8", "always_on_u8", "kpi_012_u8",
        "generalized_011_u8", "generalized_0115_u8",
    ]
    candidate_labels = {
        "none_u8": "no exchange", "always_on_u8": "always-on",
        "kpi_012_u8": "KPI change", "generalized_011_u8": "generalized tau=0.110",
        "generalized_0115_u8": "generalized tau=0.115",
    }
    markers = {name: marker for name, marker in zip(candidate_order, ("X", "P", "s", "o", "^"))}
    method_colors = {name: color for name, color in zip(
        candidate_order, (COLORS["gray"], COLORS["black"], COLORS["green"], COLORS["sky"], COLORS["purple"])
    )}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, application in zip(axes, ("humanitarian", "utility_restoration")):
        values = data[data.application.eq(application)]
        for candidate in candidate_order:
            row = values[values.candidate.eq(candidate)]
            if row.empty:
                continue
            value = row.iloc[0]
            ax.errorbar(
                value.mean_wire_bytes, value.mean_primary_error,
                xerr=1.96 * value.wire_bytes_se, yerr=1.96 * value.primary_error_se,
                fmt=markers[candidate], color=method_colors[candidate],
                ecolor="#AAAAAA", capsize=2, label=candidate_labels[candidate],
            )
        frontier = values[values.nondominated].sort_values("mean_wire_bytes")
        ax.plot(frontier.mean_wire_bytes, frontier.mean_primary_error, "--", color=COLORS["black"], label="nondominated")
        ax.set_title(application.replace("_", " ")); ax.set_xlabel("actual sketch bytes / panel")
    axes[0].set_ylabel("distributed-state error")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Development pilot scheduler tradeoff (n=6 panels/application)", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); _save(fig, results_root, name)


def disruption_trajectory(results_root: Path) -> None:
    name = "v8_disruption_partition_trajectory"
    stage = _evidence_stage(results_root)
    frame = _frame(results_root, stage)
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    selected = _display_trigger(results_root)
    candidates = frame[
        frame.candidate_name.eq(selected) & frame.complexity.eq("medium")
        & frame.fragmentation.eq("high")
    ].sort_values(["application", "environment_seed"])
    rows: List[Dict[str, Any]] = []
    for application in ("humanitarian", "utility_restoration"):
        match = candidates[candidates.application.eq(application)]
        if match.empty:
            continue
        run_id = str(match.iloc[0].run_id)
        metadata = match.iloc[0]
        environment = make_environment(
            application, str(metadata.complexity), str(metadata.coupling),
            str(metadata.fragmentation), str(metadata.network_disruption),
            str(metadata.topology_family), int(metadata.environment_seed),
            str(metadata.information_condition), sketch_policy="none",
        )
        run_dir = results_root / "raw" / stage / run_id
        if not run_dir.exists():
            continue
        estimates = pd.DataFrame(read_csv_gzip(run_dir / "estimation.csv.gz"))
        triggers = pd.DataFrame(read_csv_gzip(run_dir / "triggers.csv.gz"))
        numeric = [
            "step", "distributed_disagreement", "evaluator_global_disagreement",
            "belief_mae", "disagreement_absolute_error",
        ]
        for column in numeric:
            estimates[column] = pd.to_numeric(estimates[column])
        triggers["step"] = pd.to_numeric(triggers["step"])
        triggers["transmit"] = triggers["transmit"].astype(str).str.lower().eq("true")
        estimates["evaluator_disrupted"] = estimates["evaluator_disrupted"].astype(str).str.lower().eq("true")
        aggregate = estimates.groupby("step").agg({
            "distributed_disagreement": "mean", "evaluator_global_disagreement": "mean",
            "belief_mae": "mean", "disagreement_absolute_error": "mean",
            "evaluator_disrupted": "max",
        }).reset_index()
        transmissions = triggers.groupby("step").transmit.sum()
        for _, value in aggregate.iterrows():
            rows.append({
                "application": application, "stage": stage, "run_id": run_id,
                **value.to_dict(), "transmissions": int(transmissions.get(value.step, 0)),
                "disruption_step": int(environment.disruption_step),
                "recovery_step": int(environment.recovery_step),
            })
    _source(results_root, name, rows)
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex="col")
    for column, application in enumerate(("humanitarian", "utility_restoration")):
        values = data[data.application.eq(application)]
        if not values.empty:
            axes[0, column].axvspan(
                float(values.disruption_step.iloc[0]),
                float(values.recovery_step.iloc[0]),
                color=COLORS["red"], alpha=0.10, label="disruption / partition",
            )
        axes[0, column].plot(values.step, values.evaluator_global_disagreement, color=COLORS["black"], label="evaluator global")
        axes[0, column].plot(values.step, values.distributed_disagreement, color=APP_COLORS[application], ls="--", label="distributed")
        axes[0, column].set_title(application.replace("_", " "))
        axes[1, column].vlines(
            values.step, 0, values.transmissions,
            color=APP_COLORS[application], linewidth=1.2,
        )
        axes[1, column].set_xlabel("simulation step")
    axes[0, 0].set_ylabel("Jensen-Shannon disagreement"); axes[1, 0].set_ylabel("triggered transmissions")
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Representative %s panels (lowest prespecified seed)" % stage.replace("_", " "), fontsize=10)
    fig.tight_layout(); _save(fig, results_root, name)


def primary_forest(results_root: Path) -> None:
    name = "v8_primary_effect_forest"
    stage = _evidence_stage(results_root)
    no_go_path = results_root / "statistics" / "v8_pilot_no_go_intervals.csv"
    path = no_go_path if no_go_path.exists() else results_root / stage / "primary_bootstrap_intervals.csv"
    if no_go_path.exists():
        stage = "development pilot no-go"
    if not path.exists():
        stage = "development"
        path = results_root / stage / "panel_bootstrap_intervals.csv"
    data = pd.read_csv(path)
    metrics = [
        value for value in (
            "message_reduction", "wire_byte_reduction", "sketch_byte_reduction",
            "primary_estimation_error_increase", "primary_error_increase",
            "primary_error_advantage_vs_comparator", "relative_service_degradation",
            "harmful_action_rate_degradation", "reward_degradation",
        ) if value in set(data.metric)
    ]
    shown = data[data.metric.isin(metrics)].copy()
    protocol = (
        json.loads((results_root / "protocol" / "v8_frozen_protocol.json").read_text())
        if (results_root / "protocol" / "v8_frozen_protocol.json").exists()
        else {}
    )
    margins = protocol.get("primary_margins", {})
    margin_map = {
        "message_reduction": margins.get("H1_message_reduction_lower_95", 0.25),
        "wire_byte_reduction": margins.get("H1_wire_byte_reduction_lower_95", 0.25),
        "sketch_byte_reduction": margins.get("H1_wire_byte_reduction_lower_95", 0.25),
        "primary_estimation_error_increase": margins.get("H1_primary_error_increase_upper_95", 0.02),
        "primary_error_increase": margins.get("H1_primary_error_increase_upper_95", 0.02),
        "primary_error_advantage_vs_comparator": margins.get("H2_practical_primary_error_advantage", 0.001),
        "relative_service_degradation": margins.get("H3_relative_service_degradation_upper_95", 0.02),
        "harmful_action_rate_degradation": margins.get("H3_harmful_action_rate_degradation_upper_95", 0.02),
        "reward_degradation": margins.get("H3_reward_degradation_upper_95", 0.02),
    }
    shown["decision_margin"] = shown.metric.map(margin_map)
    shown["pass_direction"] = shown.metric.map(lambda value: (
        "lower bound >= margin" if value in (
            "message_reduction", "wire_byte_reduction",
            "sketch_byte_reduction",
            "primary_error_advantage_vs_comparator",
        ) else "upper bound <= margin"
    ))
    shown["label"] = shown.application.str.replace("_", " ") + " — " + shown.metric.str.replace("_", " ")
    _source(results_root, name, shown.to_dict("records"))
    y = np.arange(len(shown))
    colors = [APP_COLORS[value] for value in shown.application]
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.32 * len(shown))))
    for index, (_, row) in enumerate(shown.iterrows()):
        ax.errorbar(row["mean"], index, xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]], fmt="o", color=colors[index], capsize=2)
        if np.isfinite(row["decision_margin"]):
            ax.plot(row["decision_margin"], index, marker="|", color=COLORS["red"],
                    markersize=11, markeredgewidth=2)
    ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(shown.label); ax.invert_yaxis()
    ax.set_xlabel("paired panel effect; red ticks are predeclared pilot margins")
    ax.set_title("%s evidence; 95%% panel-bootstrap intervals" % stage.replace("_", " "))
    fig.tight_layout(); _save(fig, results_root, name)


def downstream_performance(results_root: Path) -> None:
    name = "v8_downstream_performance_vs_communication"
    stage = _evidence_stage(results_root)
    frame = _frame(results_root, stage)
    rows = []
    for (application, candidate), values in frame.groupby(["application", "candidate_name"]):
        actions = (values.autonomous_beneficial_actions + values.autonomous_harmful_actions + values.autonomous_neutral_actions).clip(lower=1)
        rows.append({
            "stage": stage, "application": application, "candidate": candidate,
            "independent_panels": int(values.environment_seed.nunique()),
            "mean_wire_bytes": float(values.sketch_on_wire_bytes.mean()),
            "mean_service_loss": float(values.service_loss.mean()),
            "mean_harmful_action_rate": float((values.autonomous_harmful_actions / actions).mean()),
            "mean_autonomous_reward": float(values.normalized_autonomous_reward.mean()),
        })
    data = pd.DataFrame(rows); _source(results_root, name, rows)
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    key_candidates = {
        "always_on_u8", "none_u8",
        protocol.get("primary_trigger", "generalized_0125_u8"),
        protocol.get("strongest_nonentropic_comparator", "kpi_012_u8"),
    }
    candidate_order = ["none_u8", "always_on_u8", "kpi_012_u8", "generalized_011_u8", "generalized_0115_u8"]
    labels = {"none_u8": "none", "always_on_u8": "always-on", "kpi_012_u8": "KPI", "generalized_011_u8": "gen .110", "generalized_0115_u8": "gen .115"}
    markers = {name: marker for name, marker in zip(candidate_order, ("X", "P", "s", "o", "^"))}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex="row")
    metrics = ("mean_service_loss", "mean_harmful_action_rate", "mean_autonomous_reward")
    for row_index, application in enumerate(("humanitarian", "utility_restoration")):
        values = data[data.application.eq(application)]
        for column, metric in enumerate(metrics):
            ax = axes[row_index, column]
            for candidate in candidate_order:
                point = values[values.candidate.eq(candidate)]
                if point.empty:
                    continue
                value = point.iloc[0]
                ax.scatter(value.mean_wire_bytes, value[metric], marker=markers[candidate],
                           color=APP_COLORS[application], edgecolor="white", linewidth=0.5,
                           label=labels[candidate])
            ax.set_ylabel(metric.replace("mean_", "").replace("_", " "))
            if row_index == 1:
                ax.set_xlabel("actual sketch bytes")
            if column == 0:
                ax.text(0.02, 0.94, application.replace("_", " "), transform=ax.transAxes,
                        va="top", fontsize=9.5, weight="bold")
    handles, legend_labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    title = (
        "Deterministic-rule policy diagnostics (development pilot; n=6/application)"
        if "pilot" in stage else "Frozen-policy outcomes (%s)" % stage.replace("_", " ")
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); _save(fig, results_root, name)


def scaling_topology(results_root: Path) -> None:
    name = "v8_scaling_topology_robustness"
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    selected_name = _display_trigger(results_root)
    # The final development stage contains both prospectively eligible trigger
    # candidates and the repaired two-hop routing semantics.  Prefer it after
    # it is complete; retain the broader first development grid as a fallback
    # for pre-freeze figure smoke tests.
    stage = (
        "hysteresis_repair_pilot_v3"
        if (results_root / "statistics" / "v8_pilot_no_go_summary.json").exists()
        else
        "development_final"
        if (results_root / "development_final" / "episode_summary.csv").exists()
        else "development"
    )
    frame = _frame(results_root, stage)
    if selected_name not in set(frame.candidate_name.dropna()):
        frame = _frame(results_root, "development")
    selected = frame[frame.candidate_name.eq(selected_name)]
    if selected.empty:
        raise RuntimeError("selected V8 trigger is absent from scaling source data")
    rows = []
    for keys, values in selected.groupby(["application", "complexity", "topology_family"]):
        rows.append({
            "application": keys[0], "complexity": keys[1], "topology_family": keys[2],
            "independent_panels": int(values.environment_seed.nunique()),
            "mean_agent_count": float(values.agent_count.mean()),
            "mean_wire_bytes_per_agent": float((values.sketch_on_wire_bytes / values.agent_count).mean()),
            "mean_primary_error": float(values.primary_distributed_state_error.mean()),
            "mean_recovery_steps": float(values.consensus_recovery_steps.mean()),
        })
    data = pd.DataFrame(rows); _source(results_root, name, rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for application, values in data.groupby("application"):
        ordered = values.groupby("mean_agent_count").mean(numeric_only=True).reset_index()
        axes[0].plot(ordered.mean_agent_count, ordered.mean_wire_bytes_per_agent, "o-", color=APP_COLORS[application], label=application.replace("_", " "))
        axes[1].plot(ordered.mean_agent_count, ordered.mean_primary_error, "s--", color=APP_COLORS[application])
        axes[2].plot(ordered.mean_agent_count, ordered.mean_recovery_steps, "^-.", color=APP_COLORS[application])
    axes[0].set_ylabel("wire bytes / agent"); axes[1].set_ylabel("distributed-state error"); axes[2].set_ylabel("recovery steps")
    for ax in axes: ax.set_xlabel("agents")
    axes[0].legend(frameon=False)
    fig.suptitle("Development pilot scaling diagnostic (descriptive panel means)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); _save(fig, results_root, name)


def trigger_ablation(results_root: Path) -> None:
    no_go = (results_root / "statistics" / "v8_pilot_no_go_summary.json").exists()
    name = "v8_trigger_feasibility_no_go" if no_go else "v8_trigger_ablation"
    stage = (
        "ablations" if (results_root / "ablations" / "episode_summary.csv").exists()
        else "hysteresis_repair_pilot_v3" if no_go else "development"
    )
    frame = _frame(results_root, stage)
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    selected_trigger = _display_trigger(results_root)
    if no_go:
        selection = json.loads(
            (results_root / stage / "hysteresis_repair_selection.json").read_text()
        )
        rows = []
        for candidate in selection["candidates"]:
            for application in candidate["applications"]:
                rows.append({
                    "stage": stage, "application": application["application"],
                    "candidate": candidate["candidate_name"],
                    "independent_panels": application["independent_panels"],
                    "mean_activation_rate": application["activation_rate_mean"],
                    "information_score_fraction": application["information_score_fraction"],
                    "pre_disruption_transmission_rate": application[
                        "pre_disruption_noninitial_transmission_rate"
                    ],
                })
        data = pd.DataFrame(rows); _source(results_root, name, rows)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        candidate_markers = {"generalized_011_u8": "o", "generalized_0115_u8": "^"}
        for application, values in data.groupby("application"):
            for _, row in values.iterrows():
                marker = candidate_markers[row.candidate]
                axes[0].scatter(
                    row.information_score_fraction,
                    row.pre_disruption_transmission_rate,
                    color=APP_COLORS[application], marker=marker,
                )
                axes[1].scatter(
                    row.mean_activation_rate,
                    row.pre_disruption_transmission_rate,
                    color=APP_COLORS[application], marker=marker,
                )
        axes[0].axvline(0.05, color=COLORS["green"], ls="--", label="information gate")
        axes[0].axhline(0.10, color=COLORS["red"], ls=":", label="nominal limit")
        axes[1].axhline(0.10, color=COLORS["red"], ls=":")
        axes[0].set(xlabel="information-score share of noninitial traffic", ylabel="pre-disruption transmission rate")
        axes[1].set(xlabel="overall trigger activation rate", ylabel="pre-disruption transmission rate")
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=APP_COLORS["humanitarian"], label="humanitarian", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=APP_COLORS["utility_restoration"], label="utility restoration", markersize=7),
            Line2D([0], [0], marker="o", color=COLORS["gray"], lw=0, label="tau=0.110", markersize=7),
            Line2D([0], [0], marker="^", color=COLORS["gray"], lw=0, label="tau=0.115", markersize=7),
            Line2D([0], [0], color=COLORS["green"], ls="--", label="information gate"),
            Line2D([0], [0], color=COLORS["red"], ls=":", label="nominal limit"),
        ]
        axes[0].legend(handles=legend_handles, frameon=False, loc="center left", fontsize=8.8)
        fig.suptitle("Development pilot: information-driven but nominally overactive", fontsize=10)
        fig.tight_layout(); _save(fig, results_root, name)
        return
    candidates = (
        "v7_shannon_006_u8", "spectrum_008_u8", "js_007_u8",
        "aoi_30_u8", "uncertainty_008_u8", selected_trigger,
    )
    rows = []
    for (application, candidate), values in frame[frame.candidate_name.isin(candidates)].groupby(["application", "candidate_name"]):
        rows.append({
            "stage": stage, "application": application, "candidate": candidate,
            "independent_panels": int(values.environment_seed.nunique()),
            "mean_wire_bytes": float(values.sketch_on_wire_bytes.mean()),
            "mean_primary_error": float(values.primary_distributed_state_error.mean()),
            "mean_detection_delay": float(values.mean_detection_delay_steps.mean()),
            "mean_activation_rate": float(values.trigger_activation_rate.mean()),
        })
    data = pd.DataFrame(rows); _source(results_root, name, rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for application, values in data.groupby("application"):
        axes[0].scatter(values.mean_wire_bytes, values.mean_primary_error, color=APP_COLORS[application], label=application.replace("_", " "))
        axes[1].scatter(values.mean_activation_rate, values.mean_detection_delay, color=APP_COLORS[application])
        for _, row in values.iterrows():
            short = row.candidate.replace("_u8", "")
            if row.candidate == selected_trigger:
                short = "full selected"
            axes[0].annotate(short, (row.mean_wire_bytes, row.mean_primary_error), xytext=(2, 2), textcoords="offset points", fontsize=9.0)
    axes[0].set(xlabel="actual sketch bytes", ylabel="distributed-state error")
    axes[1].set(xlabel="activation rate", ylabel="detection delay (steps)")
    axes[0].legend(frameon=False)
    fig.suptitle("Development-only trigger ablations", fontsize=10)
    fig.tight_layout(); _save(fig, results_root, name)


def generate_v8_figures(results_root: Path) -> List[str]:
    configure_style()
    architecture_figure(results_root)
    application_networks(results_root)
    communication_frontier(results_root)
    disruption_trajectory(results_root)
    primary_forest(results_root)
    downstream_performance(results_root)
    scaling_topology(results_root)
    trigger_ablation(results_root)
    return sorted(value.name for value in (results_root / "figures" / "pdf").glob("*.pdf"))
