"""Data-derived vector figures for the V7 development or later stages."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from .v5_experiments import atomic_json, write_csv
from .v7_io import read_csv_artifact
from .v7_protocol import development_manifest
from .v7_topology import generate_graph, topology_diagnostics


PALETTE = {
    "humanitarian": "#0072B2", "utility_restoration": "#D55E00",
    "baseline": "#6B7280", "entropic": "#009E73", "warning": "#CC79A7",
    "neutral": "#E69F00", "dark": "#26384A",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.5, "axes.titlesize": 12.0,
        "legend.fontsize": 9.5, "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(
    results_root: Path, name: str, figure: plt.Figure,
    source_rows: Sequence[Mapping[str, Any]], description: str,
) -> Dict[str, Any]:
    pdf = results_root / "figures" / "pdf" / (name + ".pdf")
    png = results_root / "figures" / "png" / (name + ".png")
    source = results_root / "figures" / "source_data" / (name + ".csv")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    write_csv(source, list(source_rows))
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {
        "figure": name, "pdf": str(pdf.relative_to(results_root)),
        "preview": str(png.relative_to(results_root)),
        "source_data": str(source.relative_to(results_root)),
        "description": description,
    }


def _pilot_candidates(results_root: Path) -> pd.DataFrame:
    return read_csv_artifact(results_root / "pilots_iteration3" / "candidate_decisions.csv")


def _pilot_effects(results_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        results_root / "pilots_iteration3" / "analysis" / "paired_incremental_effects.csv"
    )


def _effect_data(results_root: Path) -> pd.DataFrame:
    """Use formal matched dynamic panels when available, pilots otherwise."""
    formal = results_root / "statistics" / "dynamic_paired_panel_effects.csv"
    if formal.exists():
        frame = pd.read_csv(formal)
        frame = frame.rename(columns={"harm_reduction": "incremental_harm_reduction"})
        frame["evidence_stage"] = "formal development"
        return frame
    frame = _pilot_effects(results_root)
    frame["evidence_stage"] = "retained feasibility pilot"
    return frame


def _reference_candidates(results_root: Path) -> pd.DataFrame:
    formal = (
        results_root / "development_formal_reference" / "risk_analysis"
        / "crossfit_predictions.csv"
    )
    if formal.exists():
        return pd.read_csv(formal)
    return _pilot_candidates(results_root)


def _reference_episodes(results_root: Path) -> pd.DataFrame:
    formal = results_root / "development_formal_reference" / "episode_summary.csv"
    if formal.exists():
        return pd.read_csv(formal)
    return pd.read_csv(results_root / "pilots_iteration3" / "episode_summary.csv")


def _evidence_label(results_root: Path) -> str:
    if (results_root / "statistics" / "dynamic_primary_analysis.json").exists():
        return "Formal development (not validation or holdout)"
    return "Retained feasibility pilot (not formal evidence)"


def _complexity_comparison(results_root: Path) -> Dict[str, Any]:
    audit = pd.read_csv(results_root / "development" / "audits" / "v6_complexity_audit.csv")
    rows = audit.to_dict("records")
    # Store explicit V7 values beside the machine-audited V6 values.
    values = [
        {"study": "V6", "metric": "agents", "value": 12},
        {"study": "V6", "metric": "horizon", "value": 12},
        {"study": "V6", "metric": "decision epochs", "value": 6},
        {"study": "V6", "metric": "operational nodes", "value": 4},
        {"study": "V7 small", "metric": "agents", "value": 12},
        {"study": "V7 small", "metric": "horizon", "value": 30},
        {"study": "V7 small", "metric": "decision epochs", "value": 10},
        {"study": "V7 small", "metric": "operational nodes", "value": 8},
        {"study": "V7 medium", "metric": "agents", "value": 28},
        {"study": "V7 medium", "metric": "horizon", "value": 60},
        {"study": "V7 medium", "metric": "decision epochs", "value": 15},
        {"study": "V7 medium", "metric": "operational nodes", "value": 16},
        {"study": "V7 large", "metric": "agents", "value": 52},
        {"study": "V7 large", "metric": "horizon", "value": 100},
        {"study": "V7 large", "metric": "decision epochs", "value": 20},
        {"study": "V7 large", "metric": "operational nodes", "value": 30},
    ]
    frame = pd.DataFrame(values)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.5))
    colors = ["#9CA3AF", "#56B4E9", "#0072B2", "#D55E00"]
    for axis, (metric, subset) in zip(axes, frame.groupby("metric", sort=True)):
        axis.bar(np.arange(len(subset)), subset.value, color=colors)
        axis.set_xticks(np.arange(len(subset)), subset.study, rotation=32, ha="right")
        axis.set_title(metric.capitalize())
        axis.set_ylabel("Count")
    fig.suptitle("V7 increases persistent-agent scale, horizon, and coupled state")
    return _save(results_root, "v6_v7_complexity_comparison", fig, values, "Audited V6 versus configured V7 complexity.")


def _network_figure(results_root: Path, application: str) -> Dict[str, Any]:
    family = "modular" if application == "humanitarian" else "grid"
    graph = generate_graph(family, 28, 787777 if application == "humanitarian" else 787778)
    position = nx.spring_layout(graph, seed=7877)
    communities = [int(graph.nodes[node].get("community", 0)) for node in graph]
    rows = []
    for node in graph:
        rows.append({"record_type": "node", "node": node, "x": position[node][0], "y": position[node][1], "community": graph.nodes[node].get("community", 0)})
    for first, second, data in graph.edges(data=True):
        rows.append({"record_type": "edge", "node": first, "target": second, "reliability": data.get("reliability"), "latency": data.get("latency")})
    fig, axis = plt.subplots(figsize=(7.4, 5.4))
    nx.draw_networkx_edges(graph, position, ax=axis, alpha=0.45, edge_color="#64748B")
    nx.draw_networkx_nodes(graph, position, ax=axis, node_color=communities, cmap="viridis", node_size=180, edgecolors="white", linewidths=0.8)
    # Compact IDs are offset from nodes so labels do not collide with the
    # explanatory footer at final two-column paper size.
    label_position = {node: (xy[0], xy[1] + 0.045) for node, xy in position.items()}
    nx.draw_networkx_labels(graph, label_position, ax=axis, font_size=7.5)
    axis.set_title(("Humanitarian logistics" if application == "humanitarian" else "Utility physical–communication–crew") + " multilayer proxy network")
    axis.text(0.01, -0.02, "Stored graph instance; colors denote structural communities", transform=axis.transAxes, fontsize=9.5)
    axis.margins(0.12)
    axis.axis("off")
    name = "humanitarian_multilayer_network" if application == "humanitarian" else "utility_multilayer_network"
    return _save(results_root, name, fig, rows, "Stored medium-size V7 graph instance and structural communities.")


def _architecture(results_root: Path) -> Dict[str, Any]:
    nodes = [
        ("Private observations\nand beliefs", 0.12, 0.72),
        ("Independent\nLevel-1 policies", 0.31, 0.72),
        ("Distributed entropy\nand consensus", 0.54, 0.72),
        ("Level-2 risk and\ncommunication control", 0.77, 0.72),
        ("Execute / communicate /\nabstain / escalate", 0.77, 0.30),
        ("Coupled domain\nstate transitions", 0.31, 0.30),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    rows = [{"record_type": "node", "label": label, "x": x, "y": y} for label, x, y in nodes]
    rows += [{"record_type": "edge", "source": nodes[a][0], "target": nodes[b][0]} for a, b in edges]
    fig, axis = plt.subplots(figsize=(10.5, 4.7))
    for index, (label, x, y) in enumerate(nodes):
        color = PALETTE["entropic"] if index in (2, 3) else PALETTE["humanitarian"]
        axis.text(x, y, label, ha="center", va="center", color="white", fontsize=10.5,
                  bbox=dict(boxstyle="round,pad=0.65", facecolor=color, edgecolor=PALETTE["dark"], linewidth=1.2))
    arrow_segments = [
        ((0.20, 0.72), (0.23, 0.72)), ((0.39, 0.72), (0.45, 0.72)),
        ((0.63, 0.72), (0.68, 0.72)), ((0.77, 0.62), (0.77, 0.40)),
        ((0.68, 0.30), (0.40, 0.30)), ((0.27, 0.36), (0.14, 0.62)),
    ]
    for start, end in arrow_segments:
        axis.annotate("", xy=end, xytext=start,
                      arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], lw=1.4),
                      zorder=1)
    axis.text(0.54, 0.93, "All sketch and operational traffic is explicitly delivered and costed", ha="center", fontsize=10)
    axis.set_xlim(-0.01, 0.92); axis.set_ylim(0.10, 1.0); axis.axis("off")
    axis.set_title("Independent agents with a separate generalized-entropic Level-2 controller")
    return _save(results_root, "v7_independent_agent_entropy_architecture", fig, rows, "V7 decentralized execution and Level-2 risk-control boundary.")


def _coverage_data(frame: pd.DataFrame) -> pd.DataFrame:
    if "harmful_label" in frame:
        accepted = frame.copy()
        accepted["counterfactual_harmful"] = accepted.harmful_label.astype(bool)
        methods = {
            "Strongest non-entropic": "risk_strongest_nonentropic",
            "Shannon + JS": "risk_shannon_js",
            "Generalized entropic": "risk_generalized_entropic",
        }
        group_column = "panel_id"
    else:
        accepted = frame[
            frame.counterfactual_evaluated.astype(bool)
            & frame.counterfactual_action_accepted.astype(bool)
        ].copy()
        methods = {
            "KPI confidence": "risk_kpi_confidence",
            "Predictive uncertainty": "risk_predictive_uncertainty",
            "Shannon + JS": "risk_shannon_js",
            "Generalized entropic": "risk_combined_generalized_entropic",
        }
        group_column = "run_id"
    rows = []
    for (application, panel_id), subset in accepted.groupby(["application", group_column], sort=True):
        for label, column in methods.items():
            for coverage in np.linspace(0.1, 1.0, 10):
                count = max(1, int(round(coverage * len(subset))))
                selected = subset.sort_values(column, kind="mergesort").head(count)
                rows.append({
                    "application": application, "panel_id": panel_id,
                    "method": label, "coverage": coverage,
                    "harm_rate": float(selected.counterfactual_harmful.mean()),
                    "mean_utility": float(selected.counterfactual_causal_utility.mean()),
                })
    return pd.DataFrame(rows)


def _coverage_figure(results_root: Path, metric: str, name: str) -> Dict[str, Any]:
    values = _coverage_data(_reference_candidates(results_root))
    summary = values.groupby(["application", "method", "coverage"])[metric].agg(["mean", "sem"]).reset_index()
    summary["ci_low"] = summary["mean"] - 1.96 * summary["sem"].fillna(0)
    summary["ci_high"] = summary["mean"] + 1.96 * summary["sem"].fillna(0)
    if metric == "harm_rate":
        summary["ci_low"] = summary.ci_low.clip(lower=0.0, upper=1.0)
        summary["ci_high"] = summary.ci_high.clip(lower=0.0, upper=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    styles = {
        "Strongest non-entropic": (PALETTE["baseline"], "o", "-"),
        "KPI confidence": (PALETTE["baseline"], "o", "-"),
        "Predictive uncertainty": (PALETTE["neutral"], "s", "--"),
        "Shannon + JS": (PALETTE["humanitarian"], "^", "-."),
        "Generalized entropic": (PALETTE["entropic"], "D", ":"),
    }
    for axis, application in zip(axes, ("humanitarian", "utility_restoration")):
        for method, subset in summary[summary.application == application].groupby("method"):
            color, marker, line = styles[method]
            axis.plot(subset.coverage, subset["mean"], color=color, marker=marker, linestyle=line, label=method)
            axis.fill_between(subset.coverage, subset.ci_low, subset.ci_high, color=color, alpha=0.12)
        axis.set_title(application.replace("_", " ").title())
        axis.set_xlabel("Autonomous-action coverage")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Harmful-action rate" if metric == "harm_rate" else "Mean causal utility")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(("Risk–coverage" if metric == "harm_rate" else "Utility–coverage") + " · " + _evidence_label(results_root))
    return _save(results_root, name, fig, summary.to_dict("records"), "Panel-level coverage curves from grouped cross-fitted formal-development predictions.")


def _phase_and_interaction(results_root: Path, name: str, mode: str) -> Dict[str, Any]:
    frame = _effect_data(results_root)
    if "information_condition" in frame:
        frame = frame[frame.information_condition.eq("private_fragmented")].copy()
    mapping = {"low": 0, "medium": 1, "high": 2}
    frame["coupling_value"] = frame.coupling.map(mapping)
    frame["fragmentation_value"] = frame.fragmentation.map(mapping)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=(mode == "phase"))
    plotted_rows = []
    for axis, application in zip(axes, ("humanitarian", "utility_restoration")):
        subset = frame[frame.application == application]
        grouped = subset.groupby(["coupling", "fragmentation"]).incremental_harm_reduction.agg(["mean", "sem", "size"]).reset_index()
        grouped["application"] = application
        grouped["coupling_value"] = grouped.coupling.map(mapping)
        grouped["fragmentation_value"] = grouped.fragmentation.map(mapping)
        plotted_rows.extend(grouped.to_dict("records"))
        if mode == "phase":
            scatter = axis.scatter(grouped.fragmentation_value, grouped.coupling_value, c=grouped["mean"], cmap="RdBu", vmin=-0.10, vmax=0.10, s=180, edgecolor="black")
            for _, value in grouped.iterrows():
                label_offset = 14 if value.coupling == "low" else -16
                axis.annotate("n=%d" % int(value["size"]),
                              (value.fragmentation_value, value.coupling_value),
                              xytext=(0, label_offset), textcoords="offset points",
                              ha="center", va="center", fontsize=8.5)
            axis.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_xlabel("Private-information fragmentation")
            axis.set_ylabel("Coupling")
        else:
            for coupling in ("low", "medium", "high"):
                values = grouped[grouped.coupling.eq(coupling)]
                if values.empty:
                    continue
                axis.errorbar(values.fragmentation_value, values["mean"],
                              yerr=1.96 * values["sem"].fillna(0), marker="o",
                              linestyle="none", capsize=3, label="%s coupling" % coupling)
            axis.axhline(0, color="black", lw=0.8)
            axis.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_xlabel("Fragmentation")
            axis.set_ylabel("KPI harm rate − entropic harm rate")
        axis.set_title(application.replace("_", " ").title())
    if mode == "phase":
        fig.colorbar(scatter, ax=axes, label="Incremental harm reduction", shrink=0.85)
    else:
        axes[1].legend(frameon=False)
    fig.suptitle(("Complexity phase map" if mode == "phase" else "Harm reduction versus coupling and fragmentation") + " · " + _evidence_label(results_root))
    return _save(results_root, name, fig, plotted_rows, "Discrete prospective complexity cells with panel means and normal descriptive intervals; no smoothing or confirmatory claim.")


def _scaling(results_root: Path) -> Dict[str, Any]:
    frame = _effect_data(results_root)
    frame["agent_count"] = frame.complexity.map({"small": 12, "medium": 28, "large": 52})
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    rows = []
    for application, subset in frame.groupby("application"):
        grouped = subset.groupby("agent_count").incremental_harm_reduction.agg(["mean", "sem"]).reset_index()
        grouped["application"] = application
        rows.extend(grouped.to_dict("records"))
        axis.errorbar(grouped.agent_count, grouped["mean"], yerr=1.96 * grouped["sem"].fillna(0), marker="o", linestyle="none", capsize=3, label=application.replace("_", " ").title(), color=PALETTE[application])
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xlabel("Persistent autonomous agents")
    axis.set_ylabel("Incremental harm reduction")
    axis.set_title("Scaling diagnostic · " + _evidence_label(results_root))
    axis.legend(frameon=False)
    axis.set_xticks([12, 28, 52], ["Small\n12", "Medium\n28", "Large\n52"])
    return _save(results_root, "entropy_value_scaling", fig, rows, "Observed discrete formal-development scale levels with panel standard errors; points are intentionally not interpolated.")


def _communication(results_root: Path) -> Dict[str, Any]:
    formal = results_root / "statistics" / "communication_primary_effects.csv"
    path = formal if formal.exists() else results_root / "pilots" / "analysis" / "communication_reductions.csv"
    frame = pd.read_csv(path)
    if "status" in frame and len(frame) == 1:
        frame = pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    if not frame.empty and formal.exists():
        for _, row in frame.iterrows():
            color = PALETTE[str(row.application)]
            axes[0].scatter(row.message_reduction, row.harm_degradation, s=90, color=color,
                            label=str(row.application).replace("_", " ").title(), edgecolor="black")
            axes[1].scatter(row.byte_reduction, row.maximum_event_estimation_mae, s=90,
                            color=color, edgecolor="black")
        axes[0].axvline(0.20, color=PALETTE["warning"], linestyle="--", label="20% reduction target")
        axes[0].axhline(0.02, color=PALETTE["dark"], linestyle=":", label="Harm NI margin")
        axes[0].set_xlabel("Total-message reduction")
        axes[0].set_ylabel("Harm-rate degradation")
        axes[0].legend(frameon=False, fontsize=8.8)
        axes[1].axvline(0.20, color=PALETTE["warning"], linestyle="--")
        axes[1].axhline(0.08, color=PALETTE["dark"], linestyle=":")
        axes[1].set_xlabel("Total-byte reduction")
        axes[1].set_ylabel("Maximum distributed-estimation MAE")
        fig.text(0.5, 0.01, "Same always-act operational controller: zero harm difference is a monitoring-cost result, not selective-safety evidence.", ha="center", fontsize=9)
    elif not frame.empty:
        frame = frame[frame.measure.isin(("total_messages", "total_bytes"))].copy()
        for axis, measure in zip(axes, ("total_messages", "total_bytes")):
            values = frame[frame.measure == measure]
            axis.bar(np.arange(len(values)), values.relative_reduction, color=[PALETTE.get(value, PALETTE["neutral"]) for value in values.application])
            axis.set_xticks(np.arange(len(values)), values.application.str.replace("_", " "), rotation=20)
            axis.axhline(0.20, color=PALETTE["warning"], linestyle="--", label="20% target")
            axis.set_ylabel("Fraction reduced")
            axis.set_title(measure.replace("_", " ").title())
            axis.legend(frameon=False)
    fig.suptitle("Event-triggered versus always-on fully counted communication · " + _evidence_label(results_root))
    fig.subplots_adjust(bottom=0.20)
    return _save(results_root, "communication_safety_pareto", fig, frame.to_dict("records"), "Matched formal-development communication reductions, safety difference, and estimation error; every message type is counted.")


def _trajectory(results_root: Path, name: str, recovery: bool = False) -> Dict[str, Any]:
    frame = _reference_candidates(results_root).copy()
    frame["horizon"] = frame.complexity.map({"small": 30, "medium": 60, "large": 100})
    frame["onset_fraction"] = frame.application.map({"humanitarian": 0.18, "utility_restoration": 0.16})
    frame["relative_time"] = frame.step / frame.horizon - frame.onset_fraction
    if recovery:
        frame = frame[frame.network_disruption.eq("high")]
    frame["relative_time_bin"] = (frame.relative_time * 20).round() / 20
    values = frame.groupby(["application", "relative_time_bin"])[["shannon_local", "js_disagreement", "consensus", "consensus_residual"]].mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    for application, subset in values.groupby("application"):
        color = PALETTE[application]
        if recovery:
            axes[0].plot(subset.relative_time_bin, subset.consensus, color=color, marker="o", label=application.replace("_", " ").title())
            axes[1].plot(subset.relative_time_bin, subset.consensus_residual, color=color, marker="o")
        else:
            axes[0].plot(subset.relative_time_bin, subset.shannon_local, color=color, marker="o", label=application.replace("_", " ").title())
            axes[1].plot(subset.relative_time_bin, subset.js_disagreement, color=color, marker="o")
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("Consensus" if recovery else "Local Shannon entropy")
    axes[1].set_ylabel("Consensus residual" if recovery else "JS disagreement")
    axes[1].set_xlabel("Episode fraction relative to disruption onset")
    for axis in axes:
        axis.axvline(0, color=PALETTE["warning"], linestyle="--", linewidth=1.1)
        if recovery:
            axis.axvline(0.32, color=PALETTE["entropic"], linestyle=":", linewidth=1.1)
        axis.grid(alpha=0.2)
    axes[0].annotate("disruption", xy=(0, axes[0].get_ylim()[1]),
                     xytext=(4, -5), textcoords="offset points", va="top",
                     color=PALETTE["warning"], fontsize=9)
    if recovery:
        axes[0].annotate("configured reconnection", xy=(0.32, axes[0].get_ylim()[1]),
                         xytext=(4, -5), textcoords="offset points", va="top",
                         color=PALETTE["entropic"], fontsize=9)
    fig.suptitle(("Consensus trajectory during high network disruption" if recovery else "Entropy and disagreement around disruptions") + " · " + _evidence_label(results_root))
    return _save(results_root, name, fig, values.to_dict("records"), "Formal reference probes aligned to application-specific disruption onset; the dotted recovery marker is the configured partition-release time.")


def _topology_robustness(results_root: Path) -> Dict[str, Any]:
    rows = []
    families = ("ring", "chain", "grid", "random_geometric", "small_world", "scale_free", "modular")
    for index, family in enumerate(families):
        graph = generate_graph(family, 28, 787800 + index)
        diagnostic = topology_diagnostics(graph, family)
        rows.append({key: value for key, value in diagnostic.__dict__.items() if key != "graph6_sha256"})
    frame = pd.DataFrame(rows)
    frame["display_family"] = frame.family.str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axes[0].barh(frame.display_family, frame.mean_degree, color=PALETTE["humanitarian"])
    axes[0].set_xlabel("Mean degree")
    axes[1].barh(frame.display_family, frame.clustering_coefficient, color=PALETTE["utility_restoration"])
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_xlabel("Clustering coefficient")
    fig.suptitle("Graph-structurally distinct V7 topology families")
    return _save(results_root, "graph_topology_robustness", fig, frame.to_dict("records"), "Structural diagnostics for stored topology instances.")


def _family_comparison(results_root: Path) -> Dict[str, Any]:
    formal = results_root / "development_formal_reference" / "risk_analysis" / "ranking_metrics.csv"
    if formal.exists():
        frame = pd.read_csv(formal).rename(columns={"feature_block": "method"})
        methods = ["strongest_nonentropic", "shannon_js", "generalized_entropic"]
    else:
        frame = pd.read_csv(results_root / "pilots_iteration3" / "analysis" / "risk_ranking_metrics.csv")
        methods = ["kpi_confidence", "predictive_uncertainty", "shannon_js", "generalized_tsallis_gini", "graph_disagreement", "combined_generalized_entropic"]
    subset = frame[frame.method.isin(methods)].groupby(["application", "method"]).roc_auc.mean().reset_index()
    fig, axis = plt.subplots(figsize=(10.0, 4.5))
    x = np.arange(len(methods)); width = 0.36
    for offset, application in zip((-width / 2, width / 2), ("humanitarian", "utility_restoration")):
        values = subset[subset.application == application].set_index("method").reindex(methods).roc_auc
        axis.bar(x + offset, values, width, label=application.replace("_", " ").title(), color=PALETTE[application])
    axis.set_xticks(x, [value.replace("_", "\n") for value in methods])
    axis.set_ylabel("Harm-ranking ROC AUC")
    axis.set_ylim(0, 1)
    axis.set_title("Same-capacity feature-block comparison · " + _evidence_label(results_root))
    axis.legend(frameon=False)
    return _save(results_root, "entropy_family_comparison", fig, subset.to_dict("records"), "Diagnostic ranking by prespecified information family.")


def _cascade_and_equity(results_root: Path, name: str, equity: bool = False) -> Dict[str, Any]:
    episodes = _reference_episodes(results_root)
    if equity:
        subset = episodes[episodes.application == "humanitarian"][["run_id", "complexity", "allocation_inequality_gini", "resource_waste", "delivery_completion"]]
        fig, axis = plt.subplots(figsize=(7.2, 4.3))
        colors = subset.complexity.map({"small": "#440154", "medium": "#21918C", "large": "#FDE725"})
        axis.scatter(subset.allocation_inequality_gini, subset.delivery_completion, c=colors, s=70)
        axis.set_xlabel("Allocation inequality (economic Gini)")
        axis.set_ylabel("Delivered commodity units")
        axis.set_title("Humanitarian allocation and service · " + _evidence_label(results_root))
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, label=label)
                   for label, color in (("Small", "#440154"), ("Medium", "#21918C"), ("Large", "#FDE725"))]
        axis.legend(handles=handles, title="Complexity", frameon=False)
    else:
        subset = episodes[["application", "run_id", "maximum_cascade_depth", "service_reaching_actions", "physical_actions"]]
        subset = subset.copy()
        subset["service_reaching_fraction"] = subset.service_reaching_actions / subset.physical_actions.clip(lower=1)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
        depth = subset.groupby("application").maximum_cascade_depth.max()
        axes[0].bar(depth.index.str.replace("_", " "), depth.values,
                    color=[PALETTE[value] for value in depth.index])
        axes[0].set_ylabel("Maximum recorded chain depth")
        axes[0].set_ylim(0, max(depth.max() + 0.6, 1))
        for index, value in enumerate(depth.values):
            axes[0].text(index, value + 0.08, str(int(value)), ha="center")
        for application, values in subset.groupby("application"):
            axes[1].scatter(values.physical_actions, values.service_reaching_fraction,
                            s=45, alpha=0.7, label=application.replace("_", " ").title(),
                            color=PALETTE[application])
        axes[1].set_xlabel("Accepted physical actions")
        axes[1].set_ylabel("Fraction reaching service")
        axes[1].legend(frameon=False)
        fig.suptitle("Multi-step causal chains and downstream service · " + _evidence_label(results_root))
    return _save(results_root, name, fig, subset.to_dict("records"), "Stored episode-level causal-chain or allocation outcomes.")


def _forest(results_root: Path, name: str) -> Dict[str, Any]:
    if name == "primary_effect_forest" and (results_root / "statistics" / "high_complexity_dynamic_effects.csv").exists():
        high = pd.read_csv(results_root / "statistics" / "high_complexity_dynamic_effects.csv")
        interaction = json.loads((results_root / "statistics" / "dynamic_primary_analysis.json").read_text(encoding="utf-8"))["interaction"]
        rows = [{
            "application": "pooled", "complexity": "coupling × fragmentation",
            "panels": int(interaction["panels"]),
            "effect": float(interaction["coupling_fragmentation_interaction"]),
            "ci_low": float(interaction["ci95_low"]), "ci_high": float(interaction["ci95_high"]),
        }]
        rows.extend({
            "application": str(row.application), "complexity": "high coupling + high fragmentation",
            "panels": int(row.panels), "effect": float(row.harm_reduction),
            "ci_low": float(row.harm_ci95_low), "ci_high": float(row.harm_ci95_high),
        } for _, row in high.iterrows())
        title = "Prospectively tested primary effects · formal development"
    else:
        frame = _effect_data(results_root)
        rows = []
        rng = np.random.RandomState(787799)
        for (application, complexity), subset in frame.groupby(["application", "complexity"], sort=True):
            observed = subset.incremental_harm_reduction.to_numpy(dtype=float)
            draws = np.mean(rng.choice(observed, size=(10000, len(observed)), replace=True), axis=1)
            rows.append({"application": application, "complexity": complexity, "panels": len(subset), "effect": float(observed.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975))})
        title = "Effect heterogeneity by scale · " + _evidence_label(results_root)
    values = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8.3, 4.6))
    labels = values.application.str.replace("_", " ") + " · " + values.complexity + " (n=" + values.panels.astype(str) + ")"
    y = np.arange(len(values))
    axis.errorbar(values.effect, y, xerr=[values.effect - values.ci_low, values.ci_high - values.effect], fmt="o", color=PALETTE["dark"], capsize=3)
    axis.axvline(0, color="black", lw=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("KPI harm rate − generalized-entropic harm rate")
    axis.set_title(title)
    return _save(results_root, name, fig, rows, "Panel effects with 10,000-replicate cluster bootstrap intervals; formal development, not confirmation.")


def generate_v7_figures(results_root: Path) -> Dict[str, Any]:
    _style()
    catalog = [
        _complexity_comparison(results_root),
        _network_figure(results_root, "humanitarian"),
        _network_figure(results_root, "utility_restoration"),
        _architecture(results_root),
        _phase_and_interaction(results_root, "complexity_phase_diagram", "phase"),
        _phase_and_interaction(results_root, "harm_reduction_coupling_fragmentation", "lines"),
        _scaling(results_root),
        _coverage_figure(results_root, "harm_rate", "risk_coverage"),
        _coverage_figure(results_root, "mean_utility", "utility_coverage"),
        _communication(results_root),
        _trajectory(results_root, "entropy_consensus_disruption_trajectories"),
        _trajectory(results_root, "consensus_recovery_after_partitions", recovery=True),
        _topology_robustness(results_root),
        _forest(results_root, "regime_level_effect_forest"),
        _family_comparison(results_root),
        _cascade_and_equity(results_root, "causal_chain_cascade_depth"),
        _cascade_and_equity(results_root, "humanitarian_resource_equity", equity=True),
        _forest(results_root, "primary_effect_forest"),
    ]
    # Learned-agent figures are generated only if those prospectively gated
    # stages actually ran; no placeholder PDFs imply nonexistent evidence.
    if (results_root / "training" / "seed_manifest.csv").exists():
        training = pd.read_csv(results_root / "training" / "seed_manifest.csv")
        curves = []
        for path in sorted((results_root / "training" / "curves").glob("*.csv")):
            curves.append(pd.read_csv(path))
        if curves:
            values = pd.concat(curves, ignore_index=True)
            fig, axis = plt.subplots(figsize=(8.5, 4.6))
            for (method, seed), subset in values.groupby(["method", "rl_seed"]):
                axis.plot(subset.training_episode, subset.mean_reward, alpha=0.45, label="%s / %s" % (method, seed))
            axis.set_xlabel("Training episodes"); axis.set_ylabel("Mean trajectory reward")
            axis.set_title("Sequential decentralized PPO: every retained seed")
            catalog.append(_save(results_root, "ppo_learning_curves_seed_variability", fig, values.to_dict("records"), "All retained PPO seed curves."))
    qwen_path = results_root / "qwen" / "qwen_formal_qualification_decisions.csv"
    if qwen_path.exists():
        values = pd.read_csv(qwen_path)
        summary = values.groupby("application").agg(decisions=("agent_id", "size"), harmful=("harmful", "mean"), beneficial=("beneficial", "mean"), utility=("causal_utility", "mean")).reset_index()
        fig, axis = plt.subplots(figsize=(7.3, 4.3))
        axis.bar(np.arange(len(summary)) - 0.18, summary.harmful, 0.36, label="Harmful", color=PALETTE["warning"])
        axis.bar(np.arange(len(summary)) + 0.18, summary.beneficial, 0.36, label="Beneficial", color=PALETTE["entropic"])
        axis.set_xticks(np.arange(len(summary)), summary.application.str.replace("_", " "))
        axis.set_ylabel("Fraction of decisions"); axis.set_title("Real-Qwen qualification (simulated domains)")
        axis.legend(frameon=False)
        catalog.append(_save(results_root, "qwen_action_delegation_funnel", fig, summary.to_dict("records"), "Formal Qwen action outcomes; not human evidence."))
    write_csv(results_root / "tables" / "figure_catalog.csv", catalog)
    report = {
        "figures_generated": len(catalog),
        "evidence_label": _evidence_label(results_root),
        "conditional_figures_omitted_when_stage_not_run": True,
    }
    atomic_json(results_root / "figures" / "generation_summary.json", report)
    return report
