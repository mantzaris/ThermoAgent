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
    return pd.read_csv(results_root / "pilots_iteration3" / "candidate_decisions.csv")


def _pilot_effects(results_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        results_root / "pilots_iteration3" / "analysis" / "paired_incremental_effects.csv"
    )


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
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    nx.draw_networkx_edges(graph, position, ax=axis, alpha=0.45, edge_color="#64748B")
    nx.draw_networkx_nodes(graph, position, ax=axis, node_color=communities, cmap="viridis", node_size=180, edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(graph, position, ax=axis, font_size=7)
    axis.set_title(("Humanitarian logistics" if application == "humanitarian" else "Utility physical–communication–crew") + " multilayer proxy network")
    axis.text(0.01, 0.01, "Stored graph instance; colors denote structural communities", transform=axis.transAxes, fontsize=9)
    axis.axis("off")
    name = "humanitarian_multilayer_network" if application == "humanitarian" else "utility_multilayer_network"
    return _save(results_root, name, fig, rows, "Stored medium-size V7 graph instance and structural communities.")


def _architecture(results_root: Path) -> Dict[str, Any]:
    nodes = [
        ("Private observations\nand beliefs", 0.08, 0.72),
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
    for first, second in edges:
        axis.annotate("", xy=nodes[second][1:], xytext=nodes[first][1:], arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], lw=1.4))
    axis.text(0.54, 0.93, "All sketch and operational traffic is explicitly delivered and costed", ha="center", fontsize=10)
    axis.set_xlim(0, 0.9); axis.set_ylim(0.12, 1.0); axis.axis("off")
    axis.set_title("Independent agents with a separate generalized-entropic Level-2 controller")
    return _save(results_root, "v7_independent_agent_entropy_architecture", fig, rows, "V7 decentralized execution and Level-2 risk-control boundary.")


def _coverage_data(frame: pd.DataFrame) -> pd.DataFrame:
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
    rows = []
    for (application, run_id), subset in accepted.groupby(["application", "run_id"], sort=True):
        for label, column in methods.items():
            for coverage in np.linspace(0.1, 1.0, 10):
                count = max(1, int(round(coverage * len(subset))))
                selected = subset.sort_values(column, kind="mergesort").head(count)
                rows.append({
                    "application": application, "run_id": run_id,
                    "method": label, "coverage": coverage,
                    "harm_rate": float(selected.counterfactual_harmful.mean()),
                    "mean_utility": float(selected.counterfactual_causal_utility.mean()),
                })
    return pd.DataFrame(rows)


def _coverage_figure(results_root: Path, metric: str, name: str) -> Dict[str, Any]:
    values = _coverage_data(_pilot_candidates(results_root))
    summary = values.groupby(["application", "method", "coverage"])[metric].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    styles = {
        "KPI confidence": (PALETTE["baseline"], "o", "-"),
        "Predictive uncertainty": (PALETTE["neutral"], "s", "--"),
        "Shannon + JS": (PALETTE["humanitarian"], "^", "-."),
        "Generalized entropic": (PALETTE["entropic"], "D", ":"),
    }
    for axis, application in zip(axes, ("humanitarian", "utility_restoration")):
        for method, subset in summary[summary.application == application].groupby("method"):
            color, marker, line = styles[method]
            axis.plot(subset.coverage, subset["mean"], color=color, marker=marker, linestyle=line, label=method)
            axis.fill_between(subset.coverage, subset["mean"] - 1.96 * subset["sem"].fillna(0), subset["mean"] + 1.96 * subset["sem"].fillna(0), color=color, alpha=0.12)
        axis.set_title(application.replace("_", " ").title())
        axis.set_xlabel("Autonomous-action coverage")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Harmful-action rate" if metric == "harm_rate" else "Mean causal utility")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(("Risk–coverage" if metric == "harm_rate" else "Utility–coverage") + " · " + _evidence_label(results_root))
    return _save(results_root, name, fig, summary.to_dict("records"), "Coverage curve using retained pilot counterfactual probes.")


def _phase_and_interaction(results_root: Path, name: str, mode: str) -> Dict[str, Any]:
    frame = _pilot_effects(results_root)
    mapping = {"low": 0, "medium": 1, "high": 2}
    frame["coupling_value"] = frame.coupling.map(mapping)
    frame["fragmentation_value"] = frame.fragmentation.map(mapping)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=(mode == "phase"))
    for axis, application in zip(axes, ("humanitarian", "utility_restoration")):
        subset = frame[frame.application == application]
        if mode == "phase":
            scatter = axis.scatter(subset.fragmentation_value, subset.coupling_value, c=subset.incremental_harm_reduction, cmap="RdBu", vmin=-0.10, vmax=0.10, s=100, edgecolor="black")
            axis.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_xlabel("Private-information fragmentation")
            axis.set_ylabel("Coupling")
        else:
            grouped = subset.groupby(["coupling", "fragmentation"]).incremental_harm_reduction.mean().reset_index()
            for coupling, values in grouped.groupby("coupling"):
                axis.plot(values.fragmentation.map(mapping), values.incremental_harm_reduction, marker="o", label="%s coupling" % coupling)
            axis.axhline(0, color="black", lw=0.8)
            axis.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            axis.set_xlabel("Fragmentation")
            axis.set_ylabel("KPI harm rate − entropic harm rate")
        axis.set_title(application.replace("_", " ").title())
    if mode == "phase":
        fig.colorbar(scatter, ax=axes, label="Incremental harm reduction", shrink=0.85)
    else:
        axes[1].legend(frameon=False)
    fig.suptitle(("Complexity phase map" if mode == "phase" else "Harm reduction versus coupling and fragmentation") + " · retained feasibility pilot")
    return _save(results_root, name, fig, frame.to_dict("records"), "Discrete prospective complexity cells; no smoothing or confirmatory claim.")


def _scaling(results_root: Path) -> Dict[str, Any]:
    frame = _pilot_effects(results_root)
    frame["agent_count"] = frame.complexity.map({"small": 12, "medium": 28, "large": 52})
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for application, subset in frame.groupby("application"):
        grouped = subset.groupby("agent_count").incremental_harm_reduction.agg(["mean", "sem"]).reset_index()
        axis.errorbar(grouped.agent_count, grouped["mean"], yerr=1.96 * grouped["sem"].fillna(0), marker="o", capsize=3, label=application.replace("_", " ").title(), color=PALETTE[application])
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xlabel("Persistent autonomous agents")
    axis.set_ylabel("Incremental harm reduction")
    axis.set_title("Scaling diagnostic · retained feasibility pilot")
    axis.legend(frameon=False)
    return _save(results_root, "entropy_value_scaling", fig, frame.to_dict("records"), "Observed discrete pilot scale levels with panel standard errors.")


def _communication(results_root: Path) -> Dict[str, Any]:
    path = results_root / "pilots" / "analysis" / "communication_reductions.csv"
    frame = pd.read_csv(path)
    if "status" in frame and len(frame) == 1:
        frame = pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    if not frame.empty:
        frame = frame[frame.measure.isin(("total_messages", "total_bytes"))].copy()
        for axis, measure in zip(axes, ("total_messages", "total_bytes")):
            values = frame[frame.measure == measure]
            axis.bar(np.arange(len(values)), values.relative_reduction, color=[PALETTE.get(value, PALETTE["neutral"]) for value in values.application])
            axis.set_xticks(np.arange(len(values)), values.application.str.replace("_", " "), rotation=20)
            axis.axhline(0.20, color=PALETTE["warning"], linestyle="--", label="20% target")
            axis.set_ylabel("Fraction reduced")
            axis.set_title(measure.replace("_", " ").title())
            axis.legend(frameon=False)
    fig.suptitle("Event-triggered versus always-on fully counted communication · retained pilot")
    return _save(results_root, "communication_safety_pareto", fig, frame.to_dict("records"), "Matched pilot message and byte reductions; operational and sketch traffic included.")


def _trajectory(results_root: Path, name: str, recovery: bool = False) -> Dict[str, Any]:
    frame = _pilot_candidates(results_root)
    values = frame.groupby(["application", "step"])[["shannon_local", "js_disagreement", "consensus", "consensus_residual"]].mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    for application, subset in values.groupby("application"):
        color = PALETTE[application]
        if recovery:
            axes[0].plot(subset.step, subset.consensus, color=color, label=application.replace("_", " ").title())
            axes[1].plot(subset.step, subset.consensus_residual, color=color)
        else:
            axes[0].plot(subset.step, subset.shannon_local, color=color, label=application.replace("_", " ").title())
            axes[1].plot(subset.step, subset.js_disagreement, color=color)
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("Consensus" if recovery else "Local Shannon entropy")
    axes[1].set_ylabel("Consensus residual" if recovery else "JS disagreement")
    axes[1].set_xlabel("Simulation step")
    for axis in axes: axis.grid(alpha=0.2)
    fig.suptitle(("Consensus recovery around partitions" if recovery else "Entropy and disagreement around disruptions") + " · retained pilot")
    return _save(results_root, name, fig, values.to_dict("records"), "Event-aligned means from stored candidate decisions.")


def _topology_robustness(results_root: Path) -> Dict[str, Any]:
    rows = []
    families = ("ring", "chain", "grid", "random_geometric", "small_world", "scale_free", "modular")
    for index, family in enumerate(families):
        graph = generate_graph(family, 28, 787800 + index)
        diagnostic = topology_diagnostics(graph, family)
        rows.append({key: value for key, value in diagnostic.__dict__.items() if key != "graph6_sha256"})
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axes[0].barh(frame.family, frame.mean_degree, color=PALETTE["humanitarian"])
    axes[0].set_xlabel("Mean degree")
    axes[1].barh(frame.family, frame.clustering_coefficient, color=PALETTE["utility_restoration"])
    axes[1].set_xlabel("Clustering coefficient")
    fig.suptitle("Graph-structurally distinct V7 topology families")
    return _save(results_root, "graph_topology_robustness", fig, frame.to_dict("records"), "Structural diagnostics for stored topology instances.")


def _family_comparison(results_root: Path) -> Dict[str, Any]:
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
    axis.set_title("Entropy-family comparison · retained feasibility pilot")
    axis.legend(frameon=False)
    return _save(results_root, "entropy_family_comparison", fig, subset.to_dict("records"), "Diagnostic ranking by prespecified information family.")


def _cascade_and_equity(results_root: Path, name: str, equity: bool = False) -> Dict[str, Any]:
    episodes = pd.read_csv(results_root / "pilots_iteration3" / "episode_summary.csv")
    if equity:
        subset = episodes[episodes.application == "humanitarian"][["run_id", "complexity", "allocation_inequality_gini", "resource_waste", "delivery_completion"]]
        fig, axis = plt.subplots(figsize=(7.2, 4.3))
        axis.scatter(subset.allocation_inequality_gini, subset.delivery_completion, c=subset.complexity.map({"small": 0, "medium": 1, "large": 2}), cmap="viridis", s=70)
        axis.set_xlabel("Allocation inequality (economic Gini)")
        axis.set_ylabel("Delivered commodity units")
        axis.set_title("Humanitarian allocation and service · retained pilot")
    else:
        subset = episodes[["application", "run_id", "maximum_cascade_depth", "service_reaching_actions", "physical_actions"]]
        fig, axis = plt.subplots(figsize=(7.2, 4.3))
        for application, values in subset.groupby("application"):
            axis.scatter(values.maximum_cascade_depth, values.service_reaching_actions, s=65, alpha=0.8, label=application.replace("_", " ").title(), color=PALETTE[application])
        axis.set_xlabel("Maximum recorded causal-chain depth")
        axis.set_ylabel("Service-reaching physical actions")
        axis.set_title("Delayed causal chains reach downstream service")
        axis.legend(frameon=False)
    return _save(results_root, name, fig, subset.to_dict("records"), "Stored episode-level causal-chain or allocation outcomes.")


def _forest(results_root: Path, name: str) -> Dict[str, Any]:
    frame = _pilot_effects(results_root)
    rows = []
    for (application, complexity), subset in frame.groupby(["application", "complexity"], sort=True):
        mean = float(subset.incremental_harm_reduction.mean())
        sem = float(subset.incremental_harm_reduction.sem()) if len(subset) > 1 else 0.0
        rows.append({"application": application, "complexity": complexity, "panels": len(subset), "effect": mean, "ci_low": mean - 1.96 * sem, "ci_high": mean + 1.96 * sem})
    values = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8.3, 4.6))
    labels = values.application.str.replace("_", " ") + " · " + values.complexity + " (n=" + values.panels.astype(str) + ")"
    y = np.arange(len(values))
    axis.errorbar(values.effect, y, xerr=[values.effect - values.ci_low, values.ci_high - values.effect], fmt="o", color=PALETTE["dark"], capsize=3)
    axis.axvline(0, color="black", lw=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("KPI harm rate − generalized-entropic harm rate")
    axis.set_title("Development diagnostic effect forest · retained pilot")
    return _save(results_root, name, fig, rows, "Panel means and normal diagnostic intervals; not confirmatory inference.")


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
