"""Data-derived publication figures for generalized-entropic V6 evidence."""

from __future__ import annotations

import gzip
import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .dashboard.v6 import V6DashboardFrame, V6DashboardReplay, frame_svg_v6
from .v6_entropy import generalized_disagreement, shannon_entropy, tsallis_entropy


COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "black": "#20242A", "gray": "#6C7480", "light": "#E9EDF2",
}
APP_COLORS = {
    "commercial": COLORS["blue"], "humanitarian": COLORS["green"],
    "utility_restoration": COLORS["orange"],
}
APP_LABELS = {
    "commercial": "Commercial boundary", "humanitarian": "Humanitarian",
    "utility_restoration": "Utility restoration",
}
FIGURE_DATA_SOURCES = {
    "generalized_entropic_architecture": "figures/data/architecture.csv",
    "independent_agent_operator_flow": "figures/data/agent_operator_flow.csv",
    "entropy_family_curves": "figures/data/entropy_family_curves.csv",
    "entropy_spectrum_examples": "figures/data/entropy_spectrum_examples.csv",
    "uncertainty_disagreement_phase_plane": "figures/data/uncertainty_disagreement_phase_plane.csv",
    "graph_weighted_consensus_network": "figures/data/graph_weighted_consensus_network.csv",
    "risk_coverage": "figures/data/risk_coverage.csv",
    "harm_coverage": "figures/data/harm_coverage.csv",
    "utility_coverage": "figures/data/utility_coverage.csv",
    "operator_workload_service_pareto": "figures/data/operator_workload_service_pareto.csv",
    "communication_safety_pareto": "figures/data/communication_safety_pareto.csv",
    "entropy_family_effect_forest": "figures/data/entropy_family_effect_forest.csv",
    "primary_dynamic_effect_forest": "figures/data/primary_dynamic_effect_forest.csv",
    "fragmented_public_interaction": "figures/data/fragmented_public_interaction.csv",
    "v5_same_score_abstention": "figures/data/v5_same_score_abstention.csv",
    "coverage_matched_escalation": "figures/data/coverage_matched_escalation.csv",
    "sequential_rl_learning_curves": "figures/data/sequential_rl_learning_curves.csv",
    "rl_seed_evaluation": "figures/data/rl_seed_evaluation.csv",
    "qwen_agent_evaluation": "figures/data/qwen_agent_evaluation.csv",
    "calibration_conformal_risk": "figures/data/calibration_conformal_risk.csv",
    "regime_heterogeneity": "figures/data/regime_heterogeneity.csv",
    "consensus_recovery_timing": "figures/data/consensus_recovery_timing.csv",
    "utility_cyber_physical_network": "figures/data/utility_cyber_physical_network.csv",
    "entropy_family_ablation": "figures/data/entropy_family_ablation.csv",
    "operator_dashboard": "figures/data/operator_dashboard.csv",
    "matched_operator_dashboard": "figures/data/matched_operator_dashboard.csv",
    "causal_chain_funnel": "figures/data/causal_chain_funnel.csv",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.0, "axes.titlesize": 12.0,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2, "figure.titlesize": 13.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def _save(fig: Any, root: Path, name: str, stamp: bool = True) -> str:
    pdf = root / "figures" / "pdf" / (name + ".pdf")
    png = root / "figures" / "png" / (name + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    if stamp:
        fig.text(
            0.995, 0.008, "DEVELOPMENT ONLY · SIMULATED OPERATOR · NO REAL-HUMAN EVIDENCE",
            ha="right", va="bottom", fontsize=9.0, color=COLORS["gray"],
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    fig.savefig(pdf, format="pdf", metadata={
        "Title": name,
        "Subject": "Generalized Entropic Consensus V6",
    })
    fig.savefig(png, format="png", dpi=240)
    plt.close(fig)
    return str(pdf.relative_to(root))


def _data(root: Path, name: str, frame: pd.DataFrame) -> None:
    path = root / "figures" / "data" / (name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except TypeError:
        frame.to_csv(path, index=False, line_terminator="\n")


def _read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists() and path.suffix == ".csv":
        compressed = path.with_suffix(path.suffix + ".gz")
        if compressed.exists():
            path = compressed
    return pd.read_csv(path)


def architecture(root: Path) -> str:
    rows = [
        (0.4, 7.25, "Independent agents\nprivate belief · memory\nutility · authority", "blue"),
        (6.0, 7.25, "Ad-hoc messages\ncompressed entropy spectra\nlogged communication", "green"),
        (11.5, 7.25, "Distributed estimator\nuncertainty · disagreement\nconsensus residual", "orange"),
        (11.5, 4.35, "Delegation controller\nexecute · communicate\nabstain · escalate", "purple"),
        (6.0, 4.35, "Bounded simulated operator\nqueue · minutes\nfinite budget", "red"),
        (0.4, 4.35, "Typed operational action\nrole mask\naccept · reject · counter", "blue"),
        (3.0, 1.4, "Dynamic state transition\nresources · commitments\nservice", "green"),
        (9.0, 1.4, "Evaluator-only branch\nmatched stochastic tape\ncounterfactual analysis", "orange"),
    ]
    _data(root, "architecture", pd.DataFrame(rows, columns=["x", "y", "component", "color"]))
    box_width = 4.1
    box_height = 1.55
    fig, ax = plt.subplots(figsize=(10.5, 6.3)); ax.set(xlim=(0, 16), ylim=(0, 10)); ax.axis("off")
    for x, y, label, color in rows:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), box_width, box_height, boxstyle="round,pad=0.06",
            facecolor=mpl.colors.to_rgba(COLORS[color], 0.12),
            edgecolor=COLORS[color], linewidth=1.6,
        ))
        ax.text(x + box_width / 2, y + box_height / 2, label, ha="center", va="center", fontsize=9.4)
    arrows = [
        ((4.5, 8.03), (6.0, 8.03)),
        ((10.1, 8.03), (11.5, 8.03)),
        ((13.55, 7.25), (13.55, 5.90)),
        ((11.5, 5.13), (10.1, 5.13)),
        ((6.0, 5.13), (4.5, 5.13)),
        ((2.45, 4.35), (4.05, 2.95)),
        ((7.1, 2.18), (9.0, 2.18)),
    ]
    for first, second in arrows:
        ax.annotate("", xy=second, xytext=first, arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(8, 0.55, "Environment validates actions; it never substitutes an oracle decision.", ha="center")
    ax.set_title("Generalized-entropic selective autonomy")
    return _save(fig, root, "generalized_entropic_architecture", stamp=False)


def agent_operator_flow(root: Path) -> str:
    stages = ["Private\nobservation", "Local\nproposal", "Peer\noffers", "Distributed\nconsensus", "Delegate or\nescalate", "Bounded\naction", "Service\noutcome"]
    values = pd.DataFrame({"order": range(len(stages)), "stage": stages})
    _data(root, "agent_operator_flow", values)
    fig, ax = plt.subplots(figsize=(7.5, 3.3)); ax.axis("off")
    for index, label in enumerate(stages):
        x = 0.6 + index * 1.65
        ax.add_patch(patches.FancyBboxPatch((x, 1.1), 1.35, 0.9, boxstyle="round,pad=.04", facecolor=mpl.colors.to_rgba(COLORS["blue" if index < 3 else "orange" if index < 5 else "green"], .13), edgecolor=COLORS["black"]))
        ax.text(x + .675, 1.55, label, ha="center", va="center", fontsize=9)
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 1.62, 1.55), xytext=(x + 1.36, 1.55), arrowprops={"arrowstyle": "->"})
    ax.text(5.55, .48, "Evaluator-only counterfactual", ha="center", color=COLORS["red"])
    ax.annotate("", xy=(10.5, 1.05), xytext=(5.55, .58), arrowprops={"arrowstyle": "->", "linestyle": "--", "color": COLORS["red"]})
    ax.set(xlim=(0, 12), ylim=(0, 2.8)); ax.set_title("Independent-agent, ad-hoc network, estimator, and oversight flow")
    return _save(fig, root, "independent_agent_operator_flow")


def entropy_curves(root: Path) -> str:
    dominant = np.linspace(1 / 6, .995, 180)
    beliefs = [np.asarray([value] + [(1 - value) / 5] * 5) for value in dominant]
    rows = []
    for q in (.5, 1., 1.5, 2., 3.):
        for value, belief in zip(dominant, beliefs):
            rows.append({"dominant_probability": value, "q": q, "entropy": tsallis_entropy(belief, q)})
    frame = pd.DataFrame(rows); _data(root, "entropy_family_curves", frame)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = [COLORS["purple"], COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    for (q, subset), color in zip(frame.groupby("q"), colors):
        ax.plot(subset.dominant_probability, subset.entropy, label="q = %g" % q, color=color, lw=2)
    ax.set(xlabel="Probability of dominant incident mode", ylabel="Normalized generalized entropy", ylim=(-.02, 1.02), title="Shannon, Tsallis, and Gini-Simpson uncertainty")
    ax.legend(ncol=3)
    return _save(fig, root, "entropy_family_curves")


def entropy_spectra(root: Path) -> str:
    examples = {
        "Broad uncertainty": [.22, .19, .18, .16, .14, .11],
        "Rare-state tail": [.74, .12, .07, .04, .02, .01],
        "Dominant confidence": [.94, .02, .015, .01, .01, .005],
        "Two-mode ambiguity": [.47, .45, .03, .02, .02, .01],
    }
    rows = []
    for name, probabilities in examples.items():
        belief = np.asarray(probabilities) / np.sum(probabilities)
        for q in (.5, 1., 1.5, 2., 3.):
            rows.append({"belief_type": name, "q": q, "entropy": tsallis_entropy(belief, q)})
    frame = pd.DataFrame(rows); _data(root, "entropy_spectrum_examples", frame)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for index, (name, subset) in enumerate(frame.groupby("belief_type", sort=False)):
        ax.plot(subset.q, subset.entropy, marker=["o", "s", "^", "D"][index], lw=2, label=name)
    ax.set(xlabel="Tsallis order q", ylabel="Normalized entropy", title="Entropy-spectrum signatures", xticks=[.5, 1, 1.5, 2, 3])
    ax.legend()
    return _save(fig, root, "entropy_spectrum_examples")


def phase_plane(root: Path) -> str:
    frame = _read(root, "development/formal_reference/candidate_decisions.csv")
    frame = frame[(frame.information_condition == "private_fragmented") & frame.application.isin(["humanitarian", "utility_restoration"])].copy()
    frame["harmful"] = frame.evaluator_harmful_if_executed.astype(str).str.lower().isin(["true", "1"])
    sample = frame.sample(min(1400, len(frame)), random_state=66080)
    _data(root, "uncertainty_disagreement_phase_plane", sample[["application", "operational_energy", "shannon_local", "js_disagreement", "consensus_residual", "harmful"]])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9), sharex=True, sharey=True)
    for ax, application in zip(axes, ("humanitarian", "utility_restoration")):
        subset = sample[sample.application == application]
        ax.scatter(subset.shannon_local, subset.js_disagreement, c=subset.harmful.astype(int), cmap=mpl.colors.ListedColormap([COLORS["green"], COLORS["red"]]), s=15, alpha=.48)
        ax.set(title=APP_LABELS[application], xlabel="Local uncertainty (Shannon)")
    axes[0].set_ylabel("Epistemic disagreement (Jensen–Shannon)")
    fig.suptitle("Aleatoric uncertainty and epistemic disagreement are distinct")
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=COLORS["green"], label="Beneficial/neutral proposal"),
            Line2D([], [], marker="o", ls="", color=COLORS["red"], label="Harmful proposal"),
        ],
        loc="upper left",
        fontsize=8.7,
    )
    return _save(fig, root, "uncertainty_disagreement_phase_plane")


def _representative_episode(root: Path, application: str) -> Path:
    paths = sorted((root / "raw" / "development_dynamic").glob(
        "v6-development_dynamic-%s-compound-private_fragmented-e*-combined_generalized_entropic_crossfit-*/episode.json*" % application
    ))
    if not paths:
        paths = sorted((root / "raw" / "development_dynamic").glob(
            "v6-development_dynamic-%s-*-private_fragmented-*/episode.json*" % application
        ))
    if not paths:
        raise FileNotFoundError("no representative V6 episode for %s" % application)
    return paths[0]


def _events(path: Path) -> List[Dict[str, Any]]:
    ledger = next(path.parent.glob("events.jsonl*"))
    opener = gzip.open if ledger.suffix == ".gz" else open
    with opener(ledger, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def graph_consensus(root: Path) -> str:
    episode = _representative_episode(root, "humanitarian")
    events = _events(episode)
    sketches = [value for value in events if value["kind"] == "v6_sketch" and value["step"] == 2]
    beliefs = {
        value["actor"]: np.asarray(value["payload"]["belief_summary"], dtype=float)
        for value in sketches
    }
    edges = pd.DataFrame([{
        "sender": value["actor"], "recipient": value["payload"]["recipient"],
        "reliability": value["payload"]["reliability"], "step": value["step"],
        "sender_entropy": shannon_entropy(beliefs[value["actor"]]),
        "recipient_entropy": shannon_entropy(beliefs[value["payload"]["recipient"]]),
        "pairwise_disagreement": generalized_disagreement(
            [beliefs[value["actor"]], beliefs[value["payload"]["recipient"]]],
            [.5, .5],
            1.0,
        ),
    } for value in sketches])
    _data(root, "graph_weighted_consensus_network", edges)
    nodes = sorted(set(edges.sender).union(edges.recipient))
    positions = {node: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes))) for index, node in enumerate(nodes)}
    node_entropy = {node: shannon_entropy(beliefs[node]) for node in nodes}
    node_cmap = mpl.cm.get_cmap("cividis")
    edge_cmap = mpl.cm.get_cmap("magma")
    node_normalizer = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    edge_maximum = max(float(edges.pairwise_disagreement.max()), 1e-6)
    edge_normalizer = mpl.colors.Normalize(vmin=0.0, vmax=edge_maximum)

    fig, ax = plt.subplots(figsize=(8.2, 6.8)); ax.axis("off")
    for row in edges.itertuples(index=False):
        x1, y1 = positions[row.sender]; x2, y2 = positions[row.recipient]
        ax.plot(
            [x1, x2], [y1, y2],
            color=edge_cmap(edge_normalizer(row.pairwise_disagreement)),
            alpha=.35 + .55 * row.reliability,
            lw=.6 + 2.2 * row.reliability,
        )
    for node, (x, y) in positions.items():
        ax.scatter(
            [x], [y], s=150,
            color=[node_cmap(node_normalizer(node_entropy[node]))],
            edgecolor=COLORS["black"], zorder=3,
        )
        if "_regional_hub_" in node:
            role = "Hub"
        elif "_clinic_" in node:
            role = "Clinic"
        else:
            role = "NGO"
        ax.text(x, y - .11, "%s %s" % (role, node.split("_")[-2]), ha="center", va="top", fontsize=8.6)
    edge_map = mpl.cm.ScalarMappable(norm=edge_normalizer, cmap=edge_cmap)
    edge_bar = fig.colorbar(edge_map, ax=ax, fraction=.033, pad=.02)
    edge_bar.set_label("Pairwise JS disagreement", fontsize=9.2)
    entropy_handles = [
        Line2D(
            [], [], marker="o", ls="", markeredgecolor=COLORS["black"],
            markerfacecolor=node_cmap(node_normalizer(value)),
            label="Node H = %.1f" % value,
        )
        for value in (0.0, 0.5, 1.0)
    ]
    entropy_handles.append(Line2D([], [], color=COLORS["gray"], lw=2.4, label="Edge width = reliability"))
    ax.legend(
        handles=entropy_handles,
        loc="upper center",
        bbox_to_anchor=(.5, -.035),
        ncol=4,
        fontsize=8.2,
        title="Delivered-sketch encodings",
        title_fontsize=8.5,
    )
    ax.set_title("Graph-weighted consensus over delivered ad-hoc sketches")
    return _save(fig, root, "graph_weighted_consensus_network")


def coverage_curves(root: Path, metric: str, name: str, ylabel: str) -> str:
    frame = _read(root, "development/risk_analysis/risk_coverage_panel_results.csv")
    frame = frame[(frame.application.isin(["humanitarian", "utility_restoration"])) & (frame.information_condition == "private_fragmented")]
    grouped = frame.groupby(["application", "feature_block", "coverage_target"], as_index=False)[metric].agg(["mean", "sem"]).reset_index()
    _data(root, name, grouped)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True)
    methods = ["kpi_confidence", "predictive_uncertainty", "shannon_js", "generalized_tsallis_gini", "combined_generalized_entropic"]
    for ax, app in zip(axes, ("humanitarian", "utility_restoration")):
        for index, method in enumerate(methods):
            subset = grouped[(grouped.application == app) & (grouped.feature_block == method)]
            ax.errorbar(subset.coverage_target, subset["mean"], yerr=1.96 * subset["sem"].fillna(0), marker=["o", "s", "^", "D", "P"][index], lw=1.7, label=method.replace("_", " "))
        ax.set(title=APP_LABELS[app], xlabel="Autonomous-action coverage")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="best", fontsize=9.0)
    fig.suptitle(ylabel + " across matched action coverage")
    return _save(fig, root, name)


def operator_pareto(root: Path) -> str:
    frame = _read(root, "development/dynamic/episode_summary.csv")
    grouped = frame.groupby(["application", "controller"], as_index=False).agg(service_loss=("service_loss", "mean"), operator_minutes=("operator_minutes", "mean"))
    _data(root, "operator_workload_service_pareto", grouped)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for app, subset in grouped.groupby("application"):
        ax.scatter(subset.operator_minutes, subset.service_loss, s=70, label=APP_LABELS[app], color=APP_COLORS[app])
        for row in subset.itertuples(index=False):
            ax.annotate("generalized" if "combined" in row.controller else "baseline", (row.operator_minutes, row.service_loss), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Simulated-operator minutes per episode", ylabel="Primary service loss", title="Operator effort and dynamic service performance")
    ax.legend()
    return _save(fig, root, "operator_workload_service_pareto")


def communication_pareto(root: Path) -> str:
    costs = _read(root, "development/communication/sketch_costs.csv")
    errors = _read(root, "development/communication/distributed_estimation_error.csv")
    frame = costs.merge(errors, on="sketch_policy", validate="one_to_one")
    _data(root, "communication_safety_pareto", frame)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for index, row in enumerate(frame.itertuples(index=False)):
        ax.scatter(row.total_bytes_mean, row.distributed_estimation_mae, s=85, marker=["o", "s", "^", "D"][index], label=row.sketch_policy.replace("_", " "))
    ax.set(xlabel="Total communicated bytes per episode", ylabel="Distributed-estimation MAE", title="Communication cost versus consensus accuracy")
    ax.legend()
    return _save(fig, root, "communication_safety_pareto")


def effect_forest(root: Path) -> str:
    selections = _read(
        root,
        "development/entropy_family/entropy_family_panel_selections.csv",
    )
    measures = [
        "tsallis_q_0_5", "tsallis_q_1_5", "tsallis_q_2", "tsallis_q_3",
        "gini_simpson", "jensen_shannon", "jensen_tsallis_q_0_5",
        "jensen_tsallis_q_2", "graph_weighted_disagreement",
    ]
    rows: List[Dict[str, Any]] = []
    rng = np.random.RandomState(66801)
    for application in ("humanitarian", "utility_restoration"):
        subset = selections[
            (selections.application == application)
            & (selections.information_condition == "private_fragmented")
        ]
        reference = subset[subset.entropy_measure == "shannon_local"][[
            "cluster_id", "harmful_action_rate",
        ]]
        for measure in measures:
            candidate = subset[subset.entropy_measure == measure][[
                "cluster_id", "harmful_action_rate",
            ]]
            paired = reference.merge(
                candidate, on="cluster_id", suffixes=("_shannon", "_candidate"),
                validate="one_to_one",
            )
            differences = (
                paired.harmful_action_rate_shannon
                - paired.harmful_action_rate_candidate
            ).to_numpy(dtype=float)
            bootstrap = np.mean(
                differences[rng.randint(0, len(differences), size=(10000, len(differences)))],
                axis=1,
            )
            rows.append({
                "application": application,
                "entropy_measure": measure,
                "reference": "shannon_local",
                "independent_panels": len(differences),
                "paired_harm_rate_reduction": float(np.mean(differences)),
                "ci95_low": float(np.quantile(bootstrap, .025)),
                "ci95_high": float(np.quantile(bootstrap, .975)),
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 66801,
            })
    frame = pd.DataFrame(rows)
    _data(root, "entropy_family_effect_forest", frame)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6.2), sharex=True, sharey=True)
    labels = [value.replace("jensen_", "JT ").replace("tsallis_", "Tsallis ").replace("_", " ") for value in measures]
    for ax, application in zip(axes, ("humanitarian", "utility_restoration")):
        subset = frame[frame.application == application].set_index("entropy_measure").loc[measures]
        y = np.arange(len(subset))[::-1]
        ax.errorbar(
            subset.paired_harm_rate_reduction, y,
            xerr=[
                subset.paired_harm_rate_reduction - subset.ci95_low,
                subset.ci95_high - subset.paired_harm_rate_reduction,
            ],
            fmt="o", color=APP_COLORS[application], capsize=2.5,
        )
        ax.axvline(0, color=COLORS["black"], lw=1)
        ax.set(
            yticks=y,
            title=APP_LABELS[application],
        )
        if ax is axes[0]:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Prespecified entropy-family effects at 50% action coverage")
    fig.text(
        .5,
        .055,
        "Harmful-action-rate reduction versus Shannon (positive favors row)",
        ha="center",
        va="bottom",
        fontsize=11.0,
    )
    return _save(fig, root, "entropy_family_effect_forest")


def dynamic_effect_forest(root: Path) -> str:
    frame = _read(root, "development/dynamic/paired_dynamic_effects.csv")
    _data(root, "primary_dynamic_effect_forest", frame)
    labels = [APP_LABELS[row.application] + " · " + row.information_condition.replace("_", " ") for row in frame.itertuples(index=False)]
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for position, row in zip(y, frame.itertuples(index=False)):
        ax.errorbar(row.harm_rate_reduction, position, xerr=[[row.harm_rate_reduction - row.harm_ci95_low], [row.harm_ci95_high - row.harm_rate_reduction]], fmt="o", color=APP_COLORS[row.application], capsize=3)
    ax.axvline(0, color=COLORS["black"], lw=1)
    ax.axvline(.03, color=COLORS["red"], lw=1, ls="--", label="Frozen practical threshold")
    ax.set(yticks=y, yticklabels=labels, xlabel="Harm-rate reduction vs strongest non-entropic baseline", title="Cross-fitted dynamic selective-safety effects")
    ax.legend()
    return _save(fig, root, "primary_dynamic_effect_forest")


def fragmented_public(root: Path) -> str:
    frame = _read(root, "development/dynamic/fragmentation_interaction.csv")
    _data(root, "fragmented_public_interaction", frame)
    fig, ax = plt.subplots(figsize=(7.0, 4.3)); y = np.arange(len(frame))[::-1]
    for position, row in zip(y, frame.itertuples(index=False)):
        ax.errorbar(row.private_minus_public_harm_reduction, position, xerr=[[row.private_minus_public_harm_reduction-row.ci95_low], [row.ci95_high-row.private_minus_public_harm_reduction]], fmt="o", color=APP_COLORS[row.application], capsize=3)
    ax.axvline(0, color=COLORS["black"]); ax.axvline(.02, color=COLORS["red"], ls="--")
    ax.set(yticks=y, yticklabels=[APP_LABELS[value] for value in frame.application], xlabel="Private-minus-public incremental harm reduction", title="Prespecified fragmentation-mechanism interaction")
    return _save(fig, root, "fragmented_public_interaction")


def v5_abstention(root: Path, name: str, metric: str, title: str) -> str:
    frame = _read(root, "v5_reanalysis/abstention_policy_summary.csv")
    selected = frame[frame.policy.isin(["original_safe", "same_score_no_consensus", "coverage_matched_no_consensus", "operator_budget_matched_escalation", "mandatory_intervention"])].copy()
    grouped = selected.groupby(["application", "policy"], as_index=False)[metric].mean()
    _data(root, name, grouped)
    policy_order = [
        "original_safe",
        "same_score_no_consensus",
        "coverage_matched_no_consensus",
        "operator_budget_matched_escalation",
        "mandatory_intervention",
    ]
    policy_colors = {
        policy: color for policy, color in zip(
            policy_order,
            [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]],
        )
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.1), sharey=True)
    for ax, app in zip(axes, ("commercial", "humanitarian", "utility_restoration")):
        subset = grouped[grouped.application == app].set_index("policy").reindex(policy_order).reset_index()
        positions = np.arange(len(subset))
        ax.barh(positions, subset[metric], color=[policy_colors[value] for value in subset.policy])
        ax.set(yticks=positions, title=APP_LABELS[app])
        if ax is axes[0]:
            ax.set_yticklabels([value.replace("_", " ") for value in subset.policy])
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_xlabel(metric.replace("_", " ")); fig.suptitle(title)
    fig.text(
        .995, .012,
        "POST-DEVELOPMENT V5 REANALYSIS FOR V6 DESIGN · V5 GATES UNCHANGED",
        ha="right", fontsize=9.0, color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, .05, 1, .95))
    return _save(fig, root, name, stamp=False)


def training_curves(root: Path) -> str:
    files = sorted((root / "training" / "curves").glob("*.csv"))
    frame = pd.concat([pd.read_csv(value) for value in files], ignore_index=True)
    _data(root, "sequential_rl_learning_curves", frame)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.4))
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    for color, (method, subset) in zip(palette, frame.groupby("method", sort=True)):
        for _, seed_rows in subset.groupby("rl_seed", sort=True):
            axes[0].plot(
                seed_rows.training_episode, seed_rows.mean_trajectory_reward,
                color=color, alpha=.22, lw=.8,
            )
            axes[1].plot(
                seed_rows.training_episode, seed_rows.policy_entropy,
                color=color, alpha=.22, lw=.8,
            )
        for ax, metric in zip(axes, ("mean_trajectory_reward", "policy_entropy")):
            grouped = subset.groupby("training_episode")[metric].agg(["mean", "sem"]).reset_index()
            error = 1.96 * grouped["sem"].fillna(0).to_numpy(dtype=float)
            x = grouped.training_episode.to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            ax.plot(
                x, mean, color=color, lw=2.0,
                label=method.replace("ppo_", "").replace("_", " "),
            )
            ax.fill_between(x, mean - error, mean + error, color=color, alpha=.10, linewidth=0)
    axes[0].set(xlabel="Training episode", ylabel="Mean trajectory reward", title="Seed trajectories and 95% mean intervals")
    axes[1].set(xlabel="Training episode", ylabel="Policy entropy", title="Exploration stability")
    axes[0].legend(
        fontsize=8.7,
        loc="best",
        title="Thin: seed · bold/band: mean ±95%",
        title_fontsize=8.5,
    )
    return _save(fig, root, "sequential_rl_learning_curves")


def seed_evaluation(root: Path) -> str:
    frame = _read(root, "training/seed_manifest.csv")
    _data(root, "rl_seed_evaluation", frame)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    methods = list(frame.method.unique())
    for index, method in enumerate(methods):
        subset = frame[frame.method == method]
        x = np.full(len(subset), index) + np.linspace(-.09, .09, len(subset))
        ax.scatter(x, subset.evaluation_mean_reward, s=48)
        ax.plot([index-.2, index+.2], [subset.evaluation_mean_reward.mean()]*2, color=COLORS["black"], lw=2)
    ax.set(xticks=range(len(methods)), xticklabels=[value.replace("ppo_", "").replace("_", "\n") for value in methods], ylabel="Evaluation trajectory reward", title="Independent sequential-PPO seeds")
    return _save(fig, root, "rl_seed_evaluation")


def qwen_evaluation(root: Path) -> str:
    frame = _read(root, "qwen/decision_epochs.csv")
    rows = []
    for app, subset in frame.groupby("application"):
        physical = subset[subset.accepted_physical_action.astype(str).str.lower().isin(["true", "1"])]
        rows.append({"application": app, "decisions": len(subset), "physical_actions": len(physical), "first_pass": subset.first_pass_valid.mean(), "harm_rate": physical.harmful.mean() if len(physical) else 0, "mean_effect": physical.causal_effect.mean() if len(physical) else 0, "abstention": (subset.delegation=="abstain").mean(), "escalation": (subset.delegation=="escalate_operator").mean()})
    summary = pd.DataFrame(rows); _data(root, "qwen_agent_evaluation", summary)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    x=np.arange(len(summary)); width=.22
    tick_labels = []
    for row in summary.itertuples(index=False):
        label = APP_LABELS[row.application]
        if row.application in {"commercial", "utility_restoration"}:
            label = label.replace(" ", "\n")
        tick_labels.append("%s\nphysical n=%d" % (label, row.physical_actions))
    for offset, key, label in [(-width,"first_pass","First-pass valid"),(0,"harm_rate","Harmful physical"),(width,"abstention","Abstention")]:
        axes[0].bar(x+offset,summary[key],width,label=label)
    axes[0].set(xticks=x,xticklabels=tick_labels,ylim=(0,1),ylabel="Fraction",title="Behavioral rates"); axes[0].legend(fontsize=9.0)
    axes[1].bar(x,summary.mean_effect,color=[APP_COLORS[v] for v in summary.application]); axes[1].axhline(0,color=COLORS["black"]); axes[1].set(xticks=x,xticklabels=tick_labels,ylabel="Mean causal effect",title="Accepted physical actions")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.5)
    return _save(fig, root, "qwen_agent_evaluation")


def calibration(root: Path) -> str:
    frame = _read(root, "development/risk_analysis/crossfit_risk_predictions.csv")
    frame = frame[(frame.feature_block.isin(["predictive_uncertainty","combined_generalized_entropic"])) & frame.application.isin(["humanitarian","utility_restoration"]) & (frame.information_condition=="private_fragmented")].copy()
    frame["bin"] = pd.cut(frame.predicted_harm_risk, np.linspace(0,1,11), include_lowest=True)
    grouped = frame.groupby(["feature_block","bin"], observed=True).agg(predicted=("predicted_harm_risk","mean"), observed=("harmful_label","mean"), n=("harmful_label","size")).reset_index()
    grouped["bin"] = grouped["bin"].astype(str); _data(root,"calibration_conformal_risk",grouped)
    fig,ax=plt.subplots(figsize=(6.4,5.0)); ax.plot([0,1],[0,1],color=COLORS["gray"],ls="--",label="Ideal")
    for method,subset in grouped.groupby("feature_block"):
        ax.plot(subset.predicted,subset.observed,marker="o",lw=2,label=method.replace("_"," "))
    ax.set(xlabel="Predicted harmful-action risk",ylabel="Observed harmful-action frequency",title="Cross-fitted reliability at selective-risk scores",xlim=(0,1),ylim=(0,1)); ax.legend()
    return _save(fig,root,"calibration_conformal_risk")


def regime_heterogeneity(root: Path) -> str:
    frame=_read(root,"development/dynamic/regime_dynamic_effects.csv"); frame=frame[(frame.information_condition=="private_fragmented") & frame.application.isin(["humanitarian","utility_restoration"])]
    _data(root,"regime_heterogeneity",frame)
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.5),sharey=True)
    for ax,app in zip(axes,("humanitarian","utility_restoration")):
        subset=frame[frame.application==app]; y=np.arange(len(subset))[::-1]
        ax.errorbar(subset.harm_rate_reduction,y,xerr=[subset.harm_rate_reduction-subset.harm_ci95_low,subset.harm_ci95_high-subset.harm_rate_reduction],fmt="o",capsize=2,color=APP_COLORS[app]); ax.axvline(0,color=COLORS["black"]); ax.set(yticks=y,xlabel="Harm reduction",title=APP_LABELS[app])
        if ax is axes[0]:
            ax.set_yticklabels([value.replace("_", " ") for value in subset.regime])
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Development effect heterogeneity by disruption regime")
    return _save(fig,root,"regime_heterogeneity")


def timing_recovery(root: Path) -> str:
    frame=_read(root,"development/sketch_reference/distributed_consensus.csv"); frame=frame[(frame.sketch_policy=="event_triggered") & frame.application.isin(["humanitarian","utility_restoration"])]
    grouped=frame.groupby(["application","step"],as_index=False).agg(consensus=("consensus","mean"),error=("evaluator_distributed_error","mean"),residual=("consensus_residual","mean")); _data(root,"consensus_recovery_timing",grouped)
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.0),sharex=True)
    for app,subset in grouped.groupby("application"):
        axes[0].plot(subset.step,subset.consensus,marker="o",label=APP_LABELS[app]); axes[1].plot(subset.step,subset.error,marker="s",label=APP_LABELS[app])
    for ax in axes: ax.axvline(2,color=COLORS["red"],ls="--",label="Disruption")
    axes[0].set(xlabel="Simulator step",ylabel="Consensus score",title="Consensus trajectory"); axes[1].set(xlabel="Simulator step",ylabel="Distributed-estimation MAE",title="Recovery under message aging"); axes[0].legend()
    return _save(fig,root,"consensus_recovery_timing")


def utility_network(root: Path) -> str:
    episode = _representative_episode(root, "utility_restoration")
    events = _events(episode)
    snapshot = next(
        value["payload"] for value in events if value["kind"] == "v6_panel_snapshot"
    )
    agents = sorted(snapshot["agent_ids"])

    def role(agent: str) -> str:
        return agent.replace("utility_restoration_", "").rsplit("_", 2)[0]

    def incident_number(agent: str) -> str:
        return agent.rsplit("_", 2)[1]

    layer_for_role = {
        "distribution_node": "service",
        "critical_load": "service",
        "communications": "telemetry",
        "cyber_defense": "telemetry",
        "field_crew": "restoration",
        "resource_allocation": "restoration",
    }
    layers = {
        name: [agent for agent in agents if layer_for_role[role(agent)] == name]
        for name in ("service", "telemetry", "restoration")
    }
    y_values = {"service": 2.45, "telemetry": 1.35, "restoration": 0.25}
    positions: Dict[str, Tuple[float, float]] = {}
    node_rows: List[Dict[str, Any]] = []
    for layer, values in layers.items():
        xs = np.linspace(0.6, 6.4, len(values)) if len(values) > 1 else np.asarray([3.5])
        for x, agent in zip(xs, values):
            positions[agent] = (float(x), y_values[layer])
            node_rows.append({
                "record_type": "agent_node", "source": agent, "target": "",
                "layer": layer, "role": role(agent), "incident": incident_number(agent),
                "step": 2, "reliability": np.nan, "action": "",
            })
    incidents = sorted({"utility_restoration_incident_%s" % incident_number(value) for value in agents})
    incident_positions = {
        incident: (float(x), 3.42)
        for incident, x in zip(incidents, np.linspace(1.0, 6.0, len(incidents)))
    }
    for incident in incidents:
        node_rows.append({
            "record_type": "service_incident", "source": incident, "target": "",
            "layer": "service_zone", "role": "service incident",
            "incident": incident.rsplit("_", 1)[-1], "step": 2,
            "reliability": np.nan, "action": "",
        })

    edge_rows: List[Dict[str, Any]] = []
    # Role-to-incident edges encode the actual incident scope represented in
    # the replay rather than inventing an electrical feeder topology.
    for agent in agents:
        if layer_for_role[role(agent)] == "service":
            edge_rows.append({
                "record_type": "service_scope_edge", "source": agent,
                "target": "utility_restoration_incident_%s" % incident_number(agent),
                "layer": "service", "role": role(agent),
                "incident": incident_number(agent), "step": 2,
                "reliability": 1.0, "action": "service_scope",
            })
    delivered = [
        value for value in events
        if value["kind"] == "v6_sketch" and value["step"] == 2
    ]
    seen_edges = set()
    for value in delivered:
        key = (value["actor"], value["payload"]["recipient"])
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_rows.append({
            "record_type": "delivered_sketch", "source": key[0], "target": key[1],
            "layer": "communication", "role": role(key[0]),
            "incident": value["payload"]["incident_id"].rsplit("_", 1)[-1],
            "step": 2, "reliability": value["payload"]["reliability"],
            "action": "entropy_sketch",
        })
    scheduled = [value for value in events if value["kind"] == "v6_action_scheduled"]
    for value in scheduled:
        proposal = value["payload"]["proposal"]
        edge_rows.append({
            "record_type": "restoration_action", "source": value["actor"],
            "target": proposal["incident_id"], "layer": "restoration",
            "role": proposal["role"],
            "incident": proposal["incident_id"].rsplit("_", 1)[-1],
            "step": value["step"], "reliability": 1.0,
            "action": proposal["action"],
        })
    rows = pd.DataFrame(node_rows + edge_rows)
    rows["source_episode"] = str(episode.relative_to(root))
    rows["cluster_id"] = snapshot["cluster_id"]
    _data(root, "utility_cyber_physical_network", rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.axis("off")
    for layer, y in y_values.items():
        ax.axhspan(y - .35, y + .35, color=COLORS["light"], alpha=.45, zorder=0)
        ax.text(-.05, y, layer.title() + " layer", ha="right", va="center", fontsize=10, fontweight="bold")
    for incident, (x, y) in incident_positions.items():
        ax.scatter([x], [y], s=170, marker="D", color=COLORS["red"], edgecolor=COLORS["black"], zorder=5)
        ax.text(x, y + .19, "Incident " + incident.rsplit("_", 1)[-1], ha="center", va="bottom", fontsize=9)
    for row in edge_rows:
        source = positions[row["source"]]
        target = incident_positions[row["target"]] if row["target"] in incident_positions else positions[row["target"]]
        if row["record_type"] == "delivered_sketch":
            color, style, width, alpha = COLORS["blue"], "--", .7 + row["reliability"], .22
        elif row["record_type"] == "restoration_action":
            color, style, width, alpha = COLORS["orange"], "-", 1.7, .72
        else:
            color, style, width, alpha = COLORS["green"], "-", 1.3, .55
        ax.plot([source[0], target[0]], [source[1], target[1]], color=color, ls=style, lw=width, alpha=alpha, zorder=1)
    node_colors = {"service": COLORS["green"], "telemetry": COLORS["purple"], "restoration": COLORS["orange"]}
    role_labels = {
        "distribution_node": "Distribution",
        "critical_load": "Critical load",
        "communications": "Comms",
        "cyber_defense": "Cyber",
        "field_crew": "Crew",
        "resource_allocation": "Resource",
    }
    for agent, (x, y) in positions.items():
        layer = layer_for_role[role(agent)]
        ax.scatter([x], [y], s=145, color=node_colors[layer], edgecolor=COLORS["black"], zorder=4)
        ax.text(
            x, y - .18,
            role_labels[role(agent)] + "\n" + incident_number(agent),
            ha="center", va="top", fontsize=9, linespacing=.9,
        )
    legend = [
        Line2D([], [], color=COLORS["green"], lw=1.5, label="Service scope"),
        Line2D([], [], color=COLORS["blue"], lw=1.5, ls="--", label="Delivered entropy sketch"),
        Line2D([], [], color=COLORS["orange"], lw=1.8, label="Scheduled restoration action"),
        Line2D([], [], marker="D", color="none", markerfacecolor=COLORS["red"], markeredgecolor=COLORS["black"], label="Simulated service incident"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.16), ncol=2)
    ax.set(xlim=(-.15, 7.0), ylim=(-.55, 3.9))
    ax.set_title("Three-layer utility-restoration replay after an abstract cyber-physical disruption")
    return _save(fig,root,"utility_cyber_physical_network")


def dashboard_export(root: Path) -> str:
    episodes = sorted((root / "raw" / "development_dynamic").glob(
        "v6-development_dynamic-utility_restoration-*-private_fragmented-e*-combined_generalized_entropic_crossfit-*/episode.json*"
    ))
    replay = None
    frame = None
    episode = None
    selection_rule = "first lexicographic utility panel with a populated authorized view"
    for candidate in episodes:
        candidate_replay = V6DashboardReplay(candidate)
        frames = [value for value in candidate_replay.frames if value.view_hashes]
        if frames:
            episode, replay, frame = candidate, candidate_replay, frames[0]
            break
    if replay is None or frame is None or episode is None:
        qwen_episodes = sorted((root / "raw" / "qwen").glob(
            "v6-qwen-utility_restoration-*/episode.json*"
        ))
        for candidate in qwen_episodes:
            candidate_replay = V6DashboardReplay(candidate)
            frames = [value for value in candidate_replay.frames if value.view_hashes]
            if frames:
                episode, replay, frame = candidate, candidate_replay, frames[0]
                selection_rule = "first lexicographic real-Qwen utility panel with a populated authorized view"
                break
    if (replay is None or frame is None or episode is None) and episodes:
        episode = episodes[0]
        replay = V6DashboardReplay(episode)
        frame = replay.frames[-1]
        selection_rule = "first lexicographic utility panel; no operator alert occurred"
    if replay is None or frame is None or episode is None:
        raise RuntimeError("no utility replay is available for dashboard export")
    destination=root/"dashboard_exports"; destination.mkdir(parents=True,exist_ok=True)
    svg=destination/"utility_restoration_populated_replay.svg"; svg.write_text(frame_svg_v6(frame)+"\n",encoding="utf-8")
    metadata=pd.DataFrame([{"episode":str(episode.relative_to(root)),"step":frame.step,"view_hash":frame.view_hashes[-1] if frame.view_hashes else "no_alert","replay_digest":replay.digest(),"selection_rule":selection_rule}]); _data(root,"operator_dashboard",metadata)
    converter=shutil.which("rsvg-convert")
    if converter is None: raise RuntimeError("rsvg-convert is required")
    pdf=root/"figures"/"pdf"/"operator_dashboard.pdf"; png=root/"figures"/"png"/"operator_dashboard.png"; pdf.parent.mkdir(parents=True,exist_ok=True); png.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([converter,"-f","pdf","-o",str(pdf),str(svg)],check=True)
    subprocess.run([converter,"-f","png","-w","2400","-o",str(png),str(svg)],check=True)
    return str(pdf.relative_to(root))


def matched_dashboard_export(root: Path) -> str:
    """Export two real, matched replay frames without outcome-based selection."""
    summaries = _read(root, "development/dynamic/episode_summary.csv")
    baseline_names = sorted(
        value for value in summaries.controller.unique()
        if value.endswith("_crossfit") and "combined_generalized" not in value
    )
    if not baseline_names:
        raise RuntimeError("no frozen non-entropic cross-fit controller found")
    baseline = baseline_names[0]
    keys = ["application", "regime", "information_condition", "environment_seed"]
    left = summaries[summaries.controller == baseline]
    right = summaries[summaries.controller == "combined_generalized_entropic_crossfit"]
    matched = left.merge(right, on=keys, suffixes=("_baseline", "_entropic"), validate="one_to_one")
    matched = matched[
        (matched.application == "utility_restoration")
        & (matched.information_condition == "private_fragmented")
    ].sort_values(keys, kind="mergesort")
    selected: Tuple[Any, V6DashboardReplay, V6DashboardReplay, V6DashboardFrame, V6DashboardFrame, str] | None = None
    fallback: Tuple[Any, V6DashboardReplay, V6DashboardReplay, V6DashboardFrame, V6DashboardFrame, str] | None = None
    for row in matched.itertuples(index=False):
        left_path = root / "raw" / "development_dynamic" / row.run_id_baseline / "episode.json.gz"
        right_path = root / "raw" / "development_dynamic" / row.run_id_entropic / "episode.json.gz"
        left_replay = V6DashboardReplay(left_path)
        right_replay = V6DashboardReplay(right_path)
        left_frames = [value for value in left_replay.frames if value.view_hashes]
        right_frames = [value for value in right_replay.frames if value.view_hashes]
        if left_frames and right_frames:
            step = max(left_frames[0].step, right_frames[0].step)
            selected = (
                row, left_replay, right_replay,
                left_replay.frame(step), right_replay.frame(step),
                "first lexicographic matched panel with populated authorized views in both methods",
            )
            break
        if fallback is None and (left_frames or right_frames):
            step = (left_frames or right_frames)[0].step
            fallback = (
                row, left_replay, right_replay,
                left_replay.frame(step), right_replay.frame(step),
                "first lexicographic matched panel with an authorized view in either method; the other panel honestly shows no alert",
            )
    selected = selected or fallback
    if selected is None:
        raise RuntimeError("no lexicographically selected matched panel has a populated replay view")
    row, left_replay, right_replay, left_frame, right_frame, selection_rule = selected
    left_svg = frame_svg_v6(left_frame)
    right_svg = frame_svg_v6(right_frame)

    def body(value: str) -> str:
        return value[value.find(">") + 1:value.rfind("</svg>")]

    combined = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="570" viewBox="0 0 1600 570">'
        '<rect width="100%" height="100%" fill="#F6F8FA"/>'
        '<style>text{font-family:Liberation Sans,Arial,sans-serif;fill:#18212F}</style>'
        '<text x="400" y="30" text-anchor="middle" font-size="22" font-weight="700">'
        + html.escape(baseline.replace("_crossfit", "").replace("_", " "))
        + '</text><text x="1200" y="30" text-anchor="middle" font-size="22" font-weight="700">'
        'KPI + generalized-entropic controller</text>'
        '<g transform="translate(0,42) scale(.6667)">' + body(left_svg) + '</g>'
        '<g transform="translate(800,42) scale(.6667)">' + body(right_svg) + '</g>'
        '<text x="800" y="560" text-anchor="middle" font-size="14">'
        'Matched panel selected lexicographically before inspecting outcomes · simulated operator · authorized views only</text>'
        '</svg>'
    )
    destination = root / "dashboard_exports"
    destination.mkdir(parents=True, exist_ok=True)
    svg_path = destination / "matched_kpi_entropic_replay.svg"
    svg_path.write_text(combined + "\n", encoding="utf-8")
    _data(root, "matched_operator_dashboard", pd.DataFrame([{
        "application": row.application,
        "regime": row.regime,
        "information_condition": row.information_condition,
        "environment_seed": row.environment_seed,
        "baseline_controller": baseline,
        "baseline_run_id": row.run_id_baseline,
        "entropic_run_id": row.run_id_entropic,
        "baseline_step": left_frame.step,
        "entropic_step": right_frame.step,
        "baseline_view_hash": left_frame.view_hashes[-1] if left_frame.view_hashes else "no_alert",
        "entropic_view_hash": right_frame.view_hashes[-1] if right_frame.view_hashes else "no_alert",
        "baseline_replay_digest": left_replay.digest(),
        "entropic_replay_digest": right_replay.digest(),
        "selection_rule": selection_rule,
    }]))
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required")
    pdf = root / "figures" / "pdf" / "matched_operator_dashboard.pdf"
    png = root / "figures" / "png" / "matched_operator_dashboard.png"
    subprocess.run([converter, "-f", "pdf", "-o", str(pdf), str(svg_path)], check=True)
    subprocess.run([converter, "-f", "png", "-w", "3200", "-o", str(png), str(svg_path)], check=True)
    return str(pdf.relative_to(root))


def causal_chain_funnel(root: Path) -> str:
    """Show operator and autonomous chains as separate nested populations."""
    delegations = _read(root, "development/dynamic/delegation_decisions.csv")
    actions = _read(root, "development/dynamic/completed_actions.csv")
    controller = "combined_generalized_entropic_crossfit"
    delegations = delegations[delegations.controller == controller]
    actions = actions[actions.controller == controller].copy()
    truth = lambda value: value.astype(str).str.lower().isin(["true", "1"])
    physical = truth(actions.accepted_physical_action)
    reached_next = truth(actions.reached_next_stage)
    reached_service = truth(actions.reached_service)
    beneficial = truth(actions.beneficial)
    operator = actions.source == "bounded_simulated_operator"
    autonomous = actions.source == "autonomous_agent"
    rows = [
        {"population": "operator", "stage_order": 0, "stage": "Escalation requests", "count": int((delegations.delegation_action == "escalate_operator").sum())},
        {"population": "operator", "stage_order": 1, "stage": "Completed operator actions", "count": int(operator.sum())},
        {"population": "operator", "stage_order": 2, "stage": "Accepted physical actions", "count": int((operator & physical).sum())},
        {"population": "operator", "stage_order": 3, "stage": "Reached next stage", "count": int((operator & physical & reached_next).sum())},
        {"population": "operator", "stage_order": 4, "stage": "Reached service", "count": int((operator & physical & reached_next & reached_service).sum())},
        {"population": "operator", "stage_order": 5, "stage": "Reached service and improved loss", "count": int((operator & physical & reached_next & reached_service & beneficial).sum())},
        {"population": "autonomous", "stage_order": 0, "stage": "Autonomous execution decisions", "count": int((delegations.delegation_action == "execute_autonomously").sum())},
        {"population": "autonomous", "stage_order": 1, "stage": "Completed autonomous actions", "count": int(autonomous.sum())},
        {"population": "autonomous", "stage_order": 2, "stage": "Accepted physical actions", "count": int((autonomous & physical).sum())},
        {"population": "autonomous", "stage_order": 3, "stage": "Reached next stage", "count": int((autonomous & physical & reached_next).sum())},
        {"population": "autonomous", "stage_order": 4, "stage": "Reached service", "count": int((autonomous & physical & reached_next & reached_service).sum())},
        {"population": "autonomous", "stage_order": 5, "stage": "Reached service and improved loss", "count": int((autonomous & physical & reached_next & reached_service & beneficial).sum())},
    ]
    frame = pd.DataFrame(rows)
    _data(root, "causal_chain_funnel", frame)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=False)
    for ax, population, color, title in (
        (axes[0], "operator", COLORS["red"], "Escalated operator path"),
        (axes[1], "autonomous", COLORS["blue"], "Autonomous execution path"),
    ):
        subset = frame[frame.population == population].sort_values("stage_order")
        y = np.arange(len(subset))[::-1]
        maximum = max(int(subset.iloc[0]["count"]), 1)
        widths = subset["count"].to_numpy(dtype=float) / maximum
        ax.barh(y, widths, color=mpl.colors.to_rgba(color, .78), edgecolor=color)
        for position, width, count in zip(y, widths, subset["count"]):
            if width >= .28:
                ax.text(width / 2, position, "%d" % count, ha="center", va="center", fontsize=9, color="white")
            else:
                ax.text(width + .025, position, "%d" % count, ha="left", va="center", fontsize=9, color=COLORS["black"])
        ax.set(yticks=y, yticklabels=subset.stage, xlim=(0, 1.14), xlabel="Fraction of panel-specific starting population", title=title)
    fig.suptitle("Separate causal-chain populations; counts are never pooled across paths")
    return _save(fig, root, "causal_chain_funnel")


def entropy_ablation(root: Path) -> str:
    frame=_read(root,"development/entropy_family/entropy_family_summary.csv"); frame=frame[(frame.application.isin(["humanitarian","utility_restoration"])) & (frame.information_condition=="private_fragmented")]
    _data(root,"entropy_family_ablation",frame)
    pivot=frame.pivot(index="entropy_measure",columns="application",values="harm_rate_at_50pct_coverage").sort_values("humanitarian")
    fig,ax=plt.subplots(figsize=(7.2,5.5)); y=np.arange(len(pivot)); width=.36
    ax.barh(y-width/2,pivot.humanitarian,width,label="Humanitarian",color=APP_COLORS["humanitarian"]); ax.barh(y+width/2,pivot.utility_restoration,width,label="Utility restoration",color=APP_COLORS["utility_restoration"])
    ax.set(yticks=y,yticklabels=[value.replace("_"," ") for value in pivot.index],xlabel="Harmful-action rate at 50% coverage",title="Prespecified generalized-entropy family ablation"); ax.legend()
    return _save(fig,root,"entropy_family_ablation")


def generate(root: Path) -> List[str]:
    configure_style()
    outputs = [
        architecture(root), agent_operator_flow(root), entropy_curves(root), entropy_spectra(root),
        phase_plane(root), graph_consensus(root),
        coverage_curves(root,"harmful_action_rate","risk_coverage","Harmful-action rate"),
        coverage_curves(root,"harmful_actions","harm_coverage","Harmful actions per panel"),
        coverage_curves(root,"mean_causal_utility","utility_coverage","Mean causal utility"),
        operator_pareto(root), communication_pareto(root), effect_forest(root), dynamic_effect_forest(root), fragmented_public(root),
        v5_abstention(root,"v5_same_score_abstention","harmful_action_rate","V5 fair same-score abstention reanalysis"),
        v5_abstention(root,"coverage_matched_escalation","mean_operator_minutes","Coverage- and budget-matched escalation"),
        training_curves(root), seed_evaluation(root), qwen_evaluation(root), calibration(root),
        regime_heterogeneity(root), timing_recovery(root), utility_network(root), entropy_ablation(root),
        dashboard_export(root), matched_dashboard_export(root), causal_chain_funnel(root),
    ]
    rows = []
    for relative_pdf in outputs:
        name = Path(relative_pdf).stem
        data_path = root / FIGURE_DATA_SOURCES[name]
        if not data_path.exists():
            raise FileNotFoundError("missing source data for figure %s" % name)
        rows.append({
            "figure": relative_pdf,
            "source_data": FIGURE_DATA_SOURCES[name],
            "source_data_rows": len(pd.read_csv(data_path)),
            "data_derived": True,
        })
    _data(root, "figure_provenance", pd.DataFrame(rows))
    return outputs
