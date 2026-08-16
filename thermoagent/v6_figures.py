"""Data-derived publication figures for generalized-entropic V6 evidence."""

from __future__ import annotations

import gzip
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

from .dashboard.v6 import V6DashboardReplay, frame_svg_v6
from .v6_entropy import shannon_entropy, tsallis_entropy


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
            ha="right", va="bottom", fontsize=8.5, color=COLORS["gray"],
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
        (0.4, 7.4, "Independent agents\nprivate belief · memory · utility", "blue"),
        (4.5, 7.4, "Ad-hoc messages\ncompressed entropy spectra", "green"),
        (8.6, 7.4, "Distributed estimator\nuncertainty · disagreement · residual", "orange"),
        (8.6, 4.6, "Delegation controller\nexecute · communicate · abstain", "purple"),
        (4.5, 4.6, "Bounded simulated operator\nqueue · minutes · finite budget", "red"),
        (0.4, 4.6, "Typed operational action\nrole mask · accept/reject/counter", "blue"),
        (2.45, 1.7, "Dynamic state transition\nresources · commitments · service", "green"),
        (6.65, 1.7, "Evaluator-only branch\nmatched stochastic tape", "orange"),
    ]
    _data(root, "architecture", pd.DataFrame(rows, columns=["x", "y", "component", "color"]))
    fig, ax = plt.subplots(figsize=(7.5, 6.2)); ax.set(xlim=(0, 12), ylim=(0, 10)); ax.axis("off")
    for x, y, label, color in rows:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), 3.0, 1.35, boxstyle="round,pad=0.06",
            facecolor=mpl.colors.to_rgba(COLORS[color], 0.12),
            edgecolor=COLORS[color], linewidth=1.6,
        ))
        ax.text(x + 1.5, y + 0.675, label, ha="center", va="center", fontsize=9.2)
    arrows = [((3.4, 8.08), (4.5, 8.08)), ((7.5, 8.08), (8.6, 8.08)),
              ((10.1, 7.4), (10.1, 5.95)), ((8.6, 5.28), (7.5, 5.28)),
              ((4.5, 5.28), (3.4, 5.28)), ((1.9, 4.6), (3.4, 3.05)),
              ((5.45, 2.38), (6.65, 2.38))]
    for first, second in arrows:
        ax.annotate("", xy=second, xytext=first, arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(6, 0.72, "Environment validates actions; it never substitutes an oracle decision.", ha="center")
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
        points = ax.scatter(subset.shannon_local, subset.js_disagreement, c=subset.harmful.astype(int), cmap=mpl.colors.ListedColormap([COLORS["green"], COLORS["red"]]), s=15, alpha=.48)
        ax.set(title=APP_LABELS[application], xlabel="Local uncertainty (Shannon)")
    axes[0].set_ylabel("Epistemic disagreement (Jensen–Shannon)")
    fig.suptitle("Aleatoric uncertainty and epistemic disagreement are distinct")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS["green"], label="Beneficial/neutral proposal"), Line2D([], [], marker="o", ls="", color=COLORS["red"], label="Harmful proposal")], loc="lower center", ncol=2)
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
    edges = pd.DataFrame([{
        "sender": value["actor"], "recipient": value["payload"]["recipient"],
        "reliability": value["payload"]["reliability"], "step": value["step"],
    } for value in sketches])
    _data(root, "graph_weighted_consensus_network", edges)
    nodes = sorted(set(edges.sender).union(edges.recipient))
    positions = {node: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes))) for index, node in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(7.0, 5.8)); ax.axis("off")
    for row in edges.itertuples(index=False):
        x1, y1 = positions[row.sender]; x2, y2 = positions[row.recipient]
        ax.plot([x1, x2], [y1, y2], color=COLORS["blue"], alpha=.18 + .55 * row.reliability, lw=.6 + 1.8 * row.reliability)
    for node, (x, y) in positions.items():
        ax.scatter([x], [y], s=120, color=COLORS["sky"], edgecolor=COLORS["black"], zorder=3)
        ax.text(x, y - .10, node.split("_")[-2], ha="center", va="top", fontsize=8)
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
    axes[1].legend(loc="best", fontsize=7.7)
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
            ax.annotate("generalized" if "combined" in row.controller else "baseline", (row.operator_minutes, row.service_loss), xytext=(4, 4), textcoords="offset points", fontsize=8)
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
    frame = _read(root, "development/dynamic/paired_dynamic_effects.csv")
    _data(root, "entropy_family_effect_forest", frame)
    labels = [APP_LABELS[row.application] + " · " + row.information_condition.replace("_", " ") for row in frame.itertuples(index=False)]
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for position, row in zip(y, frame.itertuples(index=False)):
        ax.errorbar(row.harm_rate_reduction, position, xerr=[[row.harm_rate_reduction - row.harm_ci95_low], [row.harm_ci95_high - row.harm_rate_reduction]], fmt="o", color=APP_COLORS[row.application], capsize=3)
    ax.axvline(0, color=COLORS["black"], lw=1)
    ax.axvline(.03, color=COLORS["red"], lw=1, ls="--", label="Frozen practical threshold")
    ax.set(yticks=y, yticklabels=labels, xlabel="Harm-rate reduction vs strongest non-entropic baseline", title="Cross-fitted dynamic selective-safety effects")
    ax.legend()
    return _save(fig, root, "entropy_family_effect_forest")


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
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.8), sharey=True)
    for ax, app in zip(axes, ("commercial", "humanitarian", "utility_restoration")):
        subset = grouped[grouped.application == app]
        ax.barh(range(len(subset)), subset[metric], color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]][:len(subset)])
        ax.set(yticks=range(len(subset)), yticklabels=[value.replace("_", " ") for value in subset.policy] if ax is axes[0] else [], title=APP_LABELS[app])
    axes[0].set_xlabel(metric.replace("_", " ")); fig.suptitle(title)
    return _save(fig, root, name, stamp=False)


def training_curves(root: Path) -> str:
    files = sorted((root / "training" / "curves").glob("*.csv"))
    frame = pd.concat([pd.read_csv(value) for value in files], ignore_index=True)
    _data(root, "sequential_rl_learning_curves", frame)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.1))
    for method, subset in frame.groupby("method"):
        grouped = subset.groupby("training_episode").mean(numeric_only=True).reset_index()
        axes[0].plot(grouped.training_episode, grouped.mean_trajectory_reward, label=method.replace("ppo_", "").replace("_", " "))
        axes[1].plot(grouped.training_episode, grouped.policy_entropy)
    axes[0].set(xlabel="Training episode", ylabel="Mean trajectory reward", title="All five-seed means")
    axes[1].set(xlabel="Training episode", ylabel="Policy entropy", title="Exploration stability")
    axes[0].legend(fontsize=7.5)
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
        rows.append({"application": app, "decisions": len(subset), "first_pass": subset.first_pass_valid.mean(), "harm_rate": physical.harmful.mean() if len(physical) else 0, "mean_effect": physical.causal_effect.mean() if len(physical) else 0, "abstention": (subset.delegation=="abstain").mean(), "escalation": (subset.delegation=="escalate_operator").mean()})
    summary = pd.DataFrame(rows); _data(root, "qwen_agent_evaluation", summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))
    x=np.arange(len(summary)); width=.22
    for offset, key, label in [(-width,"first_pass","First-pass valid"),(0,"harm_rate","Harmful physical"),(width,"abstention","Abstention")]:
        axes[0].bar(x+offset,summary[key],width,label=label)
    axes[0].set(xticks=x,xticklabels=[APP_LABELS[v] for v in summary.application],ylim=(0,1),ylabel="Fraction",title="Behavioral rates"); axes[0].legend(fontsize=7.5)
    axes[1].bar(x,summary.mean_effect,color=[APP_COLORS[v] for v in summary.application]); axes[1].axhline(0,color=COLORS["black"]); axes[1].set(xticks=x,xticklabels=[APP_LABELS[v] for v in summary.application],ylabel="Mean causal effect",title="Accepted physical actions")
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
        ax.errorbar(subset.harm_rate_reduction,y,xerr=[subset.harm_rate_reduction-subset.harm_ci95_low,subset.harm_ci95_high-subset.harm_rate_reduction],fmt="o",capsize=2,color=APP_COLORS[app]); ax.axvline(0,color=COLORS["black"]); ax.set(yticks=y,yticklabels=subset.regime if ax is axes[0] else [],xlabel="Harm reduction",title=APP_LABELS[app])
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
    episode=_representative_episode(root,"utility_restoration"); events=_events(episode)
    snapshot=next(value["payload"] for value in events if value["kind"]=="v6_panel_snapshot"); agents=sorted(snapshot["agent_ids"])
    edges=[value for value in events if value["kind"]=="v6_sketch" and value["step"]==2]
    rows=pd.DataFrame([{"sender":value["actor"],"recipient":value["payload"]["recipient"],"reliability":value["payload"]["reliability"]} for value in edges]); _data(root,"utility_cyber_physical_network",rows)
    positions={agent:(math.cos(2*math.pi*i/len(agents)),math.sin(2*math.pi*i/len(agents))) for i,agent in enumerate(agents)}
    fig,ax=plt.subplots(figsize=(7.2,5.8)); ax.axis("off")
    for row in rows.itertuples(index=False):
        x1,y1=positions[row.sender]; x2,y2=positions[row.recipient]; ax.plot([x1,x2],[y1,y2],color=COLORS["blue"],alpha=.30,lw=1+row.reliability)
    for agent,(x,y) in positions.items():
        role=agent.replace("utility_restoration_","").rsplit("_",2)[0]; color=COLORS["red"] if "cyber" in role or "communications" in role else COLORS["orange"] if "crew" in role else COLORS["green"]
        ax.scatter([x],[y],s=145,color=color,edgecolor=COLORS["black"],zorder=3); ax.text(x,y-.10,role.replace("_"," "),ha="center",fontsize=8)
    ax.set_title("Abstract defensive utility-restoration network after cyber-physical disruption")
    return _save(fig,root,"utility_cyber_physical_network")


def dashboard_export(root: Path) -> str:
    episode=_representative_episode(root,"utility_restoration"); replay=V6DashboardReplay(episode); frames=[value for value in replay.frames if value.view_hashes]
    if not frames: raise RuntimeError("representative V6 replay has no populated operator view")
    frame=frames[-1]; destination=root/"dashboard_exports"; destination.mkdir(parents=True,exist_ok=True)
    svg=destination/"utility_restoration_populated_replay.svg"; svg.write_text(frame_svg_v6(frame)+"\n",encoding="utf-8")
    metadata=pd.DataFrame([{"episode":str(episode.relative_to(root)),"step":frame.step,"view_hash":frame.view_hashes[-1],"replay_digest":replay.digest()}]); _data(root,"operator_dashboard",metadata)
    converter=shutil.which("rsvg-convert")
    if converter is None: raise RuntimeError("rsvg-convert is required")
    pdf=root/"figures"/"pdf"/"operator_dashboard.pdf"; png=root/"figures"/"png"/"operator_dashboard.png"; pdf.parent.mkdir(parents=True,exist_ok=True); png.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([converter,"-f","pdf","-o",str(pdf),str(svg)],check=True)
    subprocess.run([converter,"-f","png","-w","2400","-o",str(png),str(svg)],check=True)
    return str(pdf.relative_to(root))


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
        operator_pareto(root), communication_pareto(root), effect_forest(root), fragmented_public(root),
        v5_abstention(root,"v5_same_score_abstention","harmful_action_rate","V5 fair same-score abstention reanalysis"),
        v5_abstention(root,"coverage_matched_escalation","mean_operator_minutes","Coverage- and budget-matched escalation"),
        training_curves(root), seed_evaluation(root), qwen_evaluation(root), calibration(root),
        regime_heterogeneity(root), timing_recovery(root), utility_network(root), entropy_ablation(root),
        dashboard_export(root),
    ]
    return outputs
