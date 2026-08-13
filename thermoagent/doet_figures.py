"""Vector paper figures for the entropy-triggered v2 study."""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from .doet_analysis import _common_panel_subset


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
}

METHOD_STYLE = {
    "autonomous_no_comm": (PALETTE["gray"], "X", ":"),
    "fixed_always_on": (PALETTE["orange"], "D", "-"),
    "periodic_communication": (PALETTE["purple"], "s", "--"),
    "random_budget_matched": (PALETTE["yellow"], "v", ":"),
    "learned_no_entropy": (PALETTE["blue"], "^", "--"),
    "thermoagent": (PALETTE["sky"], "*", "-."),
    "doet_rule": (PALETTE["green"], "o", "-"),
    "doet_rl": (PALETTE["red"], "P", "-"),
    "kpi_cusum_trigger": (PALETTE["black"], "h", "--"),
    "global_entropy_trigger_oracle": (PALETTE["sky"], ">", ":"),
    "disruption_label_oracle": (PALETTE["black"], "d", ":"),
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _save(fig: Any, name: str, root: Path) -> str:
    pdf_dir = root / "figures" / "pdf"
    preview_dir = root / "figures" / "previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / (name + ".pdf")
    png = preview_dir / (name + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=220)
    plt.close(fig)
    return pdf.name


def _label(method: str) -> str:
    labels = {
        "fixed_always_on": "Always-on fixed",
        "periodic_communication": "Periodic",
        "random_budget_matched": "Random matched",
        "learned_no_entropy": "Learned, no entropy",
        "thermoagent": "ThermoAgent v1",
        "doet_rule": "DOET-rule",
        "doet_rl": "DOET-RL",
        "kpi_cusum_trigger": "Local KPI CUSUM",
        "autonomous_no_comm": "No communication",
        "global_entropy_trigger_oracle": "Global entropy oracle",
        "disruption_label_oracle": "Disruption oracle",
    }
    return labels.get(method, method.replace("_", " "))


def _ci(values: Sequence[float], seed: int = 20260813) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, mean, mean
    rng = np.random.RandomState(seed)
    draws = array[rng.randint(0, len(array), size=(4000, len(array)))].mean(axis=1)
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def architecture(root: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    boxes = [
        (0.1, 6.1, 3.1, 1.55, "Private agent\nidentity + utility\nobservation + memory", PALETTE["blue"]),
        (4.45, 6.1, 3.1, 1.55, "Distributed monitor\ncoarse sketch gossip\n$\widehat{S}_i$, change, confidence", PALETTE["purple"]),
        (8.8, 6.1, 3.1, 1.55, "Stateful trigger\nCUSUM + hysteresis\nlocal confidence", PALETTE["red"]),
        (0.1, 2.75, 3.1, 1.55, "Quiet mode\nlocal plan\nsparse gossip", PALETTE["gray"]),
        (4.45, 2.75, 3.1, 1.55, "Targeted mode\nbilateral information\nand negotiation", PALETTE["orange"]),
        (8.8, 2.75, 3.1, 1.55, "Crisis mode\ncoalition coordination\nwithin hard budgets", PALETTE["green"]),
    ]
    for x, y, width, height, text, color in boxes:
        box = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.08",
            facecolor=mpl.colors.to_rgba(color, 0.12),
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", weight="semibold", fontsize=8.9, linespacing=1.2)
    arrows = [
        ((3.2, 6.88), (4.45, 6.88), "local macrostate"),
        ((7.55, 6.88), (8.8, 6.88), "local statistic"),
    ]
    for start, end, text in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.15, "color": PALETTE["black"]})
        ax.text((start[0] + end[0]) / 2, 7.92, text, ha="center", va="center", fontsize=7.8, backgroundcolor="white")
    for end_x in (1.65, 6.0, 10.35):
        ax.annotate("", xy=(end_x, 4.3), xytext=(10.35, 6.1), arrowprops={"arrowstyle": "->", "lw": 1.05, "color": PALETTE["black"]})
    ax.text(6.0, 4.88, "local mode eligibility", ha="center", fontsize=8.2, backgroundcolor="white")
    ax.annotate("", xy=(10.8, 1.55), xytext=(1.2, 1.55), arrowprops={"arrowstyle": "<->", "linestyle": "--", "color": PALETTE["gray"], "lw": 1.2})
    ax.text(6, 1.82, "Explicit counted messages, alerts, sketches, offers, and coalition contracts", ha="center", color=PALETTE["gray"], fontsize=8.8)
    ax.text(6, 0.72, "Independent authority remains local: each agent can accept, counter, refuse, or withdraw", ha="center", weight="semibold")
    ax.text(6, 0.25, "Exact global entropy and true disruption labels are evaluator-only oracle ablations", ha="center", color=PALETTE["gray"], fontsize=8.5)
    fig.suptitle("Distributed Operational Entropy Triggering (DOET)", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "doet_architecture", root)


def trigger_dynamics(root: Path) -> str:
    case = pd.read_csv(root / "processed" / "commercial_event_case_study.csv")
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 8.6), sharex=True)
    specs = [
        ("distributed_entropy_mean", "Distributed entropy", PALETTE["blue"]),
        ("mean_trigger_statistic_agents", "Mean trigger statistic", PALETTE["red"]),
        ("trigger_active_agents", "Active agents", PALETTE["orange"]),
        ("operational_messages_this_step", "Messages this period", PALETTE["purple"]),
        ("service_loss", "Service loss", PALETTE["green"]),
    ]
    disruption = int(case["disruption_step"].iloc[0])
    activation_values = case["first_activation_step"].dropna()
    activation = int(activation_values.iloc[0]) if len(activation_values) else None
    for ax, (metric, label, color) in zip(axes, specs):
        ax.plot(case["step"], case[metric], marker="o", markersize=3.2, color=color, linewidth=1.4)
        ax.set_ylabel(label)
        ax.axvline(disruption, color=PALETTE["black"], linestyle="--", linewidth=1.0)
        if activation is not None:
            ax.axvline(activation, color=PALETTE["red"], linestyle=":", linewidth=1.2)
    axes[-1].set_xlabel("Simulator period")
    axes[0].legend(handles=[
        Line2D([0], [0], color=PALETTE["black"], linestyle="--", label="Disruption"),
        Line2D([0], [0], color=PALETTE["red"], linestyle=":", label="First activation"),
    ], loc="best", frameon=False)
    fig.suptitle("DOET trigger dynamics in a representative commercial holdout episode")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.0)
    return _save(fig, "trigger_dynamics", root)


def performance_communication_pareto(root: Path) -> str:
    frame = pd.read_csv(root / "statistics" / "pareto_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.15))
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        app = frame[frame["application"] == application]
        for _, row in app.iterrows():
            method = str(row["method"])
            color, marker, _ = METHOD_STYLE.get(method, (PALETTE["gray"], "o", "-"))
            ax.scatter(row["mean_total_communication_messages"], row["mean_primary_outcome"], s=52, marker=marker, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        frontier = app[app["pareto_nondominated_loss_messages"].astype(str).str.lower().isin(["true", "1"])].sort_values("mean_total_communication_messages")
        if len(frontier):
            ax.plot(frontier["mean_total_communication_messages"], frontier["mean_primary_outcome"], color=PALETTE["black"], linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("All messages per episode\n(including entropy sketches)")
        ax.set_ylabel("Primary loss (lower is better)")
        ax.set_title(application.capitalize())
    methods = [method for method in METHOD_STYLE if method in set(frame["method"])]
    handles = [Line2D([0], [0], color=METHOD_STYLE[method][0], marker=METHOD_STYLE[method][1], linestyle="none", label=_label(method)) for method in methods]
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.015), frameon=False,
    )
    fig.suptitle("Locked-holdout performance–communication frontier")
    fig.tight_layout(rect=(0, 0.25, 1, 0.94))
    return _save(fig, "performance_communication_pareto", root)


def noninferiority_forest(root: Path) -> str:
    frame = pd.read_csv(root / "statistics" / "main_paired_comparisons.csv")
    frame = frame[(frame["method"] == "doet_rule")]
    order = []
    for application in ("commercial", "humanitarian"):
        for scenario in ("all_non_nominal", "isolated", "communication_partition", "correlated", "compound_ood"):
            match = frame[(frame["application"] == application) & (frame["scenario"] == scenario)]
            if len(match):
                order.append(match.iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for index, row in enumerate(order):
        color = PALETTE["blue"] if row["application"] == "commercial" else PALETTE["green"]
        mean = 100 * float(row["mean_relative_degradation"])
        low = 100 * float(row["relative_degradation_ci95_low"])
        high = 100 * float(row["relative_degradation_ci95_high"])
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=3, markersize=5.5)
    ax.axvline(2.0, color=PALETTE["red"], linestyle="--", linewidth=1.2, label="2% non-inferiority margin")
    ax.axvline(0.0, color=PALETTE["black"], linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([
        "%s — %s" % (str(row["application"]).capitalize(), str(row["scenario"]).replace("_", " "))
        for row in order
    ])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Relative primary-loss degradation vs always-on fixed (%)")
    ax.legend(frameon=False, loc="best")
    ax.set_title("DOET-rule non-inferiority on the locked holdout")
    fig.tight_layout()
    return _save(fig, "noninferiority_forest", root)


def communication_reduction(root: Path) -> str:
    frame = pd.read_csv(root / "processed" / "holdout_results.csv")
    non_nominal = frame[frame["scenario_name"] != "nominal"]
    metrics = [
        ("total_communication_messages", "Messages"),
        ("total_communication_bytes", "Structured bytes"),
        ("prompt_tokens", "Prompt tokens"),
        ("llm_calls", "LLM calls"),
        ("llm_latency_seconds", "LLM latency"),
    ]
    methods = [method for method in ("doet_rule", "doet_rl", "periodic_communication", "random_budget_matched", "kpi_cusum_trigger") if method in set(non_nominal["method"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.35), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        fixed = non_nominal[non_nominal["method"] == "fixed_always_on"].set_index(["application", "scenario_name", "seed"])[metric]
        for index, method in enumerate(methods):
            values = non_nominal[non_nominal["method"] == method].set_index(["application", "scenario_name", "seed"])[metric]
            joined = pd.concat([values.rename("method"), fixed.rename("fixed")], axis=1, join="inner").dropna()
            reduction = 100 * (1.0 - joined["method"] / joined["fixed"].clip(lower=1e-9))
            mean, low, high = _ci(reduction)
            color, marker, _ = METHOD_STYLE[method]
            ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt=marker, color=color, capsize=2.5, markersize=5)
        ax.axvline(20, color=PALETTE["red"], linestyle="--", linewidth=0.9)
        ax.axvline(0, color=PALETTE["black"], linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Reduction (%)")
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([_label(method) for method in methods])
    axes[0].set_ylim(len(methods) - 0.5, -0.5)
    fig.suptitle("Communication and inference reduction versus always-on fixed")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "communication_reduction", root)


def multiple_seed_learning_curves(root: Path) -> str:
    frame = pd.read_csv(root / "training" / "learning_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharey=True)
    variants = ("no_entropy", "thermo", "doet_rl")
    for ax, variant in zip(axes, variants):
        subset = frame[frame["variant"] == variant]
        for seed, group in subset.groupby("rl_training_seed"):
            rolling = group.sort_values("episode")["reward_sum"].rolling(12, min_periods=1).mean()
            ax.plot(group.sort_values("episode")["episode"], rolling, linewidth=1.0, alpha=0.75, label=str(int(seed)))
        ax.set_title(_label("learned_no_entropy" if variant == "no_entropy" else "thermoagent" if variant == "thermo" else "doet_rl"))
        ax.set_xlabel("Training episode")
    axes[0].set_ylabel("12-episode rolling reward")
    axes[-1].legend(title="RL seed", frameon=False, loc="best", ncol=2)
    fig.suptitle("Independent coordination-policy training seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "multiple_seed_learning_curves", root)


def training_seed_variability(root: Path) -> str:
    frame = pd.read_csv(root / "statistics" / "training_seed_variability.csv")
    applications = [value for value in ("commercial", "humanitarian") if value in set(frame["application"])]
    methods = [value for value in ("learned_no_entropy", "thermoagent", "doet_rl") if value in set(frame["method"])]
    fig, axes = plt.subplots(1, len(applications), figsize=(7.2, 3.4), squeeze=False)
    for ax, application in zip(axes[0], applications):
        app = frame[frame["application"] == application]
        for index, method in enumerate(methods):
            values = app[app["method"] == method]["mean_primary_outcome"].to_numpy(dtype=float)
            color, marker, _ = METHOD_STYLE[method]
            jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(len(values))
            ax.scatter(index + jitter, values, marker=marker, color=color, s=34, alpha=0.8)
            ax.hlines(np.mean(values), index - 0.23, index + 0.23, color=color, linewidth=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(method) for method in methods], rotation=24, ha="right")
        ax.set_ylabel("Mean locked-holdout primary loss")
        ax.set_title(application.capitalize())
    fig.suptitle("Variation across independently trained RL seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "training_seed_variability", root)


def holdout_primary_results(root: Path) -> str:
    frame = pd.read_csv(root / "processed" / "holdout_results.csv")
    methods = [method for method in ("fixed_always_on", "periodic_communication", "random_budget_matched", "learned_no_entropy", "thermoagent", "doet_rule", "doet_rl", "kpi_cusum_trigger") if method in set(frame["method"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), squeeze=False)
    for row_index, application in enumerate(("commercial", "humanitarian")):
        ax = axes[row_index, 0]
        app = frame[(frame["application"] == application) & (frame["scenario_name"] != "nominal")]
        # Secondary controls use a preregistered compute-capped subset. Keep
        # the visual comparison on the exact panel intersection so method
        # means cannot be shifted by unequal scenario seeds. The full primary
        # sample remains in the non-inferiority forest.
        app = _common_panel_subset(app, set(methods))
        for index, method in enumerate(methods):
            values = app[app["method"] == method]["primary_outcome"].to_numpy(dtype=float)
            mean, low, high = _ci(values)
            color, marker, _ = METHOD_STYLE[method]
            jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(len(values))
            ax.scatter(index + jitter, values, s=9, facecolors="none", edgecolors=color, linewidths=0.5, alpha=0.45)
            ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt=marker, color=color, markersize=6, capsize=3)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(method) for method in methods], rotation=24, ha="right")
        ax.set_ylabel("Service-loss AUC" if application == "commercial" else "Cumulative unmet weighted need")
        ax.set_title(application.capitalize())
    fig.suptitle("Locked-holdout primary results (common matched panels)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "holdout_primary_results", root)


def partition_robustness(root: Path) -> str:
    frame = pd.read_csv(root / "processed" / "holdout_results.csv")
    partitions = frame[(frame["method"] == "doet_rule") & frame["scenario_name"].isin(["communication_partition", "compound_ood"])]
    fixed = frame[frame["method"] == "fixed_always_on"][["application", "scenario_name", "seed", "primary_outcome"]].rename(columns={"primary_outcome": "fixed_loss"})
    joined = partitions.merge(fixed, on=["application", "scenario_name", "seed"], validate="one_to_one")
    joined["relative_degradation"] = (joined["primary_outcome"] - joined["fixed_loss"]) / joined["fixed_loss"].abs().clip(lower=1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for application, color, marker in (("commercial", PALETTE["blue"], "o"), ("humanitarian", PALETTE["green"], "s")):
        app = joined[joined["application"] == application]
        axes[0].scatter(app["mean_consensus_rmse"], 100 * app["relative_degradation"], color=color, marker=marker, alpha=0.75, label=application.capitalize())
        grouped = app.groupby("scenario_name").agg(consensus=("mean_consensus_rmse", "mean"), activation=("trigger_activations", "mean")).reset_index()
        axes[1].plot(grouped["consensus"], grouped["activation"], color=color, marker=marker, linewidth=1.2, label=application.capitalize())
    axes[0].axhline(2, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Mean local consensus RMSE")
    axes[0].set_ylabel("Loss degradation vs fixed (%)")
    axes[1].set_xlabel("Mean local consensus RMSE")
    axes[1].set_ylabel("Mean activations per episode")
    axes[0].legend(frameon=False)
    fig.suptitle("DOET robustness under communication partitions")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "partition_robustness", root)


def trigger_ablation_effects(root: Path) -> str:
    validation = pd.read_csv(root / "validation" / "trigger_candidate_comparison.csv")
    holdout_path = root / "ablations" / "episodes.csv"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
    ordered = validation.sort_values("mean_message_reduction", ascending=False)
    y = np.arange(len(ordered))
    axes[0].errorbar(100 * ordered["mean_relative_degradation"], y, fmt="o", color=PALETTE["blue"], label="Loss degradation")
    axes[0].axvline(2, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ordered["method_variant"].str.replace("_", " "))
    axes[0].set_xlabel("Validation loss degradation (%)")
    axes[0].set_title("Trigger and hysteresis candidates")
    if holdout_path.exists():
        ablations = pd.read_csv(holdout_path)
        fixed = ablations[ablations["method"] == "doet_rule"]["primary_outcome"].mean()
        grouped = ablations.groupby(["method", "method_variant"])["primary_outcome"].mean().reset_index()
        grouped["effect"] = 100 * (grouped["primary_outcome"] - fixed) / max(abs(fixed), 1e-9)
        names = (grouped["method"].astype(str) + ":" + grouped["method_variant"].astype(str)).str.replace("_", " ")
        axes[1].barh(np.arange(len(grouped)), grouped["effect"], color=PALETTE["green"])
        axes[1].set_yticks(np.arange(len(grouped)))
        axes[1].set_yticklabels(names)
        axes[1].set_xlabel("Primary-loss change vs selected DOET (%)")
    else:
        axes[1].text(0.5, 0.5, "Locked ablations pending", transform=axes[1].transAxes, ha="center", va="center", color=PALETTE["gray"])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Signal and distributed-estimate controls")
    fig.suptitle("DOET trigger ablation effects")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "trigger_ablation_effects", root)


def event_case_study(root: Path, application: str) -> str:
    case = pd.read_csv(root / "processed" / (application + "_event_case_study.csv"))
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.2), sharex=True)
    metrics = [
        ("distributed_entropy_mean", "Distributed entropy", PALETTE["blue"]),
        ("mean_trigger_statistic_agents", "Trigger statistic", PALETTE["red"]),
        ("operational_messages_this_step", "Messages", PALETTE["orange"]),
        ("service_loss", "Service loss", PALETTE["green"]),
    ]
    disruption = int(case["disruption_step"].iloc[0])
    activations = case["first_activation_step"].dropna()
    activation = int(activations.iloc[0]) if len(activations) else None
    for ax, (metric, label, color) in zip(axes, metrics):
        ax.plot(case["step"], case[metric], color=color, marker="o", markersize=3, linewidth=1.3)
        ax.set_ylabel(label)
        ax.axvline(disruption, color=PALETTE["black"], linestyle="--", linewidth=1.0)
        if activation is not None:
            ax.axvline(activation, color=PALETTE["red"], linestyle=":", linewidth=1.1)
    axes[-1].set_xlabel("Simulator period")
    axes[0].set_title("%s DOET event sequence" % application.capitalize())
    fig.tight_layout(h_pad=1.0)
    return _save(fig, application + "_event_case_study", root)


def _events(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def network_snapshots(root: Path) -> str:
    case = pd.read_csv(root / "processed" / "commercial_event_case_study.csv")
    run_id = str(case["run_id"].iloc[0])
    run_dir = root / "raw" / "holdout_locked" / run_id
    events = _events(run_dir / "events.jsonl.gz")
    topology = next(event["payload"] for event in events if event["kind"] == "topology_snapshot")
    disruption = int(case["disruption_step"].iloc[0])
    active_rows = case[case.get("trigger_active_agents", 0) > 0]
    targeted_rows = case[
        case.get("trigger_active_agents", 0)
        > case.get("trigger_crisis_agents", 0)
    ]
    crisis_rows = case[case.get("trigger_crisis_agents", 0) > 0]
    activation_values = case["first_activation_step"].dropna()
    activation = (
        int(activation_values.iloc[0])
        if len(activation_values) else
        int(active_rows["step"].iloc[0]) if len(active_rows) else disruption + 1
    )
    targeted_step = (
        int(targeted_rows["step"].iloc[0])
        if len(targeted_rows) else activation
    )
    crisis_step = (
        int(crisis_rows["step"].iloc[0])
        if len(crisis_rows) else min(int(case["step"].max()), activation + 2)
    )
    steps = [
        max(0, disruption - 2), max(0, disruption - 1), targeted_step,
        crisis_step, int(case["step"].max()),
    ]
    titles = [
        "Quiet", "Entropy deviation", "Targeted communication",
        "Crisis coalition" if len(crisis_rows) else "No crisis activation",
        "Recovery",
    ]
    identities = topology["agents"]
    pos = {agent_id: tuple(row["location"]) for agent_id, row in identities.items()}
    physical_initial = {tuple(edge) for edge in topology["physical_edges"]}
    communication_initial = {tuple(sorted(edge)) for edge in topology["communication_edges"]}
    role_shapes = {"retailer": "s", "supplier": "o", "manufacturer": "^", "carrier": "D", "warehouse": "h"}
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.5))
    for ax, step, title in zip(axes.flat[:5], steps, titles):
        available_communication = set(communication_initial)
        observations: Dict[str, Mapping[str, Any]] = {}
        trigger: Dict[str, Mapping[str, Any]] = {}
        closed = set()
        active_messages = []
        commitment_edges = set()
        coalition_groups: Dict[str, set] = {}
        for event in events:
            if int(event["step"]) > step:
                continue
            if event["kind"] == "observation_delivery" and int(event["step"]) == step:
                observations[str(event["payload"]["recipient"])] = event["payload"]["observation"]
            elif event["kind"] == "coordination_trigger" and int(event["step"]) == step:
                trigger[str(event["actor"])] = event["payload"]
            elif event["kind"] == "disruption":
                closed.update(tuple(edge) for edge in event["payload"].get("route_closures", []))
                coordinator = event["payload"].get("coordinator_loss")
                if coordinator:
                    available_communication = {
                        edge for edge in available_communication
                        if coordinator not in edge
                    }
            elif event["kind"] == "message" and int(event["step"]) in (step, step - 1):
                active_messages.append((str(event["actor"]), str(event["payload"]["recipient"]), str(event["payload"].get("kind"))))
            elif event["kind"] == "commitment" and event["payload"].get("status") == "accepted":
                commitment_edges.add((str(event["payload"]["proposer"]), str(event["payload"]["partner"])))
            elif event["kind"] == "coalition_event":
                payload = event["payload"]
                coalition_id = str(payload.get("coalition_id", ""))
                if payload.get("action") == "propose":
                    coalition_groups[coalition_id] = set(
                        payload.get("members", [])
                    )
                elif (
                    payload.get("action") == "join_coalition"
                    and payload.get("ok")
                ):
                    coalition_groups.setdefault(coalition_id, set()).add(
                        str(event["actor"])
                    )
        graph = nx.DiGraph()
        graph.add_nodes_from(identities)
        physical = physical_initial - closed
        if (
            topology.get("communication_regime") == "partition"
            and step >= disruption
        ):
            ordered_nodes = sorted(identities)
            midpoint = len(ordered_nodes) // 2
            left_partition = set(ordered_nodes[:midpoint])
            available_communication = {
                edge for edge in available_communication
                if (edge[0] in left_partition) == (edge[1] in left_partition)
            }
        nx.draw_networkx_edges(graph, pos, edgelist=list(physical), ax=ax, edge_color="#BBBBBB", width=1.0, arrows=True, arrowsize=6)
        nx.draw_networkx_edges(graph, pos, edgelist=list(available_communication), ax=ax, edge_color=PALETTE["gray"], style="dashed", width=0.55, arrows=False, alpha=0.55)
        values = [float(trigger.get(node, {}).get("local_surprisal", 0.0)) for node in identities]
        sizes = [
            min(
                520.0,
                80.0 + 18.0 * float(
                    observations.get(node, {}).get("backlog", 0.0)
                ),
            )
            for node in identities
        ]
        for role, shape in role_shapes.items():
            nodes = [node for node, row in identities.items() if row["role"] == role]
            if not nodes:
                continue
            indices = [list(identities).index(node) for node in nodes]
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_shape=shape, node_size=[sizes[index] for index in indices], node_color=[values[index] for index in indices], cmap="viridis", vmin=0, vmax=max(1.0, max(values) if values else 1.0), edgecolors=PALETTE["black"], linewidths=0.6, ax=ax)
        for sender, recipient, kind in active_messages:
            color = PALETTE["red"] if kind == "entropy_alert" else PALETTE["orange"]
            nx.draw_networkx_edges(graph, pos, edgelist=[(sender, recipient)], ax=ax, edge_color=color, width=1.5, arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.12")
        nx.draw_networkx_edges(graph, pos, edgelist=list(commitment_edges), ax=ax, edge_color=PALETTE["green"], width=2.2, arrows=True, arrowsize=8)
        for members in coalition_groups.values():
            member_positions = [pos[node] for node in members if node in pos]
            if len(member_positions) < 2:
                continue
            x_values = [value[0] for value in member_positions]
            y_values = [value[1] for value in member_positions]
            outline = patches.Ellipse(
                ((min(x_values) + max(x_values)) / 2.0,
                 (min(y_values) + max(y_values)) / 2.0),
                max(max(x_values) - min(x_values) + 0.35, 0.55),
                max(max(y_values) - min(y_values) + 0.35, 0.55),
                fill=False, edgecolor=PALETTE["purple"], linestyle="-.",
                linewidth=1.2, alpha=0.8,
            )
            ax.add_patch(outline)
        nx.draw_networkx_labels(graph, pos, labels={node: node.split("_")[0][:3] + node[-2:] for node in identities}, font_size=6.5, ax=ax)
        ax.set_title("%s (period %d)" % (title, step))
        ax.set_axis_off()
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([0], [0], color="#BBBBBB", linewidth=1.2, label="Physical route"),
        Line2D([0], [0], color=PALETTE["gray"], linestyle="--", label="Available communication"),
        Line2D([0], [0], color=PALETTE["red"], linewidth=1.5, label="Entropy alert"),
        Line2D([0], [0], color=PALETTE["orange"], linewidth=1.5, label="Negotiation message"),
        Line2D([0], [0], color=PALETTE["green"], linewidth=2.2, label="Accepted commitment"),
        Line2D([0], [0], color=PALETTE["purple"], linestyle="-.", linewidth=1.2, label="Coalition membership"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor=PALETTE["black"], markersize=7, label="Node color: local surprisal"),
    ], loc="center", frameon=False)
    fig.suptitle("Network evolution under distributed entropy triggering")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "network_snapshots_entropy_trigger", root)


def generate(root: Path, architecture_only: bool = False) -> List[str]:
    configure_style()
    generated = [architecture(root)]
    if architecture_only:
        return generated
    required = [
        root / "statistics" / "main_paired_comparisons.csv",
        root / "processed" / "holdout_results.csv",
        root / "processed" / "commercial_event_case_study.csv",
        root / "processed" / "humanitarian_event_case_study.csv",
        root / "training" / "learning_curves.csv",
        root / "validation" / "trigger_candidate_comparison.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("DOET figure inputs are incomplete: %s" % missing)
    generated.extend([
        trigger_dynamics(root),
        performance_communication_pareto(root),
        noninferiority_forest(root),
        communication_reduction(root),
        multiple_seed_learning_curves(root),
        training_seed_variability(root),
        holdout_primary_results(root),
        partition_robustness(root),
        trigger_ablation_effects(root),
        event_case_study(root, "commercial"),
        event_case_study(root, "humanitarian"),
        network_snapshots(root),
    ])
    # These three are generated prospectively by the diagnostics/monitoring
    # pipeline and must remain present rather than silently regenerated from a
    # different post-holdout source.
    for name in (
        "original_holdout_tie_diagnostics.pdf",
        "monitoring_baseline_comparison.pdf",
        "entropy_incremental_value.pdf",
    ):
        path = root / "figures" / "pdf" / name
        if not path.exists():
            raise FileNotFoundError(path)
        generated.append(name)
    return sorted(set(generated))
