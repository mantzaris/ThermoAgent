"""Paper-facing vector figures for V11; PNG renders are QA-only and external."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .workflow import atomic_csv


COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#777777",
    "black": "#111111",
}


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _save(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(figure)
    os.replace(str(temporary), str(destination))


def _box(axis: plt.Axes, x: float, y: float, text: str, color: str, width: float = 0.20) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.075),
            width,
            0.15,
            boxstyle="round,pad=0.012",
            facecolor=color,
            edgecolor=COLORS["black"],
            linewidth=0.9,
        )
    )
    axis.text(x, y, text, ha="center", va="center", fontsize=9)


def _arrow(axis: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], dashed: bool = False) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.3,
            linestyle="--" if dashed else "-",
            color=COLORS["gray"],
        )
    )


def figure_01_architecture(results: Path) -> None:
    records = [
        {"record": "node", "name": name, "information_class": info}
        for name, info in (
            ("private signal", "agent-private"),
            ("continuous belief", "agent-private"),
            ("binary belief", "derived local"),
            ("typed action", "agent-selected"),
            ("commitment", "agent-private"),
            ("inbox/outbox", "delivered only"),
            ("scheduler", "update opportunity only"),
            ("evaluator", "offline global"),
        )
    ]
    atomic_csv(records, results / "figures/source_data/figure_01_architecture.csv")
    figure, axis = plt.subplots(figsize=(7.4, 3.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(axis, 0.10, 0.72, "private\nsignal", COLORS["sky"], 0.16)
    _box(axis, 0.30, 0.72, "reported\nprobability", COLORS["green"], 0.18)
    _box(axis, 0.50, 0.72, "derived belief\n+ typed action", COLORS["orange"], 0.19)
    _box(axis, 0.70, 0.72, "commitment +\nprivate memory", COLORS["purple"], 0.19)
    _box(axis, 0.90, 0.72, "inbox /\noutbox", COLORS["sky"], 0.16)
    for left, right in ((0.18, 0.21), (0.39, 0.405), (0.595, 0.605), (0.795, 0.815)):
        _arrow(axis, (left, 0.72), (right, 0.72))
    _box(axis, 0.32, 0.26, "scheduler\noffers turn", "#DDDDDD", 0.20)
    _box(axis, 0.62, 0.26, "environment applies\nagent-selected action", COLORS["blue"], 0.25)
    _box(axis, 0.88, 0.26, "evaluator-only\nlatent state", "#EEEEEE", 0.18)
    _arrow(axis, (0.40, 0.26), (0.49, 0.26))
    _arrow(axis, (0.50, 0.64), (0.58, 0.34))
    _arrow(axis, (0.75, 0.26), (0.79, 0.26), dashed=True)
    axis.text(0.5, 0.95, "Decentralized V11 information and authority boundary", ha="center", weight="bold", fontsize=11)
    axis.text(0.88, 0.08, "offline scoring only", ha="center", fontsize=9, color=COLORS["gray"])
    _save(figure, results / "figures/pdf/figure_01_architecture.pdf")


def figure_02_evidence_process(results: Path) -> None:
    records = [
        {"stage": index, "name": name, "agent_visible": int(visible)}
        for index, (name, visible) in enumerate(
            [
                ("latent theta", False),
                ("Bernoulli signal with reliability r", True),
                ("typed packet and binary serialization", True),
                ("explicit directed delivery", True),
                ("local evidence integration", True),
                ("offline Bayesian reference", False),
            ]
        )
    ]
    atomic_csv(records, results / "figures/source_data/figure_02_evidence_process.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
    axes[0].set_title("Generative model", weight="bold")
    _box(axes[0], 0.15, 0.65, "latent $\\theta$\n(evaluator)", "#EEEEEE", 0.22)
    _box(axes[0], 0.50, 0.65, "signal $s_i$\n$P(s_i=\\theta)=r$", COLORS["sky"], 0.25)
    _box(axes[0], 0.85, 0.65, "packet\n+ wire bytes", COLORS["green"], 0.25)
    _arrow(axes[0], (0.27, 0.65), (0.36, 0.65))
    _arrow(axes[0], (0.63, 0.65), (0.73, 0.65))
    axes[0].text(0.50, 0.30, r"Bayes reference: $\Delta\ell=\pm\log[r/(1-r)]$", ha="center", fontsize=9.5)
    axes[1].set_title("Matched intervention", weight="bold")
    _box(axes[1], 0.16, 0.68, "send /\nabstain", COLORS["orange"], 0.24)
    _box(axes[1], 0.50, 0.68, "directed\ndelivery", COLORS["purple"], 0.20)
    _box(axes[1], 0.84, 0.68, "report\n$P(\\theta=R)$", COLORS["green"], 0.24)
    _arrow(axes[1], (0.29, 0.68), (0.38, 0.68))
    _arrow(axes[1], (0.61, 0.68), (0.72, 0.68))
    axes[1].text(0.50, 0.28, "matched private state, template, order, and inference seed", ha="center", fontsize=8.8)
    _save(figure, results / "figures/pdf/figure_02_evidence_process.pdf")


def _mean_interval(values: np.ndarray, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = np.mean(values[rng.integers(0, values.size, (10000, values.size))], axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def figure_03_belief_response(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/belief_response.csv")
    selected = data[data["condition"].str.startswith("single_") & (data["expected_direction"] != 0)].copy()
    figure, axis = plt.subplots(figsize=(6.4, 3.7))
    for index, (domain, color, marker) in enumerate(
        [("route_viability", COLORS["blue"], "o"), ("repair_hypothesis", COLORS["vermillion"], "s")]
    ):
        part = selected[selected["domain"] == domain]
        rows = []
        for reliability, group in part.groupby("nominal_reliability", sort=True):
            cluster_values = group.groupby("cluster_id")["signed_logit_change"].mean().to_numpy(float)
            summary = _mean_interval(cluster_values, 1300 + index + int(100 * reliability))
            rows.append((float(reliability),) + summary)
        values = np.asarray(rows)
        axis.errorbar(
            values[:, 0],
            values[:, 1],
            yerr=[values[:, 1] - values[:, 2], values[:, 3] - values[:, 1]],
            color=color,
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=domain.replace("_", " "),
        )
        jitter = (index - 0.5) * 0.007
        points = part.groupby(["cluster_id", "nominal_reliability"], sort=True)["signed_logit_change"].mean().reset_index()
        axis.scatter(points["nominal_reliability"] + jitter, points["signed_logit_change"], s=13, alpha=0.28, color=color)
    axis.axhline(0.0, color=COLORS["gray"], lw=1)
    axis.axhline(0.10, color=COLORS["gray"], lw=1, ls="--", label="frozen practical minimum")
    axis.set(xlabel="source reliability", ylabel="signed change in belief log odds")
    axis.legend(frameon=False, ncol=2)
    axis.text(0.99, 0.02, "n=24 independent clusters/application", transform=axis.transAxes, ha="right", va="bottom", fontsize=9)
    _save(figure, results / "figures/pdf/figure_03_belief_response.pdf")


def figure_04_calibration(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/reported_vs_empirical_calibration.csv")
    figure, axis = plt.subplots(figsize=(4.7, 4.2))
    for domain, color, marker in (
        ("route_viability", COLORS["blue"], "o"),
        ("repair_hypothesis", COLORS["vermillion"], "s"),
    ):
        part = data[data["domain"] == domain]
        axis.scatter(
            part["mean_reported_probability"],
            part["empirical_right_frequency"],
            s=24,
            alpha=0.55,
            color=color,
            marker=marker,
            label=domain.replace("_", " "),
        )
    axis.plot([0, 1], [0, 1], "--", color=COLORS["gray"], lw=1.2, label="perfect calibration")
    axis.set(xlim=(-0.02, 1.02), ylim=(-0.03, 1.03), xlabel="mean reported probability right", ylabel="empirical right-choice frequency")
    axis.legend(frameon=False, loc="upper left")
    axis.text(0.98, 0.03, "2 repeated samples/cell", transform=axis.transAxes, ha="right", va="bottom", fontsize=9)
    _save(figure, results / "figures/pdf/figure_04_calibration.pdf")


def _formal_figures(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/formal_panel_metrics.csv")
    primary = data[data["panel_family"] == "primary"].copy()
    controls = data[data["panel_family"] == "control"].copy()

    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    for application, color, marker in (("humanitarian", COLORS["blue"], "o"), ("utility", COLORS["vermillion"], "s")):
        part = primary[primary["application"] == application]
        for alpha, group in part.groupby("alpha", sort=True):
            axis.scatter(np.full(len(group), alpha), group["adjusted_block_kl"], color=color, alpha=0.28, s=18, marker=marker)
        means = part.groupby("alpha")["adjusted_block_kl"].mean()
        axis.plot(means.index, means.values, color=color, marker=marker, lw=1.8, label=application)
    axis.axhline(0, color=COLORS["gray"], lw=1)
    axis.set(xlabel=r"nonreciprocity $\alpha$", ylabel="bias-corrected block time-reversal KL")
    axis.legend(frameon=False)
    _save(figure, results / "figures/pdf/figure_05_transition_current_comparison.pdf")

    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    grouped = primary.groupby(["application", "alpha"])["adjusted_block_kl"].agg(["mean", "sem"]).reset_index()
    for application, color, marker in (("humanitarian", COLORS["blue"], "o"), ("utility", COLORS["vermillion"], "s")):
        part = grouped[grouped["application"] == application]
        axis.errorbar(part["alpha"], part["mean"], yerr=1.96 * part["sem"], color=color, marker=marker, capsize=3, label=application)
    axis.set(xlabel=r"nonreciprocity $\alpha$", ylabel="trajectory irreversibility (nats / block)")
    axis.legend(frameon=False)
    _save(figure, results / "figures/pdf/figure_06_irreversibility_nonreciprocity.pdf")

    theory = pd.read_csv(results.parent / "llm_agent_entropy_v10/figures/source_data/figure_02_quadratic_onset.csv")
    atomic_csv(theory.to_dict(orient="records"), results / "figures/source_data/figure_07_v10_theory_reference.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].plot(theory["alpha"], theory["total_per_update_mean"], "o-", color=COLORS["green"], label="exact heat-bath")
    axes[0].plot(theory["alpha"], theory["quadratic_prediction_mean"], "--", color=COLORS["black"], label=r"$C\alpha^2$")
    axes[0].set(xlabel=r"$\alpha$", ylabel="exact EPR / update", title="Analytical reference")
    axes[0].legend(frameon=False)
    means = primary.groupby("alpha")["adjusted_block_kl"].mean()
    baseline = float(means.iloc[0])
    axes[1].plot(means.index, means.values - baseline, "o-", color=COLORS["purple"], label="LLM coarse lower bound")
    x = means.index.to_numpy(float) ** 2
    y = means.to_numpy(float) - baseline
    coefficient = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
    axes[1].plot(means.index, coefficient * x, "--", color=COLORS["black"], label="zero-intercept quadratic")
    axes[1].set(xlabel=r"$\alpha$", ylabel="excess irreversibility", title="Empirical LLM network")
    axes[1].legend(frameon=False)
    _save(figure, results / "figures/pdf/figure_07_quadratic_comparison.pdf")

    figure, axis = plt.subplots(figsize=(5.7, 3.7))
    layer = primary.groupby("alpha")[["belief_adjusted_block_kl", "action_adjusted_block_kl"]].mean()
    axis.plot(layer.index, layer["belief_adjusted_block_kl"], "o-", color=COLORS["blue"], label="belief layer")
    axis.plot(layer.index, layer["action_adjusted_block_kl"], "s--", color=COLORS["orange"], label="action layer")
    axis.axhline(0, color=COLORS["gray"], lw=1)
    axis.set(xlabel=r"nonreciprocity $\alpha$", ylabel="bias-corrected layer block KL")
    axis.legend(frameon=False)
    _save(figure, results / "figures/pdf/figure_08_layer_decomposition.pdf")

    currents = pd.read_csv(results / "figures/source_data/coarse_transition_currents.csv")
    selected = currents[currents["alpha"].isin([currents["alpha"].min(), currents["alpha"].max()])]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for axis, alpha in zip(axes, sorted(selected["alpha"].unique())):
        part = selected[selected["alpha"] == alpha].nlargest(18, "transition_count")
        states = sorted(set(part["source_state"]) | set(part["destination_state"]))
        angles = np.linspace(0, 2 * np.pi, len(states), endpoint=False)
        positions = {state: (np.cos(angle), np.sin(angle)) for state, angle in zip(states, angles)}
        for row in part.to_dict(orient="records"):
            start, end = positions[int(row["source_state"])], positions[int(row["destination_state"])]
            axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, lw=0.4 + 0.15 * np.sqrt(row["transition_count"]), alpha=0.45, color=COLORS["purple"]))
        for state, position in positions.items():
            axis.scatter(*position, s=30, color=COLORS["sky"], edgecolor=COLORS["black"], linewidth=0.5, zorder=3)
        axis.set_title(r"$\alpha=%.2f$" % alpha)
        axis.set_aspect("equal")
        axis.axis("off")
    _save(figure, results / "figures/pdf/figure_09_probability_currents.pdf")

    figure, axis = plt.subplots(figsize=(6.3, 3.8))
    order = sorted(controls["control"].unique())
    means = controls.groupby("control")["adjusted_block_kl"].mean().reindex(order)
    sem = controls.groupby("control")["adjusted_block_kl"].sem().reindex(order)
    positions = np.arange(len(order))
    axis.errorbar(means.values, positions, xerr=1.96 * sem.values, fmt="o", color=COLORS["blue"], capsize=3)
    axis.set_yticks(positions, [value.replace("_", " ") for value in order])
    axis.axvline(0, color=COLORS["gray"], lw=1)
    axis.set(xlabel="bias-corrected irreversibility", ylabel="control")
    _save(figure, results / "figures/pdf/figure_10_controls.pdf")

    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.4))
    for topology, marker in (("ring", "o"), ("modular", "s")):
        part = primary[primary["topology"] == topology]
        axes[0].scatter(part["alpha"], part["adjusted_block_kl"], alpha=0.35, marker=marker, label=topology)
    axes[0].set(xlabel=r"$\alpha$", ylabel="adjusted irreversibility")
    axes[0].legend(frameon=False)
    axes[1].scatter(primary["markov_cmi_history_1"], primary["markov_cmi_history_2"], c=primary["alpha"], cmap="viridis", s=24)
    axes[1].set(xlabel="history-1 conditional MI", ylabel="history-2 conditional MI")
    _save(figure, results / "figures/pdf/figure_11_topology_seed_heterogeneity.pdf")

    convergence = pd.read_csv(results / "figures/source_data/estimator_convergence.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.4))
    for alpha, group in convergence.groupby("alpha"):
        mean = group.groupby("prefix_turns")["block_time_reversal_kl"].mean()
        axes[0].plot(mean.index, mean.values, marker="o", label=r"$\alpha=%.2f$" % alpha)
    axes[0].set(xlabel="trajectory prefix (turns)", ylabel="block time-reversal KL")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].scatter(primary["turns"], primary["markov_cmi_history_2"], c=primary["alpha"], cmap="viridis", s=24)
    axes[1].set(xlabel="valid turns", ylabel="history-2 conditional MI")
    _save(figure, results / "figures/pdf/figure_12_markov_convergence.pdf")


def generate_figures(repository: Path) -> List[str]:
    _configure()
    results = Path(repository) / "results/llm_agent_entropy_v11"
    figure_01_architecture(results)
    figure_02_evidence_process(results)
    generated = ["figure_01_architecture.pdf", "figure_02_evidence_process.pdf"]
    if (results / "figures/source_data/belief_response.csv").exists():
        figure_03_belief_response(results)
        figure_04_calibration(results)
        generated.extend(["figure_03_belief_response.pdf", "figure_04_calibration.pdf"])
    if (results / "figures/source_data/formal_panel_metrics.csv").exists():
        _formal_figures(results)
        generated.extend(
            [
                "figure_05_transition_current_comparison.pdf",
                "figure_06_irreversibility_nonreciprocity.pdf",
                "figure_07_quadratic_comparison.pdf",
                "figure_08_layer_decomposition.pdf",
                "figure_09_probability_currents.pdf",
                "figure_10_controls.pdf",
                "figure_11_topology_seed_heterogeneity.pdf",
                "figure_12_markov_convergence.pdf",
            ]
        )
    return generated
