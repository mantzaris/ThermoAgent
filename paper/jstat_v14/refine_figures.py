#!/usr/bin/env python3
"""Publication-only layout refinements for selected V14 figures.

This script reads the frozen aggregate figure-source CSVs and changes only
typography, annotation placement, and panel layout. It is deliberately kept
outside the formal execution-source checksum.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/collective_agent_statmech_v14"
SOURCE = RESULT / "figures/source_data"
PDF = RESULT / "figures/pdf"
COLORS = {
    "nominal": "#0072B2",
    "field_reversal": "#D55E00",
    "network_partition": "#009E73",
    "message_corruption": "#CC79A7",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PDF / name, format="pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _shade(ax: plt.Axes) -> None:
    ax.axvspan(15, 30, color="#F0E442", alpha=0.12, lw=0)
    ax.axvline(15, color="0.35", ls="--", lw=1.1)
    ax.axvline(30, color="0.35", ls=":", lw=1.1)


def _phase_portrait(name: str, x: str, y: str, xlabel: str, ylabel: str) -> None:
    frame = pd.read_csv(SOURCE / f"{name}.csv")
    mean = frame.groupby("sweep", as_index=False)[[x, y]].mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    points = ax.scatter(mean[x], mean[y], c=mean["sweep"], cmap="viridis", s=34, zorder=3)
    ax.plot(mean[x], mean[y], color="0.5", lw=1.1, zorder=1)
    positions = {
        1: (7, 7, "start 1"),
        16: (18, 18, "quench 16"),
        31: (20, -24, "restore 31"),
        45: (-48, -20, "end 45"),
    }
    for sweep, (dx, dy, label) in positions.items():
        row = mean.loc[mean["sweep"] == sweep].iloc[0]
        ax.annotate(
            label,
            (row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7},
        )
    colorbar = fig.colorbar(points, ax=ax, label="Sweep", pad=0.02)
    colorbar.ax.tick_params(labelsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, f"{name}.pdf")


def _entropy_decomposition() -> None:
    frame = pd.read_csv(SOURCE / "figure09_entropy_decomposition.csv")
    summary = frame.groupby(["entropy_component", "sweep"], as_index=False)["value"].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, constrained_layout=True)
    specs = [
        ("mean_individual_entropy", "Individual entropy", "#0072B2", "o"),
        ("configuration_entropy", "Configuration entropy", "#D55E00", "s"),
    ]
    for component, label, color, marker in specs:
        part = summary[summary["entropy_component"] == component]
        axes[0].plot(part["sweep"], part["mean"], color=color, marker=marker, markevery=4, label=label)
        axes[0].fill_between(part["sweep"], part["mean"] - 1.96 * part["sem"], part["mean"] + 1.96 * part["sem"], color=color, alpha=0.15)
    total = summary[summary["entropy_component"] == "total_correlation"]
    axes[1].plot(total["sweep"], total["mean"], color="#009E73", marker="^", markevery=4, label="Total correlation")
    axes[1].fill_between(total["sweep"], total["mean"] - 1.96 * total["sem"], total["mean"] + 1.96 * total["sem"], color="#009E73", alpha=0.15)
    for ax in axes:
        _shade(ax)
        ax.set_xlabel("Sweep")
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Entropy (nats)")
    axes[1].set_ylabel("Total correlation (nats)")
    _save(fig, "figure09_entropy_decomposition.pdf")


def _quench_counterquench() -> None:
    frame = pd.read_csv(SOURCE / "figure14_quench_counterquench.csv")
    summary = frame.groupby("sweep", as_index=False)["macrostate_distance"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    ax.plot(summary["sweep"], summary["mean"], color=COLORS["field_reversal"], marker="s", markevery=3, label="Field reversal")
    ax.fill_between(summary["sweep"], summary["mean"] - 1.96 * summary["sem"], summary["mean"] + 1.96 * summary["sem"], color=COLORS["field_reversal"], alpha=0.16)
    _shade(ax)
    ax.text(15.4, ax.get_ylim()[1] * 0.93, "Quench", ha="left", va="top", fontsize=8.5)
    ax.text(30.4, ax.get_ylim()[1] * 0.93, "Restore", ha="left", va="top", fontsize=8.5)
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Macrostate distance")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "figure14_quench_counterquench.pdf")


def _margin_refresh() -> None:
    frame = pd.read_csv(SOURCE / "figure05_memory_cluster_effects.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    for study, marker, color in (("V12", "o", "#0072B2"), ("V13", "s", "#D55E00")):
        part = frame[frame["study"] == study]
        ax.scatter(np.arange(len(part)), part["difference"], marker=marker, color=color, label=f"{study} clusters", alpha=0.8)
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_ylabel("Persistent − Markovized (nats/update)")
    ax.set_xlabel("Independent matched comparison")
    ax.legend(frameon=False)
    _save(fig, "figure05_memory_cluster_effects.pdf")

    for number, metric, ylabel in (
        (6, "adjusted_irreversibility_nats_per_update", "Adjusted reversal divergence (nats/update)"),
        (7, "shuffle_floor_nats_per_update", "Time-shuffle floor (nats/update)"),
    ):
        stem = "block_length_sensitivity" if number == 6 else "bias_floor"
        data = pd.read_csv(SOURCE / f"figure{number:02d}_{stem}.csv")
        summary = data.groupby(["study", "block_length"], as_index=False)[metric].mean()
        fig, ax = plt.subplots(figsize=(5.9, 3.4), constrained_layout=True)
        for study, part in summary.groupby("study"):
            ax.plot(part["block_length"], part[metric], marker="o", label=study.replace("_", " "))
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("Block length")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        _save(fig, f"figure{number:02d}_{stem}.pdf")

    recovery = pd.read_csv(SOURCE / "figure15_peak_integrated_recovery.csv")
    order = ["nominal", "field_reversal", "network_partition", "message_corruption"]
    labels = ["Nominal", "Field\nreversal", "Partition", "Corruption"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    for index, (condition, marker) in enumerate(zip(order, markers)):
        values = recovery[recovery["disruption"] == condition]["maximum_post_quench_distance"].to_numpy(float)
        ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, color=COLORS[condition], marker=marker, zorder=3)
        ax.errorbar(index, np.mean(values), yerr=1.96 * np.std(values, ddof=1) / np.sqrt(len(values)), fmt="_", color="black", capsize=4, ms=15)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Maximum post-quench distance")
    _save(fig, "figure15_peak_integrated_recovery.pdf")

    for number, contribution in ((18, True), (19, False)):
        stem = "observable_family_contribution" if number == 18 else "leave_family_out"
        data = pd.read_csv(SOURCE / f"figure{number:02d}_{stem}.csv")
        means = data.groupby("ablation", as_index=False)["maximum_distance"].mean()
        if contribution:
            full = float(means.loc[means["ablation"] == "all", "maximum_distance"].iloc[0])
            means["value"] = full - means["maximum_distance"]
            xlabel = "Distance loss after family removal"
        else:
            means["value"] = means["maximum_distance"]
            xlabel = "Field-reversal maximum distance"
        means = means.sort_values("value")
        display_labels = means["ablation"].map(
            lambda value: "All families"
            if value == "all"
            else "Remove "
            + (value[len("without_") :] if value.startswith("without_") else value).replace("_", " ")
        )
        fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
        ax.barh(display_labels, means["value"], color="#56B4E9")
        ax.axvline(0, color="0.35", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Observable set")
        _save(fig, f"figure{number:02d}_{stem}.pdf")


def main() -> None:
    _style()
    _entropy_decomposition()
    _phase_portrait(
        "figure11_energy_entropy_phase_space",
        "reference_energy_per_agent",
        "configuration_entropy",
        "Reference energy per agent",
        "Configuration entropy (nats)",
    )
    _phase_portrait(
        "figure12_belief_action_phase_space",
        "belief_magnetization",
        "action_magnetization",
        "Belief magnetization",
        "Action magnetization",
    )
    _quench_counterquench()
    _margin_refresh()
    names = [
        "figure05_memory_cluster_effects.pdf",
        "figure06_block_length_sensitivity.pdf",
        "figure07_bias_floor.pdf",
        "figure09_entropy_decomposition.pdf",
        "figure11_energy_entropy_phase_space.pdf",
        "figure12_belief_action_phase_space.pdf",
        "figure14_quench_counterquench.pdf",
        "figure15_peak_integrated_recovery.pdf",
        "figure18_observable_family_contribution.pdf",
        "figure19_leave_family_out.pdf",
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "publication layout only; frozen source CSV values unchanged",
        "files": [
            {
                "path": f"results/collective_agent_statmech_v14/figures/pdf/{name}",
                "sha256": hashlib.sha256((PDF / name).read_bytes()).hexdigest(),
            }
            for name in names
        ],
    }
    destination = RESULT / "reproducibility/publication_figure_refinement.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
