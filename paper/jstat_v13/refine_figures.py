#!/usr/bin/env python3
"""Apply presentation-only refinements to selected V13 vector figures.

The frozen execution source and numerical source tables are not changed.  This
script reads the exact CSVs produced by the frozen analysis and rewrites only
the corresponding paper-facing PDFs to clarify legends, arrows, and labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/collective_agent_statmech_v13"
SOURCE = RESULTS / "figures/source_data"
PDF = RESULTS / "figures/pdf"

COLORS = {
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


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(
        PDF / filename,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "ThermoAgent V13 paper-only figure refinement"},
    )
    plt.close(fig)


def figure_01() -> None:
    nodes = pd.read_csv(SOURCE / "figure_01_nodes.csv")
    edges = pd.read_csv(SOURCE / "figure_01_edges.csv")
    expected_nodes = set(nodes["node"])
    expected_edges = set(zip(edges["source"], edges["target"]))
    assert {"LLM agent i", "agent j", "evaluator Z(t)"}.issubset(expected_nodes)

    positions = {
        "private observation": (0.13, 0.70),
        "bounded memory": (0.13, 0.36),
        "LLM agent i": (0.39, 0.53),
        "inbox/outbox": (0.62, 0.70),
        "local action": (0.62, 0.36),
        "agent j": (0.87, 0.70),
        "environment": (0.87, 0.36),
        "evaluator Z(t)": (0.50, 0.065),
    }
    palette = {
        "local": COLORS["green"],
        "agent": COLORS["blue"],
        "network": COLORS["orange"],
        "action": COLORS["red"],
        "environment": COLORS["purple"],
        "evaluator": COLORS["gray"],
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.25))

    deployable = [
        ("private observation", "LLM agent i", None, None),
        ("bounded memory", "LLM agent i", None, None),
        ("LLM agent i", "inbox/outbox", None, None),
        ("inbox/outbox", "agent j", "delivered", (0.745, 0.75)),
        ("LLM agent i", "local action", None, None),
        ("local action", "environment", "typed effect", (0.745, 0.41)),
    ]
    for source, target, label, label_xy in deployable:
        assert (source, target) in expected_edges
        arrow = FancyArrowPatch(
            positions[source],
            positions[target],
            arrowstyle="-|>",
            mutation_scale=11,
            connectionstyle="arc3,rad=0",
            color=COLORS["gray"],
            linewidth=1.25,
            shrinkA=24,
            shrinkB=24,
            zorder=1,
        )
        ax.add_patch(arrow)
        if label is not None and label_xy is not None:
            ax.text(
                *label_xy,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=COLORS["gray"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
                zorder=2,
            )

    assert ("environment", "private observation") in expected_edges
    observation_path = MplPath(
        [
            (0.87, 0.43),
            (0.90, 0.91),
            (0.32, 0.93),
            (0.19, 0.73),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        FancyArrowPatch(
            path=observation_path,
            arrowstyle="-|>",
            mutation_scale=11,
            color=COLORS["gray"],
            linewidth=1.25,
            zorder=1,
        )
    )
    ax.text(
        0.53,
        0.90,
        "next local observation",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
        zorder=2,
    )

    for source in ("LLM agent i", "agent j", "environment"):
        assert (source, "evaluator Z(t)") in expected_edges
        ax.add_patch(
            FancyArrowPatch(
                positions[source],
                positions["evaluator Z(t)"],
                arrowstyle="-|>",
                mutation_scale=9,
                color="#AAAAAA",
                linestyle="--",
                linewidth=0.9,
                shrinkA=24,
                shrinkB=26,
                zorder=0,
            )
        )

    for row in nodes.itertuples():
        x, y = positions[row.node]
        ax.text(
            x,
            y,
            row.node,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": palette[row.boundary],
                "edgecolor": "white",
                "linewidth": 1.0,
            },
            zorder=3,
        )

    ax.axhline(0.17, color="#BBBBBB", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(0.50, 0.975, "Independent-agent information and authority boundary", ha="center", fontweight="bold")
    ax.text(0.68, 0.065, "offline records only; never included in an agent prompt", ha="left", va="center", color=COLORS["gray"], fontsize=7.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    save(fig, "figure_01_micro_to_macro_architecture.pdf")


def figure_07() -> None:
    data = pd.read_csv(SOURCE / "figure_07_effective_energy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for coupling, marker, color in ((0.35, "o", COLORS["blue"]), (0.80, "s", COLORS["orange"])):
        group = data[np.isclose(data["coupling_strength"], coupling)]
        axes[0].scatter(
            group["mean_abs_belief_magnetization"],
            group["mean_reference_energy_per_agent"],
            marker=marker,
            color=color,
            s=35,
            alpha=0.75,
            label=f"J={coupling:.2f}",
        )
    axes[0].set(
        xlabel="mean |belief magnetization|",
        ylabel="reference energy / agent",
        title="Effective energy and order",
    )
    axes[0].legend(frameon=False)
    for noise, marker, color in ((0.50, "o", COLORS["blue"]), (0.85, "s", COLORS["orange"])):
        group = data[np.isclose(data["sampling_temperature"], noise)]
        axes[1].scatter(
            group["belief_susceptibility"],
            group["energy_fluctuation_N_var_e"],
            marker=marker,
            color=color,
            s=35,
            alpha=0.75,
            label=f"noise {noise:.2f}",
        )
    axes[1].set(
        xlabel="belief susceptibility",
        ylabel=r"$N\,\mathrm{Var}(e_{ref})$",
        title="Fluctuation correspondence",
    )
    axes[1].legend(frameon=False)
    save(fig, "figure_07_energy_fluctuations.pdf")


def figure_12() -> None:
    data = pd.read_csv(SOURCE / "figure_12_magnetization_entropy.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.15), sharex=True, sharey=True)
    phases = (
        ("baseline", "o", COLORS["blue"]),
        ("disruption", "s", COLORS["red"]),
        ("recovery", "^", COLORS["green"]),
    )
    conditions = ("nominal", "field_reversal", "network_partition", "message_corruption")
    for axis, condition in zip(axes.ravel(), conditions):
        group = data[data["disruption"] == condition]
        for phase, marker, color in phases:
            phase_group = group[group["phase"] == phase]
            axis.scatter(
                phase_group["belief_magnetization"],
                phase_group["configuration_entropy"],
                marker=marker,
                color=color,
                alpha=0.5,
                s=18,
            )
        axis.set_title(condition.replace("_", " "))
    for axis in axes[-1]:
        axis.set_xlabel("belief magnetization")
    for axis in axes[:, 0]:
        axis.set_ylabel("configuration entropy (nats)")
    handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color=color, label=phase, markersize=6)
        for phase, marker, color in phases
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figure_12_magnetization_entropy_trajectories.pdf")


def _draw_directed_edge(
    ax: plt.Axes,
    pos: dict[int, np.ndarray],
    source: int,
    target: int,
    width: float,
    color: str,
    linestyle: str,
    alpha: float,
) -> None:
    curvature = 0.075 if source < target else -0.075
    ax.add_patch(
        FancyArrowPatch(
            pos[source],
            pos[target],
            arrowstyle="-|>",
            mutation_scale=6.5,
            connectionstyle=f"arc3,rad={curvature}",
            linewidth=width,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            shrinkA=7,
            shrinkB=7,
            zorder=1,
        )
    )


def figure_13() -> None:
    nodes = pd.read_csv(SOURCE / "figure_13_snapshot_nodes.csv")
    edges = pd.read_csv(SOURCE / "figure_13_snapshot_edges.csv")
    baseline = edges[edges["phase"] == "baseline"]
    support = nx.Graph()
    support.add_nodes_from(sorted(nodes["node"].unique()))
    support.add_edges_from((int(row.source), int(row.target)) for row in baseline.itertuples())
    pos = nx.spring_layout(support, seed=13)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35))
    for axis, phase in zip(axes, ("baseline", "disruption", "recovery")):
        node_frame = nodes[nodes["phase"] == phase].sort_values("node")
        edge_frame = edges[edges["phase"] == phase]
        for row in edge_frame.itertuples():
            source, target = int(row.source), int(row.target)
            if int(row.active) == 1 and int(row.message_count) > 0:
                _draw_directed_edge(
                    axis,
                    pos,
                    source,
                    target,
                    width=0.35 + 0.10 * float(row.message_count),
                    color=COLORS["gray"],
                    linestyle="-",
                    alpha=0.48,
                )
            elif int(row.cross_community) == 1:
                _draw_directed_edge(
                    axis,
                    pos,
                    source,
                    target,
                    width=0.65,
                    color=COLORS["red"],
                    linestyle="--",
                    alpha=0.38,
                )
        nx.draw_networkx_nodes(
            support,
            pos,
            nodelist=[int(row.node) for row in node_frame.itertuples()],
            node_color=[COLORS["blue"] if row.belief < 0 else COLORS["orange"] for row in node_frame.itertuples()],
            node_size=[75 + 155 * row.confidence_uncertainty for row in node_frame.itertuples()],
            edgecolors=[COLORS["green"] if row.action > 0 else COLORS["black"] for row in node_frame.itertuples()],
            linewidths=1.5,
            ax=axis,
        )
        axis.set_title(phase)
        axis.axis("off")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"], label="belief -1", markersize=7),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["green"], label="belief +1 / action +1 border", markersize=7),
        Line2D([0], [0], color=COLORS["gray"], marker=">", markersize=4, label="delivered message current"),
        Line2D([0], [0], color=COLORS["red"], linestyle="--", marker=">", markersize=4, label="partition-blocked direction"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], label="node size: confidence uncertainty", markersize=9),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015), fontsize=7.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "figure_13_network_partition_snapshots.pdf")


def figure_18() -> None:
    data = pd.read_csv(SOURCE / "figure_18_surrogate_direct.csv")
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    for coupling, marker, color in ((0.35, "o", COLORS["blue"]), (0.80, "s", COLORS["orange"])):
        group = data[np.isclose(data["coupling_strength"], coupling)]
        ax.scatter(
            group["surrogate_order"],
            group["mean_abs_belief_magnetization"],
            marker=marker,
            color=color,
            s=48,
            label=f"J={coupling:.2f}",
        )
    limit = max(data["surrogate_order"].max(), data["mean_abs_belief_magnetization"].max()) * 1.08
    ax.plot([0, limit], [0, limit], linestyle="--", color=COLORS["gray"], label="equality")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="fitted-surrogate order",
        ylabel="direct LLM order",
    )
    ax.legend(frameon=False, loc="upper left")
    save(fig, "figure_18_surrogate_vs_llm.pdf")


def figure_21() -> None:
    data = pd.read_csv(SOURCE / "figure_21_finite_size.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    metrics = (
        ("mean_abs_belief_magnetization", "mean |belief magnetization|"),
        ("belief_susceptibility", "belief susceptibility"),
        ("belief_integrated_autocorrelation_time_updates", "correlation time (updates)"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for n, marker, color in ((8, "o", COLORS["blue"]), (16, "s", COLORS["orange"])):
            group = data[data["n_agents"] == n]
            axis.scatter(group["coupling_strength"], group[metric], marker=marker, color=color, alpha=0.65, label=f"N={n}")
        axis.set_xlabel("coupling J")
        axis.set_ylabel(label)
    axes[-1].legend(frameon=False)
    save(fig, "figure_21_finite_size.pdf")


def main() -> None:
    configure()
    PDF.mkdir(parents=True, exist_ok=True)
    figure_01()
    figure_07()
    figure_12()
    figure_13()
    figure_18()
    figure_21()
    print("refined 6 V13 paper-facing PDFs from frozen source CSVs")


if __name__ == "__main__":
    main()
