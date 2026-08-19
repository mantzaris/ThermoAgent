#!/usr/bin/env python3
"""Regenerate V12 figures that failed post-analysis visual QA.

This is a presentation-only repair.  It reads the compact, frozen aggregate
figure-source tables and does not recalculate any formal statistic.  Figure 9
uses the descriptively selected matched panel documented in its selection
table; that panel is not used for inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import networkx as nx
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
GREY = "#777777"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def box(axis: plt.Axes, x: float, y: float, text: str, color: str, width: float) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.065),
        width,
        0.13,
        boxstyle="round,pad=0.012",
        facecolor="white",
        edgecolor=color,
        linewidth=1.6,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9.2, zorder=4)


def figure_01(pdf: Path, source: Path) -> None:
    nodes = pd.read_csv(source / "figure_01_architecture_nodes.csv")
    edges = pd.read_csv(source / "figure_01_architecture_edges.csv")
    # Widen the two right-hand columns while preserving the recorded topology.
    positions = {
        "private observation": (0.11, 0.77, 0.19),
        "private memory": (0.11, 0.48, 0.19),
        "delivered inbox": (0.11, 0.19, 0.19),
        "LLM agent i\nbelief, action, c, m": (0.43, 0.48, 0.23),
        "typed action": (0.69, 0.72, 0.16),
        "model-chosen signal": (0.69, 0.27, 0.17),
        "delivery graph": (0.89, 0.27, 0.13),
        "local workload": (0.89, 0.72, 0.16),
    }
    colors = {"local": SKY, "agent": ORANGE, "output": GREEN, "network": PURPLE, "environment": RED}
    figure, axis = plt.subplots(figsize=(7.4, 3.6))
    for row in edges.itertuples(index=False):
        start = nodes.iloc[int(row.source)]
        end = nodes.iloc[int(row.target)]
        sx, sy, _ = positions[start.label]
        ex, ey, _ = positions[end.label]
        axis.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (ex, ey),
                arrowstyle="-|>",
                mutation_scale=11,
                color=GREY,
                linewidth=1.15,
                connectionstyle="arc3,rad=0.035",
                zorder=1,
            )
        )
    display_labels = {"model-chosen signal": "model-chosen\nsignal", "delivery graph": "delivery\ngraph"}
    for row in nodes.itertuples(index=False):
        x, y, width = positions[row.label]
        box(axis, x, y, display_labels.get(row.label, row.label), colors[row.kind], width)
    axis.text(
        0.50,
        0.94,
        "Scheduler offers one random-sequential update; it never selects the response",
        ha="center",
        fontsize=10,
    )
    axis.text(
        0.50,
        0.035,
        "Evaluator-only global state remains outside every agent context",
        ha="center",
        color=RED,
        fontsize=9.5,
    )
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    save(figure, pdf / "figure_01_agent_architecture.pdf")


def circular_positions(node_ids: list[int]) -> dict[int, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, len(node_ids), endpoint=False)
    return {node: np.asarray((np.cos(angle), np.sin(angle))) for node, angle in zip(node_ids, angles)}


def figure_02(pdf: Path, source: Path) -> None:
    data = pd.read_csv(source / "figure_02_network_construction.csv")
    node_ids = sorted(set(data["source"]) | set(data["target"]))
    positions = circular_positions(node_ids)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.35))
    for axis, condition, title in zip(
        axes,
        ("reciprocal", "nonreciprocal"),
        ("Reciprocal, $\\alpha=0$", "Directed, $\\alpha=0.8$"),
    ):
        subset = data[data["condition"] == condition]
        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        nx.draw_networkx_nodes(graph, positions, node_color=SKY, edgecolors=BLACK, node_size=360, ax=axis)
        nx.draw_networkx_labels(graph, positions, font_size=9, ax=axis)
        weight_map = {(int(r.source), int(r.target)): float(r.weight) for r in subset.itertuples(index=False)}
        for row in subset.itertuples(index=False):
            reverse = weight_map.get((int(row.target), int(row.source)), 0.0)
            axis.add_patch(
                FancyArrowPatch(
                    positions[int(row.source)],
                    positions[int(row.target)],
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.7 + 2.0 * float(row.weight),
                    color=PURPLE if float(row.weight) > reverse else GREY,
                    alpha=0.82,
                    connectionstyle="arc3,rad=0.11",
                )
            )
        axis.set_title(title)
        axis.set(xlim=(-1.30, 1.30), ylim=(-1.22, 1.22))
        axis.axis("off")
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.17, top=0.88, wspace=0.12)
    figure.text(
        0.5,
        0.055,
        "Same support, unit weighted in/out degree, and one opportunity per valid update",
        ha="center",
        fontsize=9.2,
    )
    save(figure, pdf / "figure_02_network_construction.pdf")


def figure_08(pdf: Path, source: Path) -> None:
    data = pd.read_csv(source / "figure_08_correlation_fluctuation_relaxation.csv")
    data = data[data["family"] == "collective_network"]
    metrics = (
        ("belief_correlation_distance_1", "Neighbor correlation", BLUE),
        ("belief_susceptibility", "Belief susceptibility\n$N\\,\\mathrm{Var}(m_b)$", ORANGE),
        ("belief_integrated_autocorrelation_time_updates", "Correlation time\n(attempted updates)", GREEN),
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), constrained_layout=True)
    for axis, (metric, label, color) in zip(axes, metrics):
        axis.scatter(data["alpha"], data[metric], color=color, alpha=0.58, s=22)
        axis.set_xlabel("Nonreciprocity $\\alpha$")
        axis.set_ylabel(label, labelpad=5)
        axis.set_xticks([0.0, 0.2, 0.5, 0.8])
    save(figure, pdf / "figure_08_correlations_fluctuations.pdf")


def figure_09(pdf: Path, source: Path) -> None:
    data = pd.read_csv(source / "figure_09_probability_currents.csv")
    selected_ids = [
        "collective_n16_modular_k0.80_t0.85_g0_a0.00_reciprocal",
        "collective_n16_modular_k0.80_t0.85_g0_a0.80_forward",
    ]
    titles = ("Reciprocal reference", "Directed, $\\alpha=0.8$")
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    states = [0, 1, 2, 3]
    positions = circular_positions(states)
    for axis, panel_id, title in zip(axes, selected_ids, titles):
        subset = data[data["panel_id"] == panel_id]
        for state in states:
            axis.scatter(*positions[state], s=180, color=SKY, edgecolor=BLACK, zorder=3)
            axis.text(*positions[state], str(state), ha="center", va="center", fontsize=9, zorder=4)
        resolved = subset[subset["current"].abs() >= 1e-6]
        if resolved.empty:
            axis.text(0, 0, "No resolved current\n$|J|<10^{-6}$", ha="center", va="center", fontsize=9.5)
        else:
            scale = float(resolved["current"].abs().max())
            for row in resolved.itertuples(index=False):
                source_state, target_state = (
                    (int(row.source_state), int(row.target_state))
                    if row.current >= 0
                    else (int(row.target_state), int(row.source_state))
                )
                axis.add_patch(
                    FancyArrowPatch(
                        positions[source_state],
                        positions[target_state],
                        arrowstyle="-|>",
                        mutation_scale=10,
                        linewidth=1.0 + 3.0 * abs(float(row.current)) / scale,
                        color=RED,
                        alpha=0.80,
                        connectionstyle="arc3,rad=0.15",
                    )
                )
            axis.text(
                0,
                0,
                "resolved $|J|=0.0101$",
                ha="center",
                va="center",
                fontsize=9.0,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 2.0},
                zorder=5,
            )
        axis.set_title(title + "\nprojected macrostate currents")
        axis.set(xlim=(-1.35, 1.35), ylim=(-1.25, 1.25))
        axis.axis("off")
    figure.text(
        0.5,
        0.015,
        "Descriptive matched panel selected for current visibility; not used for inference",
        ha="center",
        fontsize=8.7,
        color=GREY,
    )
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.84, wspace=0.18)
    save(figure, pdf / "figure_09_probability_currents.pdf")


def figure_12(pdf: Path, source: Path) -> None:
    """Render the estimator-shape comparison from its frozen source table."""

    data = pd.read_csv(source / "figure_12_reference_surrogate_llm.csv")
    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    for label, color, marker in (
        ("V10 heat-bath", BLACK, "^"),
        ("fitted surrogate", GREEN, "s"),
        ("V12 Qwen", PURPLE, "o"),
    ):
        line = data[data["source"] == label].sort_values("alpha")
        if line.empty:
            continue
        maximum = max(float(line["total_per_update_mean"].abs().max()), 1e-12)
        axis.plot(
            line["alpha"],
            line["total_per_update_mean"] / maximum,
            color=color,
            marker=marker,
            linewidth=1.7,
            label=label,
        )
    axis.set(
        xlabel="Nonreciprocity $\\alpha$",
        ylabel="Irreversibility normalized\nwithin estimator",
    )
    axis.set_title(
        "Normalized shapes; estimators are not numerically equivalent",
        loc="left",
        fontsize=9.0,
        pad=8,
    )
    axis.legend(frameon=False)
    save(figure, pdf / "figure_12_effective_model_comparison.pdf")


def figure_13(pdf: Path, source: Path) -> None:
    data = pd.read_csv(source / "figure_13_size_dependence.csv")
    figure, axes = plt.subplots(1, 3, figsize=(8.6, 3.25), constrained_layout=True)
    metrics = (
        ("order", "Mean |$m_b$|"),
        ("susceptibility", "Belief susceptibility\n$N\\,\\mathrm{Var}(m_b)$"),
        ("irreversibility", "Adjusted block KL\n(nats / transition)"),
    )
    for alpha, color, marker in ((0.0, BLUE, "o"), (0.8, ORANGE, "s")):
        line = data[np.isclose(data["alpha"], alpha)].sort_values("n_agents")
        for axis, (metric, _) in zip(axes, metrics):
            axis.plot(line["n_agents"], line[metric], color=color, marker=marker, linewidth=1.8, label="$\\alpha=%.1f$" % alpha)
    for axis, (_, label) in zip(axes, metrics):
        axis.set_xlabel("Agents $N$")
        axis.set_ylabel(label, labelpad=5)
        axis.set_xticks([8, 16])
    axes[-1].legend(frameon=False, loc="best")
    save(figure, pdf / "figure_13_size_dependence.pdf")


def figure_16(pdf: Path, source: Path) -> None:
    nodes = pd.read_csv(source / "figure_16_network_snapshot_nodes.csv")
    edges = pd.read_csv(source / "figure_16_network_snapshot_edges.csv")
    node_ids = [int(value) for value in nodes["agent"]]
    positions = circular_positions(node_ids)
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    beliefs = {int(r.agent): int(r.belief) for r in nodes.itertuples(index=False)}
    actions = {int(r.agent): int(r.action) for r in nodes.itertuples(index=False)}
    figure, axis = plt.subplots(figsize=(5.5, 4.55))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[ORANGE if beliefs[node] == 1 else BLUE for node in node_ids],
        edgecolors=[RED if beliefs[node] != actions[node] else BLACK for node in node_ids],
        linewidths=2,
        node_size=450,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=9, ax=axis)
    for row in edges.itertuples(index=False):
        axis.add_patch(
            FancyArrowPatch(
                positions[int(row.source)],
                positions[int(row.target)],
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.55 + 2.0 * float(row.weight),
                color=GREY,
                alpha=0.68,
                connectionstyle="arc3,rad=0.11",
            )
        )
    handles = [
        Patch(facecolor=BLUE, edgecolor=BLACK, label="belief $-1$"),
        Patch(facecolor=ORANGE, edgecolor=BLACK, label="belief $+1$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=RED, markeredgewidth=2, label="belief-action mismatch"),
    ]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8.4)
    axis.set_title("Directed LLM-agent network at sweep 8.0")
    axis.set(xlim=(-1.28, 1.28), ylim=(-1.20, 1.22))
    axis.axis("off")
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.18, top=0.88)
    save(figure, pdf / "figure_16_network_snapshot.pdf")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "results/llm_agent_statmech_v12/figures"
    pdf = root / "pdf"
    source = root / "source_data"
    configure()
    figure_01(pdf, source)
    figure_02(pdf, source)
    figure_08(pdf, source)
    figure_09(pdf, source)
    figure_12(pdf, source)
    figure_13(pdf, source)
    figure_16(pdf, source)
    print("Regenerated seven V12 PDFs from compact aggregate source data.")


if __name__ == "__main__":
    main()
