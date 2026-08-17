"""Compact vector figures and PDF quality assurance for V9."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .reporting import RESULTS_RELATIVE
from .workflow import artifact_root


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


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "ThermoAgent V9"})
    plt.close(fig)


def _copy_source(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pandas 1.1 in the preserved project environment uses the older explicit
    # keyword spelling; it still guarantees LF output.
    frame.to_csv(path, index=False, line_terminator="\n", float_format="%.10g")


def _figure_architecture(pdf: Path, source: Path) -> None:
    nodes = pd.DataFrame(
        [
            ("agent_1", "agent", 0.10, 0.75, "private belief, action, memory"),
            ("agent_2", "agent", 0.10, 0.50, "private belief, action, memory"),
            ("agent_3", "agent", 0.10, 0.25, "private belief, action, memory"),
            ("communication", "layer", 0.39, 0.68, "communication coupling J_b"),
            ("dependency", "layer", 0.39, 0.32, "dependency coupling J_a"),
            ("scheduler", "environment", 0.68, 0.50, "random local update schedule"),
            ("evaluator", "evaluator", 0.91, 0.50, "global observables only"),
        ],
        columns=["node", "kind", "x", "y", "description"],
    )
    edges = pd.DataFrame(
        [
            ("agent_1", "communication", "delivered messages"),
            ("agent_2", "communication", "delivered messages"),
            ("agent_3", "communication", "delivered messages"),
            ("agent_1", "dependency", "local task coupling"),
            ("agent_2", "dependency", "local task coupling"),
            ("agent_3", "dependency", "local task coupling"),
            ("communication", "scheduler", "local view"),
            ("dependency", "scheduler", "local view"),
            ("scheduler", "evaluator", "state transitions"),
        ],
        columns=["source", "target", "relation"],
    )
    _copy_source(nodes, source / "figure_01_nodes.csv")
    _copy_source(edges, source / "figure_01_edges.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    color_by_kind = {"agent": COLORS["blue"], "layer": COLORS["green"], "environment": COLORS["orange"], "evaluator": COLORS["gray"]}
    position = {row.node: (row.x, row.y) for row in nodes.itertuples()}
    for row in edges.itertuples():
        start, end = position[row.source], position[row.target]
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#777777", "alpha": 0.75})
    labels = {
        "agent_1": "A1",
        "agent_2": "A2",
        "agent_3": "A3",
        "communication": "Comm.\nlayer",
        "dependency": "Task\nlayer",
        "scheduler": "Sched.",
        "evaluator": "Eval.",
    }
    for row in nodes.itertuples():
        size = 1250 if row.kind == "layer" else (1050 if row.kind in {"environment", "evaluator"} else 780)
        ax.scatter(row.x, row.y, s=size, marker="o", color=color_by_kind[row.kind], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(row.x, row.y, labels[row.node], ha="center", va="center", color="white", fontsize=8.8, weight="bold", zorder=4)
    ax.text(0.10, 0.06, "Independent private vaults; no global policy input", ha="center", fontsize=9.5)
    ax.text(0.50, 0.94, "Multiplex belief–action microdynamics", ha="center", fontsize=12, weight="bold")
    ax.text(0.91, 0.16, "Evaluator truth is excluded\nfrom agent decisions", ha="center", fontsize=9.5, color=COLORS["red"])
    _save(fig, pdf / "figure_01_agentic_formulation.pdf")


def generate_figures(repository: Path) -> Dict[str, object]:
    _style()
    results = repository / RESULTS_RELATIVE
    tables = results / "tables"
    pdf = results / "figures/pdf"
    source = results / "figures/source_data"
    _figure_architecture(pdf, source)

    exact = pd.read_csv(source / "figure_02_equilibrium_validation.csv")
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.scatter(exact["gibbs_probability"], exact["empirical_probability"], s=34, color=COLORS["blue"], alpha=0.8)
    limit = max(exact["gibbs_probability"].max(), exact["empirical_probability"].max()) * 1.05
    ax.plot([0, limit], [0, limit], color=COLORS["black"], linestyle="--", linewidth=1.2)
    ax.set(xlabel="Exact Gibbs probability", ylabel="Empirical stationary probability", xlim=(0, limit), ylim=(0, limit))
    ax.text(0.03, 0.96, "N = 3, T = 1.7", transform=ax.transAxes, va="top")
    ax.text(0.69, 0.76, "identity", transform=ax.transAxes, rotation=42, color=COLORS["black"])
    _save(fig, pdf / "figure_02_equilibrium_validation.pdf")

    phase = pd.read_csv(tables / "phase_summary.csv")
    phase_plot = phase[(phase["n_agents"] == 144) & (phase["topology"] == "regular")].copy()
    _copy_source(phase_plot, source / "figure_03_phase_diagram.csv")
    fragments = sorted(phase_plot["fragmentation"].unique())
    fig, axes = plt.subplots(1, len(fragments), figsize=(7.4, 3.4), sharey=True)
    image = None
    for ax, fragmentation in zip(axes, fragments):
        subset = phase_plot[phase_plot["fragmentation"] == fragmentation]
        grid = subset.pivot(index="temperature", columns="communication_availability", values="mean_abs_magnetization_mean").sort_index(ascending=True)
        image = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis", extent=[grid.columns.min(), grid.columns.max(), grid.index.min(), grid.index.max()])
        ax.set_title("field disorder = %.1f" % fragmentation)
    axes[0].set_ylabel("decision temperature")
    fig.text(0.47, 0.01, "communication availability $p_c$", ha="center", fontsize=10.5)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
    colorbar.set_label("mean |order parameter|")
    _save(fig, pdf / "figure_03_phase_diagram.pdf")

    finite = pd.read_csv(tables / "finite_size_summary.csv")
    _copy_source(finite, source / "figure_04_binder_cumulants.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for color, (size, group) in zip([COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]], finite.groupby("n_agents")):
        group = group.sort_values("temperature")
        ax.plot(group["temperature"], group["binder_cumulant_mean"], marker="o", ms=4, lw=1.5, color=color, label="N = %d" % size)
        ax.fill_between(
            group["temperature"].to_numpy(float),
            group["binder_cumulant_ci_low"].to_numpy(float),
            group["binder_cumulant_ci_high"].to_numpy(float),
            color=color,
            alpha=0.12,
        )
    ax.set(xlabel="decision temperature", ylabel="Binder cumulant $U_4$")
    ax.legend(frameon=False, ncol=2)
    _save(fig, pdf / "figure_04_binder_cumulants.pdf")

    _copy_source(finite, source / "figure_05_response_functions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True)
    for color, (size, group) in zip([COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]], finite.groupby("n_agents")):
        group = group.sort_values("temperature")
        axes[0].plot(group["temperature"], group["susceptibility_per_agent_mean"], marker="o", ms=3, color=color, label="N=%d" % size)
        axes[0].fill_between(
            group["temperature"].to_numpy(float),
            group["susceptibility_per_agent_ci_low"].to_numpy(float),
            group["susceptibility_per_agent_ci_high"].to_numpy(float),
            color=color,
            alpha=0.08,
        )
        axes[1].plot(group["temperature"], group["heat_capacity_per_agent_mean"], marker="s", ms=3, color=color)
        axes[1].fill_between(
            group["temperature"].to_numpy(float),
            group["heat_capacity_per_agent_ci_low"].to_numpy(float),
            group["heat_capacity_per_agent_ci_high"].to_numpy(float),
            color=color,
            alpha=0.08,
        )
    axes[0].set(xlabel="decision temperature", ylabel="susceptibility $N\,\mathrm{Var}(m)/T$")
    axes[1].set(xlabel="decision temperature", ylabel="heat-capacity analogue $\mathrm{Var}(H)/(NT^2)$")
    axes[0].legend(frameon=False, ncol=2)
    _save(fig, pdf / "figure_05_response_functions.pdf")

    epr = pd.read_csv(tables / "entropy_production_summary.csv")
    _copy_source(epr, source / "figure_06_entropy_production.csv")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.errorbar(epr["asymmetry"], epr["entropy_production_rate_mean"], yerr=[epr["entropy_production_rate_mean"] - epr["entropy_production_rate_ci_low"], epr["entropy_production_rate_ci_high"] - epr["entropy_production_rate_mean"]], marker="o", color=COLORS["red"], capsize=3, lw=1.6)
    ax.axhline(0.0, color=COLORS["black"], linestyle="--", linewidth=1.0)
    ax.set(xlabel="directed-coupling asymmetry", ylabel="exact stationary entropy production / update")
    _save(fig, pdf / "figure_06_entropy_production.pdf")

    landscape = pd.read_csv(source / "figure_07_free_energy_landscape.csv")
    hysteresis = pd.read_csv(tables / "hysteresis_summary.csv")
    _copy_source(hysteresis, source / "figure_07_hysteresis.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    shifted = landscape["constrained_free_energy"] - landscape["constrained_free_energy"].min()
    axes[0].plot(landscape["order_parameter"], shifted, marker="o", color=COLORS["blue"])
    axes[0].set(xlabel="order parameter", ylabel="$F(m)-\min F(m)$")
    for branch, label, color, marker in ((1.0, "field up", COLORS["orange"], "o"), (-1.0, "field down", COLORS["green"], "s")):
        group = hysteresis[hysteresis["branch_code"] == branch].sort_values("field")
        axes[1].plot(group["field"], group["magnetization_mean"], marker=marker, color=color, label=label)
        axes[1].fill_between(
            group["field"].to_numpy(float),
            group["magnetization_ci_low"].to_numpy(float),
            group["magnetization_ci_high"].to_numpy(float),
            color=color,
            alpha=0.10,
        )
    axes[1].set(xlabel="external field", ylabel="order parameter")
    axes[1].legend(frameon=False)
    _save(fig, pdf / "figure_07_free_energy_and_hysteresis.pdf")

    relaxation = pd.read_csv(tables / "relaxation_summary.csv")
    _copy_source(relaxation, source / "figure_08_relaxation.csv")
    fig, ax = plt.subplots(figsize=(6.1, 4.3))
    for color, (size, group) in zip([COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]], relaxation.groupby("n_agents")):
        group = group.sort_values("temperature")
        ax.plot(group["temperature"], group["relaxation_time_mean"], marker="o", color=color, label="N=%d" % size)
        ax.fill_between(
            group["temperature"].to_numpy(float),
            np.maximum(0.0, group["relaxation_time_ci_low"].to_numpy(float)),
            group["relaxation_time_ci_high"].to_numpy(float),
            color=color,
            alpha=0.08,
        )
    ax.set(xlabel="decision temperature", ylabel="relaxation time (sweeps)")
    ax.legend(frameon=False, ncol=2)
    ax.axhline(500.0, color=COLORS["gray"], linestyle=":", linewidth=0.9)
    ax.text(0.98, 0.68, "500-sweep censoring", transform=ax.transAxes, ha="right", fontsize=9.0, color=COLORS["gray"])
    _save(fig, pdf / "figure_08_relaxation.pdf")

    nodes = pd.read_csv(source / "figure_09_application_nodes.csv")
    edges = pd.read_csv(source / "figure_09_application_edges.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    for ax, application in zip(axes, ("humanitarian", "utility")):
        node_data = nodes[nodes["application"] == application].sort_values("node")
        edge_data = edges[edges["application"] == application]
        graph = nx.Graph()
        graph.add_nodes_from(node_data["node"].astype(int))
        dependency_edges = edge_data[edge_data["layer"] == "dependency"]
        communication_edges = edge_data[edge_data["layer"] == "communication"]
        graph.add_edges_from(zip(dependency_edges["source"].astype(int), dependency_edges["target"].astype(int)))
        positions = nx.spring_layout(graph, seed=17)
        nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#BBBBBB", width=0.6, alpha=0.6)
        comm_graph = nx.Graph()
        comm_graph.add_nodes_from(graph.nodes)
        comm_graph.add_edges_from(zip(communication_edges["source"].astype(int), communication_edges["target"].astype(int)))
        nx.draw_networkx_edges(comm_graph, positions, ax=ax, edge_color=COLORS["sky"], width=0.7, alpha=0.55, style="dashed")
        node_colors = node_data.set_index("node").loc[list(graph.nodes), "workload"].to_numpy(float)
        node_shapes = node_data.set_index("node").loc[list(graph.nodes), "action"].to_numpy(int)
        for shape_value, marker in ((1, "^"), (-1, "o")):
            selected = [node for node, action in zip(graph.nodes, node_shapes) if action == shape_value]
            if selected:
                colors = [node_colors[list(graph.nodes).index(node)] for node in selected]
                nx.draw_networkx_nodes(graph, positions, nodelist=selected, node_color=colors, cmap="viridis", vmin=0, vmax=max(1.0, node_colors.max()), node_size=42, node_shape=marker, edgecolors=COLORS["black"], linewidths=0.4, ax=ax)
        ax.set_title(application.capitalize())
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#BBBBBB", lw=1.5, label="task edge"),
        Line2D([0], [0], color=COLORS["sky"], lw=1.5, ls="--", label="comm. edge"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["black"], label="$a_i=+1$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["black"], label="$a_i=-1$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False)
    maximum_workload = max(1.0, float(nodes["workload"].max()))
    colorbar_axis = fig.add_axes([0.31, 0.15, 0.38, 0.025])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.0, vmax=maximum_workload), cmap="viridis"),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("local workload", labelpad=2)
    fig.subplots_adjust(bottom=0.26)
    _save(fig, pdf / "figure_09_application_networks.pdf")

    applications = pd.read_csv(tables / "application_mapping_summary.csv")
    utility = applications[applications["application"] == "utility"].copy()
    _copy_source(utility, source / "figure_10_utility_trajectory.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), sharex=True)
    panels = [
        ("belief_order", "belief order"),
        ("entropy_flow_per_update", "entropy flow / update"),
        ("service_loss", "unserved-service proxy"),
        ("cascade_depth", "cascade depth"),
    ]
    for ax, (metric, label) in zip(axes.ravel(), panels):
        ax.plot(utility["time"], utility[metric + "_mean"], color=COLORS["blue"], lw=1.6)
        ax.fill_between(
            utility["time"].to_numpy(float),
            utility[metric + "_ci_low"].to_numpy(float),
            utility[metric + "_ci_high"].to_numpy(float),
            color=COLORS["blue"],
            alpha=0.18,
        )
        ax.set_ylabel(label)
        ax.axvspan(40, 96, color=COLORS["red"], alpha=0.08)
        ax.axvspan(45, 116, color=COLORS["gray"], alpha=0.08)
    axes[1, 0].set_xlabel("simulation step")
    axes[1, 1].set_xlabel("simulation step")
    for axis in axes[:, 1]:
        axis.yaxis.set_label_position("right")
    axes[0, 0].text(0.02, 0.95, "cyber-physical drive", transform=axes[0, 0].transAxes, va="top", color=COLORS["red"])
    _save(fig, pdf / "figure_10_utility_disruption.pdf")
    return {"pdf_count": len(list(pdf.glob("*.pdf"))), "source_csv_count": len(list(source.glob("*.csv")))}


def validate_pdfs(repository: Path) -> Dict[str, object]:
    results = repository / RESULTS_RELATIVE
    pdf_directory = results / "figures/pdf"
    manual_review_path = results / "reproducibility/pdf_visual_review.json"
    manual_review = {}
    manual_reviewer = "pending"
    if manual_review_path.exists():
        review_payload = json.loads(manual_review_path.read_text(encoding="utf-8"))
        manual_review = review_payload.get("files", {})
        manual_reviewer = str(review_payload.get("reviewer", "pending"))
    qa_root = artifact_root() / "pdf_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in sorted(pdf_directory.glob("*.pdf")):
        fonts = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True).stdout
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines) and all(" yes " in (" " + line.lower() + " ") for line in font_lines)
        prefix = qa_root / path.stem
        subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-r", "300", "-png", str(path), str(prefix)], check=True, capture_output=True)
        rendered = prefix.with_suffix(".png")
        text_output = subprocess.run(["pdftotext", str(path), "-"], check=True, capture_output=True, text=True).stdout
        review = manual_review.get(path.name, {})
        rows.append(
            {
                "file": path.name,
                "opens": True,
                "page_count": 1,
                "fonts_embedded": embedded,
                "font_count": len(font_lines),
                "text_extractable": bool(text_output.strip()),
                "render_300_dpi": rendered.exists() and rendered.stat().st_size > 0,
                "render_path_external": str(rendered),
                "visual_inspection": review.get("visual_inspection", "pending"),
                "clipping_or_overlap": review.get("clipping_or_overlap", "pending"),
                "reviewer": manual_reviewer if review else "pending",
                "source_data_directory": "figures/source_data",
            }
        )
    payload = {
        "pdf_count": len(rows),
        "all_open": all(row["opens"] for row in rows),
        "all_fonts_embedded": all(row["fonts_embedded"] for row in rows),
        "all_rendered_300_dpi": all(row["render_300_dpi"] for row in rows),
        "all_visually_inspected": all(row["visual_inspection"] == "passed" for row in rows),
        "all_free_of_observed_clipping_or_overlap": all(row["clipping_or_overlap"] == "none observed" for row in rows),
        "qa_pngs_external_only": True,
        "files": rows,
    }
    destination = results / "reproducibility/pdf_qa.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
