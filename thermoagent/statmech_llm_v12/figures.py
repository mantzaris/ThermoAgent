"""Data-derived vector figures for the V12 JSTAT package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

from .experiment import _graph_for_panel, formal_panel_design
from .workflow import artifact_root, load_yaml


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
GREY = "#777777"


def _configure() -> None:
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


def _paths(repository: Path) -> Tuple[Path, Path]:
    results = Path(repository) / "results/llm_agent_statmech_v12"
    pdf = results / "figures/pdf"
    source = results / "figures/source_data"
    pdf.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    return pdf, source


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _box(axis: plt.Axes, x: float, y: float, text: str, color: str, width: float = 0.18) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.07), width, 0.14, boxstyle="round,pad=0.015", facecolor=color, alpha=0.16,
        edgecolor=color, linewidth=1.3
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9.2)


def _figure_01(repository: Path, pdf: Path, source: Path) -> None:
    nodes = pd.DataFrame(
        [
            ("private observation", 0.10, 0.77, "local"),
            ("private memory", 0.10, 0.48, "local"),
            ("delivered inbox", 0.10, 0.19, "local"),
            ("LLM agent i\nbelief, action, c, m", 0.43, 0.48, "agent"),
            ("typed action", 0.72, 0.70, "output"),
            ("model-chosen signal", 0.72, 0.30, "output"),
            ("delivery graph", 0.91, 0.30, "network"),
            ("local workload", 0.91, 0.70, "environment"),
        ],
        columns=["label", "x", "y", "kind"],
    )
    edges = pd.DataFrame(
        [(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (4, 7), (5, 6), (6, 2)], columns=["source", "target"]
    )
    _csv(nodes.assign(record_type="node"), source / "figure_01_architecture_nodes.csv")
    _csv(edges, source / "figure_01_architecture_edges.csv")
    figure, axis = plt.subplots(figsize=(7.0, 3.5))
    colors = {"local": SKY, "agent": ORANGE, "output": GREEN, "network": PURPLE, "environment": RED}
    for row in nodes.itertuples():
        _box(axis, row.x, row.y, row.label, colors[row.kind], 0.20 if row.kind == "agent" else 0.17)
    for row in edges.itertuples():
        start = nodes.iloc[row.source]
        end = nodes.iloc[row.target]
        axis.add_patch(
            FancyArrowPatch((start.x, start.y), (end.x, end.y), arrowstyle="-|>", mutation_scale=11, color=GREY,
                            linewidth=1.1, connectionstyle="arc3,rad=0.04")
        )
    axis.text(0.43, 0.93, "Scheduler offers one random-sequential update; it never selects the response", ha="center", fontsize=10)
    axis.text(0.43, 0.04, "Evaluator-only global state remains outside every agent context", ha="center", color=RED, fontsize=9.5)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    _save(figure, pdf / "figure_01_agent_architecture.pdf")


def _figure_02(repository: Path, pdf: Path, source: Path) -> None:
    protocol = load_yaml(repository / "configs/statmech_v12/protocol_frozen.yaml")
    definitions = formal_panel_design(protocol)
    selected = next(row for row in definitions if row["family"] == "collective_network" and row["n_agents"] == 8 and row["topology"] == "ring" and np.isclose(row["alpha"], 0.8) and row["orientation"] == "forward")
    graph = _graph_for_panel(selected)
    edge_rows = []
    for condition, weights in (("reciprocal", graph.symmetric), ("nonreciprocal", graph.weights)):
        for i, j in zip(*np.nonzero(weights)):
            edge_rows.append({"condition": condition, "source": i, "target": j, "weight": weights[i, j]})
    _csv(pd.DataFrame(edge_rows), source / "figure_02_network_construction.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    positions = nx.circular_layout(nx.from_numpy_array(graph.adjacency))
    for axis, title, weights in zip(axes, ("Reciprocal, $\\alpha=0$", "Directed, $\\alpha=0.8$"), (graph.symmetric, graph.weights)):
        nx.draw_networkx_nodes(nx.from_numpy_array(graph.adjacency), positions, node_color=SKY, edgecolors=BLACK, node_size=380, ax=axis)
        nx.draw_networkx_labels(nx.from_numpy_array(graph.adjacency), positions, font_size=9, ax=axis)
        for i, j in zip(*np.nonzero(weights)):
            arrow = FancyArrowPatch(positions[i], positions[j], arrowstyle="-|>", mutation_scale=8,
                                    linewidth=0.6 + 2.2 * weights[i, j], color=PURPLE if weights[i, j] > weights[j, i] else GREY,
                                    alpha=0.8, connectionstyle="arc3,rad=0.10")
            axis.add_patch(arrow)
        axis.set_title(title)
        axis.axis("off")
    figure.text(0.5, 0.02, "Same support, unit weighted in/out degree, and one opportunity per valid update", ha="center")
    _save(figure, pdf / "figure_02_network_construction.pdf")


def _figure_03(repository: Path, pdf: Path, source: Path) -> None:
    frame = pd.read_csv(artifact_root() / "formal/microscopic_response.csv")
    valid = frame[frame["valid_after_repair"] == 1]
    grouped = valid.groupby(["private_field", "neighbor_field", "coupling_strength"], as_index=False).agg(
        probability_plus=("belief_after", lambda values: np.mean(np.asarray(values) == 1)),
        n=("belief_after", "size"),
    )
    _csv(grouped, source / "figure_03_individual_response_surface.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    markers = {-1: "o", 0: "s", 1: "^"}
    colors = {-1: BLUE, 0: GREY, 1: ORANGE}
    for axis, coupling in zip(axes, sorted(grouped["coupling_strength"].unique())):
        subset = grouped[np.isclose(grouped["coupling_strength"], coupling)]
        for private in (-1, 0, 1):
            line = subset[subset["private_field"] == private].sort_values("neighbor_field")
            axis.plot(line["neighbor_field"], line["probability_plus"], marker=markers[private], color=colors[private],
                      label="private field %+d" % private, linewidth=1.8)
        axis.axhline(0.5, color="#BBBBBB", linestyle="--", linewidth=1)
        axis.set_title("Neighbor relevance %.2f" % coupling)
        axis.set_xlabel("Delivered neighbor field")
        axis.set_xticks([-1, 0, 1])
    axes[0].set_ylabel("Pr(next belief = +1)")
    axes[1].legend(frameon=False, loc="best")
    _save(figure, pdf / "figure_03_individual_response.pdf")


def _figure_04(repository: Path, pdf: Path, source: Path) -> None:
    frame = pd.read_csv(artifact_root() / "formal/microscopic_response.csv")
    valid = frame[frame["valid_after_repair"] == 1].copy()
    valid["belief_switched"] = valid["belief_after"] != valid["current_belief"]
    valid["action_switched"] = valid["action_after"] != valid["current_action"]
    grouped = valid.groupby(["regime", "current_belief", "neighbor_field"], as_index=False).agg(
        belief_switch_rate=("belief_switched", "mean"),
        action_switch_rate=("action_switched", "mean"),
        n=("belief_switched", "size"),
    )
    _csv(grouped, source / "figure_04_transition_persistence.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for axis, metric, title in zip(axes, ("belief_switch_rate", "action_switch_rate"), ("Belief transitions", "Action transitions")):
        for regime, color, marker in (("markovized", BLUE, "o"), ("persistent_memory", ORANGE, "s")):
            line = grouped[grouped["regime"] == regime].groupby("neighbor_field", as_index=False)[metric].mean()
            axis.plot(line["neighbor_field"], line[metric], color=color, marker=marker, linewidth=1.8, label=regime.replace("_", " "))
        axis.set_title(title)
        axis.set_xlabel("Neighbor field")
        axis.set_xticks([-1, 0, 1])
    axes[0].set_ylabel("Switch probability")
    axes[1].legend(frameon=False)
    _save(figure, pdf / "figure_04_transition_persistence.pdf")


def _representative_files() -> Tuple[Path, Path]:
    root = artifact_root() / "formal/panels"
    reciprocal = root / "collective_n8_ring_k0.80_t0.85_g0_a0.00_reciprocal.csv"
    directed = root / "collective_n8_ring_k0.80_t0.85_g0_a0.80_forward.csv"
    if not reciprocal.exists() or not directed.exists():
        raise FileNotFoundError("predeclared representative panels are unavailable")
    return reciprocal, directed


def _trajectory_source(source: Path) -> pd.DataFrame:
    reciprocal, directed = _representative_files()
    rows = []
    for condition, path in (("reciprocal", reciprocal), ("directed", directed)):
        frame = pd.read_csv(path)
        frame = frame.assign(condition=condition)
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    _csv(output[["condition", "update", "sweep", "beliefs", "actions", "belief_magnetization", "action_magnetization", "belief_disagreement", "reference_energy_per_agent", "messages_delivered"]], source)
    return output


def _figure_05(repository: Path, pdf: Path, source: Path) -> None:
    frame = _trajectory_source(source / "figure_05_representative_trajectories.csv")
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.7), sharex=True)
    for axis, condition in zip(axes, ("reciprocal", "directed")):
        subset = frame[frame["condition"] == condition]
        matrix = np.vstack([_bits for _bits in (np.asarray([int(v) for v in text.split(";")]) for text in subset["beliefs"])])
        for agent in range(matrix.shape[1]):
            axis.step(subset["sweep"], matrix[:, agent] + 0.05 * agent, where="post", linewidth=1.1, label="agent %d" % agent if condition == "reciprocal" else None)
        axis.set_ylabel("Belief spin")
        axis.set_title(condition.capitalize() + " communication")
        axis.set_yticks([-1, 1])
    axes[-1].set_xlabel("Sweeps")
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    _save(figure, pdf / "figure_05_network_trajectories.pdf")


def _figure_06(repository: Path, pdf: Path, source: Path) -> None:
    frame = _trajectory_source(source / "figure_06_collective_time_series.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    metrics = [
        ("belief_magnetization", "Belief magnetization"),
        ("action_magnetization", "Action magnetization"),
        ("belief_disagreement", "Disagreement density"),
        ("reference_energy_per_agent", "Reference energy / agent"),
    ]
    for axis, (metric, label) in zip(axes.flat, metrics):
        for condition, color, style in (("reciprocal", BLUE, "-"), ("directed", ORANGE, "--")):
            line = frame[frame["condition"] == condition]
            axis.plot(line["sweep"], line[metric], color=color, linestyle=style, linewidth=1.6, label=condition)
        axis.set_ylabel(label)
    axes[1, 0].set_xlabel("Sweeps")
    axes[1, 1].set_xlabel("Sweeps")
    axes[0, 1].legend(frameon=False)
    _save(figure, pdf / "figure_06_collective_time_series.pdf")


def _panel_table() -> pd.DataFrame:
    return pd.read_csv(artifact_root() / "analysis/panel_statistics.csv")


def _figure_07(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = panel[panel["family"] == "collective_network"].groupby(
        ["n_agents", "topology", "coupling_strength", "sampling_temperature", "alpha"], as_index=False
    ).agg(order=("mean_abs_belief_magnetization", "mean"), disagreement=("mean_belief_disagreement", "mean"), n=("panel_id", "size"))
    _csv(data, source / "figure_07_collective_regimes.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for topology, color, marker in (("ring", BLUE, "o"), ("modular", ORANGE, "s")):
        subset = data[data["topology"] == topology]
        axes[0].scatter(subset["disagreement"], subset["order"], c=subset["alpha"], cmap="viridis", marker=marker, s=42, label=topology)
    for temperature, color, style in ((0.50, BLUE, "-"), (0.85, ORANGE, "--")):
        for alpha, marker in ((0.0, "o"), (0.8, "s")):
            line = data[np.isclose(data["sampling_temperature"], temperature) & np.isclose(data["alpha"], alpha)].groupby(
                "coupling_strength", as_index=False
            )["order"].mean()
            axes[1].plot(
                line["coupling_strength"], line["order"], color=color, linestyle=style,
                marker=marker, linewidth=1.7, label="noise %.2f, $\\alpha=%.1f$" % (temperature, alpha)
            )
    axes[0].set(xlabel="Disagreement density", ylabel="Mean |belief magnetization|", title="Collective finite-size regimes")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Neighbor relevance", ylabel="Mean |belief magnetization|", title="Coupling--noise response")
    axes[1].legend(frameon=False, fontsize=7.8)
    _save(figure, pdf / "figure_07_collective_regimes.pdf")


def _figure_08(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = panel[panel["family"].isin(["collective_network", "relaxation"])].copy()
    _csv(data[["family", "n_agents", "topology", "alpha", "sampling_temperature", "belief_correlation_distance_1", "belief_correlation_distance_2", "belief_susceptibility", "belief_integrated_autocorrelation_time_updates", "initial_condition"]], source / "figure_08_correlation_fluctuation_relaxation.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    collective = data[data["family"] == "collective_network"]
    axes[0].scatter(collective["alpha"], collective["belief_correlation_distance_1"], color=BLUE, alpha=0.6, s=22)
    axes[1].scatter(collective["alpha"], collective["belief_susceptibility"], color=ORANGE, alpha=0.6, s=22)
    axes[2].scatter(collective["alpha"], collective["belief_integrated_autocorrelation_time_updates"], color=GREEN, alpha=0.6, s=22)
    for axis, label in zip(axes, ("Nearest-neighbor correlation", "$N\\,\\mathrm{Var}(m_b)$", "Integrated correlation time\n(attempted updates)")):
        axis.set_xlabel("Nonreciprocity $\\alpha$")
        axis.set_ylabel(label)
    _save(figure, pdf / "figure_08_correlations_fluctuations.pdf")


def _figure_09(repository: Path, pdf: Path, source: Path) -> None:
    currents = pd.read_csv(artifact_root() / "analysis/probability_currents.csv")
    selected_ids = ["small_n3_g0_a0.00_reciprocal", "small_n3_g0_a0.80_forward"]
    data = currents[currents["panel_id"].isin(selected_ids)].copy()
    _csv(data, source / "figure_09_probability_currents.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for axis, panel_id, title in zip(axes, selected_ids, ("Reciprocal", "Directed, $\\alpha=0.8$")):
        subset = data[data["panel_id"] == panel_id]
        states = sorted(set(subset["source_state"]) | set(subset["target_state"]))
        positions = {state: (np.cos(2 * np.pi * i / max(len(states), 1)), np.sin(2 * np.pi * i / max(len(states), 1))) for i, state in enumerate(states)}
        for state, position in positions.items():
            axis.scatter(*position, s=120, color=SKY, edgecolor=BLACK, zorder=3)
            axis.text(*position, str(state), ha="center", va="center", fontsize=7)
        scale = max(subset["current"].abs().max(), 1e-12)
        for row in subset.itertuples():
            source_state, target_state = (row.source_state, row.target_state) if row.current >= 0 else (row.target_state, row.source_state)
            axis.add_patch(FancyArrowPatch(positions[source_state], positions[target_state], arrowstyle="-|>", mutation_scale=9,
                                           linewidth=0.7 + 3 * abs(row.current) / scale, color=RED, alpha=0.7,
                                           connectionstyle="arc3,rad=0.13"))
        axis.set_title(title + "\nstrongest projected currents")
        axis.axis("off")
    _save(figure, pdf / "figure_09_probability_currents.pdf")


def _mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, mean, mean
    se = float(np.std(array, ddof=1) / np.sqrt(array.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _effect_curve(panel: pd.DataFrame, metric: str) -> pd.DataFrame:
    records = []
    subset = panel[panel["family"] == "collective_network"]
    for (n_agents, alpha), group in subset.groupby(["n_agents", "alpha"]):
        cluster = group.groupby("cluster_id")[metric].mean().to_numpy(float)
        rng = np.random.default_rng(12012000 + int(n_agents) * 100 + int(round(float(alpha) * 100)))
        boot = np.asarray([np.mean(cluster[rng.integers(0, len(cluster), len(cluster))]) for _ in range(10000)])
        records.append(
            {
                "n_agents": n_agents,
                "alpha": alpha,
                "mean": float(np.mean(cluster)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "independent_panels": len(cluster),
                "cluster_bootstrap_replicates": 10000,
            }
        )
    return pd.DataFrame(records)


def _figure_10(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = _effect_curve(panel, "markov_epr_nats_per_update")
    _csv(data, source / "figure_10_entropy_production.csv")
    figure, axis = plt.subplots(figsize=(5.1, 3.4))
    for n_agents, color, marker in ((8, BLUE, "o"), (16, ORANGE, "s")):
        line = data[data["n_agents"] == n_agents].sort_values("alpha")
        axis.errorbar(line["alpha"], line["mean"], yerr=[line["mean"] - line["ci_low"], line["ci_high"] - line["mean"]], color=color, marker=marker, capsize=3, label="$N=%d$" % n_agents)
    axis.set(xlabel="Nonreciprocity $\\alpha$", ylabel="Projected Markov EPR\n(nats / attempted update)")
    axis.legend(frameon=False)
    _save(figure, pdf / "figure_10_entropy_production.pdf")


def _figure_11(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = _effect_curve(panel, "adjusted_block_kl_nats_per_update")
    _csv(data, source / "figure_11_path_irreversibility.csv")
    figure, axis = plt.subplots(figsize=(5.1, 3.4))
    for n_agents, color, marker, style in ((8, BLUE, "o", "-"), (16, ORANGE, "s", "--")):
        line = data[data["n_agents"] == n_agents].sort_values("alpha")
        axis.errorbar(line["alpha"], line["mean"], yerr=[line["mean"] - line["ci_low"], line["ci_high"] - line["mean"]], color=color, marker=marker, linestyle=style, capsize=3, label="$N=%d$" % n_agents)
    axis.axhline(0, color=GREY, linewidth=1)
    axis.set(xlabel="Nonreciprocity $\\alpha$", ylabel="Bias-adjusted block KL\n(nats / transition)")
    axis.legend(frameon=False)
    _save(figure, pdf / "figure_11_path_irreversibility.pdf")


def _figure_12(repository: Path, pdf: Path, source: Path) -> None:
    v10 = pd.read_csv(repository / "results/llm_agent_entropy_v10/tables/quadratic_onset.csv")
    panel = _panel_table()
    llm = panel[(panel["family"] == "collective_network")].groupby("alpha", as_index=False)["adjusted_block_kl_nats_per_update"].mean()
    surrogate_path = artifact_root() / "analysis/fitted_surrogate.csv"
    surrogate = pd.read_csv(surrogate_path) if surrogate_path.exists() else pd.DataFrame(columns=["alpha", "irreversibility"])
    v10_source = v10[["alpha", "total_per_update_mean", "quadratic_prediction_mean"]].assign(source="V10 heat-bath")
    llm_source = llm.rename(columns={"adjusted_block_kl_nats_per_update": "total_per_update_mean"}).assign(quadratic_prediction_mean=np.nan, source="V12 Qwen")
    surrogate_source = surrogate.groupby("alpha", as_index=False)["irreversibility"].mean().rename(columns={"irreversibility": "total_per_update_mean"}).assign(quadratic_prediction_mean=np.nan, source="fitted surrogate")
    data = pd.concat([v10_source, llm_source, surrogate_source], ignore_index=True)
    _csv(data, source / "figure_12_reference_surrogate_llm.csv")
    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    for label, color, marker in (("V10 heat-bath", BLACK, "^"), ("fitted surrogate", GREEN, "s"), ("V12 Qwen", PURPLE, "o")):
        line = data[data["source"] == label].sort_values("alpha")
        if line.empty:
            continue
        maximum = max(float(line["total_per_update_mean"].abs().max()), 1e-12)
        axis.plot(line["alpha"], line["total_per_update_mean"] / maximum, color=color, marker=marker, linewidth=1.7, label=label)
    axis.set(xlabel="Nonreciprocity $\\alpha$", ylabel="Irreversibility normalized\nwithin estimator")
    axis.text(0.02, 0.98, "Shapes only; estimators are not numerically equivalent", transform=axis.transAxes, va="top", fontsize=8.5)
    axis.legend(frameon=False)
    _save(figure, pdf / "figure_12_effective_model_comparison.pdf")


def _figure_13(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = panel[panel["family"] == "collective_network"].groupby(["n_agents", "alpha"], as_index=False).agg(
        order=("mean_abs_belief_magnetization", "mean"),
        susceptibility=("belief_susceptibility", "mean"),
        irreversibility=("adjusted_block_kl_nats_per_update", "mean"),
        n=("cluster_id", "nunique"),
    )
    _csv(data, source / "figure_13_size_dependence.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for alpha, color, marker in ((0.0, BLUE, "o"), (0.8, ORANGE, "s")):
        line = data[np.isclose(data["alpha"], alpha)].sort_values("n_agents")
        for axis, metric in zip(axes, ("order", "susceptibility", "irreversibility")):
            axis.plot(line["n_agents"], line[metric], color=color, marker=marker, label="$\\alpha=%.1f$" % alpha)
    for axis, label in zip(axes, ("Mean |$m_b$|", "$N\\,\\mathrm{Var}(m_b)$", "Adjusted block KL")):
        axis.set_xlabel("Agents $N$")
        axis.set_ylabel(label)
        axis.set_xticks([8, 16])
    axes[-1].legend(frameon=False)
    _save(figure, pdf / "figure_13_size_dependence.pdf")


def _figure_14(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = panel[panel["family"] == "persistent_memory"].copy()
    data["regime_comparison"] = data["regime"].str.replace("_", " ")
    grouped = data.groupby(["regime_comparison", "alpha"], as_index=False).agg(
        irreversibility=("adjusted_block_kl_nats_per_update", "mean"),
        markov_cmi=("history_1_conditional_mutual_information", "mean"),
        persistence=("belief_integrated_autocorrelation_time_updates", "mean"),
        n=("cluster_id", "nunique"),
    )
    _csv(grouped, source / "figure_14_memory_comparison.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for regime, color, marker in (("markovized", BLUE, "o"), ("persistent memory", ORANGE, "s")):
        line = grouped[grouped["regime_comparison"] == regime].sort_values("alpha")
        for axis, metric in zip(axes, ("irreversibility", "markov_cmi", "persistence")):
            axis.plot(line["alpha"], line[metric], color=color, marker=marker, label=regime)
    for axis, label in zip(axes, ("Adjusted block KL", "History CMI", "Correlation time")):
        axis.set_xlabel("Nonreciprocity $\\alpha$")
        axis.set_ylabel(label)
    axes[-1].legend(frameon=False, fontsize=8)
    _save(figure, pdf / "figure_14_memory_comparison.pdf")


def _figure_15(repository: Path, pdf: Path, source: Path) -> None:
    panel = _panel_table()
    data = panel[panel["family"] == "controls"].groupby("control", as_index=False).agg(
        irreversibility=("adjusted_block_kl_nats_per_update", "mean"),
        order=("mean_abs_belief_magnetization", "mean"),
        wire_bytes=("wire_bytes", "mean"),
        n=("cluster_id", "nunique"),
    ).sort_values("irreversibility")
    _csv(data, source / "figure_15_controls.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    positions = np.arange(len(data))
    axes[0].barh(positions, data["irreversibility"], color=BLUE)
    axes[1].barh(positions, data["order"], color=ORANGE)
    for axis, label in zip(axes, ("Adjusted block KL", "Mean |belief magnetization|")):
        axis.set_yticks(positions, data["control"].str.replace("_", " "))
        axis.set_xlabel(label)
    axes[1].tick_params(labelleft=False)
    _save(figure, pdf / "figure_15_controls.pdf")


def _figure_16(repository: Path, pdf: Path, source: Path) -> None:
    _, directed_path = _representative_files()
    frame = pd.read_csv(directed_path)
    selected = frame.iloc[-1]
    protocol = load_yaml(repository / "configs/statmech_v12/protocol_frozen.yaml")
    definition = next(row for row in formal_panel_design(protocol) if row["panel_id"] == selected["panel_id"])
    graph = _graph_for_panel(definition)
    beliefs = np.asarray([int(value) for value in str(selected["beliefs"]).split(";")])
    actions = np.asarray([int(value) for value in str(selected["actions"]).split(";")])
    nodes = pd.DataFrame({"agent": np.arange(graph.n_agents), "belief": beliefs, "action": actions})
    edges = pd.DataFrame([{"source": i, "target": j, "weight": graph.weights[i, j]} for i, j in zip(*np.nonzero(graph.weights))])
    _csv(nodes, source / "figure_16_network_snapshot_nodes.csv")
    _csv(edges, source / "figure_16_network_snapshot_edges.csv")
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    network = nx.from_numpy_array(graph.adjacency)
    positions = nx.circular_layout(network)
    colors = [ORANGE if value == 1 else BLUE for value in beliefs]
    nx.draw_networkx_nodes(network, positions, node_color=colors, edgecolors=[BLACK if a == b else RED for a, b in zip(actions, beliefs)], linewidths=2, node_size=480, ax=axis)
    nx.draw_networkx_labels(network, positions, font_size=9, ax=axis)
    for row in edges.itertuples():
        axis.add_patch(FancyArrowPatch(positions[row.source], positions[row.target], arrowstyle="-|>", mutation_scale=8,
                                       linewidth=0.5 + 2 * row.weight, color=GREY, alpha=0.65, connectionstyle="arc3,rad=0.11"))
    axis.text(0.02, 0.02, "Fill: belief (-1 blue, +1 orange)\nRed outline: belief-action mismatch", transform=axis.transAxes, fontsize=9)
    axis.set_title("Directed LLM-agent network at sweep %.1f" % selected["sweep"])
    axis.axis("off")
    _save(figure, pdf / "figure_16_network_snapshot.pdf")


def generate_figures(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    _configure()
    pdf, source = _paths(repository)
    functions = [
        _figure_01,
        _figure_02,
        _figure_03,
        _figure_04,
        _figure_05,
        _figure_06,
        _figure_07,
        _figure_08,
        _figure_09,
        _figure_10,
        _figure_11,
        _figure_12,
        _figure_13,
        _figure_14,
        _figure_15,
        _figure_16,
    ]
    for function in functions:
        function(repository, pdf, source)
    return {
        "figure_count": len(list(pdf.glob("figure_*.pdf"))),
        "source_data_files": len(list(source.glob("figure_*.csv"))),
        "pdf_directory": str(pdf),
    }
