"""Data-derived vector candidate figures for the V13 paper package."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .workflow import atomic_csv, utc_now


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


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "ThermoAgent V13"})
    plt.close(figure)


def _source(frame: pd.DataFrame, path: Path) -> None:
    atomic_csv(frame.to_dict("records"), path)


def _mean_error(group: pd.core.groupby.generic.DataFrameGroupBy, metric: str) -> pd.DataFrame:
    result = group[metric].agg(["mean", "std", "count"]).reset_index()
    result["se"] = result["std"].fillna(0.0) / np.sqrt(result["count"].clip(lower=1))
    return result


def _figure_01(pdf: Path, source: Path) -> None:
    nodes = pd.DataFrame(
        [
            ("private observation", 0.08, 0.74, "local"),
            ("bounded memory", 0.08, 0.36, "local"),
            ("LLM agent i", 0.36, 0.55, "agent"),
            ("inbox/outbox", 0.62, 0.74, "network"),
            ("local action", 0.62, 0.36, "action"),
            ("agent j", 0.88, 0.74, "agent"),
            ("environment", 0.88, 0.36, "environment"),
            ("evaluator Z(t)", 0.50, 0.10, "evaluator"),
        ],
        columns=["node", "x", "y", "boundary"],
    )
    edges = pd.DataFrame(
        [
            ("private observation", "LLM agent i", "private"),
            ("bounded memory", "LLM agent i", "private"),
            ("LLM agent i", "inbox/outbox", "chosen message"),
            ("inbox/outbox", "agent j", "delivered only"),
            ("LLM agent i", "local action", "chosen action"),
            ("local action", "environment", "typed effect"),
            ("environment", "private observation", "local field"),
            ("LLM agent i", "evaluator Z(t)", "recorded"),
            ("agent j", "evaluator Z(t)", "recorded"),
            ("environment", "evaluator Z(t)", "offline only"),
        ],
        columns=["source", "target", "relation"],
    )
    _source(nodes, source / "figure_01_nodes.csv")
    _source(edges, source / "figure_01_edges.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    palette = {"local": COLORS["green"], "agent": COLORS["blue"], "network": COLORS["orange"], "action": COLORS["red"], "environment": COLORS["purple"], "evaluator": COLORS["gray"]}
    positions = {row.node: (row.x, row.y) for row in nodes.itertuples()}
    for row in edges.itertuples():
        x1, y1 = positions[row.source]; x2, y2 = positions[row.target]
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.3})
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.018, row.relation, fontsize=8, ha="center", color=COLORS["gray"])
    for row in nodes.itertuples():
        ax.text(row.x, row.y, row.node, ha="center", va="center", color="white", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.45", "facecolor": palette[row.boundary], "edgecolor": "white"})
    ax.text(0.36, 0.93, "Deployable local information and authority", ha="center", fontweight="bold")
    ax.text(0.76, 0.10, "Evaluator state is never prompted", ha="center", color=COLORS["gray"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _save(fig, pdf / "figure_01_micro_to_macro_architecture.pdf")


def _figure_02(micro: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = _mean_error(micro.groupby(["sampling_temperature", "coupling_strength", "neighbor_field"]), "belief_after")
    _source(data, source / "figure_02_individual_response.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axis, temperature in zip(axes, sorted(data["sampling_temperature"].unique())):
        subset = data[data["sampling_temperature"] == temperature]
        for coupling, marker, color in ((0.35, "o", COLORS["blue"]), (0.80, "s", COLORS["red"])):
            group = subset[np.isclose(subset["coupling_strength"], coupling)]
            probability = (group["mean"] + 1.0) / 2.0
            axis.errorbar(group["neighbor_field"], probability, yerr=group["se"] / 2.0, marker=marker, color=color, label=f"J={coupling:.2f}")
        axis.axhline(0.5, color=COLORS["gray"], ls=":")
        axis.set_title(f"decoding noise {temperature:.2f}")
        axis.set_xlabel("delivered neighbor field")
    axes[0].set_ylabel("P(latent + belief choice)")
    axes[1].legend(frameon=False)
    _save(fig, pdf / "figure_02_individual_response_law.pdf")


def _figure_03(surrogate: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = surrogate[(surrogate["n_agents"] == 64) & (surrogate["topology"] == "modular")].copy()
    _source(data, source / "figure_03_surrogate_stability.csv")
    pivot = data.pivot(index="sampling_temperature", columns="coupling_strength", values="local_belief_stability_index_mean")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="mean-field local stability index")
    ax.contour(pivot.columns, pivot.index, pivot.values, levels=[1.0], colors="white", linewidths=1.5)
    ax.set_xlabel("coupling J"); ax.set_ylabel("decision-noise setting")
    ax.set_title("Fitted-surrogate stability map (N=64 modular)")
    _save(fig, pdf / "figure_03_surrogate_stability.pdf")


def _figure_04(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "modular_primary"].copy()
    _source(data, source / "figure_04_direct_order_map.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for axis, n in zip(axes, (8, 16)):
        subset = data[data["n_agents"] == n]
        for temperature, marker, color in ((0.50, "o", COLORS["blue"]), (0.85, "s", COLORS["orange"])):
            group = subset[np.isclose(subset["sampling_temperature"], temperature)]
            axis.scatter(group["coupling_strength"], group["mean_abs_belief_magnetization"], marker=marker, color=color, alpha=0.65, label=f"noise {temperature:.2f}")
        axis.set_title(f"N={n}"); axis.set_xlabel("coupling J")
    axes[0].set_ylabel("mean |belief magnetization|")
    axes[1].legend(frameon=False)
    _save(fig, pdf / "figure_04_direct_llm_order_map.pdf")


def _figure_05(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "modular_primary"].copy()
    _source(data, source / "figure_05_fluctuation_relaxation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for temperature, marker, color in ((0.50, "o", COLORS["blue"]), (0.85, "s", COLORS["orange"])):
        group = data[np.isclose(data["sampling_temperature"], temperature)]
        axes[0].scatter(group["coupling_strength"], group["belief_susceptibility"], marker=marker, color=color, alpha=0.65, label=f"noise {temperature:.2f}")
        axes[1].scatter(group["coupling_strength"], group["belief_integrated_autocorrelation_time_updates"], marker=marker, color=color, alpha=0.65)
    axes[0].set(xlabel="coupling J", ylabel="belief susceptibility", title="Fluctuations")
    axes[1].set(xlabel="coupling J", ylabel="integrated correlation time (updates)", title="Persistence")
    axes[0].legend(frameon=False)
    _save(fig, pdf / "figure_05_susceptibility_correlation_time.pdf")


def _figure_06(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "modular_primary"].copy()
    _source(data, source / "figure_06_entropy_decomposition.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    metrics = (("configuration_entropy", "configuration entropy"), ("single_agent_marginal_entropy", "mean marginal entropy"), ("total_correlation", "total correlation"))
    for axis, (metric, title) in zip(axes, metrics):
        for coupling, color in ((0.35, COLORS["blue"]), (0.80, COLORS["red"])):
            group = data[np.isclose(data["coupling_strength"], coupling)]
            axis.scatter(group["sampling_temperature"], group[metric], color=color, alpha=0.65, label=f"J={coupling:.2f}")
        axis.set_xlabel("decision noise"); axis.set_title(title)
    axes[0].set_ylabel("nats")
    axes[-1].legend(frameon=False)
    _save(fig, pdf / "figure_06_entropy_decomposition.pdf")


def _figure_07(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "modular_primary"].copy()
    _source(data, source / "figure_07_effective_energy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].scatter(data["mean_abs_belief_magnetization"], data["mean_reference_energy_per_agent"], c=data["coupling_strength"], cmap="viridis", s=35)
    axes[0].set(xlabel="mean |belief magnetization|", ylabel="reference energy / agent", title="Effective energy and order")
    axes[1].scatter(data["belief_susceptibility"], data["energy_fluctuation_N_var_e"], c=data["sampling_temperature"], cmap="plasma", s=35)
    axes[1].set(xlabel="belief susceptibility", ylabel=r"$N\,\mathrm{Var}(e_{ref})$", title="Fluctuation correspondence")
    _save(fig, pdf / "figure_07_energy_fluctuations.pdf")


def _trajectory_plot(macro: pd.DataFrame, condition: str, filename: str, pdf: Path, source: Path) -> None:
    data = macro[(macro["family"] == "C_disruption_recovery") & (macro["disruption"] == condition)].copy()
    _source(data, source / (filename + ".csv"))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    phase_colors = {"baseline": COLORS["blue"], "disruption": COLORS["red"], "recovery": COLORS["green"]}
    for cluster, group in data.groupby("cluster_id"):
        group = group.sort_values("sweep")
        ax.plot(group["reference_energy_per_agent"], group["configuration_entropy"], color=COLORS["gray"], alpha=0.35)
        for phase, phase_group in group.groupby("phase"):
            ax.scatter(phase_group["reference_energy_per_agent"], phase_group["configuration_entropy"], s=20, color=phase_colors[phase], marker={"baseline":"o","disruption":"s","recovery":"^"}[phase], label=phase if cluster == sorted(data["cluster_id"].unique())[0] else None)
    ax.set_xlabel("reference energy / agent"); ax.set_ylabel("rolling configuration entropy (nats)")
    ax.set_title(condition.replace("_", " ").title())
    ax.legend(frameon=False)
    _save(fig, pdf / (filename + ".pdf"))


def _figure_12(macro: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = macro[macro["family"] == "C_disruption_recovery"].copy()
    _source(data, source / "figure_12_magnetization_entropy.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
    for axis, condition in zip(axes.ravel(), ("nominal", "field_reversal", "network_partition", "message_corruption")):
        group = data[data["disruption"] == condition]
        for phase, marker, color in (("baseline","o",COLORS["blue"]),("disruption","s",COLORS["red"]),("recovery","^",COLORS["green"])):
            phase_group = group[group["phase"] == phase]
            axis.scatter(phase_group["belief_magnetization"], phase_group["configuration_entropy"], marker=marker, color=color, alpha=0.5, s=18)
        axis.set_title(condition.replace("_", " "))
    for axis in axes[-1]: axis.set_xlabel("belief magnetization")
    for axis in axes[:,0]: axis.set_ylabel("configuration entropy (nats)")
    _save(fig, pdf / "figure_12_magnetization_entropy_trajectories.pdf")


def _figure_13(nodes: pd.DataFrame, edges: pd.DataFrame, pdf: Path, source: Path) -> None:
    data_nodes = nodes[(nodes["disruption"] == "network_partition")]
    data_edges = edges[(edges["disruption"] == "network_partition")]
    _source(data_nodes, source / "figure_13_snapshot_nodes.csv")
    _source(data_edges, source / "figure_13_snapshot_edges.csv")
    graph = nx.Graph()
    graph.add_nodes_from(sorted(data_nodes["node"].unique()))
    baseline_edges = data_edges[data_edges["phase"] == "baseline"]
    graph.add_edges_from((int(row.source), int(row.target)) for row in baseline_edges.itertuples() if row.source < row.target)
    pos = nx.spring_layout(graph, seed=13)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for axis, phase in zip(axes, ("baseline", "disruption", "recovery")):
        node_frame = data_nodes[data_nodes["phase"] == phase]
        edge_frame = data_edges[(data_edges["phase"] == phase) & (data_edges["active"] == 1) & (data_edges["message_count"] > 0)]
        nx.draw_networkx_edges(graph, pos, edgelist=[(int(row.source), int(row.target)) for row in edge_frame.itertuples()], width=[0.6 + 0.25 * row.message_count for row in edge_frame.itertuples()], edge_color=COLORS["gray"], arrows=True, ax=axis)
        nx.draw_networkx_nodes(graph, pos, node_color=[COLORS["blue"] if row.belief < 0 else COLORS["orange"] for row in node_frame.sort_values("node").itertuples()], node_size=[90 + 260 * row.confidence_uncertainty for row in node_frame.sort_values("node").itertuples()], edgecolors=[COLORS["green"] if row.action > 0 else COLORS["black"] for row in node_frame.sort_values("node").itertuples()], linewidths=1.5, ax=axis)
        axis.set_title(phase); axis.axis("off")
    _save(fig, pdf / "figure_13_network_partition_snapshots.pdf")


def _figure_14(macro: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = macro[macro["family"] == "C_disruption_recovery"].copy()
    _source(data, source / "figure_14_disruption_response.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for axis, condition in zip(axes.ravel(), ("nominal", "field_reversal", "network_partition", "message_corruption")):
        group = data[data["disruption"] == condition]
        summary = group.groupby("sweep")["macrostate_distance"].agg(["mean", "std", "count"])
        se = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
        axis.plot(summary.index, summary["mean"], color=COLORS["blue"])
        axis.fill_between(summary.index, summary["mean"]-1.96*se, summary["mean"]+1.96*se, color=COLORS["sky"], alpha=0.3)
        axis.axvspan(15, 30, color=COLORS["red"], alpha=0.08)
        axis.set_title(condition.replace("_", " "))
    for axis in axes[-1]: axis.set_xlabel("sweep")
    for axis in axes[:,0]: axis.set_ylabel("distance from nominal manifold")
    _save(fig, pdf / "figure_14_disruption_response_recovery.pdf")


def _figure_15(recovery: pd.DataFrame, pdf: Path, source: Path) -> None:
    _source(recovery, source / "figure_15_macrostate_departure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    order = ["nominal", "field_reversal", "network_partition", "message_corruption"]
    for index, condition in enumerate(order):
        group = recovery[recovery["disruption"] == condition]
        axes[0].scatter(np.full(len(group), index), group["maximum_disruption_distance"], color=COLORS["blue"], alpha=0.7)
        axes[1].scatter(np.full(len(group), index), group["recovery_time_sweeps"], color=COLORS["green"], alpha=0.7)
    for axis in axes:
        axis.set_xticks(range(len(order))); axis.set_xticklabels([value.replace("_", "\n") for value in order])
    axes[0].set_ylabel("maximum macrostate distance"); axes[1].set_ylabel("recovery time (sweeps)")
    _save(fig, pdf / "figure_15_macrostate_distance.pdf")


def _paired_plot(panel: pd.DataFrame, metric: str, filename: str, ylabel: str, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "memory_confirmation"].copy()
    _source(data, source / (filename + ".csv"))
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for cluster, group in data.groupby("cluster_id"):
        group = group.set_index("regime").loc[["markovized", "persistent_memory"]]
        ax.plot([0,1], group[metric], marker="o", alpha=0.7, color=COLORS["blue"])
    ax.set_xticks([0,1]); ax.set_xticklabels(["Markovized", "bounded memory"])
    ax.set_ylabel(ylabel)
    _save(fig, pdf / (filename + ".pdf"))


def _figure_18(panel: pd.DataFrame, surrogate: pd.DataFrame, pdf: Path, source: Path) -> None:
    direct = panel[panel["subset"] == "modular_primary"].groupby(["n_agents","coupling_strength","sampling_temperature"])["mean_abs_belief_magnetization"].mean().reset_index()
    fitted = surrogate[(surrogate["topology"] == "modular") & surrogate["coupling_strength"].isin([0.35,0.80]) & surrogate["sampling_temperature"].isin([0.50,0.85])].copy()
    fitted = fitted.rename(columns={"mean_abs_belief_magnetization_mean":"surrogate_order"})
    data = direct.merge(fitted[["n_agents","coupling_strength","sampling_temperature","surrogate_order"]], on=["n_agents","coupling_strength","sampling_temperature"])
    _source(data, source / "figure_18_surrogate_direct.csv")
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    ax.scatter(data["surrogate_order"], data["mean_abs_belief_magnetization"], c=data["coupling_strength"], cmap="viridis", s=50)
    limits = [0, max(data["surrogate_order"].max(), data["mean_abs_belief_magnetization"].max()) * 1.08]
    ax.plot(limits, limits, ls="--", color=COLORS["gray"])
    ax.set(xlim=limits, ylim=limits, xlabel="fitted-surrogate order", ylabel="direct LLM order")
    _save(fig, pdf / "figure_18_surrogate_vs_llm.pdf")


def _figure_19(cv: pd.DataFrame, pdf: Path, source: Path) -> None:
    _source(cv, source / "figure_19_representation_ablation.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    order = ["simple", "order_only", "full_statmech"]
    for index, representation in enumerate(order):
        group = cv[cv["representation"] == representation]
        ax.scatter(np.full(len(group), index), group["accuracy"], color=[COLORS["gray"],COLORS["orange"],COLORS["blue"]][index], s=45)
    ax.axhline(0.25, color=COLORS["red"], ls=":", label="four-class chance")
    ax.set_xticks(range(3)); ax.set_xticklabels(["simple", "order only", "full statmech"])
    ax.set_ylabel("leave-cluster-out accuracy"); ax.set_ylim(0,1.05); ax.legend(frameon=False)
    _save(fig, pdf / "figure_19_representation_ablation.pdf")


def _figure_20(effects: pd.DataFrame, v12: pd.DataFrame, pdf: Path, source: Path) -> None:
    v13 = effects[effects["hypothesis"].isin(["H1","H2","H3"])].copy()
    v13["study"] = "V13 confirmation"
    v12copy = v12.copy(); v12copy["hypothesis"] = np.where(v12copy["factor"] == "coupling_strength", "H1", np.where(v12copy["factor"] == "sampling_temperature", "H2", "H3")); v12copy["study"] = "V12 discovery"
    common = ["study","hypothesis","metric","estimate","ci_low","ci_high"]
    data = pd.concat([v12copy[common], v13[common]], ignore_index=True)
    _source(data, source / "figure_20_discovery_confirmation.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    labels=[]; y=[]
    for index, row in enumerate(data.itertuples()):
        color = COLORS["gray"] if row.study == "V12 discovery" else COLORS["blue"]
        ax.errorbar(row.estimate, index, xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]], marker="o", color=color, capsize=2)
        labels.append(f"{row.study}: {row.hypothesis} {row.metric.replace('_',' ')}"); y.append(index)
    ax.axvline(0, color=COLORS["black"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.set_xlabel("paired effect (native units)")
    _save(fig, pdf / "figure_20_v12_v13_confirmation.pdf")


def _figure_21(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    data = panel[panel["subset"] == "modular_primary"].copy()
    _source(data, source / "figure_21_finite_size.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for axis, metric, label in zip(axes, ("mean_abs_belief_magnetization","belief_susceptibility","belief_integrated_autocorrelation_time_updates"), ("order","susceptibility","correlation time")):
        for n, marker, color in ((8,"o",COLORS["blue"]),(16,"s",COLORS["red"])):
            group = data[data["n_agents"] == n]
            axis.scatter(group["coupling_strength"], group[metric], marker=marker, color=color, alpha=.65, label=f"N={n}")
        axis.set_xlabel("coupling J"); axis.set_ylabel(label)
    axes[-1].legend(frameon=False)
    _save(fig, pdf / "figure_21_finite_size.pdf")


def _figure_22(panel: pd.DataFrame, pdf: Path, source: Path) -> None:
    variables = ["mean_abs_belief_magnetization","belief_susceptibility","configuration_entropy","entropy_rate_nats_per_update","total_correlation","mean_reference_energy_per_agent","energy_fluctuation_N_var_e","belief_integrated_autocorrelation_time_updates"]
    data = panel[panel["subset"] == "modular_primary"][variables].corr()
    long = data.stack().rename("correlation").reset_index().rename(columns={"level_0":"variable_x","level_1":"variable_y"})
    _source(long, source / "figure_22_observable_redundancy.csv")
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(data.values, vmin=-1, vmax=1, cmap="coolwarm")
    labels=[value.replace("mean_","").replace("_"," ") for value in variables]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=ax, label="trajectory-level Pearson correlation")
    _save(fig, pdf / "figure_22_observable_correlation.pdf")


def generate_figures(repository: Path) -> Dict[str, object]:
    _configure()
    repository = Path(repository).resolve()
    root = repository / "results/collective_agent_statmech_v13"
    tables = root / "tables"
    pdf = root / "figures/pdf"
    source = root / "figures/source_data"
    pdf.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(tables / "panel_statistics.csv")
    macro = pd.read_csv(tables / "macrostate_trajectories.csv")
    recovery = pd.read_csv(tables / "disruption_recovery.csv")
    micro = pd.read_csv(tables / "microscopic_response.csv")
    surrogate = pd.read_csv(tables / "surrogate_phase_map.csv")
    cv = pd.read_csv(tables / "representation_cv.csv")
    effects = pd.read_csv(tables / "hypothesis_effects.csv")
    v12 = pd.read_csv(tables / "v12_discovery_effects.csv")
    nodes = pd.read_csv(tables / "network_snapshot_nodes.csv")
    edges = pd.read_csv(tables / "network_snapshot_edges.csv")
    _figure_01(pdf, source); _figure_02(micro, pdf, source); _figure_03(surrogate, pdf, source)
    _figure_04(panel, pdf, source); _figure_05(panel, pdf, source); _figure_06(panel, pdf, source); _figure_07(panel, pdf, source)
    _trajectory_plot(macro, "nominal", "figure_08_energy_entropy_nominal", pdf, source)
    _trajectory_plot(macro, "field_reversal", "figure_09_energy_entropy_field_reversal", pdf, source)
    _trajectory_plot(macro, "network_partition", "figure_10_energy_entropy_partition", pdf, source)
    _trajectory_plot(macro, "message_corruption", "figure_11_energy_entropy_corruption", pdf, source)
    _figure_12(macro, pdf, source); _figure_13(nodes, edges, pdf, source); _figure_14(macro, pdf, source); _figure_15(recovery, pdf, source)
    _paired_plot(panel, "adjusted_block_irreversibility_nats_per_update", "figure_16_memory_irreversibility", "adjusted path irreversibility (nats/update)", pdf, source)
    _paired_plot(panel, "energy_entropy_loop_area", "figure_17_memory_loop_area", "energy-entropy loop area", pdf, source)
    _figure_18(panel, surrogate, pdf, source); _figure_19(cv, pdf, source); _figure_20(effects, v12, pdf, source); _figure_21(panel, pdf, source); _figure_22(panel, pdf, source)
    purpose = [
        "information boundary", "microscopic response", "surrogate stability", "direct order map", "fluctuation response",
        "entropy decomposition", "reference energy", "nominal phase portrait", "field-reversal portrait", "partition portrait",
        "corruption portrait", "order-entropy trajectories", "network snapshots", "disruption response", "nominal distance",
        "memory irreversibility", "memory hysteresis", "surrogate comparison", "representation ablation", "discovery-confirmation",
        "finite-size comparison", "observable redundancy",
    ]
    files = sorted(pdf.glob("figure_*.pdf"))
    catalog = [
        {
            "figure": index,
            "filename": path.name,
            "scientific_purpose": purpose[index - 1],
            "recommendation": "main" if index in (1,2,3,4,5,6,9,10,11,14,16,19,20) else "supplemental_candidate",
            "source_data_prefix": path.stem,
            "limitations": "finite-size single-model simulation; effective quantities are not physical thermodynamics",
        }
        for index, path in enumerate(files, start=1)
    ]
    atomic_csv(catalog, root / "figures/figure_catalog.csv")
    return {"generated_at": utc_now(), "figure_count": len(files), "catalog_rows": len(catalog)}
