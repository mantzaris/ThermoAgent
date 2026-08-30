"""Paper-facing vector figures and exact compact source tables for V14."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from .workflow import atomic_csv, atomic_json, load_yaml, sha256_file, utc_now


COLORS = {
    "nominal": "#0072B2",
    "field_reversal": "#D55E00",
    "network_partition": "#009E73",
    "message_corruption": "#CC79A7",
    "markovized": "#0072B2",
    "persistent_memory": "#E69F00",
    "order_only": "#56B4E9",
    "simple_uncertainty": "#009E73",
    "full_statmech": "#D55E00",
}
MARKERS = {"nominal": "o", "field_reversal": "s", "network_partition": "^", "message_corruption": "D"}
LABELS = {
    "nominal": "Nominal",
    "field_reversal": "Field reversal",
    "network_partition": "Partition",
    "message_corruption": "Corruption",
    "markovized": "Markovized",
    "persistent_memory": "Persistent memory",
    "order_only": "Order only",
    "simple_uncertainty": "Simple uncertainty",
    "full_statmech": "Full stat.-mech.",
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


def _source(frame: pd.DataFrame, path: Path) -> None:
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    atomic_csv(cleaned.to_dict("records"), path)


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _phase_shading(axis) -> None:
    axis.axvspan(15, 30, color="#F0E442", alpha=0.12, lw=0)
    axis.axvline(15, color="0.35", ls="--", lw=1.1)
    axis.axvline(30, color="0.35", ls=":", lw=1.1)


def _mean_ci(frame: pd.DataFrame, x: str, y: str, groups: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(groups) + [x], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group[y].to_numpy(float)
        row = dict(zip(list(groups) + [x], keys))
        row.update(
            {
                "mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "independent_clusters": int(group["cluster_id"].nunique()) if "cluster_id" in group else int(len(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _architecture(name: str, source_dir: Path, pdf_dir: Path) -> None:
    nodes = pd.DataFrame(
        [
            ("private observation", 0.05, 0.72, "local"),
            ("private memory", 0.05, 0.28, "local"),
            ("LLM agent i", 0.38, 0.50, "agent"),
            ("inbox", 0.38, 0.82, "message"),
            ("outbox", 0.68, 0.82, "message"),
            ("typed action", 0.68, 0.25, "action"),
            ("environment", 0.92, 0.25, "environment"),
            ("scheduler", 0.38, 0.08, "scheduler"),
        ],
        columns=["label", "x", "y", "kind"],
    )
    edges = pd.DataFrame(
        [
            ("private observation", "LLM agent i", "local evidence"),
            ("private memory", "LLM agent i", "bounded state"),
            ("inbox", "LLM agent i", "delivered only"),
            ("LLM agent i", "outbox", "agent-selected"),
            ("LLM agent i", "typed action", "agent-selected"),
            ("typed action", "environment", "consequence"),
            ("scheduler", "LLM agent i", "update opportunity"),
        ],
        columns=["source", "target", "relation"],
    )
    _source(nodes.merge(edges, how="cross"), source_dir / f"{name}.csv")
    lookup = nodes.set_index("label")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    palette = {"local": "#E8F4FA", "agent": "#FDE9D9", "message": "#E6F4EA", "action": "#F3E5F5", "environment": "#EEEEEE", "scheduler": "#FFF4CC"}
    for row in nodes.itertuples():
        ax.text(row.x, row.y, row.label, ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.32", fc=palette[row.kind], ec="0.25", lw=0.9))
    for row in edges.itertuples():
        start, end = lookup.loc[row.source], lookup.loc[row.target]
        ax.annotate(
            "",
            xy=(end.x, end.y),
            xytext=(start.x, start.y),
            arrowprops=dict(
                arrowstyle="->", color="0.35", lw=1.2, shrinkA=32, shrinkB=32
            ),
        )
    ax.text(0.02, 0.96, "Private boundary", weight="bold", fontsize=10.5)
    ax.text(0.68, 0.96, "Explicit network / environment", weight="bold", fontsize=10.5)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _micro_macro(name: str, source_dir: Path, pdf_dir: Path) -> None:
    rows = pd.DataFrame(
        [
            ("micro", "belief b_i", "realized categorical choice"),
            ("micro", "action a_i", "committed typed action"),
            ("micro", "confidence c_i", "bounded report"),
            ("micro", "memory m_i", "bounded private state"),
            ("macro", "m_b, m_a", "directional order"),
            ("macro", "S, h", "state diversity and entropy rate"),
            ("macro", "e_ref, Var(e)", "reference compatibility and fluctuation"),
            ("macro", "I, T, tau", "dependence, total correlation, persistence"),
        ],
        columns=["level", "symbol", "interpretation"],
    )
    _source(rows, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.6)); ax.axis("off")
    symbol_labels = {
        "belief b_i": r"belief $b_i$",
        "action a_i": r"action $a_i$",
        "confidence c_i": r"confidence $c_i$",
        "memory m_i": r"memory $m_i$",
        "m_b, m_a": r"$m_b,\ m_a$",
        "S, h": r"$S,\ h$",
        "e_ref, Var(e)": r"$e_{\mathrm{ref}},\ \mathrm{Var}(e)$",
        "I, T, tau": r"$I,\ \mathcal{T},\ \tau$",
    }
    for index, row in rows.iterrows():
        column = 0 if row.level == "micro" else 1
        local = index if column == 0 else index - 4
        x, y = (0.24 if column == 0 else 0.76), 0.82 - 0.2 * local
        ax.text(x, y, f"{symbol_labels[row.symbol]}\n{row.interpretation}", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.28", fc="#E8F4FA" if column == 0 else "#FDE9D9", ec="0.3"))
    for y in (0.82, 0.62, 0.42, 0.22):
        ax.annotate("", xy=(0.58, y), xytext=(0.42, y), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.24, 0.96, "Agent-owned microstate", ha="center", weight="bold")
    ax.text(0.76, 0.96, "Observable reduced state", ha="center", weight="bold")
    _finish(fig, pdf_dir / f"{name}.pdf")


def _network(name: str, nodes: pd.DataFrame, edges: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    selected_nodes = nodes[(nodes["disruption"] == "field_reversal")].copy()
    selected_edges = edges[(edges["disruption"] == "field_reversal")].copy()
    node_source = selected_nodes.copy()
    node_source["record_type"] = "node"
    edge_source = selected_edges.copy()
    edge_source["record_type"] = "edge"
    _source(pd.concat([node_source, edge_source], ignore_index=True, sort=False), source_dir / f"{name}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, phase in zip(axes, ("baseline", "disruption", "recovery")):
        n = selected_nodes[selected_nodes["phase"] == phase]
        e = selected_edges[(selected_edges["phase"] == phase) & (selected_edges["active"] == 1)]
        graph = nx.DiGraph()
        graph.add_nodes_from(n["node"])
        graph.add_edges_from(zip(e["source"], e["target"]))
        pos = nx.spring_layout(graph.to_undirected(), seed=14)
        color = ["#0072B2" if int(n.set_index("node").loc[node, "belief"]) < 0 else "#D55E00" for node in graph]
        size = [180 + 220 * float(n.set_index("node").loc[node, "uncertainty"]) for node in graph]
        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowsize=6, width=0.6, alpha=0.35)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=color, node_size=size, linewidths=0.9, edgecolors="black")
        ax.set_title(phase.capitalize(), fontsize=10); ax.axis("off")
    _finish(fig, pdf_dir / f"{name}.pdf")


def _memory_forest(name: str, memory: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    _source(memory, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    data = memory.iloc[::-1].reset_index(drop=True)
    for index, row in data.iterrows():
        ax.errorbar(row.estimate, index, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]], fmt="o", color="#0072B2" if "V12" in row.study else "#D55E00", capsize=3)
    ax.axvline(0, color="0.3", lw=1, ls="--")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([value.replace("_", " ") for value in data["study"]])
    ax.set_xlabel("Memory effect on adjusted irreversibility (nats/update)")
    _finish(fig, pdf_dir / f"{name}.pdf")


def _memory_clusters(name: str, repository: Path, source_dir: Path, pdf_dir: Path) -> None:
    v13 = pd.read_csv(
        repository / "results/JSTAT/stages/replication/tables/panel_statistics.csv"
    )
    v13 = v13[v13["subset"] == "memory_confirmation"]
    rows = []
    for cluster, group in v13.groupby("cluster_id"):
        values = group.set_index("regime")["adjusted_block_irreversibility_nats_per_update"]
        rows.append({"study": "V13", "cluster_id": cluster, "markovized": values["markovized"], "persistent": values["persistent_memory"], "difference": values["persistent_memory"]-values["markovized"]})
    v12 = pd.read_csv(
        repository / "results/JSTAT/stages/discovery/tables/memory_effects.csv"
    )
    v12 = v12[v12["metric"] == "adjusted_block_kl_nats_per_update"]
    for row in v12.itertuples():
        rows.append({"study": "V12", "cluster_id": row.matched_arm, "markovized": np.nan, "persistent": np.nan, "difference": row.persistent_minus_markovized})
    frame = pd.DataFrame(rows); _source(frame, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for study, marker, color in (("V12", "o", "#0072B2"), ("V13", "s", "#D55E00")):
        subset = frame[frame["study"] == study]
        ax.scatter(np.arange(len(subset)), subset["difference"], marker=marker, color=color, label=f"{study} clusters", alpha=0.8)
    ax.axhline(0, color="0.3", ls="--", lw=1); ax.set_ylabel("Persistent − Markovized (nats/update)"); ax.set_xlabel("Independent matched comparison"); ax.legend(frameon=False)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _line_groups(name: str, frame: pd.DataFrame, x: str, y: str, group: str, xlabel: str, ylabel: str, source_dir: Path, pdf_dir: Path, shade: bool = False) -> None:
    _source(frame[[column for column in frame.columns if column in {"cluster_id", "panel_id", x, y, group, "phase"}]], source_dir / f"{name}.csv")
    summary = frame.groupby([group, x], as_index=False)[y].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fallback_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    fallback_markers = ("o", "s", "^", "D", "v")
    for index, (value, subset) in enumerate(summary.groupby(group)):
        color = COLORS.get(str(value), fallback_colors[index % len(fallback_colors)])
        marker = MARKERS.get(str(value), fallback_markers[index % len(fallback_markers)])
        label = LABELS.get(str(value), str(value).replace("_", " "))
        ax.plot(subset[x], subset["mean"], marker=marker, label=label, color=color)
        ax.fill_between(subset[x], subset["mean"]-1.96*subset["sem"].fillna(0), subset["mean"]+1.96*subset["sem"].fillna(0), color=color, alpha=0.16)
    if shade: _phase_shading(ax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False, ncol=2)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _phase_portrait(name: str, macro: pd.DataFrame, condition: str, x: str, y: str, xlabel: str, ylabel: str, source_dir: Path, pdf_dir: Path) -> None:
    frame = macro[macro["disruption"] == condition].copy()
    _source(frame[["cluster_id", "sweep", "phase", x, y]], source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    mean = frame.groupby("sweep", as_index=False)[[x, y]].mean()
    points = ax.scatter(mean[x], mean[y], c=mean["sweep"], cmap="viridis", s=34, zorder=3)
    ax.plot(mean[x], mean[y], color="0.5", lw=1)
    offsets = {0: (6, 6), 14: (8, 8), 29: (25, 0), 44: (42, 8)}
    for index in (0, 14, 29, 44):
        if index < len(mean):
            ax.annotate(
                str(index + 1),
                (mean.iloc[index][x], mean.iloc[index][y]),
                xytext=offsets[index],
                textcoords="offset points",
                fontsize=8,
            )
    fig.colorbar(points, ax=ax, label="Sweep"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _summary_bars(name: str, frame: pd.DataFrame, metric: str, ylabel: str, source_dir: Path, pdf_dir: Path) -> None:
    _source(frame[["cluster_id", "disruption", metric]], source_dir / f"{name}.csv")
    order = ["nominal", "field_reversal", "network_partition", "message_corruption"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for index, condition in enumerate(order):
        values = frame[frame["disruption"] == condition][metric].to_numpy(float)
        ax.scatter(np.full(len(values), index) + np.linspace(-.08,.08,len(values)), values, color=COLORS[condition], marker=MARKERS[condition], zorder=3)
        ax.errorbar(index, np.mean(values), yerr=1.96*np.std(values,ddof=1)/np.sqrt(len(values)), fmt="_", color="black", capsize=4, ms=15)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Nominal", "Field\nreversal", "Partition", "Corruption"]); ax.set_ylabel(ylabel)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _representation(name: str, folds: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    _source(folds, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    order = ["order_only", "simple_uncertainty", "full_statmech"]
    for index, representation in enumerate(order):
        values = folds[folds["representation"] == representation]["balanced_accuracy"].to_numpy(float)
        ax.scatter(np.full(len(values), index)+np.linspace(-.08,.08,len(values)), values, color=COLORS[representation], label=LABELS[representation] if index == 0 else None)
        ax.errorbar(index, np.mean(values), yerr=1.96*np.std(values,ddof=1)/np.sqrt(len(values)), fmt="_", color="black", capsize=4, ms=15)
    ax.axhline(.25, color="0.4", ls="--", lw=1)
    ax.set_ylim(0,1.02)
    ax.set_ylabel("LOCO balanced accuracy")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[v] for v in order], rotation=12)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _confusions(name: str, predictions: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    _source(predictions, source_dir / f"{name}.csv")
    labels = ["nominal", "field_reversal", "network_partition", "message_corruption"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, representation in zip(axes, ("order_only", "simple_uncertainty", "full_statmech")):
        selected = predictions[predictions["representation"] == representation]
        matrix = confusion_matrix(selected["truth"], selected["prediction"], labels=labels)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i,j in np.ndindex(matrix.shape): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8)
        ax.set_title(LABELS[representation], fontsize=9.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(["N","F","P","C"])
        ax.set_yticks(range(4)); ax.set_yticklabels(["N","F","P","C"] if ax is axes[0] else [])
    axes[0].set_ylabel("Observed"); axes[1].set_xlabel("Predicted")
    _finish(fig, pdf_dir / f"{name}.pdf")


def _robustness(name: str, frame: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    selected = frame[(frame["ablation"] == "all") & (frame["nominal_fit_window"] == "all_nominal") & (frame["rolling_window_sweeps"] == 5)].copy()
    _source(selected, source_dir / f"{name}.csv")
    summary = selected.groupby(["estimator", "disruption"], as_index=False)["maximum_distance"].mean()
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for condition in ("nominal","field_reversal","network_partition","message_corruption"):
        subset = summary[summary["disruption"] == condition]
        ax.plot(subset["estimator"], subset["maximum_distance"], marker=MARKERS[condition], color=COLORS[condition], label=LABELS[condition])
    ax.set_ylabel("Mean maximum macrostate distance"); ax.set_xlabel("Covariance / distance estimator"); ax.legend(frameon=False, ncol=2)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _family_ablation(name: str, frame: pd.DataFrame, source_dir: Path, pdf_dir: Path, contribution: bool = False) -> None:
    selected = frame[(frame["disruption"] == "field_reversal") & (frame["nominal_fit_window"] == "all_nominal") & (frame["rolling_window_sweeps"] == 5) & (frame["deleted_observable"] == "none") & (frame["estimator"] == "shrinkage") & np.isclose(frame["ridge_fraction"],.1)].copy()
    _source(selected, source_dir / f"{name}.csv")
    means = selected.groupby("ablation", as_index=False)["maximum_distance"].mean()
    if contribution:
        full = float(means[means["ablation"] == "all"]["maximum_distance"].iloc[0])
        means["plotted_value"] = full - means["maximum_distance"]
        xlabel = "Distance loss after family removal"
    else:
        means["plotted_value"] = means["maximum_distance"]
        xlabel = "Field-reversal maximum distance"
    means = means.sort_values("plotted_value")
    display_labels = means["ablation"].replace({"all": "none (all retained)"})
    display_labels = display_labels.str.replace("without_", "", regex=False).str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.4)); ax.barh(display_labels, means["plotted_value"], color="#56B4E9")
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel("Removed observable family")
    _finish(fig, pdf_dir / f"{name}.pdf")


def _two_time_panels(name: str, frame: pd.DataFrame, variables: Sequence[Tuple[str, str]], source_dir: Path, pdf_dir: Path) -> None:
    columns = ["cluster_id", "panel_id", "disruption", "sweep", "phase"] + [value[0] for value in variables]
    _source(frame[columns], source_dir / f"{name}.csv")
    fig, axes = plt.subplots(1, len(variables), figsize=(7.0, 3.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (variable, label) in zip(axes, variables):
        summary = frame.groupby("sweep")[variable].agg(["mean", "sem"]).reset_index()
        ax.plot(summary["sweep"], summary["mean"], color="#D55E00")
        ax.fill_between(summary["sweep"], summary["mean"] - 1.96 * summary["sem"].fillna(0), summary["mean"] + 1.96 * summary["sem"].fillna(0), color="#D55E00", alpha=0.16)
        _phase_shading(ax); ax.set_xlabel("Sweep"); ax.set_ylabel(label)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _memory_block(name: str, frame: pd.DataFrame, metric: str, ylabel: str, source_dir: Path, pdf_dir: Path) -> None:
    available = frame[np.isfinite(pd.to_numeric(frame.get(metric), errors="coerce"))].copy()
    if available.empty:
        available = pd.DataFrame(
            [{"study": "external_raw_unavailable", "block_length": 3, metric: np.nan, "status": "not plotted"}]
        )
    _source(available, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(5.9, 3.4))
    finite = available[np.isfinite(pd.to_numeric(available[metric], errors="coerce"))]
    if not finite.empty:
        group_cols = ["study", "block_length"] if "block_length" in available else ["study"]
        summary = finite.groupby(group_cols, as_index=False)[metric].mean()
        for study, subset in summary.groupby("study"):
            x = subset["block_length"] if "block_length" in subset else np.arange(len(subset))
            ax.plot(x, subset[metric], marker="o", label=study.replace("_"," "))
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "External raw sensitivity unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0,color="0.4",ls="--",lw=1); ax.set_xlabel("Block length"); ax.set_ylabel(ylabel)
    _finish(fig, pdf_dir / f"{name}.pdf")


def _v13_topology(name: str, repository: Path, source_dir: Path, pdf_dir: Path) -> None:
    frame = pd.read_csv(
        repository / "results/JSTAT/stages/replication/tables/panel_statistics.csv"
    )
    selected = frame[frame["subset"].isin(["modular_primary","ring_replication"])][["cluster_id","topology","n_agents","coupling_strength","sampling_temperature","mean_abs_belief_magnetization","belief_susceptibility"]]
    _source(selected, source_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(5.8,3.4))
    for topology, marker, color in (("modular","o","#0072B2"),("ring","s","#D55E00")):
        subset=selected[selected.topology==topology]
        grouped=subset.groupby("n_agents",as_index=False)["mean_abs_belief_magnetization"].mean()
        ax.plot(grouped.n_agents,grouped.mean_abs_belief_magnetization,marker=marker,color=color,label=topology.capitalize())
    ax.set_xlabel("Agents N"); ax.set_ylabel("Mean |belief magnetization|"); ax.legend(frameon=False)
    _finish(fig,pdf_dir/f"{name}.pdf")


def _surrogate(name: str, repository: Path, source_dir: Path, pdf_dir: Path) -> None:
    direct=pd.read_csv(repository/"results/JSTAT/stages/replication/tables/panel_statistics.csv")
    direct=direct[direct.subset=="modular_primary"].groupby(["coupling_strength","sampling_temperature"],as_index=False).mean(numeric_only=True)
    surrogate=pd.read_csv(repository/"results/JSTAT/stages/replication/tables/surrogate_phase_map.csv")
    surrogate=surrogate[(surrogate.topology=="modular")&(surrogate.n_agents==16)&surrogate.coupling_strength.isin([.35,.8])&surrogate.sampling_temperature.isin([.5,.85])]
    rows=[]
    for row in direct.itertuples(): rows.append({"source":"Direct LLM","coupling":row.coupling_strength,"noise":row.sampling_temperature,"order":row.mean_abs_belief_magnetization})
    for row in surrogate.itertuples(): rows.append({"source":"Kinetic surrogate","coupling":row.coupling_strength,"noise":row.sampling_temperature,"order":row.mean_abs_belief_magnetization_mean})
    frame=pd.DataFrame(rows); _source(frame,source_dir/f"{name}.csv")
    fig,ax=plt.subplots(figsize=(5.8,3.4))
    for source,marker,color in (("Direct LLM","o","#0072B2"),("Kinetic surrogate","s","#D55E00")):
        subset=frame[(frame.source==source)&np.isclose(frame.noise,.5)]
        ax.plot(subset.coupling,subset.order,marker=marker,color=color,label=source)
    ax.set_xlabel("Coupling J"); ax.set_ylabel("Mean |belief magnetization|"); ax.legend(frameon=False)
    _finish(fig,pdf_dir/f"{name}.pdf")


def _effect_forest(name: str, effects: pd.DataFrame, memory: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    all_field=effects.copy(); all_field["label"]=all_field["hypothesis"]+": "+all_field["estimand"]
    all_field["role"]=np.where(all_field["hypothesis"]=="H3","historical_invalid_directional_test","confirmatory")
    field=all_field[all_field["hypothesis"]!="H3"].copy()
    mem=memory.iloc[:2].copy(); mem["label"]=mem["study"]; mem["hypothesis"]="H1"
    source=pd.concat([all_field[["label","hypothesis","estimate","ci_low","ci_high","role"]],mem.assign(role="discovery_or_replication")[["label","hypothesis","estimate","ci_low","ci_high","role"]]],ignore_index=True)
    _source(source,source_dir/f"{name}.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    left=mem.reset_index(drop=True)
    for i,row in left.iterrows(): axes[0].errorbar(row.estimate,i,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt="o",capsize=3)
    axes[0].axvline(0,color=".4",ls="--"); axes[0].set_yticks(range(len(left))); axes[0].set_yticklabels(left.study.str.replace("_"," ")); axes[0].set_xlabel("Irreversibility effect")
    right=field.reset_index(drop=True)
    for i,row in right.iterrows(): axes[1].errorbar(row.estimate,i,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt="s",capsize=3,color="#D55E00")
    axes[1].axvline(0,color=".4",ls="--"); axes[1].set_yticks(range(len(right))); axes[1].set_yticklabels(right.hypothesis); axes[1].set_xlabel("V14 confirmatory effect")
    axes[1].text(
        0.98,
        0.94,
        "H3 not shown\ninvalid directional test",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9, pad=2.0),
    )
    _finish(fig,pdf_dir/f"{name}.pdf")


def _permutation_audit(name: str, null: pd.DataFrame, summary: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    source=null.copy()
    for row in summary.itertuples():
        source["observed_"+row.metric]=float(row.observed)
    _source(source,source_dir/f"{name}.csv")
    metrics=("full_statmech_balanced_accuracy","full_minus_order_only_balanced_accuracy","full_minus_simple_uncertainty_balanced_accuracy")
    labels=("Full accuracy","Full - order","Full - uncertainty")
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),constrained_layout=True)
    for ax,metric,label in zip(axes,metrics,labels):
        values=null[metric].to_numpy(float)
        observed=float(summary.loc[summary.metric==metric,"observed"].iloc[0])
        pvalue=float(summary.loc[summary.metric==metric,"upper_tail_empirical_p"].iloc[0])
        ax.hist(values,bins=25,color="#56B4E9",alpha=.85)
        ax.axvline(observed,color="#D55E00",lw=2,label="Observed")
        ax.set_xlabel(label); ax.set_ylabel("Permutations" if ax is axes[0] else "")
        ax.text(.04,.94,"p = %.3f"%pvalue,transform=ax.transAxes,va="top",fontsize=8.8)
    axes[0].legend(frameon=False,fontsize=8.5)
    _finish(fig,pdf_dir/f"{name}.pdf")


def _information_bias_audit(name: str, clusters: pd.DataFrame, source_dir: Path, pdf_dir: Path) -> None:
    selected=clusters[clusters.metric.isin(["total_correlation_raw","total_correlation_bias_adjusted","pairwise_mutual_information_raw","pairwise_mutual_information_bias_adjusted"])].copy()
    _source(selected,source_dir/f"{name}.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.0),constrained_layout=True)
    families=(("total_correlation_raw","total_correlation_bias_adjusted"),("pairwise_mutual_information_raw","pairwise_mutual_information_bias_adjusted"))
    titles=("Total correlation","Pairwise mutual information")
    for ax,metrics,title in zip(axes,families,titles):
        for metric,marker,color in ((metrics[0],"o","#0072B2"),(metrics[1],"s","#D55E00")):
            subset=selected[selected.metric==metric]
            for window,group in subset.groupby("window_sweeps"):
                ax.scatter(np.full(len(group),window)+(0.08 if "adjusted" in metric else -0.08),group.field_minus_nominal,marker=marker,color=color,alpha=.8)
            mean=subset.groupby("window_sweeps",as_index=False).field_minus_nominal.mean()
            ax.plot(mean.window_sweeps+(0.08 if "adjusted" in metric else -0.08),mean.field_minus_nominal,marker=marker,color=color,label="Bias-adjusted" if "adjusted" in metric else "Raw")
        ax.axhline(0,color=".4",ls="--",lw=1); ax.set_xticks([3,5,7]); ax.set_xlabel("Rolling window (sweeps)"); ax.set_title(title,fontsize=10)
    axes[0].set_ylabel("Field - nominal (nats)"); axes[1].legend(frameon=False,fontsize=8.5)
    _finish(fig,pdf_dir/f"{name}.pdf")


def generate_figures(repository: Path) -> Dict[str, object]:
    _style()
    repository=Path(repository).resolve(); root=repository/"results/JSTAT/stages/corrected_quench"
    tables=root/"tables"; pdf_dir=root/"figures/pdf"; source_dir=root/"figures/source_data"
    pdf_dir.mkdir(parents=True,exist_ok=True); source_dir.mkdir(parents=True,exist_ok=True)
    macro=pd.read_csv(tables/"macrostate_trajectories.csv")
    panels=pd.read_csv(tables/"panel_statistics.csv")
    recovery=pd.read_csv(tables/"quench_recovery.csv")
    robustness=pd.read_csv(tables/"macrostate_distance_robustness.csv")
    folds=pd.read_csv(tables/"representation_cv.csv")
    predictions=pd.read_csv(tables/"representation_predictions.csv")
    memory=pd.read_csv(tables/"memory_discovery_replication.csv")
    legacy=pd.read_csv(tables/"legacy_memory_block_sensitivity.csv")
    depths=pd.read_csv(tables/"conditional_memory_depth.csv")
    nodes=pd.read_csv(tables/"network_snapshot_nodes.csv"); edges=pd.read_csv(tables/"network_snapshot_edges.csv")
    effects=pd.read_csv(tables/"hypothesis_effects.csv")
    permutation_null=pd.read_csv(tables/"representation_permutation_null.csv")
    permutation_summary=pd.read_csv(tables/"representation_permutation_summary.csv")
    information_clusters=pd.read_csv(tables/"information_estimator_cluster_contrasts.csv")
    catalog=[]
    def register(number,name,purpose,source_table,claim,recommendation="supplementary",limitation="Finite-size, one pinned model"):
        catalog.append({"figure":number,"filename":name+".pdf","purpose":purpose,"source_table":source_table,"estimand":claim,"recommendation":recommendation,"supported_claim":claim,"limitation":limitation})
    _architecture("figure01_agent_architecture",source_dir,pdf_dir); register(1,"figure01_agent_architecture","Independent-agent boundaries","figure01_agent_architecture.csv","Agents own state and decisions","main")
    _micro_macro("figure02_micro_macro_mapping",source_dir,pdf_dir); register(2,"figure02_micro_macro_mapping","Micro-to-macro formulation","figure02_micro_macro_mapping.csv","Observable reduced state","main")
    _network("figure03_network_states",nodes,edges,source_dir,pdf_dir); register(3,"figure03_network_states","Network state snapshots","network_snapshot_nodes/edges.csv","Explicit directed communication")
    _memory_forest("figure04_memory_discovery_replication",memory,source_dir,pdf_dir); register(4,"figure04_memory_discovery_replication","V12 discovery and V13 replication","memory_discovery_replication.csv","Memory-associated irreversibility","main")
    _memory_clusters("figure05_memory_cluster_effects",repository,source_dir,pdf_dir); register(5,"figure05_memory_cluster_effects","Raw matched memory effects","V12/V13 immutable aggregate tables","Cluster heterogeneity")
    _memory_block("figure06_block_length_sensitivity",legacy,"adjusted_irreversibility_nats_per_update","Adjusted reversal divergence (nats/update)",source_dir,pdf_dir); register(6,"figure06_block_length_sensitivity","Block-length sensitivity","legacy_memory_block_sensitivity.csv","Estimator robustness")
    _memory_block("figure07_bias_floor",legacy,"shuffle_floor_nats_per_update","Time-shuffle floor (nats/update)",source_dir,pdf_dir); register(7,"figure07_bias_floor","Finite-sample bias floor","legacy_memory_block_sensitivity.csv","Bias adjustment")
    _line_groups("figure08_conditional_memory_depth",depths,"memory_depth","conditional_mutual_information_nats","disruption","Memory depth k","Conditional mutual information (nats)",source_dir,pdf_dir); register(8,"figure08_conditional_memory_depth","Conditional memory depth","conditional_memory_depth.csv","History dependence")
    entropy_long=macro.melt(id_vars=["cluster_id","disruption","sweep","phase"],value_vars=["mean_individual_entropy","configuration_entropy","total_correlation"],var_name="entropy_component",value_name="value")
    _line_groups("figure09_entropy_decomposition",entropy_long,"sweep","value","entropy_component","Sweep","Entropy / dependence (nats)",source_dir,pdf_dir,True); register(9,"figure09_entropy_decomposition","Entropy decomposition","macrostate_trajectories.csv","Distinct uncertainty and dependence")
    field=macro[macro.disruption=="field_reversal"]
    _two_time_panels("figure10_energy_entropy_quench",field,[("reference_energy_per_agent","Reference energy / agent"),("configuration_entropy","Configuration entropy (nats)")],source_dir,pdf_dir); register(10,"figure10_energy_entropy_quench","Energy and entropy during quench","macrostate_trajectories.csv","Quench response","main")
    _phase_portrait("figure11_energy_entropy_phase_space",macro,"field_reversal","reference_energy_per_agent","configuration_entropy","Reference energy per agent","Configuration entropy (nats)",source_dir,pdf_dir); register(11,"figure11_energy_entropy_phase_space","Energy-entropy path","macrostate_trajectories.csv","Quench/counter-quench path","main")
    _phase_portrait("figure12_belief_action_phase_space",macro,"field_reversal","belief_magnetization","action_magnetization","Belief magnetization","Action magnetization",source_dir,pdf_dir); register(12,"figure12_belief_action_phase_space","Belief-action lag portrait","macrostate_trajectories.csv","Layered response")
    _line_groups("figure13_macrostate_distance",macro,"sweep","macrostate_distance","disruption","Sweep","LOCO macrostate distance",source_dir,pdf_dir,True); register(13,"figure13_macrostate_distance","Macrostate departure","macrostate_trajectories.csv","Field-quench departure","main")
    _line_groups("figure14_quench_counterquench",field,"sweep","macrostate_distance","phase","Sweep","Macrostate distance",source_dir,pdf_dir,True); register(14,"figure14_quench_counterquench","Quench and restoration","macrostate_trajectories.csv","Recovery path")
    _summary_bars("figure15_peak_integrated_recovery",recovery,"maximum_post_quench_distance","Maximum post-quench distance",source_dir,pdf_dir); register(15,"figure15_peak_integrated_recovery","Peak responses","quench_recovery.csv","Condition response")
    _summary_bars("figure16_hysteresis_route_asymmetry",recovery,"energy_entropy_signed_loop_area","Signed energy-entropy loop area",source_dir,pdf_dir); register(16,"figure16_hysteresis_route_asymmetry","Route asymmetry","quench_recovery.csv","Finite-system hysteresis")
    _two_time_panels("figure17_susceptibility_correlation",field,[("belief_susceptibility","Susceptibility"),("integrated_correlation_time","Correlation time (updates)")],source_dir,pdf_dir); register(17,"figure17_susceptibility_correlation","Fluctuation and persistence","macrostate_trajectories.csv","Dynamic response")
    _family_ablation("figure18_observable_family_contribution",robustness,source_dir,pdf_dir,True); register(18,"figure18_observable_family_contribution","Observable-family contribution","macrostate_distance_robustness.csv","Separation attribution")
    _family_ablation("figure19_leave_family_out",robustness,source_dir,pdf_dir,False); register(19,"figure19_leave_family_out","Leave-family-out sensitivity","macrostate_distance_robustness.csv","Separation robustness")
    _robustness("figure20_distance_robustness",robustness,source_dir,pdf_dir); register(20,"figure20_distance_robustness","Distance estimator audit","macrostate_distance_robustness.csv","Scale robustness","main")
    _representation("figure21_representation_ablation",folds,source_dir,pdf_dir); register(21,"figure21_representation_ablation","Representation comparison","representation_cv.csv","Full vs reduced representation","main")
    _confusions("figure22_cluster_confusion",predictions,source_dir,pdf_dir); register(22,"figure22_cluster_confusion","Cluster-held-out confusion","representation_predictions.csv","Condition separation")
    _line_groups("figure23_condition_trajectories",macro,"sweep","macrostate_distance","disruption","Sweep","LOCO macrostate distance",source_dir,pdf_dir,True); register(23,"figure23_condition_trajectories","Four matched conditions","macrostate_trajectories.csv","Perturbation comparison")
    _v13_topology("figure24_topology_context",repository,source_dir,pdf_dir); register(24,"figure24_topology_context","V13 topology context","immutable V13 panel statistics","Topology boundary")
    _surrogate("figure25_surrogate_vs_llm",repository,source_dir,pdf_dir); register(25,"figure25_surrogate_vs_llm","Reference surrogate comparison","immutable V13 direct/surrogate aggregates","Effective-model limit","main")
    _effect_forest("figure26_claims_forest",effects,memory,source_dir,pdf_dir); register(26,"figure26_claims_forest","Compatible replication and V14 effects","hypothesis_effects and memory tables","Claim disposition","main")
    _permutation_audit("figure27_representation_permutation",permutation_null,permutation_summary,source_dir,pdf_dir); register(27,"figure27_representation_permutation","Cluster-preserving representation permutation audit","representation_permutation_null.csv","Representation robustness")
    _information_bias_audit("figure28_information_bias",information_clusters,source_dir,pdf_dir); register(28,"figure28_information_bias","Finite-sample dependence audit","information_estimator_cluster_contrasts.csv","Raw versus bias-adjusted dependence")
    catalog_frame=pd.DataFrame(catalog); atomic_csv(catalog_frame.to_dict("records"),root/"figures/figure_catalog.csv")
    summary={"generated_at":utc_now(),"figure_count":len(catalog),"pdf_count":len(list(pdf_dir.glob("*.pdf"))),"source_table_count":len(list(source_dir.glob("*.csv"))),"catalog_sha256":sha256_file(root/"figures/figure_catalog.csv")}
    atomic_json(summary,root/"reproducibility/figure_generation.json")
    return summary


__all__=["generate_figures"]
