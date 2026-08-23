"""Compact, source-backed vector figures for the V14 audit and V15 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .workflow import atomic_csv, atomic_json, sha256_file, utc_now


PALETTE = {
    "qwen": "#0072B2",
    "granite": "#D55E00",
    "nominal_markovized": "#7F7F7F",
    "field_markovized": "#009E73",
    "field_persistent": "#CC79A7",
    "field_scrambled": "#E69F00",
    "direct": "#0072B2",
    "surrogate": "#D55E00",
}
MARKERS = {
    "qwen": "o",
    "granite": "s",
    "nominal_markovized": "o",
    "field_markovized": "s",
    "field_persistent": "^",
    "field_scrambled": "D",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _save(
    figure: plt.Figure,
    name: str,
    source: pd.DataFrame,
    result: Path,
    catalog: List[Dict[str, object]],
    purpose: str,
    estimand: str,
    recommendation: str,
    claim: str,
    limitation: str,
) -> None:
    source_path = result / "figures/source_data" / (name + ".csv")
    pdf_path = result / "figures/pdf" / (name + ".pdf")
    atomic_csv(source, source_path)
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)
    catalog.append(
        {
            "filename": pdf_path.name,
            "purpose": purpose,
            "source_table": source_path.relative_to(result).as_posix(),
            "estimand": estimand,
            "recommendation": recommendation,
            "supported_claim": claim,
            "important_limitation": limitation,
            "pdf_sha256": sha256_file(pdf_path),
            "source_sha256": sha256_file(source_path),
        }
    )


def _phase_background(axis: plt.Axes) -> None:
    axis.axvspan(15.5, 30.5, color="#F0E442", alpha=0.14, lw=0)
    axis.axvline(15.5, color="#666666", ls="--", lw=1.0)
    axis.axvline(30.5, color="#666666", ls=":", lw=1.0)


def _architecture(result: Path, catalog: List[Dict[str, object]]) -> None:
    rows = pd.DataFrame(
        [
            ("private_observation", 0.08, 0.76, "local"),
            ("bounded_memory", 0.08, 0.48, "local"),
            ("delivered_inbox", 0.08, 0.20, "local"),
            ("LLM_local_transition", 0.39, 0.48, "agent"),
            ("belief_action_packet", 0.69, 0.68, "observable"),
            ("typed_local_action", 0.69, 0.28, "observable"),
            ("delivery_graph", 0.91, 0.68, "network"),
            ("environment", 0.91, 0.28, "network"),
            ("observable_projection_Y", 0.39, 0.08, "evaluator"),
            ("rolling_macrostate_Z", 0.69, 0.08, "evaluator"),
        ],
        columns=("component", "x", "y", "boundary"),
    )
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    colors = {"local": "#56B4E9", "agent": "#009E73", "observable": "#E69F00", "network": "#CC79A7", "evaluator": "#999999"}
    for row in rows.itertuples():
        ax.scatter(row.x, row.y, s=1050, marker="s", color=colors[row.boundary], alpha=0.22, edgecolor=colors[row.boundary], lw=1.5)
        ax.text(row.x, row.y, str(row.component).replace("_", "\n"), ha="center", va="center", fontsize=9)
    arrows = ((0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 7), (4, 8), (5, 8), (8, 9))
    for source, target in arrows:
        a, b = rows.iloc[source], rows.iloc[target]
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"})
    ax.text(0.02, 0.98, r"$\Xi_t$: complete augmented simulator state", transform=ax.transAxes, va="top", fontsize=10.5, weight="bold")
    ax.text(0.02, 0.02, r"$Y_t=\phi(\Xi_t)$; $Z_t=\psi(Y_{t-w+1:t})$", transform=ax.transAxes, va="bottom", fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    _save(fig, "figure01_augmented_state_architecture", rows, result, catalog, "Define agent boundaries and the augmented-to-macrostate projection", "state and information-flow definitions", "main", "LLM decisions are local while macroscopic analysis is evaluator-side", "The projection need not be Markov")


def _memory_replication(repository: Path, result: Path, catalog: List[Dict[str, object]]) -> None:
    legacy = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/memory_discovery_replication.csv")
    effects = pd.read_csv(result / "tables/hypothesis_effects.csv")
    panels = pd.read_csv(result / "tables/panel_statistics.csv")
    rows: List[Dict[str, object]] = []
    for item in legacy[legacy["study"] != "descriptive_fixed_effect_synthesis"].itertuples():
        rows.append({"study": item.study, "model": "Qwen", "contrast": "persistent-Markovized", "estimate": item.estimate, "ci_low": item.ci_low, "ci_high": item.ci_high, "role": item.role})
    for model, group in panels.groupby("model_key"):
        values = []
        for cluster, matched in group.groupby("cluster_id"):
            persistent = matched[matched["condition"] == "field_persistent"].iloc[0]
            markov = matched[matched["condition"] == "field_markovized"].iloc[0]
            values.append(float(persistent.adjusted_pathwise_irreversibility_nats_per_update - markov.adjusted_pathwise_irreversibility_nats_per_update))
        rng = np.random.default_rng(15158201 + (model == "granite"))
        boot = np.asarray([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(10000)])
        rows.append({"study": "V15_%s" % model, "model": model.title(), "contrast": "persistent-Markovized", "estimate": float(np.mean(values)), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "role": "prospective_cross_model"})
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    y = np.arange(len(source))[::-1]
    for index, row in enumerate(source.itertuples()):
        color = PALETTE["granite"] if str(row.model).lower() == "granite" else PALETTE["qwen"]
        ax.errorbar(row.estimate, y[index], xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.set_yticks(y, [str(value).replace("_", " ") for value in source["study"]])
    ax.set_xlabel("Bias-adjusted path-reversal difference (nats/update)")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "figure02_memory_discovery_replication", source, result, catalog, "Separate memory discovery, replication, and cross-model extension", "persistent minus Markovized adjusted block divergence", "main", "Memory-associated temporal asymmetry is evaluated across distinct study roles", "V12/V13 are not prospectively pooled with V15")


def _v14_quench(repository: Path, result: Path, catalog: List[Dict[str, object]]) -> None:
    frame = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/macrostate_trajectories.csv")
    source = frame[frame["disruption"].isin(("nominal", "field_reversal"))][["cluster_id", "disruption", "sweep", "phase", "reference_energy_per_agent", "configuration_entropy", "macrostate_distance"]].copy()
    aggregate = source.groupby(["disruption", "sweep", "phase"], as_index=False).agg(energy=("reference_energy_per_agent", "mean"), entropy=("configuration_entropy", "mean"), distance=("macrostate_distance", "mean"))
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.7), sharex=True)
    for disruption, group in aggregate.groupby("disruption"):
        color = "#0072B2" if disruption == "nominal" else "#D55E00"
        label = disruption.replace("_", " ")
        for axis, metric, ylabel in zip(axes, ("energy", "entropy", "distance"), ("Effective reference energy / agent", "Configuration entropy (nats)", "Macrostate distance")):
            axis.plot(group["sweep"], group[metric], label=label, color=color, ls="-" if disruption == "field_reversal" else "--")
            _phase_background(axis)
            axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Sweep")
    _save(fig, "figure03_v14_quench_time_series", source, result, catalog, "Show V14 quench and counter-quench response in complementary coordinates", "cluster-mean time-resolved observables", "main", "Field reversal drives energy, entropy, and distance pulses", "Derived V14 trajectories; correction does not alter raw choices")


def _v14_recovery(repository: Path, result: Path, catalog: List[Dict[str, object]]) -> None:
    macro = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/macrostate_trajectories.csv")
    recovery = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/quench_recovery.csv")
    selected = macro[macro["disruption"] == "field_reversal"][["cluster_id", "sweep", "phase", "macrostate_distance", "training_nominal_threshold_95"]].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for cluster, group in selected.groupby("cluster_id"):
        ax.plot(group["sweep"], group["macrostate_distance"], lw=1.3, alpha=0.75, label=cluster)
    _phase_background(ax)
    ax.set_xlabel("Sweep")
    ax.set_ylabel("LOCO macrostate distance")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    _save(fig, "figure04_v14_cluster_recovery", selected, result, catalog, "Expose V14 cluster heterogeneity and protocol-consistent recovery", "cluster trajectories with training-only thresholds", "main", "All recovery statements use held-out-cluster-excluded thresholds", "Distance scale is metric-specific; historical H3 sign test is invalid")


def _v14_audit(repository: Path, result: Path, catalog: List[Dict[str, object]]) -> None:
    info = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/information_estimator_contrast_summary.csv")
    perm = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/representation_permutation_summary.csv")
    info = info[info["metric"].isin(("raw_total_correlation", "adjusted_total_correlation", "normalized_total_correlation"))].copy()
    info["panel"] = "dependence"
    perm = perm.copy()
    perm["panel"] = "permutation"
    source = pd.concat([info, perm], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for metric, group in info.groupby("metric"):
        axes[0].plot(group["window_sweeps"], group["estimate"], marker="o", label=metric.replace("_", " "))
        axes[0].fill_between(group["window_sweeps"], group["ci_low"], group["ci_high"], alpha=0.15)
    axes[0].axhline(0, color="#555555", lw=1, ls="--")
    axes[0].set_xlabel("Rolling window (sweeps)")
    axes[0].set_ylabel("Field - nominal dependence")
    axes[0].legend(frameon=False, fontsize=7.5)
    shown = perm[perm["metric"].isin(("full_statmech_balanced_accuracy", "full_minus_order_only_balanced_accuracy"))]
    x = np.arange(len(shown))
    axes[1].vlines(x, shown["null_q025"], shown["null_q975"], color="#777777", lw=3, label="Permutation 95% interval")
    axes[1].scatter(x, shown["observed"], color="#0072B2", marker="o", label="Observed")
    axes[1].set_xticks(x, ["Full accuracy", "Full - order"], rotation=15)
    axes[1].set_ylabel("Observed (null 95% interval)")
    axes[1].legend(frameon=False, fontsize=7.5)
    _save(fig, "figure05_v14_delayed_audit", source, result, catalog, "Report delayed prespecified dependence and permutation audits", "window sensitivity and cluster-preserving nulls", "supplement", "Raw and bias-adjusted dependence plus full-pipeline permutation results are explicit", "Completed after formal outcomes because of implementation omissions")


def _v15_quench(result: Path, catalog: List[Dict[str, object]]) -> None:
    frame = pd.read_csv(result / "tables/macrostate_trajectories.csv")
    source = frame[frame["condition"].isin(("nominal_markovized", "field_markovized"))][["model_key", "cluster_id", "condition", "sweep", "phase", "macrostate_distance", "reference_energy_per_agent", "configuration_entropy"]].copy()
    aggregate = source.groupby(["model_key", "condition", "sweep", "phase"], as_index=False).agg(distance=("macrostate_distance", "mean"), distance_sd=("macrostate_distance", "std"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharey=False)
    for axis, model in zip(axes, ("qwen", "granite")):
        for condition, group in aggregate[aggregate["model_key"] == model].groupby("condition"):
            axis.plot(group["sweep"], group["distance"], color=PALETTE[condition], marker=None, ls="-" if condition == "field_markovized" else "--", label=condition.replace("_", " "))
            axis.fill_between(group["sweep"], np.maximum(0, group["distance"]-group["distance_sd"]), group["distance"]+group["distance_sd"], color=PALETTE[condition], alpha=0.12)
        _phase_background(axis)
        axis.set_title(model.title())
        axis.set_xlabel("Sweep")
        axis.set_ylabel("LOCO macrostate distance")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "figure06_cross_model_quench", source, result, catalog, "Compare matched field quench response across model families", "model-specific LOCO macrostate distance", "main", "Independent-model replication is assessed without cross-model scale pooling", "Distances are fitted within model")


def _v15_memory(result: Path, catalog: List[Dict[str, object]]) -> None:
    panels = pd.read_csv(result / "tables/panel_statistics.csv")
    rows = []
    for (model, cluster), group in panels.groupby(["model_key", "cluster_id"]):
        lookup = group.set_index("condition")["adjusted_pathwise_irreversibility_nats_per_update"]
        rows.append({"model_key": model, "cluster_id": cluster, "persistent_minus_markovized": float(lookup["field_persistent"]-lookup["field_markovized"]), "persistent_minus_scrambled": float(lookup["field_persistent"]-lookup["field_scrambled"])})
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    offsets = {"qwen": -0.08, "granite": 0.08}
    for model, group in source.groupby("model_key"):
        for index, contrast in enumerate(("persistent_minus_markovized", "persistent_minus_scrambled")):
            x = np.full(len(group), index + offsets[model])
            ax.scatter(x, group[contrast], color=PALETTE[model], marker=MARKERS[model], alpha=0.85, label=model.title() if index == 0 else None)
            ax.plot([index+offsets[model]-0.05, index+offsets[model]+0.05], [group[contrast].mean()]*2, color=PALETTE[model], lw=3)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks((0, 1), ("Persistent - Markovized", "Persistent - scrambled"))
    ax.set_ylabel("Adjusted path divergence (nats/update)")
    ax.legend(frameon=False)
    _save(fig, "figure07_v15_memory_controls", source, result, catalog, "Test genuine memory against Markovized and prompt-matched scrambled history", "paired cluster differences", "main", "Separates historical content from prompt length and format", "Coarse-grained temporal asymmetry, not exact entropy production")


def _irreversibility_sensitivity(result: Path, catalog: List[Dict[str, object]]) -> None:
    frame = pd.read_csv(result / "tables/irreversibility_sensitivity.csv")
    source = frame.copy()
    selected = frame[np.isclose(frame["pseudocount"], 0.5)].copy()
    aggregate = selected.groupby(["model_key", "condition", "block_length"], as_index=False).agg(adjusted=("adjusted_irreversibility_nats_per_update", "mean"), floor=("shuffle_floor_nats_per_update", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharey=True)
    for axis, model in zip(axes, ("qwen", "granite")):
        for condition, group in aggregate[aggregate["model_key"] == model].groupby("condition"):
            axis.plot(group["block_length"], group["adjusted"], marker=MARKERS[condition], color=PALETTE[condition], label=condition.replace("field_", ""))
        axis.axhline(0, color="#555555", lw=1, ls="--")
        axis.set_title(model.title())
        axis.set_xlabel("Block length")
        axis.set_ylabel("Adjusted divergence (nats/update)")
    axes[0].legend(frameon=False, fontsize=7.5)
    _save(fig, "figure08_path_reversal_sensitivity", source, result, catalog, "Audit block length, pseudocount, and shuffled bias floor", "bias-adjusted block reversal divergence", "supplement", "Memory contrasts can be checked against estimator choices", "Observable coarse-graining and finite length remain limiting")


def _effects(result: Path, catalog: List[Dict[str, object]]) -> None:
    source = pd.read_csv(result / "tables/hypothesis_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    distance = source[source["unit"] == "distance_units"]
    information = source[source["unit"] == "nats_per_attempted_update"]
    for axis, frame, ylabel in ((axes[0], distance, "Distance units"), (axes[1], information, "Nats / attempted update")):
        y = np.arange(len(frame))[::-1]
        axis.errorbar(frame["estimate"], y, xerr=[frame["estimate"]-frame["ci_low"], frame["ci_high"]-frame["estimate"]], fmt="o", capsize=3, color="#0072B2")
        axis.axvline(0, color="#555555", lw=1, ls="--")
        axis.set_yticks(y, frame["hypothesis"])
        axis.set_xlabel(ylabel)
    _save(fig, "figure09_confirmatory_effects", source, result, catalog, "Show formal effects without mixing incompatible units", "H1-H4 cluster-level effects and 95% intervals", "main", "Formal dispositions remain visible whether positive, null, or mixed", "Separate axes are required for distance and information units")


def _surrogate(result: Path, catalog: List[Dict[str, object]]) -> None:
    source = pd.read_csv(result / "tables/v14_direct_surrogate_quench_trajectories.csv")
    selected = source[source["disruption"] == "field_reversal"]
    aggregate = selected.groupby(["source", "sweep", "phase"], as_index=False).agg(belief=("belief_magnetization", "mean"), action=("action_magnetization", "mean"), energy=("reference_energy_per_agent", "mean"), entropy=("configuration_entropy", "mean"), response=("shared_response_distance", "mean"))
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.0), sharex=True)
    metrics = (("belief", "Belief magnetization"), ("action", "Action magnetization"), ("energy", "Reference energy / agent"), ("entropy", "Configuration entropy"), ("response", "Shared response distance"))
    for axis, (metric, label) in zip(axes.flat, metrics):
        for source_name, group in aggregate.groupby("source"):
            key = "direct" if source_name == "Direct Qwen" else "surrogate"
            axis.plot(group["sweep"], group[metric], color=PALETTE[key], ls="-" if key == "direct" else "--", label=source_name)
        _phase_background(axis)
        axis.set_ylabel(label)
    axes.flat[-1].axis("off")
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Sweep")
    _save(fig, "figure10_direct_surrogate_quench", source, result, catalog, "Evaluate a V13-fitted kinetic closure out of sample on V14 quench paths", "time-resolved shared observables", "main", "Surrogate successes and failures are compared without quench refitting", "The surrogate uses Qwen microscopic data and is not a cross-model replacement")


def _surrogate_sizes(result: Path, catalog: List[Dict[str, object]]) -> None:
    source = pd.read_csv(result / "tables/surrogate_size_quench.csv")
    disruption = source[(source["disruption"] == "field_reversal") & (source["phase"] == "disruption")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0))
    for axis, metric, label in zip(axes, ("belief_magnetization_mean", "configuration_entropy_mean", "susceptibility_mean"), ("Belief magnetization", "Configuration entropy", "Susceptibility")):
        for size, group in disruption.groupby("n_agents"):
            axis.plot(group["sweep"], group[metric], label="N=%d" % size)
        axis.set_xlabel("Sweep")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7.5)
    _save(fig, "figure11_surrogate_size_sensitivity", source, result, catalog, "Place direct N=16 anchors in a denser inexpensive effective-model size context", "CPU kinetic-surrogate quench response", "supplement", "Effective-model size trends are explicit comparison results", "Not direct-LLM finite-size scaling")


def _prompt_control(result: Path, catalog: List[Dict[str, object]]) -> None:
    source = pd.read_csv(result / "tables/memory_prompt_balance.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for model, group in source.groupby("model_key"):
        ax.scatter(group["persistent_mean_prompt_tokens"], group["scrambled_mean_prompt_tokens"], color=PALETTE[model], marker=MARKERS[model], label=model.title())
    bounds = [float(min(source[["persistent_mean_prompt_tokens", "scrambled_mean_prompt_tokens"]].min())), float(max(source[["persistent_mean_prompt_tokens", "scrambled_mean_prompt_tokens"]].max()))]
    ax.plot(bounds, bounds, color="#555555", ls="--", lw=1)
    ax.set_xlabel("Persistent-history prompt tokens")
    ax.set_ylabel("Scrambled-history prompt tokens")
    ax.legend(frameon=False)
    _save(fig, "figure12_memory_prompt_balance", source, result, catalog, "Verify the scrambled-history prompt-length control", "per-cluster mean token counts", "supplement", "Prompt length and format are approximately matched", "Semantic content cannot be exactly token-matched turn by turn")


def generate_figures(repository: Path) -> Dict[str, object]:
    _style()
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    for directory in (result / "figures/pdf", result / "figures/source_data"):
        directory.mkdir(parents=True, exist_ok=True)
    catalog: List[Dict[str, object]] = []
    _architecture(result, catalog)
    _memory_replication(repository, result, catalog)
    _v14_quench(repository, result, catalog)
    _v14_recovery(repository, result, catalog)
    _v14_audit(repository, result, catalog)
    _v15_quench(result, catalog)
    _v15_memory(result, catalog)
    _irreversibility_sensitivity(result, catalog)
    _effects(result, catalog)
    _surrogate(result, catalog)
    _surrogate_sizes(result, catalog)
    _prompt_control(result, catalog)
    atomic_csv(catalog, result / "figures/figure_catalog.csv")
    summary = {
        "generated_at": utc_now(),
        "figure_count": len(catalog),
        "pdf_count": len(list((result / "figures/pdf").glob("*.pdf"))),
        "source_data_count": len(list((result / "figures/source_data").glob("*.csv"))),
        "catalog_sha256": sha256_file(result / "figures/figure_catalog.csv"),
    }
    atomic_json(summary, result / "reproducibility/figure_generation.json")
    return summary


__all__ = ["generate_figures"]
