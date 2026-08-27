"""Compact, source-backed vector figures for the V14 audit and V15 study."""

from __future__ import annotations

from copy import copy
import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
            # Figures are authored at about 7.1 inches and normally placed at
            # 0.9--0.95 text width.  These source sizes retain approximately
            # nine-point or larger type after manuscript scaling.
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
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


def _validate_text_containment(
    figure: plt.Figure,
    labeled_boxes: Sequence[Tuple[str, FancyBboxPatch, plt.Text]],
    padding_points: float = 2.5,
) -> None:
    """Fail figure generation when a rendered label exceeds its box interior."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding_pixels = padding_points * figure.dpi / 72.0
    violations: List[str] = []
    for component, patch, label in labeled_boxes:
        box_bounds = patch.get_window_extent(renderer=renderer)
        text_bounds = label.get_window_extent(renderer=renderer)
        contained = (
            text_bounds.x0 >= box_bounds.x0 + padding_pixels
            and text_bounds.x1 <= box_bounds.x1 - padding_pixels
            and text_bounds.y0 >= box_bounds.y0 + padding_pixels
            and text_bounds.y1 <= box_bounds.y1 - padding_pixels
        )
        if not contained:
            violations.append(component)
    if violations:
        raise AssertionError(
            "architecture labels exceed padded box interiors: %s"
            % ", ".join(violations)
        )


def _architecture(result: Path, catalog: List[Dict[str, object]]) -> None:
    rows = pd.DataFrame(
        [
            ("private_observation", 0.120, 0.720, "local", 0.190, 0.105),
            ("bounded_memory", 0.120, 0.540, "local", 0.190, 0.105),
            ("delivered_inbox", 0.120, 0.360, "local", 0.190, 0.105),
            ("LLM_local_transition", 0.360, 0.540, "agent", 0.200, 0.140),
            ("belief_action_packet", 0.600, 0.680, "observable", 0.180, 0.120),
            ("typed_local_action", 0.600, 0.400, "observable", 0.180, 0.120),
            ("delivery_graph", 0.860, 0.680, "network", 0.165, 0.120),
            ("environment", 0.860, 0.400, "network", 0.165, 0.120),
            ("observable_projection_Y", 0.390, 0.130, "evaluator", 0.245, 0.135),
            ("rolling_macrostate_Z", 0.730, 0.130, "evaluator", 0.235, 0.135),
        ],
        columns=("component", "x", "y", "boundary", "width", "height"),
    )
    fig, ax = plt.subplots(figsize=(7.1, 4.45))
    colors = {
        "local": "#D7EFF9",
        "agent": "#D9F0E8",
        "observable": "#FFF0C9",
        "network": "#F3DDEB",
        "evaluator": "#E7E7E7",
    }
    labels = {
        "private_observation": "Private\nobservation",
        "bounded_memory": "Bounded\nmemory",
        "delivered_inbox": "Delivered\ninbox",
        "LLM_local_transition": "Local LLM\ntransition",
        "belief_action_packet": "Belief/action\npacket",
        "typed_local_action": "Typed local\naction",
        "delivery_graph": "Delivery\ngraph",
        "environment": "Environment",
        "observable_projection_Y": "Observable\nprojection $Y_t$",
        "rolling_macrostate_Z": "Rolling\nmacrostate $Z_t$",
    }
    patches: Dict[str, FancyBboxPatch] = {}
    positions: Dict[str, Tuple[float, float]] = {}
    labeled_boxes: List[Tuple[str, FancyBboxPatch, plt.Text]] = []
    for row in rows.itertuples():
        patch = FancyBboxPatch(
            (row.x - row.width / 2.0, row.y - row.height / 2.0),
            row.width,
            row.height,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor=colors[row.boundary],
            edgecolor="#777777",
            linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(patch)
        label = ax.text(
            row.x,
            row.y,
            labels[row.component],
            ha="center",
            va="center",
            fontsize=9.0,
            linespacing=1.12,
            zorder=4,
        )
        patches[row.component] = patch
        positions[row.component] = (row.x, row.y)
        labeled_boxes.append((row.component, patch, label))

    def arrow(source: str, target: str, rad: float = 0.0) -> None:
        connection = FancyArrowPatch(
            posA=positions[source],
            posB=positions[target],
            patchA=patches[source],
            patchB=patches[target],
            arrowstyle="-|>",
            mutation_scale=11.5,
            linewidth=1.2,
            color="#444444",
            connectionstyle="arc3,rad=%s" % rad,
            shrinkA=2.0,
            shrinkB=1.5,
            zorder=3,
        )
        ax.add_patch(connection)

    arrow("private_observation", "LLM_local_transition")
    arrow("bounded_memory", "LLM_local_transition")
    arrow("delivered_inbox", "LLM_local_transition")
    arrow("LLM_local_transition", "belief_action_packet")
    arrow("LLM_local_transition", "typed_local_action")
    arrow("belief_action_packet", "delivery_graph")
    arrow("typed_local_action", "environment")
    arrow("belief_action_packet", "observable_projection_Y", -0.08)
    arrow("typed_local_action", "observable_projection_Y", 0.08)
    arrow("observable_projection_Y", "rolling_macrostate_Z")
    ax.text(
        0.02,
        0.965,
        r"Augmented process $\Xi_t$ and measured projections",
        va="top",
        fontsize=11.0,
        weight="bold",
    )
    ax.text(
        0.02,
        0.885,
        r"$Y_t=\phi(\Xi_t)$; $Z_t=\psi(Y_{t-w+1:t})$ (evaluator-side)",
        va="top",
        fontsize=8.8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _validate_text_containment(fig, labeled_boxes)
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
        for axis, metric, ylabel in zip(
            axes,
            ("energy", "entropy", "distance"),
            ("Reference energy / agent", "Config. entropy (nats)", "LOCO macrostate distance"),
        ):
            axis.plot(group["sweep"], group[metric], label=label, color=color, ls="-" if disruption == "field_reversal" else "--")
            _phase_background(axis)
            axis.set_ylabel(ylabel, fontsize=9.5, labelpad=5)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Sweep")
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.08, top=0.99, hspace=0.20)
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
    ax.legend(ncol=3, frameon=False, fontsize=10.5)
    _save(fig, "figure04_v14_cluster_recovery", selected, result, catalog, "Expose V14 cluster heterogeneity and protocol-consistent recovery", "cluster trajectories with training-only thresholds", "main", "All recovery statements use held-out-cluster-excluded thresholds", "Distance scale is metric-specific; historical H3 sign test is invalid")


def _v14_audit(repository: Path, result: Path, catalog: List[Dict[str, object]]) -> None:
    info = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/information_estimator_contrast_summary.csv")
    perm = pd.read_csv(repository / "results/collective_agent_statmech_v14/tables/representation_permutation_summary.csv")
    # Keep quantities with the same units on this axis.  The V14 audit table
    # uses the observable-first naming convention; the former aliases matched
    # no rows and silently produced an empty panel.
    info = info[
        info["metric"].isin(
            (
                "total_correlation_raw",
                "total_correlation_null_mean",
                "total_correlation_bias_adjusted",
            )
        )
    ].copy()
    info["panel"] = "dependence"
    perm = perm.copy()
    perm["panel"] = "permutation"
    source = pd.concat([info, perm], ignore_index=True, sort=False)
    fig = plt.figure(figsize=(7.1, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), hspace=0.42, wspace=0.32)
    raw_axis = fig.add_subplot(grid[0, 0])
    adjusted_axis = fig.add_subplot(grid[0, 1])
    permutation_axis = fig.add_subplot(grid[1, :])
    raw = info[info["metric"].isin(("total_correlation_raw", "total_correlation_null_mean"))]
    adjusted = info[info["metric"] == "total_correlation_bias_adjusted"]
    for metric, group in raw.groupby("metric"):
        label = "Raw contrast" if metric.endswith("_raw") else "Marginal-shift null"
        raw_axis.plot(
            group["window_sweeps"],
            group["estimate"],
            marker="o",
            label=label,
        )
        raw_axis.fill_between(group["window_sweeps"], group["ci_low"], group["ci_high"], alpha=0.14)
    raw_axis.set_xlabel("Window (sweeps)")
    raw_axis.set_ylabel("TC contrast (nats)")
    raw_axis.legend(frameon=False, fontsize=7.5)
    adjusted_axis.plot(adjusted["window_sweeps"], adjusted["estimate"], marker="o", color="#009E73")
    adjusted_axis.fill_between(adjusted["window_sweeps"], adjusted["ci_low"], adjusted["ci_high"], color="#009E73", alpha=0.16)
    adjusted_axis.axhline(0, color="#555555", lw=1, ls="--")
    adjusted_axis.set_xlabel("Window (sweeps)")
    adjusted_axis.set_ylabel("Adjusted TC (nats)")
    shown = perm[perm["metric"].isin(("full_statmech_balanced_accuracy", "full_minus_order_only_balanced_accuracy"))]
    x = np.arange(len(shown))
    permutation_axis.vlines(x, shown["null_q025"], shown["null_q975"], color="#777777", lw=4, label="Permutation 95% interval")
    permutation_axis.scatter(x, shown["observed"], color="#0072B2", marker="o", zorder=3, label="Observed")
    permutation_axis.axhline(0, color="#BBBBBB", lw=0.8)
    permutation_axis.set_xticks(x, ["Full accuracy", "Full - order"])
    permutation_axis.set_ylabel("Accuracy / difference")
    permutation_axis.legend(frameon=False, ncol=2, loc="upper center")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.11, top=0.99)
    _save(fig, "figure05_v14_delayed_audit", source, result, catalog, "Report delayed prespecified dependence and permutation audits", "window sensitivity and cluster-preserving nulls", "supplement", "Raw and bias-adjusted dependence plus full-pipeline permutation results are explicit", "Completed after formal outcomes because of implementation omissions")


def _v15_quench(result: Path, catalog: List[Dict[str, object]]) -> None:
    frame = pd.read_csv(result / "tables/macrostate_trajectories.csv")
    source = frame[frame["condition"].isin(("nominal_markovized", "field_markovized"))][["model_key", "cluster_id", "condition", "sweep", "phase", "macrostate_distance", "reference_energy_per_agent", "configuration_entropy"]].copy()
    aggregate = source.groupby(["model_key", "condition", "sweep", "phase"], as_index=False).agg(distance=("macrostate_distance", "mean"), distance_sd=("macrostate_distance", "std"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=False)
    handles = []
    labels = []
    for axis, model in zip(axes, ("qwen", "granite")):
        for condition, group in aggregate[aggregate["model_key"] == model].groupby("condition"):
            line, = axis.plot(group["sweep"], group["distance"], color=PALETTE[condition], marker=None, ls="-" if condition == "field_markovized" else "--", label=condition.replace("_", " "))
            axis.fill_between(group["sweep"], np.maximum(0, group["distance"]-group["distance_sd"]), group["distance"]+group["distance_sd"], color=PALETTE[condition], alpha=0.12)
            if model == "qwen":
                handles.append(line)
                labels.append(condition.replace("_", " "))
        _phase_background(axis)
        axis.set_title(model.title())
        axis.set_xlabel("Sweep")
        axis.set_ylabel("LOCO macrostate distance")
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.83, wspace=0.34)
    _save(fig, "figure06_cross_model_quench", source, result, catalog, "Compare matched field quench response across model families", "model-specific LOCO macrostate distance", "main", "Independent-model replication is assessed without cross-model scale pooling", "Distances are fitted within model")


def _v15_memory(result: Path, catalog: List[Dict[str, object]]) -> None:
    panels = pd.read_csv(result / "tables/panel_statistics.csv")
    rows = []
    for (model, cluster), group in panels.groupby(["model_key", "cluster_id"]):
        lookup = group.set_index("condition")["adjusted_pathwise_irreversibility_nats_per_update"]
        rows.append({"model_key": model, "cluster_id": cluster, "persistent_minus_markovized": float(lookup["field_persistent"]-lookup["field_markovized"]), "persistent_minus_scrambled": float(lookup["field_persistent"]-lookup["field_scrambled"])})
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    offsets = {"qwen": -0.08, "granite": 0.08}
    for model, group in source.groupby("model_key"):
        for index, contrast in enumerate(("persistent_minus_markovized", "persistent_minus_scrambled")):
            x = np.full(len(group), index + offsets[model])
            ax.scatter(x, group[contrast], color=PALETTE[model], marker=MARKERS[model], alpha=0.85, label=model.title() if index == 0 else None)
            ax.plot([index+offsets[model]-0.05, index+offsets[model]+0.05], [group[contrast].mean()]*2, color=PALETTE[model], lw=3)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks((0, 1), ("Persistent - Markovized", "Persistent - scrambled"))
    ax.set_ylabel("Adjusted path divergence\n(nats / update)", fontsize=10.5)
    ax.legend(frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.25, top=0.98)
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
    axes[0].legend(frameon=False, fontsize=10.5)
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
    trajectories = pd.read_csv(
        result / "tables/v14_direct_surrogate_quench_trajectories.csv"
    )
    selected = trajectories[trajectories["disruption"] == "field_reversal"]
    metric_columns = {
        "belief": "belief_magnetization",
        "action": "action_magnetization",
        "overlap": "belief_action_overlap",
        "energy": "reference_energy_per_agent",
        "entropy": "configuration_entropy",
        "susceptibility": "belief_susceptibility",
        "correlation_time": "integrated_correlation_time",
        "response": "shared_response_distance",
    }
    rng = np.random.default_rng(15158310)
    aggregate_rows: List[Dict[str, object]] = []
    for (source_name, sweep, phase), group in selected.groupby(
        ["source", "sweep", "phase"], sort=True
    ):
        row: Dict[str, object] = {
            "source": source_name,
            "sweep": int(sweep),
            "phase": phase,
            "independent_clusters": int(group["cluster_id"].nunique()),
            "bootstrap_replicates": 10000,
        }
        for metric, column in metric_columns.items():
            values = group[column].to_numpy(float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                estimate = ci_low = ci_high = float("nan")
            else:
                indices = rng.integers(
                    0, finite.size, size=(10000, finite.size), endpoint=False
                )
                bootstrap = finite[indices].mean(axis=1)
                estimate = float(np.mean(finite))
                ci_low = float(np.quantile(bootstrap, 0.025))
                ci_high = float(np.quantile(bootstrap, 0.975))
            row[metric] = estimate
            row[metric + "_ci_low"] = ci_low
            row[metric + "_ci_high"] = ci_high
        aggregate_rows.append(row)
    source = pd.DataFrame(aggregate_rows)
    aggregate = source
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.9), sharex=True)
    metrics = (
        ("belief", "Belief magnetization"),
        ("action", "Action magnetization"),
        ("overlap", "Belief--action overlap"),
        ("energy", "Reference energy / agent"),
        ("entropy", "Configuration entropy"),
        ("susceptibility", "Belief susceptibility"),
        ("correlation_time", "Correlation-time estimate"),
        ("response", "Shared response distance"),
    )
    for axis, (metric, label) in zip(axes.flat, metrics):
        for source_name, group in aggregate.groupby("source"):
            key = "direct" if source_name == "Direct Qwen" else "surrogate"
            axis.plot(group["sweep"], group[metric], color=PALETTE[key], ls="-" if key == "direct" else "--", label=source_name)
            axis.fill_between(
                group["sweep"],
                group[metric + "_ci_low"],
                group[metric + "_ci_high"],
                color=PALETTE[key],
                alpha=0.10,
                lw=0,
            )
        _phase_background(axis)
        axis.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_axis = fig.add_axes([0.25, 0.965, 0.50, 0.025])
    legend_axis.axis("off")
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
    )
    for label, axis in zip("abcdefgh", axes.flat):
        axis.text(
            0.0,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=12,
            weight="bold",
        )
    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Sweep")
    fig.subplots_adjust(top=0.925, hspace=0.34, wspace=0.35)
    _save(fig, "figure10_direct_surrogate_quench", source, result, catalog, "Evaluate a V13-fitted kinetic closure out of sample on V14 quench paths", "time-resolved shared observables", "main", "Surrogate successes and failures are compared without quench refitting", "The surrogate uses Qwen microscopic data and is not a cross-model replacement")


def _surrogate_sizes(result: Path, catalog: List[Dict[str, object]]) -> None:
    source = pd.read_csv(result / "tables/surrogate_size_quench.csv")
    disruption = source[(source["disruption"] == "field_reversal") & (source["phase"] == "disruption")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.15))
    handles = []
    legend_labels = []
    for axis, metric, label in zip(axes, ("belief_magnetization_mean", "configuration_entropy_mean", "susceptibility_mean"), ("Belief magnetization", "Configuration entropy", "Susceptibility")):
        for size, group in disruption.groupby("n_agents"):
            line, = axis.plot(group["sweep"], group[metric], label="N=%d" % size)
            if axis is axes[0]:
                handles.append(line)
                legend_labels.append("N=%d" % size)
        axis.set_xlabel("Sweep")
        axis.set_title(label, fontsize=10.5)
    fig.legend(handles, legend_labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=9.0)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.78, wspace=0.36)
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


def _collective_correlations(result: Path, catalog: List[Dict[str, object]]) -> None:
    matrices = pd.read_csv(result / "tables/connected_correlation_matrix_means.csv")
    profiles = pd.read_csv(result / "tables/connected_correlation_profile_summary.csv")
    condition = "field_persistent"
    selected_matrices = matrices[
        (matrices["condition"] == condition) & (matrices["phase"] == "disruption")
    ].copy()
    selected_profiles = profiles[profiles["condition"] == condition].copy()
    selected_matrices["source_type"] = "mean_connected_matrix"
    selected_profiles["source_type"] = "graph_distance_profile"
    source = pd.concat([selected_matrices, selected_profiles], ignore_index=True, sort=False)

    fig = plt.figure(figsize=(7.15, 6.25))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), hspace=0.34, wspace=0.28)
    heat_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    profile_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
    off_diagonal = selected_matrices[
        selected_matrices["agent_i"] != selected_matrices["agent_j"]
    ]
    extrema = off_diagonal["connected_correlation"].abs().max()
    limit = max(float(extrema), 1.0e-6)
    # ``Colormap.copy`` is unavailable in older Matplotlib releases used by
    # some repository-test hosts.  A standard shallow copy is sufficient
    # because only this figure's bad-value color is changed.
    correlation_cmap = copy(plt.get_cmap("RdBu_r"))
    correlation_cmap.set_bad("#E6E6E6")
    mesh = None
    for model_index, model in enumerate(("qwen", "granite")):
        matrix_rows = selected_matrices[selected_matrices["model_key"] == model]
        matrix = matrix_rows.pivot(
            index="agent_i", columns="agent_j", values="connected_correlation"
        ).sort_index().sort_index(axis=1)
        axis = heat_axes[model_index]
        matrix_values = matrix.to_numpy(float)
        np.fill_diagonal(matrix_values, np.nan)
        mesh = axis.pcolormesh(
            np.arange(matrix.shape[1] + 1) - 0.5,
            np.arange(matrix.shape[0] + 1) - 0.5,
            matrix_values,
            cmap=correlation_cmap,
            vmin=-limit,
            vmax=limit,
            shading="flat",
        )
        axis.axhline(7.5, color="#222222", lw=1.0)
        axis.axvline(7.5, color="#222222", lw=1.0)
        axis.set_title("%s: quench window" % model.title())
        axis.set_xlabel("Agent $j$")
        axis.set_ylabel("Agent $i$")
        axis.set_xticks((0, 4, 8, 12, 15))
        axis.set_yticks((0, 4, 8, 12, 15))

        profile_axis = profile_axes[model_index]
        model_profiles = selected_profiles[selected_profiles["model_key"] == model]
        phase_style = {
            "baseline": ("#0072B2", "o", "-"),
            "disruption": ("#D55E00", "s", "--"),
            "recovery": ("#009E73", "^", ":"),
        }
        for phase in ("baseline", "disruption", "recovery"):
            group = model_profiles[model_profiles["phase"] == phase].sort_values(
                "graph_distance"
            )
            color, marker, line = phase_style[phase]
            profile_axis.plot(
                group["graph_distance"],
                group["estimate"],
                color=color,
                marker=marker,
                ls=line,
                label=phase,
            )
            profile_axis.fill_between(
                group["graph_distance"],
                group["ci_low"],
                group["ci_high"],
                color=color,
                alpha=0.13,
                lw=0,
            )
        profile_axis.axhline(0, color="#555555", lw=0.9, ls="--")
        profile_axis.set_xlabel("Graph distance $d$")
        profile_axis.set_ylabel(r"Connected $C_b(d)$")
        profile_axis.set_xticks(sorted(model_profiles["graph_distance"].unique()))
        if model_index == 1:
            profile_axis.legend(
                frameon=False,
                ncol=1,
                loc="upper right",
                fontsize=10.5,
            )
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=heat_axes, location="right", shrink=0.82, pad=0.03)
        colorbar.set_label("Connected belief correlation")
    for label, axis in zip(("a", "b", "c", "d"), heat_axes + profile_axes):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=12, weight="bold")
    _save(
        fig,
        "figure13_graph_distance_correlations",
        source,
        result,
        catalog,
        "Relate microscopic belief covariance to modular graph distance",
        "per-trajectory connected correlations summarized across six clusters",
        "main_candidate",
        "Within persistent-history field trajectories, quench phases can reorganize spatial covariance beyond mean magnetization",
        "One finite N=16 reciprocal modular topology; node labels align only by predefined community across graph realizations; no correlation-length scaling",
    )


def _dynamical_persistence_shape(result: Path, catalog: List[Dict[str, object]]) -> None:
    curves = pd.read_csv(result / "tables/autocorrelation_curve_summary.csv")
    integrated = pd.read_csv(result / "tables/integrated_autocorrelation.csv")
    integrated_summary = pd.read_csv(
        result / "tables/integrated_autocorrelation_summary.csv"
    )
    binders = pd.read_csv(result / "tables/binder_cumulants.csv")
    binder_summary = pd.read_csv(result / "tables/binder_cumulant_summary.csv")
    binder_windows = pd.read_csv(
        result / "tables/binder_cumulant_sensitivity.csv"
    )
    binder_pooling = pd.read_csv(
        result / "tables/binder_cumulant_pooling_sensitivity.csv"
    )
    distributions = pd.read_csv(result / "tables/magnetization_distribution_summary.csv")
    plotted_conditions = (
        "nominal_markovized",
        "field_markovized",
        "field_persistent",
        "field_scrambled",
    )
    selected_curves = curves[
        curves["condition"].isin(plotted_conditions)
        & (curves["phase"] == "recovery")
    ].copy()
    selected_integrated = integrated[
        integrated["condition"].isin(plotted_conditions)
        & (integrated["phase"] == "recovery")
        & (integrated["is_primary"].astype(str).str.lower() == "true")
    ].copy()
    selected_integrated_summary = integrated_summary[
        integrated_summary["condition"].isin(plotted_conditions)
        & (integrated_summary["phase"] == "recovery")
        & (integrated_summary["is_primary"].astype(str).str.lower() == "true")
    ].copy()
    selected_binders = binders[binders["condition"] == "field_markovized"].copy()
    selected_binder_summary = binder_summary[
        binder_summary["condition"] == "field_markovized"
    ].copy()
    selected_binder_windows = binder_windows[
        binder_windows["condition"] == "field_markovized"
    ].copy()
    selected_binder_pooling = binder_pooling[
        binder_pooling["condition"] == "field_markovized"
    ].copy()
    selected_distributions = distributions[
        distributions["condition"] == "field_persistent"
    ].copy()
    source_frames = []
    for name, frame in (
        ("autocorrelation", selected_curves),
        ("integrated_autocorrelation", selected_integrated),
        ("integrated_autocorrelation_summary", selected_integrated_summary),
        ("binder", selected_binders),
        ("binder_summary", selected_binder_summary),
        ("binder_window_sensitivity", selected_binder_windows),
        ("binder_pooling_sensitivity", selected_binder_pooling),
        ("magnetization_distribution", selected_distributions),
    ):
        value = frame.copy()
        value["source_type"] = name
        source_frames.append(value)
    source = pd.concat(source_frames, ignore_index=True, sort=False)

    fig, axes = plt.subplots(3, 2, figsize=(7.15, 8.25))
    condition_labels = {
        "nominal_markovized": "Nominal",
        "field_markovized": "Field",
        "field_persistent": "Persistent",
        "field_scrambled": "Scrambled",
    }
    for model_index, model in enumerate(("qwen", "granite")):
        axis = axes[0, model_index]
        for condition in plotted_conditions:
            group = selected_curves[
                (selected_curves["model_key"] == model)
                & (selected_curves["condition"] == condition)
            ].sort_values("lag_updates")
            axis.plot(
                group["lag_sweeps"],
                group["estimate"],
                color=PALETTE[condition],
                marker=MARKERS[condition],
                markevery=8,
                ls={
                    "nominal_markovized": ":",
                    "field_markovized": "--",
                    "field_persistent": "-",
                    "field_scrambled": "-.",
                }[condition],
                label=condition_labels[condition],
            )
            axis.fill_between(
                group["lag_sweeps"],
                group["ci_low"],
                group["ci_high"],
                color=PALETTE[condition],
                alpha=0.10,
                lw=0,
            )
        axis.axhline(0, color="#555555", lw=0.8, ls=":")
        axis.set_title(model.title())
        axis.set_xlabel("Lag (sweeps)")
        axis.set_ylabel("Magnetization autocorrelation")
        if model_index == 0:
            axis.legend(frameon=False, ncol=2, loc="upper right", fontsize=10.5)

    tau_axis = axes[1, 0]
    x_positions = {
        condition: index for index, condition in enumerate(plotted_conditions)
    }
    model_offsets = {"qwen": -0.11, "granite": 0.11}
    for model in ("qwen", "granite"):
        for condition in plotted_conditions:
            group = selected_integrated[
                (selected_integrated["model_key"] == model)
                & (selected_integrated["condition"] == condition)
            ]
            x = x_positions[condition] + model_offsets[model]
            finite_group = group[
                np.isfinite(group["integrated_autocorrelation_time_updates"])
            ]
            tau_axis.scatter(
                np.full(len(finite_group), x),
                finite_group["integrated_autocorrelation_time_updates"],
                color=PALETTE[model],
                marker=MARKERS[model],
                alpha=0.82,
                label=model.title() if condition == plotted_conditions[0] else None,
            )
            if len(finite_group):
                summary = selected_integrated_summary[
                    (selected_integrated_summary["model_key"] == model)
                    & (selected_integrated_summary["condition"] == condition)
                ].iloc[0]
                if np.isfinite(float(summary["estimate"])):
                    tau_axis.errorbar(
                        x,
                        float(summary["estimate"]),
                        yerr=np.asarray(
                            [
                                [
                                    max(
                                        0.0,
                                        float(summary["estimate"] - summary["ci_low"]),
                                    )
                                ],
                                [
                                    max(
                                        0.0,
                                        float(summary["ci_high"] - summary["estimate"]),
                                    )
                                ],
                            ]
                        ),
                        color=PALETTE[model],
                        marker="_",
                        markersize=10,
                        capsize=3,
                        lw=1.5,
                    )
    tau_axis.set_xticks(range(len(plotted_conditions)))
    tau_axis.set_xticklabels(
        [condition_labels[value] for value in plotted_conditions], rotation=12
    )
    tau_axis.set_ylabel(r"Truncated $\tau_{\rm int}$ (updates)")
    tau_axis.set_title("Restoration-window persistence")
    tau_axis.legend(frameon=False, fontsize=10.5)

    binder_axis = axes[1, 1]
    phase_positions = {phase: index for index, phase in enumerate(("baseline", "disruption", "recovery"))}
    for model in ("qwen", "granite"):
        for phase in phase_positions:
            group = selected_binders[
                (selected_binders["model_key"] == model)
                & (selected_binders["phase"] == phase)
            ]
            x = phase_positions[phase] + model_offsets[model]
            finite_group = group[np.isfinite(group["binder_cumulant"])]
            binder_axis.scatter(
                np.full(len(finite_group), x),
                finite_group["binder_cumulant"],
                color=PALETTE[model],
                marker=MARKERS[model],
                alpha=0.82,
            )
            if len(finite_group):
                summary = selected_binder_summary[
                    (selected_binder_summary["model_key"] == model)
                    & (selected_binder_summary["phase"] == phase)
                ].iloc[0]
                if np.isfinite(float(summary["estimate"])):
                    binder_axis.errorbar(
                        x,
                        float(summary["estimate"]),
                        yerr=np.asarray(
                            [
                                [
                                    max(
                                        0.0,
                                        float(summary["estimate"] - summary["ci_low"]),
                                    )
                                ],
                                [
                                    max(
                                        0.0,
                                        float(summary["ci_high"] - summary["estimate"]),
                                    )
                                ],
                            ]
                        ),
                        color=PALETTE[model],
                        marker="_",
                        markersize=10,
                        capsize=3,
                        lw=1.5,
                    )
    binder_axis.set_xticks(range(3))
    binder_axis.set_xticklabels(("Baseline", "Quench", "Restoration"), rotation=12)
    binder_axis.set_ylabel(r"Binder $U_4$")
    binder_axis.set_title("Field-Markovized shape")

    phase_style = {
        "baseline": ("#0072B2", "-"),
        "disruption": ("#D55E00", "--"),
        "recovery": ("#009E73", ":"),
    }
    for model_index, model in enumerate(("qwen", "granite")):
        axis = axes[2, model_index]
        for phase in ("baseline", "disruption", "recovery"):
            group = selected_distributions[
                (selected_distributions["model_key"] == model)
                & (selected_distributions["phase"] == phase)
            ].sort_values("belief_magnetization")
            color, line = phase_style[phase]
            axis.plot(
                group["belief_magnetization"],
                group["estimate"],
                color=color,
                ls=line,
                marker="o",
                label=phase,
            )
            axis.fill_between(
                group["belief_magnetization"],
                group["ci_low"],
                group["ci_high"],
                color=color,
                alpha=0.10,
                lw=0,
            )
        axis.set_xlabel("Belief magnetization")
        axis.set_ylabel("Occupancy probability")
        axis.set_title("%s persistent-history" % model.title())
        axis.set_xlim(-1.03, 1.03)
        if model_index == 0:
            axis.legend(frameon=False, ncol=1, loc="upper left", fontsize=10.5)
    for label, axis in zip("abcdef", axes.flat):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=12, weight="bold")
    fig.tight_layout(h_pad=1.35, w_pad=1.0)
    _save(
        fig,
        "figure14_persistence_and_binder",
        source,
        result,
        catalog,
        "Connect temporal persistence to finite-size order-parameter shape",
        "phase-resolved autocorrelation, truncated correlation sums, Binder cumulants, and occupancies",
        "supplement_candidate",
        "Memory and quench response can alter persistence and distribution shape beyond mean order",
        "Short, potentially nonstationary N=16 phase windows; no equilibrium correlation-time, critical-slowing-down, or Binder-crossing claim",
    )


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
    _collective_correlations(result, catalog)
    _dynamical_persistence_shape(result, catalog)
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


def generate_figure1(repository: Path) -> Dict[str, object]:
    """Regenerate the architecture figure without touching quantitative figures."""

    _style()
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    for directory in (result / "figures/pdf", result / "figures/source_data"):
        directory.mkdir(parents=True, exist_ok=True)

    generated: List[Dict[str, object]] = []
    _architecture(result, generated)
    if len(generated) != 1:
        raise AssertionError("targeted architecture generation produced an unexpected catalog")

    catalog_path = result / "figures/figure_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    filename = generated[0]["filename"]
    matches = catalog["filename"] == filename
    if int(matches.sum()) != 1:
        raise ValueError("expected exactly one architecture row in %s" % catalog_path)
    replacement = generated[0]
    if set(replacement) != set(catalog.columns):
        raise ValueError("architecture catalog fields do not match the existing catalog")
    for column in catalog.columns:
        catalog.loc[matches, column] = replacement[column]
    atomic_csv(catalog, catalog_path)

    summary = {
        "generated_at": utc_now(),
        "figure_count": len(catalog),
        "pdf_count": len(list((result / "figures/pdf").glob("*.pdf"))),
        "source_data_count": len(list((result / "figures/source_data").glob("*.csv"))),
        "catalog_sha256": sha256_file(catalog_path),
    }
    atomic_json(summary, result / "reproducibility/figure_generation.json")
    return summary


__all__ = ["generate_figure1", "generate_figures"]
