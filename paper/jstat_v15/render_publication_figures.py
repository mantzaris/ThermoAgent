#!/usr/bin/env python3
"""Post-formal typography and layout fixes for the V15 publication PDFs.

The formal execution and analysis source is frozen. This renderer reads only
sealed aggregate tables and replaces presentation-layer PDFs whose first render
failed manual visual QA. It does not calculate or alter a scientific estimand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/collective_agent_statmech_v15"
PDF = RESULT / "figures/pdf"
SOURCE = RESULT / "figures/source_data"

PALETTE = {
    "qwen": "#0072B2",
    "granite": "#D55E00",
    "nominal_markovized": "#7F7F7F",
    "field_markovized": "#009E73",
    "direct": "#0072B2",
    "surrogate": "#D55E00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save(figure: plt.Figure, filename: str) -> None:
    figure.savefig(PDF / filename, format="pdf")
    plt.close(figure)


def phase(axis: plt.Axes) -> None:
    axis.axvspan(15.5, 30.5, color="#F0E442", alpha=0.14, lw=0)
    axis.axvline(15.5, color="#666666", ls="--", lw=1.0)
    axis.axvline(30.5, color="#666666", ls=":", lw=1.0)


def architecture() -> None:
    components = [
        ("private\nobservation", 0.04, 0.73, 0.17, 0.13, "#D7EFF9"),
        ("bounded\nmemory", 0.04, 0.45, 0.17, 0.13, "#D7EFF9"),
        ("delivered\ninbox", 0.04, 0.17, 0.17, 0.13, "#D7EFF9"),
        ("LLM local\ntransition", 0.32, 0.43, 0.20, 0.17, "#D9F0E8"),
        ("belief/action\npacket", 0.59, 0.64, 0.16, 0.14, "#FFF0C9"),
        ("typed local\naction", 0.59, 0.32, 0.16, 0.14, "#FFF0C9"),
        ("delivery\ngraph", 0.83, 0.64, 0.14, 0.14, "#F3DDEB"),
        ("environment", 0.83, 0.32, 0.14, 0.14, "#F3DDEB"),
        ("observable\nprojection $Y_t$", 0.48, 0.06, 0.22, 0.14, "#E7E7E7"),
        ("rolling\nmacrostate $Z_t$", 0.78, 0.06, 0.19, 0.14, "#E7E7E7"),
    ]
    fig, axis = plt.subplots(figsize=(7.1, 4.25))

    def arrow(start, end, rad=0.0):
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#444444",
                "lw": 1.2,
                "connectionstyle": "arc3,rad=%s" % rad,
            },
            zorder=1,
        )

    arrow((0.21, 0.795), (0.32, 0.555))
    arrow((0.21, 0.515), (0.32, 0.515))
    arrow((0.21, 0.235), (0.32, 0.475))
    arrow((0.52, 0.555), (0.59, 0.71))
    arrow((0.52, 0.475), (0.59, 0.39))
    arrow((0.75, 0.71), (0.83, 0.71))
    arrow((0.75, 0.39), (0.83, 0.39))
    arrow((0.64, 0.64), (0.57, 0.20), -0.10)
    arrow((0.64, 0.32), (0.63, 0.20), 0.08)
    arrow((0.70, 0.13), (0.78, 0.13))

    for label, x, y, width, height, color in components:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            facecolor=color,
            edgecolor="#777777",
            linewidth=1.0,
            zorder=2,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9, zorder=3)

    axis.text(0.02, 0.97, r"$\Xi_t$: complete augmented simulator state", va="top", fontsize=11, weight="bold")
    axis.text(0.02, 0.015, r"$Y_t=\phi(\Xi_t)$; $Z_t=\psi(Y_{t-w+1:t})$", va="bottom", fontsize=10.5)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    save(fig, "figure01_augmented_state_architecture.pdf")


def v14_quench() -> None:
    source = pd.read_csv(SOURCE / "figure03_v14_quench_time_series.csv")
    aggregate = source.groupby(["disruption", "sweep", "phase"], as_index=False).agg(
        energy=("reference_energy_per_agent", "mean"),
        entropy=("configuration_entropy", "mean"),
        distance=("macrostate_distance", "mean"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.7), sharex=True)
    labels = (("energy", "Reference energy\nper agent"), ("entropy", "Configuration entropy\n(nats)"), ("distance", "Macrostate distance"))
    for disruption, group in aggregate.groupby("disruption"):
        color = "#0072B2" if disruption == "nominal" else "#D55E00"
        for axis, (metric, ylabel) in zip(axes, labels):
            axis.plot(group["sweep"], group[metric], label=disruption.replace("_", " "), color=color, ls="-" if disruption == "field_reversal" else "--")
            phase(axis)
            axis.set_ylabel(ylabel, fontsize=9.5, labelpad=8)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Sweep")
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.08, top=0.99, hspace=0.20)
    save(fig, "figure03_v14_quench_time_series.pdf")


def delayed_audit() -> None:
    info = pd.read_csv(ROOT / "results/collective_agent_statmech_v14/tables/information_estimator_contrast_summary.csv")
    perm = pd.read_csv(ROOT / "results/collective_agent_statmech_v14/tables/representation_permutation_summary.csv")
    raw = info[info.metric.isin(("total_correlation_raw", "total_correlation_null_mean"))]
    adjusted = info[info.metric == "total_correlation_bias_adjusted"]
    shown = perm[perm.metric.isin(("full_statmech_balanced_accuracy", "full_minus_order_only_balanced_accuracy"))]
    fig = plt.figure(figsize=(7.1, 5.1))
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1, 1.05), hspace=0.42, wspace=0.32)
    left = fig.add_subplot(grid[0, 0])
    middle = fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, :])
    for metric, group in raw.groupby("metric"):
        label = "Raw contrast" if metric.endswith("_raw") else "Marginal-shift null"
        left.plot(group.window_sweeps, group.estimate, marker="o", label=label)
        left.fill_between(group.window_sweeps, group.ci_low, group.ci_high, alpha=0.14)
    left.set_xlabel("Window (sweeps)")
    left.set_ylabel("TC contrast (nats)")
    left.legend(frameon=False, fontsize=7.5)
    middle.plot(adjusted.window_sweeps, adjusted.estimate, marker="o", color="#009E73")
    middle.fill_between(adjusted.window_sweeps, adjusted.ci_low, adjusted.ci_high, color="#009E73", alpha=0.16)
    middle.axhline(0, color="#555555", ls="--", lw=1)
    middle.set_xlabel("Window (sweeps)")
    middle.set_ylabel("Adjusted TC (nats)")
    x = np.arange(len(shown))
    bottom.vlines(x, shown.null_q025, shown.null_q975, color="#777777", lw=4, label="Permutation 95% interval")
    bottom.scatter(x, shown.observed, color="#0072B2", marker="o", zorder=3, label="Observed")
    bottom.axhline(0, color="#BBBBBB", lw=0.8)
    bottom.set_xticks(x, ("Full accuracy", "Full - order"))
    bottom.set_ylabel("Accuracy / difference")
    bottom.legend(frameon=False, ncol=2, loc="upper center")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.11, top=0.99)
    save(fig, "figure05_v14_delayed_audit.pdf")


def cross_model_quench() -> None:
    source = pd.read_csv(SOURCE / "figure06_cross_model_quench.csv")
    aggregate = source.groupby(["model_key", "condition", "sweep", "phase"], as_index=False).agg(
        distance=("macrostate_distance", "mean"), distance_sd=("macrostate_distance", "std")
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    handles = []
    labels = []
    for axis, model in zip(axes, ("qwen", "granite")):
        for condition, group in aggregate[aggregate.model_key == model].groupby("condition"):
            line, = axis.plot(group.sweep, group.distance, color=PALETTE[condition], ls="-" if condition == "field_markovized" else "--")
            axis.fill_between(group.sweep, np.maximum(0, group.distance - group.distance_sd), group.distance + group.distance_sd, color=PALETTE[condition], alpha=0.12)
            if model == "qwen":
                handles.append(line)
                labels.append(condition.replace("_", " "))
        phase(axis)
        axis.set_title(model.title())
        axis.set_xlabel("Sweep")
        axis.set_ylabel("LOCO macrostate distance")
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.83, wspace=0.30)
    save(fig, "figure06_cross_model_quench.pdf")


def direct_surrogate() -> None:
    source = pd.read_csv(SOURCE / "figure10_direct_surrogate_quench.csv")
    selected = source[source.disruption == "field_reversal"]
    aggregate = selected.groupby(["source", "sweep", "phase"], as_index=False).agg(
        belief=("belief_magnetization", "mean"),
        action=("action_magnetization", "mean"),
        energy=("reference_energy_per_agent", "mean"),
        entropy=("configuration_entropy", "mean"),
        response=("shared_response_distance", "mean"),
    )
    fig = plt.figure(figsize=(7.1, 6.7))
    grid = GridSpec(3, 2, figure=fig, height_ratios=(1, 1, 1.05), hspace=0.23, wspace=0.32)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :])]
    metrics = (("belief", "Belief magnetization"), ("action", "Action magnetization"), ("energy", "Reference energy / agent"), ("entropy", "Configuration entropy"), ("response", "Shared response distance"))
    for axis, (metric, label) in zip(axes, metrics):
        for source_name, group in aggregate.groupby("source"):
            key = "direct" if source_name == "Direct Qwen" else "surrogate"
            axis.plot(group.sweep, group[metric], color=PALETTE[key], ls="-" if key == "direct" else "--", label=source_name)
        phase(axis)
        axis.set_ylabel(label, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Sweep")
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.07, top=0.99)
    save(fig, "figure10_direct_surrogate_quench.pdf")


def surrogate_sizes() -> None:
    source = pd.read_csv(SOURCE / "figure11_surrogate_size_sensitivity.csv")
    disruption = source[(source.disruption == "field_reversal") & (source.phase == "disruption")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0))
    metrics = (("belief_magnetization_mean", "Belief $m$"), ("configuration_entropy_mean", "Configuration entropy"), ("susceptibility_mean", "Susceptibility"))
    for axis, (metric, label) in zip(axes, metrics):
        for size, group in disruption.groupby("n_agents"):
            axis.plot(group.sweep, group[metric], label="$N=%d$" % size)
        axis.set_xlabel("Sweep")
        axis.set_ylabel(label, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.98, wspace=0.45)
    save(fig, "figure11_surrogate_size_sensitivity.pdf")


def update_catalog() -> None:
    catalog_path = RESULT / "figures/figure_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    for index, row in catalog.iterrows():
        catalog.loc[index, "pdf_sha256"] = sha256(PDF / row.filename)
        catalog.loc[index, "source_sha256"] = sha256(RESULT / row.source_table)
    try:
        catalog.to_csv(catalog_path, index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5
        catalog.to_csv(catalog_path, index=False, line_terminator="\n")
    summary = {
        "figure_count": int(len(catalog)),
        "pdf_count": len(list(PDF.glob("*.pdf"))),
        "source_data_count": len(list(SOURCE.glob("*.csv"))),
        "catalog_sha256": sha256(catalog_path),
        "postformal_publication_renderer_sha256": sha256(Path(__file__)),
        "presentation_only": True,
    }
    target = RESULT / "reproducibility/figure_generation.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    style()
    architecture()
    v14_quench()
    delayed_audit()
    cross_model_quench()
    direct_surrogate()
    surrogate_sizes()
    update_catalog()


if __name__ == "__main__":
    main()
