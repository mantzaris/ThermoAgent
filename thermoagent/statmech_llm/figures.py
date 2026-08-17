"""Compact vector figures for the V10 analytical results."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from .workflow import _atomic_csv, _atomic_json, artifact_root, sha256_file, utc_now


COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#777777",
}


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pdf")
    fig.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(fig)
    os.replace(str(temporary), str(path))


def _architecture(results: Path) -> None:
    nodes = [
        ("private evidence", 0.05, 0.77, "local"),
        ("agent belief b_i", 0.25, 0.77, "agent"),
        ("agent action a_i", 0.45, 0.77, "agent"),
        ("local heat bath", 0.25, 0.28, "theory"),
        ("typed LLM decision", 0.70, 0.77, "llm"),
        ("private memory", 0.70, 0.48, "local"),
        ("inbox / outbox", 0.88, 0.77, "message"),
        ("environment transition", 0.70, 0.18, "environment"),
        ("evaluator-only scoring", 0.91, 0.18, "evaluator"),
    ]
    edges = [
        ("private evidence", "agent belief b_i", "authorized"),
        ("agent belief b_i", "agent action a_i", "K coupling"),
        ("agent belief b_i", "local heat bath", "analytical"),
        ("private evidence", "typed LLM decision", "authorized"),
        ("private memory", "typed LLM decision", "authorized"),
        ("inbox / outbox", "typed LLM decision", "delivered only"),
        ("typed LLM decision", "inbox / outbox", "directed message"),
        ("typed LLM decision", "environment transition", "validated tool"),
        ("environment transition", "evaluator-only scoring", "offline only"),
    ]
    _atomic_csv(
        [
            {"record": "node", "name": name, "x": x, "y": y, "kind": kind, "source": "protocol"}
            for name, x, y, kind in nodes
        ]
        + [
            {"record": "edge", "name": "%s -> %s" % (source, target), "x": "", "y": "", "kind": kind, "source": "protocol"}
            for source, target, kind in edges
        ],
        results / "figures/source_data/figure_01_architecture.csv",
    )
    def add_box(
        axis: plt.Axes,
        center: Tuple[float, float],
        size: Tuple[float, float],
        label: str,
        facecolor: str,
        fontsize: float = 8.8,
    ) -> None:
        x, y = center
        width, height = size
        axis.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.012",
                facecolor=facecolor,
                edgecolor=COLORS["black"],
                linewidth=0.9,
                zorder=2,
            )
        )
        axis.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=3)

    def add_arrow(
        axis: plt.Axes,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        curvature: float = 0.0,
        linestyle: str = "-",
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.3,
                linestyle=linestyle,
                color=COLORS["gray"],
                connectionstyle="arc3,rad=%.2f" % curvature,
                zorder=1,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.55))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    left, right = axes
    left.set_title("Analytical stochastic-agent\nreference", weight="bold", pad=8, fontsize=10.2)
    left.text(-0.02, 1.02, "a", transform=left.transAxes, fontsize=12, weight="bold")
    add_box(left, (0.16, 0.69), (0.25, 0.18), "private\nevidence", COLORS["sky"])
    add_box(left, (0.50, 0.69), (0.25, 0.18), r"belief  $b_i$", COLORS["green"])
    add_box(left, (0.84, 0.69), (0.25, 0.18), r"action  $a_i$", COLORS["green"])
    add_box(left, (0.50, 0.30), (0.27, 0.18), "local\nheat bath", COLORS["yellow"])
    add_arrow(left, (0.29, 0.69), (0.37, 0.69))
    add_arrow(left, (0.63, 0.69), (0.71, 0.69))
    add_arrow(left, (0.50, 0.60), (0.50, 0.40))
    left.text(0.67, 0.75, r"$K$", ha="center", fontsize=9.2)

    right.set_title("Independent LLM-agent\nrealization", weight="bold", pad=8, fontsize=10.2)
    right.text(-0.02, 1.02, "b", transform=right.transAxes, fontsize=12, weight="bold")
    add_box(right, (0.16, 0.72), (0.25, 0.18), "private\nevidence", COLORS["sky"])
    add_box(right, (0.16, 0.42), (0.25, 0.18), "private\nmemory", COLORS["sky"])
    add_box(right, (0.50, 0.58), (0.27, 0.20), "typed LLM\ndecision", COLORS["orange"])
    add_box(right, (0.84, 0.72), (0.25, 0.18), "inbox /\noutbox", COLORS["purple"])
    add_box(right, (0.50, 0.20), (0.29, 0.18), "environment\ntransition", COLORS["blue"], 8.1)
    add_box(right, (0.84, 0.20), (0.27, 0.18), "evaluator-only\nscoring", "#DDDDDD", 8.1)
    add_arrow(right, (0.29, 0.72), (0.38, 0.63))
    add_arrow(right, (0.29, 0.42), (0.37, 0.53))
    add_arrow(right, (0.71, 0.69), (0.63, 0.62), curvature=0.12)
    add_arrow(right, (0.62, 0.57), (0.72, 0.66), curvature=0.12)
    add_arrow(right, (0.50, 0.48), (0.50, 0.30))
    add_arrow(right, (0.65, 0.20), (0.71, 0.20), linestyle="--")
    fig.subplots_adjust(wspace=0.13)
    _save(fig, results / "figures/pdf/figure_01_architecture.pdf")


def _quadratic(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_02_quadratic_onset.csv")
    alpha = data["alpha"].to_numpy(float)
    observed = data["total_per_update_mean"].to_numpy(float)
    low = data["total_per_update_ci_low"].to_numpy(float)
    high = data["total_per_update_ci_high"].to_numpy(float)
    prediction = data["quadratic_prediction_mean"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes[0].fill_between(alpha, low, high, color=COLORS["sky"], alpha=0.32, label="95% graph-orientation CI")
    axes[0].plot(alpha, observed, "o-", color=COLORS["blue"], lw=1.8, ms=4.8, label="exact stationary EPR")
    axes[0].plot(alpha, prediction, "--", color=COLORS["vermillion"], lw=1.8, label=r"$\langle C\rangle\alpha^2$")
    axes[0].set(xlabel=r"nonreciprocity $\alpha$", ylabel="EPR (nats / attempted update)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontsize=12, weight="bold")
    positive = alpha > 0
    axes[1].plot(
        alpha[positive],
        observed[positive] / alpha[positive] ** 2,
        "o-",
        color=COLORS["green"],
        lw=1.6,
        label=r"exact $\sigma/\alpha^2$",
    )
    axes[1].axhline(data["coefficient_prediction_mean"].iloc[0], color=COLORS["vermillion"], ls="--", lw=1.6, label="perturbative C")
    axes[1].set_xscale("log")
    axes[1].set(xlabel=r"$\alpha$ (log scale)", ylabel=r"$\sigma/\alpha^2$")
    axes[1].legend(frameon=False)
    axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, results / "figures/pdf/figure_02_quadratic_onset.pdf")


def _coefficient(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_03_coefficient.csv")
    topologies = ["path", "ring", "star", "complete"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)
    for axis, coupling in zip(axes, [0.0, 0.4, 0.8]):
        part_k = data[np.isclose(data["belief_action_coupling"], coupling)]
        for topology, color in zip(topologies, colors):
            part = part_k[part_k["topology"] == topology].sort_values("temperature")
            axis.fill_between(
                part["temperature"].to_numpy(float),
                part["coefficient_per_update_ci_low"].to_numpy(float),
                part["coefficient_per_update_ci_high"].to_numpy(float),
                color=color,
                alpha=0.14,
            )
            axis.plot(
                part["temperature"],
                part["coefficient_per_update_mean"],
                marker="o",
                ms=3.5,
                lw=1.35,
                color=color,
                label=topology,
            )
        axis.set_title(r"belief--action $K=%.1f$" % coupling)
        axis.set_xlabel("decision temperature T")
    axes[0].set_ylabel(r"quadratic coefficient $C$ / update")
    axes[-1].legend(frameon=False, loc="upper right")
    for index, axis in enumerate(axes):
        axis.text(-0.18, 1.05, chr(ord("a") + index), transform=axis.transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.16)
    _save(fig, results / "figures/pdf/figure_03_coefficient.pdf")


def _scaling(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_04_size_scaling.csv")
    selected = data[
        np.isclose(data["temperature"], 1.65) & np.isclose(data["alpha"], 0.35)
    ].copy()
    topologies = ["ring", "small_world", "modular"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for topology, color in zip(topologies, colors):
        part = selected[selected["topology"] == topology].sort_values("n_agents")
        n = part["n_agents"].to_numpy(float)
        per_update = part["pathwise_irreversibility_per_update_mean"].to_numpy(float)
        low = part["pathwise_irreversibility_per_update_ci_low"].to_numpy(float)
        high = part["pathwise_irreversibility_per_update_ci_high"].to_numpy(float)
        axes[0].errorbar(n, per_update, yerr=[per_update - low, high - per_update], marker="o", lw=1.4, capsize=2.5, color=color, label=topology)
        total_sweep = 2.0 * n * per_update
        axes[1].plot(n, total_sweep, marker="s", lw=1.4, color=color, label=topology)
    axes[0].set(xlabel="agents N", ylabel="pathwise irreversibility / update")
    axes[1].set(xlabel="agents N", ylabel="pathwise irreversibility / sweep")
    for axis in axes:
        axis.set_xscale("log", base=2)
    axes[0].legend(frameon=False)
    axes[0].text(-0.14, 1.04, "a", transform=axes[0].transAxes, fontsize=12, weight="bold")
    axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, results / "figures/pdf/figure_04_size_scaling.pdf")


def _currents(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_05_probability_currents.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    reciprocal = data[np.isclose(data["alpha"], 0.0)]
    axes[0].axhline(0.0, color=COLORS["black"], lw=1)
    axes[0].scatter(reciprocal["rank"], reciprocal["current"], color=COLORS["blue"], s=28)
    axes[0].set(xlabel="ranked state-pair current", ylabel="stationary probability current", title=r"reciprocal $\alpha=0$")
    axes[0].text(0.5, 0.58, "all currents vanish\nwithin numerical precision", ha="center", transform=axes[0].transAxes)
    nonreciprocal = data[np.isclose(data["alpha"], 0.5)].sort_values("rank")
    signs = np.sign(nonreciprocal["current"].to_numpy(float))
    colors = [COLORS["vermillion"] if sign > 0 else COLORS["blue"] for sign in signs]
    axes[1].bar(nonreciprocal["rank"], nonreciprocal["absolute_current"], color=colors, width=0.78)
    axes[1].set(xlabel="ranked directed state edge", ylabel="absolute stationary current", title=r"nonreciprocal $\alpha=0.5$")
    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.06, chr(ord("a") + index), transform=axis.transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, results / "figures/pdf/figure_05_probability_currents.pdf")


def _temperature_surface(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_06_temperature_nonreciprocity.csv")
    selected = data[(data["n_agents"] == 64) & (data["topology"] == "small_world")]
    alphas = sorted(selected["alpha"].unique())
    colors = [COLORS["black"], COLORS["sky"], COLORS["orange"], COLORS["vermillion"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    for alpha, color in zip(alphas, colors):
        part = selected[np.isclose(selected["alpha"], alpha)].sort_values("temperature")
        axes[0].plot(part["temperature"], part["pathwise_irreversibility_per_update_mean"], marker="o", color=color, lw=1.5, label=r"$\alpha=%.2g$" % alpha)
        axes[1].plot(part["temperature"], part["pathwise_irreversibility_per_update_mean"] / max(alpha ** 2, 1.0), marker="s", color=color, lw=1.5)
    axes[0].set(xlabel="decision temperature T", ylabel="pathwise irreversibility / update")
    axes[1].set(xlabel="decision temperature T", ylabel=r"irreversibility / $\alpha^2$ (nonzero $\alpha$)")
    axes[0].legend(frameon=False)
    axes[0].text(-0.14, 1.04, "a", transform=axes[0].transAxes, fontsize=12, weight="bold")
    axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, results / "figures/pdf/figure_06_temperature_response.pdf")


def _applications(results: Path) -> None:
    records = [
        ("humanitarian", "depot", 0.13, 0.55, "resource"),
        ("humanitarian", "field team", 0.34, 0.76, "evidence"),
        ("humanitarian", "carrier", 0.55, 0.55, "action"),
        ("humanitarian", "clinic", 0.82, 0.76, "service"),
        ("humanitarian", "shelter", 0.82, 0.33, "service"),
        ("utility", "component operator", 0.15, 0.55, "resource"),
        ("utility", "telemetry relay", 0.38, 0.76, "evidence"),
        ("utility", "crew coordinator", 0.59, 0.55, "action"),
        ("utility", "critical load", 0.84, 0.76, "service"),
        ("utility", "safety monitor", 0.84, 0.33, "service"),
    ]
    _atomic_csv(
        [
            {"application": app, "node": node, "x": x, "y": y, "role_class": role, "status": "illustrative mapping; no formal dynamic LLM trajectory"}
            for app, node, x, y, role in records
        ],
        results / "figures/source_data/figure_07_application_mappings.csv",
    )
    edges = {
        "humanitarian": [("depot", "field team"), ("field team", "carrier"), ("carrier", "clinic"), ("carrier", "shelter")],
        "utility": [
            ("component operator", "telemetry relay"),
            ("telemetry relay", "crew coordinator"),
            ("crew coordinator", "critical load"),
            ("crew coordinator", "safety monitor"),
        ],
    }
    palette = {"resource": COLORS["green"], "evidence": COLORS["purple"], "action": COLORS["orange"], "service": COLORS["sky"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    for panel, (axis, application, title) in enumerate(
        zip(axes, ("humanitarian", "utility"), ("Humanitarian coordination", "Defensive utility restoration"))
    ):
        subset = [row for row in records if row[0] == application]
        positions = {node: (x, y) for _, node, x, y, _ in subset}
        for source, target in edges[application]:
            axis.add_patch(
                FancyArrowPatch(
                    positions[source],
                    positions[target],
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.3,
                    color=COLORS["gray"],
                    zorder=1,
                )
            )
        for _, node, x, y, role in subset:
            display = node.replace(" ", "\n") if len(node) > 12 else node
            axis.add_patch(
                FancyBboxPatch(
                    (x - 0.11, y - 0.075),
                    0.22,
                    0.15,
                    boxstyle="round,pad=0.012",
                    facecolor=palette[role],
                    edgecolor=COLORS["black"],
                    linewidth=0.9,
                    zorder=2,
                )
            )
            axis.text(x, y, display, ha="center", va="center", fontsize=8.1, zorder=3)
        axis.set_title(("a  " if panel == 0 else "b  ") + title, loc="left", weight="bold", pad=5)
        axis.set_xlim(0, 1)
        axis.set_ylim(0.12, 0.93)
        axis.axis("off")
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["green"], edgecolor="black", label="resource"),
            Patch(facecolor=COLORS["purple"], edgecolor="black", label="evidence"),
            Patch(facecolor=COLORS["orange"], edgecolor="black", label="action"),
            Patch(facecolor=COLORS["sky"], edgecolor="black", label="service/safety"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )
    fig.text(0.5, 0.015, "Illustrative mapping only; the formal dynamic Qwen stage was not unlocked.", ha="center", fontsize=9.2)
    fig.subplots_adjust(bottom=0.22, wspace=0.08)
    _save(fig, results / "figures/pdf/figure_07_application_mappings.pdf")


def _qwen_pilot(results: Path) -> None:
    data = pd.read_csv(results / "figures/source_data/figure_08_qwen_pilot.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25))
    evidence = data[data["panel"] == "private_evidence"]
    styles = {
        "left option first": (COLORS["blue"], "o", "-"),
        "right option first": (COLORS["orange"], "s", "--"),
    }
    for condition, part in evidence.groupby("condition", sort=True):
        part = part.sort_values("x")
        color, marker, linestyle = styles[str(condition)]
        mean = part["mean_right_choice"].to_numpy(float)
        axes[0].errorbar(
            part["x"],
            mean,
            yerr=[mean - part["wilson_ci_low"].to_numpy(float), part["wilson_ci_high"].to_numpy(float) - mean],
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2.5,
            label=str(condition),
        )
    axes[0].axhline(0.5, color=COLORS["gray"], lw=1, ls=":")
    axes[0].set(xlabel="controlled private-evidence field", ylabel="probability of plan_right", ylim=(-0.04, 1.04))
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(-0.15, 1.05, "a", transform=axes[0].transAxes, fontsize=12, weight="bold")

    message = data[data["panel"] == "delivered_message"]
    message_styles = {
        "prior left": (COLORS["vermillion"], "o", "-"),
        "prior right": (COLORS["green"], "s", "--"),
    }
    for condition, part in message.groupby("condition", sort=True):
        part = part.sort_values("x")
        color, marker, linestyle = message_styles[str(condition)]
        mean = part["mean_right_choice"].to_numpy(float)
        axes[1].errorbar(
            part["x"],
            mean,
            yerr=[mean - part["wilson_ci_low"].to_numpy(float), part["wilson_ci_high"].to_numpy(float) - mean],
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2.5,
            label=str(condition),
        )
    axes[1].set_xticks([-1, 1], ["supports left", "supports right"])
    axes[1].set(xlabel="new delivered peer message", ylabel="probability of plan_right", ylim=(-0.04, 1.04))
    axes[1].legend(
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.67),
    )
    axes[1].text(
        0.5,
        0.46,
        "message response " + r"$\Delta=0.00$" + "\n" + r"required $\Delta\geq0.20$",
        transform=axes[1].transAxes,
        fontsize=9,
        ha="center",
        va="center",
    )
    axes[1].text(-0.15, 1.05, "b", transform=axes[1].transAxes, fontsize=12, weight="bold")
    fig.suptitle("Qwen qualification pilots (development only; formal network study not unlocked)", fontsize=10.5)
    fig.subplots_adjust(wspace=0.34, top=0.84)
    _save(fig, results / "figures/pdf/figure_08_qwen_pilot.pdf")


def generate_figures(repository: Path) -> List[str]:
    _configure()
    results = repository / "results/llm_agent_entropy_v10"
    _architecture(results)
    _quadratic(results)
    _coefficient(results)
    _scaling(results)
    _currents(results)
    _temperature_surface(results)
    _applications(results)
    _qwen_pilot(results)
    return sorted(path.name for path in (results / "figures/pdf").glob("*.pdf"))


def validate_pdfs(repository: Path, manual_reviewed: bool = False) -> Dict[str, object]:
    results = repository / "results/llm_agent_entropy_v10"
    pdf_root = results / "figures/pdf"
    render_root = artifact_root() / "pdf_qa"
    render_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for path in sorted(pdf_root.glob("*.pdf")):
        document = fitz.open(path)
        if document.page_count < 1:
            raise RuntimeError("empty PDF: %s" % path)
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(300.0 / 72.0, 300.0 / 72.0), alpha=False)
        render = render_root / (path.stem + ".png")
        pixmap.save(render)
        extracted = "".join(document[index].get_text() for index in range(document.page_count))
        document.close()
        fonts = subprocess.run(
            ["pdffonts", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines) and all(" yes " in (" " + line + " ") for line in font_lines)
        if not embedded or len(extracted.strip()) < 20:
            raise RuntimeError("PDF font/text validation failed: %s" % path)
        records.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "page_count": 1,
                "fonts_embedded": embedded,
                "text_extractable": True,
                "render_dpi": 300,
                "render_path_external": str(render),
                "render_width": pixmap.width,
                "render_height": pixmap.height,
                "manual_original_size_review": "passed" if manual_reviewed else "pending",
                "manual_300_dpi_review": "passed" if manual_reviewed else "pending",
                "clipping_or_overlap": "none observed" if manual_reviewed else "pending",
            }
        )
    payload = {
        "generated_at": utc_now(),
        "pdf_count": len(records),
        "manual_reviewed": bool(manual_reviewed),
        "records": records,
    }
    _atomic_json(payload, results / "reproducibility/pdf_qa.json")
    return payload
