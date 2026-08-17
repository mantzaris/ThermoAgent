"""Post-freeze presentation-only repairs for V10 paper figures.

The frozen numerical and analysis source is not modified. This script reads the
already generated compact figure-source CSV files and corrects only layout,
legend, and uncertainty-display defects found during manual PDF QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/llm_agent_entropy_v10"
SOURCE = RESULTS / "figures/source_data"
PDF = RESULTS / "figures/pdf"

COLORS: Dict[str, str] = {
    "black": "#000000",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#8A8A8A",
    "lightgray": "#D9D9D9",
}

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
    }
)


def _save(fig: plt.Figure, filename: str) -> None:
    PDF.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PDF / filename,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "ThermoAgent V10 post-freeze presentation renderer"},
    )
    plt.close(fig)


def _box(
    axis: plt.Axes,
    center: Tuple[float, float],
    size: Tuple[float, float],
    text: str,
    color: str,
    fontsize: float = 9.3,
) -> None:
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor=COLORS["black"],
        linewidth=1.0,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(
    axis: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    style: str = "-",
    color: str = COLORS["gray"],
    curvature: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.3,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={curvature}",
        zorder=1,
    )
    axis.add_patch(arrow)


def architecture() -> None:
    data = pd.read_csv(SOURCE / "figure_01_architecture.csv")
    expected = set(data.loc[data["record"] == "node", "name"])
    required = {
        "private evidence",
        "agent belief b_i",
        "agent action a_i",
        "local heat bath",
        "typed LLM decision",
        "private memory",
        "inbox / outbox",
        "environment transition",
        "evaluator-only scoring",
    }
    if expected != required:
        raise ValueError("architecture source-data labels changed")

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.55))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    left, right = axes
    left.set_title("Analytical stochastic-agent\nreference", weight="bold", pad=8, fontsize=10.2)
    left.text(-0.02, 1.02, "a", transform=left.transAxes, fontsize=12, weight="bold")
    _box(left, (0.16, 0.69), (0.25, 0.18), "private\nevidence", COLORS["sky"])
    _box(left, (0.50, 0.69), (0.25, 0.18), "belief  $b_i$", COLORS["green"])
    _box(left, (0.84, 0.69), (0.25, 0.18), "action  $a_i$", COLORS["green"])
    _box(left, (0.50, 0.30), (0.27, 0.18), "local\nheat bath", COLORS["yellow"])
    _arrow(left, (0.29, 0.69), (0.37, 0.69))
    _arrow(left, (0.63, 0.69), (0.71, 0.69))
    _arrow(left, (0.50, 0.60), (0.50, 0.40))
    left.text(0.67, 0.75, "$K$", ha="center", fontsize=9.2)

    right.set_title("Independent LLM-agent\nrealization", weight="bold", pad=8, fontsize=10.2)
    right.text(-0.02, 1.02, "b", transform=right.transAxes, fontsize=12, weight="bold")
    _box(right, (0.16, 0.72), (0.25, 0.18), "private\nevidence", COLORS["sky"])
    _box(right, (0.16, 0.42), (0.25, 0.18), "private\nmemory", COLORS["sky"])
    _box(right, (0.50, 0.58), (0.27, 0.20), "typed LLM\ndecision", COLORS["orange"])
    _box(right, (0.84, 0.72), (0.25, 0.18), "inbox /\noutbox", COLORS["purple"])
    _box(right, (0.50, 0.20), (0.29, 0.18), "environment\ntransition", COLORS["blue"], 8.1)
    _box(right, (0.84, 0.20), (0.27, 0.18), "evaluator-only\nscoring", COLORS["lightgray"], 8.1)
    _arrow(right, (0.29, 0.72), (0.38, 0.63))
    _arrow(right, (0.29, 0.42), (0.37, 0.53))
    _arrow(right, (0.71, 0.69), (0.63, 0.62), curvature=0.12)
    _arrow(right, (0.62, 0.57), (0.72, 0.66), curvature=0.12)
    _arrow(right, (0.50, 0.48), (0.50, 0.30))
    _arrow(right, (0.65, 0.20), (0.71, 0.20), style="--")
    fig.subplots_adjust(wspace=0.13)
    _save(fig, "figure_01_architecture.pdf")


def probability_currents() -> None:
    data = pd.read_csv(SOURCE / "figure_05_probability_currents.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    reciprocal = data[np.isclose(data["alpha"], 0.0)]
    axes[0].axhline(0.0, color=COLORS["black"], lw=1)
    axes[0].scatter(reciprocal["rank"], reciprocal["current"], color=COLORS["blue"], s=28)
    axes[0].set(
        xlabel="ranked state-pair current",
        ylabel="stationary probability current",
        title=r"reciprocal $\alpha=0$",
    )
    axes[0].text(
        0.5,
        0.58,
        "all currents vanish\nwithin numerical precision",
        ha="center",
        transform=axes[0].transAxes,
    )
    nonreciprocal = data[np.isclose(data["alpha"], 0.5)].sort_values("rank")
    signs = np.sign(nonreciprocal["current"].to_numpy(float))
    colors = [COLORS["vermillion"] if sign > 0 else COLORS["blue"] for sign in signs]
    axes[1].bar(nonreciprocal["rank"], nonreciprocal["absolute_current"], color=colors, width=0.78)
    axes[1].set(
        xlabel="ranked directed state edge",
        ylabel="absolute stationary current",
        title=r"nonreciprocal $\alpha=0.5$",
    )
    axes[1].legend(
        handles=[
            Patch(facecolor=COLORS["vermillion"], label="signed current $>0$"),
            Patch(facecolor=COLORS["blue"], label="signed current $<0$"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.4,
    )
    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.06, chr(ord("a") + index), transform=axis.transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "figure_05_probability_currents.pdf")


def temperature_response() -> None:
    data = pd.read_csv(SOURCE / "figure_06_temperature_nonreciprocity.csv")
    selected = data[(data["n_agents"] == 64) & (data["topology"] == "small_world")].copy()
    alphas = sorted(selected["alpha"].unique())
    palette = [COLORS["black"], COLORS["sky"], COLORS["orange"], COLORS["vermillion"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for alpha, color in zip(alphas, palette):
        subset = selected[np.isclose(selected["alpha"], alpha)].sort_values("temperature")
        x = subset["temperature"].to_numpy(float)
        mean = subset["pathwise_irreversibility_per_update_mean"].to_numpy(float)
        low = subset["pathwise_irreversibility_per_update_ci_low"].to_numpy(float)
        high = subset["pathwise_irreversibility_per_update_ci_high"].to_numpy(float)
        axes[0].plot(x, mean, marker="o", color=color, label=rf"$\alpha={alpha:g}$")
        axes[0].fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        if alpha > 0:
            axes[1].plot(x, mean / alpha**2, marker="s", color=color, label=rf"$\alpha={alpha:g}$")
            axes[1].fill_between(x, low / alpha**2, high / alpha**2, color=color, alpha=0.14, linewidth=0)
    axes[0].set(xlabel="decision temperature $T$", ylabel="pathwise irreversibility / update")
    axes[1].set(xlabel="decision temperature $T$", ylabel=r"irreversibility / $\alpha^2$ ($\alpha>0$)")
    axes[0].legend(loc="center right", frameon=False, fontsize=8.4)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.4)
    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.06, chr(ord("a") + index), transform=axis.transAxes, fontsize=12, weight="bold")
    fig.subplots_adjust(wspace=0.37)
    _save(fig, "figure_06_temperature_response.pdf")


def _application_panel(axis: plt.Axes, application: str, title: str) -> None:
    data = pd.read_csv(SOURCE / "figure_07_application_mappings.csv")
    subset = data[data["application"] == application]
    labels = list(subset["node"])
    if application == "humanitarian":
        positions = {
            "depot": (0.13, 0.55),
            "field team": (0.34, 0.76),
            "carrier": (0.55, 0.55),
            "clinic": (0.82, 0.76),
            "shelter": (0.82, 0.33),
        }
        edges = [("depot", "field team"), ("field team", "carrier"), ("carrier", "clinic"), ("carrier", "shelter")]
    else:
        positions = {
            "component operator": (0.15, 0.55),
            "telemetry relay": (0.38, 0.76),
            "crew coordinator": (0.59, 0.55),
            "critical load": (0.84, 0.76),
            "safety monitor": (0.84, 0.33),
        }
        edges = [
            ("component operator", "telemetry relay"),
            ("telemetry relay", "crew coordinator"),
            ("crew coordinator", "critical load"),
            ("crew coordinator", "safety monitor"),
        ]
    if set(labels) != set(positions):
        raise ValueError(f"application source-data labels changed for {application}")
    role_colors = {
        "resource": COLORS["green"],
        "evidence": COLORS["purple"],
        "action": COLORS["orange"],
        "service": COLORS["sky"],
    }
    for source, destination in edges:
        _arrow(axis, positions[source], positions[destination])
    for _, row in subset.iterrows():
        label = str(row["node"])
        display = label.replace(" ", "\n") if len(label) > 12 else label
        _box(axis, positions[label], (0.22, 0.15), display, role_colors[str(row["role_class"])], 8.1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0.12, 0.93)
    axis.axis("off")
    axis.set_title(title, loc="left", weight="bold", pad=5)


def application_mappings() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    _application_panel(axes[0], "humanitarian", "a  Humanitarian coordination")
    _application_panel(axes[1], "utility", "b  Defensive utility restoration")
    handles = [
        Patch(facecolor=COLORS["green"], edgecolor="black", label="resource"),
        Patch(facecolor=COLORS["purple"], edgecolor="black", label="evidence"),
        Patch(facecolor=COLORS["orange"], edgecolor="black", label="action"),
        Patch(facecolor=COLORS["sky"], edgecolor="black", label="service/safety"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.015,
        "Illustrative mapping only; the formal dynamic Qwen stage was not unlocked.",
        ha="center",
        fontsize=9.2,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.08)
    _save(fig, "figure_07_application_mappings.pdf")


def main() -> None:
    architecture()
    probability_currents()
    temperature_response()
    application_mappings()


if __name__ == "__main__":
    main()
