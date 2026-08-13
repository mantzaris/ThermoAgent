"""Consistent vector publication figures and rendered previews."""

from __future__ import annotations

import json
import gzip
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .mechanics import MacrostateCalibration, local_surprisal


PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#222222", "gray": "#777777",
}
METHOD_STYLE = {
    "centralized_lookahead": (PALETTE["black"], "o", "-"),
    "centralized_llm": (PALETTE["purple"], "P", "--"),
    "scripted_independent": (PALETTE["gray"], "s", "--"),
    "autonomous_no_comm": (PALETTE["red"], "X", ":"),
    "autonomous_fixed_comm": (PALETTE["orange"], "D", "-."),
    "learned_no_entropy": (PALETTE["blue"], "^", "--"),
    "thermoagent": (PALETTE["green"], "*", "-"),
    "random_gate": (PALETTE["yellow"], "v", ":"),
    "entropy_llm_only": (PALETTE["orange"], "d", "--"),
    "no_episodic_memory": (PALETTE["red"], "p", "-."),
    "global_entropy_oracle": (PALETTE["sky"], "h", "--"),
    "shuffled_entropy": (PALETTE["purple"], ">", ":"),
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _save(fig: Any, name: str, results_root: Path) -> Tuple[Path, Path]:
    pdf_dir = results_root / "figures" / "pdf"
    preview_dir = results_root / "figures" / "previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / (name + ".pdf")
    png = preview_dir / (name + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=180)
    plt.close(fig)
    return pdf, png


def _ci(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, mean, mean
    rng = np.random.RandomState(20260811)
    samples = array[rng.randint(0, len(array), size=(3000, len(array)))].mean(axis=1)
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def architecture_figure(results_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    boxes = [
        (0.25, 4.8, 2.8, 1.45, "Independent agent\nprivate state\nprivate memory", PALETTE["blue"]),
        (4.6, 4.8, 2.8, 1.45, "RL coordination\nmetapolicy\nlocal features only", PALETTE["green"]),
        (8.95, 4.8, 2.8, 1.45, "Frozen LLM planner\nseparate context", PALETTE["purple"]),
        (7.4, 1.35, 3.0, 1.45, "Typed tools\nquantitative simulator", PALETTE["orange"]),
        (1.6, 1.35, 3.0, 1.45, "Distributed monitor\nmacrostate gossip", PALETTE["red"]),
    ]
    for x, y, w, h, label, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=mpl.colors.to_rgba(color, 0.12), edgecolor=color, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="semibold", fontsize=9)
    arrows = [
        ((3.05, 5.52), (4.6, 5.52), "local features", (3.82, 6.52)),
        ((7.4, 5.52), (8.95, 5.52), "coordination option", (8.18, 6.52)),
        ((10.35, 4.8), (9.0, 2.8), "validated JSON", (10.05, 3.65)),
        ((7.4, 2.1), (2.1, 4.8), "result and observation", (4.65, 3.3)),
        ((7.4, 1.85), (4.6, 1.85), "coarse sketch", (6.0, 2.10)),
        ((3.1, 2.8), (5.3, 4.8), "$S,\\Delta F,I_i$", (3.95, 3.8)),
    ]
    for start, end, label, label_position in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": PALETTE["black"]})
        ax.text(*label_position, label, ha="center", va="center", fontsize=8.5, backgroundcolor="white")
    ax.annotate("", xy=(10.8, 0.65), xytext=(1.2, 0.65), arrowprops={"arrowstyle": "<->", "lw": 1.2, "linestyle": "--", "color": PALETTE["gray"]})
    ax.text(6, 0.88, "Explicit logged messages, offers, commitments, and coalitions", ha="center", fontsize=9, color=PALETTE["gray"])
    ax.text(6, 0.12, "Exact global state is evaluator-only (except the named oracle ablation)", ha="center", color=PALETTE["gray"], fontsize=8.5)
    ax.set_title("ThermoAgent architecture and privacy boundary")
    _save(fig, "system_architecture", results_root)


def entropy_dynamics(time_series: pd.DataFrame, results_root: Path) -> None:
    frame = time_series[(time_series["method"] == "thermoagent") & time_series["scenario_name"].astype(str).str.contains("compound")]
    if frame.empty:
        frame = time_series[time_series["method"] == "thermoagent"]
    application = "commercial" if "commercial" in frame["application"].values else frame.iloc[0]["application"]
    frame = frame[frame["application"] == application]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.2), sharex=True)
    specs = [
        ("exact_entropy", "Operational entropy, $S$", PALETTE["blue"]),
        ("exact_energy", "Operational energy, $U$", PALETTE["orange"]),
        ("exact_free_energy", "Free-energy gap, $\\Delta F$", PALETTE["red"]),
        ("fulfillment_rate", "Cumulative fulfillment", PALETTE["green"]),
    ]
    for ax, (metric, label, color) in zip(axes, specs):
        grouped = frame.groupby("step")[metric]
        x = sorted(grouped.groups)
        stats_rows = [_ci(grouped.get_group(step)) for step in x]
        mean = np.asarray([row[0] for row in stats_rows])
        low = np.asarray([row[1] for row in stats_rows])
        high = np.asarray([row[2] for row in stats_rows])
        ax.plot(x, mean, color=color, marker="o", markersize=3, linewidth=1.5)
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.set_ylabel(label)
    disruption = int(frame[frame["disruption_active"].astype(bool)]["step"].min()) if frame["disruption_active"].any() else None
    if disruption is not None:
        for ax in axes:
            ax.axvline(disruption, color=PALETTE["black"], linestyle="--", linewidth=1, label="Disruption")
    detection_path = results_root / "statistics" / "detection_episodes.csv"
    if detection_path.exists():
        detections = pd.read_csv(detection_path)
        detected = detections[
            (detections["signal"] == "absolute_free_energy_deviation")
            & detections["run_id"].isin(set(frame["run_id"]))
        ]["detection_step"].dropna()
        if not detected.empty:
            detection = float(detected.median())
            for ax in axes:
                ax.axvline(
                    detection,
                    color=PALETTE["red"],
                    linestyle=":",
                    linewidth=1.2,
                    label=r"Median $|\Delta F|$ detection",
                )
    axes[-1].set_xlabel("Simulator period")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.suptitle("Disruption-aligned monitoring and logistics response (%s)" % application.capitalize())
    fig.tight_layout(rect=(0.025, 0, 1, 0.96), h_pad=1.4)
    _save(fig, "entropy_dynamics", results_root)


def main_performance(episodes: pd.DataFrame, results_root: Path) -> None:
    frame = episodes[(episodes["stage"] == "main") & (episodes["completion_status"] == "complete")]
    applications = [app for app in ("commercial", "humanitarian") if app in frame["application"].values]
    fig, axes = plt.subplots(len(applications), 1, figsize=(7.2, 3.5 * len(applications)), squeeze=False)
    for row_index, application in enumerate(applications):
        ax = axes[row_index, 0]
        app = frame[frame["application"] == application]
        scenarios = list(dict.fromkeys(app["scenario_name"].astype(str)))
        methods = [m for m in METHOD_STYLE if m in app["method"].values]
        offset = 0.72 / max(1, len(methods))
        for method_index, method in enumerate(methods):
            color, marker, _ = METHOD_STYLE[method]
            for scenario_index, scenario in enumerate(scenarios):
                values = app[(app["method"] == method) & (app["scenario_name"] == scenario)]["primary_outcome"].astype(float).values
                if not len(values):
                    continue
                mean, low, high = _ci(values)
                y = scenario_index - 0.36 + offset * (method_index + 0.5)
                jitter = np.linspace(-offset * 0.18, offset * 0.18, len(values))
                ax.scatter(values, y + jitter, s=13, alpha=0.42, color=color, marker=marker, linewidths=0)
                ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], color=color, marker=marker, markersize=5.5, capsize=2.5, linewidth=1.1)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([s.replace("_", " ") for s in scenarios], fontsize=8.5)
        ax.set_ylim(len(scenarios) - 0.55, -0.55)
        ax.set_xlabel("Service-loss AUC (lower is better)" if application == "commercial" else "Cumulative unmet weighted need (lower is better)")
        ax.set_title(application.capitalize())
    handles = [Line2D([0], [0], color=METHOD_STYLE[m][0], marker=METHOD_STYLE[m][1], linestyle="none", label=m.replace("_", " ")) for m in methods]
    axes[0, 0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("Primary episode outcomes (points are paired scenario seeds)", y=0.99)
    fig.tight_layout(rect=(0, 0, 0.84, 0.94))
    _save(fig, "main_performance", results_root)


def communication_pareto(episodes: pd.DataFrame, results_root: Path) -> None:
    frame = episodes[(episodes["stage"] == "main") & (episodes["completion_status"] == "complete")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        app = frame[frame["application"] == application]
        for method, group in app.groupby("method"):
            color, marker, _ = METHOD_STYLE.get(method, (PALETTE["gray"], "o", "-"))
            communication_column = (
                "total_communication_messages"
                if "total_communication_messages" in group
                else "messages"
            )
            x, xl, xh = _ci(group[communication_column].astype(float))
            y, yl, yh = _ci(group["primary_outcome"].astype(float))
            ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], color=color, marker=marker, markersize=7, capsize=2, linestyle="none")
        ax.set_xlabel("Operational + monitor messages per episode")
        ax.set_ylabel("Primary loss (lower is better)")
        ax.set_title(application.capitalize())
    methods = [method for method in METHOD_STYLE if method in frame["method"].values]
    handles = [Line2D([0], [0], color=METHOD_STYLE[m][0], marker=METHOD_STYLE[m][1], linestyle="none", label=m.replace("_", " ")) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Communication–performance trade-off")
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    _save(fig, "communication_performance_pareto", results_root)


def _factor_values(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    private = series.astype(str).str.extract(r"-p([0-9.]+)-o")[0].astype(float).to_numpy()
    objective = series.astype(str).str.extract(r"-o([0-9.]+)$")[0].astype(float).to_numpy()
    return private, objective


def _fixed_deployable_benchmark(app: pd.DataFrame) -> pd.DataFrame:
    """Choose one deployable comparator per factor cell, then preserve pairing.

    Selecting the lower outcome separately for every seed would create a
    clairvoyant ensemble that is not itself deployable.  We instead select the
    method with the lower across-seed mean in each scenario/size cell (with a
    deterministic name tie-break), then retain that fixed method's seed rows.
    The map is descriptive and labels this as the best *observed* comparator.
    """

    candidates = app[
        app["method"].isin(["centralized_llm", "scripted_independent"])
    ].copy()
    cell_keys = ["scenario_name"]
    if "n_agents" in candidates:
        cell_keys.append("n_agents")
    means = (
        candidates.groupby(cell_keys + ["method"], as_index=False)[
            "primary_outcome"
        ]
        .mean()
        .sort_values(cell_keys + ["primary_outcome", "method"])
    )
    selected = means.groupby(cell_keys, as_index=False).first()[
        cell_keys + ["method"]
    ]
    fixed = candidates.merge(selected, on=cell_keys + ["method"], how="inner")
    return fixed.rename(columns={"primary_outcome": "benchmark"})


def necessity_map(episodes: pd.DataFrame, results_root: Path) -> None:
    frame = episodes[(episodes["stage"] == "main") & (episodes["completion_status"] == "complete")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    all_z: List[float] = []
    prepared: List[Tuple[Any, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        app = frame[frame["application"] == application]
        moderate = app[app["scenario"].astype(str).str.contains("-moderate-")]
        if not moderate.empty:
            app = moderate
        thermo = app[app["method"] == "thermoagent"]
        # The full-information numerical controller is an intentionally
        # unattainable oracle and therefore cannot define whether deployable
        # autonomy is *necessary* under privacy.  Use the best legally informed
        # central LLM or scripted-independent comparator for this map; the
        # oracle remains visible in the main performance figure and tables.
        bench = _fixed_deployable_benchmark(app)
        pair_keys = ["scenario_name", "seed"]
        if "n_agents" in thermo and "n_agents" in bench:
            pair_keys.append("n_agents")
        paired = thermo.merge(bench[pair_keys + ["benchmark"]], on=pair_keys)
        scenario_column = "scenario_thermo" if "scenario_thermo" in paired.columns else "scenario"
        grouped = paired.groupby("scenario_name").agg(
            thermo=("primary_outcome", "mean"),
            benchmark=("benchmark", "mean"),
            factor_code=(scenario_column, "first"),
        ).reset_index()
        denominator = pd.concat([grouped["benchmark"].abs(), grouped["thermo"].abs()], axis=1).max(axis=1).clip(lower=1e-9)
        grouped["advantage"] = 100.0 * (grouped["benchmark"] - grouped["thermo"]) / denominator
        x, y = _factor_values(grouped["factor_code"])
        z = grouped["advantage"].to_numpy(float)
        prepared.append((ax, x, y, z, application))
        all_z.extend(z.tolist())
    bound = max(1e-9, float(np.max(np.abs(all_z))))
    norm = mpl.colors.Normalize(vmin=-bound, vmax=bound)
    mappable = None
    for ax, x, y, z, application in prepared:
        if len(x) >= 3 and len(np.unique(np.c_[x, y], axis=0)) >= 3:
            mappable = ax.tricontourf(x, y, z, levels=np.linspace(-bound, bound, 9), cmap="RdBu", norm=norm, extend="both")
        scatter = ax.scatter(x, y, c=z, cmap="RdBu", norm=norm, edgecolor="black", s=55, zorder=3)
        if mappable is None:
            mappable = scatter
        for xi, yi, zi in zip(x, y, z):
            ax.text(xi, yi + 0.06, "%+.0f%%" % zi, ha="center", fontsize=8)
        ax.axhline(0.5, color=PALETTE["gray"], lw=0.5, alpha=0.3)
        ax.axvline(0.5, color=PALETTE["gray"], lw=0.5, alpha=0.3)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.12)
        ax.set_xlabel("Private-information level")
        ax.set_title(application.capitalize())
    axes[0].set_ylabel("Objective-misalignment level")
    fig.colorbar(mappable, ax=axes, label="Normalized autonomous advantage (%)", fraction=0.05, pad=0.05)
    fig.suptitle("Where autonomous coordination is justified", y=0.99)
    fig.text(
        0.5, 0.01,
        "Positive values favor ThermoAgent over the best observed fixed legal-central or scripted comparator per cell.",
        ha="center", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.20, top=0.82, wspace=0.28)
    _save(fig, "agentic_necessity_map", results_root)


def recovery_curves(time_series: pd.DataFrame, results_root: Path) -> None:
    frame = time_series[(time_series["stage"] == "main") & time_series["scenario_name"].astype(str).str.contains("compound")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        app = frame[frame["application"] == application]
        for method, group in app.groupby("method"):
            color, marker, line = METHOD_STYLE.get(method, (PALETTE["gray"], "o", "-"))
            grouped = group.groupby("step")["service_loss"]
            x = sorted(grouped.groups)
            stat = [_ci(grouped.get_group(step)) for step in x]
            mean, low, high = [np.asarray([row[i] for row in stat]) for i in range(3)]
            ax.plot(x, mean, color=color, linestyle=line, marker=marker, markevery=3, markersize=3, label=method.replace("_", " "))
            ax.fill_between(x, low, high, color=color, alpha=0.1, linewidth=0)
        if not app.empty and app["disruption_active"].any():
            ax.axvline(int(app[app["disruption_active"].astype(bool)]["step"].min()), color=PALETTE["black"], linestyle="--", lw=1)
        ax.set_xlabel("Simulator period")
        ax.set_ylabel("Service loss")
        ax.set_title(application.capitalize())
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("Compound-shock recovery trajectories", y=0.99)
    fig.tight_layout(rect=(0, 0, 0.82, 0.88))
    _save(fig, "recovery_curves", results_root)


def ablation_effects(episodes: pd.DataFrame, results_root: Path) -> None:
    frame = episodes[
        (episodes["stage"] == "ablations")
        & (episodes["completion_status"] == "complete")
    ]
    preferred_order = [
        "thermoagent", "learned_no_entropy", "entropy_llm_only",
        "no_episodic_memory", "autonomous_fixed_comm", "autonomous_no_comm",
        "random_gate", "shuffled_entropy", "global_entropy_oracle",
    ]
    labels = {
        "thermoagent": "ThermoAgent",
        "learned_no_entropy": "No entropy/free energy",
        "entropy_llm_only": "Entropy to LLM; no RL gate",
        "no_episodic_memory": "No episodic memory",
        "autonomous_fixed_comm": "Fixed communication",
        "autonomous_no_comm": "No communication",
        "random_gate": "Activity-matched random gate",
        "shuffled_entropy": "Shuffled delayed entropy",
        "global_entropy_oracle": "Exact-global oracle",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.7), sharey=True)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        app = frame[frame["application"] == application]
        observed = set(app["method"].astype(str))
        methods = [method for method in preferred_order if method in observed]
        values = []
        errors = []
        colors = []
        for method in methods:
            mean, low, high = _ci(app[app["method"] == method]["primary_outcome"].astype(float))
            values.append(mean)
            errors.append([mean - low, high - mean])
            colors.append(METHOD_STYLE.get(method, (PALETTE["gray"], "o", "-"))[0])
        positions = np.arange(len(methods))
        ax.barh(positions, values, color=colors, alpha=0.78)
        if methods:
            ax.errorbar(values, positions, xerr=np.asarray(errors).T, fmt="none", color=PALETTE["black"], capsize=3, lw=1)
        for index, method in enumerate(methods):
            points = app[app["method"] == method]["primary_outcome"].astype(float).to_numpy()
            offsets = np.linspace(-0.12, 0.12, len(points)) if len(points) > 1 else np.zeros(len(points))
            ax.scatter(points, index + offsets, s=13, facecolors="white", edgecolors=PALETTE["black"], linewidths=0.55, zorder=3)
        ax.set_yticks(positions)
        ax.set_yticklabels([labels[method] for method in methods])
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(application.capitalize())
    axes[0].invert_yaxis()
    fig.suptitle("Parameter-matched monitoring and coordination ablations", y=0.99)
    # Figure.supxlabel was introduced after the oldest supported Matplotlib.
    # Figure.text provides the same shared-label layout on Matplotlib 3.3.
    fig.text(
        0.5,
        0.015,
        "Primary episode loss (lower is better)",
        ha="center",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    _save(fig, "ablation_effects", results_root)


def _event_rows(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _formed_coalition_members(
    events: Sequence[Mapping[str, Any]], step: int
) -> Tuple[Optional[str], set[str]]:
    """Reconstruct the most recently *joined* coalition at one period."""

    coalition_events = [
        event for event in events
        if event.get("kind") == "coalition_event"
        and int(event.get("step", -1)) <= step
    ]
    joins = [
        event for event in coalition_events
        if event.get("payload", {}).get("action") == "join_coalition"
        and event.get("payload", {}).get("ok")
    ]
    if not joins:
        return None, set()
    coalition_id = str(joins[-1]["payload"]["coalition_id"])
    proposal = next(
        (
            event for event in coalition_events
            if str(event.get("payload", {}).get("coalition_id")) == coalition_id
            and event.get("payload", {}).get("action") == "propose"
        ),
        None,
    )
    members: set[str] = set()
    if proposal is not None:
        proposer = proposal.get("payload", {}).get("proposer", proposal.get("actor"))
        if proposer:
            members.add(str(proposer))
    for event in coalition_events:
        payload = event.get("payload", {})
        if str(payload.get("coalition_id")) != coalition_id or not payload.get("ok", True):
            continue
        if payload.get("action") == "join_coalition":
            members.add(str(event.get("actor")))
        elif payload.get("action") == "withdraw_coalition":
            members.discard(str(event.get("actor")))
    return coalition_id, members


def network_snapshots(results_root: Path, application: str) -> None:
    candidates = sorted((results_root / "raw" / "main").glob("*thermoagent*compound*/events.jsonl*"))
    candidates = [path for path in candidates if application in path.parent.name]
    if not candidates:
        candidates = sorted((results_root / "raw").glob("*/*%s*thermoagent*/events.jsonl*" % application))
    events = _event_rows(candidates[0])
    observation_events = [e for e in events if e["kind"] == "observation_delivery"]
    topology_events = [e for e in events if e["kind"] == "topology_snapshot"]
    topology = topology_events[0]["payload"] if topology_events else {}
    agents = sorted(topology.get("agents", {})) or sorted({e["payload"]["recipient"] for e in observation_events})
    horizon = max(e["step"] for e in observation_events) + 1
    disruption_events = [event for event in events if event["kind"] == "disruption"]
    disruption_step = disruption_events[0]["step"] if disruption_events else horizon // 3
    detection_step = disruption_step
    detection_path = results_root / "statistics" / "detection_episodes.csv"
    if detection_path.exists():
        episode = json.loads((candidates[0].parent / "episode.json").read_text(encoding="utf-8"))
        detections = pd.read_csv(detection_path)
        match = detections[
            (detections["run_id"] == episode["run_id"])
            & (detections["signal"] == "absolute_free_energy_deviation")
        ]["detection_step"].dropna()
        if not match.empty:
            detection_step = int(match.iloc[0])
    negotiation_kinds = {
        "information_request", "quote_request", "offer", "counteroffer",
        "priority_request", "challenge", "coalition_proposal",
    }
    negotiation_steps = [
        event["step"] for event in events
        if event["kind"] == "message"
        and event["step"] >= disruption_step
        and event["payload"].get("kind") in negotiation_kinds
    ]
    coalition_steps = [
        event["step"] for event in events
        if event["kind"] == "coalition_event"
        and event["step"] >= disruption_step
        and event["payload"].get("action") == "join_coalition"
        and event["payload"].get("ok")
    ]
    negotiation_step = min(negotiation_steps) if negotiation_steps else min(horizon - 1, disruption_step + 1)
    coalition_step = min(coalition_steps) if coalition_steps else min(horizon - 1, 2 * horizon // 3)
    steps = [0, detection_step, negotiation_step, coalition_step, horizon - 1]
    labels = [
        "Nominal",
        "Monitor detection" if detection_step != disruption_step else "Shock onset",
        "Negotiation escalation" if negotiation_steps else "No negotiation observed",
        "Coalition response" if coalition_steps else "No coalition formed",
        "Recovery endpoint",
    ]
    positions = {
        agent: np.asarray(topology["agents"][agent]["location"], dtype=float)
        for agent in agents
    } if topology.get("agents") else nx.circular_layout(agents, scale=1.0)
    source_roles = {"supplier", "manufacturer", "warehouse", "ngo", "agency", "depot"}
    demand_roles = {"retailer", "clinic", "community"}
    transport_roles = {"carrier", "transport"}
    roles = {
        agent: topology.get("agents", {}).get(agent, {}).get("role", agent.rsplit("_", 1)[0])
        for agent in agents
    }
    physical = [tuple(edge) for edge in topology.get("physical_edges", [])]
    if not physical:
        physical = [(a, b) for a in agents for b in agents if roles[a] in source_roles and roles[b] in demand_roles]
    initial_communication = [tuple(edge) for edge in topology.get("communication_edges", [])]
    if not initial_communication:
        initial_communication = [(agents[i], agents[(i + 1) % len(agents)]) for i in range(len(agents))]
    calibration_path = results_root / "reproducibility" / "macrostate_calibration.json"
    if calibration_path.exists():
        calibration_value = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = MacrostateCalibration(
            thresholds=np.asarray(calibration_value["thresholds"], dtype=float),
            alpha=float(calibration_value["alpha"]),
            temperature=float(calibration_value["temperature"]),
            energy_weights=tuple(calibration_value["energy_weights"]),
            role_references={
                str(role): list(reference)
                for role, reference in calibration_value.get("role_references", {}).items()
            },
        )
    else:
        calibration = MacrostateCalibration(np.asarray([[0.20, 0.55]] * 3, dtype=float))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.2))
    flat_axes = list(axes.flat)
    for ax, step, label in zip(flat_axes, steps, labels):
        ax.axis("off")
        ax.set_xlim(-1.28, 1.28)
        ax.set_ylim(-1.28, 1.28)
        ax.text(
            0.5, 1.02, label + "\nperiod %d" % step,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
        )
        observations: Dict[str, Dict[str, Any]] = {}
        for event in observation_events:
            if event["step"] <= step:
                observations[event["payload"]["recipient"]] = event["payload"]["observation"]
        messages = [e for e in events if e["kind"] == "message" and max(0, step - 2) <= e["step"] <= step and not e["payload"].get("dropped")]
        commitments = [e for e in events if e["kind"] == "commitment" and e["step"] <= step]
        coalition_events = [
            event for event in events
            if event["kind"] == "coalition_event" and event["step"] <= step
        ]
        disruptions = [e for e in events if e["kind"] == "disruption" and e["step"] <= step]
        closed_edges = {
            tuple(edge) for event in disruptions
            for edge in event["payload"].get("route_closures", [])
        }
        current_physical = [edge for edge in physical if edge not in closed_edges]
        current_communication = list(initial_communication)
        if disruptions and topology.get("communication_regime") == "partition":
            midpoint = len(agents) // 2
            first = set(agents[:midpoint])
            current_communication = [
                edge for edge in current_communication
                if (edge[0] in first) == (edge[1] in first)
            ]
        if disruptions:
            lost_coordinators = {
                event["payload"].get("coordinator_loss") for event in disruptions
                if event["payload"].get("coordinator_loss")
            }
            current_communication = [
                edge for edge in current_communication
                if not any(agent in lost_coordinators for agent in edge)
            ]
        recent_flows = []
        for event in events:
            if event["kind"] != "tool_call" or not max(0, step - 2) <= event["step"] <= step:
                continue
            if event["payload"].get("tool") in ("schedule_shipment", "transfer_resource"):
                recent_flows.append((event["actor"], event["payload"]["arguments"].get("target")))
        graph = nx.DiGraph()
        graph.add_nodes_from(agents)
        nx.draw_networkx_edges(graph, positions, edgelist=current_physical, ax=ax, width=0.5, alpha=0.18, edge_color=PALETTE["black"], arrows=False)
        if recent_flows:
            nx.draw_networkx_edges(graph, positions, edgelist=recent_flows, ax=ax, width=1.8, alpha=0.65, edge_color=PALETTE["black"], arrows=True, arrowsize=7)
        nx.draw_networkx_edges(graph, positions, edgelist=current_communication, ax=ax, width=0.6, alpha=0.35, style="dashed", edge_color=PALETTE["sky"], arrows=False)
        negotiations = [
            event for event in messages
            if event["payload"].get("kind") in (
                "information_request", "quote_request", "offer", "counteroffer",
                "priority_request", "challenge", "coalition_proposal",
            )
        ]
        if negotiations:
            nx.draw_networkx_edges(graph, positions, edgelist=[(e["payload"]["sender"], e["payload"]["recipient"]) for e in negotiations], ax=ax, width=1.1, edge_color=PALETTE["orange"], arrows=True, arrowsize=7, connectionstyle="arc3,rad=0.08")
        if commitments:
            nx.draw_networkx_edges(graph, positions, edgelist=[(e["payload"]["proposer"], e["payload"]["partner"]) for e in commitments], ax=ax, width=2.0, edge_color=PALETTE["green"], arrows=True, arrowsize=7)
        node_values: Dict[str, float] = {}
        sizes: Dict[str, float] = {}
        for agent in agents:
            obs = observations.get(agent, {})
            pressure = float(obs.get("backlog", 0.0)) / max(float(obs.get("local_forecast", 1.0)), 1.0)
            features = [
                min(1.0, max(pressure, float(obs.get("service_shortfall", 0.0)))),
                min(1.0, float(obs.get("impairment", 0.0))),
                min(1.0, 0.6 * float(obs.get("commitment_strain", 0.0)) + 0.4 * (1.0 - float(obs.get("communication_reliability", 1.0)))),
            ]
            node_values[agent] = local_surprisal(
                calibration.encode(features), calibration.role_reference(roles[agent])
            )
            sizes[agent] = 65 + 110 * min(1.0, pressure + float(obs.get("commitment_strain", 0.0)))
        role_groups = [
            (source_roles, "o"), (demand_roles, "s"), (transport_roles, "^")
        ]
        for role_set, shape in role_groups:
            nodes = [agent for agent in agents if roles[agent] in role_set]
            if not nodes:
                continue
            nx.draw_networkx_nodes(
                graph, positions, nodelist=nodes, ax=ax,
                node_color=[node_values[agent] for agent in nodes], cmap="viridis", vmin=0, vmax=5.5,
                node_size=[sizes[agent] for agent in nodes], node_shape=shape,
                edgecolors=[PALETTE["red"] if observations.get(agent, {}).get("impairment", 0) > 0.5 else PALETTE["black"] for agent in nodes],
                linewidths=[1.8 if observations.get(agent, {}).get("impairment", 0) > 0.5 else 0.7 for agent in nodes],
            )
        nx.draw_networkx_labels(graph, positions, labels={a: roles[a][:3].upper() + a[-2:] for a in agents}, font_size=7.5, ax=ax)
        coalition_id, formed_members = _formed_coalition_members(events, step)
        if coalition_id is not None:
            members = [member for member in formed_members if member]
            if members:
                xy = np.asarray([positions[m] for m in members if m in positions])
                if len(xy) >= 3 and np.linalg.matrix_rank(xy - xy.mean(axis=0)) >= 2:
                    center = xy.mean(axis=0)
                    expanded = center + 1.12 * (xy - center)
                    hull = ConvexHull(expanded)
                    ax.add_patch(patches.Polygon(
                        expanded[hull.vertices], closed=True, fill=False,
                        edgecolor=PALETTE["purple"], linestyle="--", linewidth=1.4,
                    ))
                elif len(xy):
                    center = xy.mean(axis=0)
                    radius = max(0.15, float(np.linalg.norm(xy - center, axis=1).max() + 0.08))
                    ax.add_patch(patches.Circle(center, radius, fill=False, edgecolor=PALETTE["purple"], linestyle="--", linewidth=1.4))
        # NetworkX may autoscale after drawing; restore deterministic padding
        # so labels at the cardinal points remain inside the page.
        ax.set_xlim(-1.28, 1.28)
        ax.set_ylim(-1.28, 1.28)
    flat_axes[-1].axis("off")
    legend = [
        Line2D([0], [0], color=PALETTE["black"], lw=1, alpha=0.4, label="Physical route"),
        Line2D([0], [0], color=PALETTE["black"], lw=2, alpha=0.7, label="Recent material flow"),
        Line2D([0], [0], color=PALETTE["sky"], lw=1, ls="--", label="Communication"),
        Line2D([0], [0], color=PALETTE["orange"], lw=1.5, label="Recent negotiation"),
        Line2D([0], [0], color=PALETTE["green"], lw=2.2, label="Accepted commitment"),
        Line2D([0], [0], color=PALETTE["purple"], lw=1.4, ls="--", label="Coalition"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PALETTE["black"], label="Source", markersize=6),
        Line2D([0], [0], marker="s", color="none", markeredgecolor=PALETTE["black"], label="Demand", markersize=6),
        Line2D([0], [0], marker="^", color="none", markeredgecolor=PALETTE["black"], label="Transport", markersize=6),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PALETTE["gray"], label="Low pressure/load", markersize=4),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PALETTE["gray"], label="High pressure/load", markersize=9),
    ]
    color_axis = flat_axes[-1].inset_axes([0.14, 0.82, 0.72, 0.055])
    color_scale = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=5.5), cmap="viridis"
    )
    colorbar = fig.colorbar(color_scale, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Local surprisal $I_i$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    flat_axes[-1].legend(
        handles=legend, loc="center", bbox_to_anchor=(0.5, 0.38), ncol=2,
        frameon=False, fontsize=7.5, title="Network encoding",
    )
    fig.suptitle("%s coordination network through disruption and recovery" % application.capitalize())
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    _save(fig, "network_snapshots_%s" % application, results_root)


def agentic_metrics(agent_metrics: pd.DataFrame, results_root: Path) -> None:
    frame = agent_metrics[(agent_metrics["stage"] == "main") & (agent_metrics["completion_status"] == "complete")]
    metrics = [
        "valid_tool_call_rate",
        "agreement_rate",
        "plan_revisions",
        "useful_coalition_precision",
        "commitment_breaches",
        "contradiction_rate",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    methods = [m for m in METHOD_STYLE if m in frame["method"].values]
    for metric_index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        values: List[float] = []
        errors: List[List[float]] = []
        colors: List[str] = []
        for method in methods:
            mean, low, high = _ci(frame[frame["method"] == method][metric].astype(float))
            values.append(mean)
            errors.append([mean - low, high - mean])
            colors.append(METHOD_STYLE[method][0])
        y = np.arange(len(methods))
        if methods:
            for index, (value, error, color) in enumerate(zip(values, errors, colors)):
                ax.errorbar(value, index, xerr=[[error[0]], [error[1]]], fmt="o", color=color, markersize=5, capsize=2, lw=1)
                points = frame[frame["method"] == methods[index]][metric].astype(float).to_numpy()
                offsets = np.linspace(-0.14, 0.14, len(points)) if len(points) > 1 else np.zeros(len(points))
                ax.scatter(
                    points,
                    index + offsets,
                    s=9,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.45,
                    alpha=0.7,
                )
        ax.set_yticks(y)
        if metric_index % 3 == 0:
            ax.set_yticklabels([m.replace("autonomous_", "").replace("_", " ") for m in methods], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(metric.replace("_", " "))
        ax.set_ylim(len(methods) - 0.5, -0.5)
    fig.suptitle("Observable agentic behavior by method", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "agentic_metrics", results_root)


def generate_all(results_root: Path) -> List[str]:
    configure_style()
    episodes = pd.read_csv(results_root / "processed" / "episodes.csv")
    time_series = pd.read_csv(results_root / "processed" / "time_series.csv")
    agents = pd.read_csv(results_root / "processed" / "agent_metrics.csv")
    architecture_figure(results_root)
    entropy_dynamics(time_series, results_root)
    main_performance(episodes, results_root)
    communication_pareto(episodes, results_root)
    necessity_map(episodes, results_root)
    recovery_curves(time_series, results_root)
    ablation_effects(episodes, results_root)
    network_snapshots(results_root, "commercial")
    network_snapshots(results_root, "humanitarian")
    agentic_metrics(agents, results_root)
    return sorted(path.name for path in (results_root / "figures" / "pdf").glob("*.pdf"))


def validate_pdfs(results_root: Path) -> Dict[str, Any]:
    pdfs = sorted((results_root / "figures" / "pdf").glob("*.pdf"))
    preview_dir = results_root / "figures" / "previews"
    qa_dir = results_root / "reproducibility" / "pdf_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    tools = {name: shutil.which(name) for name in ("pdfinfo", "pdffonts", "pdftoppm")}
    poppler_available = all(tools.values())
    pymupdf: Any = None
    if not poppler_available:
        try:
            import pymupdf as pymupdf_module

            pymupdf = pymupdf_module
        except ImportError as error:
            raise RuntimeError(
                "PDF QA requires Poppler tools or the isolated PyMuPDF dependency"
            ) from error
    records = []
    for pdf in pdfs:
        render_prefix = qa_dir / (pdf.stem + "-render")
        render = render_prefix.with_suffix(".png")
        if poppler_available:
            info = subprocess.run([tools["pdfinfo"], str(pdf)], capture_output=True, text=True, check=True).stdout
            fonts = subprocess.run([tools["pdffonts"], str(pdf)], capture_output=True, text=True, check=True).stdout
            subprocess.run([tools["pdftoppm"], "-f", "1", "-singlefile", "-png", "-r", "150", str(pdf), str(render_prefix)], capture_output=True, check=True)
            fonts_detected = len(fonts.splitlines()) > 2
            backend = "poppler"
        else:
            document = pymupdf.open(str(pdf))
            if document.page_count < 1:
                raise RuntimeError("PDF has no pages: %s" % pdf)
            font_rows = sorted({
                "%s | %s | %s" % (row[1], row[2], row[3])
                for page in document
                for row in page.get_fonts(full=True)
            })
            info = json.dumps({
                "page_count": document.page_count,
                "metadata": document.metadata,
                "page_size_points": [
                    float(document[0].rect.width), float(document[0].rect.height)
                ],
                "backend": "PyMuPDF %s" % getattr(pymupdf, "__version__", "unknown"),
            }, indent=2, sort_keys=True)
            fonts = "\n".join(font_rows) + "\n"
            document[0].get_pixmap(dpi=150, alpha=False).save(str(render))
            fonts_detected = bool(font_rows)
            backend = "pymupdf"
            document.close()
        if not render.exists() or render.stat().st_size < 1000:
            raise RuntimeError("render failed for %s" % pdf)
        (qa_dir / (pdf.stem + ".pdfinfo.txt")).write_text(info, encoding="utf-8")
        (qa_dir / (pdf.stem + ".fonts.txt")).write_text(fonts, encoding="utf-8")
        records.append({"pdf": pdf.name, "opens": True, "rendered": str(render.relative_to(results_root)), "fonts_detected": fonts_detected, "validation_backend": backend, "visual_inspection": "pending manual preview review"})
    report = {"tools": tools, "pymupdf_available": pymupdf is not None, "figures": records}
    (qa_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def mark_visual_qa(results_root: Path, reviewer: str, note: str) -> Dict[str, Any]:
    """Record a human/vision review after rendered previews were inspected."""
    report_path = results_root / "reproducibility" / "pdf_qa" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("figures"):
        raise ValueError("no rendered figures are available for visual review")
    for record in report["figures"]:
        render = results_root / record["rendered"]
        if not record.get("opens") or not record.get("fonts_detected") or not render.exists():
            raise ValueError("mechanical QA is incomplete for %s" % record.get("pdf"))
        record["visual_inspection"] = "passed"
        record["visual_inspection_note"] = note
        record["visual_reviewer"] = reviewer
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
