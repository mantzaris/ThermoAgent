"""Publication-facing vector figures for the ThermoHITL v3 no-go study."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .dashboard.replay import DashboardReplay, frame_svg


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#20242A",
    "gray": "#6C7480",
    "light": "#E9EDF2",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _save(fig: Any, name: str, root: Path) -> str:
    pdf_dir = root / "figures" / "pdf"
    preview_dir = root / "figures" / "previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / (name + ".pdf")
    preview = preview_dir / (name + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(preview, format="png", dpi=240)
    plt.close(fig)
    return pdf.name


def _blocked(name: str, root: Path, title: str, detail: str) -> str:
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch(
        (0.05, 0.20), 0.90, 0.60, transform=ax.transAxes,
        boxstyle="round,pad=0.025", facecolor="#F4F5F7",
        edgecolor=COLORS["red"], linewidth=1.6,
    ))
    ax.text(0.5, 0.62, "PROSPECTIVELY NOT RUN", transform=ax.transAxes,
            ha="center", va="center", fontsize=15, weight="bold", color=COLORS["red"])
    ax.text(0.5, 0.43, detail, transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, wrap=True)
    ax.text(0.5, 0.29, "Zero observations are displayed; no values were imputed.",
            transform=ax.transAxes, ha="center", va="center", color=COLORS["gray"])
    ax.set_title(title)
    record_root = root / "reproducibility" / "not_run_figures"
    record_root.mkdir(parents=True, exist_ok=True)
    pdf = record_root / (name + ".pdf")
    preview = record_root / (name + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(preview, format="png", dpi=240)
    plt.close(fig)
    return str(pdf.relative_to(root))


def _summary(root: Path, stage: str) -> pd.DataFrame:
    path = root / stage / "episode_summary.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _episode_paths(root: Path, stage: str) -> List[Path]:
    return sorted((root / "raw" / stage).glob("*/episode.json"))


def _episode(
    root: Path,
    application: str,
    regime: str = "correlated",
    stage: str = "development_trigger_candidate_n10_v4",
) -> Tuple[Path, Dict[str, Any]]:
    candidates = [
        path for path in _episode_paths(root, stage)
        if ("-" + application + "-") in path.parent.name
        and ("-" + regime + "-") in path.parent.name
    ]
    if not candidates:
        candidates = [
            path for path in _episode_paths(root, stage)
            if ("-" + application + "-") in path.parent.name
        ]
    if not candidates:
        raise FileNotFoundError("no v3 episode for %s/%s" % (application, regime))
    path = candidates[len(candidates) // 2]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _events(episode_path: Path) -> List[Dict[str, Any]]:
    path = next(iter(sorted(episode_path.parent.glob("events.jsonl*"))))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _regime(scenario: str) -> str:
    for value in ("nominal", "moderate", "correlated", "compound"):
        if "-" + value + "-" in str(scenario):
            return value
    return "other"


def export_populated_dashboard_replays(root: Path) -> List[str]:
    """Export real, schema-validated replay frames rather than a mock dashboard."""

    export_root = root / "dashboard" / "populated_replays"
    export_root.mkdir(parents=True, exist_ok=True)
    exported: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian"):
        episode_path, _ = _episode(root, application, "compound")
        replay = DashboardReplay(episode_path)
        frame = max(
            replay.frames,
            key=lambda value: (
                bool(value.explanation.get("features")),
                len(value.alert_queue),
                len(value.interventions),
            ),
        )
        output = export_root / (application + "_populated_replay.svg")
        output.write_text(frame_svg(frame) + "\n", encoding="utf-8")
        exported.append(str(output.relative_to(root)))
        metadata.append({
            **replay.metadata(),
            "selected_step": frame.step,
            "export": str(output.relative_to(root)),
            "selection_rule": "authorized payload present, then largest queue and intervention history",
            "data_populated": True,
        })
    metadata_path = export_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    exported.append(str(metadata_path.relative_to(root)))
    return exported


def thermohitl_architecture(root: Path) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    boxes = [
        (0.25, 5.2, 2.55, 1.5, "Independent agent\nprivate state · memory\nutility · commitments", "blue"),
        (3.25, 5.2, 2.55, 1.5, "Distributed monitor\nenergy · entropy · slope\ndisagreement · confidence", "green"),
        (6.25, 5.2, 2.40, 1.5, "Independent request\nadjustable-autonomy\nstate", "orange"),
        (9.10, 5.2, 2.55, 1.5, "Attention allocator\nfinite slots · queue\nworkload · latency", "purple"),
        (9.00, 2.2, 2.65, 1.5, "Simulated operator\nauthorized view\nbounded intervention", "red"),
        (5.35, 2.2, 2.65, 1.5, "Typed directive\nauthority / information\nchange", "orange"),
        (1.70, 2.2, 2.65, 1.5, "Autonomous response\nvalidated commitment/action\nmaterial flow", "blue"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=mpl.colors.to_rgba(COLORS[color], 0.11),
            edgecolor=COLORS[color], linewidth=1.5,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)
    arrows = [
        ((2.80, 5.95), (3.25, 5.95)), ((5.80, 5.95), (6.25, 5.95)),
        ((8.65, 5.95), (9.10, 5.95)), ((10.40, 5.2), (10.3, 3.7)),
        ((9.00, 2.95), (8.00, 2.95)), ((5.35, 2.95), (4.35, 2.95)),
        ((2.3, 3.7), (1.55, 5.2)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": COLORS["black"]})
    ax.annotate("", xy=(8.8, 1.0), xytext=(3.1, 1.0),
                arrowprops={"arrowstyle": "<->", "lw": 1.2, "linestyle": "--", "color": COLORS["gray"]})
    ax.text(5.95, 1.30, "Event ledger: view hash → operator action → agent action → arrival → loss",
            ha="center", fontsize=9, color=COLORS["gray"])
    ax.text(6, 0.48, "No hidden central domain planner; evaluator-global state is analysis-only",
            ha="center", fontsize=8.8, color=COLORS["gray"])
    ax.set_title("ThermoHITL adaptive-autonomy architecture")
    return _save(fig, "thermohitl_architecture", root)


def _draw_network(ax: Any, frame: Any, title: str) -> None:
    network = frame.network
    nodes = network.get("nodes", [])
    positions = {node["agent_id"]: np.asarray(node.get("location", [0, 0]), dtype=float) for node in nodes}
    for left, right in network.get("physical_edges", []):
        if left in positions and right in positions:
            p, q = positions[left], positions[right]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#9AA3AD", lw=2.3, zorder=1)
    for left, right in network.get("communication_edges", []):
        if left in positions and right in positions:
            p, q = positions[left], positions[right]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=COLORS["sky"], lw=0.9,
                    ls="--", alpha=0.7, zorder=1)
    for node in nodes:
        p = positions[node["agent_id"]]
        level = int(node.get("autonomy_level", 0))
        ax.scatter(p[0], p[1], s=85 + 30 * level, color=COLORS["blue"],
                   edgecolor=COLORS["red"] if level >= 3 else COLORS["black"], lw=1.3, zorder=3)
        abbreviations = {
            "manufacturer": "M", "supplier": "S", "carrier": "C",
            "warehouse": "W", "retailer": "R", "ngo": "N",
            "agency": "A", "transport": "T", "depot": "D",
            "clinic": "Cl", "community": "Co",
        }
        role = str(node.get("role", "agent"))
        suffix = str(node["agent_id"]).rsplit("_", 1)[-1]
        label = abbreviations.get(role, role[:2].title()) + suffix
        ax.text(p[0], p[1] - 0.13, label, ha="center", va="top", fontsize=9.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3})
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect("equal"); ax.axis("off")


def operator_dashboard_overview(root: Path) -> str:
    path, _ = _episode(root, "commercial")
    replay = DashboardReplay(path)
    frame = max(
        replay.frames,
        key=lambda value: (
            bool(value.explanation.get("features")),
            len(value.alert_queue),
            len(value.interventions),
        ),
    )
    fig = plt.figure(figsize=(8.4, 5.2))
    ax_net = fig.add_axes((0.035, 0.25, 0.27, 0.62))
    _draw_network(ax_net, frame, "Network and autonomy")
    axes = [
        fig.add_axes((0.345, 0.57, 0.285, 0.28)),
        fig.add_axes((0.675, 0.57, 0.285, 0.28)),
        fig.add_axes((0.345, 0.25, 0.285, 0.28)),
        fig.add_axes((0.675, 0.25, 0.285, 0.28)),
    ]
    thermo = frame.thermodynamics
    latest = frame.interventions[-1] if frame.interventions else {}
    prediction = frame.explanation.get("prediction", {})
    labels = [
        ("Thermodynamic state", [
            "energy %.3f" % thermo["energy"],
            "entropy %.3f · anomaly %.2fσ" % (thermo["entropy"], thermo["entropy_anomaly"]),
            "disagreement %.3f" % thermo["disagreement"],
            "consensus confidence %.3f" % thermo["consensus_confidence"],
        ]),
        ("Alert queue", [
            "%d queued incident(s)" % len(frame.alert_queue),
            "reason %s" % (frame.explanation.get("alert_reason") or "none"),
            "predicted collapse %s periods" % (prediction.get("predicted_steps_until_collapse", "—")),
            "view payload hashes %d" % len(frame.view_hashes),
        ]),
        ("Latest bounded intervention", [
            "event %s" % latest.get("event", "none"),
            "tool %s" % latest.get("tool", "—"),
            "status %s" % latest.get("code", "—"),
            "autonomy level %d" % thermo["autonomy_level"],
        ]),
        ("Operator workload", [
            "workload %.3f" % frame.workload["workload"],
            "active interventions %d" % frame.workload["active_interventions"],
            "fatigue %.3f" % frame.workload["fatigue"],
            "minutes %.1f" % frame.workload["operator_minutes"],
        ]),
    ]
    for ax, (title, lines) in zip(axes, labels):
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0.03, 0.05), 0.94, 0.88, transform=ax.transAxes,
                     boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#BBC4CE"))
        ax.text(0.08, 0.82, title, transform=ax.transAxes, weight="bold", fontsize=10.2)
        for index, line in enumerate(lines):
            ax.text(0.10, 0.64 - 0.15 * index, "• " + line, transform=ax.transAxes, fontsize=9.2)
    bottom = fig.add_axes((0.045, 0.035, 0.91, 0.16)); bottom.axis("off")
    bottom.text(0.00, 0.78, "Execution-time information boundary", weight="bold", fontsize=9.5)
    bottom.text(0.00, 0.44,
                "The simulated operator and dashboard consume the same schema-validated, hashed payload. "
                "Private observations, RNG state, future disruptions, and counterfactual outcomes are excluded.",
                fontsize=8.2, wrap=True)
    bottom.text(0.00, 0.05,
                "Replay controls: play · pause · step · rewind · jump to alert · compare branch · export SVG",
                color=COLORS["gray"], fontsize=8.7)
    fig.suptitle(
        "Functional ThermoHITL operator dashboard — populated ledger replay, step %d" % frame.step,
        y=0.97,
    )
    return _save(fig, "operator_dashboard_overview", root)


def energy_entropy_phase_plane(root: Path) -> str:
    calibration = json.loads(
        (root / "calibration" / "thermodynamic_calibration_n10.json").read_text(encoding="utf-8")
    )["applications"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.25), sharex=True, sharey=True)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        episode_path, episode = _episode(root, application, "compound")
        frame = pd.DataFrame(episode["time_series"])
        values = calibration[application]
        z_energy = (
            frame.distributed_energy_mean.to_numpy(dtype=float) - float(values["energy_center"])
        ) / float(values["energy_scale"])
        entropy_anomaly = frame.entropy_anomaly_mean.to_numpy(dtype=float)

        x_grid = np.linspace(0.0, max(7.0, float(entropy_anomaly.max()) + 0.5), 180)
        y_grid = np.linspace(min(-2.0, float(z_energy.min()) - 0.5), max(7.0, float(z_energy.max()) + 0.5), 180)
        xx, yy = np.meshgrid(x_grid, y_grid)
        # This is the exact two-variable projection of the frozen prospective
        # v3 rule. Slope, disagreement, local disruption risk, and workload are
        # held at their nominal zero residuals; actual requests retain all terms.
        projected_score = 0.30 * np.maximum(yy, 0.0) + 0.22 * xx
        regions = np.where(projected_score >= 1.1, 2, np.where(projected_score >= 0.6, 1, 0))
        ax.contourf(
            xx,
            yy,
            regions,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=[COLORS["green"], COLORS["orange"], COLORS["red"]],
            alpha=0.09,
        )
        boundary = (1.1 - 0.22 * x_grid) / 0.30
        ax.plot(x_grid, boundary, color=COLORS["red"], ls="--", lw=1.3,
                label="projected actionable boundary")
        ax.add_patch(patches.Rectangle(
            (0.0, -1.0), 1.0, 2.0, facecolor="none", edgecolor=COLORS["green"],
            lw=1.1, ls=":", label="nominal ±1σ region",
        ))
        ax.plot(entropy_anomaly, z_energy, color=COLORS["black"], lw=1.25,
                marker="o", ms=3.2, label="episode trajectory")
        events = _events(episode_path)
        disruption_steps = [int(event["step"]) for event in events if event["kind"] == "disruption"]
        intervention_steps = [int(event["step"]) for event in events if event["kind"] == "operator_action"]
        disruption = disruption_steps[0] if disruption_steps else max(2, len(frame) // 3)
        if 0 <= disruption < len(frame):
            ax.scatter(entropy_anomaly[disruption], z_energy[disruption], s=65, marker="X",
                       color=COLORS["orange"], edgecolor=COLORS["black"], linewidth=0.5,
                       zorder=5, label="disruption")
        if intervention_steps:
            step = intervention_steps[0]
            ax.scatter(entropy_anomaly[step], z_energy[step], s=95, marker="*",
                       color=COLORS["purple"], edgecolor=COLORS["black"], linewidth=0.5,
                       zorder=5, label="first intervention")
        ax.scatter(entropy_anomaly[0], z_energy[0], s=45, marker="s",
                   color=COLORS["green"], edgecolor=COLORS["black"], linewidth=0.5,
                   label="start")
        ax.set_title(application.capitalize())
        ax.set_xlabel("Entropy anomaly, |Ŝ − μₛ| / σₛ")
        ax.set_xlim(x_grid.min(), x_grid.max()); ax.set_ylim(y_grid.min(), y_grid.max())
    axes[0].set_ylabel("Standardized operational energy, (Ê − μₑ) / σₑ")
    handles, labels = axes[0].get_legend_handles_labels()
    handles = [
        patches.Patch(facecolor=mpl.colors.to_rgba(COLORS["green"], 0.12), label="quiet/autonomy region"),
        patches.Patch(facecolor=mpl.colors.to_rgba(COLORS["orange"], 0.12), label="monitor or abstain region"),
        patches.Patch(facecolor=mpl.colors.to_rgba(COLORS["red"], 0.12), label="alert-eligible region"),
    ] + handles
    labels = [item.get_label() for item in handles]
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Data-derived energy–entropy phase plane — development evidence\n"
        "Shading is the frozen 2-D rule projection; actual alerts use all authorized local terms"
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    return _save(fig, "energy_entropy_phase_plane", root)


def network_operator_sequence(root: Path) -> str:
    path, episode = _episode(root, "commercial", "correlated")
    replay = DashboardReplay(path)
    intervention_steps = [row["step"] for row in episode.get("counterfactuals", [])]
    disruption = max(2, len(replay.frames) // 3)
    first = episode["operator_metrics"].get("first_post_disruption_request_step") or disruption
    intervention = intervention_steps[0] if intervention_steps else min(first + 2, len(replay.frames) - 1)
    steps = [0, disruption, first, intervention, len(replay.frames) - 1]
    titles = ["quiet autonomy", "disruption", "request/queue", "bounded intervention", "post-intervention"]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.2))
    for ax, step, title in zip(axes.ravel(), steps, titles):
        _draw_network(ax, replay.frame(step), "%s\n$t=%d$" % (title, step))
    legend_ax = axes.ravel()[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([0], [0], color="#9AA3AD", lw=2.3, label="physical logistics edge"),
        Line2D([0], [0], color=COLORS["sky"], lw=1.2, ls="--", label="communication edge"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"],
               markeredgecolor=COLORS["black"], markersize=9, label="quiet agent"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"],
               markeredgecolor=COLORS["red"], markeredgewidth=1.5, markersize=11,
               label="elevated autonomy"),
    ], frameon=False, loc="center", fontsize=10.5)
    fig.suptitle("Commercial event sequence: network, communication, and autonomy level")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.4)
    return _save(fig, "network_operator_sequence", root)


def trigger_and_intervention_dynamics(root: Path) -> str:
    episode_path, episode = _episode(root, "humanitarian", "compound")
    frame = pd.DataFrame(episode["time_series"])
    recorded = _events(episode_path)
    priority = np.full(len(frame), np.nan)
    intervention_steps: List[int] = []
    for event in recorded:
        if event["kind"] == "operator_view":
            step = int(event["step"])
            score = event["payload"].get("payload", {}).get("incident", {}).get("priority_score")
            if score is not None and 0 <= step < len(priority):
                priority[step] = max(float(score), priority[step]) if np.isfinite(priority[step]) else float(score)
        elif event["kind"] == "operator_action":
            intervention_steps.append(int(event["step"]))

    def scaled(column: str) -> np.ndarray:
        values = frame[column].to_numpy(dtype=float)
        low, high = float(values.min()), float(values.max())
        return (values - low) / max(high - low, 1e-12)

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 8.1), sharex=True)
    axes[0].plot(frame.step, frame.distributed_energy_mean, color=COLORS["orange"], lw=1.5, label="energy")
    twin0 = axes[0].twinx()
    twin0.plot(frame.step, frame.entropy_anomaly_mean, color=COLORS["blue"], ls="--", lw=1.35, label="entropy anomaly")
    twin0.axhline(1.5, color=COLORS["blue"], ls=":", lw=0.9, label=r"$\tau_{on}=1.5$")
    axes[0].set_ylabel("Energy"); twin0.set_ylabel("Entropy anomaly", color=COLORS["blue"])
    handles = axes[0].get_lines() + twin0.get_lines()
    axes[0].legend(handles, [item.get_label() for item in handles], frameon=False, loc="upper left", ncol=3)

    axes[1].plot(frame.step, scaled("entropy_slope_mean"), color=COLORS["purple"], label="entropy slope")
    axes[1].plot(frame.step, scaled("exact_free_energy_diagnostic"), color=COLORS["gray"], ls="--", label="free energy diagnostic")
    axes[1].plot(frame.step, scaled("disagreement_mean"), color=COLORS["green"], ls=":", label="disagreement")
    axes[1].set_ylabel("Within-series\nnormalized")
    axes[1].legend(frameon=False, loc="upper left", ncol=3)

    axes[2].scatter(frame.step, priority, color=COLORS["red"], marker="D", s=25, label="intervention priority score")
    axes[2].fill_between(frame.step, 0, 1, where=frame.operator_queue_length.to_numpy() > 0,
                         transform=axes[2].get_xaxis_transform(), color=COLORS["purple"], alpha=0.10,
                         label="alert queued")
    axes[2].set_ylabel("Priority score")
    axes[2].legend(frameon=False, loc="upper left", ncol=2)

    axes[3].plot(frame.step, frame.operator_workload, color=COLORS["red"], label="operator workload")
    twin3 = axes[3].twinx()
    twin3.step(frame.step, frame.maximum_autonomy_level, where="post", color=COLORS["orange"], ls="--", label="autonomy level")
    axes[3].set_ylabel("Workload"); twin3.set_ylabel("Autonomy level", color=COLORS["orange"])
    handles = axes[3].get_lines() + twin3.get_lines()
    axes[3].legend(handles, [item.get_label() for item in handles], frameon=False, loc="upper left", ncol=2)

    axes[4].plot(frame.step, frame.service_loss, color=COLORS["black"], lw=1.6)
    axes[4].set_ylabel("Service loss")
    disruption = max(2, len(frame) // 3)
    request = episode["operator_metrics"].get("first_post_disruption_request_step")
    for ax in axes:
        ax.axvline(disruption, color=COLORS["black"], ls="--", lw=0.9)
        if request is not None:
            ax.axvline(request, color=COLORS["red"], ls=":", lw=1.1)
        for step in intervention_steps:
            ax.axvline(step, color=COLORS["purple"], alpha=0.18, lw=0.7)
    axes[-1].set_xlabel("Simulator period")
    fig.suptitle("Trigger, simulated-operator intervention, and service dynamics\nDevelopment case")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.7)
    return _save(fig, "trigger_and_intervention_dynamics", root)


def operator_view_incremental_value(root: Path) -> str:
    source = pd.read_csv(root / "monitoring" / "causal_allocation_utility.csv")
    names = ["causal_local_kpi_logistic", "causal_kpi_plus_thermodynamic_logistic"]
    source = source[source.detector.isin(names)].copy()
    source["regime"] = source.scenario.map(_regime)
    paired = source.pivot_table(
        index=["application", "run_id", "regime"],
        columns="detector",
        values="realized_counterfactual_utility",
        aggfunc="first",
    ).dropna().reset_index()
    paired["difference"] = paired[names[1]] - paired[names[0]]

    def interval(values: np.ndarray, denominator: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        values = np.asarray(values, dtype=float)
        rng = np.random.RandomState(42031)
        indices = rng.randint(0, len(values), size=(10000, len(values)))
        if denominator is None:
            draws = values[indices].mean(axis=1)
            point = float(values.mean())
        else:
            denominator = np.asarray(denominator, dtype=float)
            draws = values[indices].mean(axis=1) / np.maximum(
                np.abs(denominator[indices].mean(axis=1)), 1e-12
            )
            point = float(values.mean() / max(abs(float(denominator.mean())), 1e-12))
        low, high = np.quantile(draws, [0.025, 0.975])
        return point, float(low), float(high)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    app_order = ("commercial", "humanitarian")
    x = np.arange(len(app_order)); width = 0.34
    for detector_index, (detector, label, color) in enumerate((
        (names[0], "KPI only", COLORS["blue"]),
        (names[1], "KPI + thermodynamics", COLORS["orange"]),
    )):
        means = []
        errors = [[], []]
        for application in app_order:
            values = paired.loc[paired.application == application, detector].to_numpy(dtype=float)
            point, low, high = interval(values)
            means.append(point); errors[0].append(point - low); errors[1].append(high - point)
        axes[0, 0].bar(x + (detector_index - 0.5) * width, means, width, color=color,
                       label=label, alpha=0.88)
        axes[0, 0].errorbar(x + (detector_index - 0.5) * width, means, yerr=errors,
                            fmt="none", ecolor=COLORS["black"], capsize=3, lw=1)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels([item.capitalize() for item in app_order])
    axes[0, 0].set_ylabel("Budgeted causal utility")
    axes[0, 0].set_title("Absolute utility at matched budget")
    axes[0, 0].legend(frameon=False)

    difference_points = []
    difference_errors = [[], []]
    relative_points = []
    relative_errors = [[], []]
    for application in app_order:
        selected = paired[paired.application == application]
        point, low, high = interval(selected.difference.to_numpy(dtype=float))
        difference_points.append(point); difference_errors[0].append(point - low); difference_errors[1].append(high - point)
        point, low, high = interval(
            selected.difference.to_numpy(dtype=float), selected[names[0]].to_numpy(dtype=float)
        )
        relative_points.append(100 * point); relative_errors[0].append(100 * (point - low)); relative_errors[1].append(100 * (high - point))
    colors = [COLORS["red"], COLORS["green"]]
    axes[0, 1].bar(x, difference_points, color=colors, alpha=0.88)
    axes[0, 1].errorbar(x, difference_points, yerr=difference_errors, fmt="none",
                        ecolor=COLORS["black"], capsize=4, lw=1)
    axes[0, 1].axhline(0, color=COLORS["black"], lw=0.9)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels([item.capitalize() for item in app_order])
    axes[0, 1].set_ylabel("Paired utility difference")
    axes[0, 1].set_title("Thermodynamic view − KPI-only")

    axes[1, 0].bar(x, relative_points, color=colors, alpha=0.88)
    axes[1, 0].errorbar(x, relative_points, yerr=relative_errors, fmt="none",
                        ecolor=COLORS["black"], capsize=4, lw=1)
    axes[1, 0].axhline(5, color=COLORS["black"], ls="--", lw=1, label="prospective +5% gate")
    axes[1, 0].axhline(0, color=COLORS["gray"], lw=0.8)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels([item.capitalize() for item in app_order])
    axes[1, 0].set_ylabel("Relative utility gain (%)")
    axes[1, 0].set_title("Relative gain with cluster bootstrap CI")
    axes[1, 0].legend(frameon=False)

    regime_order = ["moderate", "correlated", "compound"]
    positions = np.arange(len(regime_order))
    for index, (application, marker, color) in enumerate((
        ("commercial", "o", COLORS["blue"]),
        ("humanitarian", "s", COLORS["orange"]),
    )):
        points = []
        errors = [[], []]
        for regime in regime_order:
            values = paired.loc[
                (paired.application == application) & (paired.regime == regime), "difference"
            ].to_numpy(dtype=float)
            point, low, high = interval(values)
            points.append(point); errors[0].append(point - low); errors[1].append(high - point)
        offset = (index - 0.5) * 0.12
        axes[1, 1].errorbar(positions + offset, points, yerr=errors, color=color,
                            marker=marker, capsize=3, lw=1.2, label=application.capitalize())
    axes[1, 1].axhline(0, color=COLORS["black"], lw=0.9)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels([item.capitalize() + "\n(n=2/app)" for item in regime_order])
    axes[1, 1].set_ylabel("Paired utility difference")
    axes[1, 1].set_title("Per-regime decision utility")
    axes[1, 1].legend(frameon=False)

    counts = paired.groupby("application").run_id.nunique().to_dict()
    fig.suptitle(
        "Operator-view incremental value — DEVELOPMENT ONLY\n"
        "Independent matched environment panels: commercial n=%d; humanitarian n=%d"
        % (counts.get("commercial", 0), counts.get("humanitarian", 0))
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.7, w_pad=1.3)
    return _save(fig, "operator_view_incremental_value", root)


def loss_operator_effort_pareto(root: Path) -> str:
    frame = _summary(root, "development_gate_preliminary_v3_n10")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55))
    styles = {
        "autonomous_no_human": (COLORS["gray"], "o"),
        "local_kpi_trigger": (COLORS["blue"], "s"),
        "thermohitl_rule": (COLORS["green"], "^"),
    }
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        selected = frame[(frame.application == application) & ~frame.scenario.str.contains("-nominal-")]
        for method, group in selected.groupby("method"):
            color, marker = styles.get(method, (COLORS["gray"], "o"))
            ax.scatter(group.operator_minutes, group.primary_outcome, s=20, alpha=0.35,
                       color=color, marker=marker)
            ax.scatter(group.operator_minutes.mean(), group.primary_outcome.mean(), s=80,
                       color=color, marker=marker, edgecolor=COLORS["black"], lw=0.7,
                       label=method.replace("_", " "))
        ax.set_ylabel("Primary loss (lower is better)")
        ax.set_title(application.capitalize())
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.text(0.43, 0.02, "Simulated-operator minutes per episode", ha="center")
    fig.suptitle("Development loss–operator-effort frontier (not confirmatory)")
    fig.tight_layout(rect=(0, 0.07, 0.86, 0.92))
    return _save(fig, "loss_operator_effort_pareto", root)


def primary_effect_forest(root: Path) -> str:
    path = root / "statistics" / "development_paired_effects.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.regime == "aggregate"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    y = np.arange(len(frame))
    mean = 100 * frame.mean_relative_difference.to_numpy()
    low = 100 * frame.relative_ci95_low.to_numpy(); high = 100 * frame.relative_ci95_high.to_numpy()
    colors = [COLORS["blue"] if "human" in value else COLORS["orange"] for value in frame.comparison]
    for idx in range(len(frame)):
        ax.errorbar(mean[idx], y[idx], xerr=[[mean[idx] - low[idx]], [high[idx] - mean[idx]]],
                    marker="o", color=colors[idx], capsize=3, lw=1.4)
    labels = [
        "%s: %s" % (
            row.application.capitalize(),
            "KPI-triggered human vs autonomy"
            if "KPI-triggered" in row.comparison
            else "fixed communication vs no communication",
        )
        for row in frame.itertuples()
    ]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color=COLORS["black"], lw=0.9)
    ax.set_xlabel("Relative loss difference, treatment − reference (%)\nnegative favors treatment")
    ax.set_title("Development paired effects with 10,000-replicate bootstrap intervals\n"
                 "Exploratory evidence; no confirmatory holdout", fontsize=11)
    fig.tight_layout()
    return _save(fig, "primary_effect_forest", root)


def causal_intervention_effects(root: Path) -> str:
    frame = pd.read_csv(root / "counterfactuals" / "paired_intervention_effects.csv")
    frame = frame[frame.method.isin(["local_kpi_trigger", "thermohitl_rule"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=False)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        groups = []
        labels = []
        for method in ("local_kpi_trigger", "thermohitl_rule"):
            values = frame[(frame.application == application) & (frame.method == method)].intervention_effect.astype(float)
            groups.append(values.to_numpy()); labels.append(method.replace("_", " "))
        ax.violinplot(groups, showmedians=True, widths=0.75)
        for idx, values in enumerate(groups, start=1):
            jitter = np.linspace(-0.08, 0.08, min(len(values), 80))
            sample = values[:80]
            ax.scatter(idx + jitter[:len(sample)], sample, s=8, alpha=0.35, color=COLORS["blue"])
        ax.axhline(0, color=COLORS["black"], lw=0.8)
        ax.set_xticks((1, 2)); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Paired loss without − with intervention")
        ax.set_title(application.capitalize())
    fig.suptitle("Per-intervention paired counterfactual effects — development evidence")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "causal_intervention_effects", root)


def intervention_funnel(root: Path) -> str:
    episodes = list(_episodes_for_figure(root, "development_trigger_candidate_n10_v4"))
    causal = [probe for row in episodes for probe in row.get("counterfactuals", [])]
    autonomous_chain = [
        ("Accepted actions", sum(row["metrics"]["material_actions_accepted"] for row in episodes)),
        ("Entered transit", sum(row["metrics"]["material_actions_entered_transit"] for row in episodes)),
        ("Reached next stage", sum(row["metrics"]["material_actions_next_stage"] for row in episodes)),
        ("Reached demand", sum(row["metrics"]["material_actions_reached_demand"] for row in episodes)),
    ]
    probe_chain = [
        ("Paired probes", len(causal)),
        ("Agent accepted", sum(bool(row.get("agent_accepted")) for row in causal)),
        ("Material accepted", sum(bool(row.get("material_action_accepted")) for row in causal)),
        ("Reached next stage", sum(bool(row.get("material_reached_next_stage")) for row in causal)),
        ("Reached demand", sum(bool(row.get("material_reached_demand")) for row in causal)),
        ("Outcome changed", sum(bool(row.get("primary_outcome_changed")) for row in causal)),
        ("Beneficial complete chain", sum(
            bool(row.get("agent_accepted") and row.get("material_action_accepted")
                 and row.get("material_reached_demand") and row.get("primary_outcome_changed")
                 and row.get("intervention_effect", 0) > 0) for row in causal
        )),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    for ax, chain, title, color in (
        (axes[0], autonomous_chain, "Episode-wide autonomous actions", COLORS["blue"]),
        (axes[1], probe_chain, "Paired intervention probes only", COLORS["purple"]),
    ):
        y = np.arange(len(chain)); values = np.asarray([value for _, value in chain], dtype=float)
        bars = ax.barh(y, values, color=color, alpha=0.83, edgecolor=COLORS["black"], lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([label for label, _ in chain]); ax.invert_yaxis()
        ax.set_xlabel("Count")
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.015, bar.get_y() + bar.get_height() / 2,
                    "%d" % value, va="center", fontsize=10)
        ax.set_xlim(0, max(values) * 1.16)
        ax.set_title(title)
    requests = sum(row["metrics"]["operator_requests"] for row in episodes)
    interventions = sum(row["metrics"]["operator_interventions"] for row in episodes)
    fig.text(
        0.5, 0.015,
        "Distinct denominators: left = all autonomous material actions; right = %d counterfactual probes. "
        "Episode context: %d requests and %d interventions."
        % (len(causal), requests, interventions),
        ha="center", fontsize=10, color=COLORS["gray"],
    )
    fig.suptitle("Separate actionability and causal-chain populations — development only")
    fig.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=2.2)
    return _save(fig, "intervention_funnel", root)


def _episodes_for_figure(root: Path, stage: str) -> Iterable[Dict[str, Any]]:
    for path in _episode_paths(root, stage):
        yield json.loads(path.read_text(encoding="utf-8"))


def operator_workload_performance(root: Path) -> str:
    frame = _summary(root, "development_trigger_candidate_n10_v4")
    disrupted = frame[~frame.scenario.str.contains("-nominal-")].copy()
    disrupted["relative_loss"] = disrupted.groupby("application")["primary_outcome"].transform(
        lambda values: values / max(float(values.median()), 1e-9)
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for application, marker, color in (("commercial", "o", "blue"), ("humanitarian", "s", "orange")):
        group = disrupted[disrupted.application == application]
        axes[0].scatter(group.operator_minutes, group.relative_loss, marker=marker,
                        color=COLORS[color], alpha=0.7, label=application.capitalize())
        axes[1].scatter(group.operator_requests, group.relative_loss, marker=marker,
                        color=COLORS[color], alpha=0.7, label=application.capitalize())
    axes[0].set_xlabel("Simulated-operator minutes"); axes[1].set_xlabel("Operator requests")
    for ax in axes:
        ax.set_ylabel("Primary loss / application median"); ax.legend(frameon=False)
    axes[0].set_title("Effort and outcome"); axes[1].set_title("Queue demand and outcome")
    fig.suptitle("Operator workload and performance — final development trigger candidate")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "operator_workload_performance", root)


def attention_allocation_heatmap(root: Path) -> str:
    _, episode = _episode(root, "humanitarian", "compound")
    frame = pd.DataFrame(episode["time_series"])
    raw = np.vstack([
        frame.human_requests.diff().fillna(frame.human_requests).clip(lower=0),
        frame.operator_queue_length,
        frame.operator_active,
        frame.maximum_autonomy_level,
        frame.entropy_anomaly_mean,
        frame.disagreement_mean,
    ]).astype(float)
    minima = raw.min(axis=1, keepdims=True)
    ranges = np.maximum(raw.max(axis=1, keepdims=True) - minima, 1e-12)
    data = (raw - minima) / ranges
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="cividis")
    ax.set_yticks(range(6)); ax.set_yticklabels([
        "new requests", "queue length", "active slots", "max autonomy level",
        "entropy anomaly", "disagreement",
    ])
    ax.set_xlabel("Simulator period"); ax.set_title("Attention allocation over time — humanitarian development case")
    fig.colorbar(image, ax=ax, label="within-row normalized value", fraction=0.025, pad=0.02)
    fig.tight_layout()
    return _save(fig, "attention_allocation_heatmap", root)


def monitoring_incremental_value(root: Path) -> str:
    frame = pd.read_csv(root / "monitoring" / "monitoring_baselines.csv")
    detectors = ["local_kpi_risk", "entropy_anomaly", "energy_severity", "disagreement", "free_energy_diagnostic", "local_kpi_logistic", "kpi_plus_thermodynamic_logistic"]
    labels = ["KPI risk", "Entropy", "Energy", "Disagreement", "Free energy", "KPI model", "KPI + thermo"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        selected = frame[frame.application == application].set_index("detector")
        values = [float(selected.loc[value, "average_precision"]) for value in detectors]
        bars = ax.barh(np.arange(len(values)), values, color=[COLORS["blue"] if "kpi" in value else COLORS["orange"] for value in detectors])
        ax.axvline(float(selected.prevalence.iloc[0]), color=COLORS["black"], ls="--", lw=1, label="prevalence")
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Average precision"); ax.set_title(application.capitalize())
    axes[1].legend(frameon=False)
    fig.suptitle("Monitoring signals versus ordinary same-information KPIs\nDevelopment labels; free energy remains diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "monitoring_incremental_value", root)


def training_seed_curves(root: Path) -> str:
    return _blocked(
        "training_seed_curves", root,
        "Independent RL training seeds",
        "Not run: cross-application thermodynamic information Gate 5 failed before training.\n"
        "Required seeds: ≥5 per primary learned method; observed seeds: 0.",
    )


def trigger_timing_and_false_alarms(root: Path) -> str:
    frame = _summary(root, "development_trigger_candidate_n10_v4")
    frame["regime"] = frame.scenario.map(_regime)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    disrupted = frame[frame.regime != "nominal"]
    rows = disrupted.groupby(["application", "regime"]).agg(
        timely=("timely_activation", "mean"), missed=("missed_activation", "mean"),
        pre_false=("pre_disruption_false_activation", "mean"), requests=("operator_requests", "mean"),
    ).reset_index()
    x = np.arange(3); width = 0.36
    for idx, application in enumerate(("commercial", "humanitarian")):
        selected = rows[rows.application == application].set_index("regime").reindex(["moderate", "correlated", "compound"])
        axes[0].bar(x + (idx - .5) * width, 100 * selected.timely, width,
                    label=application.capitalize(), color=[COLORS["blue"], COLORS["orange"]][idx])
        axes[1].bar(x + (idx - .5) * width, selected.requests, width,
                    label=application.capitalize(), color=[COLORS["blue"], COLORS["orange"]][idx])
    axes[0].axhline(75, color=COLORS["black"], ls="--", lw=1, label="75% gate")
    axes[0].set_ylabel("Timely activation (%)"); axes[1].set_ylabel("Mean requests per episode")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(["Moderate", "Correlated", "Compound"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01))
    axes[1].set_title("Nonzero, non-always-on use")
    axes[0].set_title("Activation before sustained collapse")
    fig.suptitle("Trigger timing and false alarms — final development candidate\nNominal and pre-disruption false activation: 0%")
    fig.tight_layout(rect=(0, 0.14, 1, 0.89))
    return _save(fig, "trigger_timing_and_false_alarms", root)


def partition_robustness(root: Path) -> str:
    rows: List[Dict[str, Any]] = []
    for episode in _episodes_for_figure(root, "dense_causal_development_n10_v1"):
        series = pd.DataFrame(episode["time_series"])
        rows.append({
            "application": episode["application"],
            "communication": "partition" if "-partition-" in episode["scenario"] else "reliable",
            "entropy_rmse": series.entropy_estimation_rmse.mean(),
            "energy_rmse": series.energy_estimation_rmse.mean(),
            "primary_loss": episode["metrics"]["primary_outcome"],
            "requests": episode["metrics"]["operator_requests"],
        })
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.2))
    metrics = [("entropy_rmse", "Entropy estimate RMSE"), ("requests", "Operator requests"), ("primary_loss", "Primary loss")]
    for ax, (metric, label) in zip(axes, metrics):
        grouped = frame.groupby(["application", "communication"])[metric].mean().unstack()
        grouped.plot.bar(ax=ax, color=[COLORS["red"], COLORS["blue"]], width=0.72)
        ax.set_ylabel(label); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
        ax.get_legend().remove()
        if metric == "requests" and float(frame.requests.abs().sum()) == 0.0:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.52, "Zero requests in both\nv1 diagnostic conditions",
                    transform=ax.transAxes, ha="center", va="center", color=COLORS["red"])
    fig.suptitle("Partition robustness — commercial-only development diagnostic")
    handles = [
        patches.Patch(facecolor=COLORS["red"], label="partition"),
        patches.Patch(facecolor=COLORS["blue"], label="reliable"),
    ]
    fig.legend(handles=handles, frameon=False, title="Network", loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.075))
    fig.text(0.5, 0.01,
             "The all-agent v1 diagnostic was efficiency-aborted before humanitarian episodes; it was not promoted.",
             ha="center", fontsize=10, color=COLORS["red"])
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    return _save(fig, "partition_robustness", root)


def _case_study(root: Path, application: str, name: str) -> str:
    _, episode = _episode(root, application, "correlated")
    frame = pd.DataFrame(episode["time_series"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    axes[0, 0].plot(frame.step, frame.distributed_energy_mean, color=COLORS["orange"], label="energy")
    axes[0, 0].plot(frame.step, frame.distributed_entropy_mean, color=COLORS["blue"], ls="--", label="entropy")
    axes[0, 0].legend(frameon=False); axes[0, 0].set_ylabel("Distributed state")
    axes[0, 1].plot(frame.step, frame.entropy_anomaly_mean, color=COLORS["blue"], label="entropy anomaly")
    axes[0, 1].plot(frame.step, frame.disagreement_mean, color=COLORS["green"], ls=":", label="disagreement")
    axes[0, 1].legend(frameon=False); axes[0, 1].set_ylabel("Alert features")
    axes[1, 0].step(frame.step, frame.human_requests, where="post", color=COLORS["purple"], label="requests")
    axes[1, 0].step(frame.step, frame.human_interventions, where="post", color=COLORS["red"], label="interventions")
    axes[1, 0].legend(frameon=False); axes[1, 0].set_ylabel("Cumulative count")
    axes[1, 1].plot(frame.step, frame.service_loss, color=COLORS["black"], label="service loss")
    axes[1, 1].plot(frame.step, frame.fulfillment_rate, color=COLORS["green"], ls="--", label="fulfillment")
    axes[1, 1].legend(frameon=False); axes[1, 1].set_ylabel("Logistics outcome")
    disruption = max(2, len(frame) // 3)
    first = episode["operator_metrics"].get("first_post_disruption_request_step")
    for ax in axes.ravel():
        ax.axvline(disruption, color=COLORS["black"], ls="--", lw=.8)
        if first is not None:
            ax.axvline(first, color=COLORS["red"], ls=":", lw=1)
        ax.set_xlabel("Simulator period")
    fig.suptitle("%s development ledger: thermodynamic alert to material outcome" % application.capitalize())
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, name, root)


def thermodynamic_ablation(root: Path) -> str:
    frame = pd.read_csv(root / "monitoring" / "monitoring_baselines.csv")
    detectors = ["entropy_anomaly", "energy_severity", "free_energy_diagnostic", "disagreement", "kpi_plus_thermodynamic_logistic"]
    labels = ["Entropy only", "Energy only", "Free energy", "Disagreement", "Combined + KPI"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True)
    for ax, application in zip(axes, ("commercial", "humanitarian")):
        selected = frame[frame.application == application].set_index("detector")
        ap = [selected.loc[item, "average_precision"] for item in detectors]
        ax.barh(np.arange(len(ap)), ap, color=[COLORS["blue"], COLORS["orange"], COLORS["gray"], COLORS["green"], COLORS["purple"]])
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Average precision"); ax.set_title(application.capitalize())
    fig.suptitle("Thermodynamic feature ablation — development monitoring labels")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "thermodynamic_ablation", root)


def actionability_diagnostics(root: Path) -> str:
    frame = pd.read_csv(root / "statistics" / "actionability_summary.csv")
    mock = frame[frame.planner == "deterministic mock"]
    first_attempt = frame[frame.stage == "development_real_llm_actionability"]
    retry = frame[frame.stage == "development_real_llm_actionability_retry1"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55), sharey=True)
    metrics = ["first_pass_validity", "valid_after_one_repair", "accepted_to_next_stage", "accepted_to_demand"]
    labels = ["First pass", "After ≤1 repair", "Accepted → next stage", "Accepted → demand"]
    thresholds = np.asarray([90, 98, 70, 30], dtype=float)
    x = np.arange(len(metrics)); width = .36
    panels = [
        (mock, "Deterministic mechanics"),
        (first_attempt, "Retained Qwen v8 failure"),
        (retry, "Qualified Qwen v9 retry"),
    ]
    for axis_index, (selected, title) in enumerate(panels):
        ax = axes[axis_index]
        for idx, application in enumerate(("commercial", "humanitarian")):
            row = selected[selected.application == application]
            values = [float(row[item].iloc[0]) if not row.empty else 0 for item in metrics]
            ax.bar(x + (idx - .5) * width, 100 * np.asarray(values), width,
                   color=[COLORS["blue"], COLORS["orange"]][idx], label=application.capitalize())
        ax.scatter(x, thresholds, marker="_", s=220, linewidths=1.3,
                   color=COLORS["black"], zorder=5, label="prospective threshold" if axis_index == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Rate (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("V3 structured-output and material-action funnel")
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    return _save(fig, "actionability_diagnostics", root)


def run(root: Path) -> Dict[str, Any]:
    configure_style()
    dashboard_exports = export_populated_dashboard_replays(root)
    figures = [
        thermohitl_architecture(root),
        operator_dashboard_overview(root),
        energy_entropy_phase_plane(root),
        network_operator_sequence(root),
        trigger_and_intervention_dynamics(root),
        operator_view_incremental_value(root),
        loss_operator_effort_pareto(root),
        primary_effect_forest(root),
        causal_intervention_effects(root),
        intervention_funnel(root),
        operator_workload_performance(root),
        attention_allocation_heatmap(root),
        monitoring_incremental_value(root),
        training_seed_curves(root),
        trigger_timing_and_false_alarms(root),
        partition_robustness(root),
        _case_study(root, "commercial", "commercial_case_study"),
        _case_study(root, "humanitarian", "humanitarian_case_study"),
        thermodynamic_ablation(root),
        actionability_diagnostics(root),
    ]
    return {
        "figures": figures,
        "count": len(figures),
        "populated_dashboard_exports": dashboard_exports,
        "evidence_boundary": "development only; simulated operator; no holdout",
    }
