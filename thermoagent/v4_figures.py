"""Publication-facing figures for the prospectively stopped ThermoHITL v4 study."""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .dashboard.v4 import V4DashboardReplay, V4DashboardFrame, frame_svg_v4
from .v4_analysis import FEATURE_BLOCKS, crossfit_scores


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
APP_LABELS = {
    "commercial": "Commercial",
    "humanitarian": "Humanitarian",
    "utility_restoration": "Utility restoration",
}
APP_COLORS = {
    "commercial": COLORS["blue"],
    "humanitarian": COLORS["green"],
    "utility_restoration": COLORS["orange"],
}
REGIME_LABELS = {
    "isolated_physical": "Isolated physical",
    "telemetry_integrity": "Telemetry integrity",
    "partition": "Partition",
    "correlated": "Correlated",
    "compound": "Compound",
    "aggregate": "Aggregate",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 14.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _stamp(fig: Any, text: str = "DEVELOPMENT ONLY · SIMULATED OPERATOR") -> None:
    # Keep the evidence-status stamp inside the tight-layout export bounds so
    # PDF renderers do not clip descenders or the bottom edge at paper size.
    fig.text(0.995, 0.02, text, ha="right", va="bottom", fontsize=10, color=COLORS["gray"])


def _save(fig: Any, root: Path, name: str, stamp: bool = True) -> str:
    pdf_dir = root / "figures" / "pdf"
    preview_dir = root / "figures" / "previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    if stamp:
        _stamp(fig)
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.94))
    pdf = pdf_dir / (name + ".pdf")
    preview = preview_dir / (name + ".png")
    fig.savefig(pdf, format="pdf", metadata={"Title": name, "Subject": "ThermoHITL v4 development evidence"})
    fig.savefig(preview, format="png", dpi=240)
    plt.close(fig)
    return str(pdf.relative_to(root))


def _read(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative)


def _episode_path(root: Path, application: str, regime: str = "compound") -> Path:
    candidates = sorted((root / "raw" / "development_gate_trigger").glob(
        "*-%s-thermohitl_v4_rule-%s-private_fragmented-e*/episode.json"
        % (application, regime)
    ))
    if not candidates:
        raise FileNotFoundError("no trigger episode for %s/%s" % (application, regime))
    for path in candidates:
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode["metrics"].get("complete_causal_chains", 0) > 0:
            return path
    return candidates[0]


def _events(path: Path) -> List[Dict[str, Any]]:
    ledger = next(path.parent.glob("events.jsonl*"))
    opener = gzip.open if ledger.suffix == ".gz" else open
    with opener(ledger, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _first_populated_frame(root: Path, application: str) -> Tuple[Path, V4DashboardReplay, V4DashboardFrame]:
    path = _episode_path(root, application)
    replay = V4DashboardReplay(path)
    populated = [frame for frame in replay.frames if frame.view_hashes]
    if not populated:
        raise RuntimeError("selected replay has no authorized operator payload")
    return path, replay, populated[0]


def architecture(root: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [
        (0.2, 7.4, 3.2, 1.45, "Independent agents\nprivate state & memory\nindependent authority", "blue"),
        (4.4, 7.4, 3.2, 1.45, "Bounded gossip\nprivate sketches\nexplicit messages", "green"),
        (8.6, 7.4, 3.2, 1.45, "Distributed monitor\nenergy · entropy\ndisagreement · confidence", "orange"),
        (8.6, 4.6, 3.2, 1.45, "Request or abstain\nlocal trigger\nadjustable autonomy", "purple"),
        (4.4, 4.6, 3.2, 1.45, "Attention allocator\none slot & queue\nbudgeted workload", "red"),
        (0.2, 4.6, 3.2, 1.45, "Simulated operator\nauthorized view\nbounded choices", "red"),
        (0.2, 1.8, 3.2, 1.45, "Typed intervention\nverify or authorize\nresource / information", "orange"),
        (4.4, 1.8, 3.2, 1.45, "Agent response\ncommitments & tools\nmaterial / service flow", "blue"),
        (8.6, 1.8, 3.2, 1.45, "Outcome and replay\nloss & conservation\ncausal branch", "green"),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.08",
            facecolor=mpl.colors.to_rgba(COLORS[color], 0.11),
            edgecolor=COLORS[color], linewidth=1.6,
        ))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10.2)
    arrows = [
        ((3.4, 8.12), (4.4, 8.12)), ((7.6, 8.12), (8.6, 8.12)),
        ((10.2, 7.4), (10.2, 6.05)), ((8.6, 5.32), (7.6, 5.32)),
        ((4.4, 5.32), (3.4, 5.32)), ((1.8, 4.6), (1.8, 3.25)),
        ((3.4, 2.52), (4.4, 2.52)), ((7.6, 2.52), (8.6, 2.52)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": COLORS["black"]})
    ax.text(6.0, 0.82, "Event sourcing links authorized view hash → action → response → arrival → outcome",
            ha="center", fontsize=10.5, color=COLORS["gray"])
    ax.text(6.0, 0.34, "No hidden central domain planner; evaluator and counterfactual state are analysis-only",
            ha="center", fontsize=10.5, color=COLORS["gray"])
    ax.set_title("ThermoHITL v4: distributed observability for scarce human attention")
    return _save(fig, root, "thermohitl_v4_architecture", stamp=False)


def three_application_diagram(root: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.7), sharey=True)
    content = {
        "commercial": ("Suppliers\nCarriers · hubs\nRetailers", "Service-loss AUC", "Pre-specified\nKPI-sufficient boundary"),
        "humanitarian": ("NGOs · depots\nTransport\nCritical regions", "Weighted unmet need", "Fragmented needs\nand mandates"),
        "utility_restoration": ("Zones\nSubstations · crews\nCritical loads", "Critical unserved-load\nAUC", "Ambiguous telemetry\n+ restoration resources"),
    }
    for ax, app in zip(axes, content):
        roles, outcome, role = content[app]
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.82, transform=ax.transAxes,
                    boxstyle="round,pad=0.025", facecolor=mpl.colors.to_rgba(APP_COLORS[app], 0.10),
                    edgecolor=APP_COLORS[app], linewidth=2))
        ax.text(0.5, 0.80, APP_LABELS[app], transform=ax.transAxes, ha="center", weight="bold", fontsize=13)
        ax.text(0.5, 0.62, roles, transform=ax.transAxes, ha="center", va="center", wrap=True)
        ax.text(0.5, 0.43, "Primary outcome\n" + outcome, transform=ax.transAxes, ha="center", va="center", fontsize=10.5)
        ax.text(0.5, 0.23, role, transform=ax.transAxes, ha="center", va="center", fontsize=10.5, color=COLORS["gray"], wrap=True)
    fig.suptitle("Matched oversight mechanism across three autonomous-system applications")
    _stamp(fig, "STUDY DESIGN · COMMERCIAL IS A PRE-SPECIFIED BOUNDARY APPLICATION")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    return _save(fig, root, "three_application_comparison", stamp=False)


def _draw_network(ax: Any, frame: V4DashboardFrame, title: str) -> None:
    nodes = frame.network.get("nodes", [])
    positions = {node["agent_id"]: np.asarray(node.get("location", [0.0, 0.0]), dtype=float) for node in nodes}
    styles = (
        ("service_edges", COLORS["green"], 3.0, "-"),
        ("logistics_edges", COLORS["gray"], 2.2, "-"),
        ("communication_edges", COLORS["blue"], 1.2, "--"),
        ("authorized_emergency_edges", COLORS["red"], 4.4, "-"),
    )
    for key, color, width, line in styles:
        for left, right in frame.network.get(key, []):
            if left in positions and right in positions:
                p, q = positions[left], positions[right]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width, ls=line, alpha=0.85, zorder=1)
    for incident in frame.network.get("visible_incidents", []):
        point = np.asarray(incident.get("location", [0.0, 0.0]), dtype=float)
        low = incident.get("telemetry_confidence_state") == "low"
        ax.scatter(point[0], point[1], s=850, facecolors="none", edgecolors=COLORS["red"] if low else COLORS["orange"],
                   linewidths=2.4, linestyle="--" if low else "-", zorder=2)
    for node in nodes:
        point = positions[node["agent_id"]]
        ax.scatter(point[0], point[1], s=150, color=COLORS["sky"], edgecolor=COLORS["black"], lw=1.3, zorder=3)
        role = str(node["role"])
        abbreviations = {
            "distribution_zone": "Zone", "substation": "Sub", "microgrid": "MG",
            "crew_dispatch": "Crew", "parts_depot": "Parts", "mobile_generation": "Gen",
            "critical_load": "Load", "incident_coordinator": "Coord",
        }
        suffix = str(node["agent_id"]).rsplit("_", 1)[-1]
        label = abbreviations.get(role, role.replace("_", " ").title()) + " " + suffix
        ax.text(point[0], point[1] - 0.14, label, ha="center", va="top", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)


def utility_multilayer_network(root: Path) -> str:
    _, _, frame = _first_populated_frame(root, "utility_restoration")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _draw_network(ax, frame, "Utility-restoration multilayer network at operator allocation")
    handles = [
        Line2D([0], [0], color=COLORS["green"], lw=3, label="Service/power"),
        Line2D([0], [0], color=COLORS["gray"], lw=2.2, label="Restoration route"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.4, ls="--", label="Communication"),
        Line2D([0], [0], marker="o", color=COLORS["orange"], mfc="none", lw=0, markersize=14, label="Visible incident"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    return _save(fig, root, "utility_restoration_multilayer_network")


def utility_cyber_event_sequence(root: Path) -> str:
    path = _episode_path(root, "utility_restoration")
    events = _events(path)
    event_map = {
        "disruption": ("Abstract cyber-physical\ndisruption", COLORS["red"]),
        "human_request": ("Independent\nrequest", COLORS["purple"]),
        "attention_allocation": ("Attention\nallocated", COLORS["orange"]),
        "operator_action": ("Bounded operator\nverification", COLORS["blue"]),
        "plan_revision": ("Autonomous\nreplan", COLORS["sky"]),
        "service_transition": ("Critical service\nrestored", COLORS["green"]),
    }
    selected = []
    for kind in event_map:
        candidates = [event for event in events if event["kind"] == kind]
        if candidates:
            selected.append((candidates[0]["step"], kind, event_map[kind]))
    selected.sort()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(selected))[::-1]
    for ordinate, (step, kind, (label, color)) in zip(y, selected):
        ax.hlines(ordinate, 5.0, 12.0, color=COLORS["light"], lw=1.3)
        ax.scatter([step], [ordinate], s=115, color=color, edgecolor="white", lw=1.2, zorder=3)
        ax.text(step + 0.14, ordinate, "step %d" % step, va="center", fontsize=10.5, color=color)
    ax.set_xlim(4.8, 12.2); ax.set_ylim(-0.6, len(selected) - 0.4)
    ax.set_xlabel("Simulation step")
    ax.set_yticks(y)
    ax.set_yticklabels([value[2][0].replace("\n", " ") for value in selected])
    ax.set_title("Defensive abstract utility event chain (no real system or attack technique)")
    return _save(fig, root, "utility_cyber_disruption_event_sequence")


def phase_plane(root: Path) -> str:
    frame = _read(root, "development/development_gate_monitoring/candidate_interventions.csv")
    frame = frame[frame.information_condition == "private_fragmented"].copy()
    severity = (
        0.42 * frame.service_deficit + 0.18 * frame.backlog
        + 0.14 * frame.safety_stress + 0.12 * frame.resource_scarcity
        + 0.14 * frame.actionability_flag
    )
    frame["severity_energy_contribution"] = 1.05 * severity + 0.30 * np.maximum(0.0, frame.standardized_energy)
    frame["uncertainty_contribution"] = (
        0.20 * frame.entropy_anomaly
        + 0.10 * np.maximum(0.0, frame.entropy_slope / 0.05)
        + 2.80 * frame.belief_disagreement
        - 0.25 * (1.0 - frame.consensus_confidence)
    )
    frame["alert"] = frame.severity_energy_contribution + frame.uncertainty_contribution >= 1.15
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    xmax = max(1.4, float(frame.severity_energy_contribution.max()) * 1.05)
    ymax = max(1.4, float(frame.uncertainty_contribution.max()) * 1.05)
    x = np.linspace(0, xmax, 300)
    ax.fill_between(x, np.maximum(0.0, 1.15 - x), ymax, color=mpl.colors.to_rgba(COLORS["red"], 0.08), label="Alert region")
    ax.fill_between(x, 0, np.minimum(ymax, np.maximum(0.0, 1.15 - x)), color=mpl.colors.to_rgba(COLORS["green"], 0.08), label="Autonomy / abstention")
    ax.plot(x, 1.15 - x, color=COLORS["red"], lw=2.2, label="Frozen boundary (score = 1.15)")
    for app, group in frame.groupby("application"):
        ax.scatter(group.severity_energy_contribution, group.uncertainty_contribution,
                   s=34, alpha=0.62, color=APP_COLORS[app], label=APP_LABELS[app],
                   marker={"commercial": "o", "humanitarian": "s", "utility_restoration": "^"}[app])

    # Overlay one auditable trajectory from an actual utility-restoration replay.
    # Its coordinates use the same frozen score decomposition as request_score_v4;
    # the workload term is kept on the horizontal coordinate so x + y = score.
    episode_path = _episode_path(root, "utility_restoration", "compound")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    with gzip.open(episode_path.with_name("events.jsonl.gz"), "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    observations = {
        (event["step"], event["payload"]["recipient"]): event["payload"]["observation"]
        for event in events if event["kind"] == "observation_delivery"
    }
    thermodynamic = {
        (event["step"], event["payload"]["incident_id"]): event["payload"]["features"]
        for event in events if event["kind"] == "thermodynamic_state"
    }
    workload = {int(row["step"]): float(row["operator_workload"]) for row in episode["time_series"]}
    trajectory = []
    for step in (5, 7, 10):
        observation = observations[(step, "zone_01")]
        thermo = thermodynamic[(step, "hospital_zone")]
        local_severity = (
            0.42 * observation["local_service_deficit"]
            + 0.18 * observation["local_backlog"]
            + 0.14 * observation["local_safety_stress"]
            + 0.12 * observation["local_resource_scarcity"]
            + 0.14 * observation["local_actionability_flag"]
        )
        horizontal = (
            1.05 * local_severity
            + 0.30 * max(0.0, thermo["standardized_energy"])
            - 0.20 * workload[step]
        )
        vertical = (
            0.20 * thermo["entropy_anomaly"]
            + 0.10 * max(0.0, thermo["entropy_slope"] / 0.05)
            + 2.80 * thermo["belief_disagreement"]
            - 0.25 * (1.0 - thermo["consensus_confidence"])
        )
        trajectory.append((horizontal, vertical, step))
    tx = [value[0] for value in trajectory]
    ty = [value[1] for value in trajectory]
    ax.plot(tx, ty, color=COLORS["black"], lw=1.8, marker="D", ms=6, zorder=5)
    ax.annotate("Nominal t=5", (tx[0], ty[0]), xytext=(10, 12), textcoords="offset points", fontsize=10)
    ax.annotate(
        "Disruption t=6\nalert + intervention t=7",
        (tx[1], ty[1]), xytext=(-104, 20), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["black"], "lw": 1.0}, fontsize=10,
    )
    ax.annotate("Post-intervention t=10", (tx[2], ty[2]), xytext=(-118, -30), textcoords="offset points", fontsize=10)
    ax.axvspan(0, 0.35, color=COLORS["gray"], alpha=0.05)
    ax.text(0.72, 0.15, "Nominal calibration region",
            fontsize=10, color=COLORS["gray"], ha="center", va="center")
    ax.set(xlim=(0, xmax), ylim=(0, ymax),
           xlabel="Severity + standardized-energy − workload contribution",
           ylabel="Entropy, slope, disagreement, and confidence contribution")
    ax.set_title("Actual prospective trigger coordinates and boundary")
    handles, labels = ax.get_legend_handles_labels()
    region_legend = ax.legend(handles[:3], labels[:3], loc="upper left", frameon=False)
    ax.add_artist(region_legend)
    ax.legend(handles[3:], labels[3:], loc="center", bbox_to_anchor=(0.58, 0.72), frameon=False)
    return _save(fig, root, "standardized_energy_entropy_disagreement_phase_plane")


def export_dashboard_replays(root: Path) -> List[str]:
    destination = root / "dashboard_exports"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    outputs = []
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("vector dashboard PDF export requires rsvg-convert")
    for application in APP_LABELS:
        episode_path, replay, frame = _first_populated_frame(root, application)
        svg_path = destination / (application + "_authorized_replay.svg")
        pdf_path = destination / (application + "_authorized_replay.pdf")
        svg_path.write_text(frame_svg_v4(frame) + "\n", encoding="utf-8")
        # Keep the direct librsvg output: it retains embedded, selectable
        # vector text. The replay SVG deliberately uses Liberation Sans after
        # cross-render QA found a DejaVu Sans subset defect in Poppler.
        subprocess.run(
            [converter, "-w", "540", "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            check=True,
        )
        outputs.extend([str(svg_path.relative_to(root)), str(pdf_path.relative_to(root))])
        records.append({
            "application": application,
            "run_id": replay.metadata()["run_id"],
            "step": frame.step,
            "view_sha256": frame.view_hashes[-1],
            "replay_digest": replay.digest(),
            "svg": str(svg_path.relative_to(root)),
            "pdf": str(pdf_path.relative_to(root)),
            "operator_payload_only": True,
            "evidence_stage": "development",
            "simulated_operator": True,
        })
    (destination / "metadata.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(str((destination / "metadata.json").relative_to(root)))
    # The utility export is the publication dashboard figure. It remains a
    # vector conversion of the functional replay export, not a redrawn mockup.
    publication = root / "figures" / "pdf" / "operator_dashboard_populated.pdf"
    publication.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination / "utility_restoration_authorized_replay.pdf", publication)
    subprocess.run([
        "pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "240",
        str(publication), str(root / "figures" / "previews" / "operator_dashboard_populated"),
    ], check=True, capture_output=True)
    outputs.append(str(publication.relative_to(root)))
    return outputs


def operator_view_comparison(root: Path) -> str:
    candidate = _read(root, "development/development_gate_monitoring/candidate_interventions.csv")
    example = candidate[(candidate.application == "utility_restoration") & (candidate.information_condition == "private_fragmented")].iloc[0]
    kpi_fields = ["service deficit", "backlog", "lateness", "safety stress", "resource scarcity"]
    thermo_fields = ["standardized energy", "entropy anomaly", "entropy slope", "belief disagreement", "consensus confidence"]
    display_labels = {
        "service deficit": "Service deficit",
        "backlog": "Backlog",
        "lateness": "Lateness",
        "safety stress": "Safety stress",
        "resource scarcity": "Resource scarcity",
        "standardized energy": "Std. energy",
        "entropy anomaly": "Entropy anomaly",
        "entropy slope": "Entropy slope",
        "belief disagreement": "Disagreement",
        "consensus confidence": "Consensus confidence",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4))
    for ax, title, fields, color in (
        (axes[0], "A. KPI-only view", kpi_fields, COLORS["blue"]),
        (axes[1], "B. KPI + thermo", kpi_fields + thermo_fields, COLORS["green"]),
    ):
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0.03, 0.05), 0.94, 0.88, transform=ax.transAxes,
                    boxstyle="round,pad=0.02", facecolor=mpl.colors.to_rgba(color, 0.08), edgecolor=color, lw=1.8))
        ax.text(0.5, 0.86, title, transform=ax.transAxes, ha="center", weight="bold", fontsize=12)
        for index, field in enumerate(fields):
            key = field.replace(" ", "_")
            value = example.get(key, np.nan)
            rendered = "%.3f" % value if pd.notna(value) else "—"
            y = 0.75 - 0.065 * index
            ax.text(0.10, y, display_labels[field], transform=ax.transAxes, ha="left", fontsize=9.4)
            ax.text(0.90, y, rendered, transform=ax.transAxes, ha="right", fontsize=9.4, family="monospace")
        ax.text(0.5, 0.125, "Matched incident and timing\nOne attention slot", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["gray"], fontsize=8.8)
    fig.suptitle("Matched information boundary: what thermodynamic observability adds")
    return _save(fig, root, "operator_view_kpi_vs_thermodynamic")


def _case_timeline(root: Path, application: str, name: str) -> str:
    path = _episode_path(root, application)
    episode = json.loads(path.read_text(encoding="utf-8"))
    data = pd.DataFrame(episode["time_series"])
    events = _events(path)
    disruption = min(event["step"] for event in events if event["kind"] == "disruption")
    request_steps = [event["step"] for event in events if event["kind"] == "human_request"]
    action_steps = [event["step"] for event in events if event["kind"] == "operator_action"]
    service_steps = [event["step"] for event in events if event["kind"] == "service_transition"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.5), sharex=True)
    axes[0].plot(data.step, data.operational_energy, color=COLORS["red"], lw=2, label="Operational energy")
    axes[0].plot(data.step, data.entropy_anomaly / max(data.entropy_anomaly.max(), 1e-9), color=COLORS["orange"], lw=2, ls="--", label="Entropy anomaly (scaled)")
    axes[0].plot(data.step, data.belief_disagreement, color=COLORS["purple"], lw=2, ls=":", label="Belief disagreement")
    axes[0].legend(frameon=False, ncol=3); axes[0].set_ylabel("State")
    axes[1].plot(data.step, data.consensus_confidence, color=COLORS["blue"], lw=2, label="Consensus confidence")
    axes[1].plot(data.step, data.consensus_error, color=COLORS["gray"], lw=2, ls="--", label="Consensus error")
    axes[1].legend(frameon=False, ncol=2); axes[1].set_ylabel("Consensus")
    axes[2].step(data.step, data.requests, where="mid", color=COLORS["purple"], lw=2, label="Requests")
    axes[2].step(data.step, data.operator_interventions, where="mid", color=COLORS["green"], lw=2, label="Interventions")
    axes[2].plot(data.step, data.operator_workload, color=COLORS["orange"], lw=2, label="Workload")
    axes[2].legend(frameon=False, ncol=3); axes[2].set_ylabel("Oversight")
    axes[3].plot(data.step, data.loss, color=COLORS["black"], lw=2.4, label="Primary loss per step")
    axes[3].step(data.step, data.material_actions_reached_service, where="mid", color=COLORS["green"], lw=2, label="Cumulative service arrivals")
    axes[3].legend(frameon=False, ncol=2); axes[3].set_ylabel("Outcome"); axes[3].set_xlabel("Simulation step")
    for ax in axes:
        ax.axvline(disruption, color=COLORS["red"], lw=1.4, ls="--")
        for value in action_steps:
            ax.axvline(value, color=COLORS["green"], lw=1.1, ls=":")
    axes[0].set_title("%s case: disruption at %d, request at %s, intervention at %s, service arrival at %s" % (
        APP_LABELS[application], disruption,
        request_steps[0] if request_steps else "none",
        action_steps[0] if action_steps else "none",
        service_steps[0] if service_steps else "none",
    ))
    return _save(fig, root, name)


def trigger_timeline(root: Path) -> str:
    return _case_timeline(root, "utility_restoration", "trigger_and_intervention_timeline")


def alert_timing(root: Path) -> str:
    frame = _read(root, "development/development_gate_trigger/episode_summary.csv")
    private = frame[frame.information_condition == "private_fragmented"].copy()
    private["activation_delay"] = private.first_request_step - 6
    private["false_rate"] = private.pre_disruption_false_activation.astype(float)
    summary = private.groupby(["application", "regime"], as_index=False).agg(
        timely=("timely_activation", "mean"),
        false=("false_rate", "mean"),
        delay=("activation_delay", "mean"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    regimes = [value for value in REGIME_LABELS if value != "aggregate"]
    x = np.arange(len(regimes)); width = 0.24
    for index, app in enumerate(APP_LABELS):
        values = summary[summary.application == app].set_index("regime")
        axes[0].bar(x + (index - 1) * width, [values.loc[r, "timely"] for r in regimes], width,
                    color=APP_COLORS[app], label=APP_LABELS[app])
        axes[1].bar(x + (index - 1) * width, [values.loc[r, "delay"] for r in regimes], width,
                    color=APP_COLORS[app], label=APP_LABELS[app])
    axes[0].axhline(0.75, color=COLORS["red"], ls="--")
    axes[0].text(len(regimes) - 0.55, 0.765, "Frozen gate: 75%", ha="right", va="bottom", color=COLORS["red"], fontsize=10)
    axes[0].set_ylabel("Timely activation"); axes[0].set_ylim(0, 1.08)
    axes[1].set_ylabel("Delay (steps)")
    axes[0].set_xticks(x); axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xticks(x); axes[1].set_xticklabels([REGIME_LABELS[r] for r in regimes], rotation=25, ha="right")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    axes[0].set_title("A. Activation after disruption")
    axes[1].set_title("B. Delay from disruption onset")
    fig.suptitle("Trigger activation and delay (no false alerts observed)")
    return _save(fig, root, "alert_timing_and_false_alarms")


def workload_performance(root: Path) -> str:
    frame = _read(root, "development/development_gate_human/episode_summary.csv")
    frame = frame[frame.method.isin(["autonomy_no_operator", "thermohitl_v4_rule"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    aggregate = frame.groupby(["application", "method"], as_index=False).agg(
        minutes=("operator_minutes", "mean"), loss=("primary_outcome", "mean")
    )
    for app, group in aggregate.groupby("application"):
        group = group.sort_values("minutes")
        axes[0].plot(group.minutes, group.loss, marker="o", ms=8, lw=2,
                     color=APP_COLORS[app], label=APP_LABELS[app])
    paired = _read(root, "statistics/human_causal_paired_effects.csv")
    paired = paired[paired.regime != "aggregate"]
    offsets = {"commercial": -0.012, "humanitarian": 0.0, "utility_restoration": 0.012}
    for app, group in paired.groupby("application"):
        axes[1].scatter(np.full(len(group), 0.34 + offsets[app]), group.mean_relative_loss_reduction * 100,
                        s=65, color=APP_COLORS[app], label=APP_LABELS[app], alpha=0.75)
    axes[0].set(xlabel="Mean simulated operator minutes", ylabel="Mean primary episode loss", title="A. Fixed-effort outcome tradeoff")
    axes[1].axhline(0, color=COLORS["black"], lw=1)
    axes[1].set(xlabel="Peak simulated workload (jittered by application)", ylabel="Loss reduction by regime (%)", title="B. Outcome variation at bounded workload")
    axes[0].legend(frameon=False)
    return _save(fig, root, "operator_workload_vs_service_performance")


def budgeted_utility(root: Path) -> str:
    frame = _read(root, "statistics/budgeted_causal_utility.csv")
    frame = frame[frame.information_condition == "private_fragmented"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    x = np.arange(len(frame)); width = 0.33
    axes[0].bar(x - width / 2, frame.kpi_mean_causal_utility, width, color=COLORS["blue"], label="KPI-only")
    axes[0].bar(x + width / 2, frame.thermodynamic_mean_causal_utility, width, color=COLORS["green"], label="KPI + thermodynamics")
    axes[0].axhline(0, color=COLORS["black"], lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels([APP_LABELS[v] for v in frame.application])
    axes[0].set_ylabel("Budgeted causal utility (loss units)"); axes[0].legend(frameon=False)
    axes[0].set_title("A. Absolute utility at one intervention per panel")
    gain = frame.paired_mean_utility_gain.to_numpy()
    low = frame.utility_gain_ci95_low.to_numpy(); high = frame.utility_gain_ci95_high.to_numpy()
    axes[1].errorbar(x, gain, yerr=np.vstack([gain - low, high - gain]), fmt="o", ms=8,
                     color=COLORS["purple"], capsize=5, lw=2)
    axes[1].axhline(0, color=COLORS["black"], lw=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels([APP_LABELS[v] for v in frame.application])
    axes[1].set_ylabel("Paired utility gain (95% cluster bootstrap CI)")
    axes[1].set_title("B. Paired incremental value (60 independent panels per application)")
    fig.suptitle("Same-information, fixed-budget causal decision utility")
    return _save(fig, root, "budgeted_causal_utility_cluster_intervals")


def effect_forest(root: Path) -> str:
    coordination = _read(root, "statistics/coordination_paired_effects.csv")
    human = _read(root, "statistics/human_causal_paired_effects.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.0), sharex=False)
    for ax, frame, title in (
        (axes[0], coordination, "Fixed communication − no communication"),
        (axes[1], human, "ThermoHITL rule − autonomy only"),
    ):
        rows = frame[frame.regime != "aggregate"].copy()
        rows["label"] = rows.application.map(APP_LABELS) + " · " + rows.regime.map(REGIME_LABELS)
        rows = rows.sort_values(["application", "regime"])
        y = np.arange(len(rows))
        mean = rows.mean_treatment_minus_reference.to_numpy()
        low = rows.difference_ci95_low.to_numpy(); high = rows.difference_ci95_high.to_numpy()
        colors = [APP_COLORS[value] for value in rows.application]
        for index in range(len(rows)):
            ax.errorbar(mean[index], y[index], xerr=[[mean[index]-low[index]], [high[index]-mean[index]]],
                        fmt="o", color=colors[index], capsize=3, lw=1.7)
        ax.axvline(0, color=COLORS["black"], lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(rows.label)
        ax.invert_yaxis(); ax.set_xlabel("Paired loss difference (negative favors treatment)"); ax.set_title(title)
    fig.suptitle("Development paired effects by application and disruption regime (12 panels each)")
    return _save(fig, root, "per_application_regime_effect_forest")


def feature_ablation(root: Path) -> str:
    frame = _read(root, "statistics/feature_block_performance.csv")
    frame = frame[frame.information_condition == "private_fragmented"]
    order = list(FEATURE_BLOCKS)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    x = np.arange(len(order)); width = 0.24
    for index, app in enumerate(APP_LABELS):
        group = frame[frame.application == app].set_index("feature_block")
        values = [group.loc[value, "mean_budgeted_causal_utility"] for value in order]
        ax.bar(x + (index - 1) * width, values, width, color=APP_COLORS[app], label=APP_LABELS[app])
    ax.axhline(0, color=COLORS["black"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels([value.replace("_", " ") for value in order], rotation=28, ha="right")
    ax.set_ylabel("Mean budgeted causal utility")
    ax.set_title("Pre-specified feature-block ablation under private fragmented information")
    ax.legend(frameon=False, ncol=3)
    return _save(fig, root, "thermodynamic_feature_block_ablation")


def fragmentation_benefit(root: Path) -> str:
    selections = _read(root, "statistics/candidate_selection_by_cluster.csv")
    private = selections[selections.information_condition == "private_fragmented"].copy()
    candidates = _read(root, "development/development_gate_monitoring/candidate_interventions.csv")
    frag = candidates[candidates.information_condition == "private_fragmented"].groupby("cluster_id", as_index=False).belief_disagreement.max()
    private = private.merge(frag, on="cluster_id", how="left")
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for app, group in private.groupby("application"):
        ax.scatter(group.belief_disagreement, group.paired_utility_gain, s=48, alpha=0.64,
                   color=APP_COLORS[app], label=APP_LABELS[app])
        if group.belief_disagreement.nunique() > 1:
            coefficient = np.polyfit(group.belief_disagreement, group.paired_utility_gain, 1)
            x = np.linspace(group.belief_disagreement.min(), group.belief_disagreement.max(), 60)
            ax.plot(x, np.polyval(coefficient, x), color=APP_COLORS[app], lw=2)
    ax.axhline(0, color=COLORS["black"], lw=1)
    ax.set(xlabel="Maximum local belief disagreement in matched panel",
           ylabel="Thermodynamic minus KPI-only causal utility",
           title="Conditional incremental value under fragmented observability")
    ax.legend(frameon=False)
    return _save(fig, root, "conditional_benefit_vs_information_fragmentation")


def calibration_reliability(root: Path) -> str:
    candidates = _read(root, "development/development_gate_monitoring/candidate_interventions.csv")
    candidates = candidates[(candidates.information_condition == "private_fragmented") & candidates.application.isin(["humanitarian", "utility_restoration"])].copy()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True, sharey=True)
    for ax, app in zip(axes, ("humanitarian", "utility_restoration")):
        group = candidates[candidates.application == app].reset_index(drop=True)
        for block, color, marker in (("local_kpi_only", COLORS["blue"], "o"), ("complete_thermodynamic", COLORS["green"], "s")):
            scores, _ = crossfit_scores(group, FEATURE_BLOCKS[block], c_grid=(0.05, 0.2, 1.0), budget=1)
            bins = pd.qcut(scores, q=min(5, len(np.unique(scores))), duplicates="drop")
            calibration = pd.DataFrame({"score": scores, "label": group.beneficial.astype(float), "bin": bins}).groupby("bin", observed=True).agg(predicted=("score", "mean"), observed=("label", "mean"), n=("label", "size"))
            ax.plot(calibration.predicted, calibration.observed, color=color, marker=marker, lw=2, label=block.replace("_", " "))
        ax.plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=1)
        ax.set_title(APP_LABELS[app])
    axes[1].set_xlabel("Cross-fitted predicted benefit probability")
    axes[0].set_ylabel("Observed beneficial fraction")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Cluster-separated development reliability (secondary diagnostic)")
    return _save(fig, root, "calibration_and_reliability")


def causal_funnels(root: Path) -> str:
    counter = _read(root, "counterfactuals/development_gate_human.csv")
    human = _read(root, "development/development_gate_human/episode_summary.csv")
    stage_matrix = np.column_stack([
        counter.request_entered_queue.astype(bool),
        counter.allocator_selected.astype(bool),
        counter.operator_received_authorized_view.astype(bool),
        counter.operator_acted.astype(bool),
        (counter.agent_commitment_changed.astype(bool) | counter.accepted_action_changed.astype(bool)),
        counter.material_or_service_flow_changed.astype(bool),
        counter.reached_demand_or_critical_service.astype(bool),
        counter.primary_outcome_changed.astype(bool),
    ])
    values = np.logical_and.accumulate(stage_matrix, axis=1).mean(axis=0)
    labels = [
        "Queued", "Allocated", "Authorized view", "Operator acted",
        "Commitment/action changed", "Flow changed", "Reached service", "Outcome changed",
    ]
    action_values = [
        float(human.structured_attempts.sum()),
        float(human.material_actions_accepted.sum()),
        float(human.material_actions_next_stage.sum()),
        float(human.material_actions_reached_service.sum()),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2))
    axes[0].barh(np.arange(len(values)), values, color=COLORS["purple"])
    axes[0].set_yticks(np.arange(len(values))); axes[0].set_yticklabels(labels); axes[0].invert_yaxis(); axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Fraction of 180 counterfactual probes")
    axes[0].set_title("A. Counterfactual intervention chain")
    axes[1].barh(np.arange(4), action_values, color=COLORS["blue"])
    axes[1].set_yticks(np.arange(4)); axes[1].set_yticklabels(["Structured attempts", "Accepted actions", "Next physical stage", "Reached service"]); axes[1].invert_yaxis()
    axes[1].set_xlabel("Count across 360 online episodes")
    axes[1].set_title("B. Autonomous material-action stages")
    fig.suptitle("Separate populations: counterfactual probes are not pooled with autonomous actions")
    return _save(fig, root, "causal_intervention_and_action_funnels")


def intervention_distribution(root: Path) -> str:
    frame = _read(root, "counterfactuals/development_gate_human.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    bins = np.linspace(frame.intervention_effect.min() - 0.02, frame.intervention_effect.max() + 0.02, 18)
    for app, group in frame.groupby(frame.run_id.str.extract(r"development_gate_human-([^-]+)")[0]):
        axes[0].hist(group.intervention_effect, bins=bins, alpha=0.45, color=APP_COLORS[app], label=APP_LABELS[app])
    axes[0].axvline(0, color=COLORS["black"], lw=1); axes[0].set_xlabel("Paired loss reduction"); axes[0].set_ylabel("Counterfactual probes")
    axes[0].set_title("A. Intervention-effect distribution"); axes[0].legend(frameon=False)
    categories = []
    for app in APP_LABELS:
        group = frame[frame.run_id.str.contains("-" + app + "-")]
        categories.append([float((group.intervention_effect < -1e-12).mean()), float((group.intervention_effect.abs() <= 1e-12).mean()), float((group.intervention_effect > 1e-12).mean())])
    bottom = np.zeros(3)
    x = np.arange(3)
    for index, label in enumerate(("Harmful", "Neutral", "Beneficial")):
        values = np.asarray([row[index] for row in categories])
        axes[1].bar(x, values, bottom=bottom, color=(COLORS["red"], COLORS["gray"], COLORS["green"])[index], label=label)
        bottom += values
    axes[1].set_xticks(x); axes[1].set_xticklabels(list(APP_LABELS.values()), rotation=15)
    axes[1].set_ylim(0, 1.20); axes[1].set_ylabel("Fraction of probes"); axes[1].set_title("B. Harmful, neutral, beneficial")
    axes[1].legend(frameon=False, ncol=3, loc="upper center")
    return _save(fig, root, "harmful_neutral_beneficial_interventions")


def partition_robustness(root: Path) -> str:
    rows = []
    for path in sorted((root / "raw" / "development_gate_trigger").glob("*/episode.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        values = pd.DataFrame(episode["time_series"])
        rows.append({
            "application": episode["application"],
            "regime": episode["regime"],
            "information_condition": episode["information_condition"],
            "consensus_error": float(values.consensus_error.mean()),
            "consensus_confidence": float(values.consensus_confidence.mean()),
            "loss": float(episode["metrics"]["primary_outcome"]),
            "activated": float(episode["metrics"]["first_request_step"] is not None),
        })
    frame = pd.DataFrame(rows)
    private = frame[frame.information_condition == "private_fragmented"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    for app, group in private.groupby("application"):
        connected = group[group.regime != "partition"]
        partitioned = group[group.regime == "partition"]
        axes[0].scatter(connected.consensus_error, connected.consensus_confidence, s=38, alpha=0.42, color=APP_COLORS[app])
        axes[0].scatter(partitioned.consensus_error, partitioned.consensus_confidence, s=72, alpha=0.85, color=APP_COLORS[app], marker="X", label=APP_LABELS[app])
        axes[1].scatter(connected.consensus_error, connected.loss, s=38, alpha=0.42, color=APP_COLORS[app])
        axes[1].scatter(partitioned.consensus_error, partitioned.loss, s=72, alpha=0.85, color=APP_COLORS[app], marker="X", label=APP_LABELS[app])
    axes[0].set(ylabel="Mean consensus confidence", title="A. Confidence degrades with estimation error")
    axes[1].set(xlabel="Mean distributed-estimation error", ylabel="Primary episode loss", title="B. Outcome association under partitions")
    axes[0].legend(frameon=False, title="Partition panels (X)")
    return _save(fig, root, "distributed_consensus_error_under_partitions")


def commercial_boundary(root: Path) -> str:
    utility = _read(root, "statistics/budgeted_causal_utility.csv")
    utility = utility[utility.information_condition == "private_fragmented"].set_index("application")
    coordination = _read(root, "statistics/coordination_paired_effects.csv")
    coordination = coordination[coordination.regime == "aggregate"].set_index("application")
    human = _read(root, "statistics/human_causal_paired_effects.csv")
    human = human[human.regime == "aggregate"].set_index("application")
    x = np.arange(3)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    axes[0].bar(x, [coordination.loc[a, "mean_relative_loss_reduction"] * 100 for a in APP_LABELS], color=[APP_COLORS[a] for a in APP_LABELS])
    axes[0].axhline(5, color=COLORS["red"], ls="--", label="Frozen Gate 3: 5%")
    axes[0].set_ylabel("Fixed-communication\nloss reduction (%)"); axes[0].legend(frameon=False, loc="upper right")
    axes[1].bar(x, [utility.loc[a, "relative_utility_gain"] * 100 for a in APP_LABELS], color=[APP_COLORS[a] for a in APP_LABELS])
    axes[1].axhline(5, color=COLORS["red"], ls="--", label="Practical incremental-value target")
    axes[1].set_ylabel("Thermodynamic causal-utility\ngain (%)"); axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(list(APP_LABELS.values()), rotation=16)
    axes[0].set_title("A. Coordination necessity")
    axes[1].set_title("B. Same-information thermodynamic value")
    fig.suptitle("Commercial boundary and utility-restoration near-threshold stop")
    return _save(fig, root, "commercial_boundary_condition")


def generate_all(root: Path) -> List[str]:
    configure_style()
    outputs = [
        architecture(root),
        three_application_diagram(root),
        utility_multilayer_network(root),
        utility_cyber_event_sequence(root),
        phase_plane(root),
    ]
    outputs.extend(export_dashboard_replays(root))
    outputs.extend([
        operator_view_comparison(root),
        trigger_timeline(root),
        alert_timing(root),
        workload_performance(root),
        budgeted_utility(root),
        effect_forest(root),
        feature_ablation(root),
        fragmentation_benefit(root),
        calibration_reliability(root),
        causal_funnels(root),
        intervention_distribution(root),
        partition_robustness(root),
        commercial_boundary(root),
        _case_timeline(root, "humanitarian", "humanitarian_development_case_study"),
        _case_timeline(root, "utility_restoration", "utility_restoration_development_case_study"),
    ])
    manifest = {
        "evidence_stage": "development",
        "simulated_operator": True,
        "validation_run": False,
        "rl_training_run": False,
        "holdout_run": False,
        "publication_pdfs": sorted(path.name for path in (root / "figures" / "pdf").glob("*.pdf")),
        "dashboard_exports": sorted(str(path.relative_to(root)) for path in (root / "dashboard_exports").glob("*")),
    }
    (root / "figures" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
