"""Data-derived publication figures for the stopped V5 development study."""

from __future__ import annotations

import gzip
import json
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

from .dashboard.v5 import V5DashboardReplay, V5DashboardFrame, frame_svg_v5
from .v5_analysis import paired_bootstrap


COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#20242A", "gray": "#6C7480",
    "light": "#E9EDF2",
}
APP_LABELS = {
    "commercial": "Commercial boundary",
    "humanitarian": "Humanitarian",
    "utility_restoration": "Utility restoration",
}
APP_COLORS = {
    "commercial": COLORS["blue"], "humanitarian": COLORS["green"],
    "utility_restoration": COLORS["orange"],
}
APP_MARKERS = {"commercial": "o", "humanitarian": "s", "utility_restoration": "^"}
REGIME_LABELS = {
    "nominal": "Nominal", "isolated_physical": "Isolated", "telemetry_integrity": "Telemetry",
    "partition": "Partition", "correlated": "Correlated", "compound": "Compound", "ood": "OOD",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.5, "axes.titlesize": 12.0,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "figure.titlesize": 13.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def _stamp(fig: Any, text: str = "DEVELOPMENT ONLY · SIMULATED OPERATOR") -> None:
    fig.text(0.995, 0.012, text, ha="right", va="bottom", fontsize=9.0, color=COLORS["gray"])


def _save(fig: Any, root: Path, name: str, stamp: bool = True) -> str:
    pdf_dir = root / "figures" / "pdf"
    png_dir = root / "figures" / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    if stamp:
        _stamp(fig)
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.95))
    pdf = pdf_dir / f"{name}.pdf"
    png = png_dir / f"{name}.png"
    fig.savefig(pdf, format="pdf", metadata={
        "Title": name, "Subject": "ThermoHITL V5 development evidence",
    })
    fig.savefig(png, format="png", dpi=240)
    plt.close(fig)
    return str(pdf.relative_to(root))


def _write_data(root: Path, name: str, frame: pd.DataFrame) -> None:
    path = root / "figures" / "data" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5
        frame.to_csv(path, index=False, line_terminator="\n")


def _read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    return pd.read_csv(path)


def _representative_episode(root: Path, application: str, regime: str = "compound") -> Path:
    paths = sorted((root / "raw" / "development_primary_v2").glob(
        f"v5-{application}-{regime}-private_fragmented-event_triggered-e*/episode.json"
    ))
    if not paths:
        raise FileNotFoundError(f"no V5 episode for {application}/{regime}")
    return paths[0]


def _events(episode_path: Path) -> List[Dict[str, Any]]:
    ledger = next(episode_path.parent.glob("events.jsonl*"))
    opener = gzip.open if ledger.suffix == ".gz" else open
    with opener(ledger, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _populated_frame(root: Path, application: str) -> Tuple[Path, V5DashboardReplay, V5DashboardFrame]:
    path = _representative_episode(root, application)
    replay = V5DashboardReplay(path)
    populated = [value for value in replay.frames if value.alert_queue]
    if not populated:
        raise RuntimeError(f"no populated V5 dashboard frame for {application}")
    return path, replay, populated[-1]


def architecture(root: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set(xlim=(0, 12), ylim=(0, 10)); ax.axis("off")
    boxes = [
        (0.2, 7.5, "Independent agents\nprivate observation · memory\nutility · typed authority", "blue"),
        (4.4, 7.5, "Bounded messages\noperational + compressed\nthermodynamic sketches", "green"),
        (8.6, 7.5, "Distributed observability\nbelief entropy · JS disagreement\nconfidence · energy", "orange"),
        (8.6, 4.7, "Local request / abstention\nqueue entry + reason code\nno evaluator state", "purple"),
        (4.4, 4.7, "Budgeted allocator\ntwo incidents per panel\nfinite operator minutes", "red"),
        (0.2, 4.7, "Simulated operator\nauthorized dashboard\nbounded intervention", "red"),
        (0.2, 1.9, "Agent response\naccept · reject · counter\ncommitment revision", "blue"),
        (4.4, 1.9, "Validated flow\nmaterial / crew / service\nconservation", "green"),
        (8.6, 1.9, "Matched causal branch\nstored stochastic tape\nloss + effort", "orange"),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), 3.2, 1.35, boxstyle="round,pad=0.07",
            facecolor=mpl.colors.to_rgba(COLORS[color], 0.11),
            edgecolor=COLORS[color], linewidth=1.6,
        ))
        ax.text(x + 1.6, y + 0.675, label, ha="center", va="center", fontsize=8.7)
    for start, end in [
        ((3.4, 8.18), (4.4, 8.18)), ((7.6, 8.18), (8.6, 8.18)),
        ((10.2, 7.5), (10.2, 6.05)), ((8.6, 5.38), (7.6, 5.38)),
        ((4.4, 5.38), (3.4, 5.38)), ((1.8, 4.7), (1.8, 3.25)),
        ((3.4, 2.58), (4.4, 2.58)), ((7.6, 2.58), (8.6, 2.58)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": COLORS["black"]})
    ax.text(6, 0.78, "Evaluator state is analysis-only; no centralized policy replaces agent decisions.", ha="center", color=COLORS["gray"])
    ax.text(6, 0.35, "V5 stopped prospectively before validation after development gates failed.", ha="center", color=COLORS["red"], weight="bold")
    ax.set_title("ThermoHITL V5 autonomous-agent and human-oversight architecture")
    return _save(fig, root, "v5_architecture", stamp=False)


def three_applications(root: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.9))
    content = {
        "commercial": ("Suppliers · carriers\nwarehouses · retailers", "Service-loss AUC", "Boundary application\nKPI actionability may suffice"),
        "humanitarian": ("NGOs · hubs · carriers\nclinics · regions", "Weighted unmet need", "Primary domain\nfragmented need evidence"),
        "utility_restoration": ("Zones · relays · crews\ncyber defense · critical loads", "Critical unserved-load AUC", "Primary domain\nambiguous telemetry"),
    }
    for ax, app in zip(axes, APP_LABELS):
        roles, outcome, purpose = content[app]
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch(
            (0.04, 0.08), 0.92, 0.84, transform=ax.transAxes,
            boxstyle="round,pad=0.025", facecolor=mpl.colors.to_rgba(APP_COLORS[app], 0.10),
            edgecolor=APP_COLORS[app], linewidth=2,
        ))
        ax.text(0.5, 0.80, APP_LABELS[app], transform=ax.transAxes, ha="center", weight="bold", fontsize=12)
        ax.text(0.5, 0.61, roles, transform=ax.transAxes, ha="center", va="center")
        ax.text(0.5, 0.41, f"Primary outcome\n{outcome}", transform=ax.transAxes, ha="center", va="center")
        ax.text(0.5, 0.21, purpose, transform=ax.transAxes, ha="center", va="center", color=COLORS["gray"])
    fig.suptitle("One prospective oversight question across three independent-agent systems")
    return _save(fig, root, "three_application_overview")


def _draw_network(ax: Any, frame: V5DashboardFrame, entropy_overlay: bool = False) -> None:
    nodes = frame.network["nodes"]
    positions = {node["agent_id"]: np.asarray(node["location"], dtype=float) for node in nodes}
    for key, color, width, style in (
        ("service_edges", COLORS["green"], 3.0, "-"),
        ("logistics_edges", COLORS["gray"], 2.0, "-"),
        ("communication_edges", COLORS["blue"], 1.2, "--"),
    ):
        for left, right in frame.network.get(key, []):
            if left in positions and right in positions:
                p, q = positions[left], positions[right]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width, ls=style, alpha=0.72, zorder=1)
    abbreviations = {
        "distribution_node": "Zone", "field_crew": "Crew", "communications": "Comms",
        "cyber_defense": "Cyber", "resource_allocation": "Resources", "critical_load": "Load",
        "regional_coordinator": "Coord", "supplier": "Supplier", "carrier": "Carrier",
        "warehouse": "Warehouse", "retailer": "Retailer", "coordinator": "Coord",
        "ngo": "NGO", "regional_hub": "Hub", "clinic": "Clinic",
    }
    for index, node in enumerate(nodes):
        point = positions[node["agent_id"]]
        disagreement = float(node.get("disagreement") or 0.0)
        confidence = float(node.get("consensus_confidence") or 1.0)
        entropy = float(node.get("entropy") or 0.0)
        if entropy_overlay:
            ax.scatter(point[0], point[1], s=270 + 380 * entropy, facecolors="none",
                       edgecolors=COLORS["purple"], lw=1.4 + 2.0 * disagreement, alpha=0.75, zorder=2)
        color = COLORS["red"] if confidence < 0.42 else COLORS["sky"]
        ax.scatter(point[0], point[1], s=105, color=color, edgecolor=COLORS["black"], lw=1.1, zorder=3)
        label = abbreviations.get(node["role"], node["role"].replace("_", " ").title())
        vertical = 0.11 if index % 2 else -0.11
        ax.text(point[0], point[1] + vertical, label, ha="center",
                va="bottom" if vertical > 0 else "top", fontsize=7.9)
    ax.set_aspect("equal"); ax.axis("off")


def utility_network(root: Path) -> str:
    _, _, frame = _populated_frame(root, "utility_restoration")
    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    _draw_network(ax, frame)
    handles = [
        Line2D([0], [0], color=COLORS["green"], lw=3, label="Service/power edge"),
        Line2D([0], [0], color=COLORS["gray"], lw=2, label="Restoration route"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.3, ls="--", label="Communication link"),
        Line2D([0], [0], marker="o", mfc=COLORS["red"], mec=COLORS["black"], lw=0, label="Low-consensus agent"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.set_title("Abstract cyber-physical utility restoration network")
    return _save(fig, root, "utility_restoration_multilayer_network")


def distributed_network(root: Path) -> str:
    _, _, frame = _populated_frame(root, "utility_restoration")
    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    _draw_network(ax, frame, entropy_overlay=True)
    ax.text(0.02, 0.03, "Halo size: local belief entropy\nHalo width: disagreement\nRed fill: confidence < 0.42",
            transform=ax.transAxes, fontsize=9.5, bbox={"facecolor": "white", "edgecolor": COLORS["light"], "alpha": 0.9})
    ax.set_title("Distributed sketches over an ad-hoc communication network")
    return _save(fig, root, "distributed_entropy_communication_network")


def phase_plane(root: Path) -> str:
    candidates = _read(root, "development/development_primary_v2/candidate_interventions.csv")
    candidates = candidates[candidates.information_condition == "private_fragmented"].copy()
    selections = _read(root, "statistics/panel_budget_selections.csv")
    selected = selections[
        (selections.feature_block == "kpi_plus_entropy_disagreement")
        & (selections.information_condition == "private_fragmented")
    ]["selected_candidates"].fillna("").str.split(";").explode()
    selected_ids = set(selected[selected.astype(bool)].astype(str))
    candidates["selected_by_prospective_triage"] = candidates.candidate_id.astype(str).isin(selected_ids)
    # One row per incident keeps its nine counterfactual actions from visually
    # pretending to be nine independent thermodynamic states.
    numeric = [
        "operational_energy", "mean_belief_entropy", "js_disagreement",
        "consensus_confidence",
    ]
    incident = candidates.groupby(
        ["cluster_id", "application", "regime", "incident_id"], as_index=False,
    ).agg({**{value: "first" for value in numeric}, "selected_by_prospective_triage": "max"})
    _write_data(root, "energy_entropy_disagreement_phase_plane", incident[
        ["application", "regime", "incident_id", "operational_energy", "mean_belief_entropy",
         "js_disagreement", "consensus_confidence", "selected_by_prospective_triage"]
    ])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.5), sharey=True)
    for ax, app in zip(axes, ("humanitarian", "utility_restoration")):
        frame = incident[incident.application == app]
        visible = frame.sample(min(800, len(frame)), random_state=55111)
        scatter = ax.scatter(
            visible.operational_energy, visible.js_disagreement,
            c=visible.consensus_confidence, cmap="viridis", vmin=0, vmax=1,
            s=18, marker=APP_MARKERS[app], alpha=0.30, edgecolors="none",
        )
        chosen = visible[visible.selected_by_prospective_triage]
        ax.scatter(
            chosen.operational_energy, chosen.js_disagreement,
            c=chosen.consensus_confidence, cmap="viridis", vmin=0, vmax=1,
            s=52, marker=APP_MARKERS[app], alpha=0.90,
            edgecolors=COLORS["red"], linewidths=0.9,
        )
        ax.axhline(0.0, color=COLORS["gray"], lw=0.8)
        ax.axvline(frame.operational_energy.median(), color=COLORS["gray"], ls="--", lw=1.0,
                   label="Nominal empirical center")
        ax.set_title(APP_LABELS[app]); ax.set_xlabel("Operational energy (normalized loss stress)")
    axes[0].set_ylabel("Inter-agent Jensen–Shannon disagreement")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gray"], markersize=5, label="Not selected"),
        Line2D([0], [0], marker="o", color=COLORS["red"], markerfacecolor="white", markersize=7, label="Selected intervention"),
        Line2D([0], [0], color=COLORS["gray"], ls="--", label="Empirical center"),
    ]
    axes[1].legend(handles=legend, frameon=False, loc="upper right")
    fig.colorbar(scatter, ax=axes, fraction=0.04, pad=0.03, label="Consensus confidence")
    fig.suptitle("Actual V5 state coordinates and cross-fitted budget selections")
    return _save(fig, root, "energy_entropy_disagreement_phase_plane")


def export_dashboard_replays(root: Path) -> List[str]:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required for vector dashboard exports")
    destination = root / "dashboard_exports"
    destination.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    outputs: List[str] = []
    for application in APP_LABELS:
        episode, replay, frame = _populated_frame(root, application)
        svg = destination / f"{application}_populated_replay.svg"
        pdf = destination / f"{application}_populated_replay.pdf"
        png = destination / f"{application}_populated_replay.png"
        svg.write_text(frame_svg_v5(frame) + "\n", encoding="utf-8")
        subprocess.run([converter, "-w", "900", "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
        subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "240", str(pdf), str(png.with_suffix(""))], check=True, capture_output=True)
        records.append({
            "application": application, "episode": str(episode.relative_to(root)),
            "step": frame.step, "replay_digest": replay.digest(),
            "authorized_operator_payload_only": True, "simulated_operator": True,
            "svg": str(svg.relative_to(root)), "pdf": str(pdf.relative_to(root)),
        })
        outputs.extend([str(svg.relative_to(root)), str(pdf.relative_to(root)), str(png.relative_to(root))])
    metadata = destination / "metadata.json"
    metadata.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(str(metadata.relative_to(root)))
    publication = root / "figures" / "pdf" / "populated_operator_dashboard.pdf"
    publication.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination / "utility_restoration_populated_replay.pdf", publication)
    subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "240", str(publication), str(root / "figures" / "png" / "populated_operator_dashboard")], check=True, capture_output=True)
    outputs.append(str(publication.relative_to(root)))
    return outputs


def dashboard_comparison(root: Path) -> str:
    candidates = _read(root, "development/development_primary_v2/candidate_interventions.csv")
    row = candidates[(candidates.application == "utility_restoration") & (candidates.information_condition == "private_fragmented")].iloc[0]
    kpi = ["visible_severity", "visible_backlog", "visible_delay", "resource_scarcity", "safety_risk", "commitment_strain"]
    thermo = ["mean_belief_entropy", "entropy_dispersion", "js_disagreement", "entropy_slope", "consensus_residual", "consensus_confidence"]
    labels = {value: value.replace("_", " ").title() for value in kpi + thermo}
    records = [{"view": "KPI-only", "field": field, "value": row[field]} for field in kpi]
    records += [{"view": "KPI + entropy/disagreement", "field": field, "value": row[field]} for field in kpi + thermo]
    _write_data(root, "dashboard_view_comparison", pd.DataFrame(records))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.6))
    for ax, title, fields, color in (
        (axes[0], "A. KPI-only", kpi, COLORS["blue"]),
        (axes[1], "B. KPI + entropy/disagreement", kpi + thermo, COLORS["green"]),
    ):
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0.03, 0.04), 0.94, 0.90, transform=ax.transAxes,
                    boxstyle="round,pad=0.02", facecolor=mpl.colors.to_rgba(color, 0.07), edgecolor=color, lw=1.8))
        ax.text(0.5, 0.87, title, transform=ax.transAxes, ha="center", weight="bold", fontsize=11.5)
        for index, field in enumerate(fields):
            y = 0.77 - index * 0.055
            ax.text(0.10, y, labels[field], transform=ax.transAxes, fontsize=8.8)
            ax.text(0.90, y, f"{row[field]:.3f}", transform=ax.transAxes, ha="right", fontsize=8.8, family="monospace")
        ax.text(0.5, 0.10, "Same incident · same time · same budget", transform=ax.transAxes,
                ha="center", color=COLORS["gray"], fontsize=9.2)
    fig.suptitle("Matched simulated-operator information conditions")
    return _save(fig, root, "operator_dashboard_kpi_vs_entropy")


def causal_funnel(root: Path) -> str:
    summary = _read(root, "development/development_primary_v2/episode_summary.csv")
    candidates = _read(root, "development/development_primary_v2/candidate_interventions.csv")
    # Panel A is the independent-agent action population; panel B is the
    # separate counterfactual candidate population. They are not a shared funnel.
    autonomous = [
        ("Panel actions", int(summary.incidents.sum())),
        ("Accepted", int(summary.fixed_accepted_actions.sum())),
        ("Reached service", int(summary.fixed_service_reaching_actions.sum())),
        ("Changed outcome", int(summary.coordination_changed_outcome.sum())),
    ]
    probes = [
        ("Counterfactual probes", len(candidates)),
        ("Accepted", int(candidates.accepted_action.sum())),
        ("Next stage", int(candidates.reached_next_stage.sum())),
        ("Reached service", int(candidates.reached_service.sum())),
        ("Beneficial", int(candidates.beneficial.sum())),
        ("Harmful", int(candidates.harmful.sum())),
    ]
    _write_data(root, "causal_populations", pd.DataFrame(
        [{"population": "autonomous", "stage": k, "count": v} for k, v in autonomous]
        + [{"population": "counterfactual probes", "stage": k, "count": v} for k, v in probes]
    ))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.6))
    for ax, values, title, color in (
        (axes[0], autonomous, "A. Autonomous material actions", COLORS["blue"]),
        (axes[1], probes, "B. Counterfactual intervention probes", COLORS["orange"]),
    ):
        labels, counts = zip(*values)
        y = np.arange(len(labels))[::-1]
        ax.barh(y, counts, color=color, alpha=0.80)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_title(title); ax.set_xlabel("Count")
        for yy, value in zip(y, counts):
            ax.text(value, yy, f" {value:,}", va="center", fontsize=9.2)
    fig.suptitle("Separate causal populations—no mixed-denominator funnel")
    return _save(fig, root, "causal_alert_to_outcome_funnels")


def feature_block_value(root: Path) -> str:
    selections = _read(root, "statistics/panel_budget_selections.csv")
    selections = selections[selections.information_condition == "private_fragmented"]
    blocks = ["local_kpi_only", "energy_only", "entropy_disagreement_only", "kpi_plus_energy", "kpi_plus_entropy_disagreement", "complete_thermodynamic", "exploratory_free_energy"]
    rows = []
    for (app, block), group in selections.groupby(["application", "feature_block"]):
        interval = paired_bootstrap(group.selected_effect, replicates=10000, seed=55121)
        rows.append({"application": app, "feature_block": block, "mean": interval["mean"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"], "clusters": len(group)})
    data = pd.DataFrame(rows)
    _write_data(root, "feature_block_incremental_value", data)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 5.4), sharey=True)
    for ax, app in zip(axes, APP_LABELS):
        frame = data[data.application == app].set_index("feature_block").reindex(blocks)
        x = np.arange(len(blocks))
        ax.errorbar(x, frame["mean"], yerr=[frame["mean"] - frame["ci_low"], frame["ci_high"] - frame["mean"]],
                    fmt=APP_MARKERS[app], color=APP_COLORS[app], capsize=2.5, lw=1.3)
        ax.axhline(0, color=COLORS["black"], lw=0.8); ax.set_title(APP_LABELS[app]); ax.set_xticks(x)
        ax.set_xticklabels(["KPI", "Energy", "Entropy\n+ disagreement", "KPI\n+ energy", "KPI + entropy\n+ disagreement", "Complete", "Free energy"], rotation=70, ha="right")
        ax.text(0.02, 0.97, f"n={int(frame.clusters.min())} panels/block", transform=ax.transAxes, va="top", fontsize=8.5)
    axes[0].set_ylabel("Budgeted causal intervention utility")
    fig.suptitle("Cross-fitted feature-block utility (cluster-bootstrap 95% CI)")
    return _save(fig, root, "feature_block_incremental_value")


def primary_forest(root: Path) -> str:
    data = _read(root, "statistics/primary_incremental_value.csv")
    data = data[data.information_condition == "private_fragmented"].copy()
    _write_data(root, "primary_cluster_effect_forest", data)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(data))[::-1]
    for yy, (_, row) in zip(y, data.iterrows()):
        color = APP_COLORS[row.application]
        ax.errorbar(row.absolute_gain, yy, xerr=[[row.absolute_gain - row.gain_ci95_low], [row.gain_ci95_high - row.absolute_gain]],
                    fmt=APP_MARKERS[row.application], color=color, capsize=4, markersize=7)
    ax.axvline(0, color=COLORS["black"], lw=1.0)
    ax.axvline(0.05, color=COLORS["green"], ls="--", lw=1.0, label="Frozen practical target (+0.05)")
    ax.set_yticks(y); ax.set_yticklabels([APP_LABELS[value] for value in data.application])
    ax.set_xlabel("KPI + entropy/disagreement minus KPI-only utility")
    ax.set_title("Primary development effect: all point estimates are negative")
    ax.legend(frameon=False)
    return _save(fig, root, "paired_cluster_effect_forest")


def fragmentation_interaction(root: Path) -> str:
    data = _read(root, "statistics/fragmentation_interaction.csv")
    _write_data(root, "fragmentation_public_interaction", data)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.0))
    x = np.arange(len(data)); width = 0.32
    axes[0].bar(x - width / 2, data.private_mean_gain, width, color=COLORS["purple"], label="Private fragmented")
    axes[0].bar(x + width / 2, data.public_mean_gain, width, color=COLORS["sky"], label="Public shared")
    axes[0].axhline(0, color=COLORS["black"], lw=0.8); axes[0].set_ylabel("Thermodynamic incremental utility")
    axes[0].set_xticks(x); axes[0].set_xticklabels([APP_LABELS[a] for a in data.application], rotation=25, ha="right")
    axes[0].legend(frameon=False)
    axes[1].errorbar(data.absolute_interaction, x,
        xerr=[data.absolute_interaction - data.ci95_low, data.ci95_high - data.absolute_interaction],
        fmt="o", color=COLORS["green"], capsize=4)
    axes[1].axvline(0, color=COLORS["black"], lw=0.8); axes[1].set_yticks(x)
    axes[1].set_yticklabels([APP_LABELS[a] for a in data.application]); axes[1].set_xlabel("Private minus public gain (95% CI)")
    fig.suptitle("Fragmented-information mechanism interaction is not supported")
    return _save(fig, root, "fragmented_vs_public_interaction")


def _pareto_rows(root: Path) -> pd.DataFrame:
    summary = _read(root, "development/development_primary_v2/episode_summary.csv")
    selections = _read(root, "statistics/panel_budget_selections.csv")
    selections = selections[selections.information_condition == "private_fragmented"]
    rows: List[Dict[str, Any]] = []
    for app, base in summary[summary.information_condition == "private_fragmented"].groupby("application"):
        sketch_bytes = float(base.sketch_bytes.mean())
        operational_bytes = float(base.operational_bytes.mean())
        baseline_loss = float(base.no_communication_loss.mean())
        rows.append({"application": app, "method": "No human", "loss": baseline_loss, "operator_minutes": 0.0, "communication_bytes": 0.0})
        for block, label, thermodynamic in (
            ("local_kpi_only", "KPI-only triage", False),
            ("kpi_plus_entropy_disagreement", "Entropy-assisted triage", True),
        ):
            frame = selections[(selections.application == app) & (selections.feature_block == block)]
            rows.append({"application": app, "method": label, "loss": float(frame.loss_after_selection.mean()),
                         "operator_minutes": float(frame.operator_minutes.mean()),
                         "communication_bytes": operational_bytes + (sketch_bytes if thermodynamic else 0.0)})
        rows.append({"application": app, "method": "Bounded oracle", "loss": float(base.bounded_oracle_loss.mean()),
                     "operator_minutes": float(base.bounded_oracle_operator_minutes.mean()),
                     "communication_bytes": operational_bytes + sketch_bytes})
    return pd.DataFrame(rows)


def communication_pareto(root: Path) -> str:
    data = _pareto_rows(root)
    _write_data(root, "communication_cost_service_loss", data)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.8), sharey=False)
    method_colors = {"No human": COLORS["gray"], "KPI-only triage": COLORS["blue"], "Entropy-assisted triage": COLORS["purple"], "Bounded oracle": COLORS["green"]}
    for ax, app in zip(axes, APP_LABELS):
        frame = data[data.application == app]
        for _, row in frame.iterrows():
            ax.scatter(row.communication_bytes, row.loss, color=method_colors[row.method], s=55, marker=APP_MARKERS[app])
        ax.set_title(APP_LABELS[app]); ax.set_xlabel("Bytes / panel")
    axes[0].set_ylabel("Primary loss after selection")
    handles = [Line2D([0], [0], marker="o", lw=0, color=color, label=label) for label, color in method_colors.items()]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Communication cost includes thermodynamic sketch traffic")
    return _save(fig, root, "communication_cost_service_loss_pareto")


def operator_pareto(root: Path) -> str:
    data = _pareto_rows(root)
    _write_data(root, "operator_effort_service_loss", data)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.8))
    method_colors = {"No human": COLORS["gray"], "KPI-only triage": COLORS["blue"], "Entropy-assisted triage": COLORS["purple"], "Bounded oracle": COLORS["green"]}
    for ax, app in zip(axes, APP_LABELS):
        frame = data[data.application == app]
        for _, row in frame.iterrows():
            ax.scatter(row.operator_minutes, row.loss, color=method_colors[row.method], s=55, marker=APP_MARKERS[app])
        ax.set_title(APP_LABELS[app]); ax.set_xlabel("Operator minutes")
    axes[0].set_ylabel("Primary loss after selection")
    handles = [Line2D([0], [0], marker="o", lw=0, color=color, label=label) for label, color in method_colors.items()]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Operator effort versus service outcome")
    return _save(fig, root, "operator_effort_service_loss_pareto")


def intervention_distribution(root: Path) -> str:
    candidates = _read(root, "development/development_primary_v2/candidate_interventions.csv")
    candidates = candidates[candidates.information_condition == "private_fragmented"]
    _write_data(root, "intervention_effect_distribution", candidates[["application", "regime", "action", "causal_effect", "beneficial", "harmful"]])
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.8), sharex=True, sharey=True)
    bins = np.linspace(-0.36, 0.42, 40)
    for ax, app in zip(axes, APP_LABELS):
        values = candidates[candidates.application == app].causal_effect
        ax.hist(values, bins=bins, color=APP_COLORS[app], alpha=0.78)
        ax.axvline(0, color=COLORS["black"], lw=0.9); ax.set_title(APP_LABELS[app]); ax.set_xlabel("Causal effect")
        ax.text(0.04, 0.94, f"harmful {(values < 0).mean():.1%}\nneutral {(values.abs() <= 1e-12).mean():.1%}\nbeneficial {(values > 0).mean():.1%}",
                transform=ax.transAxes, va="top", fontsize=8.6)
    axes[0].set_ylabel("Candidate interventions")
    fig.suptitle("The simulator permits beneficial, neutral, and bounded harmful choices")
    return _save(fig, root, "intervention_harm_benefit_distribution")


def abstention(root: Path) -> str:
    data = _read(root, "statistics/low_consensus_abstention.csv")
    _write_data(root, "low_consensus_abstention", data)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.9))
    x = np.arange(len(data)); width = 0.32
    axes[0].bar(x - width/2, data.forced_harmful_interventions, width, color=COLORS["red"], label="Forced selection")
    axes[0].bar(x + width/2, data.safe_harmful_interventions, width, color=COLORS["green"], label="Low-confidence abstention")
    axes[0].set_ylabel("Harmful interventions"); axes[0].legend(frameon=False)
    axes[1].bar(x, data.safe_minus_forced_utility, color=[APP_COLORS[a] for a in data.application])
    axes[1].axhline(0, color=COLORS["black"], lw=0.8); axes[1].set_ylabel("Safe minus forced utility")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([APP_LABELS[a] for a in data.application], rotation=25, ha="right")
    fig.suptitle("Low-consensus abstention is exercised and reduces harm")
    return _save(fig, root, "low_consensus_abstention")


def training_curves(root: Path) -> str:
    paths = sorted((root / "training" / "curves").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("V5 multi-seed curves are not complete")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    _write_data(root, "multiseed_training_curves", data)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8), sharey=True)
    for ax, method in zip(axes, ("ippo_kpi_only", "ippo_entropy_disagreement")):
        frame = data[data.method == method]
        for seed, group in frame.groupby("rl_seed"):
            ax.plot(group.environment_steps, group.sample_reward, alpha=0.65, lw=1.1, label=str(seed))
        ax.set_title(method.replace("ippo_", "IPPO ").replace("_", " ").title())
        ax.set_xlabel("Training decision epochs"); ax.legend(title="RL seed", frameon=False, fontsize=7.8)
    axes[0].set_ylabel("On-policy sample reward")
    fig.suptitle("All five independent RL training seeds")
    return _save(fig, root, "multiseed_rl_learning_curves")


def training_evaluation(root: Path) -> str:
    manifest = _read(root, "training/seed_manifest.csv")
    _write_data(root, "multiseed_policy_evaluation", manifest)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    methods = list(manifest.method.unique()); x = np.arange(len(methods))
    for index, method in enumerate(methods):
        values = manifest[manifest.method == method].evaluation_mean_reward
        ax.scatter(np.full(len(values), index) + np.linspace(-0.07, 0.07, len(values)), values,
                   color=COLORS["blue"] if "kpi_only" in method else COLORS["purple"], s=55)
        ax.hlines(values.mean(), index - 0.18, index + 0.18, color=COLORS["black"], lw=2)
    ax.set_xticks(x); ax.set_xticklabels([value.replace("ippo_", "IPPO ").replace("_", " ").title() for value in methods])
    ax.set_ylabel("Development evaluation reward"); ax.axhline(0, color=COLORS["black"], lw=0.8)
    ax.set_title("Between-training-seed decentralized policy performance")
    return _save(fig, root, "multiseed_policy_evaluation")


def calibration(root: Path) -> str:
    predictions = _read(root, "statistics/candidate_crossfit_predictions.csv")
    predictions = predictions[
        (predictions.information_condition == "private_fragmented")
        & predictions.feature_block.isin(["local_kpi_only", "kpi_plus_entropy_disagreement"])
        & predictions.application.isin(["humanitarian", "utility_restoration"])
    ].copy()
    predictions["bin"] = predictions.groupby(["application", "feature_block"])["predicted_value"].transform(
        lambda value: pd.qcut(value, 8, labels=False, duplicates="drop")
    )
    data = predictions.groupby(["application", "feature_block", "bin"], as_index=False).agg(
        predicted=("predicted_value", "mean"), observed=("causal_effect", "mean"), rows=("candidate_id", "size"), clusters=("cluster_id", "nunique"),
    )
    _write_data(root, "calibration_reliability", data)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8), sharex=True, sharey=True)
    styles = {"local_kpi_only": (COLORS["blue"], "o", "KPI-only"), "kpi_plus_entropy_disagreement": (COLORS["purple"], "s", "KPI + entropy/disagreement")}
    limits = [float(min(data.predicted.min(), data.observed.min())), float(max(data.predicted.max(), data.observed.max()))]
    for ax, app in zip(axes, ("humanitarian", "utility_restoration")):
        for block, (color, marker, label) in styles.items():
            frame = data[(data.application == app) & (data.feature_block == block)]
            ax.plot(frame.predicted, frame.observed, color=color, marker=marker, label=label)
        ax.plot(limits, limits, color=COLORS["gray"], ls="--", label="Perfect calibration")
        ax.set_title(APP_LABELS[app]); ax.set_xlabel("Predicted causal utility")
    axes[0].set_ylabel("Observed causal utility"); axes[1].legend(frameon=False)
    fig.suptitle("Cross-fitted action-value reliability")
    return _save(fig, root, "calibration_reliability")


def competitive_selection(root: Path) -> str:
    primary = _read(root, "statistics/primary_incremental_value.csv")
    primary = primary[primary.information_condition == "private_fragmented"]
    data = primary[["application", "paired_win_fraction", "paired_tie_fraction", "paired_loss_fraction", "clusters"]]
    _write_data(root, "competitive_panel_selection", data)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(data)); left = np.zeros(len(data))
    for column, label, color in (
        ("paired_loss_fraction", "Thermo worse", COLORS["red"]),
        ("paired_tie_fraction", "Same outcome", COLORS["gray"]),
        ("paired_win_fraction", "Thermo better", COLORS["green"]),
    ):
        ax.barh(y, data[column], left=left, color=color, label=label)
        left += data[column].to_numpy()
    ax.set_yticks(y); ax.set_yticklabels([f"{APP_LABELS[a]} (n={n})" for a, n in zip(data.application, data.clusters)])
    ax.set_xlabel("Fraction of independent panels"); ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.set_title("Competitive-panel selection outcomes")
    return _save(fig, root, "competitive_panel_selection_comparison")


def regime_heterogeneity(root: Path) -> str:
    selections = _read(root, "statistics/panel_budget_selections.csv")
    frame = selections[
        selections.information_condition.eq("private_fragmented")
        & selections.feature_block.isin(["local_kpi_only", "kpi_plus_entropy_disagreement"])
    ]
    left = frame[frame.feature_block == "local_kpi_only"]
    right = frame[frame.feature_block == "kpi_plus_entropy_disagreement"]
    paired = left.merge(right, on=["cluster_id", "application", "regime", "environment_seed"], suffixes=("_kpi", "_thermo"))
    paired["gain"] = paired.selected_effect_thermo - paired.selected_effect_kpi
    rows = []
    for (app, regime), group in paired.groupby(["application", "regime"]):
        interval = paired_bootstrap(group.gain, replicates=10000, seed=55131)
        rows.append({"application": app, "regime": regime, **interval})
    data = pd.DataFrame(rows)
    _write_data(root, "regime_heterogeneity", data)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 5.2), sharey=True)
    regimes = list(REGIME_LABELS)
    for ax, app in zip(axes, APP_LABELS):
        values = data[data.application == app].set_index("regime").reindex(regimes)
        y = np.arange(len(regimes))[::-1]
        ax.errorbar(values["mean"], y, xerr=[values["mean"] - values.ci_low, values.ci_high - values["mean"]],
                    fmt=APP_MARKERS[app], color=APP_COLORS[app], capsize=2.5)
        ax.axvline(0, color=COLORS["black"], lw=0.8); ax.set_title(APP_LABELS[app]); ax.set_xlabel("Thermo − KPI utility")
        ax.set_yticks(y); ax.set_yticklabels([REGIME_LABELS[r] for r in regimes] if app == "commercial" else [])
    fig.suptitle("Regime heterogeneity does not rescue the primary no-go")
    return _save(fig, root, "regime_specific_heterogeneity")


def trigger_timing(root: Path) -> str:
    data = _read(root, "statistics/trigger_feasibility.csv")
    _write_data(root, "trigger_timing_false_activation", data)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    x = np.arange(len(data)); width = 0.30
    axes[0].bar(x - width/2, data.timely_activation_fraction, width, color=COLORS["green"], label="Timely")
    axes[0].bar(x + width/2, data.missed_eligible_fraction, width, color=COLORS["red"], label="Missed")
    axes[0].axhline(0.75, color=COLORS["black"], ls="--", lw=1, label="Timely target")
    axes[0].legend(frameon=False); axes[0].set_ylabel("Fraction")
    axes[1].bar(x, data.nominal_false_activation_fraction, color=COLORS["orange"])
    axes[1].axhline(0.10, color=COLORS["red"], ls="--", lw=1, label="Maximum")
    axes[1].legend(frameon=False); axes[1].set_ylabel("Nominal false activation")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([APP_LABELS[a] for a in data.application], rotation=25, ha="right")
    fig.suptitle("Trigger feasibility fails on false activations and utility misses")
    return _save(fig, root, "trigger_timing_and_false_alarms")


def generate(root: Path) -> List[str]:
    configure_style()
    outputs = [
        architecture(root), three_applications(root), utility_network(root), distributed_network(root),
        phase_plane(root), dashboard_comparison(root), causal_funnel(root), feature_block_value(root),
        primary_forest(root), fragmentation_interaction(root), communication_pareto(root), operator_pareto(root),
        intervention_distribution(root), abstention(root), training_curves(root), training_evaluation(root),
        calibration(root), competitive_selection(root), regime_heterogeneity(root), trigger_timing(root),
    ]
    outputs.extend(export_dashboard_replays(root))
    return outputs
