"""Privacy-preserving deterministic dashboard replay for ThermoHITL v4.

The deployable replay deliberately reconstructs only topology, explicit alert
messages, bounded operator actions, and schema-validated operator-view payloads.
Evaluator time-series fields and private per-agent thermodynamic events are not
used to populate the display.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..events import EventLedger
from ..v4_types import OperatorViewV4, validate_operator_view_v4


@dataclass
class V4DashboardFrame:
    """One operator-authorized dashboard frame reconstructed from the ledger."""

    step: int
    application: str
    scenario: str
    method: str
    network: Dict[str, Any]
    thermodynamics: Dict[str, Any]
    alert_queue: List[Dict[str, Any]]
    interventions: List[Dict[str, Any]]
    workload: Dict[str, Any]
    explanation: Dict[str, Any]
    material_progress: List[Dict[str, Any]]
    view_hashes: List[str]
    information_boundary: str = "operator-authorized payload only"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _topology_network(topology: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "nodes": [dict(value) for value in topology["agents"].values()],
        "service_edges": [list(value) for value in topology.get("service_edges", [])],
        "communication_edges": [
            list(value) for value in topology.get("communication_edges", [])
        ],
        "logistics_edges": [list(value) for value in topology.get("logistics_edges", [])],
        # Compatibility with the dependency-light web renderer.
        "physical_edges": [list(value) for value in topology.get("logistics_edges", [])],
        "authorized_emergency_edges": [],
        "visible_incidents": [],
    }


class V4DashboardReplay:
    """GPU-free replay that enforces the tested v4 operator-view boundary."""

    def __init__(self, episode_path: Path, event_path: Optional[Path] = None) -> None:
        self.episode_path = Path(episode_path)
        self.episode = json.loads(self.episode_path.read_text(encoding="utf-8"))
        if event_path is None:
            candidates = sorted(self.episode_path.parent.glob("events.jsonl*"))
            if len(candidates) != 1:
                raise FileNotFoundError("expected one event ledger beside v4 episode")
            event_path = candidates[0]
        self.event_path = Path(event_path)
        self.ledger = EventLedger.read_jsonl(self.event_path)
        if self.ledger.digest() != self.episode["event_ledger_digest"]:
            raise ValueError("dashboard ledger digest does not match episode")
        self._frames = self._build_frames()

    def _validated_views(self) -> Dict[int, List[Dict[str, Any]]]:
        output: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.ledger.events:
            if event.kind != "operator_view_v4":
                continue
            payload = dict(event.payload["payload"])
            view = OperatorViewV4(**payload)
            validate_operator_view_v4(view)
            digest = view.digest()
            if digest != event.payload["sha256"]:
                raise ValueError("operator-view hash mismatch at step %d" % event.step)
            output.setdefault(event.step, []).append(
                {"payload": view.as_dict(), "sha256": digest}
            )
        return output

    def _build_frames(self) -> List[V4DashboardFrame]:
        topology = next(
            event.payload for event in self.ledger.events
            if event.kind == "topology_snapshot"
        )
        events_by_step: Dict[int, List[Any]] = {}
        for event in self.ledger.events:
            events_by_step.setdefault(event.step, []).append(event)
        views_by_step = self._validated_views()
        base_network = _topology_network(topology)
        queues: Dict[str, Dict[str, Any]] = {}
        interventions: List[Dict[str, Any]] = []
        material: List[Dict[str, Any]] = []
        disruption_seen = False
        latest_authorized_view: Optional[Dict[str, Any]] = None
        frames: List[V4DashboardFrame] = []

        for row in self.episode["time_series"]:
            step = int(row["step"])
            current_views = views_by_step.get(step, [])
            if current_views:
                latest_authorized_view = current_views[-1]["payload"]
            for event in events_by_step.get(step, []):
                if event.kind == "disruption":
                    disruption_seen = True
                elif event.kind == "operator_queue" and event.payload.get("action") == "enqueued":
                    request = dict(event.payload["request"])
                    queues[str(request["request_id"])] = request
                elif event.kind == "attention_allocation":
                    queues.pop(str(event.payload["request_id"]), None)
                elif event.kind in {"operator_action", "operator_result"}:
                    interventions.append({"event": event.kind, "step": step, **event.payload})
                    interventions = interventions[-8:]
                elif event.kind in {"material_progress", "service_transition"}:
                    material.append({"event": event.kind, "step": step, **event.payload})
                    material = material[-12:]

            if latest_authorized_view is None:
                network = json.loads(json.dumps(base_network))
                features: Mapping[str, Any] = {}
                alert: Mapping[str, Any] = {}
                workload: Dict[str, Any] = {
                    "workload": 0.0,
                    "fatigue": 0.0,
                    "queue_length": len(queues),
                    "available_slots": 1,
                    "operator_minutes": 0.0,
                    "intervention_budget_remaining": 1,
                }
                condition = "no_operator_payload_yet"
                provenance: Mapping[str, Any] = {
                    "evaluator_only_fields_excluded": True,
                    "timestamp_step": step,
                }
            else:
                network = json.loads(json.dumps(latest_authorized_view["public_network"]))
                network["physical_edges"] = list(network.get("logistics_edges", []))
                features = latest_authorized_view["features"]
                alert = latest_authorized_view["alert"]
                workload = dict(latest_authorized_view["workload"])
                workload["queue_length"] = len(queues)
                condition = str(latest_authorized_view["condition"])
                provenance = latest_authorized_view["provenance"]

            if interventions:
                autonomy_level = 4
            elif current_views:
                autonomy_level = 3
            elif queues:
                autonomy_level = 2
            elif disruption_seen:
                autonomy_level = 1
            else:
                autonomy_level = 0
            for node in network.get("nodes", []):
                node["autonomy_level"] = autonomy_level

            # All values below originate in the authorized view. Missing values
            # remain null; evaluator-global episode metrics are never substituted.
            thermodynamics = {
                "energy": features.get("standardized_energy"),
                "standardized_energy": features.get("standardized_energy"),
                "operational_energy": features.get("operational_energy"),
                "entropy": features.get("entropy_anomaly"),
                "entropy_anomaly": features.get("entropy_anomaly"),
                "entropy_residual": features.get("entropy_residual"),
                "entropy_slope": features.get("entropy_slope"),
                "free_energy": features.get("free_energy_diagnostic"),
                "disagreement": features.get("belief_disagreement"),
                "consensus_confidence": features.get("consensus_confidence"),
                "service_loss": features.get("service_deficit"),
                "trigger_active": bool(current_views),
                "autonomy_level": autonomy_level,
                "intervention_score": alert.get("priority_score"),
                "prospective_threshold": 1.15,
            }
            frames.append(V4DashboardFrame(
                step=step,
                application=self.episode["application"],
                scenario=self.episode["regime"],
                method=self.episode["method"],
                network=network,
                thermodynamics=thermodynamics,
                alert_queue=list(queues.values()),
                interventions=list(interventions),
                workload=workload,
                explanation={
                    "view_condition": condition,
                    "alert_reason": alert.get("reason_code"),
                    "prediction": {
                        key: alert.get(key) for key in (
                            "predicted_benefit", "prediction_uncertainty",
                            "predicted_steps_until_collapse", "priority_score",
                        )
                    },
                    "feature_provenance": dict(provenance),
                    "features": dict(features),
                    "operator_payload_only": True,
                },
                material_progress=list(material),
                view_hashes=[value["sha256"] for value in current_views],
            ))
        return frames

    @property
    def frames(self) -> List[V4DashboardFrame]:
        return list(self._frames)

    def frame(self, step: int) -> V4DashboardFrame:
        if not self._frames:
            raise IndexError("dashboard replay has no frames")
        bounded = max(0, min(int(step), self._frames[-1].step))
        return next((value for value in self._frames if value.step == bounded), self._frames[-1])

    def digest(self) -> str:
        blob = json.dumps(
            [frame.as_dict() for frame in self._frames],
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.episode["run_id"],
            "application": self.episode["application"],
            "scenario": self.episode["regime"],
            "method": self.episode["method"],
            "operator_profile": self.episode["manifest_fields"].get("operator_profile"),
            "operator_view": self.episode["manifest_fields"].get("operator_view"),
            "steps": len(self._frames),
            "replay_digest": self.digest(),
            "gpu_required": False,
            "evidence_boundary": "simulated operator; development-only; no actual human evidence",
            "information_boundary": "schema-validated operator payload; no evaluator-global state",
        }


def frame_svg_v4(frame: V4DashboardFrame, width: int = 1200, height: int = 760) -> str:
    """Return a publication-readable vector export of a populated v4 frame."""

    nodes = frame.network.get("nodes", [])
    positions = {
        str(node["agent_id"]): node.get("location", [0.0, 0.0]) for node in nodes
    }

    def xy(agent_id: str) -> tuple[float, float]:
        left, top = positions[agent_id]
        return 300.0 + 210.0 * float(left), 300.0 - 190.0 * float(top)

    condition_label = str(frame.explanation.get("view_condition", "none")).replace("_", " ")
    reason_label = {
        "fragmented_belief_disagreement": "belief disagreement",
        "low_consensus_confidence": "low consensus confidence",
        "severity": "local severity",
    }.get(str(frame.explanation.get("alert_reason")), str(frame.explanation.get("alert_reason") or "none").replace("_", " "))
    latest_action = "none"
    if frame.interventions:
        action_value = str(frame.interventions[-1].get("action", frame.interventions[-1].get("code", "none")))
        latest_action = {
        "bounded_intervention_applied": "applied after displayed view",
            "authorize_verification": "verification authorized",
        }.get(action_value, action_value.replace("_", " "))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%" height="100%" fill="#F6F8FA"/>',
        # Liberation Sans avoids a Cairo/Poppler glyph-subset defect observed
        # with one DejaVu Sans dashboard export while retaining embedded,
        # selectable vector text. Arial and the generic family are fallbacks.
        '<style>text{font-family:Liberation Sans,Arial,sans-serif;fill:#18212F}.title{font-size:32px;font-weight:700}.head{font-size:25px;font-weight:700}.label{font-size:23px}.small{font-size:20px}.panel{fill:#fff;stroke:#C7D0DB;stroke-width:1.2}.service{stroke:#009E73;stroke-width:3}.logistics{stroke:#7A8798;stroke-width:2.2}.comm{stroke:#0072B2;stroke-width:1.4;stroke-dasharray:6 4}.emergency{stroke:#D55E00;stroke-width:5}</style>',
        '<text x="28" y="36" class="title">ThermoHITL v4 operator replay — %s — step %d</text>' % (html.escape(frame.application.replace("_", " ")), frame.step),
        '<text x="28" y="58" class="small">Simulated operator · development evidence · authorized payload only</text>',
        '<rect x="24" y="78" width="575" height="505" rx="8" class="panel"/>',
        '<text x="44" y="108" class="head">Multilayer network</text>',
    ]
    for css, key in (("service", "service_edges"), ("logistics", "logistics_edges"), ("comm", "communication_edges"), ("emergency", "authorized_emergency_edges")):
        for left, right in frame.network.get(key, []):
            if left in positions and right in positions:
                x1, y1 = xy(left); x2, y2 = xy(right)
                parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="%s"/>' % (x1, y1, x2, y2, css))
    for incident in frame.network.get("visible_incidents", []):
        location = incident.get("location", [0.0, 0.0])
        x, y = 300 + 210 * float(location[0]), 300 - 190 * float(location[1])
        low = incident.get("telemetry_confidence_state") == "low"
        parts.append('<circle cx="%.1f" cy="%.1f" r="35" fill="none" stroke="%s" stroke-width="3" stroke-dasharray="%s"/>' % (x, y, "#D55E00" if low else "#E69F00", "5 4" if low else "none"))
    for node in nodes:
        x, y = xy(str(node["agent_id"]))
        level = int(node.get("autonomy_level", 0))
        role = str(node.get("role", "agent"))
        role_labels = {
            "distribution_zone": "Zone", "substation": "Sub", "microgrid": "MG",
            "crew_dispatch": "Crew", "parts_depot": "Parts", "mobile_generation": "Gen",
            "critical_load": "Load", "incident_coordinator": "Coord",
            "supplier": "Supplier", "manufacturer": "Maker", "carrier": "Carrier",
            "warehouse": "Hub", "retailer": "Retail", "coordinator": "Coord",
            "ngo": "NGO", "agency": "Agency", "transport": "Transport",
            "depot": "Depot", "clinic": "Clinic", "community": "Community",
        }
        suffix = str(node["agent_id"]).rsplit("_", 1)[-1]
        node_label = role_labels.get(role, role.replace("_", " ").title()) + " " + suffix
        parts.extend([
            '<circle cx="%.1f" cy="%.1f" r="%d" fill="#56B4E9" stroke="#18212F" stroke-width="2"/>' % (x, y, 16 + level),
            '<text x="%.1f" y="%.1f" text-anchor="middle" class="small">%s</text>' % (x, y + 38, html.escape(node_label)),
            '<text x="%.1f" y="%.1f" text-anchor="middle" class="small">L%d</text>' % (x, y + 4, level),
        ])
    parts.extend([
        '<line x1="54" y1="542" x2="86" y2="542" class="service"/><text x="92" y="548" class="small">service</text>',
        '<line x1="285" y1="542" x2="317" y2="542" class="logistics"/><text x="323" y="548" class="small">restoration route</text>',
        '<line x1="54" y1="573" x2="86" y2="573" class="comm"/><text x="92" y="579" class="small">communication</text>',
        '<circle cx="301" cy="572" r="12" fill="none" stroke="#E69F00" stroke-width="3"/><text x="323" y="579" class="small">visible incident</text>',
        '<rect x="620" y="78" width="552" height="235" rx="8" class="panel"/>',
        '<text x="642" y="108" class="head">Authorized thermodynamic view</text>',
    ])
    thermo_rows = (
        ("Standardized energy", "standardized_energy"),
        ("Entropy anomaly", "entropy_anomaly"),
        ("Entropy slope", "entropy_slope"),
        ("Belief disagreement", "disagreement"),
        ("Consensus confidence", "consensus_confidence"),
        ("Free-energy diagnostic", "free_energy"),
    )
    for index, (label, key) in enumerate(thermo_rows):
        value = frame.thermodynamics.get(key)
        rendered = "not authorized" if value is None else "%.3f" % float(value)
        y = 142 + 27 * index
        parts.append('<text x="648" y="%d" class="label">%s</text><text x="1138" y="%d" text-anchor="end" class="label">%s</text>' % (y, html.escape(label), y, rendered))
    parts.extend([
        '<rect x="620" y="330" width="552" height="253" rx="8" class="panel"/>',
        '<text x="642" y="360" class="head">Alert, workload, and bounded response</text>',
        '<text x="648" y="394" class="label">View condition</text><text x="1138" y="394" text-anchor="end" class="label">%s</text>' % html.escape(condition_label),
        '<text x="648" y="423" class="label">Reason</text><text x="1138" y="423" text-anchor="end" class="label">%s</text>' % html.escape(reason_label),
        '<text x="648" y="452" class="label">Queue length</text><text x="1138" y="452" text-anchor="end" class="label">%d</text>' % len(frame.alert_queue),
        '<text x="648" y="481" class="label">Operator workload</text><text x="1138" y="481" text-anchor="end" class="label">%.2f</text>' % float(frame.workload.get("workload", 0.0)),
        '<text x="648" y="510" class="label">Minutes before decision</text><text x="1138" y="510" text-anchor="end" class="label">%.1f</text>' % float(frame.workload.get("operator_minutes", 0.0)),
        '<text x="648" y="539" class="small">Latest action</text><text x="1138" y="539" text-anchor="end" class="small">%s</text>' % html.escape(latest_action),
        '<rect x="24" y="604" width="1148" height="125" rx="8" class="panel"/>',
        '<text x="44" y="634" class="head">Data provenance and causal progress</text>',
        '<text x="44" y="661" class="small">View hash: %s</text>' % html.escape((frame.view_hashes[-1][:20] + "…") if frame.view_hashes else "no operator payload at this step"),
        '<text x="44" y="686" class="small">Material/service events visible so far: %d · evaluator-global and counterfactual fields excluded</text>' % len(frame.material_progress),
        '<text x="44" y="711" class="small">The display is a replay of a simulated operator condition, not evidence from a human participant.</text>',
        '</svg>',
    ])
    return "".join(parts)
