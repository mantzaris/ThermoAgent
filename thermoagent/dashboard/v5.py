"""Privacy-preserving V5 dashboard replay and vector export."""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..events import EventLedger
from ..v5_environment import payload_digest


@dataclass
class V5DashboardFrame:
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
    information_boundary: str = "V5 operator-authorized payload only"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class V5DashboardReplay:
    """GPU-free replay that never reads evaluator-only candidate effects."""

    def __init__(self, episode_path: Path, event_path: Optional[Path] = None) -> None:
        self.episode_path = Path(episode_path)
        self.episode = json.loads(self.episode_path.read_text(encoding="utf-8"))
        if event_path is None:
            candidates = list(self.episode_path.parent.glob("events.jsonl*"))
            if len(candidates) != 1:
                raise FileNotFoundError("expected one V5 event ledger beside episode")
            event_path = candidates[0]
        self.ledger = EventLedger.read_jsonl(Path(event_path))
        if self.ledger.digest() != self.episode["event_ledger_digest"]:
            raise ValueError("V5 dashboard ledger digest mismatch")
        self._frames = self._build_frames()

    def _views(self) -> Dict[int, List[Dict[str, Any]]]:
        values: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.ledger.public_events():
            if event.kind != "v5_operator_view":
                continue
            payload = dict(event.payload)
            digest = str(payload.pop("payload_sha256"))
            if payload_digest(payload) != digest:
                raise ValueError("V5 dashboard operator payload hash mismatch")
            prohibited = {"true_mode", "correct_action", "fragmentation", "stochastic_tape", "base_loss"}
            if prohibited.intersection(payload.get("features", {})):
                raise ValueError("V5 dashboard privacy leak")
            values.setdefault(event.step, []).append({"payload": payload, "sha256": digest})
        return values

    def _network(self, topology: Mapping[str, Any]) -> Dict[str, Any]:
        incidents = {str(value["incident_id"]): value for value in topology["incidents"]}
        nodes: List[Dict[str, Any]] = []
        scoped_agents: Dict[str, List[tuple[str, Mapping[str, Any]]]] = {}
        for agent_id, identity in topology["agents"].items():
            scoped_agents.setdefault(str(identity["incident_scope"][0]), []).append((agent_id, identity))
        for scope, scoped in sorted(scoped_agents.items()):
            center = incidents[scope]["location"]
            for index, (agent_id, identity) in enumerate(sorted(scoped)):
                angle = 2.0 * math.pi * index / max(len(scoped), 1) + 0.25
                radius = 0.22
                nodes.append({
                    "agent_id": agent_id,
                    "role": identity["role"],
                    "incident_scope": scope,
                    "location": [
                        float(center[0]) + radius * math.cos(angle),
                        float(center[1]) + radius * math.sin(angle),
                    ],
                    "autonomy_level": 0,
                })
        communication_edges: List[List[str]] = []
        logistics_edges: List[List[str]] = []
        by_scope: Dict[str, List[str]] = {}
        for node in nodes:
            by_scope.setdefault(node["incident_scope"], []).append(node["agent_id"])
        for agents in by_scope.values():
            communication_edges.extend([[agents[index], agents[(index + 1) % len(agents)]] for index in range(len(agents))])
            logistics_edges.extend([[agents[0], value] for value in agents[1:]])
        service_edges = [
            [agents[-1], next_agents[0]]
            for agents, next_agents in zip(list(by_scope.values()), list(by_scope.values())[1:] + list(by_scope.values())[:1])
        ]
        return {
            "nodes": nodes,
            "service_edges": service_edges,
            "communication_edges": communication_edges,
            "logistics_edges": logistics_edges,
            "physical_edges": logistics_edges,
            "authorized_emergency_edges": [],
            "visible_incidents": [dict(value) for value in incidents.values()],
        }

    def _build_frames(self) -> List[V5DashboardFrame]:
        topology = next(event.payload for event in self.ledger.events if event.kind == "topology_snapshot")
        network_base = self._network(topology)
        views = self._views()
        public_by_step: Dict[int, List[Any]] = {}
        for event in self.ledger.public_events():
            public_by_step.setdefault(event.step, []).append(event)
        maximum_step = max([event.step for event in self.ledger.events] or [0])
        current_views: Dict[str, Dict[str, Any]] = {}
        interventions: List[Dict[str, Any]] = []
        progress: List[Dict[str, Any]] = []
        frames: List[V5DashboardFrame] = []
        for step in range(maximum_step + 1):
            hashes: List[str] = []
            for row in views.get(step, []):
                current_views[str(row["payload"]["incident_id"])] = row["payload"]
                hashes.append(row["sha256"])
            for event in public_by_step.get(step, []):
                if event.kind in ("attention_allocation", "operator_action"):
                    interventions.append({"event": event.kind, "step": step, **event.payload})
                elif event.kind in ("material_progress", "restoration_action"):
                    progress.append({"event": event.kind, "step": step, **event.payload})
            ranked_views = sorted(
                current_views.values(),
                key=lambda value: float(value["features"].get("visible_severity", 0.0)),
                reverse=True,
            )
            selected = ranked_views[0] if ranked_views else {"features": {}, "incident_id": None}
            features = selected["features"]
            queue = [
                {
                    "incident_id": value["incident_id"],
                    "severity": value["features"].get("visible_severity"),
                    "entropy": value["features"].get("mean_belief_entropy"),
                    "disagreement": value["features"].get("js_disagreement"),
                    "consensus_confidence": value["features"].get("consensus_confidence"),
                    "reason": "severity plus distributed uncertainty",
                }
                for value in ranked_views
            ]
            network = json.loads(json.dumps(network_base))
            autonomy = 4 if interventions else 2 if ranked_views else 0
            for node in network["nodes"]:
                node["autonomy_level"] = autonomy
                scoped = current_views.get(node["incident_scope"], {"features": {}})["features"]
                node["energy"] = scoped.get("operational_energy")
                node["entropy"] = scoped.get("mean_belief_entropy")
                node["disagreement"] = scoped.get("js_disagreement")
                node["consensus_confidence"] = scoped.get("consensus_confidence")
            frames.append(V5DashboardFrame(
                step=step,
                application=str(topology["application"]),
                scenario=str(topology["regime"]),
                method="V5 competitive development replay",
                network=network,
                thermodynamics={
                    "energy": features.get("operational_energy"),
                    "entropy": features.get("mean_belief_entropy"),
                    "entropy_anomaly": features.get("entropy_dispersion"),
                    "entropy_slope": features.get("entropy_slope"),
                    "free_energy": features.get("free_energy"),
                    "disagreement": features.get("js_disagreement"),
                    "consensus_confidence": features.get("consensus_confidence"),
                    "service_loss": features.get("visible_severity"),
                    "autonomy_level": autonomy,
                    "intervention_score": None,
                    "prospective_threshold": None,
                },
                alert_queue=queue,
                interventions=interventions[-8:],
                workload={
                    "operator_budget": int(topology["operator_budget"]),
                    "selected_interventions": len([value for value in interventions if value["event"] == "operator_action"]),
                    "queue_length": len(queue),
                    "operator_minutes": sum(6.0 for value in interventions if value["event"] == "operator_action"),
                    "simulated_operator": True,
                },
                explanation={
                    "view_condition": "KPI plus distributed entropy/disagreement",
                    "alert_reason": "competitive incident ranking",
                    "prediction": {"selected_incident": selected.get("incident_id")},
                    "features": dict(features),
                    "operator_payload_only": True,
                    "counterfactual_outcomes_excluded": True,
                },
                material_progress=progress[-12:],
                view_hashes=hashes,
            ))
        return frames

    @property
    def frames(self) -> List[V5DashboardFrame]:
        return list(self._frames)

    def frame(self, step: int) -> V5DashboardFrame:
        return self._frames[max(0, min(int(step), len(self._frames) - 1))]

    def digest(self) -> str:
        blob = json.dumps([value.as_dict() for value in self._frames], sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        summary = self.episode.get("summary", {})
        return {
            "run_id": summary.get("run_id", self.episode.get("run_id")),
            "application": summary.get("application", self.episode.get("application")),
            "scenario": summary.get("regime", self.episode.get("regime")),
            "method": "V5 competitive development replay",
            "operator_profile": "bounded simulated operator",
            "operator_view": "KPI plus distributed entropy/disagreement",
            "steps": len(self._frames),
            "replay_digest": self.digest(),
            "gpu_required": False,
            "evidence_boundary": "development-only simulated operator; no actual human evidence",
            "information_boundary": "hashed deployable payload; evaluator counterfactuals excluded",
        }


def frame_svg_v5(frame: V5DashboardFrame, width: int = 1200, height: int = 760) -> str:
    nodes = frame.network["nodes"]
    positions = {
        node["agent_id"]: (315 + 215 * float(node["location"][0]), 315 - 190 * float(node["location"][1]))
        for node in nodes
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%" height="100%" fill="#F6F8FA"/>',
        '<style>text{font-family:Liberation Sans,Arial,sans-serif;fill:#18212F}.title{font-size:30px;font-weight:700}.head{font-size:21px;font-weight:700}.label{font-size:16px}.small{font-size:14px}.panel{fill:#fff;stroke:#C7D0DB;stroke-width:1.2}</style>',
        '<text x="28" y="40" class="title">ThermoHITL V5 populated replay — %s</text>' % html.escape(frame.application.replace("_", " ")),
        '<text x="28" y="64" class="small">Simulated operator · development evidence · evaluator outcomes excluded</text>',
        '<rect x="24" y="84" width="610" height="610" rx="8" class="panel"/>',
        '<rect x="655" y="84" width="520" height="610" rx="8" class="panel"/>',
        '<text x="45" y="116" class="head">Independent-agent network</text>',
        '<text x="678" y="116" class="head">Competitive alert queue</text>',
    ]
    for key, color, dash, size in (
        ("service_edges", "#009E73", "", 3.2),
        ("logistics_edges", "#7A8798", "", 2.0),
        ("communication_edges", "#0072B2", "6 4", 1.3),
    ):
        for left, right in frame.network.get(key, []):
            if left in positions and right in positions:
                x1, y1 = positions[left]; x2, y2 = positions[right]
                parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f" %s/>' % (x1, y1, x2, y2, color, size, 'stroke-dasharray="%s"' % dash if dash else ""))
    abbreviations = {
        "distribution_node": "Zone", "field_crew": "Crew", "communications": "Comms",
        "cyber_defense": "Cyber", "resource_allocation": "Resources", "critical_load": "Load",
        "regional_coordinator": "Coord", "supplier": "Supplier", "carrier": "Carrier",
        "warehouse": "Warehouse", "retailer": "Retailer", "coordinator": "Coord",
        "ngo": "NGO", "regional_hub": "Hub", "clinic": "Clinic",
    }
    for index, node in enumerate(nodes):
        x, y = positions[node["agent_id"]]
        disagreement = float(node.get("disagreement") or 0.0)
        confidence = float(node.get("consensus_confidence") or 1.0)
        color = "#D55E00" if confidence < 0.42 else "#E69F00" if disagreement > 0.10 else "#56B4E9"
        parts.append('<circle cx="%.1f" cy="%.1f" r="18" fill="%s" stroke="#26384A" stroke-width="2"/>' % (x, y, color))
        label_y = y + 33 if index % 2 == 0 else y - 27
        parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" class="small">%s</text>' % (
            x, label_y, html.escape(abbreviations.get(node["role"], node["role"].replace("_", " "))),
        ))
    for index, alert in enumerate(frame.alert_queue[:4]):
        y = 155 + index * 94
        confidence = alert.get("consensus_confidence")
        parts.extend([
            '<rect x="678" y="%d" width="470" height="78" rx="6" fill="#F3F6F9" stroke="#D6DDE5"/>' % (y - 24),
            '<text x="695" y="%d" class="label">%d. %s</text>' % (y, index + 1, html.escape(str(alert["incident_id"]))),
            '<text x="695" y="%d" class="small">severity %s · entropy %s · disagreement %s</text>' % (y + 24, _fmt(alert.get("severity")), _fmt(alert.get("entropy")), _fmt(alert.get("disagreement"))),
            '<text x="695" y="%d" class="small">consensus confidence %s</text>' % (y + 46, _fmt(confidence)),
        ])
    thermo = frame.thermodynamics
    parts.extend([
        '<text x="678" y="570" class="head">Authorized selected-incident view</text>',
        '<text x="695" y="602" class="label">Energy %s   Entropy %s   Disagreement %s</text>' % (_fmt(thermo.get("energy")), _fmt(thermo.get("entropy")), _fmt(thermo.get("disagreement"))),
        '<text x="695" y="630" class="label">Consensus %s   Autonomy level %s</text>' % (_fmt(thermo.get("consensus_confidence")), _fmt(thermo.get("autonomy_level"))),
        '<text x="695" y="661" class="small">View hashes: %s</text>' % html.escape(", ".join(frame.view_hashes)[:56] or "none at this step"),
        '</svg>',
    ])
    return "".join(parts)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else "%.3f" % float(value)
