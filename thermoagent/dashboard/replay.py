"""Deterministic dashboard frames from the event-sourced v3 ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..events import EventLedger
from ..human_operator import OperatorView, validate_operator_view


@dataclass
class DashboardFrame:
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

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _band_to_number(value: str) -> float:
    return {"low": 0.2, "nominal": 0.5, "high": 0.85}.get(str(value), 0.5)


class DashboardReplay:
    """GPU-free deterministic reconstruction of what the operator could see."""

    def __init__(self, episode_path: Path, event_path: Optional[Path] = None) -> None:
        self.episode_path = episode_path
        self.episode = json.loads(episode_path.read_text(encoding="utf-8"))
        if event_path is None:
            candidates = sorted(episode_path.parent.glob("events.jsonl*"))
            if len(candidates) != 1:
                raise FileNotFoundError("expected one event ledger beside episode")
            event_path = candidates[0]
        self.event_path = event_path
        self.ledger = EventLedger.read_jsonl(event_path)
        self._frames = self._build_frames()

    def _validated_views(self) -> Dict[int, List[Dict[str, Any]]]:
        output: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.ledger.events:
            if event.kind != "operator_view":
                continue
            view = OperatorView(**event.payload)
            validate_operator_view(view)
            output.setdefault(event.step, []).append(view.as_dict())
        return output

    def _build_frames(self) -> List[DashboardFrame]:
        topology = next(
            event.payload for event in self.ledger.events
            if event.kind == "topology_snapshot"
        )
        views_by_step = self._validated_views()
        events_by_step: Dict[int, List[Any]] = {}
        for event in self.ledger.events:
            events_by_step.setdefault(event.step, []).append(event)
        queues: Dict[str, Dict[str, Any]] = {}
        recent_interventions: List[Dict[str, Any]] = []
        material: List[Dict[str, Any]] = []
        current_autonomy = {
            agent_id: 0 for agent_id in topology["agents"]
        }
        current_physical = {tuple(edge) for edge in topology["physical_edges"]}
        current_communication = {tuple(edge) for edge in topology["communication_edges"]}
        frames: List[DashboardFrame] = []
        for row in self.episode["time_series"]:
            step = int(row["step"])
            for event in events_by_step.get(step, []):
                if event.kind == "operator_queue" and event.payload.get("action") == "enqueued":
                    queues[str(event.payload["incident_id"])] = dict(event.payload)
                elif event.kind == "attention_allocation":
                    queues.pop(str(event.payload["incident_id"]), None)
                elif event.kind in ("operator_action", "operator_result"):
                    recent_interventions.append({
                        "event": event.kind,
                        "step": step,
                        **event.payload,
                    })
                    recent_interventions = recent_interventions[-8:]
                elif event.kind == "autonomy_transition" and event.actor in current_autonomy:
                    if "to" in event.payload:
                        current_autonomy[event.actor] = int(event.payload["to"])
                elif event.kind == "material_progress":
                    material.append({"step": step, **event.payload})
                    material = material[-12:]
                elif event.kind == "disruption":
                    for edge in event.payload.get("route_closures", []):
                        current_physical.discard(tuple(edge))
                    for edge in event.payload.get("additional_authority_bottleneck_edges", []):
                        current_physical.discard(tuple(edge))
            views = views_by_step.get(step, [])
            latest_view = views[-1]["payload"] if views else None
            if latest_view:
                network = dict(latest_view["public_network"])
                for node in network.get("nodes", []):
                    node["autonomy_level"] = current_autonomy.get(node["agent_id"], node.get("autonomy_level", 0))
            else:
                network = {
                    "nodes": [
                        {
                            "agent_id": agent_id,
                            "role": values["role"],
                            "location": values["location"],
                            "autonomy_level": current_autonomy[agent_id],
                        }
                        for agent_id, values in topology["agents"].items()
                    ],
                    "physical_edges": [list(edge) for edge in sorted(current_physical)],
                    "communication_edges": [list(edge) for edge in sorted(current_communication)],
                    "active_shipments": [],
                    "authorized_emergency_edges": [],
                }
            thermodynamics = {
                "energy": float(row.get("distributed_energy_mean", 0.0)),
                "entropy": float(row.get("distributed_entropy_mean", 0.0)),
                "entropy_anomaly": float(row.get("entropy_anomaly_mean", 0.0)),
                "entropy_slope": float(row.get("entropy_slope_mean", 0.0)),
                "free_energy": float(row.get("exact_free_energy_diagnostic", 0.0)),
                "disagreement": float(row.get("disagreement_mean", 0.0)),
                "consensus_confidence": float(row.get("consensus_confidence_mean", 0.0)),
                "service_loss": float(row.get("service_loss", 0.0)),
                "trigger_active": int(row.get("human_requests", 0)) > 0,
                "autonomy_level": int(row.get("maximum_autonomy_level", 0)),
            }
            workload = {
                "workload": float(row.get("operator_workload", 0.0)),
                "fatigue": float(row.get("operator_fatigue", 0.0)),
                "queue_length": int(row.get("operator_queue_length", 0)),
                "active_interventions": int(row.get("operator_active", 0)),
                "operator_minutes": float(row.get("operator_minutes", 0.0)),
            }
            explanation = {
                "view_condition": latest_view.get("condition") if latest_view else "no_active_alert",
                "alert_reason": latest_view.get("incident", {}).get("reason") if latest_view else None,
                "prediction": {
                    key: latest_view.get("incident", {}).get(key)
                    for key in (
                        "expected_loss_without", "expected_loss_with",
                        "expected_benefit", "prediction_uncertainty",
                        "predicted_steps_until_collapse",
                    )
                } if latest_view else {},
                "feature_provenance": latest_view.get("provenance") if latest_view else {},
                "features": latest_view.get("features") if latest_view else {},
            }
            frames.append(DashboardFrame(
                step=step,
                application=self.episode["application"],
                scenario=self.episode["scenario"],
                method=self.episode["method"],
                network=network,
                thermodynamics=thermodynamics,
                alert_queue=list(queues.values()),
                interventions=list(recent_interventions),
                workload=workload,
                explanation=explanation,
                material_progress=list(material),
                view_hashes=[view["sha256"] for view in views],
            ))
        return frames

    @property
    def frames(self) -> List[DashboardFrame]:
        return list(self._frames)

    def frame(self, step: int) -> DashboardFrame:
        if not self._frames:
            raise IndexError("dashboard replay has no frames")
        step = max(0, min(int(step), self._frames[-1].step))
        return next((frame for frame in self._frames if frame.step == step), self._frames[-1])

    def digest(self) -> str:
        blob = json.dumps(
            [frame.as_dict() for frame in self._frames],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.episode["run_id"],
            "application": self.episode["application"],
            "scenario": self.episode["scenario"],
            "method": self.episode["method"],
            "operator_profile": self.episode["operator_profile"],
            "operator_view": self.episode["operator_view"],
            "steps": len(self._frames),
            "replay_digest": self.digest(),
            "gpu_required": False,
            "evidence_boundary": "simulated operator; no actual human evidence",
        }


def frame_svg(frame: DashboardFrame, width: int = 1100, height: int = 680) -> str:
    """Export one dependency-free vector dashboard state."""

    nodes = frame.network.get("nodes", [])
    positions: Dict[str, Sequence[float]] = {}
    for index, node in enumerate(nodes):
        location = node.get("location")
        if not isinstance(location, list) or len(location) != 2:
            angle = 2.0 * 3.141592653589793 * index / max(len(nodes), 1)
            location = [float(np.cos(angle)), float(np.sin(angle))]
        positions[node["agent_id"]] = location

    def xy(agent_id: str) -> tuple[float, float]:
        left, top = positions[agent_id]
        return 270.0 + 190.0 * float(left), 260.0 + 190.0 * float(top)

    def esc(value: Any) -> str:
        import html

        return html.escape(str(value))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%%" height="100%%" fill="#f7f8fa"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#18212f}.title{font-size:21px;font-weight:700}.label{font-size:12px}.small{font-size:10px}.panel{fill:white;stroke:#c9d1dc;stroke-width:1.2}.physical{stroke:#7a8798;stroke-width:3}.comm{stroke:#4c78a8;stroke-width:1.4;stroke-dasharray:5 4}.authorized{stroke:#e45756;stroke-width:5}</style>',
        '<text x="28" y="34" class="title">ThermoHITL operator replay — step %d</text>' % frame.step,
        '<rect x="20" y="54" width="510" height="410" rx="8" class="panel"/>',
        '<text x="38" y="82" class="label">Logistics and communication network</text>',
    ]
    for left, right in frame.network.get("physical_edges", []):
        if left in positions and right in positions:
            x1, y1 = xy(left); x2, y2 = xy(right)
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="physical"/>' % (x1, y1, x2, y2))
    for left, right in frame.network.get("communication_edges", []):
        if left in positions and right in positions:
            x1, y1 = xy(left); x2, y2 = xy(right)
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="comm"/>' % (x1, y1, x2, y2))
    for left, right in frame.network.get("authorized_emergency_edges", []):
        if left in positions and right in positions:
            x1, y1 = xy(left); x2, y2 = xy(right)
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="authorized"/>' % (x1, y1, x2, y2))
    colors = {"low": "#72b7b2", "nominal": "#f2cf5b", "high": "#e45756"}
    for node in nodes:
        x, y = xy(node["agent_id"])
        color = colors.get(node.get("energy_band", "nominal"), "#b9c2cf")
        radius = 15 + 2 * int(node.get("autonomy_level", 0))
        parts.extend([
            '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" stroke="#26384a" stroke-width="2"/>' % (x, y, radius, color),
            '<text x="%.1f" y="%.1f" text-anchor="middle" class="small">%s</text>' % (x, y + radius + 14, esc(node["agent_id"])),
        ])
    parts.extend([
        '<rect x="550" y="54" width="530" height="190" rx="8" class="panel"/>',
        '<text x="570" y="82" class="label">Thermodynamic system view</text>',
    ])
    thermo_rows = [
        ("Operational energy", frame.thermodynamics["energy"]),
        ("Operational entropy", frame.thermodynamics["entropy"]),
        ("Entropy anomaly", frame.thermodynamics["entropy_anomaly"]),
        ("Entropy slope", frame.thermodynamics["entropy_slope"]),
        ("Disagreement", frame.thermodynamics["disagreement"]),
        ("Service loss", frame.thermodynamics["service_loss"]),
    ]
    for index, (label, value) in enumerate(thermo_rows):
        y = 108 + 21 * index
        parts.append('<text x="570" y="%d" class="small">%s</text><text x="850" y="%d" class="small">%.3f</text>' % (y, esc(label), y, float(value)))
    parts.extend([
        '<rect x="550" y="260" width="255" height="204" rx="8" class="panel"/>',
        '<text x="570" y="288" class="label">Energy–entropy phase plane</text>',
        '<line x1="590" y1="430" x2="775" y2="430" stroke="#26384a"/><line x1="590" y1="430" x2="590" y2="310" stroke="#26384a"/>',
        '<line x1="682" y1="310" x2="682" y2="430" stroke="#b9c2cf" stroke-dasharray="4 3"/><line x1="590" y1="370" x2="775" y2="370" stroke="#b9c2cf" stroke-dasharray="4 3"/>',
    ])
    px = 590 + 185 * max(0.0, min(1.0, frame.thermodynamics["entropy"]))
    py = 430 - 120 * max(0.0, min(1.0, frame.thermodynamics["energy"]))
    parts.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#e45756"/><text x="682" y="452" class="small">entropy →</text><text x="562" y="370" class="small" transform="rotate(-90 562 370)">energy →</text>' % (px, py))
    parts.extend([
        '<rect x="825" y="260" width="255" height="204" rx="8" class="panel"/>',
        '<text x="845" y="288" class="label">Operator workload</text>',
    ])
    for index, key in enumerate(("workload", "fatigue", "queue_length", "active_interventions", "operator_minutes")):
        parts.append('<text x="845" y="%d" class="small">%s: %s</text>' % (316 + 25 * index, esc(key.replace("_", " ")), esc(frame.workload.get(key, 0))))
    parts.extend([
        '<rect x="20" y="482" width="1060" height="170" rx="8" class="panel"/>',
        '<text x="38" y="510" class="label">Alert, explanation, and intervention provenance</text>',
        '<text x="38" y="538" class="small">View: %s | reason: %s</text>' % (esc(frame.explanation.get("view_condition")), esc(frame.explanation.get("alert_reason"))),
        '<text x="38" y="562" class="small">Queue: %d | recent interventions: %d | material stages: %d</text>' % (len(frame.alert_queue), len(frame.interventions), len(frame.material_progress)),
        '<text x="38" y="586" class="small">Payload hashes: %s</text>' % esc(", ".join(value[:12] for value in frame.view_hashes) or "none"),
        '<text x="38" y="626" class="small">Simulated-operator evidence only; evaluator-global state is excluded except in explicitly labeled oracle views.</text>',
        '</svg>',
    ])
    return "".join(parts)
