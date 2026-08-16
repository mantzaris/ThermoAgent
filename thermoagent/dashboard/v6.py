"""GPU-free V6 generalized-entropic replay dashboard and vector export."""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..events import EventLedger
from ..v6_environment import payload_digest
from ..v6_experiments import read_episode_json


PROHIBITED = {
    "true_mode", "correct_action", "stochastic_tape", "future_outcome",
    "counterfactual_effect", "evaluator_distributed_error",
}


def _contains(value: Any, keys: set) -> bool:
    if isinstance(value, Mapping):
        return bool(keys.intersection(value)) or any(_contains(item, keys) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains(item, keys) for item in value)
    return False


@dataclass
class V6DashboardFrame:
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
    alternatives: List[str]
    view_hashes: List[str]
    information_boundary: str = "operator-authorized V6 payload only"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class V6DashboardReplay:
    def __init__(self, episode_path: Path, event_path: Optional[Path] = None) -> None:
        self.episode_path = Path(episode_path)
        self.episode = read_episode_json(self.episode_path)
        if event_path is None:
            candidates = list(self.episode_path.parent.glob("events.jsonl*"))
            if len(candidates) != 1:
                raise FileNotFoundError("expected one V6 event ledger")
            event_path = candidates[0]
        self.ledger = EventLedger.read_jsonl(Path(event_path))
        expected = self.episode.get("event_ledger_digest") or self.episode.get("summary", {}).get("event_ledger_digest")
        if expected and self.ledger.digest() != expected:
            raise ValueError("V6 dashboard ledger digest mismatch")
        self._frames = self._build_frames()

    def _operator_views(self) -> Dict[int, List[Dict[str, Any]]]:
        output: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.ledger.events:
            if event.kind != "operator_view" or not event.payload.get("v6"):
                continue
            if event.private_to != "simulated_operator":
                raise ValueError("V6 operator view has incorrect audience")
            view = dict(event.payload["authorized_view"])
            if payload_digest(view) != event.payload["payload_sha256"]:
                raise ValueError("V6 operator view hash mismatch")
            if _contains(view, PROHIBITED):
                raise ValueError("V6 dashboard privacy leak")
            output.setdefault(event.step, []).append({
                "incident_id": event.payload["incident_id"],
                "view": view,
                "sha256": event.payload["payload_sha256"],
            })
        return output

    @staticmethod
    def _role(agent_id: str, application: str) -> str:
        prefix = application + "_"
        value = agent_id[len(prefix):] if agent_id.startswith(prefix) else agent_id
        return value.rsplit("_", 2)[0]

    def _network(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        agent_ids = sorted(snapshot["agent_ids"])
        nodes = []
        for index, agent_id in enumerate(agent_ids):
            angle = 2.0 * math.pi * index / max(len(agent_ids), 1)
            nodes.append({
                "agent_id": agent_id,
                "role": self._role(agent_id, str(snapshot["application"])),
                "location": [math.cos(angle), math.sin(angle)],
                "autonomy_level": 0,
            })
        communication = []
        service = []
        for index, agent_id in enumerate(agent_ids):
            communication.append([agent_id, agent_ids[(index + 1) % len(agent_ids)]])
            if index % 3 == 2:
                service.append([agent_ids[index - 2], agent_id])
        return {
            "nodes": nodes,
            "communication_edges": communication,
            "service_edges": service,
            "logistics_edges": service,
            "physical_edges": service,
            "authorized_emergency_edges": [],
        }

    def _build_frames(self) -> List[V6DashboardFrame]:
        snapshot = next(value.payload for value in self.ledger.events if value.kind == "v6_panel_snapshot")
        network_base = self._network(snapshot)
        views = self._operator_views()
        events_by_step: Dict[int, List[Any]] = {}
        for event in self.ledger.events:
            events_by_step.setdefault(event.step, []).append(event)
        maximum = max([value.step for value in self.ledger.events] or [0])
        active: Dict[str, Dict[str, Any]] = {}
        interventions: List[Dict[str, Any]] = []
        frames: List[V6DashboardFrame] = []
        operator_minutes = 0.0
        for step in range(maximum + 1):
            hashes = []
            for item in views.get(step, []):
                active[str(item["incident_id"])] = item["view"]
                hashes.append(item["sha256"])
            for event in events_by_step.get(step, []):
                if event.kind in ("v6_operator_escalation", "v6_operator_response", "v6_delegation_decision"):
                    interventions.append({"event": event.kind, "step": step, **event.payload})
                if event.kind == "v6_operator_response":
                    operator_minutes += float(event.payload["operator_minutes"])
            ranked = sorted(
                active.items(),
                key=lambda value: (
                    float(value[1].get("js_disagreement", 0.0))
                    + float(value[1].get("consensus_residual", 0.0)),
                    value[0],
                ), reverse=True,
            )
            incident_id, selected = ranked[0] if ranked else (None, {})
            proposal = dict(selected.get("proposal", {}))
            queue = [{
                "incident_id": key,
                "proposed_action": value.get("proposal", {}).get("action"),
                "estimated_causal_utility": value.get("proposal", {}).get("action_value"),
                "action_value_margin": value.get("proposal", {}).get("value_margin"),
                "shannon": value.get("shannon_local"),
                "tsallis_q_0_5": value.get("tsallis_0_5_local"),
                "gini_simpson": value.get("gini_simpson_local"),
                "disagreement": value.get("js_disagreement"),
                "graph_disagreement": value.get("graph_disagreement"),
                "consensus": value.get("consensus"),
                "consensus_residual": value.get("consensus_residual"),
                "contributors": value.get("contributors", []),
                "missing_agents": value.get("missing_agents", []),
            } for key, value in ranked]
            network = json.loads(json.dumps(network_base))
            contributors = set(selected.get("contributors", []))
            missing = set(selected.get("missing_agents", []))
            for node in network["nodes"]:
                node["autonomy_level"] = 5 if incident_id else 0
                node["consensus_status"] = "missing" if node["agent_id"] in missing else "contributor" if node["agent_id"] in contributors else "unobserved"
            frames.append(V6DashboardFrame(
                step=step,
                application=str(snapshot["application"]),
                scenario=str(snapshot["regime"]),
                method=str(self.episode.get("summary", {}).get("controller", "V6 replay")),
                network=network,
                thermodynamics={
                    "energy": selected.get("operational_energy"),
                    "effective_temperature": selected.get("effective_temperature"),
                    "entropy": selected.get("shannon_local"),
                    "tsallis_q_0_5": selected.get("tsallis_0_5_local"),
                    "gini_simpson": selected.get("gini_simpson_local"),
                    "entropy_anomaly": selected.get("pooled_uncertainty"),
                    "pooled_uncertainty": selected.get("pooled_uncertainty"),
                    "entropy_slope": selected.get("entropy_slope"),
                    "free_energy": selected.get("free_energy_diagnostic"),
                    "disagreement": selected.get("js_disagreement"),
                    "graph_disagreement": selected.get("graph_disagreement"),
                    "consensus_confidence": selected.get("consensus"),
                    "consensus_residual": selected.get("consensus_residual"),
                    "service_loss": selected.get("local_kpis", {}).get("visible_severity"),
                    "autonomy_level": 5 if incident_id else 0,
                    "intervention_score": proposal.get("action_value"),
                    "prospective_threshold": None,
                },
                alert_queue=queue,
                interventions=interventions[-10:],
                workload={
                    "operator_budget": 4,
                    "operator_minutes": operator_minutes,
                    "queue_length": max(0, sum(value["event"] == "v6_operator_escalation" for value in interventions) - sum(value["event"] == "v6_operator_response" for value in interventions)),
                    "simulated_operator": True,
                },
                explanation={
                    "view_condition": "authorized generalized-entropic consensus view",
                    "alert_reason": "selective-risk and low-consensus escalation",
                    "prediction": {
                        "incident_id": incident_id,
                        "proposed_action": proposal.get("action"),
                        "estimated_action_value": proposal.get("action_value"),
                        "action_value_margin": proposal.get("value_margin"),
                    },
                    "contributors": selected.get("contributors", []),
                    "missing_agents": selected.get("missing_agents", []),
                    "operator_payload_only": True,
                    "counterfactual_outcomes_excluded": True,
                },
                alternatives=[
                    "execute autonomously", "communicate", "request evidence",
                    "defer", "abstain", "escalate operator",
                ],
                view_hashes=hashes,
            ))
        return frames

    @property
    def frames(self) -> List[V6DashboardFrame]:
        return list(self._frames)

    def frame(self, step: int) -> V6DashboardFrame:
        return self._frames[max(0, min(int(step), len(self._frames) - 1))]

    def digest(self) -> str:
        blob = json.dumps([value.as_dict() for value in self._frames], sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        summary = self.episode.get("summary", {})
        return {
            "run_id": self.episode.get("run_id", summary.get("run_id")),
            "application": self.episode.get("application", summary.get("application")),
            "scenario": self.episode.get("regime", summary.get("regime")),
            "method": summary.get("controller", "V6 replay"),
            "operator_profile": "bounded simulated operator",
            "operator_view": "generalized entropy, disagreement, and consensus",
            "steps": len(self._frames),
            "replay_digest": self.digest(),
            "gpu_required": False,
            "evidence_boundary": "simulated operator; no real-human evidence",
            "information_boundary": "hashed authorized payload; evaluator outcomes excluded",
        }


def frame_svg_v6(frame: V6DashboardFrame, width: int = 1200, height: int = 760) -> str:
    nodes = frame.network["nodes"]
    positions = {
        value["agent_id"]: (
            315 + 215 * float(value["location"][0]),
            355 + 245 * float(value["location"][1]),
        ) for value in nodes
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%" height="100%" fill="#F6F8FA"/>',
        '<style>text{font-family:Liberation Sans,Arial,sans-serif;fill:#18212F}.title{font-size:28px;font-weight:700}.head{font-size:19px;font-weight:700}.label{font-size:14px}.small{font-size:12px}.panel{fill:#fff;stroke:#C7D0DB;stroke-width:1.2}</style>',
        '<text x="28" y="40" class="title">V6 generalized-entropic selective-autonomy replay</text>',
        '<text x="28" y="64" class="small">Simulated operator · authorized view · evaluator counterfactuals excluded</text>',
        '<rect x="24" y="84" width="610" height="640" rx="8" class="panel"/>',
        '<rect x="655" y="84" width="520" height="640" rx="8" class="panel"/>',
        '<text x="45" y="116" class="head">Independent agents and ad-hoc consensus</text>',
        '<text x="678" y="116" class="head">Selective-autonomy queue</text>',
    ]
    for left, right in frame.network.get("communication_edges", []):
        if left in positions and right in positions:
            x1, y1 = positions[left]; x2, y2 = positions[right]
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0072B2" stroke-width="1.3" stroke-dasharray="6 4"/>' % (x1, y1, x2, y2))
    colors = {"contributor": "#56B4E9", "missing": "#D55E00", "unobserved": "#B8C2CC"}
    for node in nodes:
        x, y = positions[node["agent_id"]]
        parts.append('<circle cx="%.1f" cy="%.1f" r="17" fill="%s" stroke="#26384A" stroke-width="2"/>' % (x, y, colors[node["consensus_status"]]))
        parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" class="small">%s</text>' % (x, y + 31, html.escape(node["role"].replace("_", " "))))
    for index, alert in enumerate(frame.alert_queue[:4]):
        y = 150 + index * 102
        parts.extend([
            '<rect x="678" y="%d" width="470" height="88" rx="6" fill="#F3F6F9" stroke="#D6DDE5"/>' % (y - 22),
            '<text x="695" y="%d" class="label">%d. %s · %s</text>' % (y, index + 1, html.escape(str(alert["incident_id"])), html.escape(str(alert["proposed_action"]))),
            '<text x="695" y="%d" class="small">H %.3f · Hq .5 %.3f · Gini-Simpson %.3f</text>' % (y + 23, _number(alert["shannon"]), _number(alert["tsallis_q_0_5"]), _number(alert["gini_simpson"])),
            '<text x="695" y="%d" class="small">JS %.3f · graph %.3f · consensus %.3f</text>' % (y + 44, _number(alert["disagreement"]), _number(alert["graph_disagreement"]), _number(alert["consensus"])),
            '<text x="695" y="%d" class="small">missing %d · estimated value %.3f</text>' % (y + 65, len(alert["missing_agents"]), _number(alert["estimated_causal_utility"])),
        ])
    parts.extend([
        '<text x="678" y="590" class="head">Bounded choice</text>',
        '<text x="695" y="620" class="label">Execute · communicate · request evidence</text>',
        '<text x="695" y="646" class="label">Defer · abstain · escalate operator</text>',
        '<text x="695" y="680" class="small">Operator minutes %.2f · queue %s</text>' % (_number(frame.workload.get("operator_minutes")), frame.workload.get("queue_length")),
        '<text x="695" y="705" class="small">View hash %s</text>' % html.escape((frame.view_hashes[-1] if frame.view_hashes else "none")[:48]),
        '</svg>',
    ])
    return "".join(parts)


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0
