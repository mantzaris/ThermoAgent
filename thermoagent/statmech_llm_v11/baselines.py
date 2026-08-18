"""Deployable scripted reference agents used as non-LLM controls."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .core import EvidencePacket, ProviderResult, bayesian_probability_right


class ScriptedBayesianProvider:
    """Independent typed policy that combines the same authorized packets exactly."""

    def decide(self, prompt: str, seed: int) -> ProviderResult:
        del seed
        envelope = json.loads(prompt.split("\nCONTROLLED_TASK=", 1)[1].split("\nLOCAL_DELIVERY_MODE=", 1)[0])
        view = envelope["authorized_local_view"]
        private = EvidencePacket.from_mapping(view["private_evidence"])
        delivered = [EvidencePacket.from_mapping(value) for value in view["delivered_evidence"]]
        probability = bayesian_probability_right([private] + delivered)
        belief = "right" if probability >= 0.5 else "left"
        payload: Mapping[str, object] = {
            "probability_right": probability,
            "belief_choice": belief,
            "action_choice": "select_" + belief,
            "commitment_status": "provisional",
            "outgoing_evidence_action": "send_private_evidence",
            "reason_code": "combined_evidence" if delivered else "private_evidence",
            "explanation": "Exact Bayesian reference over authorized packets.",
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return ProviderResult(payload, True, False, 0, 0, 0.0, digest)
