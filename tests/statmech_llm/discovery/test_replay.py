import hashlib
import json

import pandas as pd
import pytest

from thermoagent.statmech_llm.discovery.provider import InvalidStructuredDecision
from thermoagent.statmech_llm.discovery.replay import (
    RecordedDecisionProvider,
    RecordedDecisionStore,
    compare_frames,
)


def _record(tmp_path, prompt="local state", valid=True):
    response = json.dumps(
        {
            "belief_choice": "amber",
            "action_choice": "cobalt",
            "confidence": 0.55,
            "commitment_status": "provisional",
            "memory_state": "conflict",
            "outgoing_signal": "amber",
            "tool_action": "execute_selected",
            "reason_code": "neighbor_messages",
        }
    )
    value = {
        "prompt": prompt,
        "seed": 17,
        "inference_sampling_temperature": 0.72,
        "responses": [response],
        "valid": valid,
        "first_pass_valid": valid,
        "repaired": False,
        "prompt_tokens": 31,
        "generated_tokens": 22,
        "latency_seconds": 0.25,
    }
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / ("call_00000001_%s.json" % digest[:12])).write_bytes(payload)
    return digest


def test_recorded_provider_validates_content_prompt_seed_and_temperature(tmp_path):
    digest = _record(tmp_path)
    provider = RecordedDecisionProvider(RecordedDecisionStore(tmp_path), [digest])
    result = provider.decide("local state", 17, 0.72)
    assert result.payload["belief_choice"] == "amber"
    assert result.raw_artifact_sha256 == digest
    provider.assert_consumed()

    mismatch = RecordedDecisionProvider(RecordedDecisionStore(tmp_path), [digest])
    with pytest.raises(RuntimeError, match="prompt differs"):
        mismatch.decide("changed state", 17, 0.72)


def test_recorded_invalid_decision_remains_invalid(tmp_path):
    digest = _record(tmp_path, valid=False)
    provider = RecordedDecisionProvider(RecordedDecisionStore(tmp_path), [digest])
    with pytest.raises(InvalidStructuredDecision):
        provider.decide("local state", 17, 0.72)


def test_frame_comparison_detects_transition_corruption():
    recorded = pd.DataFrame([{"state_before": 2, "energy": -1.25, "beliefs": "-1;1"}])
    assert compare_frames(recorded, recorded.to_dict("records")) == []
    corrupted = [{"state_before": 3, "energy": -1.25, "beliefs": "-1;1"}]
    assert compare_frames(recorded, corrupted) == ["state_before:1"]
