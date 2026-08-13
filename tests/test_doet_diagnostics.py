import json
from pathlib import Path

import numpy as np

from thermoagent.doet_diagnostics import event_signature, semantic_payload, softmax


def test_semantic_payload_removes_only_generated_identifiers() -> None:
    value = {
        "message_id": "M001",
        "target": "retailer_01",
        "nested": {"commitment_id": "C001", "quantity": 3.0},
    }
    assert semantic_payload(value) == {
        "nested": {"quantity": 3.0},
        "target": "retailer_01",
    }


def test_message_signature_tracks_kind_and_recipient_not_id() -> None:
    event = {
        "actor": "supplier_01",
        "kind": "message",
        "step": 4,
        "payload": {
            "message_id": "M001",
            "kind": "summary",
            "recipient": "retailer_01",
        },
    }
    signature = event_signature(event, "messages")
    assert json.loads(signature) == {"kind": "summary", "recipient": "retailer_01"}


def test_softmax_is_normalized_and_shift_invariant() -> None:
    values = np.asarray([-1.0, 0.0, 2.0])
    assert np.isclose(softmax(values).sum(), 1.0)
    assert np.allclose(softmax(values), softmax(values + 100.0))
