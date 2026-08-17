import struct

import numpy as np
import pytest

from thermoagent.v8_wire import (
    CHECKSUM,
    HEADER,
    WireFormatError,
    decode_belief_sketch,
    encode_belief_sketch,
)


@pytest.mark.parametrize("encoding,maximum_l1", [
    ("fp32", 1e-6),
    ("fp16", 5e-4),
    ("uint8_simplex", 0.02),
])
def test_v8_wire_round_trip_counts_exact_actual_bytes(encoding, maximum_l1):
    belief = (0.63, 0.21, 0.10, 0.04, 0.015, 0.005)
    encoded = encode_belief_sketch(
        origin_sender_id=7,
        transmitter_id=9,
        target_asset_id=12,
        sent_step=42,
        confidence=0.713,
        belief=belief,
        encoding=encoding,
        hop_count=2,
        flags=3,
    )
    decoded = decode_belief_sketch(encoded.wire_bytes)
    assert encoded.total_bytes == HEADER.size + encoded.payload_bytes + CHECKSUM.size
    assert decoded.total_bytes == len(encoded.wire_bytes)
    assert decoded.origin_sender_id == 7
    assert decoded.transmitter_id == 9
    assert decoded.target_asset_id == 12
    assert decoded.sent_step == 42
    assert decoded.hop_count == 2
    assert decoded.flags == 3
    assert decoded.confidence == pytest.approx(0.713, abs=1.0 / 65535.0)
    assert sum(decoded.belief) == pytest.approx(1.0)
    assert np.sum(np.abs(np.asarray(decoded.belief) - np.asarray(belief))) <= maximum_l1
    assert encoded.quantization_l1_error <= maximum_l1


def test_v8_wire_bytes_are_deterministic():
    arguments = dict(
        origin_sender_id=1, transmitter_id=1, target_asset_id=3,
        sent_step=8, confidence=0.8, belief=(0.1, 0.2, 0.7),
        encoding="fp16", hop_count=0,
    )
    first = encode_belief_sketch(**arguments)
    second = encode_belief_sketch(**arguments)
    assert first.wire_bytes == second.wire_bytes


def test_v8_wire_rejects_truncation_corruption_and_unknown_version():
    encoded = encode_belief_sketch(
        origin_sender_id=1, transmitter_id=1, target_asset_id=2,
        sent_step=3, confidence=0.9, belief=(0.8, 0.1, 0.1),
    ).wire_bytes
    with pytest.raises(WireFormatError, match="shorter|length"):
        decode_belief_sketch(encoded[:-1])
    corrupted = bytearray(encoded)
    corrupted[HEADER.size] ^= 0x01
    with pytest.raises(WireFormatError, match="integrity"):
        decode_belief_sketch(bytes(corrupted))
    unknown = bytearray(encoded)
    unknown[4] = 99
    body = bytes(unknown[:-CHECKSUM.size])
    import zlib
    unknown[-CHECKSUM.size:] = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    with pytest.raises(WireFormatError, match="schema version"):
        decode_belief_sketch(bytes(unknown))


def test_v8_uint8_simplex_is_exact_and_handles_small_tail_states():
    encoded = encode_belief_sketch(
        origin_sender_id=1, transmitter_id=1, target_asset_id=2,
        sent_step=3, confidence=0.9,
        belief=(0.97, 0.01, 0.01, 0.005, 0.003, 0.002),
        encoding="uint8_simplex",
    )
    decoded = decode_belief_sketch(encoded.wire_bytes)
    payload = encoded.wire_bytes[HEADER.size:-CHECKSUM.size]
    assert sum(payload) == 255
    assert sum(decoded.belief) == pytest.approx(1.0)
