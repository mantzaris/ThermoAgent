"""Deterministic binary wire format for V8 belief sketches.

The byte count reported by V8 is the length of the exact byte string returned
by :func:`encode_belief_sketch`.  The format is deliberately small, auditable,
and implemented only with the Python standard library.  It is a simulated wire
protocol, not a claim about a deployed network stack.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .v6_entropy import probability_vector


MAGIC = b"TBV8"
SCHEMA_VERSION = 1
ENCODING_CODES = {"fp32": 1, "fp16": 2, "uint8_simplex": 3}
ENCODING_NAMES = {value: key for key, value in ENCODING_CODES.items()}
# magic, version, encoding, flags, hop, origin, transmitter, asset, step,
# confidence_q16, state_count, payload_length
HEADER = struct.Struct(">4sBBBBHHHIHHH")
CHECKSUM = struct.Struct(">I")
MAX_REGISTRY_ID = 65535
MAX_STEP = 2**32 - 1


class WireFormatError(ValueError):
    """Raised when a V8 wire message is malformed or unsupported."""


@dataclass(frozen=True)
class EncodedBeliefSketch:
    wire_bytes: bytes
    header_bytes: int
    payload_bytes: int
    integrity_bytes: int
    encoding: str
    quantization_l1_error: float

    @property
    def total_bytes(self) -> int:
        return len(self.wire_bytes)


@dataclass(frozen=True)
class DecodedBeliefSketch:
    schema_version: int
    encoding: str
    flags: int
    hop_count: int
    origin_sender_id: int
    transmitter_id: int
    target_asset_id: int
    sent_step: int
    confidence: float
    belief: Tuple[float, ...]
    header_bytes: int
    payload_bytes: int
    integrity_bytes: int
    total_bytes: int


def _registry_id(value: int, label: str) -> int:
    result = int(value)
    if not 0 <= result <= MAX_REGISTRY_ID:
        raise WireFormatError("%s must fit an unsigned 16-bit registry ID" % label)
    return result


def _simplex_uint8(probabilities: np.ndarray) -> np.ndarray:
    """Largest-remainder quantization whose integer masses sum exactly to 255."""
    scaled = probabilities * 255.0
    quantized = np.floor(scaled).astype(np.int64)
    remaining = int(255 - int(quantized.sum()))
    if remaining:
        fractions = scaled - quantized
        # Stable index tie-break makes the wire bytes deterministic.
        order = np.lexsort((np.arange(len(fractions)), -fractions))
        quantized[order[:remaining]] += 1
    if int(quantized.sum()) != 255 or np.any(quantized < 0) or np.any(quantized > 255):
        raise WireFormatError("uint8 simplex quantization invariant failed")
    return quantized.astype(np.uint8)


def _encode_payload(probabilities: np.ndarray, encoding: str) -> Tuple[bytes, np.ndarray]:
    if encoding == "fp32":
        payload = struct.pack(">%df" % len(probabilities), *probabilities.tolist())
        decoded = np.asarray(struct.unpack(">%df" % len(probabilities), payload), dtype=float)
    elif encoding == "fp16":
        payload = struct.pack(">%de" % len(probabilities), *probabilities.tolist())
        decoded = np.asarray(struct.unpack(">%de" % len(probabilities), payload), dtype=float)
    elif encoding == "uint8_simplex":
        quantized = _simplex_uint8(probabilities)
        payload = quantized.tobytes(order="C")
        decoded = quantized.astype(float) / 255.0
    else:
        raise WireFormatError("unsupported belief encoding: %s" % encoding)
    decoded = probability_vector(decoded)
    return payload, decoded


def encode_belief_sketch(
    *,
    origin_sender_id: int,
    transmitter_id: int,
    target_asset_id: int,
    sent_step: int,
    confidence: float,
    belief: Sequence[float],
    encoding: str = "fp16",
    hop_count: int = 0,
    flags: int = 0,
    schema_version: int = SCHEMA_VERSION,
) -> EncodedBeliefSketch:
    """Serialize a complete, self-framing V8 belief sketch."""
    if int(schema_version) != SCHEMA_VERSION:
        raise WireFormatError("encoder supports schema version %d only" % SCHEMA_VERSION)
    if encoding not in ENCODING_CODES:
        raise WireFormatError("unsupported belief encoding: %s" % encoding)
    origin = _registry_id(origin_sender_id, "origin_sender_id")
    transmitter = _registry_id(transmitter_id, "transmitter_id")
    asset = _registry_id(target_asset_id, "target_asset_id")
    step = int(sent_step)
    if not 0 <= step <= MAX_STEP:
        raise WireFormatError("sent_step must fit an unsigned 32-bit integer")
    hop = int(hop_count)
    if not 0 <= hop <= 255:
        raise WireFormatError("hop_count must fit an unsigned byte")
    flag_value = int(flags)
    if not 0 <= flag_value <= 255:
        raise WireFormatError("flags must fit an unsigned byte")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise WireFormatError("confidence must be finite and in [0, 1]")
    probabilities = probability_vector(belief)
    if len(probabilities) > 65535:
        raise WireFormatError("belief state count exceeds the wire limit")
    payload, decoded = _encode_payload(probabilities, encoding)
    if len(payload) > 65535:
        raise WireFormatError("belief payload exceeds the wire limit")
    confidence_q16 = int(round(confidence_value * 65535.0))
    header = HEADER.pack(
        MAGIC,
        SCHEMA_VERSION,
        ENCODING_CODES[encoding],
        flag_value,
        hop,
        origin,
        transmitter,
        asset,
        step,
        confidence_q16,
        len(probabilities),
        len(payload),
    )
    body = header + payload
    checksum = CHECKSUM.pack(zlib.crc32(body) & 0xFFFFFFFF)
    wire = body + checksum
    return EncodedBeliefSketch(
        wire_bytes=wire,
        header_bytes=HEADER.size,
        payload_bytes=len(payload),
        integrity_bytes=CHECKSUM.size,
        encoding=encoding,
        quantization_l1_error=float(np.sum(np.abs(probabilities - decoded))),
    )


def decode_belief_sketch(wire_bytes: bytes) -> DecodedBeliefSketch:
    """Validate and decode one complete V8 belief-sketch frame."""
    wire = bytes(wire_bytes)
    minimum = HEADER.size + CHECKSUM.size
    if len(wire) < minimum:
        raise WireFormatError("belief sketch is shorter than the minimum frame")
    header = wire[: HEADER.size]
    (
        magic,
        version,
        encoding_code,
        flags,
        hop,
        origin,
        transmitter,
        asset,
        step,
        confidence_q16,
        state_count,
        payload_length,
    ) = HEADER.unpack(header)
    if magic != MAGIC:
        raise WireFormatError("belief-sketch magic does not match")
    if version != SCHEMA_VERSION:
        raise WireFormatError("unsupported belief-sketch schema version: %d" % version)
    if encoding_code not in ENCODING_NAMES:
        raise WireFormatError("unknown belief encoding code: %d" % encoding_code)
    if state_count < 2:
        raise WireFormatError("belief sketch must contain at least two states")
    expected = HEADER.size + payload_length + CHECKSUM.size
    if len(wire) != expected:
        raise WireFormatError("belief-sketch length does not match its header")
    expected_crc = CHECKSUM.unpack(wire[-CHECKSUM.size :])[0]
    observed_crc = zlib.crc32(wire[:-CHECKSUM.size]) & 0xFFFFFFFF
    if expected_crc != observed_crc:
        raise WireFormatError("belief-sketch integrity check failed")
    payload = wire[HEADER.size : -CHECKSUM.size]
    encoding = ENCODING_NAMES[encoding_code]
    if encoding == "fp32":
        unit = 4
        if payload_length != state_count * unit:
            raise WireFormatError("FP32 payload length is inconsistent")
        values = struct.unpack(">%df" % state_count, payload)
    elif encoding == "fp16":
        unit = 2
        if payload_length != state_count * unit:
            raise WireFormatError("FP16 payload length is inconsistent")
        values = struct.unpack(">%de" % state_count, payload)
    else:
        if payload_length != state_count:
            raise WireFormatError("uint8 simplex payload length is inconsistent")
        masses = np.frombuffer(payload, dtype=np.uint8)
        if int(masses.sum()) != 255:
            raise WireFormatError("uint8 simplex payload does not sum to 255")
        values = masses.astype(float) / 255.0
    belief = probability_vector(values)
    return DecodedBeliefSketch(
        schema_version=version,
        encoding=encoding,
        flags=flags,
        hop_count=hop,
        origin_sender_id=origin,
        transmitter_id=transmitter,
        target_asset_id=asset,
        sent_step=step,
        confidence=float(confidence_q16 / 65535.0),
        belief=tuple(float(value) for value in belief),
        header_bytes=HEADER.size,
        payload_bytes=payload_length,
        integrity_bytes=CHECKSUM.size,
        total_bytes=len(wire),
    )
