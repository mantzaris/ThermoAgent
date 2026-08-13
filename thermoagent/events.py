"""Append-only event ledger with deterministic IDs and replay-friendly JSONL."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EVENT_KINDS = {
    "topology_snapshot",
    "observation_delivery",
    "memory_retrieval",
    "llm_request",
    "llm_structured_response",
    "tool_call",
    "tool_result",
    "message",
    "message_delivery",
    "public_signal",
    "macrostate_sketch",
    "coordination_trigger",
    "trigger_local_state",
    "coordination_activity",
    "trigger_alert_result",
    "offer",
    "counteroffer",
    "commitment",
    "coalition_event",
    "disruption",
    "environment_transition",
    "plan_revision",
    "metric",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class Event:
    event_id: str
    step: int
    kind: str
    actor: str
    payload: Dict[str, Any]
    private_to: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLedger:
    """A deterministic sequence; timestamps belong in the run manifest."""

    def __init__(self) -> None:
        self._events: List[Event] = []

    def append(
        self,
        step: int,
        kind: str,
        actor: str,
        payload: Dict[str, Any],
        private_to: Optional[str] = None,
    ) -> Event:
        if kind not in EVENT_KINDS:
            raise ValueError("unknown event kind: %s" % kind)
        event = Event(
            event_id="E%08d" % (len(self._events) + 1),
            step=int(step),
            kind=kind,
            actor=str(actor),
            payload=_jsonable(payload),
            private_to=private_to,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def visible_to(self, agent_id: str) -> List[Event]:
        return [e for e in self._events if e.private_to in (None, agent_id)]

    def public_events(self) -> List[Event]:
        return [e for e in self._events if e.private_to is None]

    def write_jsonl(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            # gzip's default wall-clock timestamp makes identical ledgers have
            # different bytes. A blank embedded filename and mtime=0 make run
            # checksums reproducible without changing the JSONL content.
            with path.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", mtime=0
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed, encoding="utf-8", newline=""
                    ) as handle:
                        for event in self._events:
                            handle.write(
                                json.dumps(event.as_dict(), sort_keys=True)
                                + "\n"
                            )
        else:
            with path.open("w", encoding="utf-8", newline="") as handle:
                for event in self._events:
                    handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
        return sha256_file(path)

    @classmethod
    def read_jsonl(cls, path: Path) -> "EventLedger":
        ledger = cls()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            lines = list(handle)
        for line in lines:
            row = json.loads(line)
            expected = "E%08d" % (len(ledger._events) + 1)
            if row["event_id"] != expected:
                raise ValueError("non-contiguous event ledger")
            ledger._events.append(Event(**row))
        return ledger

    def digest(self) -> str:
        blob = "\n".join(json.dumps(e.as_dict(), sort_keys=True) for e in self._events)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
