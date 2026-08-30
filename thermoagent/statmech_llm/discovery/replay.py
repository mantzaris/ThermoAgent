"""Deterministic regeneration of V12 transitions from external LLM records.

Raw prompts and generations stay outside Git. Each formal row stores only a
SHA-256 pointer. Replay validates the pointer, prompt, inference seed, and
temperature before feeding the recorded autonomous choice through the same
independent-agent transition code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .core import AgentDecision, ProviderResult
from .experiment import (
    _alpha_arms,
    _graph_for_panel,
    _hysteresis_panel_rows,
    _micro_response_rows,
    _panel_seed,
    formal_panel_design,
)
from .provider import InvalidStructuredDecision
from .simulation import run_trajectory
from .workflow import artifact_root, atomic_json, load_yaml, sha256_file, utc_now


class RecordedDecisionStore:
    """Resolve content-addressed external records without copying their text."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._paths: Dict[str, List[Path]] = {}
        for path in self.root.glob("call_*.json"):
            prefix = path.stem.rsplit("_", 1)[-1]
            self._paths.setdefault(prefix, []).append(path)

    def load(self, digest: str) -> Mapping[str, object]:
        if len(digest) != 64:
            raise ValueError("record pointer is not a full SHA-256 digest")
        candidates = self._paths.get(digest[:12], [])
        matches = [path for path in candidates if sha256_file(path) == digest]
        if len(matches) != 1:
            raise RuntimeError("external model record does not resolve uniquely: %s" % digest[:12])
        value = json.loads(matches[0].read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("external model record is not an object")
        return value


class RecordedDecisionProvider:
    """Replay a prespecified sequence of content-addressed Qwen decisions."""

    def __init__(self, store: RecordedDecisionStore, digests: Sequence[str]) -> None:
        self.store = store
        self.digests = [str(value) for value in digests]
        self.index = 0

    def decide(
        self, prompt: str, seed: int, sampling_temperature: Optional[float] = None
    ) -> ProviderResult:
        if self.index >= len(self.digests):
            raise RuntimeError("replay requested more decisions than recorded")
        digest = self.digests[self.index]
        self.index += 1
        record = self.store.load(digest)
        if str(record["prompt"]) != prompt:
            raise RuntimeError("replay prompt differs from recorded prompt")
        if int(record["seed"]) != int(seed):
            raise RuntimeError("replay inference seed differs from recorded seed")
        expected_temperature = float(record["inference_sampling_temperature"])
        actual_temperature = expected_temperature if sampling_temperature is None else float(sampling_temperature)
        if not np.isclose(expected_temperature, actual_temperature, atol=1e-12):
            raise RuntimeError("replay sampling temperature differs from recorded temperature")
        responses = record["responses"]
        if not isinstance(responses, list) or not responses:
            raise ValueError("recorded response list is empty")
        result = ProviderResult(
            payload={},
            first_pass_valid=bool(record["first_pass_valid"]),
            repaired=bool(record["repaired"]),
            prompt_tokens=int(record["prompt_tokens"]),
            generated_tokens=int(record["generated_tokens"]),
            latency_seconds=float(record["latency_seconds"]),
            raw_artifact_sha256=digest,
        )
        if not bool(record["valid"]):
            raise InvalidStructuredDecision("recorded decision was invalid", result)
        payload = json.loads(str(responses[-1]).strip())
        AgentDecision.from_mapping(payload)
        return ProviderResult(
            payload=payload,
            first_pass_valid=result.first_pass_valid,
            repaired=result.repaired,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            latency_seconds=result.latency_seconds,
            raw_artifact_sha256=digest,
        )

    def assert_consumed(self) -> None:
        if self.index != len(self.digests):
            raise RuntimeError("replay consumed %d of %d records" % (self.index, len(self.digests)))


def _digests(frame: pd.DataFrame) -> List[str]:
    values = frame["raw_artifact_sha256"].fillna("").astype(str).tolist()
    if any(len(value) != 64 for value in values):
        raise RuntimeError("one or more transition rows lacks an external raw-record digest")
    return values


def compare_frames(recorded: pd.DataFrame, regenerated: Sequence[Mapping[str, object]]) -> List[str]:
    """Return compact column mismatch counts for a replayed transition table."""

    replayed = pd.DataFrame(regenerated)
    if recorded.shape != replayed.shape:
        return ["shape %s != %s" % (recorded.shape, replayed.shape)]
    if set(recorded.columns) != set(replayed.columns):
        return ["column sets differ"]
    replayed = replayed.loc[:, recorded.columns]
    failures: List[str] = []
    for column in recorded.columns:
        left = recorded[column]
        right = replayed[column]
        if pd.api.types.is_numeric_dtype(left):
            matched = np.isclose(
                pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            matched = (
                left.fillna("").astype(str).to_numpy()
                == right.fillna("").astype(str).to_numpy()
            )
        if not bool(np.all(matched)):
            failures.append("%s:%d" % (column, int(np.sum(~matched))))
    return failures


def replay_formal(repository: Path) -> Dict[str, object]:
    """Regenerate every formal transition without invoking the model."""

    repository = Path(repository).resolve()
    protocol = load_yaml(repository / "configs/statmech_llm/discovery/protocol.yaml")
    root = artifact_root() / "formal"
    store = RecordedDecisionStore(artifact_root() / "raw" / "formal")
    units: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    micro = pd.read_csv(root / "microscopic_response.csv")
    provider = RecordedDecisionProvider(store, _digests(micro))
    replayed_micro = _micro_response_rows(provider, protocol)
    provider.assert_consumed()
    mismatches = compare_frames(micro, replayed_micro)
    units.append({"unit": "microscopic_response", "rows": len(micro), "mismatches": len(mismatches)})
    if mismatches:
        failures.append({"unit": "microscopic_response", "details": mismatches[:20]})

    for panel in formal_panel_design(protocol):
        path = root / "panels" / (str(panel["panel_id"]) + ".csv")
        recorded = pd.read_csv(path)
        provider = RecordedDecisionProvider(store, _digests(recorded))
        replayed = run_trajectory(
            provider,
            _graph_for_panel(panel),  # type: ignore[arg-type]
            _panel_seed(panel),
            int(panel["sweeps"]),
            str(panel["regime"]),
            float(panel["coupling_strength"]),
            float(panel["sampling_temperature"]),
            str(panel["initial_condition"]),
            control=str(panel["control"]),
            metadata={
                "family": panel["family"],
                "cluster_id": panel["cluster_id"],
                "panel_id": panel["panel_id"],
                "orientation": panel["orientation"],
                "burn_in_sweeps": panel["burn_in_sweeps"],
            },
        )
        provider.assert_consumed()
        mismatches = compare_frames(recorded, replayed)
        units.append({"unit": panel["panel_id"], "rows": len(recorded), "mismatches": len(mismatches)})
        if mismatches:
            failures.append({"unit": panel["panel_id"], "details": mismatches[:20]})

    settings = protocol["hysteresis"]  # type: ignore[index]
    for replicate in range(int(settings["graph_environment_replicates"])):
        for alpha, orientation, reverse in _alpha_arms(settings["nonreciprocity_levels"]):  # type: ignore[index]
            panel_id = "hysteresis_g%d_a%.2f_%s" % (replicate, alpha, orientation)
            recorded = pd.read_csv(root / "hysteresis" / (panel_id + ".csv"))
            provider = RecordedDecisionProvider(store, _digests(recorded))
            replay_id, replayed = _hysteresis_panel_rows(
                provider, settings, replicate, alpha, orientation, reverse
            )
            if replay_id != panel_id:
                raise RuntimeError("hysteresis replay panel identity changed")
            provider.assert_consumed()
            mismatches = compare_frames(recorded, replayed)
            units.append({"unit": panel_id, "rows": len(recorded), "mismatches": len(mismatches)})
            if mismatches:
                failures.append({"unit": panel_id, "details": mismatches[:20]})

    summary: Dict[str, object] = {
        "generated_at": utc_now(),
        "method": "content-addressed recorded-decision deterministic transition regeneration",
        "units_checked": len(units),
        "rows_checked": int(sum(int(item["rows"]) for item in units)),
        "units_with_mismatches": len(failures),
        "mismatch_details": failures[:20],
        "raw_transcripts_copied_into_repository": False,
        "status": "passed" if not failures else "failed",
    }
    atomic_json(summary, artifact_root() / "reproducibility" / "replay_summary.json")
    return summary
