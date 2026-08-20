"""Content-addressed deterministic regeneration of every V13 transition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from thermoagent.statmech_llm_v12.replay import (
    RecordedDecisionProvider,
    RecordedDecisionStore,
    compare_frames,
)

from .experiment import (
    formal_panel_design,
    graph_for_panel,
    microscopic_response_rows,
    panel_seed,
)
from .simulation import run_v13_trajectory
from .workflow import artifact_root, atomic_json, load_yaml, utc_now


def _digests(frame: pd.DataFrame) -> List[str]:
    values = frame["raw_artifact_sha256"].fillna("").astype(str).tolist()
    if any(len(value) != 64 for value in values):
        raise RuntimeError("transition row lacks a full external record digest")
    return values


def replay_formal(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    protocol = load_yaml(repository / "configs/statmech_v13/protocol_frozen_v1.2.yaml")
    root = artifact_root() / "formal"
    store = RecordedDecisionStore(artifact_root() / "raw" / "formal")
    units: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    micro = pd.read_csv(root / "microscopic_response.csv")
    provider = RecordedDecisionProvider(store, _digests(micro))
    regenerated = microscopic_response_rows(provider, protocol)
    provider.assert_consumed()
    mismatches = compare_frames(micro, regenerated)
    units.append({"unit": "microscopic_response", "rows": len(micro), "mismatches": len(mismatches)})
    if mismatches:
        failures.append({"unit": "microscopic_response", "details": mismatches[:20]})

    for panel in formal_panel_design(protocol):
        path = root / "panels" / (str(panel["panel_id"]) + ".csv")
        recorded = pd.read_csv(path)
        provider = RecordedDecisionProvider(store, _digests(recorded))
        regenerated = run_v13_trajectory(
            provider,
            graph_for_panel(panel),
            panel_seed(panel),
            int(panel["sweeps"]),
            str(panel["regime"]),
            float(panel["coupling_strength"]),
            float(panel["sampling_temperature"]),
            str(panel["initial_condition"]),
            str(panel["disruption"]),
            panel["periods_sweeps"],  # type: ignore[arg-type]
            metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
        )
        provider.assert_consumed()
        mismatches = compare_frames(recorded, regenerated)
        units.append({"unit": panel["panel_id"], "rows": len(recorded), "mismatches": len(mismatches)})
        if mismatches:
            failures.append({"unit": panel["panel_id"], "details": mismatches[:20]})

    summary: Dict[str, object] = {
        "generated_at": utc_now(),
        "method": "content-addressed autonomous-decision transition regeneration",
        "units_checked": len(units),
        "rows_checked": int(sum(int(item["rows"]) for item in units)),
        "units_with_mismatches": len(failures),
        "mismatch_details": failures,
        "raw_transcripts_copied_into_repository": False,
        "status": "passed" if not failures else "failed",
    }
    atomic_json(summary, artifact_root() / "reproducibility" / "replay_summary.json")
    return summary
