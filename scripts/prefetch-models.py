#!/usr/bin/env python3
"""Download and verify the two pinned study-model snapshots outside Git."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from thermoagent.statmech_llm.provider import MODEL_SPECS


def tree_summary(root: Path) -> dict:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "bytes": int(sum(path.stat().st_size for path in files)),
    }


def atomic_json(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> int:
    artifact_root = Path(
        os.environ.get(
            "THERMOAGENT_ARTIFACT_ROOT",
            "/workspace/ThermoAgent-JSTAT-artifacts",
        )
    ).resolve()
    repository = Path(__file__).resolve().parents[1]
    if artifact_root == repository or repository in artifact_root.parents:
        raise RuntimeError("model verification output must remain outside Git")
    api = HfApi()
    rows = []
    for key in ("qwen", "granite"):
        specification = MODEL_SPECS[key]
        resolved = api.model_info(
            specification.identifier,
            revision=specification.revision,
        ).sha
        if resolved != specification.revision:
            raise RuntimeError(
                "%s revision resolved to %s instead of %s"
                % (specification.identifier, resolved, specification.revision)
            )
        snapshot = Path(
            snapshot_download(
                repo_id=specification.identifier,
                revision=specification.revision,
                resume_download=True,
            )
        ).resolve()
        if snapshot.name != specification.revision:
            raise RuntimeError("downloaded snapshot directory does not match pinned revision")
        rows.append(
            {
                "model_key": key,
                "identifier": specification.identifier,
                "requested_revision": specification.revision,
                "resolved_revision": resolved,
                "snapshot_directory_name": snapshot.name,
                **tree_summary(snapshot),
            }
        )
    output = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": rows,
        "model_weights_in_repository": False,
    }
    destination = artifact_root / "reproducibility/model_snapshot_verification.json"
    atomic_json(output, destination)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
