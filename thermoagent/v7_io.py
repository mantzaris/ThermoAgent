"""Transparent readers for compact, Git-facing V7 research artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def compressed_path(path: Path) -> Path:
    """Return the conventional gzip sibling for an uncompressed path."""
    return Path(str(path) + ".gz")


def resolve_artifact(path: Path) -> Path:
    """Resolve an artifact, preferring its plain form and accepting gzip."""
    if path.exists():
        return path
    candidate = compressed_path(path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def read_json_artifact(path: Path) -> Dict[str, Any]:
    resolved = resolve_artifact(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt", encoding="utf-8") as handle:
            return dict(json.load(handle))
    return dict(json.loads(resolved.read_text(encoding="utf-8")))


def read_csv_artifact(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV whether it is stored plainly or losslessly gzipped."""
    return pd.read_csv(resolve_artifact(path), **kwargs)


def episode_artifacts(raw_root: Path) -> List[Path]:
    """Return one canonical episode artifact per run directory."""
    plain = {path.parent: path for path in raw_root.glob("**/episode.json")}
    compressed = {path.parent: path for path in raw_root.glob("**/episode.json.gz")}
    duplicates = sorted(set(plain).intersection(compressed))
    if duplicates:
        raise RuntimeError(
            "both plain and compressed episode artifacts exist: %s"
            % ", ".join(str(path) for path in duplicates[:5])
        )
    values = {**compressed, **plain}
    return [values[key] for key in sorted(values, key=lambda value: str(value))]
