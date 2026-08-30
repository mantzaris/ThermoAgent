#!/usr/bin/env python3
"""Compare reconstructed science tables with the retained reference package."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCIENCE_TABLES = {
    "hypothesis_effects.csv": (),
    "quench_recovery.csv": (),
    "macrostate_trajectories.csv": (),
    "macrostate_trajectories_all_windows.csv": (),
    "nominal_distance_diagnostics.csv": (),
    "irreversibility_sensitivity.csv": (),
    "memory_prompt_balance.csv": (),
    # Wall-clock latency is expected to differ in a fresh execution.  All
    # remaining columns are deterministic scientific or accounting outputs.
    "panel_statistics.csv": ("latency_seconds",),
}

# Cross-platform LAPACK reductions can differ by a few ulps even when every
# reconstructed state and every scientific estimand is identical.  Retain the
# stringent absolute tolerance for values near zero, and add a machine-scale
# relative allowance for large derived distances.  This is approximately
# 1.42e-14 (64 binary64 epsilons), not a scientific effect-size tolerance.
DEFAULT_RELATIVE_TOLERANCE = 64.0 * np.finfo(float).eps
JSON_NUMERIC_COLUMNS = {"cluster_values"}

OPTIONAL_PRIMARY_AUDITS = (
    "cluster_seed_audit_passed",
    "memory_control_audit",
)


def _selected_primary_science(payload: Mapping[str, object]) -> Dict[str, object]:
    """Select deterministic scientific/accounting fields from primary JSON.

    Fresh wall times, generation latency, tree byte digests, and timestamps are
    deliberately excluded.  Calls and tokens remain included because seeded
    reconstruction is expected to reproduce them exactly.
    """

    completion = payload["formal_completion"]
    if not isinstance(completion, Mapping):
        raise TypeError("formal_completion must be a mapping")
    model_rows = completion.get("models", [])
    if not isinstance(model_rows, list):
        raise TypeError("formal completion models must be a list")
    selected_models = []
    for row in model_rows:
        if not isinstance(row, Mapping):
            raise TypeError("formal model completion must be a mapping")
        selected_models.append(
            {
                key: row[key]
                for key in (
                    "status",
                    "model_key",
                    "model_id",
                    "model_revision",
                    "protocol_sha256",
                    "execution_source_sha256",
                    "planned_trajectories",
                    "completed_trajectories",
                    "planned_decisions",
                    "observed_decision_rows",
                    "model_calls",
                    "prompt_tokens",
                    "generated_tokens",
                    "invalid_after_repair",
                    "invalid_after_repair_fraction",
                )
            }
        )
    selected_models.sort(key=lambda row: str(row["model_key"]))
    selected = {
        "confirmatory_dispositions": payload["confirmatory_dispositions"],
        "formal_trajectories": payload["formal_trajectories"],
        "independent_clusters_per_model": payload["independent_clusters_per_model"],
        "model_keys": payload["model_keys"],
        "privacy_mutations": payload["privacy_mutations"],
        "nonfinite_primary_features": payload["nonfinite_primary_features"],
        "formal_completion": {
            key: completion[key]
            for key in (
                "status",
                "dynamic_trajectories",
                "observed_decision_rows",
                "model_calls",
                "prompt_tokens",
                "generated_tokens",
                "protocol_sha256",
                "execution_source_sha256",
            )
        },
        "formal_models": selected_models,
    }
    # These audits were added by the post-reconstruction analysis and are not
    # present in the immutable committed reference package. Compare them when
    # two extended packages are supplied, without making the frozen comparison
    # fail merely because the historical JSON predates those derived fields.
    for optional in OPTIONAL_PRIMARY_AUDITS:
        if optional in payload:
            selected[optional] = payload[optional]
    return selected


def _aligned_primary_science(
    reference: Mapping[str, object], reconstructed: Mapping[str, object]
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Select the same deterministic JSON scope on both package versions.

    The immutable committed package predates a small number of derived audit
    fields.  Such a field is compared when both packages contain it and is
    omitted from both sides otherwise.  Core confirmatory results, privacy,
    trajectory counts, calls, and token accounting are never optional.
    """

    left = _selected_primary_science(reference)
    right = _selected_primary_science(reconstructed)
    for name in OPTIONAL_PRIMARY_AUDITS:
        if name not in reference or name not in reconstructed:
            left.pop(name, None)
            right.pop(name, None)
    return left, right


def _compare_json_values(
    reference: object,
    reconstructed: object,
    tolerance: float,
    location: str = "$",
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> list:
    """Return deterministic mismatch records for nested JSON-compatible data."""

    if isinstance(reference, Mapping) and isinstance(reconstructed, Mapping):
        if set(reference) != set(reconstructed):
            return [
                {
                    "location": location,
                    "reference_keys": sorted(str(key) for key in reference),
                    "reconstructed_keys": sorted(str(key) for key in reconstructed),
                }
            ]
        mismatches = []
        for key in sorted(reference, key=str):
            mismatches.extend(
                _compare_json_values(
                    reference[key],
                    reconstructed[key],
                    tolerance,
                    "%s.%s" % (location, key),
                    relative_tolerance,
                )
            )
            if len(mismatches) >= 20:
                break
        return mismatches[:20]
    if isinstance(reference, list) and isinstance(reconstructed, list):
        if len(reference) != len(reconstructed):
            return [
                {
                    "location": location,
                    "reference_length": len(reference),
                    "reconstructed_length": len(reconstructed),
                }
            ]
        mismatches = []
        for index, (left, right) in enumerate(zip(reference, reconstructed)):
            mismatches.extend(
                _compare_json_values(
                    left,
                    right,
                    tolerance,
                    "%s[%d]" % (location, index),
                    relative_tolerance,
                )
            )
            if len(mismatches) >= 20:
                break
        return mismatches[:20]
    numeric = (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(reconstructed, (int, float))
        and not isinstance(reconstructed, bool)
    )
    equal = (
        bool(
            np.isclose(
                float(reference),
                float(reconstructed),
                atol=tolerance,
                rtol=relative_tolerance,
            )
        )
        if numeric
        else reference == reconstructed
    )
    return (
        []
        if equal
        else [
            {
                "location": location,
                "reference": reference,
                "reconstructed": reconstructed,
            }
        ]
    )


def _normalized(frame: pd.DataFrame, excluded: Sequence[str]) -> pd.DataFrame:
    output = frame.drop(columns=list(excluded), errors="ignore").copy()
    columns = sorted(output.columns)
    output = output[columns]
    sort_columns = [
        name
        for name in (
            "model_key",
            "cluster_id",
            "panel_id",
            "condition",
            "window_sweeps",
            "sweep",
            "block_length",
            "pseudocount",
            "hypothesis",
        )
        if name in output.columns
    ]
    if sort_columns:
        output = output.sort_values(sort_columns, kind="mergesort")
    return output.reset_index(drop=True)


def compare_frames(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    excluded: Sequence[str],
    tolerance: float,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> Dict[str, object]:
    left = _normalized(reference, excluded)
    right = _normalized(reconstructed, excluded)
    result: Dict[str, object] = {
        "reference_rows": len(left),
        "reconstructed_rows": len(right),
        "reference_columns": list(left.columns),
        "reconstructed_columns": list(right.columns),
        "excluded_columns": list(excluded),
    }
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        result.update({"status": "shape_or_schema_mismatch", "mismatch_cells": None})
        return result
    mismatches = []
    maximum_absolute_difference = 0.0
    for column in left.columns:
        if column in JSON_NUMERIC_COLUMNS:
            lvalue = left[column].fillna("null").astype(str).to_numpy()
            rvalue = right[column].fillna("null").astype(str).to_numpy()
            equal = np.ones(len(lvalue), dtype=bool)
            for index, (left_json, right_json) in enumerate(zip(lvalue, rvalue)):
                try:
                    left_value = json.loads(left_json)
                    right_value = json.loads(right_json)
                except json.JSONDecodeError:
                    equal[index] = left_json == right_json
                    continue
                equal[index] = not bool(
                    _compare_json_values(
                        left_value,
                        right_value,
                        tolerance,
                        "$",
                        relative_tolerance,
                    )
                )
        elif pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            lvalue = left[column].to_numpy(float)
            rvalue = right[column].to_numpy(float)
            equal = np.isclose(
                lvalue,
                rvalue,
                atol=float(tolerance),
                rtol=float(relative_tolerance),
                equal_nan=True,
            )
            finite = np.isfinite(lvalue) & np.isfinite(rvalue)
            if np.any(finite):
                maximum_absolute_difference = max(
                    maximum_absolute_difference,
                    float(np.max(np.abs(lvalue[finite] - rvalue[finite]))),
                )
        else:
            lvalue = left[column].fillna("<NA>").astype(str).to_numpy()
            rvalue = right[column].fillna("<NA>").astype(str).to_numpy()
            equal = lvalue == rvalue
        for index in np.flatnonzero(~equal)[:20]:
            mismatches.append(
                {
                    "row": int(index),
                    "column": column,
                    "reference": str(left.iloc[index][column]),
                    "reconstructed": str(right.iloc[index][column]),
                }
            )
        if len(mismatches) >= 20:
            break
    result.update(
        {
            "status": "matched" if not mismatches else "value_mismatch",
            "mismatch_cells_preview": mismatches,
            "maximum_absolute_numeric_difference": maximum_absolute_difference,
        }
    )
    return result


def _write_json(value: object, destination: Path) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
    )
    args = parser.parse_args()
    comparisons = {}
    for name, excluded in SCIENCE_TABLES.items():
        reference_path = args.reference / "tables" / name
        reconstructed_path = args.reconstructed / "tables" / name
        if not reference_path.is_file() or not reconstructed_path.is_file():
            comparisons[name] = {
                "status": "missing",
                "reference_exists": reference_path.is_file(),
                "reconstructed_exists": reconstructed_path.is_file(),
            }
            continue
        comparisons[name] = compare_frames(
            pd.read_csv(reference_path),
            pd.read_csv(reconstructed_path),
            excluded,
            args.tolerance,
            args.relative_tolerance,
        )
    primary_name = "statistics/primary_results.json"
    reference_primary_path = args.reference / primary_name
    reconstructed_primary_path = args.reconstructed / primary_name
    if not reference_primary_path.is_file() or not reconstructed_primary_path.is_file():
        comparisons[primary_name] = {
            "status": "missing",
            "reference_exists": reference_primary_path.is_file(),
            "reconstructed_exists": reconstructed_primary_path.is_file(),
        }
    else:
        reference_payload = json.loads(
            reference_primary_path.read_text(encoding="utf-8")
        )
        reconstructed_payload = json.loads(
            reconstructed_primary_path.read_text(encoding="utf-8")
        )
        reference_primary, reconstructed_primary = _aligned_primary_science(
            reference_payload, reconstructed_payload
        )
        primary_mismatches = _compare_json_values(
            reference_primary,
            reconstructed_primary,
            args.tolerance,
            "$",
            args.relative_tolerance,
        )
        comparisons[primary_name] = {
            "status": "matched" if not primary_mismatches else "value_mismatch",
            "scope": (
                "confirmatory dispositions, privacy/memory audits, and exact "
                "non-latency formal calls/tokens/accounting"
            ),
            "mismatch_values_preview": primary_mismatches,
        }
    status = "passed" if all(value["status"] == "matched" for value in comparisons.values()) else "failed"
    output = {
        "status": status,
        "absolute_tolerance": args.tolerance,
        "relative_tolerance": args.relative_tolerance,
        "tolerance_interpretation": (
            "binary64 roundoff allowance only; not a scientific effect-size tolerance"
        ),
        "scope": "deterministic scientific and non-latency accounting columns",
        "comparisons": comparisons,
    }
    _write_json(output, args.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
