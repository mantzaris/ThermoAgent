"""Publication-equivalence and repository-integrity validation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd


EXPECTED_PROTOCOL_SHA256 = (
    "863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9"
)
EXPECTED_MODELS = {
    "qwen": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "granite": (
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    ),
}
FORBIDDEN_TRACKED_SUFFIXES = {
    ".bin",
    ".jsonl",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".safetensors",
    ".tar",
    ".xz",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid SHA-256 entry: %s" % line)
        if relative in values:
            raise ValueError("duplicate publication path: %s" % relative)
        values[relative] = digest
    return values


def displayed_equations(source: str) -> List[Dict[str, object]]:
    """Extract displayed equation blocks exactly as frozen at consolidation."""

    pattern = re.compile(
        r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", re.DOTALL
    )
    rows: List[Dict[str, object]] = []
    for index, match in enumerate(pattern.finditer(source), start=1):
        block = match.group(0)
        rows.append(
            {
                "index": index,
                "environment": match.group(1),
                "labels": re.findall(r"\\label\{([^}]+)\}", match.group(2)),
                "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def _tracked_paths(repository: Path) -> Iterable[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repository,
        text=True,
    )
    return (
        line
        for line in output.splitlines()
        if line and (repository / line).is_file()
    )


def _pdf_text_sha256(path: Path) -> str:
    """Hash the text extracted by Poppler, independent of PDF metadata."""

    extracted = subprocess.check_output(["pdftotext", str(path), "-"])
    return hashlib.sha256(extracted).hexdigest()


def _validate_figure_catalog(
    catalog: pd.DataFrame,
    figures: List[Path],
    source_tables: List[Path],
    paper: Path,
    result: Path,
) -> Dict[str, object]:
    """Check that each canonical figure has one source table and frozen hash."""

    required = {
        "filename",
        "source_table",
        "pdf_sha256",
        "source_sha256",
    }
    errors: List[str] = []
    if not required.issubset(catalog.columns):
        return {"passed": False, "errors": ["missing required catalog columns"]}
    if catalog["filename"].duplicated().any():
        errors.append("duplicate figure filename")
    expected_figures = {path.name for path in figures}
    if set(catalog["filename"]) != expected_figures:
        errors.append("catalog figure set differs from canonical PDF set")
    expected_sources = {
        "source_data/%s.csv" % path.stem for path in figures
    }
    if set(catalog["source_table"]) != expected_sources:
        errors.append("catalog source-table set differs from canonical source set")
    if {path.name for path in source_tables} != {
        Path(path).name for path in expected_sources
    }:
        errors.append("canonical source-data directory differs from catalog")
    for row in catalog.to_dict(orient="records"):
        pdf_path = paper / "figures" / str(row["filename"])
        source_path = result / str(row["source_table"])
        if not pdf_path.is_file() or sha256_file(pdf_path) != row["pdf_sha256"]:
            errors.append("PDF hash mismatch: %s" % row["filename"])
        if (
            not source_path.is_file()
            or sha256_file(source_path) != row["source_sha256"]
        ):
            errors.append("source hash mismatch: %s" % row["source_table"])
    return {"passed": not errors, "errors": errors}


def _validate_reconstruction_records(result: Path) -> Dict[str, object]:
    """Validate the retained replay and reference-comparison dispositions."""

    reproducibility = result / "stages/cross_model/reproducibility"
    replay = json.loads(
        (reproducibility / "replay_summary.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (reproducibility / "reconstructed_vs_committed.json").read_text(
            encoding="utf-8"
        )
    )
    comparison_statuses = {
        name: record.get("status")
        for name, record in comparison.get("comparisons", {}).items()
    }
    checks = {
        "replay_status": replay.get("status") == "passed",
        "replay_trajectories": replay.get("units_checked") == 48,
        "replay_decisions": replay.get("rows_checked") == 34560,
        "replay_zero_mismatches": replay.get("units_with_mismatches") == 0
        and replay.get("mismatch_details") == [],
        "comparison_status": comparison.get("status") == "passed",
        "comparison_tables": bool(comparison_statuses)
        and set(comparison_statuses.values()) == {"matched"},
    }
    return {"passed": all(checks.values()), "checks": checks}


def validate_publication(repository: Path) -> Dict[str, object]:
    """Validate immutable scientific content after semantic path consolidation."""

    repository = Path(repository).resolve()
    result = repository / "results/JSTAT"
    paper = repository / "paper/JSTAT"
    reference = json.loads(
        (result / "reproducibility/publication_reference.json").read_text(
            encoding="utf-8"
        )
    )
    expected_hashes = _hash_manifest(
        result / "reproducibility/publication_hashes.sha256"
    )
    hash_mismatches = {
        relative: {
            "expected": expected,
            "observed": sha256_file(repository / relative)
            if (repository / relative).is_file()
            else None,
        }
        for relative, expected in expected_hashes.items()
        if not (repository / relative).is_file()
        or sha256_file(repository / relative) != expected
    }

    observed_equations = displayed_equations(
        (paper / "main.tex").read_text(encoding="utf-8")
    )
    expected_equations = reference["displayed_equations"]
    equation_match = observed_equations == expected_equations

    primary = json.loads(
        (
            result
            / "stages/cross_model/statistics/primary_results.json"
        ).read_text(encoding="utf-8")
    )["confirmatory_dispositions"]
    primary_match = all(
        all(primary[hypothesis][key] == value for key, value in expected.items())
        for hypothesis, expected in reference["primary_results"].items()
    )

    protocol_path = repository / "configs/statmech_llm/cross_model/protocol.yaml"
    protocol_hash_match = sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256
    protocol_text = protocol_path.read_text(encoding="utf-8")
    model_identity_match = all(
        identifier in protocol_text and revision in protocol_text
        for identifier, revision in EXPECTED_MODELS.values()
    )

    figures = sorted((paper / "figures").glob("*.pdf"))
    source_tables = sorted((result / "source_data").glob("*.csv"))
    catalog = pd.read_csv(result / "figure_catalog.csv")
    catalog_validation = _validate_figure_catalog(
        catalog, figures, source_tables, paper, result
    )
    reconstruction_validation = _validate_reconstruction_records(result)
    main_source = (paper / "main.tex").read_text(encoding="utf-8")
    figure_reference_match = all(
        main_source.count("\\resultfigure{%s}" % path.name) == 1
        for path in figures
    )

    tracked = tuple(_tracked_paths(repository))
    forbidden_tracked = [
        path
        for path in tracked
        if Path(path).suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
        and not path.startswith("paper/JSTAT/")
    ]
    obsolete_path_pattern = re.compile(
        r"(^|/)(?:[^/]*_v(?:4|5|6|7|8|9|10|11|12|13|14|15))(?:/|$)"
    )
    obsolete_tracked_paths = [path for path in tracked if obsolete_path_pattern.search(path)]

    checks = {
        "immutable_hashes": not hash_mismatches,
        "displayed_equations": equation_match,
        "primary_results": primary_match,
        "protocol_hash": protocol_hash_match,
        "model_identities": model_identity_match,
        "publication_figure_count": len(figures) == 14,
        "publication_source_count": len(source_tables) == 14,
        "figure_catalog_count": len(catalog) == 14,
        "figure_catalog_hashes": bool(catalog_validation["passed"]),
        "figure_references_exactly_once": figure_reference_match,
        "manuscript_extracted_text": _pdf_text_sha256(paper / "main.pdf")
        == reference["main_pdf_extracted_text_sha256"],
        "retained_replay_and_reconstruction": bool(
            reconstruction_validation["passed"]
        ),
        "forbidden_tracked_artifacts_absent": not forbidden_tracked,
        "obsolete_version_paths_absent": not obsolete_tracked_paths,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "hash_entries": len(expected_hashes),
        "hash_mismatches": hash_mismatches,
        "equation_count": len(observed_equations),
        "figure_count": len(figures),
        "source_data_count": len(source_tables),
        "figure_catalog_errors": catalog_validation["errors"],
        "reconstruction_checks": reconstruction_validation["checks"],
        "forbidden_tracked_artifacts": forbidden_tracked,
        "obsolete_version_paths": obsolete_tracked_paths,
    }


__all__ = ["displayed_equations", "sha256_file", "validate_publication"]
