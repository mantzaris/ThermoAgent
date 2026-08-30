from pathlib import Path

from thermoagent.statmech_llm.repository_audit import (
    FINAL_SCRIPT_ROLES,
    figure_inventory,
    script_inventory,
)
from thermoagent.statmech_llm.validation import (
    _validate_reconstruction_records,
    sha256_file,
)

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def test_figure_inventory_classifies_every_retained_stage_artifact():
    inventory = figure_inventory(ROOT)
    expected = sum(
        1
        for path in (ROOT / "results/JSTAT/stages").glob("*/figures/**/*")
        if path.is_file()
    )
    assert len(inventory) == expected == 153
    assert not inventory["classification"].str.startswith("5 ").any()
    assert (inventory["classification"].str.startswith("1 ")).sum() == 14
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_script_inventory_uses_positive_final_closure():
    inventory = script_inventory(ROOT)
    retained = inventory[inventory["disposition"] == "retain"]
    assert set(Path(path).name for path in retained["current_path"]) == set(
        FINAL_SCRIPT_ROLES
    )
    candidates = inventory[inventory["disposition"] != "retain"]
    assert not (candidates["evidence_deletion_safe"] == "").any()


def test_publication_figure_catalog_is_a_complete_hash_bridge():
    catalog = pd.read_csv(ROOT / "results/JSTAT/figure_catalog.csv")
    figures = sorted((ROOT / "paper/JSTAT/figures").glob("*.pdf"))
    sources = sorted((ROOT / "results/JSTAT/source_data").glob("*.csv"))
    assert len(catalog) == len(figures) == len(sources) == 14
    assert set(catalog["filename"]) == {path.name for path in figures}
    assert set(catalog["source_table"]) == {
        "source_data/%s.csv" % path.stem for path in figures
    }
    for row in catalog.to_dict(orient="records"):
        assert sha256_file(ROOT / "paper/JSTAT/figures" / row["filename"]) == row[
            "pdf_sha256"
        ]
        assert sha256_file(ROOT / "results/JSTAT" / row["source_table"]) == row[
            "source_sha256"
        ]


def test_retained_replay_and_reconstruction_records_are_complete():
    result = _validate_reconstruction_records(ROOT / "results/JSTAT")
    assert result["passed"] is True
    assert all(result["checks"].values())
