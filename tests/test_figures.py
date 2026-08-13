from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thermoagent.figures import (
    _fixed_deployable_benchmark,
    _formed_coalition_members,
    validate_pdfs,
)


def test_pdf_validation_opens_detects_fonts_and_renders(tmp_path: Path):
    pdf_dir = tmp_path / "figures" / "pdf"
    pdf_dir.mkdir(parents=True)
    figure, axis = plt.subplots(figsize=(2.0, 1.5))
    axis.plot([0, 1], [0, 1])
    axis.set_title("QA fixture")
    figure.savefig(pdf_dir / "fixture.pdf")
    plt.close(figure)

    report = validate_pdfs(tmp_path)
    assert len(report["figures"]) == 1
    record = report["figures"][0]
    assert record["opens"] and record["fonts_detected"]
    assert (tmp_path / record["rendered"]).stat().st_size > 1000


def test_necessity_benchmark_is_one_fixed_method_per_cell_not_seed_oracle():
    frame = pd.DataFrame([
        {"scenario_name": "factor", "seed": 1, "n_agents": 11,
         "method": "centralized_llm", "primary_outcome": 0.0},
        {"scenario_name": "factor", "seed": 2, "n_agents": 11,
         "method": "centralized_llm", "primary_outcome": 10.0},
        {"scenario_name": "factor", "seed": 1, "n_agents": 11,
         "method": "scripted_independent", "primary_outcome": 10.0},
        {"scenario_name": "factor", "seed": 2, "n_agents": 11,
         "method": "scripted_independent", "primary_outcome": 0.0},
    ])
    selected = _fixed_deployable_benchmark(frame)
    assert selected["method"].nunique() == 1
    assert len(selected) == 2
    assert selected["benchmark"].mean() == 5.0


def test_network_figure_does_not_draw_unilateral_coalition_proposal():
    proposal = {
        "kind": "coalition_event", "step": 2, "actor": "a",
        "payload": {
            "action": "propose", "coalition_id": "K1", "proposer": "a"
        },
    }
    assert _formed_coalition_members([proposal], 2) == (None, set())
    join = {
        "kind": "coalition_event", "step": 3, "actor": "b",
        "payload": {"action": "join_coalition", "coalition_id": "K1", "ok": True},
    }
    coalition_id, members = _formed_coalition_members([proposal, join], 3)
    assert coalition_id == "K1"
    assert members == {"a", "b"}
