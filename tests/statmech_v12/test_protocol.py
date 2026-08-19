from pathlib import Path

from thermoagent.statmech_llm_v12.experiment import expected_decisions, formal_panel_design
from thermoagent.statmech_llm_v12.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_formal_decision_count_and_matched_alpha_arms():
    protocol = load_yaml(ROOT / "configs/statmech_v12/protocol_template.yaml")
    assert expected_decisions(protocol) == protocol["compute"]["expected_primary_decisions"]
    panels = formal_panel_design(protocol)
    collective = [row for row in panels if row["family"] == "collective_network"]
    for cluster in sorted({row["cluster_id"] for row in collective}):
        arms = [row for row in collective if row["cluster_id"] == cluster]
        assert len(arms) == 7
        assert sum(row["alpha"] == 0.0 for row in arms) == 1
        assert {row["orientation"] for row in arms if row["alpha"] > 0.0} == {"forward", "transpose"}
    factors = {
        (row["coupling_strength"], row["sampling_temperature"])
        for row in collective
    }
    assert factors == {(0.35, 0.50), (0.35, 0.85), (0.80, 0.50), (0.80, 0.85)}
    memory = [row for row in panels if row["family"] == "persistent_memory"]
    assert {row["regime"] for row in memory} == {"markovized", "persistent_memory"}


def test_v12_raw_artifact_patterns_are_ignored():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/llm_agent_statmech_v12/raw/" in text
    assert "paper/jstat_v12/*.aux" in text
