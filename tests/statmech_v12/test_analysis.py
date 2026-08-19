import pandas as pd

from thermoagent.statmech_llm_v12.analysis import _micro_models, _panel_statistics
from thermoagent.statmech_llm_v12.experiment import _graph_for_panel, formal_panel_design
from thermoagent.statmech_llm_v12.provider import KineticIsingProvider
from thermoagent.statmech_llm_v12.simulation import run_trajectory
from thermoagent.statmech_llm_v12.workflow import load_yaml


def test_panel_analysis_uses_attempted_update_units_and_bias_floors():
    protocol = load_yaml(__import__("pathlib").Path(__file__).resolve().parents[2] / "configs/statmech_v12/protocol_template.yaml")
    protocol["analysis"]["time_shuffle_replicates_per_panel"] = 5
    protocol["analysis"]["information_permutation_replicates_per_panel"] = 5
    definition = next(
        row
        for row in formal_panel_design(protocol)
        if row["family"] == "small_network" and row["n_agents"] == 4 and row["alpha"] == 0.8
    )
    graph = _graph_for_panel(definition)
    rows = run_trajectory(
        KineticIsingProvider(),
        graph,
        120,
        20,
        "markovized",
        0.7,
        0.72,
        "disordered",
        metadata={"panel_id": definition["panel_id"], "cluster_id": definition["cluster_id"]},
    )
    definition = dict(definition)
    definition["burn_in_sweeps"] = 2
    summary, currents = _panel_statistics(pd.DataFrame(rows), protocol, definition)
    assert summary["attempted_updates"] == 80
    assert summary["retained_updates"] == 72
    assert summary["markov_epr_nats_per_sweep"] == 4 * summary["markov_epr_nats_per_update"]
    assert "belief_layer_adjusted_block_kl_nats_per_update" in summary
    assert "directed_edge_transfer_entropy_permutation_floor" in summary
    assert summary["privacy_mutations"] == 0
    assert isinstance(currents, list)


def test_microscopic_models_cover_belief_and_action_response_families():
    rows = []
    for cell in range(48):
        for neighbor in (-1, 1):
            rows.append(
                {
                    "valid_after_repair": 1,
                    "belief_after": 1 if (cell + neighbor) % 3 else -1,
                    "action_after": 1 if (cell - neighbor) % 4 else -1,
                    "private_field": [-1, 0, 1][cell % 3],
                    "neighbor_field": neighbor,
                    "coupling_strength": [0.35, 0.8][cell % 2],
                    "current_belief": [-1, 1][cell % 2],
                    "current_action": [-1, 1][(cell // 2) % 2],
                    "sampling_temperature": [0.5, 0.85][(cell // 3) % 2],
                    "regime": ["markovized", "persistent_memory"][cell % 2],
                    "amber_first": cell % 2,
                    "latent_plus_label": ["amber", "cobalt"][cell % 2],
                    "paraphrase": (cell // 2) % 2,
                    "information_state_id": "state_%d_%d" % (cell, neighbor),
                    "replicate": cell % 2,
                    "current_belief": [-1, 1][cell % 2],
                }
            )
    models, _ = _micro_models(pd.DataFrame(rows), 100, 17)
    assert {(row["response_layer"], row["model"]) for row in models} == {
        (layer, model)
        for layer in ("belief", "action")
        for model in ("kinetic_logistic", "nonlinear_additive", "persistence_interaction")
    }
