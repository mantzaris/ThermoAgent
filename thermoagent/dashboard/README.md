# ThermoHITL operator dashboard

The dashboard is a dependency-light local web application. Replay mode reads a
v3, v4, v5, or v6 `episode.json` and adjacent compressed event ledger; it does not load a
model or require a GPU. V4 replay detects `operator_view_v4` events and refuses
to reconstruct a richer display from evaluator time-series fields. Live mode
runs a deterministic small mock-planner scenario and serves the same interface.

```bash
./scripts/run-human-operator-dashboard.sh --episode \
  results/human_operator_v3/raw/development/<run-id>/episode.json

./scripts/run-human-operator-dashboard.sh --live

./scripts/run-human-operator-v4-dashboard.sh --episode \
  results/human_operator_v4/raw/development_gate_trigger/<run-id>/episode.json

./scripts/run-human-operator-v5-dashboard.sh --episode \
  results/human_operator_v5/raw/development_primary_v2/<run-id>/episode.json

./scripts/run-v6-dashboard.sh --episode \
  results/generalized_entropic_consensus_v6/raw/development_dynamic/<run-id>/episode.json.gz
```

Open `http://127.0.0.1:8765`. The application supports play, pause, step,
rewind, jump-to-alert, and SVG/JSON export. Application, scenario, seed, method,
operator profile, and information condition are selected by choosing the
corresponding episode artifact; no state from another episode is silently
combined.

## V4 populated exports

`./scripts/generate-human-operator-v4-figures.sh` produces actual populated
commercial, humanitarian, and utility replay SVG/PDF exports under
`results/human_operator_v4/dashboard_exports/`. They are generated from the
functional replay frames and retain source ledger/view hashes; they are not
hand-drawn dashboard mockups.

## V5 populated exports

`./scripts/generate-human-operator-v5-figures.sh` produces actual populated
commercial, humanitarian, and utility replay SVG/PDF/PNG exports under
`results/human_operator_v5/dashboard_exports/`. V5 ranks four simultaneous
incidents, exposes only the hashed deployable KPI/entropy/disagreement payload,
and excludes evaluator counterfactual effects and true incident modes.

## V6 selective-autonomy replay

V6 adds the proposed autonomous action, action-value margin, Shannon and
Tsallis uncertainty, Gini-Simpson impurity, pooled uncertainty, Jensen-Shannon
and graph disagreement, consensus residual, missing contributors, delegation
recommendation, operator queue, and communication provenance. The ordinary
frame remains exactly the hashed authorized simulated-operator payload.

The **Evaluator analysis** control is a separate, red-labeled research view.
It reads only `private_to="evaluator"` matched counterfactual ledger events and
shows loss with/without an action under the common stochastic tape. It is never
loaded by default, never enters an operator payload, and is unavailable to
agents, learned policies, and the simulated operator.

## Data dictionary

- `network.nodes`: public identity/location and current autonomy level. V4 does
  not expose private node-level thermodynamic state.
- `physical_edges`: currently authorized material edges in the operator view.
- `communication_edges`: currently available explicit communication links.
- `authorized_emergency_edges`: bounded operator-authorized route edges.
- `thermodynamics`: only fields present in the latest authorized operator
  payload: energy/entropy, anomaly/slope, diagnostic free energy, disagreement,
  consensus confidence, and authorized service KPI.
- `alert_queue`: incident IDs and queue provenance; raw private state is absent.
- `interventions`: recent bounded operator action/result events.
- `workload`: simulated workload, fatigue, queue length, active slots, and
  accumulated operator minutes.
- `view_hashes`: SHA-256 digests of the exact serialized payloads consumed by
  the simulated operator.

## Information boundary

The dashboard renders the same schema-validated payload used by the tested
operator policy. Normal views cannot contain raw private observations,
inventory, cost, RNG state, future disruptions, evaluator-global entropy or
energy, or counterfactual outcomes. Only the explicitly labeled oracle view may
contain privileged fields. Tests validate both field redaction and payload
hashes.

## Human-study boundary

V3, v4, v5, and v6 evaluate simulated operators only. The interface is technical
preparation for a future IRB-approved study. It does not establish human
usability, trust, fatigue, cognitive workload, or safety. See
`results/human_operator_v5/protocol/future_human_study_protocol.md` and
`future_human_trial_schema.json` for the current technical preparation.
