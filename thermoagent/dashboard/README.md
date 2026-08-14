# ThermoHITL operator dashboard

The dashboard is a dependency-light local web application. Replay mode reads a
v3 `episode.json` and adjacent compressed event ledger; it does not load a
model or require a GPU. Live mode runs a deterministic small mock-planner
scenario and serves the same view/replay interface.

```bash
./scripts/run-human-operator-dashboard.sh --episode \
  results/human_operator_v3/raw/development/<run-id>/episode.json

./scripts/run-human-operator-dashboard.sh --live
```

Open `http://127.0.0.1:8765`. The application supports play, pause, step,
rewind, jump-to-alert, and SVG/JSON export. Application, scenario, seed, method,
operator profile, and information condition are selected by choosing the
corresponding episode artifact; no state from another episode is silently
combined.

## Data dictionary

- `network.nodes`: public identity/location, coarse shared thermodynamic bands,
  and current autonomy level.
- `physical_edges`: currently authorized material edges in the operator view.
- `communication_edges`: currently available explicit communication links.
- `authorized_emergency_edges`: bounded operator-authorized route edges.
- `thermodynamics`: distributed energy/entropy, anomaly/slope, diagnostic free
  energy, disagreement, consensus confidence, and service loss.
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

V3 evaluates simulated operators only. The interface is technical preparation
for a future IRB-approved study. It does not establish human usability, trust,
fatigue, cognitive workload, or safety. See
`results/human_operator_v3/protocol/future_human_study_protocol.md` for
the proposed future-study separation.
