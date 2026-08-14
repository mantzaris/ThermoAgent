# ThermoHITL simulated operator and dashboard

Status: implemented and tested.

The dashboard is an execution-time view over a schema-validated operator
payload, not a post-hoc illustration. Allocation and operator policies consume
the same serialized payload that the dashboard renders. Every payload receives
a deterministic SHA-256 digest in the event ledger.

Normal view schemas are `local_kpi`, `entropy_only`, `energy_only`,
`thermodynamic`, and `thermodynamic_plus_disagreement`. A separate
`evaluator_oracle` schema is privileged and unattainable. Tests will fail if a
normal payload includes raw private state, future disruptions, RNG state,
counterfactual outcomes, or evaluator-only global values.

The GitHub-facing implementation will use a dependency-light local web server
and SVG/HTML replay client unless the existing environment already provides a
compatible dashboard framework. This keeps replay mode GPU-free and reproducible
from a fresh clone. Live simulation and deterministic ledger replay will share
the same view builder.

No real human participants are part of v3. A future-study protocol, trial
schema, randomization hook, action/response-time logging, questionnaire
integration points, and IRB boundary will be prepared without fabricated human
evidence.

## Delivered implementation

`thermoagent/dashboard/` provides a dependency-light web app, deterministic
GPU-free replay, live mock mode, play/pause/step/rewind/jump-to-alert controls,
matched-branch inspection, and SVG/JSON export. Its network, thermodynamic,
phase-plane, alert, intervention, workload, explanation, provenance, and replay
panels render the same `thermohitl-dashboard-frame-v1` payload consumed by the
simulated operator.

Tests reject private/evaluator fields in normal views, verify payload hashes,
check deterministic frame digests, and require vector SVG text/shapes. The
future-human protocol and schemas are technical preparation only.
