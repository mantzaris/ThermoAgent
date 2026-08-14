# Dashboard data dictionary

The replay and live interfaces consume `thermohitl-dashboard-frame-v1` records.
All timestamps are simulator periods unless a field explicitly says seconds or
minutes.

| Field | Type | Meaning | Boundary |
|---|---|---|---|
| `step` | integer | Current simulator period | public |
| `network.nodes` | array | Public ID, role, location, coarse monitor state, autonomy level | authorized shared |
| `network.physical_edges` | pairs | Currently feasible logistics links | authorized shared |
| `network.communication_edges` | pairs | Links available for explicit messages/sketches | public network state |
| `network.authorized_emergency_edges` | pairs | Temporarily authorized operator routes | operator directive |
| `thermodynamics.distributed_energy` | number | Gossip-derived normalized operational severity | distributed estimate |
| `thermodynamics.distributed_entropy` | number | Gossip-derived normalized operational entropy | distributed estimate |
| `thermodynamics.entropy_anomaly` | number | Two-sided nominal standardized residual | distributed estimate |
| `thermodynamics.entropy_slope` | number | One-period entropy difference | distributed estimate |
| `thermodynamics.free_energy_diagnostic` | number | `E - T*S`; diagnostic only | distributed estimate |
| `thermodynamics.agent_disagreement` | number | Bounded Jensen–Shannon disagreement | distributed estimate |
| `thermodynamics.consensus_confidence` | number | Support/confidence of the gossip estimate | distributed estimate |
| `thermodynamics.service_loss` | number | Visible service-loss indicator to date | authorized KPI |
| `alert_queue` | array | Bounded incident/priority/benefit/time records | authorized operator view |
| `interventions` | array | Typed operator action and deterministic result | event ledger |
| `workload` | object | Slots, queue, workload, fatigue, latency, minutes | simulated-operator state |
| `explanation` | object | Feature attribution, contributors, provenance, timestamp | authorized operator view |
| `material_progress` | object | Counts at accepted/transit/arrival/demand stages | public execution result |
| `view_hashes` | array | SHA-256 hashes of exact operator payloads | audit metadata |

Never-present fields in normal views include raw private costs, exact private
inventory/capacity, agent RNG state, future disruption labels, evaluator-global
thermodynamic state, and counterfactual outcomes. Oracle-only fields use a
separate explicitly privileged schema.
