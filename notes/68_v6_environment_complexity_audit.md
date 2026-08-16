# V6 environment complexity audit for V7 design

Recorded on 2026-08-16 before V7 environment implementation or V7 outcome
generation. V6 remains immutable at
`8013300c23553928a0269e6be27f5baaedee7e53`.

## Verified structural facts

| Property | V6 implementation | Evidence |
|---|---|---|
| Horizon | 12 simulator steps | `V6PanelEnvironment.__init__(horizon=12)` |
| Decision epochs | steps 0, 2, 4, 6, 8, 10 (six epochs) | `V6PanelEnvironment.decision_steps` |
| Concurrent incidents | four | default `incident_count=4`; formal summaries |
| Agents | three newly created agents per incident, twelve total | `_make_agents`; `incident_count * 3` stochastic tape |
| Agent scope | every identity has `incident_scope=(incident_id,)` | `_make_agents` |
| Persistent scope | persistent during one episode, but never responsible for another incident or shared asset | agent identity and incident-indexed vault access |
| Beliefs per incident-level entropy estimate | one recipient belief plus at most two same-incident peer beliefs | `exchange_sketches` and `information_state` |
| Disruption onset | fixed at step 2 for every nonnominal incident | `_make_incidents(disruption_step=2)` |
| Topology labels | five seed-modulo labels per application | `topology_family = application + seed % 5` |
| Actual graph structure | three-agent directed clique per incident plus a bidirectional ring among incident leads | `_make_communication_edges` |
| Structural topology variation | none beyond reliability weights and regime-dependent edge availability | topology label is not passed into graph construction |
| Cross-incident ring use | not used by sketch exchange, proposal exchange, peer evidence, pooling, or action coordination | each path iterates `incident_agents[incident_id]` only |
| Domain mechanics | one common incident, resource, action-delay, direct-effect, backlog, and service-deficit transition | `V6PanelEnvironment` serves all three labels |
| Application differences | role names/permissions, utility draws, and a utility-specific integrity-mode prevalence adjustment | `APP_ROLES`, registry permissions, `_make_incidents` |
| Correct-action mapping | shared `PRIMARY_ACTION_FOR_MODE` and `SECONDARY_ACTION_FOR_MODE` | `v6_types.py` |
| Action consequence | correct/secondary/incorrect classification gives an immediate scalar deficit change; only the same incident is updated | `preview_direct_effect` and `_complete_pending` |
| Multi-stage causal coupling | action completion is delayed, but there is no resource transfer to another incident, network-flow propagation, or later cross-incident feasibility change | pending-action and service-transition code |

## Why KPI opportunity dominated

The deployable KPI vector is constructed from the same incident severity,
backlog, service deficit, shared-resource depletion, and belief components that
drive action value. Although no single KPI reveals the evaluator action label,
these variables closely track the one-incident harm function. In
private-fragmented development, the frozen KPI-confidence model achieved ROC
AUC/AP of 0.851/0.843 commercial, 0.861/0.857 humanitarian, and 0.888/0.909
utility restoration. Entropic information therefore had only residual room to
improve ranking.

V6's matched dynamic harm reductions were directionally positive but small:
0.02258 humanitarian and 0.01224 utility restoration, both below the frozen
0.03 practical threshold. The private-minus-public interactions were 0.01908
and 0.00916 and did not replicate. These results are consistent with a system
where three local beliefs sometimes add information, but shared flows,
contention, topology propagation, and long-range consequences do not exist.

## V7 requirements derived before implementation

V7 must not increase complexity by copying independent incidents. It requires
application-specific state transitions; persistent agents controlling multiple
assets; shared inventories, crews, vehicles, fuel, or spares; topology-dependent
flows; actual cross-community messages; delayed multi-step causal chains;
cascades; reconnection; and size/coupling/fragmentation as frozen experimental
factors. Entropy must remain observational: action outcomes may depend on domain
state and actions, never on an entropy value.

This audit is design evidence only. V6 is not pooled into V7 inference.
