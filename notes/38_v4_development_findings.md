# ThermoHITL v4 development findings

This note is append-only with respect to development attempts.  It does not
contain validation or holdout evidence.

## Retained implementation pilots

1. `implementation_pilot_v1_coordination` produced 51 complete episodes and
   three retained failures (`KeyError: communications_zone`).  Cause: a
   multi-scope agent could be selected for an incident while its current
   private observation referred to a different incident.  Candidate selection
   was restricted to agents whose delivered observation names the incident.
2. `implementation_pilot_v2_coordination` completed 90 episodes but showed
   mostly null or adverse communication effects.  Cause: peer messages were
   logged but did not alter the responding agent's recognition state.
   Connected-component peer beliefs were incorporated into bounded response
   selection.  Communication remains imperfect (0.55 development success
   probability), not an oracle.
3. `implementation_pilot_v3_coordination` was generated before the final
   0.55 communication probability and is retained only as an implementation
   diagnostic.  It is ineligible as formal gate evidence.
4. `implementation_pilot_v1_human` completed 90 episodes.  Humanitarian probes
   changed outcomes, but commercial and utility operator effects were zero.
5. `implementation_pilot_v2_human` completed 90 episodes.  Humanitarian loss
   improved by 17.98%, while utility remained exactly unchanged.  Ledger
   diagnosis showed that late verification changed authorized state but did
   not cause a plan revision after the one-shot material response.
6. The late-response defect was repaired prospectively: a bounded operator
   intervention now triggers one replan only when the prior response used an
   incorrect resource.  Correct prior actions are never duplicated.  Incident
   request scoring was also restricted to an agent currently observing that
   incident.
7. `implementation_pilot_v3_human` then completed 90/90 episodes.  Across 15
   matched episodes per application, loss reductions versus autonomy were
   9.08% commercial, 21.56% humanitarian, and 5.64% utility restoration.  The
   probe ledger contained 3, 8, and 3 complete causal chains respectively and
   no harmful intervention effects.  These are implementation-pilot findings,
   not formal prospective gate results.
8. `implementation_pilot_v1_dense` exposed a further scientific—not merely
   numerical—problem before formal Gate 5 execution.  Causal success varied by
   a seed hash while local belief patterns were nearly constant within a
   regime.  Therefore neither KPI nor thermodynamic features could predict the
   seed-specific benefit.  The environment is being corrected before formal
   gate data so belief fragmentation itself varies across matched panels and
   directly affects decentralized inference reliability.  Visible KPI
   severity remains matched.  This implements the pre-specified mechanism
   rather than tuning a gate threshold.
9. `implementation_pilot_v2_dense` and `implementation_pilot_v3_dense`
   identified two measurement defects before formal gate execution.  First,
   disagreement was computed after gossip averaging, which erased report
   conflict; it is now the Jensen--Shannon divergence of contributed local
   belief sketches, while consensus residual remains a separate quantity.
   Second, hash buckets happened not to cover the lowest fragmentation level
   for a key humanitarian incident in the five-seed pilot.  Sequential panel
   seeds now cycle prospectively through four fixed fragmentation levels with
   an outcome-independent incident phase.  This guarantees the intended
   matched-severity design rather than relying on accidental hash balance.
10. `implementation_pilot_v7_dense` showed that the utility feature block
    could improve selection, but the humanitarian block could not.  The audit
    found a structural coverage defect: broad-scope coordinator roles stored
    only one current observation, leaving the humanitarian water incident with
    one observer and no action-capable responder.  V4 agents now have one
    persistent regional incident scope each; coordinator authority remains
    independent and can coordinate via explicit messages, but it no longer
    implies a hidden multi-incident observation.  Every incident consequently
    has multiple observers and at least one role with material authority.
11. The conditional permutation audit on `implementation_pilot_v8_dense`
    showed that thermodynamic ranking improved budgeted utility, but permuting
    values within overly narrow severity bins often preserved incident
    identity and reproduced the gain.  This failed the intended falsification.
    Before formal gates, active incidents were therefore given matched visible
    service deficit, backlog, lateness, safety stress, commitment strain,
    congestion, and primary-resource scarcity.  Belief coherence remains
    independently varied.  This is the protocol's explicit matched-KPI design:
    a thermodynamic block can pass only by distinguishing fragmented evidence,
    not by proxying incident severity or resource identity.
12. With two operator interventions for three incidents, a permuted ranking
    could retain more than half of the true benefit simply by selecting two
    thirds of all candidates.  This violated the intended scarce-attention
    premise rather than revealing information value.  The final formal budget
    was prospectively set to one intervention per episode for all methods.
    Under the eight-seed diagnostic, this stricter budget yielded permutation
    gain fractions of 0.171 humanitarian and 0.084 utility restoration.  These
    remain pilot diagnostics; the 2,000-replicate formal test is not yet run.

The preceding entries were recorded before formal gate evaluation. The frozen
thresholds in `configs/human_operator_v4_development.yaml` were not changed.

## Formal development disposition

Formal development subsequently completed 1,584 deterministic episodes and
six real-Qwen actionability episodes. Gates 1, 2, 4, 5, 6, and 7 passed. Gate 3
failed because utility-restoration fixed communication reduced aggregate loss
by 4.4317%, below the frozen 5% practical threshold. The paired absolute effect
was -0.4843 (95% cluster-bootstrap interval [-0.7715, -0.2065]), but statistical
direction does not replace the prospective practical criterion.

The protocol therefore stopped before validation, learned-policy training, and
holdout. The authoritative gate report is
`results/human_operator_v4/development/gate_status.json`; detailed interpretation
is in `notes/40_v4_gate_results_and_disposition.md`.
