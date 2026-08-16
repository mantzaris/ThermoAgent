# V6 design audit of the frozen V5 study

This is a post-development audit. V5 remains an immutable development-stage
no-go at `c895235d02dd05ccc9315621d818def9345a398c`; its validation and holdout
remain locked. The audit writes only under
`results/generalized_entropic_consensus_v6/v5_reanalysis/`.

## Fair abstention result

The original V5 safety comparison confounded removal of the consensus cutoff
with forced nonpositive-value actions. With only the consensus cutoff removed,
lower harm persists in the two primary applications but comes with lower
action coverage. With coverage exactly matched by a development-only score
threshold, neither primary application has a 95% interval strictly below zero
for the safe-minus-no-consensus harmful-action rate. The V5 Gate 8 claim is
therefore useful motivation but not confirmed coverage-controlled evidence.

V6 will make risk-coverage and utility-coverage curves primary and will compare
controllers at matched action coverage, escalation budget, total intervention
count, and communication budget.

## Implementation findings that constrain V6

1. V5 cross-fitting grouped by environment seed only, despite the protocol's
   three-field grouping requirement. V6 will use a compound seed/topology/
   scenario-family group with explicit leakage tests.
2. Holm correction was specified but absent from V5 gate computation. This
   cannot rescue already-negative V5 primary estimates; V6 will test the
   family first and apply Holm correction to subordinate comparisons.
3. The V5 conservation residual cancels algebraically. V6 will reconstruct
   independently stored initial, remaining, consumed, transferred, and lost
   resources from the event ledger and include deliberate-fault tests.
4. V5 timing observed one post-disruption decision epoch and fixed the
   pre-disruption value to zero. V6 will use full temporal trajectories.
5. V5's learner is a contextual one-step actor-critic, not sequential IPPO,
   and disallowed actions were penalized instead of masked. V6 will use real
   sequential rollouts, GAE, PPO clipping, and hard role masks.
6. V5 Qwen reporting omitted harm and mean causal effect. The addendum reports
   both and V6 prespecifies them.
7. `material_action_accepted` includes information actions. V6 will use
   `accepted_typed_action`, `accepted_physical_action`, and service progression
   as separate fields.
8. The bounded V5 oracle uses true causal effects. It demonstrates opportunity
   only and is neither deployable nor evidence about real humans.

The complete claim-impact table with code locations is
`results/generalized_entropic_consensus_v6/v5_reanalysis/implementation_audit.csv`.
