# V4 gate results and disposition

Evidence stage: formal development. All large-scale operators are simulated.
No validation, RL, holdout, or real-human outcome evidence exists.

## Completed evidence

- 1,584/1,584 formal deterministic development episodes completed.
- 6/6 real-Qwen actionability episodes completed and passed.
- 1,590/1,590 event ledgers replayed exactly with zero mismatches.
- Maximum conservation residual: 0.
- Formal episode failures: 0.

## Gate decisions

1. Engineering integrity: pass.
2. Agent actionability: pass. Deterministic first-pass/one-repair validity was
   100%; 81.94% of accepted actions reached service. Real Qwen was 6/6 on every
   actionability stage.
3. Coordination necessity: **fail**. Commercial reduction was 24.924%,
   humanitarian 8.594%, and utility restoration 4.432%. The frozen aggregate
   threshold was 5% in each application. Utility therefore failed even though
   four disruption regimes exceeded the 2% per-regime target.
4. Human causal usefulness: pass in the required two applications.
   Humanitarian loss fell 20.957% and utility loss 13.422%; commercial fell only
   0.912% and failed its application gate.
5. Thermodynamic incremental value: pass for the required humanitarian and
   utility applications. Paired causal-utility gain over KPI-only was 0.15885
   [0.06195, 0.26130] and 0.17868 [0.10293, 0.26088], respectively. Commercial
   gain was exactly zero.
6. Trigger feasibility: pass. Every disrupted application/regime activated;
   timely activation was 100%; nominal and pre-onset false activation were 0%.
   No low-confidence operator decision occurred, so safe abstention remains
   unexercised.
7. Mechanism specificity: pass. Globally public gains were zero. Conditional
   within-KPI permutation reproduced the true gain in 0/2,000 replicates for
   each positive application; mean gain fractions were 7.88% and 7.04%.

## Hard stop

All seven gates were required. Gate 3 failure locks validation, five-seed RL
training, and holdout. The 4.432% result is not rounded up or reinterpreted as a
5% pass. Later directories contain explicit `NOT_RUN.json` dispositions and no
placeholder outcomes.

## Causal-chain evidence

The human-causal matrix records 180 bounded simulated interventions (1,440
estimated operator minutes), 68 beneficial interventions, zero harmful
interventions, and complete chains in 2 commercial, 32 humanitarian, and 34
utility panels. The full chain is ledger-verifiable: request → queue →
allocation → hashed authorized view → typed operator action → commitment/action
change → feasible movement/service → demand/critical-service arrival → primary
outcome change. Commercial chains are rare and its aggregate human gate fails.

These findings are promising development mechanism evidence only. The stop
prevents any claim of out-of-sample generalization or AIJ-ready empirical
support.
