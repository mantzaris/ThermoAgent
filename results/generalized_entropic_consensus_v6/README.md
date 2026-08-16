# Generalized Entropic Consensus V6

## Research question

Can generalized measures of decentralized uncertainty and consensus identify when independent autonomous-agent recommendations are unsafe, improving selective autonomy, abstention, communication, and bounded simulated-operator escalation at matched action coverage and operator budget?

V6 is scientifically distinct from V5. V5's immutable negative result remains unchanged: KPI plus Shannon entropy and Jensen–Shannon disagreement did not improve direct intervention ranking. V6 instead asks whether uncertainty and consensus predict *when to delegate*, not which domain action is correct.

## Evidence status

The authoritative disposition is **development no go**. Failed required gates: Gate 4 (autonomous_agent_validity), Gate 5 (primary_selective_safety), Gate 6 (utility_service_and_trigger_feasibility), Gate 7 (mechanism_specificity), Gate 9 (multiseed_learning_stability), Gate 10 (cross_application_replication). This package distinguishes pilots, frozen development, real-Qwen qualification, multi-seed sequential PPO, validation, and sealed holdout. It never treats candidate decisions within one panel as independent replicates.

Validation status: **prospectively not run**. Holdout status: **prospectively not run**.

## Integrity and retained failures

The audit replayed 5,958 stored V6
ledgers. Frozen formal/Qwen evidence accounts for
4,650 ledgers with
0 mismatches, maximum
independently reconstructed conservation residual
`0.000000000000`, and
0 privacy failures. The same
audit intentionally retains 792
privacy-invalid early pilot ledgers; these are the design iterations that
preceded the final privacy repair, not silently excluded formal episodes.

Scientific episodes remain bound to their recorded execution-source checksum.
Post-outcome source transitions are explicitly limited to lossless storage,
reporting, dashboard presentation, and replay evidence classification; none
can change a hypothesis, threshold, result, or gate outcome.

## Applications and independence

- Humanitarian logistics and abstract defensive utility restoration are the primary replication applications.
- Commercial logistics is a prespecified boundary application where ordinary KPIs may suffice.
- Every organization has a private observation, belief distribution, memory vault, utility, inbox/outbox, commitments, role-specific typed tools, and separate decision authority. Partitions block delivery. The simulator validates actions but does not replace rejected decisions with oracle actions.
- Utility cyber events are abstract simulator state changes only. No real infrastructure, protocol, credential, device, or external target was accessed.

## Measures

For belief `p_i` over six incident modes, V6 computes normalized Shannon entropy and Tsallis entropy at q = 0.5, 1, 1.5, 2, and 3. Gini–Simpson impurity is the normalized q=2 case. Reliability-weighted pooled beliefs support Jensen–Shannon and Jensen–Tsallis disagreement, graph-weighted disagreement, consensus residuals, and temporal slopes. Operational energy and free-energy-style quantities remain secondary diagnostics and are not literal thermodynamics.

All distributed-sketch messages, bytes, latency, operational messages, LLM calls, prompt tokens, generated tokens, GPU time, and simulated-operator minutes are counted.

## V5 fair-abstention addendum

`v5_reanalysis/` preserves the original V5 findings and adds same-score, coverage-matched, mandatory-action, and operator-budget-matched comparisons. It does not unlock V5 validation or revise V5 gates.

## Frozen development findings

- Humanitarian: harm-rate reduction 0.023 (95% CI 0.013 to 0.033); relative service-loss change -0.017; net causal-utility change 0.099.
- Utility Restoration: harm-rate reduction 0.012 (95% CI 0.003 to 0.022); relative service-loss change -0.009; net causal-utility change 0.051.
- Commercial: harm-rate reduction 0.017 (95% CI 0.009 to 0.026); relative service-loss change -0.009; net causal-utility change 0.066.

The gate table at `development/gate_checks.csv` is authoritative. Negative, zero, and harmful actions are retained. Simulated-operator results are not evidence about real-human usability, workload, trust, or effectiveness.

The primary selective-risk cross-fitting passed the frozen seed, topology, and
scenario-family isolation audit. A post-outcome audit found that the separate
pooled supervised learnability ceiling reused numeric environment seeds across
applications even though it isolated application-specific topology and
scenario families. That diagnostic is retained but methodologically
compromised; it cannot rescue a failed gate or unlock validation. See
`reproducibility/protocol_deviations.csv` and
`notes/64_v6_development_findings.md`.

## Real Qwen and learned agents

Primary model: `Qwen/Qwen2.5-7B-Instruct`, immutable revision `a09a35458c702b33eeacc393d103063234e8bc28`, bitsandbytes NF4, BF16 computation. Real-Qwen qualification contains 150 episodes and 2700 independent-agent decision records. Sequential role-specific PPO uses local execution observations, action masks, discounted trajectories, GAE, clipping, and five independent seeds per method. It is not the V5 contextual actor-critic.

## Compute

- Recorded reserved single-GPU Pod hours: 1.6923; measured GPU-active hours: 1.1423.
- LLM calls: 2,812.
- Prompt tokens: 3,161,730; generated tokens: 236,987.
- Recorded communication: 1,737,561 messages and 203,207,336 bytes, including operational and thermodynamic-sketch traffic during PPO training/evaluation and Qwen qualification.
- Approximate Pod cost at the recorded $0.34/hour accounting rate: $0.58.

## Artifact map

- `protocol/`: frozen protocol and checksums.
- `manifests/`: sealed input manifests and stage disposition.
- `v5_reanalysis/`: fair V5 safety reanalysis and implementation audit.
- `pilots/`: retained design iterations, including failures and superseded pilots.
- `development/`: frozen reference, cross-fitting, dynamic evaluation, learnability, communication, gates, and power evidence.
- `training/`: five-seed sequential PPO manifests, curves, and small checkpoints.
- `qwen/`: real open-weight agent decision and episode summaries.
- `raw/`: compressed event-sourced episodes.
- `tables/` and `statistics/`: publication-facing numerical summaries.
- `figures/pdf/` and `figures/png/`: vector figures and 240-DPI previews.
- `dashboard_exports/`: populated deterministic replay export.
- `reproducibility/`: replay, checksum, environment, compute, PDF QA, deviations, exclusions, and failures.

## Figure and table guide

Every paper-facing PDF is a true vector artifact with a 240-DPI preview and a
stored source-data CSV. See [`tables/figure_catalog.csv`](tables/figure_catalog.csv)
for all figure descriptions and [`tables/table_catalog.csv`](tables/table_catalog.csv)
for the statistical/table inventory. The matched dashboard exports are actual
ledger replays, not hand-entered result graphics. The evaluator-only replay
panel is visibly privileged and never enters the simulated-operator payload.

## Reproduction

```bash
./scripts/run-v6-tests.sh
./scripts/run-v6-v5-reanalysis.sh
./scripts/run-v6-pilot.sh
./scripts/freeze-v6-protocol.sh
./scripts/run-v6-development.sh
./scripts/analyze-v6-development.sh
./scripts/train-v6-multiseed.sh
./scripts/run-v6-real-qwen.sh
./scripts/replay-v6-results.sh
./scripts/evaluate-v6-gates.sh
./scripts/build-v6-report.sh
./scripts/generate-v6-figures.sh
./scripts/validate-v6-pdfs.sh
./scripts/index-v6-artifacts.sh
```

Validation and holdout scripts enforce gate locks and refuse execution when not prospectively unlocked.

## Limitations and readiness

All domains and operators are simulations. The models are abstractions, not validated logistics or critical-infrastructure digital twins. No human participants were studied. Development evidence cannot establish confirmatory generalization. `PAPER_SUMMARY.md` gives the evidence-specific publication disposition; `PAPER_OUTLINE.md` is a writing plan, not a completed manuscript.
