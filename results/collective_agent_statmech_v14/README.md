# V14 scientific audit: memory and quench response in LLM-agent networks

## Supported scientific scope

V14 uses statistical-mechanical observables as a reduced language for state-separated, locally informed Qwen-agent trajectories. Each agent instance owns its belief, typed action, confidence, commitment, workload, private field, inbox, outbox, and local context; a random-sequential scheduler only offers updates and transports the model-selected packet. Reference energy is an effective symmetric-layer observable. Reversal divergence is coarse-grained temporal asymmetry, not exact thermodynamic entropy production. Decoding temperature is not physical temperature.

V12 is the immutable discovery study and V13 the immutable prospective replication. Their memory effects remain separate: V12 0.01790 (95% CI 0.00357 to 0.03399) and V13 0.04030 (95% CI 0.02883 to 0.05856) nats per attempted update. The synthesis is descriptive and does not retroactively pool their protocols.

## Frozen V14 experiment

- Protocol: `v14-memory-quench-agent-statmech-1.0`; SHA-256 `5d8440dedbf389c02f3b448f38abfd1a370b8f9c4fafdba4760afc706e0bcfdf`.
- Execution source: `2b4276dd323bb048d8c98834ed0e9f8bfe5a0ed46e8735db3b471a6dc97e91ad`.
- Parent V13 commit: `20a9ca66041b1636bed15d5916aabcb605e6a063`.
- Model: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`; `NF4_double_quantization_BF16_compute`, sampling temperature 0.5, top-p 0.9, maximum 96 output tokens, no chain-of-thought request.
- Design: six new independent graph/environment clusters, four matched conditions, `N=16`, modular reciprocal delivery, coupling `J=0.8`, 45 sweeps (15 baseline, 15 perturbation, 15 restoration).
- Conditions: nominal, private-field reversal, inter-community partition, and 50% sender-preassigned categorical message corruption.
- Independent unit: complete matched graph/environment trajectory cluster. Agents, messages, tokens, and time steps are not independent replicates.

The complete frozen experiment ran 24 trajectories and 17,280 analyzed decisions. It used 17,280 formal model calls, 9,510,001 formal prompt tokens, and 1,391,607 generated tokens. Including the engineering pilot and any retained formal attempts, generation used 7.904 metered GPU-hours. The approximate incremental RTX 4090 cost range is USD 2.69–5.45. Raw decisions and full trajectories remain external at `/workspace/ThermoAgent-v14-artifacts`.

## Confirmatory results and versioned scientific correction

- H2, field-reversal maximum departure minus matched nominal: 133.805 (95% CI 107.949 to 184.075) regularized macrostate-distance units; exact one-sided sign-flip `p=0.01562`, Holm `p=0.04688`. Supported.
- H3 historical frozen number, early counter-quench peak minus final-five-sweep distance: 134.110 (95% CI 106.654 to 184.196) distance units. Its archived raw `p=0.01562` and Holm `p=0.04688` are retained only for provenance. Because this estimand is structurally nonnegative for almost every nonconstant trajectory, its directional sign-flip test is invalid and H3 has **no inferential support**. The complete trajectories are nevertheless consistent with finite-time return: 6/6 field-reversal panels re-enter the leave-one-cluster-out training-nominal threshold after exactly 6 sweeps, their mean final-five distance is 1.560, and the non-tautological fixed early-five-minus-late-five descriptive change is 86.605 distance units.
- H4, full-minus-order-only leave-one-cluster-out balanced accuracy: 0.333 (95% CI 0.167 to 0.458); exact sign-flip `p=0.03125`, Holm `p=0.04688`. Supported.

Across the new clusters, mean field-reversal maximum post-quench distance was 137.725, versus 3.921 for nominal evolution. This magnitude belongs to the frozen training-standardized shrinkage metric; it is not a universal physical scale. Mean LOCO balanced accuracy was 0.125 for order only, 0.458 for simple uncertainty, and 0.458 for the full statistical-mechanics representation. The prespecified 10,000-replicate cluster-preserving permutation audit gives `p=0.02510` for absolute full-representation accuracy and `p=0.00300` for its increment over order only; every fold refits imputation, standardization, and the classifier using training clusters alone.

The delayed prespecified sensitivity audit recomputes three-, five-, and seven-sweep macrostates and nominal fits separately, deletes every observable and observable family in turn, and retains raw as well as marginal-preserving-null-adjusted dependence estimates. At the primary five-sweep window, the mean field-minus-nominal adjusted-total-correlation contrast is 0.576 nats (95% cluster-bootstrap interval 0.427 to 0.704). These delayed analyses are sensitivity evidence, not a new prospective experiment.

## Interpretation and boundaries

The analysis jointly reports magnetization, belief-action alignment, uncertainty, configuration entropy, entropy rate, total correlation, mutual information, effective energy, energy fluctuations, susceptibility, correlations, pathwise irreversibility, macrostate distance, recovery, and route asymmetry. No single entropy is assigned a universal good/bad meaning. The full representation is evaluated with transparent multinomial logistic regression and leave-one-cluster-out preprocessing; no test cluster contributes to standardization, imputation, covariance fitting, or regularization.

V12's degree- and traffic-matched nonreciprocity boundary remains negative and is not reopened. V13's coupling/noise directions and four-cluster classifier were preliminary or unsupported and are not relabeled as V14 confirmation. No thermodynamic-limit phase transition, physical free energy, universal LLM behavior, controller benefit, application benefit, field validity, or human evidence is claimed.

## Reproduction order

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-tests.sh
THERMO_V14_ENABLE_QWEN=1 THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-pilot.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/freeze-statmech-v14-protocol.sh
THERMO_V14_ENABLE_QWEN=1 THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-formal.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/replay-statmech-v14.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/analyze-statmech-v14.sh
scripts/generate-statmech-v14-figures.sh
scripts/build-statmech-v14-results.sh
scripts/build-statmech-v14-paper.sh
scripts/verify-statmech-v14.sh
```
