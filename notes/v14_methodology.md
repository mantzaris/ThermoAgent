# V14 methodology: memory replication synthesis and prospective quench confirmation

Date: 2026-08-20
Parent: `collective-agent-statmech-v13` at `20a9ca66041b1636bed15d5916aabcb605e6a063`

## Why V14 is narrower

V12 was the discovery study and V13 was its prospective extension. V13 replicated the bounded-memory association with coarse-grained pathwise irreversibility but did not confirm the V12 coupling/noise directions. V13 also found a large response to a controlled private-field reversal, while its four-cluster representation analysis was necessarily preliminary. V14 therefore does not repeat the broad V12/V13 grids and does not reopen the negative nonreciprocity result. It spends new LLM computation only on an independent, prospectively frozen quench replication with enough matched clusters for cluster-level exact inference.

The scientific claim is descriptive: statistical mechanics supplies an interpretable reduced language for collective LLM-agent trajectories. Reference energy is an effective coordinate defined by a symmetric comparison model. Block reversal divergence is coarse-grained temporal asymmetry, not exact thermodynamic entropy production. Decoding temperature is a model-generation control, not physical temperature.

## Audit of the inherited process

The V13 trajectory process uses actual independent Qwen agents. Each agent owns a private field, belief, action, confidence, commitment, bounded memory, workload, inbox, outbox, and role. A random-sequential scheduler chooses one update opportunity and delivers a typed signal packet; the Qwen response selects belief, action, commitment, memory, signal, and tool action. Invalid output is not replaced with a centrally chosen scientific action. Agent-private fingerprints are checked around every update.

V14 inherits, without modifying V13 or V12 namespaces:

- `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`;
- NF4 double quantization with BF16 computation;
- sampling temperature 0.50, top-p 0.90, at most 96 generated tokens;
- the V12 prompt, typed response schema, parser, one bounded repair, serializer, and latent-label counterbalancing;
- random permutation of all `N` agents within each sweep;
- the V13 reciprocal modular graph and field-reversal, partition, corruption, and restoration operators.

The V13 field-reversal macrostate distance near 45 is not a physical scale. It was a regularized Mahalanobis distance in one fitted feature geometry. V14 prospectively audits training-only standardization, Euclidean, diagonal, shrinkage, and robust covariance methods; ridge values 0.05, 0.10, 0.25, and 0.50; nominal fitting windows; rolling windows 3, 5, and 7 sweeps; and leave-observable-family-out fits.

## Discovery and replication roles

- V12 memory effect: discovery estimate only; 24 matched clusters.
- V13 memory effect: prospective replication estimate; six matched clusters.
- V14 memory analysis: no new memory trajectories. V12 and V13 are shown separately, followed only by a labelled descriptive inverse-variance synthesis. Primary block length remains 3; block lengths 2 and 4 and fixed pseudocounts are sensitivities.
- V13 quench/representation results: preliminary motivation only; no V13 trajectory enters a V14 confirmatory confidence interval.
- V14: six new independent graph/environment clusters, each containing four matched 45-sweep arms.

## Prospective V14 formal design

Each new trajectory has `N=16` agents on a reciprocal modular graph, coupling 0.80, sampling temperature 0.50, balanced disordered initialization, and 15 baseline, 15 intervention, and 15 restoration sweeps. The four matched conditions are nominal, private-field reversal, inter-community partition, and categorical signal corruption from exactly half of senders. Six new graph/environment clusters give 24 trajectories and 17,280 LLM decisions.

Arms within a cluster share graph construction, private-field initialization, counterbalancing, random-sequential update tape, message opportunities, and inference seeds where technically meaningful. The intervention is the only intended arm-level difference. A graph/environment cluster is the inferential unit.

The engineering pilot is limited to 256 decisions and may inspect only structured validity, latent occupancy, both transition directions, perturbation scheduling and restoration, privacy, delivery, runtime, and token projection. It may not inspect whether field reversal or any other intervention produces the desired scientific effect.

## Frozen estimands and inference

H1 is historical discovery–replication evidence: persistent memory increases bias-adjusted block reversal divergence. It is not relabelled as a new V14 experiment.

The Holm-corrected V14 confirmatory family is:

1. H2: within each cluster, maximum post-quench macrostate distance under field reversal minus the matched nominal maximum is positive.
2. H3: within each field-reversal trajectory, maximum distance in the recovery period minus mean distance over the final five sweeps is positive, indicating movement back toward the restored nominal regime.
3. H4: leave-one-cluster-out balanced accuracy of the full statistical-mechanics representation minus the order-only representation is positive.

The primary test is an exhaustive one-sided sign-flip randomization over the six cluster-level effects. Cluster bootstrap intervals with 10,000 resamples summarize uncertainty. Holm correction covers H2–H4. A positive estimate alone does not imply support.

The primary nominal manifold is fitted only to nominal trajectories from training clusters. Missing/nonfinite training features are imputed with training medians; test values never enter preprocessing. Features are standardized using training data, followed by shrinkage covariance with ridge fraction 0.10. The primary rolling window is five sweeps. Classification uses fixed feature groups and L2-regularized multinomial logistic regression (`C=0.5`) with leave-one-cluster-out evaluation.

## Statistical-mechanical interpretation

- Magnetization: directional order.
- Belief–action overlap: alignment of intended and committed states.
- Susceptibility: finite-system fluctuation of belief magnetization.
- Individual entropy: local categorical/state uncertainty.
- Configuration entropy: diversity of coarse collective states.
- Entropy rate: bounded-history temporal unpredictability.
- Mutual information and total correlation: pairwise and system-wide dependence.
- Reference energy: compatibility with a symmetric interaction model, not literal energy.
- Energy variance: fluctuation in that compatibility, not physical heat capacity.
- Correlation time: persistence.
- Path reversal divergence: coarse-grained temporal asymmetry with an empirical shuffle floor.
- Macrostate distance: standardized departure from the training nominal ensemble.
- Loop area: finite-system path dependence, not thermodynamic hysteresis without a limit argument.

No free-energy-like scalar is primary. An effective temperature is not independently identifiable in this study, so `E - T S` is omitted.

## Compute choice made before formal outcomes

V13 achieved approximately 32,672 analyzed decisions in about 14.6 metered generation hours, implying approximately 1.61 seconds per decision and a V14 projection near 7.7 hours. The prospective hard generation ceiling is 10.0 metered GPU hours and 10 million prompt tokens. No more than 17,280 formal decisions may execute. The experiment is resumable by atomic trajectory and must not selectively rerun unfavorable complete panels. The precise pilot estimate is frozen into the protocol before formal execution.
