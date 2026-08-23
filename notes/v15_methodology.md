# V15 methodology and V14 scientific audit

## Provenance and scope

V15 branches from immutable V14 commit `103e4c4598ecc26a98c37a8d03ee3663f9be1070`. The frozen V14 protocol hash is `5d8440dedbf389c02f3b448f38abfd1a370b8f9c4fafdba4760afc706e0bcfdf`; its formal execution-source hash is `2b4276dd323bb048d8c98834ed0e9f8bfe5a0ed46e8735db3b471a6dc97e91ad`. V1-V13 result namespaces and V14 raw choices and trajectories are immutable. V14 derived analyses are corrected only through a versioned audit with the committed historical reports archived alongside the correction.

The paper studies state-separated, locally informed LLM-agent instances. The complete graph/environment trajectory cluster is the independent inferential unit. An agent, token, update, packet, time point, rolling window, or classifier prediction is not an independent replicate.

## V14 source audit and corrections

The source audit found five scientifically material issues.

1. `disruption_summaries()` fitted a recovery threshold from the held-out field panel's own baseline distances, although the protocol and paper described a leave-one-cluster-out training-nominal threshold. Audit version 1.1 passes an explicit cluster-to-training-threshold map from nominal-manifold fitting into recovery analysis and records the contributing training clusters.
2. Frozen H3 used `maximum recovery distance - final-five-sweep mean distance`. This is structurally nonnegative for nearly every nonconstant trajectory. The archived estimate, interval, raw sign-flip p-value, and Holm value remain reproducible, but the directional test is invalid. Machine-readable H3 fields now distinguish numerical criterion, validity, inferential support, and trajectory consistency. Recovery statements use the full path, cluster-excluded threshold re-entry, final residual, and a descriptive fixed early-five-minus-late-five contrast.
3. The protocol's 10,000 cluster-preserving permutations had not been executed. Audit version 1.1 permutes the four condition labels within each graph/environment cluster and refits imputation, scaling, logistic regression, and leave-one-cluster-out evaluation in every replicate.
4. Three- and seven-sweep rolling tables existed, but nominal geometries were fitted only for the five-sweep primary representation. Audit version 1.1 independently recomputes rolling observables, training-only scaling, covariance, thresholds, and distances for windows 3, 5, and 7. It also deletes each macrostate coordinate individually, in addition to the existing observable-family deletions.
5. The raw 32-variable total-correlation plug-in estimate has a large short-window finite-sample floor. Audit version 1.1 retains that historical coordinate and adds independently circular-shifted marginal-preserving nulls, untruncated bias-adjusted total correlation, normalized total correlation, and raw and adjusted pairwise and edge mutual information. Negative adjusted estimates are retained.

These are delayed completions and corrections of prespecified V14 analysis, not a new prospective experiment. No prompt, choice, graph, perturbation, seed, or raw outcome changes.

## Process and projections

Let the complete augmented simulator state be

`Xi_t = (all private agent states, memories, inboxes, outboxes, private fields, workloads, graph state, delivery state, quench phase, randomness state)`.

With fixed model weights, state-complete reconstructed prompts, the explicitly recorded scheduler state, and a specified random source, the simulator defines a time-inhomogeneous stochastic transition process under the quench schedule. The observable microscopic projection is

`Y_t = phi(Xi_t)`,

containing recorded categorical beliefs and actions plus bounded observable coordinates. The rolling collective representation is

`Z_t = psi(Y_{t-w+1:t})`.

Neither `Y_t` nor `Z_t` is assumed Markov. Two projected states with the same belief and action vectors can have different future laws when their private histories differ. Persistent memory is therefore a candidate hidden slow coordinate, and omitting it can increase history dependence and forward-reverse block divergence. This mechanism does not imply physical dissipation or exact thermodynamic entropy production.

## V15 prospective design

The primary Qwen model and independent Granite family are pinned before execution:

- `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`;
- `ibm-granite/granite-3.3-8b-instruct`, revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.

The initially preferred Mistral 7B family was rejected before freeze on engineering grounds: only 47/128 responses were valid after one bounded repair, 81 remained invalid, and one transition direction was absent. No network contrast or temporal-asymmetry outcome was inspected. The protocol's genuinely different, ungated fallback is IBM Granite 3.3 8B Instruct (Apache 2.0); its exact public revision was resolved before its pilot. The two zero-science Mistral infrastructure failures and the completed failed pilot remain external and enter accounting.

Both use Transformers 4.55.4, PyTorch 2.8.0+cu128, bitsandbytes 0.47.0, NF4 double quantization, BF16 computation, decoding temperature 0.5, top-p 0.9, and at most 96 generated tokens. Decoding temperature is a sampling-control parameter, not a physical temperature.

Each model has six independent reciprocal modular graph/environment clusters, `N=16`, `J=0.8`, and four matched 45-sweep arms:

1. nominal evolution with Markovized state;
2. field reversal and restoration with Markovized state;
3. field reversal and restoration with genuine bounded private memory;
4. field reversal and restoration with a length- and format-matched scrambled-history placebo.

The quench schedule is 15 baseline, 15 reversed-field, and 15 restored-field sweeps. One sweep is 16 attempted random-sequential local updates. Cluster arms share graphs, initial states, private fields, label mappings, update opportunities, recipient variates, and inference seeds where meaningful.

The scrambled tape uses only an agent's own past update-opportunity times. Each displayed entry has the genuine memory section's field names and format but deterministically randomized belief, action, and memory content. It cannot contain a future time or a peer's private state. The tape and its hash are frozen before formal outcomes. Token distributions are measured; exact token equality is not assumed.

## Prospective hypotheses and inference

- H1: Granite field reversal produces greater maximum post-quench macrostate departure than matched nominal evolution.
- H2: genuine persistent memory increases bias-adjusted block reversal divergence relative to Markovized state.
- H3: genuine persistent memory increases bias-adjusted block reversal divergence relative to scrambled history.
- H4: recovery sweeps 31-35 have greater mean distance than sweeps 41-45 in field-Markovized trajectories.

H1 receives one-sided alpha 0.02. H2-H4 form a one-sided Holm family with total alpha 0.03. Exhaustive sign flips are the primary paired test; deterministic 10,000 cluster bootstraps summarize uncertainty. H1 has six Granite clusters. H2-H4 use 12 model-qualified graph/environment units with model identity retained. No hypothesis, estimand, window, prompt, graph count, or exclusion changes after freeze.

The H4 statistic is prospectively fixed and can be positive, zero, or negative. It replaces the structurally invalid V14 H3 maximum-minus-final form.

## Information and temporal-asymmetry estimators

The primary block length is three and the primary additive pseudocount is 0.5. Blocks two and four and pseudocounts 0.1 and 1.0 are frozen sensitivities. Five hundred time-shuffle nulls per panel estimate the finite-length floor. Matched arms in the same model/cluster use the same shuffle tape to reduce Monte Carlo noise in paired contrasts. The reported adjusted measure is observed block reversal KL per attempted update minus the mean shuffled floor. It is a coarse-grained pathwise irreversibility measure.

Rolling dependence uses three-, five-, and seven-sweep windows. Raw total correlation is accompanied by 200 marginal-preserving, independently circular-shifted null replicates per rolling point. Normalization divides bias-adjusted total correlation by the sum of single-variable marginal entropies when nonzero.

## Effective-model comparison

The kinetic surrogate is fitted only to the immutable V13 microscopic-response table. No coefficient is fitted to a V14 or V15 quench trajectory. The surrogate receives the direct system's graph family, initial-condition generator, update tape, field schedule, coupling, and quench/counter-quench times. Time-resolved belief and action magnetization, overlap, effective reference energy, configuration entropy, susceptibility, shared nominal departure, peak timing, recovery, and route area are compared. Agreement is explanatory; disagreement identifies failure of the low-dimensional closure.

The symmetric-layer reference energy is an effective compatibility coordinate. It is not literal physical energy. Direct simulations at `N=16` and CPU surrogate sizes 8, 16, 32, and 64 do not establish thermodynamic-limit scaling or a phase transition.

## Engineering-only pilot and stopping rules

The pilot contains 128 decisions per model split equally between persistent and scrambled-history arms. Inspection is restricted to loading, schema validity, occupancy, both transition directions, timing, privacy, delivery, prompt-length matching, latency, tokens, and projected compute. No memory, quench, or irreversibility contrast is calculated. Formal execution may stop only for corrupted data, systematic invalidity, privacy or scheduler-authority failure, unrecoverable hardware failure, or the frozen compute ceiling. Complete unfavorable trajectories are never selectively rerun.

Qwen passed with 128/128 valid structured responses, balanced latent occupancy (0.492), and 29 transitions in each belief direction. Granite passed with 128/128 first-pass-valid responses, latent-plus occupancy 0.656, eight minus-to-plus transitions, and two plus-to-minus transitions. Their blind formal projections were respectively 7.751 and 11.255 generation hours and together 20,714,670 prompt tokens. The complete two-model design was therefore retained unchanged under the frozen ceilings.

The formal freeze records protocol SHA-256 `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`, execution-source SHA-256 `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`, schema SHA-256 `c0382247001c9c586190b81ad4a83535ceb71b03dadfd81e80cac220e9580f0d`, seed-manifest SHA-256 `d9850e5854af307364cd5504d75d0df449412817f3026b1149e8d7de6e8fdaf4`, and memory-control-manifest SHA-256 `24bae40fbc4026b66eef87768dd9ae4edbcc3e7b7a9b1f6a850aede21b32a9f4`.

## Post-formal reporting boundary

After all 48 formal trajectories, replay, and frozen analyses completed, reporting was synchronized to the sealed aggregate outputs. No formal source, protocol, seed, prompt, estimator, multiplicity rule, or trajectory was changed. Model-specific decompositions and estimator sensitivities are reported as boundary analyses: the pooled tests retain their prospective status, while per-model sign counts and sensitivity cells describe heterogeneity rather than creating new confirmatory families.

## Journal format and literature provenance

The manuscript follows the official IOP/JSTAT author guidance consulted on 21 August 2026: the JSTAT scope page, IOP's current LaTeX guidance, and IOP's article-format guidance. A mandatory proprietary class was not required; readable common LaTeX, vector figures, and an abstract within the IOP length guidance were retained. Recent literature entries were checked against primary arXiv records, while established references use publisher DOI metadata. A compact bibliographic audit accompanies the final reproducibility package rather than treating unverifiable recent citations as established literature.
