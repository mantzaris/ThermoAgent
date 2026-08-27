# V15: cross-model memory controls and field-quench replication

## Scientific scope

V15 treats state-separated, locally informed LLM-agent instances as an interacting finite stochastic system. Every persistent identity has its own local belief, action, confidence, commitment, bounded memory, workload, inbox, outbox, private field, and typed authority. The random-sequential scheduler chooses only update opportunities and packet delivery; model-generated structured responses determine the scientific state changes.

The complete augmented simulator state $\Xi_t$ includes all private agent state, graph and delivery state, the quench phase, and the specified randomness source. The recorded belief-action projection $Y_t=\phi(\Xi_t)$ and rolling collective representation $Z_t=\psi(Y_{t-w+1:t})$ need not be Markov. Genuine memory can therefore act as a hidden slow coordinate when omitted from the projection. Effective reference energy is not literal physical energy, decoding temperature is not physical temperature, and bias-adjusted path-reversal divergence is coarse-grained temporal asymmetry rather than exact thermodynamic entropy production.

## Prospective design

- Frozen protocol: `v15-cross-model-memory-quench-1.0`; SHA-256 `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`.
- Frozen execution source: `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`.
- Parent V14 commit: `103e4c4598ecc26a98c37a8d03ee3663f9be1070`.
- Models: Qwen `a09a35458c702b33eeacc393d103063234e8bc28` and Granite `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.
- Inference: NF4 double quantization, BF16 computation, decoding temperature 0.5, top-p 0.9, maximum 96 generated tokens, and one bounded greedy structured-output repair.
- Design: six independent graph/environment clusters per model, `N=16`, reciprocal modular graph, `J=0.8`, 45 sweeps (15 baseline, 15 field reversal or nominal continuation, 15 restoration).
- Matched arms: nominal Markovized, field-reversal Markovized, field-reversal genuine persistent memory, and field-reversal deterministic scrambled-history placebo.
- Independent unit: complete graph/environment trajectory cluster. Agents, updates, messages, windows, calls, and tokens are not independent replicates.

Each arm constructs a fresh agent-network object from its frozen panel seeds.
The loaded model weights and tokenizer are shared read-only for throughput, but
no conversational history, key/value cache, mutable agent object, or unseeded
scientific RNG state crosses an arm boundary. Every generation receives its
frozen per-decision seed; provider accounting and raw-record indices have no
causal input to the prompt or transition law.

The formal study ran 48 trajectories and 34,560 attempted decisions. Formal generation used 34,565 calls, 20,908,194 prompt tokens, 2,893,967 generated tokens, and 48.737 metered GPU-hours. The content-addressed raw-record audit additionally found 197 interrupted-panel decision records (197 calls, 128193 prompt tokens, 17121 generated tokens, 0.328 GPU-hours) that do not enter a completed trajectory. The incident audit counts 1 additional post-generation, pre-record infrastructure model call; its prompt tokens, generated tokens, and latency were not durably recorded, so measured token and GPU-hour totals are lower bounds. Successful Qwen/Granite engineering pilots added 256 decisions and 0.349 GPU-hours. Their retained infrastructure failures added 0 decision requests and 0.000 GPU-hours. Any rejected-model attempts made during this fresh reconstruction used 0 decision requests, 0 model calls, and 0.000 GPU-hours; no network contrast was computed from them. Total fresh measured generation was at least 49.415 hours, with an approximate measured RTX 4090 cost range of at least USD 16.80-34.10.

The fresh reconstruction does not rerun the engineering-rejected Mistral
pilot. Its original pre-freeze boundary is retained from the committed
reference accounting (historical requests:
129; historical calls:
222) and is not added to the fresh
reconstruction compute total. The original sealed execution used
19.193 measured generation GPU-hours in total; the fresh
reconstruction accounted for at least 49.415 measured generation GPU-hours. The
runtime difference is reported as an environment-dependent reproducibility
cost, not a scientific effect.

The original external raw tree was unavailable after the Pod replacement.
Fresh records are replayed at decision resolution, and the frozen reconstructed
package is compared with the committed aggregate reference before extended
reporting is authorized. The machine-readable comparison status is
`passed`. This verifies the
declared aggregate science and accounting scope; it cannot establish digest
identity with deleted historical call files.

## Frozen hypotheses

- H1 (Granite field quench versus nominal): 42.263 (95% CI 24.316 to 59.303) distance units; exact sign-flip `p=0.01562`, allocated alpha 0.02. **Supported**.
- H2 (persistent minus Markovized path divergence, pooled across model-stratified pairs): 0.05438 (95% CI 0.03459 to 0.07548) nats/update; Holm `p=0.00098` within the alpha-0.03 family. **Supported**.
- H3 (persistent minus scrambled-history path divergence): 0.04845 (95% CI 0.02190 to 0.07526) nats/update; Holm `p=0.00342`. **Supported**.
- H4 (fixed recovery sweeps 31-35 minus 41-45): 52.541 (95% CI 35.406 to 68.673) distance units; Holm `p=0.00073`. **Supported**.

The exact direction and model-specific heterogeneity are retained in `tables/hypothesis_effects.csv` and `tables/panel_statistics.csv`; the README does not reinterpret null or adverse signs. Qwen mean adjusted divergence was -0.02592, 0.00449, and -0.00239 nats/update for Markovized, persistent, and scrambled arms. The corresponding Granite means were -0.18622, -0.10788, and -0.19789. Mean persistent-minus-scrambled prompt length was 3.13 tokens.

`tables/hypothesis_model_stratified.csv` is a descriptive sensitivity, not a
replacement confirmatory analysis. It makes explicit that Qwen's
persistent-minus-scrambled mean is 0.00689 nats/update with 4
of six positive clusters and an unadjusted within-model exact sign-flip
`p=0.21875`, whereas Granite's mean is 0.09001 with 6 of six
positive clusters. The pooled H3 result is therefore not independent
confirmation within both families. Arm-level adjusted estimates may be
negative because the raw block divergence can lie below its shuffled floor;
positive paired contrasts do not establish positive absolute entropy
production.

The primary floor still uses the frozen 500 time permutations.  The extended
`tables/irreversibility_sensitivity.csv` retains the empirical permutation-null
interval, its standard deviation, and the Monte Carlo standard error and
normal-approximation interval for the floor mean.  These audit columns
reproduce the frozen raw divergence, mean floor, and adjusted value exactly;
they do not redefine H2 or H3.

## Secondary collective-observable extension

The frozen protocol and H1-H4 are unchanged. A versioned descriptive extension
(`v15-collective-observables-1.1`, SHA-256 `470ea8a1fccf013290f8a672abd980840c2297ca08f413442267503074ae95b5`) computes,
within each complete trajectory and phase, connected belief correlation by
actual shortest-path distance, magnetization autocorrelation with a primary
two-sweep lag truncation, and the Binder cumulant. One- and three-sweep lag
truncations are sensitivities. Binder window and pooling sensitivities compare
full versus early/late half-phases and cluster-first versus moment-pooled
estimates. Pair, node, and update counts are not used as
replicates. Undefined zero-variance or zero-second-moment cases remain missing.

| Model | Descriptive contrast | Estimate | 95% cluster-bootstrap interval | Independent clusters |
|---|---|---:|---:|---:|
| Qwen | disruption minus baseline at graph distance 1 | 0.00238 | -0.01750 to 0.02277 | 6 |
| Qwen | recovery minus baseline at graph distance 1 | 0.00080 | -0.01254 to 0.01453 | 6 |
| Qwen | persistent minus markovized during recovery | -2.46995 | -5.79214 to 0.17724 | 6 |
| Qwen | disruption minus baseline field markovized | -4.80504 | -7.42824 to -2.20832 | 6 |
| Qwen | recovery minus baseline field markovized | -3.99611 | -6.95969 to -1.88680 | 6 |
| Granite | disruption minus baseline at graph distance 1 | -0.02032 | -0.05486 to 0.00947 | 6 |
| Granite | recovery minus baseline at graph distance 1 | -0.02748 | -0.05776 to -0.00346 | 6 |
| Granite | persistent minus markovized during recovery | -4.14109 | -19.88286 to 11.60067 | 2 |
| Granite | disruption minus baseline field markovized | 0.16643 | -0.09447 to 0.52063 | 6 |
| Granite | recovery minus baseline field markovized | 0.13129 | -0.13195 to 0.50873 | 6 |

These are finite-window descriptive contrasts. Connected correlation exposes
spatial organization beyond mean order; truncated autocorrelation summarizes
persistence; Binder $U_4$ summarizes order-parameter shape. They do not imply
a correlation length, critical slowing down, a Binder crossing, or a phase
transition.

## V14 scientific correction

No frozen V14 decision or trajectory was altered. The versioned V14 audit now fits recovery thresholds using training clusters only, completes the frozen 10,000-replicate cluster-preserving permutation analysis, recomputes three-, five-, and seven-sweep nominal geometries, deletes individual observables, and audits finite-sample dependence bias. The historical H3 maximum-minus-final estimand, interval, raw p-value, and Holm value remain archived, but its structurally nonnegative sign makes the directional test invalid. Its machine-readable disposition is `inferential_support: false`; recovery language uses threshold re-entry, final residual, the complete path, and fixed early-versus-late descriptive changes.

## Supported boundaries

Results are finite-size and model-specific. Neither model is a human participant. No field validity, application benefit, controller advantage, performance superiority, thermodynamic-limit phase transition, physical free energy, or exact LLM entropy production is claimed. Persistent history and scrambled history are prompt-format controls; they do not make the projected binary process fully observed. Negative adjusted information quantities are retained rather than truncated.

## Reproduction order

```bash
export THERMO_V15_ARTIFACT_ROOT=/workspace/ThermoAgent-v15-reconstruction-b309f0ab
scripts/setup-statmech-v15-runpod.sh
.venv/bin/python scripts/prefetch-statmech-v15-models.py
PYTHON_BIN=.venv/bin/python scripts/run-statmech-v15-tests.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-reconstruction-pilot.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-reconstruction-pilot.sh granite
# Run the next commands against a clean checkout at the committed V15
# reference through audited reconstruction wrappers.  The 50-hour value is
# the explicitly authorized operational reconstruction guard; it does not
# alter the frozen protocol's scientific design or its historical 25-hour cap.
THERMO_V15_ENABLE_LLM=1 THERMO_V15_AUTHORIZED_GPU_HOURS=50 scripts/run-statmech-v15-reconstruction-formal.sh qwen
THERMO_V15_ENABLE_LLM=1 THERMO_V15_AUTHORIZED_GPU_HOURS=50 scripts/run-statmech-v15-reconstruction-formal.sh granite
scripts/run-statmech-v15-reconstruction-analysis.sh
scripts/run-statmech-v15-surrogate.sh
scripts/generate-statmech-v15-figures.sh
scripts/build-statmech-v15-results.sh
scripts/build-jstat-paper.sh
scripts/verify-statmech-v15.sh
```

Raw prompts, completions, and trajectory tables are external at `/workspace/ThermoAgent-v15-reconstruction-b309f0ab`. Compact aggregate tables, checksums, vector figures, and manuscript sources are repository-facing.
