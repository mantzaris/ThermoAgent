# V15: cross-model memory controls and field-quench replication

## Scientific scope

V15 treats state-separated, locally informed LLM-agent instances as an interacting finite stochastic system. Every persistent identity has its own local belief, action, confidence, commitment, bounded memory, workload, inbox, outbox, private field, and typed authority. The random-sequential scheduler chooses only update opportunities and packet delivery; model-generated structured responses determine the scientific state changes.

The complete augmented simulator state $\Xi_t$ includes all private agent state, graph and delivery state, the quench phase, and the specified randomness source. The recorded belief-action projection $Y_t=\phi(\Xi_t)$ and rolling collective representation $Z_t=\psi(Y_{t-w+1:t})$ need not be Markov. Genuine memory can therefore act as a hidden slow coordinate when omitted from the projection. Effective reference energy is not literal physical energy, decoding temperature is not physical temperature, and bias-adjusted path-reversal divergence is coarse-grained temporal asymmetry rather than exact thermodynamic entropy production.

## Prospective design

- Frozen protocol: `v15-cross-model-memory-quench-1.0`; SHA-256 `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`.
- Frozen legacy execution source: `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`.
- Clean semantic-source checksum: `f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2`.
- Parent V14 commit: `103e4c4598ecc26a98c37a8d03ee3663f9be1070`.
- Models: Qwen `a09a35458c702b33eeacc393d103063234e8bc28` and Granite `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.
- Inference: NF4 double quantization, BF16 computation, decoding temperature 0.5, top-p 0.9, maximum 96 generated tokens, and one bounded greedy structured-output repair.
- Design: six independent graph/environment clusters per model, `N=16`, reciprocal modular graph, `J=0.8`, 45 sweeps (15 baseline, 15 field reversal or nominal continuation, 15 restoration).
- Matched arms: nominal Markovized, field-reversal Markovized, field-reversal genuine persistent memory, and field-reversal deterministic scrambled-history placebo.
- Independent unit: complete graph/environment trajectory cluster. Agents, updates, messages, windows, calls, and tokens are not independent replicates.

The formal study ran 48 trajectories and 34,560 attempted decisions. Formal generation used 34,565 calls, 20,908,194 prompt tokens, 2,893,967 generated tokens, and 18.937 metered GPU-hours. Successful Qwen/Granite engineering pilots added 256 decisions and 0.141 GPU-hours. Their retained infrastructure failures added 1 decision requests and 0.000 GPU-hours. The retained, engineering-rejected Mistral attempts used 129 decision requests, 222 model calls, and 0.116 GPU-hours; no network contrast was computed from them. Total measured generation was 19.193 hours, with an approximate RTX 4090 cost range of USD 6.53-13.24.

## Frozen hypotheses

- H1 (Granite field quench versus nominal): 42.263 (95% CI 24.316 to 59.303) distance units; exact sign-flip `p=0.01562`, allocated alpha 0.02. **Supported**.
- H2 (persistent minus Markovized path divergence, pooled across model-stratified pairs): 0.05438 (95% CI 0.03459 to 0.07548) nats/update; Holm `p=0.00098` within the alpha-0.03 family. **Supported**.
- H3 (persistent minus scrambled-history path divergence): 0.04845 (95% CI 0.02190 to 0.07526) nats/update; Holm `p=0.00342`. **Supported**.
- H4 (fixed recovery sweeps 31-35 minus 41-45): 52.541 (95% CI 35.406 to 68.673) distance units; Holm `p=0.00073`. **Supported**.

The exact direction and model-specific heterogeneity are retained in `tables/hypothesis_effects.csv` and `tables/panel_statistics.csv`; the README does not reinterpret null or adverse signs. Qwen mean adjusted divergence was -0.02592, 0.00449, and -0.00239 nats/update for Markovized, persistent, and scrambled arms. The corresponding Granite means were -0.18622, -0.10788, and -0.19789. Mean persistent-minus-scrambled prompt length was 3.13 tokens.

The pooled memory result is heterogeneous. Persistent-minus-Markovized means were 0.03041 nats/update for Qwen (5/6 positive clusters) and 0.07835 for Granite (6/6). Persistent-minus-scrambled means were 0.00689 for Qwen (4/6; model-specific exact sign-flip `p=0.21875`) and 0.09001 for Granite (6/6; `p=0.015625`). These decompositions were not a second multiplicity family and do not replace the frozen pooled tests. Prespecified block-length and pseudocount sensitivities retain mostly positive directions but show substantial scale dependence; the block-4, pseudocount-1 pooled H3 contrast is 0.00024 nats/update.

H4 is a fixed-window decline, not proof of complete return. All six Qwen field-Markovized trajectories re-entered their training-nominal thresholds six sweeps after restoration. Only one of six Granite trajectories re-entered by sweep 45; five retained final-five means above their model-specific training thresholds. The result therefore supports movement toward the restored nominal regime while preserving slower or incomplete Granite recovery as a boundary finding.

## V14 scientific correction

No frozen V14 decision or trajectory was altered. The versioned V14 audit now fits recovery thresholds using training clusters only, completes the frozen 10,000-replicate cluster-preserving permutation analysis, recomputes three-, five-, and seven-sweep nominal geometries, deletes individual observables, and audits finite-sample dependence bias. The historical H3 maximum-minus-final estimand, interval, raw p-value, and Holm value remain archived, but its structurally nonnegative sign makes the directional test invalid. Its machine-readable disposition is `inferential_support: false`; recovery language uses threshold re-entry, final residual, the complete path, and fixed early-versus-late descriptive changes.

## Supported boundaries

Results are finite-size and model-specific. Neither model is a human participant. No field validity, application benefit, controller advantage, performance superiority, thermodynamic-limit phase transition, physical free energy, or exact LLM entropy production is claimed. Persistent history and scrambled history are prompt-format controls; they do not make the projected binary process fully observed. Negative adjusted information quantities are retained rather than truncated.

## Reproduction order

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python scripts/run-statmech-v15-tests.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/audit-statmech-v14.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-pilot.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-pilot.sh granite
scripts/freeze-statmech-v15-protocol.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-formal.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-formal.sh granite
scripts/replay-statmech-v15.sh
scripts/analyze-statmech-v15.sh
scripts/run-statmech-v15-surrogate.sh
scripts/generate-statmech-v15-figures.sh
.venv/bin/python paper/jstat_v15/render_publication_figures.py
scripts/build-statmech-v15-results.sh
scripts/build-statmech-v15-paper.sh
scripts/verify-statmech-v15.sh
```

Raw prompts, completions, and trajectory tables are external at `/workspace/ThermoAgent-v15-artifacts`. Compact aggregate tables, checksums, vector figures, and manuscript sources are repository-facing.

### Source-checksum audit

The frozen checksum enumerator unintentionally included one ignored 224-byte root-level Python bytecode cache. It did not affect source semantics or any outcome, and it is not committed. `reproducibility/source_checksum_audit.json` records its path and digest, the exact reconstruction of the legacy frozen checksum, and the cache-free semantic-source checksum. A clean checkout should run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python results/collective_agent_statmech_v15/reproducibility/verify_source_checksum.py
```

The command also writes `reproducibility/verification_clean.json`, which requires every non-source package check from the legacy verifier to pass and substitutes only the documented reconstruction for the absent ignored cache. The legacy `scripts/verify-statmech-v15.sh` source-equality check applies to the retained exact execution tree; all other package checks and the cache-free provenance audit are reproducible from the repository-facing package and external manifests.
