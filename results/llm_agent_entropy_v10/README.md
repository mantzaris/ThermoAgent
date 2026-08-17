# V10: entropy production in nonreciprocal decentralized agents

This compact V10 package is formulation-led. It extends the frozen V9 model
with a near-reciprocal stationary-response derivation and tests whether actual
decentralized Qwen agents satisfy the prerequisite of responding to explicitly
delivered peer evidence. They did not pass that pilot gate, so the much larger
formal LLM transition-kernel and trajectory study was prospectively not run.

## Provenance and evidence boundary

- Parent V9 branch: `statistical-mechanics-agentic-systems-v9`
- Verified pushed V9 commit: `8e8315d25684a1c582c6a7b46fbb5786bc3f0557`
- V10 working branch: `llm-agent-entropy-production-v10`
- Frozen CPU protocol: `v10.1.0-frozen-cpu`
- Protocol SHA-256:
  `fe1b8a599f82da6ce7b69ece6479b90dc7ddcb923b90e8e5a9fd3249be8efbf5`
- Frozen scientific-source SHA-256:
  `e2f566aa52d3728c12796663650b764445bff317bfa45fdf0af5711e024e1c03`
- LLM pilot amendment: `v10.1.3-llm-pilot-amendment`; current amendment SHA-256
  `5c74a592cd3b3e7c8f05bfd8616f4c3ae742380ebac9b98fe34541c29a0970b6`.
- RunPod status: the direct public-IP alias was stale, but the established
  `ssh.runpod.io` proxy reached the same online RTX 4090 Pod. No new Pod was
  created.
- Qwen stage: **384 pilot decisions completed; formal study not unlocked**.
  The final matched delivered-message response was `0.00`, below the frozen
  `0.20` gate.

The exact and pathwise stochastic-agent results are formal numerical evidence.
The Qwen results are retained development-pilot evidence only. They establish
structured decisions and private-evidence response under one pre-final prompt,
but not an empirical transition kernel, time-reversal asymmetry, a fitted
effective-temperature model under the final prompt, or dynamic-network
replication.

## Model and perturbative result

The finite microstate is `x = (b, a)`, with one private belief and one committed
action in `{-1,+1}` per persistent agent. In the reciprocal limit, symmetric
communication and dependency layers, static fields, a common temperature, and
random-sequential heat-bath updates produce a Gibbs-reversible Markov chain.

Directed communication is parameterized as

`A_c(alpha) = A_s + alpha A_a`,

where `A_s` is symmetric and `A_a` antisymmetric. Pairwise support and total
edge weight are fixed. Individual node in/out strength need not be fixed and is
reported as a diagnostic.

For `W_alpha = W_0 + alpha V + O(alpha^2)` and
`pi_alpha = pi_0 + alpha r + O(alpha^2)`, the constrained response is

`r(I-W_0) = pi_0 V`, with `r 1 = 0`.

Writing `q_xy = pi_0x W_0xy`,
`f_xy = r_x W_0xy + pi_0x V_xy`, and `j_xy = f_xy - f_yx`, the discrete-time
Schnakenberg rate has the exact near-reference expansion

`sigma(alpha) = C alpha^2 + O(alpha^3)`,

`C = 0.5 sum_{x,y:q_xy>0} j_xy^2 / q_xy >= 0`.

The linear term vanishes because both current and log affinity vanish at the
detailed-balance reference and begin at first order. Rates are reported in nats
per attempted variable update. One sweep contains `2N` attempted updates.

## Results

The formal CPU study contains 96 primary exact cells, 336 coefficient-grid
cells, 144 exact size cells, 1,280 independent trajectory cells, and 104
synthetic-estimator cells. It completed in 2,213.6 wall-clock seconds; summed
trajectory-cell CPU time was 2,016.98 seconds. The principal results are:

- H1, reciprocal analytical null: passed. The maximum exact reciprocal rate was
  `1.100864565477282e-31` nats per attempted update.
- H2, quadratic perturbative onset: passed. Across eight independent directed
  orientations, mean `C` was `0.1108861` nats per attempted update, with a
  10,000-replicate orientation-bootstrap 95% interval `[0.0964558, 0.1205296]`.
  The maximum relative difference between exact `sigma/alpha^2` and the
  predicted coefficient for `alpha <= 0.02` was `0.00056296` (0.0563%).
- A zero-intercept quadratic fit had held-out orientation RMSE `3.94e-6`, versus
  `7.51e-6` for the zero-intercept linear fit. Allowing both terms estimated a
  negligible linear coefficient (`-1.28e-7`) and did not improve held-out RMSE.
- H3/H4, topology and temperature dependence: numerical and exploratory, not
  a universal scaling law. `C` ranged from `0.0036428` to `0.1662767` across the
  frozen topology/temperature/coupling grid. Its correlation with the chosen
  antisymmetric spectral norm was only `-0.221`, so no simple spectral law is
  supported.
- The mean coefficient decomposed into `0.109269` on belief-flip transitions
  and `0.001617` on action-flip transitions. The action layer therefore carried
  about 1.46% of the coefficient despite the perturbation entering the belief
  communication layer; this is a finite-model coupling effect, not a universal
  fraction.
- In larger pathwise simulations, mean irreversibility was `0.000020`,
  `0.000735`, `0.009005`, and `0.036581` nats/update at `alpha = 0`, `0.10`,
  `0.35`, and `0.65`, respectively. These are finite stationary-path estimates,
  not exact dense-current sums.
- On the known three-state cycle at 100,000 transitions and pseudocount 0.5,
  mean absolute estimator error was `0.004057`; the matched reversible estimate
  was approximately `1e-5`, establishing a finite-sample bias floor.
- H5, LLM local-policy correspondence: development-only and incomplete. In the
  corrected 120-decision evidence pilot, the positive-minus-negative response
  difference was `0.35`, the fitted field slope was `1.2379`, the fitted
  effective decision temperature was `1.6157`, and the option-order coefficient
  was `-0.1163` (inside the frozen `+/-0.15` gate). Because the prompt was later
  clarified for message interpretation and the formal stage stopped, these are
  not a final held-out calibration result.
- H6, LLM time-reversal asymmetry: not tested. With balanced priors, Qwen
  retained the prior in all 48 matched left/right-message pairs before the
  prompt clarification and again in all 48 pairs after it. The final response
  difference and paired switch fraction were both `0.00`, versus required
  minima of `0.20`.
- H7, LLM graph/prompt/seed replication: not tested because H6's prerequisite
  message-use pilot failed and the 13,728-decision formal design remained
  locked.

Full aggregate values and confidence intervals are in
[`tables/principal_results.json`](tables/principal_results.json) and the CSV
tables in this directory.

The closest prior work already includes quadratic near-reciprocal entropy
production in a nonreciprocal kinetic Ising model. V10 does not claim novelty
for the quadratic onset alone. The narrower theoretical addition is the general
finite-kernel coefficient for a coupled belief--action model and its transition-
layer decomposition. This Qwen interface does not support the intended
nonreciprocal LLM realization: delivered messages did not alter the controlled
belief state, so varying directed influence weights would not constitute a
credible communication mechanism.

## LLM-agent architecture

Each Qwen agent has a unique context, private evidence, private memory, local
belief, committed action, inbox, outbox, role, and typed authority. The
scheduler chooses only the update opportunity. The model returns a validated
structured decision; no centralized routine substitutes the selected state.
Natural-language transcripts and per-turn records are external-only.

Five completed pilot attempts produced 384 structured decisions and 388 model
calls including four repairs: 167,031 prompt tokens, 34,224 generated tokens,
and 677.31 seconds (0.188 GPU-hours) of measured generation latency. Including
model loads, allocated GPU time is estimated at 0.23 hours, approximately
USD 0.078--0.159 at the repository's documented USD 0.34--0.69 hourly range.
First-pass validity was 96.7--100% across completed retained pilots and repaired
validity was 100%. The initial GPU attempt made no scientific decision because
PyTorch required `CUBLAS_WORKSPACE_CONFIG`; it is retained as a technical
failure. Two later pilot designs were also retained and excluded because they
aliased evidence with display order or initialized every prior to the same
choice.

The pinned design uses:

- `Qwen/Qwen2.5-7B-Instruct`
- revision `a09a35458c702b33eeacc393d103063234e8bc28`
- NF4 with double quantization and BF16 computation
- inference sampling temperature 0.65 and top-p 0.90
- one bounded repair attempt

Inference sampling temperature is not called the statistical-mechanical
temperature. An effective decision temperature is fitted only if controlled
held-out response curves support a logit approximation.

## Qualified irreversibility terminology

- Small analytical chains: exact stationary Schnakenberg entropy production.
- Fully specified stationary stochastic-agent paths: pathwise stationary
  irreversibility, validated against exact kernels.
- Empirical small LLM kernels: estimated transition-current entropy production,
  conditional on Markov-state adequacy and bias controls.
- Coarsened persistent LLM paths: block time-reversal KL or coarse-grained
  irreversibility lower bound, not automatically full entropy production.

## Compact artifacts

- `tables/`: aggregate numerical results and the literature comparison.
- `figures/source_data/`: exact compact data for each paper-facing figure.
- `figures/pdf/`: canonical vector figures with embedded fonts, including the
  data-derived Qwen pilot boundary result in Figure 8.
- `reproducibility/summary.json`: provenance, tests, environment, compute, and
  stage disposition.
- `reproducibility/external_artifacts.csv`: hashes and sizes for external raw
  aggregate artifacts.

Raw Qwen prompts/completions and GPU logs are never stored here. They remain at
`/workspace/ThermoAgent-v10-artifacts/` on the existing Pod; a compact external
copy of pilot aggregates and a 403-entry checksum manifest is retained under
`/tmp/ThermoAgent-v10-artifacts/`. No raw transcript entered Git-facing files.

Manual 300-DPI inspection found layout defects in the initially generated
Figures 1 and 7, an unexplained sign color in Figure 5, and omitted stored
uncertainty in Figure 6. The post-freeze presentation-only renderer at
`paper/jstat_v10/render_publication_figures.py` corrected those issues from the
unchanged source CSVs. No numerical result or frozen scientific source was
changed. All eight final PDFs then passed opening, embedded-font, text,
rendering, original-size, 300-DPI, clipping, and overlap checks.

## Reproduction order

Run from the repository root:

```bash
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-audit.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-tests.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-development.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-freeze.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-formal.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  MPLCONFIGDIR=/tmp/thermoagent-v10-mpl \
  ./scripts/run-statmech-v10-analysis.sh
MPLCONFIGDIR=/tmp/thermoagent-v10-mpl \
  .venv/bin/python paper/jstat_v10/render_publication_figures.py
./scripts/run-statmech-v10-paper.sh
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
THERMO_V10_EXPORT_ROOT=/tmp/ThermoAgent-JSTAT-v10-clean-export \
  ./scripts/run-statmech-v10-export.sh
```

The retained pilot commands are opt-in and must use the existing authorized
model environment:

```bash
THERMO_V10_ENABLE_QWEN=1 ./scripts/run-statmech-v10-qwen-pilot.sh
THERMO_V10_ENABLE_QWEN=1 ./scripts/run-statmech-v10-qwen-message-pilot.sh
```

The formal Qwen command is intentionally locked by the failed message gate. Do
not bypass it, lower the threshold, or treat the pilot rows as a formal kernel.

## Prohibited claims in this snapshot

- that entropy production was measured in actual Qwen-agent trajectories;
- that the pilot fitted effective decision temperature transfers to the final
  prompt or is stable across held-out paraphrases;
- that the observed coarse variables form a complete Markov state;
- that the LLM result replicates across graph sizes or prompt paraphrases;
- that a thermodynamic-limit phase transition was established;
- that the illustrative humanitarian or defensive-utility mappings establish
  field validity, operational benefit, or human benefit;
- that quadratic onset near reciprocity is itself novel.
