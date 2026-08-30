> Historical pre-consolidation record. Paths, commands, and internal study identifiers below describe the original execution layout and are not active repository interfaces.

# V12: collective dynamics and pathwise irreversibility in decentralized LLM-agent networks

## Scientific disposition

V12 is a complete prospective formal experiment with actual decentralized Qwen agents. It preserves the V11 qualification-stage no-go and all V1--V11 artifacts. The central positive result is microscopic: after latent-label counterbalancing, delivered neighbor state changed an agent's categorical transition law by 0.08333 per unit neighbor field (95% cluster-bootstrap CI 0.04167 to 0.125; 96 independent information-state clusters). The frozen network-level nonreciprocity hypotheses were not supported. Coupling and decoding-noise controls did, however, change finite-size order, fluctuations, and relaxation.

The paper therefore reports collective stochastic dynamics and carefully qualified coarse-grained irreversibility, not exact thermodynamic entropy production for the LLM process and not a positive nonreciprocity law.

## What ran

- A reciprocal engineering pilot, retained across four attempts; attempt 4 passed the prespecified engineering criteria without examining a nonreciprocity outcome.
- The complete frozen formal design: 44,352/44,352 autonomous decisions in 401 formal units, including 394 dynamic graph panels, six hysteresis trajectories, and one 576-row microscopic response unit.
- A prospectively defined independent orientation/seed replication embedded in the formal grid.
- Exact content-addressed transition regeneration for all 44,352 formal rows.
- Direct Qwen trajectories, a fitted decentralized kinetic-Ising surrogate, and the immutable exact V10 heat-bath reference.
- Sixteen data-derived vector figures and an 18-page JSTAT-oriented manuscript.

No second LLM, real-human study, application-performance experiment, thermodynamic-limit study, separate validation split, or sealed holdout was part of the frozen V12 design. Humanitarian and utility settings are interpretation examples only.

## Agent state, autonomy, and update convention

Agent `i` has local state `x_i=(b_i,a_i,c_i,m_i)`: categorical belief and action `b_i,a_i in {-1,+1}`, bounded commitment/confidence `c_i`, and bounded explicit memory `m_i`. Each persistent agent also owns a private field, workload, inbox, outbox, typed authority, and model context. The global observable state is the concatenation of these recorded local states; evaluator truth and undelivered messages are absent from prompts.

The scheduler selects one update opportunity uniformly at random, delivers permitted messages, validates syntax, and records the model's decision. It never selects or repairs a scientific belief/action on the model's behalf. One sweep is exactly `N` attempted random-sequential agent updates. The independent inferential unit is an information-state cluster for H1 or a matched graph-trajectory cluster for H2--H6—not a token, message, update, or state row.

## Frozen model and network design

- Protocol: `v12-llm-statmech-1.0`.
- Protocol SHA-256: `cee49ca7111b81ae9544a8d32a754ce4c9f15bff16333e98588147a7f7f665b6`.
- Frozen execution-source SHA-256: `0796286362ec4dde0eb4f2dc88ecea4c3bf53859e618aec1928ba8b9e8b0a154`.
- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Backend: Transformers 4.55.4 and PyTorch 2.8.0+cu128; NF4 double quantization with BF16 computation on one RTX 4090.
- Decoding: top-p 0.9; maximum 144 new tokens; the formal default decoding-noise setting is 0.72, with 0.50 and 0.85 in the collective factorial.
- Sizes: `N={3,4}` for transition-current trajectories and `N={8,16}` for collective trajectories.
- Topologies: fixed-degree ring and fixed-degree modular graphs.
- Nonreciprocity: `alpha={0,0.2,0.5,0.8}` with paired forward/transposed orientations. A divergence-free antisymmetric cycle perturbation preserves edge support, unit weighted in/out degree, payload schema, and one opportunity per valid update.
- Collective factorial: coupling `{0.35,0.80}` by decoding noise `{0.50,0.85}`.
- Controls: no-message, reciprocal, orientation reversal, content/time/sender permutation, matched natural-language placebo, persistent memory, ordered/disordered initialization, heat-bath reference, and fitted surrogate.

## Formal sample accounting

| Block | Trajectories/panels | Independent matched clusters | Decisions/updates | Sweeps |
|---|---:|---:|---:|---:|
| Microscopic response | one table | 96 information states | 576 | not applicable |
| Small transition-current network | 56 | 8 | 11,760 | 3,360 |
| Collective network | 224 | 32 | 21,504 | 1,792 |
| Matched memory comparison | 48 | 8 | 4,608 | 384 |
| Relaxation | 48 | 16 | 4,608 | 384 |
| Message controls | 18 | 3 | 864 | 108 |
| Hysteresis | 6 | 2 | 432 | 54 |
| **Total** | **400 network trajectories plus one microscopic unit** | **69 graph/environment clusters plus 96 information-state clusters** | **44,352** | **6,082 network-equivalent sweeps** |

The 394-panel analysis table excludes the six separately analyzed hysteresis panels and includes 43,344 network updates. Adding hysteresis gives 43,776 network updates; adding 576 isolated microscopic decisions gives the exact formal total of 44,352.

## Statistical-mechanics conventions

Primary microscopic states are realized categorical belief--action choices. Self-reported probability is secondary and is not treated as a calibrated transition probability. The symmetric empirical influence layer defines an equilibrium-reference observable

`H_ref = -(J_b/2)b^T W_s b -(J_a/2)a^T D_s a - K a^T b - h^T b - g^T a`,

with frozen `J_b=1`, `J_a=0.65`, and `K=0.8`. It is not literal physical energy and does not establish a Gibbs law for Qwen.

The projected Markov entropy-production statistic uses a regularized transition-current estimator and is reported in nats per attempted update and `N` times that value per sweep. The primary empirical irreversibility endpoint is length-three forward/reverse block KL minus a 200-shuffle finite-sample floor. Because closure and stationarity are imperfect—especially for short collective trajectories—this is described as coarse-grained pathwise irreversibility or a lower-bound diagnostic, not exact total entropy production. Intervals resample independent clusters with 10,000 replicates; secondary families use Holm correction.

## Frozen hypotheses and estimates

| Hypothesis | Estimate | 95% CI / diagnostic | Disposition |
|---|---:|---|---|
| H1, local neighbor response | +0.08333 latent-plus choice per unit field | 0.04167 to 0.125; `n=96` clusters | Supported |
| H2, strong-alpha path irreversibility, small | -0.0002327 nats/update | -0.001441 to 0.001013; `n=8` | Not supported |
| H2, strong-alpha path irreversibility, collective | -0.0007589 nats/update | -0.01115 to 0.009718; `n=32` | Not supported |
| H3, alpha dose slope, small | +0.0007068 | -0.0005011 to 0.002331; monotone 25% | Not supported |
| H3, alpha dose slope, collective | -0.00005391 | -0.01449 to 0.01456; monotone 15.6% | Not supported |
| H4, small quadratic coefficient | +0.006036 | 0.0000748 to 0.01404, but held-cluster RMSE 0.001937 exceeded linear 0.001925 | Not supported |
| H4, collective quadratic coefficient | +0.005473 | -0.03533 to 0.04807 | Not supported |
| H5, collective response | coupling/noise effects below | Holm-adjusted family | Supported as finite-size dynamics |
| H6, orientation/size replication | signs varied | all collective intervals crossed zero | Not supported |

For H5, increasing coupling from 0.35 to 0.80 increased mean absolute belief magnetization by 0.03298 (95% CI 0.00861 to 0.05906), susceptibility by 0.04550 (0.02282 to 0.06769), and integrated correlation time by 2.319 attempted updates (0.781 to 3.881). Increasing decoding noise from 0.50 to 0.85 reduced these by 0.05387 (-0.08354 to -0.02458), 0.07356 (-0.09923 to -0.04658), and 5.990 updates (-7.530 to -4.473), respectively. These are finite-size collective-regime changes, not a thermodynamic phase transition.

## Occupancy, currents, closure, and controls

- Microscopic response: 576 valid rows, exactly 0.5 latent-plus occupancy, and 58 belief transitions in each direction. Held-out log loss was 0.1365 for the logistic kinetic-Ising belief response, better than the prespecified nonlinear and persistence-aware alternatives; the action persistence-aware model had held-out log loss 0.2231.
- The fitted belief response coefficients were 8.32 for private field, 2.26 for delivered neighbor field, 11.85 for previous belief, and -9.15 for previous action. Option order (-0.892), latent mapping (-1.894), and paraphrase (+1.891) remained material microscopic asymmetries.
- Small trajectories occupied only 1.80 categorical projected states and 3.20 transition pairs on average, with belief--action overlap 1.0. This sparse support is a central limitation.
- Collective trajectories occupied 5.26 states and 8.04 transition pairs on average; mean overlap was 0.904, configuration entropy 1.132 nats, mean reference energy was -1.555 per agent, and correlation time was 12.55 attempted updates. The corresponding small-system means were 0.100 nats and -1.295 per agent. These energies use the frozen effective reference and are not physical energy measurements.
- Reciprocal small-system raw block KL was 0, shuffle floor 0.0002775, and adjusted value -0.0002775 nats/update. Reciprocal collective raw block KL was 0.04551, floor 0.08727, and adjusted value -0.04176.
- First-order closure diagnostics were modest in small systems (median conditional mutual information 0.00115 at reciprocal alpha) but less convincing collectively (mean 0.01916; median 0.01228); collective early/late state JS averaged 0.3095. Projected Markov-current estimates are therefore secondary diagnostics.
- The small-system projected Markov-EPR strong-alpha contrast was +0.0009061 nats/update (95% CI approximately 0 to 0.002718); the collective contrast was +0.0004354 (-0.001768 to 0.002387). These do not override the frozen primary pathwise endpoint.
- Persistent memory increased adjusted block irreversibility by 0.01790 nats/update (0.00357 to 0.03399; `n=24`), but this is a secondary coarse-grained memory effect.
- Each control had only three independent clusters. No message-content, time, sender, placebo, or no-message contrast survived Holm correction; these controls are informative but underpowered.
- Six hysteresis trajectories produced descriptive belief-loop areas 2.625--2.969. This is finite-system dynamical hysteresis only.

## Engineering, replay, communication, and compute

- Formal validity: 44,350 first-pass valid responses, two valid after one greedy bounded repair, zero invalid after repair.
- Replay: 44,352 rows in 401 units regenerated exactly; zero mismatches.
- Communication: 34,788 transmitted/delivered model-selected packets and 626,184 complete binary wire bytes; zero peer-private mutations.
- Formal calls/tokens: 44,354 calls, 24,230,610 prompt tokens, 3,543,392 generated tokens.
- All engineering plus formal work: 44,864 decision requests, 44,882 calls, 24,509,789 prompt tokens, and 3,585,830 generated tokens.
- Metered generation time: 20.3182 single-GPU hours total; formal generation plus current load was 20.0923 hours. Analysis used 76.54 CPU-seconds (80.36 wall-seconds). Estimated incremental compute cost was USD 6.91--14.02.
- The first formal analysis invocation failed before emitting outcomes because a frozen bookkeeping field was not copied into panel summaries. External artifacts preserve the failure. `scripts/analyze-statmech-v12-repair1.py` performs only that copy; no trajectory, seed, estimand, statistical rule, or scientific value changed.

## Supported and prohibited interpretations

Supported: independent LLM agents respond causally to delivered neighbor state; their finite networks exhibit measurable order, fluctuations, relaxation, currents, entropy, persistence, and memory effects; and these can be compared to a fitted effective kinetic model under explicit coarse-graining.

Unsupported: a positive nonreciprocity-induced irreversibility effect, monotone dose response, quadratic LLM analogue of V10, orientation/size replication, exact detailed balance, exact LLM entropy production, thermodynamic-limit phase transition, universal exponent, Bayesian rationality, controller superiority, application benefit, or real-human benefit.

## Files and reproduction

- Protocol: `protocol/protocol_frozen.yaml`
- Primary machine-readable results: `statistics/primary_results.json`
- Main tables: `tables/primary_hypotheses.csv`, `tables/panel_statistics.csv`, and `tables/microscopic_models.csv`
- Figures and source data: `figures/pdf/` and `figures/source_data/`
- Replay, compute, external checksums, and PDF QA: `reproducibility/`
- Manuscript: `../../paper/jstat_v12/main.tex` and `../../paper/jstat_v12/main.pdf`

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python THERMO_V12_ARTIFACT_ROOT=/workspace/ThermoAgent-v12-artifacts scripts/run-statmech-v12-tests.sh
THERMO_V12_ENABLE_QWEN=1 scripts/run-statmech-v12-pilot.sh
scripts/freeze-statmech-v12-protocol.sh
THERMO_V12_ENABLE_QWEN=1 scripts/run-statmech-v12-formal.sh
scripts/replay-statmech-v12.sh
scripts/analyze-statmech-v12.sh
scripts/generate-statmech-v12-figures.sh
/workspace/ThermoAgent/.venv/bin/python scripts/finalize-statmech-v12-figures.py
scripts/build-statmech-v12-results.sh
scripts/build-statmech-v12-paper.sh
scripts/verify-statmech-v12.sh
```

Raw prompts, completions, and trajectories remain external at `/workspace/ThermoAgent-v12-artifacts/`. Compact stage-level sizes and SHA-256 tree digests are in `reproducibility/external_artifact_trees.csv`.
