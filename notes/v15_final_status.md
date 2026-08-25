# V15 final scientific status after fresh reconstruction

## Disposition

V15 was reconstructed from the frozen protocol after the previous external
RunPod filesystem had been lost. Both exact pinned model snapshots were
downloaded into a new environment, the 48 formal graph trajectories were
generated again, and all 34,560 retained decisions replayed with zero
mismatches. The reconstructed aggregate science matches the committed V15
reference within the explicitly audited binary64 cross-platform tolerance.

The unit of inference is the complete graph/environment trajectory cluster.
Agents, updates, messages, calls, tokens, and rolling windows are not
independent replicates. All four frozen numerical criteria are met, with the
model and estimator heterogeneity below retained as part of the result.

## Frozen effects

- H1, Granite field minus nominal maximum post-quench distance: 42.26334
  distance units (95% cluster-bootstrap interval 24.31644 to 59.30340), exact
  one-sided sign-flip `p=0.015625`, supported at allocated alpha 0.02.
- H2, persistent minus Markovized adjusted path-reversal divergence: 0.05438
  nats per attempted update (0.03459 to 0.07548), exact `p=0.0004883`, Holm
  `p=0.0009766`, supported.
- H3, persistent minus the synthetic scrambled-history placebo: 0.04845 nats
  per attempted update (0.02190 to 0.07526), exact and Holm `p=0.003418`,
  supported for the frozen pooled estimand.
- H4, fixed recovery sweeps 31-35 minus sweeps 41-45: 52.54060 distance units
  (35.40627 to 68.67316), exact `p=0.0002441`, Holm `p=0.0007324`, supported.

H2 and H3 are heterogeneous. Persistent-minus-Markovized means are 0.03041
for Qwen (5/6 positive) and 0.07835 for Granite (6/6). Persistent-minus-placebo
means are 0.00689 for Qwen (4/6; model-specific exact `p=0.21875`) and 0.09001
for Granite (6/6). The pooled H3 result is not separate confirmation within
both model families and does not support a universal or homogeneous effect.

## Recovery and temporal-asymmetry boundaries

All six Qwen field-Markovized trajectories re-enter their training-only
nominal threshold within the 15-sweep restoration interval. Only one of six
Granite trajectories re-enters by sweep 45. H4 establishes a positive
fixed-window decline in macrostate distance, not complete recovery, return to
equilibrium, or entry into a physical basin.

Arm-level bias-adjusted path-divergence estimates can be negative because a raw
finite-sample divergence can lie below its shuffled floor. H2 and H3 are paired
contrasts between adjusted statistics; they do not establish positive absolute
entropy production. The contrast is sensitive to block length and
pseudocount: at block length four and pseudocount one, pooled H3 is about
0.00024 nats/update and the Granite component is slightly negative. The valid
interpretation is coarse-grained temporal asymmetry at the frozen primary
estimator, not physical dissipation or exact thermodynamic entropy production.

## Secondary statistical-mechanics extension

The frozen protocol and H1-H4 are unchanged. A post-reconstruction descriptive
extension adds three operationally measured finite-system observables:

1. connected belief correlation by actual shortest-path distance;
2. attempted-update magnetization autocorrelation and a prespecified truncated
   integrated autocorrelation time; and
3. the Binder cumulant of the phase-window belief magnetization distribution.

Trajectory-first estimates and cluster intervals show weak, model-dependent
connected-correlation changes. At graph distance one, the Qwen
disruption-minus-baseline contrast is 0.00238 (-0.01750 to 0.02277) and the
Granite contrast is -0.02032 (-0.05486 to 0.00947). The corresponding
recovery-minus-baseline values are 0.00080 (-0.01254 to 0.01453) and -0.02748
(-0.05776 to -0.00346).

The persistent-minus-Markovized restoration-window integrated-autocorrelation
contrast is -2.46995 attempted updates (-5.79214 to 0.17724) for Qwen. Only two
of six Granite contrasts are defined because four Granite phase windows have
zero magnetization variance; their mean is -4.14109 (-19.88286 to 11.60067).
These are nonstationary finite-window persistence summaries, not equilibrium
correlation times or evidence of critical slowing down.

The Qwen field-Markovized Binder disruption-minus-baseline contrast is
-4.80504 (-7.42824 to -2.20832), with a recovery contrast of -3.99611
(-6.95969 to -1.88680). Granite contrasts are 0.16643 (-0.09447 to 0.52063)
and 0.13129 (-0.13195 to 0.50873). The extreme negative Qwen values reflect a
finite-window order-parameter shape with a small second moment and are
sensitive to window and pooling choices. They do not establish a Binder
crossing, critical point, or phase transition.

## V14 correction and effective-model boundary

No V14 raw decision or trajectory changed. Audit version 1.1 excludes the
held-out cluster from threshold fitting, archives the original reports,
reclassifies the structurally invalid historical H3 sign test as
non-inferential, completes 10,000 cluster-preserving full-pipeline
permutations, recomputes 3/5/7-sweep geometries, deletes every macrostate
observable individually, and adds marginal-preserving information-estimator
null floors. All six corrected V14 field paths re-enter the training-only
threshold exactly six sweeps after restoration.

The V13-fitted kinetic surrogate was not refitted to V14 or V15 quench paths.
It generally overstates the direct Qwen shared-coordinate response and shifts
peak timing. The direct field-minus-nominal peak difference is 0.142 in the
shared five-coordinate geometry versus 1.467 for the surrogate; corresponding
energy-entropy route-area differences are 0.043 and 0.695. This is an
out-of-sample boundary on the low-dimensional closure, not a failure of the
direct experiment.

## Reconstructed environment and accounting

The fresh environment used Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
Transformers 4.55.4, bitsandbytes 0.47.0, NF4 double quantization, BF16
computation, decoding temperature 0.5, top-p 0.9, and at most 96 generated
tokens. The GPU was an NVIDIA GeForce RTX 4090. Exact model revisions were
Qwen `a09a35458c702b33eeacc393d103063234e8bc28` and Granite
`51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.

The retained formal trajectories use 34,565 calls, 20,908,194 prompt tokens,
2,893,967 generated tokens, and 48.737223 measured generation GPU-hours:
19.542721 for Qwen and 29.194502 for Granite. Successful pilots add 256
decisions and about 0.349 hours. The incident ledger retains 197 orphan calls
from an interrupted panel plus one post-generation call whose atomic token and
latency record could not be written. Fresh total metered generation is
therefore at least 49.414869 hours, and prompt/generated-token totals are lower
bounds. The estimated measured RTX 4090 cost range is at least USD 16.80 to
34.10. The difference from the original 19.193-hour execution is an
environment-dependent reconstruction cost, not a scientific effect.

The external artifact root is
`/workspace/ThermoAgent-v15-reconstruction-b309f0ab`. Its content-addressed
scientific manifest covers 35,246 files and 208,149,915 bytes with tree
SHA-256 `c9fc7857daa9672b4d255a2ebeb3d2a7f24ed63df8fb6996d0455d768217ea2e`.
The formal tree SHA-256 is
`b884e693fa764183ca36d1220a55237bb9e52e24cd7db382af80a93ebed94444`;
the raw-formal tree SHA-256 is
`215151ff3b38c25f399df01e0e327c9fc08471a959f92e0284af4bde6d13fa55`.
The original deleted external records cannot be digest-compared to the fresh
call files; the committed aggregate reference is the reconstruction target.

## Integrity and presentation

- Frozen protocol SHA-256:
  `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`.
- Frozen execution-source SHA-256:
  `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`.
- Cache-free semantic-source SHA-256:
  `f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2`.
- Final post-formal review/analysis source SHA-256:
  `4f23c83ca08a4f4b11253247bf1534ac047892a9c3f6b8ffcad5b97167473bc8`
  over the 42-file V15 source enumerator.
- Replay: 48/48 trajectories and 34,560/34,560 retained decisions, zero
  mismatches.
- Reconstruction comparison: all aggregate science and accounting checks pass;
  the largest observed cross-platform numerical difference is audited as
  binary64/LAPACK roundoff rather than hidden tolerance.
- Focused isolated regressions: 207/207 passed (51 V15 tests plus 156 executed
  V10-V14 tests; one branch-relative V14 immutability assertion is replaced by
  the stronger V15 base-immutability test).
- Remaining repository and V9 regressions: 449/449 passed. Total available
  executed tests: 656/656; no failures or skips.
- Candidate figures: 14 vector PDFs. Main manuscript: 21 pages. Supplement: 2
  pages. All 16 PDFs and 37 pages pass automated opening, embedded-font,
  rendering, and text-extraction checks. All were inspected at original
  vector size and in external 300-DPI renderings with no material clipping,
  overlap, missing glyph, or unreadable legend.
- Deterministic rebuild: two successive builds produced identical SHA-256
  manifests for 62 science tables, source-data files, vector figures, and
  manuscript PDFs after PDF metadata was pinned to `SOURCE_DATE_EPOCH`.
- Repository-facing V15 results directory: 12,092,875 bytes; V15 paper
  directory: 602,164 bytes. The verifier's complete repository-facing V15
  package is about 13.34 MB across 152 inventoried files. No individual new
  file exceeds 10 MiB.

## Supported and prohibited interpretations

Supported: state-separated, locally informed model instances form a measurable
finite interacting stochastic process; the field reversal produces a
reproducible within-model macrostate departure; genuine history increases the
frozen coarse-grained path-divergence measure relative to paired controls in
the pooled estimand; fixed-window restoration distance declines; connected
correlation, finite-window persistence, and order-parameter shape provide
additional descriptive coordinates; and the effective kinetic closure misses
identifiable direct response features.

Unsupported or prohibited: universality across models; model-homogeneous
memory effects; complete Granite recovery within 15 sweeps; positive absolute
entropy production in every arm; exact LLM entropy production; literal
physical energy or temperature; a thermodynamic-limit phase transition;
critical slowing down; task-performance, controller, human, or
application-benefit claims.

The manuscript is ready for manual scientific review, not submission. Its
strongest remaining limitation is the combination of one finite direct-system
size (`N=16`), one reciprocal modular topology and coupling/noise anchor, two
7B instruction-model families, short nonstationary phase windows, and a
model-heterogeneous memory-control effect.
