# V9 research log: statistical mechanics of decentralized agentic systems

## 2026-08-17 — provenance and storage boundary

- Fetched `origin` and verified local `HEAD` and
  `origin/entropy-triggered-belief-monitoring-v8` both resolved to
  `b86f97fa0940f11cb366c809e0e46fa888dfaba1`.
- The V8 branch had a clean worktree, an empty Git index, and a clean
  `git diff --check`. V1–V8 result paths were not edited.
- Created the permitted local branch
  `statistical-mechanics-agentic-systems-v9`. This branch will remain
  uncommitted and unpushed for human review.
- The requested `/workspace` root was present only as an unwritable path in
  this execution environment (`Permission denied`, including for the approved
  operation). Under the user's explicit fallback allowance, raw artifacts,
  scratch files, and rendered QA images are stored in
  `/tmp/ThermoAgent-v9-artifacts/` and `/tmp/ThermoAgent-v9-scratch/`. The lean
  export is stored in `/tmp/ThermoAgent-JSTAT-clean-export/`.
- Added narrowly scoped V9 ignore rules before running experiments. No raw
  trajectories, logs, arrays, archives, model files, or QA PNGs are written to
  the repository.

## Formulation decision (before numerical evaluation)

The primary microstate is

\[
x=(\boldsymbol b,\boldsymbol a),\qquad b_i,a_i\in\{-1,+1\},
\]

with persistent but equilibrium-disabled local memory and workload variables
available to driven extensions. The two symmetric multiplex layers are a
communication adjacency matrix \(A^{(c)}\) and an operational-dependency matrix
\(A^{(d)}\). The dimensionless equilibrium Hamiltonian is

\[
H(x)=-\frac{J_b}{2}\boldsymbol b^\top A^{(c)}\boldsymbol b
     -\frac{J_a}{2}\boldsymbol a^\top A^{(d)}\boldsymbol a
     -K\boldsymbol a^\top\boldsymbol b
     -\boldsymbol h^\top\boldsymbol b
     -\boldsymbol g^\top\boldsymbol a.
\]

`T` is the stochasticity of each agent's logit/heat-bath decision policy, not
an inferred physical temperature. A randomly selected agent-variable is set to
\(+1\) with probability

\[
P(s_i=+1\mid\mathcal F_i)=\{1+\exp[-2\ell_i/T]\}^{-1},
\]

where the local field \(\ell_i\) uses only its layer neighbors, own paired
variable, and local field. Symmetric static couplings, common `T`, no memory,
no delay, and no drive define the reversible reference. Directed communication,
memory, workload injection, changing fields, and partitions are explicitly
nonequilibrium extensions.

The exact detailed-balance identity follows because for any single-spin pair
\(x,y\),

\[
\frac{W_{xy}}{W_{yx}}=\exp[-(H(y)-H(x))/T],
\]

so \(\pi_xW_{xy}=\pi_yW_{yx}\) for
\(\pi_x=Z^{-1}\exp[-H(x)/T]\). A chromatic block-Gibbs sampler is used for
large systems: every color class is an independent set, so its parallel update
is the exact conditional Gibbs update. Random color order reduces scan-order
artifacts while preserving the Gibbs invariant measure.

The homogeneous mean-field fixed point is

\[
m_b=\tanh[(J_b z_c m_b+K m_a+h)/T],\qquad
m_a=\tanh[(J_a z_d m_a+K m_b+g)/T].
\]

The disordered state loses linear stability when the largest eigenvalue of
\(\begin{psmallmatrix}J_bz_c&K\\K&J_az_d\end{psmallmatrix}/T\) reaches one.
Quenched communication dilution therefore enters at leading order through
\(z_c\mapsto p_c z_c\). This is an approximation, not an exact critical point
for finite heterogeneous graphs.

For a finite stationary Markov kernel the exact total entropy-production rate
is evaluated with the Schnakenberg current expression. For driven large-system
application traces, the logged transition affinity is reported separately and
is not promoted to a fluctuation-theorem result.

## Independence boundary

Each executable agent owns a private belief, action, memory, workload, field,
commitments, inbox, and outbox. It receives immutable local views and delivered
messages. The scheduler chooses an update opportunity and applies the returned
local decision; it does not select a global action. Tests counterfactually
change one agent's private observation and verify that another private vault is
unchanged. Partitions prevent message delivery.

## Development pilot and freeze rationale

The retained pilot used 16 cells at `N={32,64}`, four temperatures, and two
communication availabilities. Mean absolute order ranged from `0.1111` to
`0.8559`; the maximum integrated autocorrelation time was `14.6443` retained
samples. A second diagnostic around the transition showed the main crossover
between approximately `T=1.4` and `T=2.2`, with noisy finite-size Binder
crossings. These values were used only to select the frozen grid and longer
sampling; they are not formal results.

The initially frozen protocol was `v9.1.0`. Before opening any formal outcome,
the run was stopped during its phase-grid stage because application accounting
did not separately reconstruct injected demand, induced cascade work, completed
work, remaining work, and consumed resources. The partial batch was retained at
`/tmp/ThermoAgent-v9-artifacts/invalidated_v9_1_0_accounting_gap_20260817T1614Z/`.
No formal comparison was analyzed. The repair added independent workload and
resource balances plus a deliberate-corruption negative test, changed every
formal seed to the fresh 19xxxx namespace, and incremented the protocol.

The replacement `v9.1.1` run was also stopped before phase-grid completion when
review found that the corrected residuals were calculated but not persisted by
the formal workflow. That partial batch is retained at
`/tmp/ThermoAgent-v9-artifacts/invalidated_v9_1_1_conservation_persistence_gap_20260817T1620Z/`.
The persistence-only repair added one compact conservation table and the
maximum residual to the principal summary. Formal seeds moved again to the
fresh 29xxxx namespace.

The final replacement frozen protocol is
`configs/statmech_v9/formal.yaml`, version `v9.1.2`.
It prespecifies exact enumeration, five lattice sizes, four structurally
different topology families, communication dilution, quenched private-field
fragmentation, relaxation, hysteresis, exact directed entropy production, and
two application mappings. Formal inference is across independent graph/seed
realizations, never across correlated Monte Carlo samples.

Projected resources before execution:

- 1,680 stationary parameter cells (240 finite-size; 1,440 phase-grid), plus
  100 relaxation cells, 66 hysteresis branch points, 61 exact small-system
  calculations, and 16 application trajectories.
- Approximately 20–30 CPU minutes on the current host; zero required GPU hours.
- Below 100 MB external working storage and below 10 MB repository-facing
  publication material. No LLM calls are required.

## JSTAT preparation decision

Current IOP guidance was checked on 2026-08-17. JSTAT identifies its scope as
statistical physics, and IOP accepts common LaTeX variants; use of the supplied
class is optional. IOP recommends a readable review manuscript with at least
12-point body text, embedded figures/tables, and permanent identifiers such as
DOIs in references. Accordingly, the repository uses a self-contained 12-point
standard LaTeX article rather than downloading an unnecessary class file.

Official sources:

- https://publishingsupport.iopscience.iop.org/journals/journal-of-statistical-mechanics-theory-and-experiment/
- https://publishingsupport.iopscience.iop.org/questions/latex-template/
- https://publishingsupport.iopscience.iop.org/questions/article-format/
- https://publishingsupport.iopscience.iop.org/questions/style-guide-journal-articles/

## Evidence labels

- **Exact:** finite-state Gibbs weights, detailed-balance residuals, stationary
  Markov distributions, equilibrium entropy production, free-energy/KL
  identity, and small-system directed entropy production.
- **Approximate:** homogeneous mean-field fixed point and diluted-degree
  stability prediction.
- **Numerical:** finite-size response curves, Binder intersections, relaxation,
  hysteresis, topology/disorder sweeps, and application trajectories.
- **Interpretive only:** humanitarian and defensive cyber-utility mappings.
  They illustrate parameter meanings and do not establish field validity.

## Formal execution and disposition

The definitive `v9.1.2` batch began at
`2026-08-17T16:21:28.605536+00:00` and completed at
`2026-08-17T16:49:54.665812+00:00`. It used local CPU execution only. The
formal source checksum immediately before outcome analysis was
`9845f4d6ef2c9282c696d67468536f1d5d2ae5b0059ce289f0cbdc0fc51f8a00`,
and the protocol checksum was
`8520d841f98722b9759971178bf58d2697d62bf25f9ca36aa5f8c13b59415eab`.

The frozen scientific results are:

- maximum detailed-balance residual `6.505213e-18`;
- maximum empirical Gibbs total-variation distance `0.0218592`;
- maximum free-energy/KL identity residual `1.863093e-15`;
- maximum equilibrium entropy-production magnitude `1.421336e-31`;
- mean exact entropy production `0.0530671` at communication asymmetry at
  least 0.5, with 95% cluster-bootstrap interval
  `[0.0434300, 0.0631312]` across 24 independent orientations;
- maximum independent application conservation residual `4.440892e-14`.

Every adjacent size pair had at least one numerical Binder intersection, but
the counts were `1, 4, 1, 3`. The first-intersection median was `T=1.458805`.
Susceptibility peaked at `T=1.55` for every tested size, but the peak height was
non-monotone at `N=256`; the log--log slope `0.250553` is therefore descriptive
and not a critical exponent. The honest disposition is a finite-size crossover
with finite-time persistence and hysteresis, not an established
thermodynamic-limit phase transition or universality class.

The minimum correlation between coarse Shannon entropy and the prespecified
generalized measures was `0.950930`. V9 did not demonstrate an independent
Tsallis or Gini-Simpson contribution, did not select a preferred `q`, and did
not establish nonextensive behavior. A finite-sample bias correction was not
separately estimated; equal sampling designs only make the frozen comparisons
internally matched.

The driven mappings remained intentionally illustrative. The utility drive
produced a service-loss pulse and numerical recovery, whereas the continuing
humanitarian workload produced persistent accumulating shortage and deeper
cascades. This difference demonstrates distinct external protocols, not
operational validity or comparative superiority.

## Post-formal presentation corrections

After the formal outputs and principal statistics were frozen, visual review
identified clipped labels, a colorbar/legend collision, an intrusive axis
label, and missing uncertainty shading on response plots. Only
`thermoagent/statmech/figures.py`, manuscript prose, and QA metadata were
changed. The corrections abbreviate diagram labels, move labels and legends,
add uncertainty bands already present in compact source tables, and mark the
500-sweep censoring boundary. They do not change a formal input, aggregate,
estimate, confidence interval, or conclusion. The formal source checksum above
remains the scientific execution checksum; the final presentation-source
checksum is recorded separately in the reproducibility summary.

All ten final PDFs were rendered at 300 DPI outside the repository and reviewed
at original resolution. Fonts were embedded and extractable; no clipping or
overlap remained. The only retained raster copies are external QA artifacts.

## Compute and publication boundary

The definitive run used about 28.4 minutes of single-process CPU wall time.
Including the two retained pre-analysis invalidated partial batches, pilot,
analysis, tests, and rendering, total incremental work was approximately 0.9
single-CPU hours. GPU hours, LLM calls, prompt tokens, generated tokens, and
incremental cloud cost were all zero.

This package is an engineering-complete formulation study and a credible theory
preprint foundation. It is not yet a strong positive JSTAT submission because
the finite-size crossing structure is incoherent, no scaling collapse or
universality analysis is established, entropy-estimator bias is not quantified,
and the application mappings are deliberately low fidelity. A journal version
would benefit from improved equilibration or cluster sampling, denser sizes and
temperatures, independent numerical replication, and a sharper analytical
treatment of the driven extensions.

## Final testing and manuscript QA

The test collector found 449 tests, including 27 V9-focused tests. The
monolithic run completed in 106.427 seconds with all 449 passing and zero
failures, errors, or skips. Its console stream yielded early at 42%, but its
finalized JUnit record removed any ambiguity. Ten bounded, nonoverlapping
process groups independently confirmed the same collection. JUnit files are
external under `/tmp/ThermoAgent-v9-artifacts/qa/`; compact accounting is stored
in `results/statmech_agentic_v9/reproducibility/test_summary.json`.

The final manuscript is 13 A4 pages. It builds without undefined references,
overfull boxes, or unresolved citations. All fonts are embedded and text is
extractable. Every page was rendered at 300 DPI outside the repository and
inspected at original rendered size; no clipping or overlap was observed. The
result is recorded in `reproducibility/manuscript_qa.json`.

The final clean publication export was built at
`/tmp/ThermoAgent-JSTAT-clean-export/` because `/workspace` was not writable.
It contains 72 files totaling approximately 12.5 MB, no Git metadata, no caches, no
PNG files, and no LaTeX intermediates. Its 27 focused tests pass and its
manuscript builds from the exported source. `EXPORT_INVENTORY.json` records the
size and SHA-256 of each exported file and the proposed lean repository layout.
