# Statistical mechanics of decentralized agentic coordination (V9)

V9 is a formulation-led study of persistent independent agents with coupled
binary beliefs and actions on distinct communication and dependency networks.
It asks when the process has an exact equilibrium reference and how directed
communication, disorder, memory, external drive, and network disruption move it
away from that reference. It does **not** reopen the negative V7/V8 controller
claims, compare entropy dashboards, or claim that entropy improves control.

The immutable parent is V8 commit
`b86f97fa0940f11cb366c809e0e46fa888dfaba1`. All V9 work is intentionally
uncommitted and unpushed for human review.

## Model and evidence boundary

The equilibrium microstate is `x=(b,a)`, where each persistent agent owns a
belief `b_i` and action `a_i` in `{-1,+1}`. The two-layer dimensionless
Hamiltonian is

```text
H = -(J_b/2) b' A_c b -(J_a/2) a' A_d a - K a' b - h' b - g' a.
```

An agent samples a local heat-bath/logit policy from its private fields, paired
variable, and authorized neighbors. The scheduler offers update opportunities
but never chooses an agent's action. The scalar `T` is specified policy noise,
not a temperature estimated from task performance.

The following distinctions are essential:

- **Exact:** the symmetric, static, common-temperature, memoryless limit; its
  Gibbs law, detailed balance, free-energy/KL identity, and finite-state
  stationary entropy production.
- **Approximate:** the homogeneous two-order-parameter mean-field equations and
  the diluted-degree stability boundary.
- **Numerical:** finite-size response functions, Binder curves, relaxation,
  hysteresis, topology/disorder sweeps, and driven mappings.
- **Interpretive only:** humanitarian logistics and defensive cyber-utility
  restoration. They are abstract parameter mappings, not field validation.

## Frozen numerical study

Protocol `v9.1.2` has SHA-256
`8520d841f98722b9759971178bf58d2697d62bf25f9ca36aa5f8c13b59415eab`.
The formal scientific source checksum is
`9845f4d6ef2c9282c696d67468536f1d5d2ae5b0059ce289f0cbdc0fc51f8a00`.
Post-run edits were limited to manuscript text, layout, uncertainty bands, and
QA metadata; no formal numeric output was rerun or changed.

The formal evaluation contains:

- 5 exact equilibrium cells and 56 exact directed-kernel cells;
- 240 finite-size cells (five sizes, six independent realizations per cell);
- 1,440 topology/disorder/communication cells (four independent realizations
  per cell);
- 80 relaxation records and 132 hysteresis branch records;
- 16 driven application trajectories (eight per illustrative mapping) and
  2,560 application-time records.

## Results

The exact equilibrium gate passed. The largest detailed-balance flux residual
was `6.5052e-18`; empirical stationary distributions were within total
variation `0.02186` of their exact Gibbs distributions; the free-energy/KL
closure residual was `1.8631e-15`; and equilibrium entropy production was at
most `1.4213e-31` per update.

Directed communication generated positive probability-current entropy
production. For asymmetry at least 0.5, the mean was `0.05307` per update with a
95% panel-bootstrap interval `[0.04343, 0.06313]` over 24 independent
orientation realizations.

The finite systems showed an ordered-to-fragmented **crossover** near the broad
response peak around `T=1.55`. A median first Binder-intersection candidate was
`T=1.459`, but the four adjacent-size comparisons had 1, 4, 1, and 3 crossings.
The susceptibility peaks were non-monotone at the largest size. Consequently,
these results do not establish a thermodynamic-limit transition, a universal
critical exponent, or a scaling collapse. The fitted peak-growth exponent
`0.251` is descriptive only. Relaxation was long or censored below the boundary
and short above it; this is finite-time persistence rather than an identified
critical-slowing-down exponent.

Prespecified coarse-grained Tsallis and Gini-Simpson measures correlated
`0.951`--`0.986` with Shannon entropy. V9 found no independent generalized-
entropy benefit, no preferred `q`, and no evidence of nonextensive scaling.

The utility mapping exhibited a drive-induced service deficit followed by
recovery; the humanitarian mapping retained an accumulating shortage under its
continuing workload protocol. These different trajectories illustrate how the
same agent microdynamics can be embedded in different drives. They are not a
comparison of application performance. The maximum independently reconstructed
workload/resource conservation residual was `4.4409e-14`.

All 449 collected repository tests passed with no failures, errors, or skips;
27 are V9-focused. The monolithic suite completed in 106.427 seconds and
bounded process-group reruns independently confirmed the collection. Exact
JUnit records remain outside Git and the compact accounting is in
`reproducibility/test_summary.json`. All ten result
figures and the 13-page manuscript were rendered at 300 DPI, checked for
embedded fonts and extractable text, and inspected at original rendered size.

## Supported and unsupported statements

Supported:

- the stated reversible local update has the explicit Gibbs stationary law;
- the exact small-system implementation satisfies detailed balance;
- directed coupling produces positive stationary entropy production in the
  tested finite kernels;
- the tested finite multiplex systems have topology-, communication-, and
  disorder-dependent coordination crossovers and finite-time metastability;
- independent agent state and authority boundaries are executable and tested.

Not supported:

- a thermodynamic-limit phase transition or universality class;
- literal physical temperature, heat, or thermodynamic work in applications;
- fluctuation-theorem, housekeeping/excess, or critical-exponent claims for the
  driven coarse application trajectories;
- a Tsallis advantage over Shannon entropy;
- entropy-controller superiority, real-human evidence, or operational field
  validity.

## Storage and reproduction

Only compact tables, exact figure-source CSVs, vector PDFs, source, tests, and a
checksum summary are repository-facing. Formal aggregates and run manifests
live outside Git at `/tmp/ThermoAgent-v9-artifacts/` because `/workspace` was
not writable in this environment. There are no per-episode ledger files or QA
PNGs in the repository. External file hashes are recorded in
`reproducibility/summary.json`.

```bash
export THERMO_V9_ARTIFACT_ROOT=/tmp/ThermoAgent-v9-reproduction-artifacts
export THERMO_V9_EXPORT_ROOT=/tmp/ThermoAgent-JSTAT-reproduction-export
./scripts/run-statmech-v9-tests.sh
./scripts/run-statmech-v9-pilot.sh
./scripts/run-statmech-v9-formal.sh
./scripts/run-statmech-v9-analysis.sh
./scripts/run-statmech-v9-export.sh
```

The full working paper is in `paper/jstat_v9/`. Tables in `tables/`, figure
source data in `figures/source_data/`, and the ten inspected vector figures in
`figures/pdf/` can be regenerated from the external formal aggregates.
The tested clean export contains 72 files and is approximately 12.5 MB at
`/tmp/ThermoAgent-JSTAT-clean-export/`; its machine-readable inventory includes
the SHA-256 of every exported file and confirms that it is not a Git repository.
