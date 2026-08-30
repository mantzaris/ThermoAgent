# Validation and correction record

## Immutable scientific reference

Before consolidation, the repository recorded SHA-256 hashes for the manuscript
sources and PDF, all 14 publication figures, all 14 figure-source CSVs, and all
final aggregate/statistical tables. It also recorded the 15 displayed equation
blocks and machine-readable hypothesis dispositions. Publication consolidation
allows path and identifier changes only; retained numerical tables and figure
source contents must match those hashes.

## Confirmatory results retained verbatim

| Hypothesis | Estimate (95% CI) | Unit | Disposition |
|---|---:|---|---|
| Granite field minus nominal peak departure | 42.263 (24.316, 59.303) | nominal-distance units | supported |
| persistent minus Markovized adjusted reversal divergence | 0.05438 (0.03459, 0.07548) | nats per attempted update | supported pooled contrast |
| persistent minus scrambled adjusted reversal divergence | 0.04845 (0.02190, 0.07526) | nats per attempted update | supported pooled contrast |
| early minus late restoration distance | 52.541 (35.406, 68.673) | nominal-distance units | supported fixed-window contrast |

The third result is heterogeneous: the Qwen mean is about 0.00689 with four of
six positive clusters and a descriptive exact value of 0.21875, whereas the
Granite mean is about 0.09001 with all six positive. It is not described as
independent confirmation in both families. At block length four and
pseudocount one, the pooled contrast is approximately 0.00024 nats per update
and the Granite component is slightly negative.

## Corrected-quench audit

The recovery threshold is learned from training clusters only within each
leave-one-cluster-out fold. A historical `maximum minus final` statistic is
structurally nonnegative for nearly every nonconstant trajectory and therefore
cannot support a directional relaxation test. Its numerical record is retained,
but its inferential disposition is false. Recovery statements instead use the
time-resolved path, training-only threshold re-entry, final residual distance,
and fixed early-versus-late windows.

The delayed audit also retains 10,000 cluster-preserving permutations with the
entire preprocessing/classification pipeline refit inside each permutation,
three-, five-, and seven-sweep nominal geometries, single-observable and
observable-family deletions, and finite-sample information-estimator checks.
The full representation outperformed order-only features but not the simple
uncertainty representation.

## Required checks

```bash
scripts/run-tests.sh
scripts/generate-figures.sh
scripts/build-jstat-paper.sh
scripts/verify-jstat-paper-assets.sh
scripts/verify-source-checksum.py
git diff --check
```

Validation additionally checks the equation count, figure count and hashes,
source-data and aggregate hashes, model revisions, absence of raw generations
and weights, PDF font embedding and text extraction, and extracted manuscript
text equivalence.
