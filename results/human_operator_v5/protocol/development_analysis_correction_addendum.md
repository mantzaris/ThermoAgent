# Development analysis correction addendum

This addendum does not change any scientific hypothesis, feature, threshold,
seed, endpoint, or primary development result.

During the first complete 199-replicate refit-permutation execution, the
application helper was found to fit only non-nominal panels. The primary
cross-fitted action-value pipeline had instead fit all development regimes and
evaluated the prespecified primary contrast on non-nominal panels. The two
inconsistent outputs are retained in `superseded/` and are ineligible.

The corrected helper fits every grouped fold over all development regimes,
permutes the thermodynamic block within the same frozen application,
information, regime, and KPI-severity strata, and excludes nominal panels only
when computing the final paired gain. It retains 199 replicates and seeds 55092
and 55093. This secondary falsification repair was made after development
primary outcomes were known, is labeled as such, and cannot unlock validation:
Gate 5 already failed from negative primary point estimates and confidence
intervals.

The corrected humanitarian run recovered the primary disrupted-panel gain of
`-0.0116410`; its permuted refit mean was `+0.0025069` with a 95% empirical
range `[-0.0096317, 0.0180420]`. The utility-restoration run recovered
`-0.0099697`; its permuted mean was `+0.0010033` with range
`[-0.0116016, 0.0120546]`. Because both true gains are negative, the
positive-effect fraction statistic is undefined by design. These results
reinforce rather than rescue the no-go decision.
