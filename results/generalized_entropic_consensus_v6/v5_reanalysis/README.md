# V5 abstention and implementation audit

Status: **post-development V5 reanalysis for V6 design only**. This analysis
does not reopen V5 validation, alter a V5 artifact, or change V5's no-go gate
disposition at commit `c895235d02dd05ccc9315621d818def9345a398c`.

## Why the original safety comparison was confounded

V5 compared a safe policy that required both positive cross-fitted predicted
action value and consensus confidence of at least 0.42 with a policy that
removed the confidence threshold *and forced selections with nonpositive
predicted value*. The contrast therefore changed two mechanisms at once.

This addendum reuses the exact frozen V5 candidate rows and cross-fitted
`kpi_plus_entropy_disagreement` scores over 80 private-fragmented panels per
application in the partition, telemetry-integrity, compound, and OOD regimes.
The independent unit remains the complete environment panel. All intervals
use 10,000 paired panel bootstrap replicates with fixed seed 66051.

## Fair-policy result

The same-score/no-consensus policy retains the positive-value rule and removes
only the 0.42 consensus threshold. Relative to it, the safe policy selected
fewer actions in every application. Humanitarian harmful-action rate fell by
2.5 percentage points at the panel level (95% CI -5.0 to -0.625 points), and
utility-restoration harm fell by 6.25 points (95% CI -11.875 to -1.875). The
commercial interval crossed zero. These are lower-harm results at lower
coverage, not a coverage-controlled safety effect.

The coverage-matched comparator removes the consensus threshold and chooses a
development-calibrated positive score threshold that exactly matches the
safe policy's total selected-action count in every application. At matched
coverage, the safe-minus-comparator harmful-rate interval crossed zero in
humanitarian logistics (-2.5 points; 95% CI -6.25 to +0.625) and touched zero
in utility restoration (-5.0 points; 95% CI -10.625 to 0.0). Causal-utility
intervals also crossed zero. Thus the original V5 Gate 8 observation does
**not** survive as a confirmed coverage-controlled effect.

The mandatory-intervention comparison reproduces the original totals (35 to
19 harmful actions in humanitarian logistics and 29 to 9 in utility
restoration), confirming that much of the headline reduction was attributable
to refusing nonpositive-score selections. The operator-budget-matched
escalation policy uses the same positive scores and two-case panel budget,
routes low-consensus selections into an explicitly accounted bounded queue,
and adds two messages/512 bytes per escalated case. It is a transparent
post-development accounting analysis, not a dynamic V5 rerun.

## Files

- `abstention_policy_panel_results.csv`: one row per policy and independent
  panel, including coverage, harm, utility, service loss, queue, and traffic.
- `abstention_policy_summary.csv`: application-policy totals and means.
- `paired_bootstrap_intervals.csv`: paired safe-minus-comparator intervals.
- `coverage_matching_calibration.csv`: post-development thresholds and exact
  action-count matching.
- `qwen_effect_audit.csv`: effect-sign and causal-utility metrics missing from
  the original Qwen summary; accepted actions are explicitly not assumed to be
  material or beneficial.
- `implementation_audit.csv`: claim-impact audit and required V6 repairs.
- `abstention_reanalysis.json`: machine-readable provenance and summary.

Exact command:

```bash
./scripts/run-v6-v5-reanalysis.sh
```

