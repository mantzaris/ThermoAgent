# Frozen-v1 holdout tie diagnosis

This is a derived, retrospective analysis of immutable v1 artifacts. The original holdout is seen and is not eligible for v2 selection or confirmation. No v1 file was changed.

## Findings

- All 16 ThermoAgent/no-entropy matched pairs are bit-for-bit equal in the raw primary outcome; the equality is not a reporting-rounding artifact.
- Exogenous cumulative-demand and service trajectories are exactly equal in every pair. Three successful ThermoAgent material calls moved resources only to intermediate nodes; zero successful material call in either method reached a demand node. Most material calls failed route or capacity validation.
- Commercial: options differ on 58.2% of common agent-decision epochs and 72.8% of union epochs; total counted messages are 7416 versus 917 because ThermoAgent includes 6206 entropy sketches. LLM calls are 743 versus 701.
- Humanitarian: options differ on 46.2% of common agent-decision epochs and 65.8% of union epochs; total counted messages are 6904 versus 883 because ThermoAgent includes 5782 entropy sketches. LLM calls are 716 versus 713.
- Entropy inputs are neither constant nor outside their designed numerical bounds. The checkpoint is behaviorally sensitive to them: zeroing the six monitor fields while holding each recorded action mask fixed changes deterministic ThermoAgent choices at the rate reported in `feature_usage.csv`. Exact training-trajectory feature ranges were not retained by v1, so that narrower question cannot be reconstructed without rerunning training; `feature_usage.csv` instead reports the explicit design bounds and the observed v1-main range.
- Action masks are not the main explanation: singleton masks are rare/absent and the policies still diverge substantially. Mask-level diagnostics and raw-versus-masked argmax effects are retained in `action_divergence.csv`.
- V1 used one RL initialization/training seed (`3001`) for each learned checkpoint. Evaluation-seed replication therefore did not provide training-seed replication.

## Causal diagnosis

The tie arose downstream of policy selection. The learned actors and LLM conversations diverged, but their rare material proposals were overwhelmingly invalid or routed to intermediates rather than demand nodes. Consequently the common purpose-specific exogenous RNG streams generated identical demand, and neither policy changed the service trajectory. ThermoAgent paid much higher counted communication cost—especially mandatory gossip sketches—without influencing the primary outcome. This motivates an event trigger, but it also requires v2 to improve operational actionability rather than merely reduce chatter.

## Table definitions

- `holdout_tie_analysis.csv`: one row per matched pair, raw float equality, service/demand trajectory equality, material consequences, communication/inference totals, and SHA-256 provenance.
- `action_divergence.csv`: pair and weighted application summaries. A common epoch exists in both trajectories; a union epoch counts a missing decision as divergence. Simulator-step divergence compares option multisets.
- `communication_divergence.csv`: semantic event signatures by agent-decision epoch. Generated message/commitment/coalition/shipment IDs are excluded so identifier renumbering alone is not counted as behavioral divergence.
- `feature_usage.csv`: observed ranges, saturation, v1-main support comparison, first-layer norms, selected-logit gradients, and a fixed-mask zero-monitor counterfactual.

## Reproduction

```bash
./scripts/run-entropy-trigger-diagnostics.sh
```
