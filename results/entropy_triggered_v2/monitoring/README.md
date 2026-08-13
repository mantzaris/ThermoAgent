# Monitoring validation against ordinary logistics indicators

This retrospective gate uses exactly one frozen `scripted_independent` trajectory per application/scenario/seed, avoiding pseudo-replication of the same exogenous panel across methods. Original-main estimates are leave-one-environment-seed-out. Models and nominal 95th-percentile thresholds are refit without the held seed. The already seen original holdout is external diagnostic evidence only.

All detector outputs use distributed entropy for deployable entropy rows. `exact_entropy_evaluator_only` is explicitly non-deployable. Entropy sketches are not free and this diagnostic does not make a communication-efficiency claim.

## Overall results

### Original main, out of seed

| Application | Detector | Prevalence | AP | AP/prevalence | ROC AUC | Recall | False alarm |
|---|---|---:|---:|---:|---:|---:|---:|
| commercial | `KPI_CUSUM` | 0.622 | 0.819 | 1.32 | 0.740 | 0.340 | 0.026 |
| commercial | `Page_Hinkley` | 0.622 | 0.894 | 1.44 | 0.872 | 0.157 | 0.020 |
| commercial | `multivariate_KPI_logistic` | 0.622 | 1.000 | 1.61 | 1.000 | 1.000 | 0.022 |
| commercial | `multivariate_KPI_plus_entropy_logistic` | 0.622 | 1.000 | 1.61 | 1.000 | 1.000 | 0.046 |
| commercial | `operational_entropy_absolute_deviation` | 0.622 | 0.797 | 1.28 | 0.860 | 0.996 | 0.237 |
| commercial | `operational_entropy_high` | 0.622 | 0.851 | 1.37 | 0.672 | 0.625 | 0.151 |
| commercial | `operational_entropy_low` | 0.622 | 0.568 | 0.91 | 0.328 | 0.371 | 0.156 |
| humanitarian | `KPI_CUSUM` | 0.622 | 0.816 | 1.31 | 0.758 | 0.208 | 0.026 |
| humanitarian | `Page_Hinkley` | 0.622 | 0.912 | 1.47 | 0.900 | 0.182 | 0.020 |
| humanitarian | `multivariate_KPI_logistic` | 0.622 | 1.000 | 1.61 | 1.000 | 1.000 | 0.024 |
| humanitarian | `multivariate_KPI_plus_entropy_logistic` | 0.622 | 1.000 | 1.61 | 1.000 | 1.000 | 0.024 |
| humanitarian | `operational_entropy_absolute_deviation` | 0.622 | 0.868 | 1.39 | 0.877 | 0.878 | 0.248 |
| humanitarian | `operational_entropy_high` | 0.622 | 0.896 | 1.44 | 0.765 | 0.629 | 0.059 |
| humanitarian | `operational_entropy_low` | 0.622 | 0.564 | 0.91 | 0.235 | 0.250 | 0.213 |

### Seen original holdout, diagnostic only

| Application | Detector | Prevalence | AP | AP/prevalence | ROC AUC | Recall | False alarm |
|---|---|---:|---:|---:|---:|---:|---:|
| commercial | `KPI_CUSUM` | 0.682 | 1.000 | 1.47 | 0.999 | 0.792 | 0.000 |
| commercial | `Page_Hinkley` | 0.682 | 0.997 | 1.46 | 0.993 | 0.400 | 0.000 |
| commercial | `multivariate_KPI_logistic` | 0.682 | 1.000 | 1.47 | 1.000 | 1.000 | 0.000 |
| commercial | `multivariate_KPI_plus_entropy_logistic` | 0.682 | 1.000 | 1.47 | 1.000 | 1.000 | 0.357 |
| commercial | `operational_entropy_absolute_deviation` | 0.682 | 0.469 | 0.69 | 0.000 | 1.000 | 1.000 |
| commercial | `operational_entropy_high` | 0.682 | 1.000 | 1.47 | 1.000 | 0.000 | 0.000 |
| commercial | `operational_entropy_low` | 0.682 | 0.469 | 0.69 | 0.000 | 1.000 | 1.000 |
| humanitarian | `KPI_CUSUM` | 0.682 | 0.998 | 1.46 | 0.995 | 0.842 | 0.000 |
| humanitarian | `Page_Hinkley` | 0.682 | 0.999 | 1.47 | 0.998 | 0.633 | 0.000 |
| humanitarian | `multivariate_KPI_logistic` | 0.682 | 1.000 | 1.47 | 1.000 | 1.000 | 0.000 |
| humanitarian | `multivariate_KPI_plus_entropy_logistic` | 0.682 | 1.000 | 1.47 | 1.000 | 1.000 | 0.429 |
| humanitarian | `operational_entropy_absolute_deviation` | 0.682 | 0.470 | 0.69 | 0.000 | 1.000 | 1.000 |
| humanitarian | `operational_entropy_high` | 0.682 | 1.000 | 1.47 | 1.000 | 0.000 | 0.000 |
| humanitarian | `operational_entropy_low` | 0.682 | 0.470 | 0.69 | 0.000 | 1.000 | 1.000 |

## Incremental-value interpretation

The definitive values are in `incremental_value.csv`. Positive AP/AUC/$R^2$ differences favor adding entropy; negative RMSE and Brier differences favor it. The ordinary model already includes current service loss, backlog/unmet-need pressure, impairment, communication volume, rolling moments, and EWMA, making this a stringent incremental test.

The full evaluator-KPI classifier is already perfect on both stages, so entropy adds exactly zero AP or ROC AUC and cannot claim independent disruption-classification value there. The high-direction entropy ranking is perfect on the seen holdout, but its main-derived nominal threshold has zero recall; absolute-deviation calibration reverses and false-alarms on every holdout-negative timepoint. This is a threshold-transfer warning, not positive confirmation.

- commercial, original_holdout_diagnostic: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `+0.0003`.
- humanitarian, original_holdout_diagnostic: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `+0.0039`.
- commercial, original_main_oof: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `-0.0001`.
- humanitarian, original_main_oof: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `-0.0002`.
- commercial, original_holdout_diagnostic: adding entropy changes future-loss RMSE by `-0.0250`, $R^2$ by `+0.2540`, and Spearman correlation by `+0.0060`.
- humanitarian, original_holdout_diagnostic: adding entropy changes future-loss RMSE by `-0.0100`, $R^2$ by `+0.0589`, and Spearman correlation by `-0.0000`.
- commercial, original_main_oof: adding entropy changes future-loss RMSE by `-0.0033`, $R^2$ by `+0.0448`, and Spearman correlation by `+0.0340`.
- humanitarian, original_main_oof: adding entropy changes future-loss RMSE by `-0.0013`, $R^2$ by `+0.0167`, and Spearman correlation by `-0.0025`.
- commercial, original_holdout_diagnostic: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `-0.0148`.
- humanitarian, original_holdout_diagnostic: adding entropy changes AP by `+0.0000`, ROC AUC by `+0.0000`, and Brier score by `-0.0197`.
- commercial, original_main_oof: adding entropy changes AP by `+0.0974`, ROC AUC by `+0.1714`, and Brier score by `-0.1522`.
- humanitarian, original_main_oof: adding entropy changes AP by `+0.0980`, ROC AUC by `+0.1696`, and Brier score by `-0.1579`.
- commercial, original_holdout_diagnostic: adding entropy changes future-loss RMSE by `+0.0095`, $R^2$ by `-0.2294`, and Spearman correlation by `+0.0140`.
- humanitarian, original_holdout_diagnostic: adding entropy changes future-loss RMSE by `+0.0033`, $R^2$ by `-0.0557`, and Spearman correlation by `+0.0193`.
- commercial, original_main_oof: adding entropy changes future-loss RMSE by `-0.0009`, $R^2$ by `+0.0208`, and Spearman correlation by `+0.1351`.
- humanitarian, original_main_oof: adding entropy changes future-loss RMSE by `-0.0029`, $R^2$ by `+0.0637`, and Spearman correlation by `+0.1943`.

Rows prefixed `restricted_local_` compare each independent agent's private local KPI vector with the same vector plus that agent's final-round distributed entropy estimate. They test entropy as a privacy-preserving compressed system statistic after the full evaluator-KPI model has already shown no classification increment.

Results are reported separately by connected, degraded, and partitioned communication and by isolated, correlated, compound, and nominal regimes in `monitoring_baselines.csv`. Episode-level detection timing is in `detection_lead_time.csv`; entropy-surprisal and ordinary-impairment localization are in `localization.csv`.

## Reproduction

```bash
./scripts/run-monitoring-validation-v2.sh
```
