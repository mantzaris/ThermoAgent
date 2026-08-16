# V6 gate disposition and no-go result

V6 is a prospective development-stage no-go. All thresholds below were frozen
before formal development outcomes were opened. The failed gates are not
reinterpreted as near-passes, and validation and holdout remain unopened.

## Gate decisions

| Gate | Decision | Evidence |
|---|---|---|
| 1. Engineering and replay integrity | Pass | Formal/Qwen replay: 4,650/4,650 exact; zero privacy or nonfinite failures; maximum independently reconstructed conservation residual 0. Deliberate accounting-corruption tests fail as intended. |
| 2. Entropy-measure validity | Pass | Bounds, identical/conflicting beliefs, q-to-1 convergence (`1.22e-8` error), and q=2/Gini-Simpson equivalence pass. Maximum univariate entropy AUC is 0.752, below the disguised-label guard. |
| 3. Learnability | Numeric pass; diagnostic caveat | Supervised ceilings and sequential policies distinguish actions from no-action and respect masks. The pooled ceiling reused numeric environment seeds across applications, so its grouped-isolation diagnostic is retained as compromised and cannot rescue later gates. |
| 4. Autonomous-agent validity | Fail | All PPO seeds act diversely, but Qwen humanitarian agents issue no physical actions; utility physical actions are 57.14% harmful; utility first-pass validity is below 98%; no Qwen escalation occurs. |
| 5. Primary selective-safety value | Fail | Private-fragmented harm-rate reduction versus `kpi_confidence`: humanitarian 0.02258 [0.01307, 0.03296], utility 0.01224 [0.00256, 0.02237]. Both are positive but below the frozen 0.03 practical threshold. |
| 6. Utility/service noninferiority and burden | Fail | Service-loss intervals satisfy noninferiority and causal utility improves, but action coverage is below 0.45 and mean escalations exceed 3.5 in both primary applications. |
| 7. Mechanism specificity | Fail | Private-minus-public interaction: humanitarian 0.01908 [0.00834, 0.02990], below 0.02; utility 0.00916 [-0.00147, 0.02001], not supported. Public actions retain nonzero effects. |
| 8. Communication feasibility | Pass | Event-triggered sketches reduce total messages 51.7–52.2% and bytes 48.4–48.8% versus always-on while retaining the frozen estimation-error bound. Partition MAE rises predictably to 0.1680. |
| 9. Multi-seed stability | Fail | All 25 runs complete with no seed removal or universal-action collapse, but combined-controller between-seed harm SD is 0.09382, above 0.08. |
| 10. Cross-application replication | Fail | Gate 5 does not pass in either required primary application. |

## Primary development effects

At the frozen matched operating point, the combined generalized-entropic
controller has positive paired harm reduction and causal utility relative to
the strongest non-entropic comparator, `kpi_confidence`:

| Application | Harm-rate reduction (95% CI) | Causal-utility gain (95% CI) | Relative service-loss change (95% CI) |
|---|---:|---:|---:|
| Humanitarian | 0.02258 [0.01307, 0.03296] | 0.09874 [0.05471, 0.14323] | -0.01748 [-0.03950, 0.00414] |
| Utility restoration | 0.01224 [0.00256, 0.02237] | 0.05111 [0.00793, 0.09564] | -0.00865 [-0.02524, 0.00823] |
| Commercial boundary | 0.01731 [0.00888, 0.02588] | 0.06562 [0.02297, 0.10934] | -0.00900 [-0.03047, 0.01449] |

These effects are suggestive development observations. They do not satisfy the
prospective practical threshold or fragmentation-mechanism requirement.

The development-only entropy-family procedure selected `q=0.5`. Full-refit
permutation tests used 200 permutations in each primary application. The
family-level comparison is supported only in humanitarian logistics
(`p=0.00498`, Holm-adjusted `0.00995`), not utility restoration (`p=0.07960`).
No cross-application generalized-entropy superiority is supported.

## Scientific disposition

The system demonstrates that generalized entropy and disagreement can be
computed privately, estimated over an ad-hoc network, and used in a complete
dynamic selective-autonomy pipeline. It does not establish the intended
cross-application safety mechanism, stable learned replication, or qualified
Qwen autonomy. This is adequate as an engineering and boundary demonstration,
not as confirmatory journal or Artificial Intelligence evidence.
