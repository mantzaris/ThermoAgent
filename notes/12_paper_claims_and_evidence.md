# Paper claims and evidence map

All final ledgers have passed replay and the frozen analysis is complete.
Engineering-supported means a software invariant is tested; it is not a
research-performance claim. “Mixed” does not authorize a positive headline.

| Planned claim | Current status | Frozen supporting artifact(s) |
|---|---|---|
| The implementation enforces independent private state and authority | Engineering-supported | Stage 0 test logs; `results/figures/pdf/system_architecture.pdf`; independence tests in `tests/` |
| Both quantitative simulators conserve material and replay from recorded decisions | Engineering-supported | 1,096/1,096 combined replay report; `results/statistics/failed_episodes.csv` |
| Real independent Qwen agents can offer, reject, counter, fail, replan, and separately join a coalition | Engineering-supported, bounded smoke only | `results/smoke/stage1_v6_invitees_only/stage1_agentic_smoke.json`; retained failed predecessors |
| Distributed monitoring approximates global entropy when connected and degrades under partition | Supported empirically | `distributed_convergence.csv`; `estimator_comparison.csv`; entropy-dynamics PDF |
| Operational entropy recognizes disruption before visible service collapse | Supported as onset detection, not pre-shock prediction | Entropy AP `0.934`, ROC AUC `0.863`, 3.0% false alarms; monitoring and detection statistics |
| The calibrated free-energy gap is a useful high-direction alarm | Unsupported | AP `0.577`, ROC AUC `0.393`, 82.6% false alarms; disrupted-minus-nominal gap negative in both applications |
| Entropic RL improves over parameter-matched non-entropic RL | Mixed, not familywise-confirmed | Main improvements `0.187` and `106.5`, Holm p `0.0856`/`0.0584`; exact holdout ties; primary/holdout/ablation tables |
| Distributed signals add value beyond raw/shuffled/oracle alternatives | Unsupported/inconclusive | Four-seed ablations all Holm-nonsignificant; ablation table and PDF |
| Autonomous agents become more valuable with privacy and misalignment | Unsupported | Every necessity-map cell negative (`-2%` to `-14%`); scenario comparisons and necessity PDF |
| Autonomous coordination offers a useful communication/performance tradeoff | Unsupported | Large sketch/message burden, low coalition precision, and no broad outcome advantage; method summary, Pareto and agentic PDFs |
| A coordination benefit transfers to humanitarian logistics | Mixed in main, unsupported on holdout | Suggestive no-entropy comparison but losses/ties against stronger controls; humanitarian primary/holdout rows |
| Effects survive unseen topology and compound/correlated combinations | Unsupported | No-entropy and scripted exact ties; fixed communication tied or better; holdout tables |

All performance claims use complete episodes and paired environment seeds as
the experimental unit. Monitoring timepoints support detector characterization
only and are not treated as independent logistics-performance replicates.
