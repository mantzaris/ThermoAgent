# V11: evidence-grounded LLM belief qualification

V11 asked whether independent Qwen agents use locally generated and delivered
probabilistic evidence well enough to support a later entropy-production study.
It is a prospectively stopped qualification study, not an LLM-network
entropy-production result.

## Disposition

The frozen qualification gate **failed**. The formal decentralized-network
stage was therefore not run.

Delivered evidence produced a positive pooled change in the reported belief
coordinate: the cluster-mean signed log-odds effect was `0.1343` (95% cluster
bootstrap CI `0.0915` to `0.1811`; 10,000 replicates; 48 independent matched
information clusters). The placebo-adjusted estimate was also `0.1343` (95% CI
`0.0905` to `0.1816`). This effect replicated when averaged separately within
route viability (`0.1886`, CI `0.1164` to `0.2684`; 24 clusters) and repair
hypotheses (`0.0799`, CI `0.0445` to `0.1197`; 24 clusters).

The response was not usable as the frozen transition kernel. Mean signed
effects over reliabilities 0.55, 0.65, 0.75, and 0.85 were `0.0463`, `0.1686`,
`0.1779`, and `0.1388`; the fitted normative-LLR slope was `0.04965`, below the
frozen `0.05` minimum, and the decline at 0.85 exceeded the monotonicity
tolerance. The right-choice fraction was `0.9224`, outside the frozen
`[0.10, 0.90]` diversity interval. A clearly post-qualification diagnostic also
found strong direction asymmetry: right-supporting packets produced a mean
signed log-odds change of `0.5574`, whereas left-supporting packets produced
`-0.2939`. That diagnostic did not change the frozen gate; it helps explain the
failure.

Reported probabilities were not calibrated as transition probabilities. Across
432 repeated-information cells (two samples per cell), mean reported
`P(right)` was `0.6697`, empirical right-choice frequency was `0.9225`, Brier
score was `0.0968`, and ECE was `0.2955`.

## Protocol and provenance

- Branch: `evidence-grounded-llm-entropy-v11`
- Immutable parent: V10 commit
  `4d372f00837bf75f90882392a92feac87dbc84b2`
- Frozen protocol: `configs/statmech_v11/qualification_frozen.yaml`
- Protocol version: `v11-qualification-1.0`
- Protocol SHA-256:
  `7bd4082f9d085222e22d195e3ff603f0f76e27c36b347bfc4132fbb164a3d03f`
- Qualification execution source-tree SHA-256:
  `1f7bcd164e4f07f033def27a8236ed0995413b4c12dcd6203549ca08d48d395e`
- Raw artifacts: `/workspace/ThermoAgent-v11-artifacts/` on the existing
  RunPod; a compact local aggregate mirror is under
  `/tmp/ThermoAgent-v11-artifacts/local/`.

V1--V10 files were not modified. V11 remains uncommitted, unstaged, and
unpushed for human review.

## Evidence-generating model

The hidden state is `theta in {left,right}`. An agent receives a private signal
with known reliability `r`, where `P(signal=theta | theta,r)=r`. A typed packet
contains the observation, source, reliability, observation/delivery times,
freshness, and evidence domain. Its normative independent-evidence increment
is `+/- log(r/(1-r))`. Qwen is not forced to reproduce this Bayesian reference.

The response separates continuous `probability_right`, the derived binary
belief, typed action, commitment, and the agent-selected send/abstain tool.
Within a matched cluster, only delivered evidence changes; private state,
template, option order, and inference seed remain fixed. The scheduler supplies
an update opportunity and validates tools but never chooses a belief or action.

## Model and execution

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- NF4 double quantization; BF16 computation
- Transformers `4.55.4`; PyTorch `2.8.0+cu128`; bitsandbytes `0.47.0`
- RTX 4090, Transformers `AutoModelForCausalLM`
- Sampling temperature `0.65`; top-p `0.90`; maximum 300 new tokens
- At most one bounded schema repair

The retained final pilot used 128 requests and 131 model calls. The decisive
qualification used exactly 864 requests, 886 calls, and 48 independent matched
clusters. Qualification accounting was 559,895 prompt tokens, 76,620 generated
tokens, and 1,536.064 seconds of measured generation latency. Retained pilot
plus qualification accounting was 639,368 prompt tokens, 87,726 generated
tokens, 1,017 model calls, and 0.4882 measured generation GPU-hours.
At the repository's previously documented RTX-4090 rate range of USD
0.34--0.69 per GPU-hour, retained generation corresponds to approximately USD
0.17--0.34. This excludes Pod idle time and invalidated calls whose earliest
latency accounting was not retained; it is not presented as an exact invoice.

Among 863 valid qualification responses, the controlled contexts delivered
1,007 evidence packets totaling 99,405 deterministic wire bytes. Agents chose
786 right actions, 58 left actions, and 19 deferrals; 850 selected outgoing
packets passed role and serialization validation. These are qualification
actions, not downstream application episodes or network traffic estimates.

Three invalidated engineering pilots are preserved externally (115 requests,
173 calls). Their early provider did not persist token accounting for responses
that remained invalid after repair, so an exact all-pilot token total cannot be
recovered without pretending re-tokenization equals the original generation
trace. The retained-study numbers above are exact; including recorded valid
invalidated rows gives conservative lower bounds of 702,235 prompt and 98,065
generated tokens. No package was installed. No formal-network calls occurred.
CPU time was not independently metered during the interactive engineering and
analysis work; the final complete 518-test run took 124.727 wall-clock seconds.

## Frozen gate results

| Component | Result | Evidence |
|---|---:|---|
| First-pass validity | pass | 97.45% versus 95% minimum |
| After-repair validity | pass | 99.88% versus 99% minimum |
| Directional effect | pass | estimate 0.1343; positive 95% CI; minimum 0.10 |
| Placebo-adjusted effect | pass | estimate 0.1343; positive 95% CI; minimum 0.10 |
| Semantic replication | pass | positive intervals in both task framings |
| Reliability monotonicity | **fail** | slope 0.04965 and high-reliability decline |
| Transition diversity | **fail** | 92.24% right choices exceeds 90% maximum |
| Order and paraphrase bounds | pass | 0.0594 order effect; 0.0113 paraphrase range |
| Prompt/treatment isolation | pass | paired prompt fingerprints verified |
| Evidence-send actionability | pass | 98.49% accepted outgoing packets |
| Overall progression | **stop** | all components were prospectively required |

## What ran and what did not

Ran: V10 audit, mathematical/reference tests, interface pilots, frozen Qwen
qualification, 10,000-replicate cluster bootstrap analysis, aggregate figures,
PDF QA, and a theory/methodology manuscript draft.

Did not run: formal reciprocal/nonreciprocal LLM networks, transition-current
estimation, trajectory irreversibility, entropy production, Markov-history
adequacy, formal controls/ablations, validation, holdout, or a replication
model. `formal_template.yaml` is an unexecuted prospective template, not a
frozen or completed protocol.

## Supported and prohibited interpretation

Supported: likelihood-bearing messages can shift Qwen's continuously reported
belief on average in two controlled semantic framings. The response is biased,
nonmonotone over reliability, and too choice-degenerate for the prespecified
network estimator.

Unsupported: that V11 measured entropy production, that directed messages
increase LLM trajectory irreversibility, that the observable LLM state is
Markov, that Qwen performs Bayesian integration, or that these simulations
establish operational or human benefit.

## Compact artifacts

- Frozen statistics: `tables/qualification_analysis.json`
- Cluster-level source: `tables/qualification_cluster_effects.csv`
- Post-gate diagnostics: `tables/qualification_exploratory_diagnostics.csv`
- Figure source: `figures/source_data/`
- Vector figures: `figures/pdf/`
- External tree checksums: `reproducibility/external_raw_checksums.json`
- Automated and manual PDF QA: `reproducibility/pdf_qa.json`
- Claims: `CLAIMS_MATRIX.md`
- Paper-facing synopsis: `PAPER_SUMMARY.md`

Raw prompts and generations are intentionally excluded from Git.

## Reproduction order

```bash
PYTHON_BIN=python3 ./scripts/run-statmech-v11-tests.sh
THERMO_V11_ARTIFACT_ROOT=/tmp/ThermoAgent-v11-artifacts/local \
  ./scripts/build-statmech-v11-results.sh
./scripts/run-statmech-v11-paper.sh
```

The Qwen commands below require the existing cached model and external artifact
root. The frozen gate prevents the second command from unlocking formal work.

```bash
THERMO_V11_ARTIFACT_ROOT=/workspace/ThermoAgent-v11-artifacts \
  ./scripts/run-statmech-v11-qualification.sh
THERMO_V11_ARTIFACT_ROOT=/workspace/ThermoAgent-v11-artifacts \
  ./scripts/run-statmech-v11-formal.sh
```
