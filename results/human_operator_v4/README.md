# ThermoHITL v4: development-stage boundary result

**Evidence status:** development only. All large-scale operators are simulated.
No human participant, validation batch, learned-policy training, or confirmatory
holdout evidence exists in v4.

ThermoHITL v4 asks whether privacy-preserving distributed operational energy,
entropy, belief disagreement, and consensus confidence can help a bounded human
supervisor allocate scarce attention among genuinely independent autonomous
agents. It adds an abstract utility-restoration application to commercial and
humanitarian logistics and tests a causal chain from an authorized operator
view through material or service outcomes.

The prospective study stopped at **Gate 3 (coordination necessity)**. Fixed
communication reduced utility-restoration loss by 4.43%, short of the frozen
5% aggregate threshold, even though its paired absolute effect excluded zero.
Per protocol, later stages were not opened. This is a useful development
boundary result, not confirmatory evidence for the proposed AIJ contribution.

## 1. Provenance and relationship to earlier studies

- Branch: `thermodynamic-human-oversight-v4`.
- Immutable v3 scientific snapshot:
  `3f844966930b1cfb5a43bdf3a4d3e744391d1018`.
- Immutable v2 snapshot:
  `c0aa6fe6c98cbce0cdd5e40a0f720a98f5facbe6`.
- Formal v4 development manifests record execution commit
  `bfd1ea24d6523809f72ec979392022d2ffb6ab0a`, protocol checksum
  `8eb867207ef638aa4b4c774d99cd464a74a3b6b45a821e081bf7c22dcf68b234`,
  and source checksum
  `de34b41d7beda8c54546f9a6d027652ff5f438f0d98789cc0dc30a965e3ae37a`.
- The final documentation/QA commit is the branch tip; embedding its hash in
  itself would be self-referential, so the handoff reports the exact value.
- Frozen v1, v2, and v3 results are not rewritten. V3 remains traceable to its
  pushed snapshot despite LF-only normalization on this descendant branch.

V4 responds directly to v3: v3 showed operator actionability and a promising
humanitarian thermodynamic result but failed commercially. V4 prospectively
treats commercial logistics as a KPI-sufficient boundary application and
requires positive incremental value in humanitarian logistics and utility
restoration. It also requires coordination itself to be consequential before
any confirmatory work.

## 2. Applications and independent-agent architecture

### Commercial supply chain

Suppliers, manufacturers, carriers, warehouses, retailers, and a coordinator
have separate private costs, inventories, capacities, forecasts, utilities,
memories, inboxes, commitments, contexts, and typed authority. The primary
outcome is service-loss AUC.

### Humanitarian relief logistics

NGOs, agencies, transport providers, depots, clinics, communities, and a
coordinator have private need beliefs, resource holdings, mandates, priorities,
and commitments. The primary outcome is cumulative weighted unmet need. This
is an abstract simulator, not a behavioral or ethical validation of real relief
operations.

### Critical-infrastructure utility restoration

Distribution zones, substations/microgrids, crew dispatch, parts depots, mobile
generation/fuel providers, critical-load representatives, and an incident
coordinator act across three coupled layers:

1. an abstract service/power graph;
2. communications and confidence-weighted telemetry; and
3. restoration logistics for crews, parts, fuel, generators, and routes.

Abstract disruptions include physical failures, telemetry-integrity loss,
contradictory observations, partitions, unavailable command channels, resource
database inconsistency, verification delay, correlated failures, and compound
combinations. The code contains no exploit logic, real protocol interactions,
credentials, scanning, or external utility targets. The primary outcome is
cumulative critical unserved-load AUC.

Every organization remains independent: it owns a private observation vault,
memory, utility, commitments, inbox/outbox, planner context, tool permissions,
RNG state, and authority. Information crosses boundaries only through logged
messages, coarse sketches, public topology, or typed operator directives. The
simulator validates actions but does not invent agents' domain choices.

## 3. Thermodynamic observability and operator boundary

Operational energy is a fixed weighted stress aggregate:

`E = .34 service deficit + .18 backlog + .10 lateness + .12 safety stress + .10 failed commitments + .08 congestion + .08 resource scarcity`.

Entropy quantities include belief entropy, alternative entropy, commitment
entropy, distributed sketch entropy, two-sided nominal entropy residual,
slope, and acceleration. Neighboring belief disagreement is bounded
Jensen–Shannon divergence. Consensus confidence and consensus error remain
separate. Free energy (`E - T_eff S`) is exploratory only because earlier
studies did not support it.

The ordinary KPI block retains all raw severity inputs. The key matched design
holds visible KPI severity nearly constant while varying private belief
coherence; thermodynamic features therefore cannot pass merely by relabeling
backlog or service deficit.

The simulated operator has one attention slot and one intervention per
episode, eight estimated minutes per intervention, queueing/workload state,
typed bounded actions, and permission to abstain. Normal views exclude raw
private agent state, evaluator truth, future disruptions, true telemetry-
corruption labels, RNG state, global oracle thermodynamics, and counterfactual
outcomes. Every exact view payload is schema validated, canonically hashed, and
logged before allocation.

## 4. What ran

| Evidence | Design | Episodes | Independent seeds |
|---|---|---:|---:|
| Coordination gate | 3 applications × 5 disrupted regimes × 2 methods | 360 | 12 |
| Human causal-usefulness gate | 3 applications × 5 regimes × 2 methods | 360 | 12 |
| Monitoring/feature blocks | 3 applications × 6 regimes × 2 information conditions | 432 | 12 |
| Trigger feasibility | same matched development grid | 432 | 12 |
| Real-Qwen qualification | 2 per application | 6 | 2 |
| **Total** | deterministic development + real Qwen | **1,590** | stage-specific |

All 1,584 formal deterministic episodes completed. Six real-Qwen episodes
qualified actionability. Retained implementation pilots and their failures are
under `superseded/` and `negative_results/`; they are not formal gate evidence.

### What did not run

Validation, five-seed RL training, and the outcome-sealed holdout were
prospectively not run after Gate 3 failed. Their directories contain explicit
`NOT_RUN.json` and `README.md` records, not placeholder result tables. No v4 RL
seed, validation effect, holdout effect, or holdout Pareto frontier exists.

## 5. Model, hardware, and software

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Immutable revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Quantization: bitsandbytes NF4; BF16 computation.
- Transformers 4.55.4; PyTorch 2.8.0+cu128.
- GPU: one NVIDIA RTX 4090, 24,564 MiB, CUDA driver capability 12.8.
- Remote execution copy: `/workspace/ThermoAgent`.
- Model cache: `/workspace/.cache/huggingface`, outside Git.
- No package change was required for v4.

The six Qwen calls used 3,942 prompt and 328 generated tokens. Recorded model
load plus generation occupied approximately 36.32 seconds, or 0.01009
single-GPU hours (about USD 0.0034 at the project's illustrative USD 0.34/hour
rate). GPU setup checks are not billed as experiment episodes. Deterministic
development used CPU only.

## 6. Prospective gates

| Gate | Result | Development finding |
|---|---|---|
| 1. Engineering integrity | Pass | Complete test suite; exact replay; zero conservation residual |
| 2. Agent actionability | Pass | 100% first-pass validity; 81.94% deterministic accepted-to-service; Qwen 6/6 |
| 3. Coordination necessity | **Fail** | Utility restoration 4.43% aggregate reduction versus 5% target |
| 4. Human causal usefulness | Pass | Humanitarian and utility restoration pass; commercial fails |
| 5. Thermodynamic incremental value | Pass | Humanitarian and utility pass; commercial exact zero gain |
| 6. Trigger feasibility | Pass | Nonzero/timely in every disrupted regime; no nominal/pre-onset false alert |
| 7. Mechanism specificity | Pass | Fragmented-information interaction and permutation falsification pass |

Passing Gates 4–7 does not override Gate 3. All seven were required.

## 7. Development findings

### Coordination necessity: the stopping result

Values below compare fixed communication with no communication; negative
absolute differences favor communication.

| Application | Mean loss: no comm | Mean loss: fixed | Paired difference (95% CI) | Relative reduction | Gate |
|---|---:|---:|---:|---:|---|
| Commercial | 8.131 | 5.700 | -2.431 [-2.953, -1.895] | 24.92% | Pass |
| Humanitarian | 8.913 | 8.084 | -0.829 [-1.222, -0.466] | 8.59% | Pass |
| Utility restoration | 9.619 | 9.135 | -0.484 [-0.771, -0.207] | **4.43%** | **Fail** |

The utility effect is directionally clear but misses the prospective practical
threshold. The threshold was not weakened to 4% after seeing this result.

### Bounded simulated-operator usefulness

ThermoHITL-rule versus autonomy-only reduced loss by 0.102 commercially
(0.91%, CI for the paired difference [-0.258, 0]), 1.959 in humanitarian
logistics (20.96%, [-2.421, -1.488]), and 1.324 in utility restoration
(13.42%, [-1.631, -1.016]). Humanitarian and utility passed the frozen gate;
commercial did not.

The primary human-causal matrix contains 180 simulated interventions totaling
1,440 estimated operator minutes. It records 68 beneficial and zero harmful
interventions, with complete causal chains in 2 commercial, 32 humanitarian,
and 34 utility panels. This does not establish safety or effectiveness for real
human operators.

### Same-information thermodynamic value

At one intervention per matched panel, KPI plus entropy/disagreement versus
KPI-only yielded:

| Application | Absolute causal-utility gain | Cluster-bootstrap 95% CI | Relative gain | Service-loss degradation | Harm increase |
|---|---:|---:|---:|---:|---:|
| Commercial | 0.0000 | [0.0000, 0.0000] | 0.0% | 0.0% | 0 |
| Humanitarian | 0.1589 | [0.0619, 0.2613] | 40.19% | +1.78% | 0 |
| Utility restoration | 0.1787 | [0.1029, 0.2609] | 83.11% | -1.10% | 0 |

Each row uses 60 independent environment clusters, 96 candidate records, and
10,000 fixed-seed cluster-bootstrap replicates. Candidate rows are not treated
as independent. The commercial exact null is the pre-specified boundary result.

Under globally public information the relative gain was zero in both primary
positive domains. Within matched KPI-severity strata, 2,000 conditional
permutations reproduced at least the true gain zero times; mean permuted gain
was 7.88% of the true humanitarian gain and 7.04% of the utility gain. These are
development mechanism checks, not confirmatory generalization.

### Trigger and distributed robustness

All disrupted application/regime cells activated and were classified timely;
nominal and pre-disruption false activation were both 0%. The mean fraction of
communication-active agent epochs was 11.2%. No low-consensus operator decision
occurred, so safe abstention under low confidence remains untested despite the
implemented rule. Partition consensus error and performance are reported, but
no confirmatory robustness claim is warranted.

### Communication and compute accounting

Across the four deterministic matrices, all counted communication totals were
588 agent messages (131,544 structured bytes) plus 771,840 thermodynamic sketch
messages (169,821,262 bytes). There were 2,304 typed tool calls. These totals
combine different development matrices and are resource accounting, not a
claim of communication superiority. The only LLM accounting is the six Qwen
qualification calls reported above.

## 8. Statistical methods

One matched environment panel is the independent unit. Analyses use paired
episode outcomes, group-separated cross-fitting, L2-regularized logistic
ranking with regularization selected inside training folds, one attention slot,
10,000-replicate cluster bootstraps, condition-number diagnostics, and
2,000-replicate conditional permutation tests within matched KPI-severity
strata. AP, ROC AUC, Brier score, and calibration are secondary to realized
budgeted causal intervention utility. All percentages are accompanied by
absolute effects because denominators can be small.

## 9. Dashboard

The dependency-light dashboard replays any formal v4 `episode.json` without a
GPU, reconstructing only topology, explicit queue/intervention events, public
material progress, and the schema-validated operator payload. It never fills
missing display values from evaluator time series. Three populated development
exports are in [`dashboard_exports/`](dashboard_exports/).

```bash
./scripts/run-human-operator-v4-dashboard.sh --episode \
  results/human_operator_v4/raw/development_gate_trigger/<run-id>/episode.json
```

## 10. Figures

The 21 paper-facing PDFs in [`figures/pdf/`](figures/pdf/) are vector outputs;
240-DPI previews are in [`figures/previews/`](figures/previews/). They cover:

1. complete v4 architecture;
2. the three-application design;
3. the utility multilayer network;
4. the abstract cyber-physical event sequence;
5. the actual standardized phase plane and prospective boundary;
6. a populated authorized dashboard replay;
7. matched KPI-only and thermodynamic views;
8. trigger/intervention dynamics;
9. activation timing and false alarms;
10. workload versus service performance;
11. budgeted causal utility with cluster intervals;
12. application/regime paired effects;
13. feature-block ablations;
14. conditional benefit versus fragmentation;
15. calibration/reliability;
16. separate causal and autonomous-action funnels;
17. harmful/neutral/beneficial effects;
18. partition consensus error;
19. the commercial boundary result;
20. a humanitarian development case; and
21. a utility-restoration development case.

RL-curve and holdout-frontier figures are intentionally absent because those
stages were not unlocked. `reproducibility/pdf_qa/` records mechanical and
manual visual QA. The three actual replay SVG/PDF exports are separate from the
publication reconstruction and hash their source payloads.

## 11. Tables and artifact map

[`tables/`](tables/) contains the experimental design, gate outcomes,
actionability, paired coordination/human effects, feature blocks, primary
incremental value, mechanism specificity, causal-chain accounting, trigger
burden, Qwen qualification, compute/communication accounting, failed runs,
hypotheses, and stage dispositions. [`INDEX.csv`](INDEX.csv) catalogs and
checksums every other v4 artifact; the index excludes itself because a stable
self-checksum is impossible.

Raw formal ledgers are individually gzip-compressed and each file is far below
GitHub's 50 MB practical limit. Retained pilot collections are stage archives
under `superseded/raw_archives/`. No model weight, cache, virtual environment,
credential, key, token, or `.env` file belongs to this namespace.

## 12. Reproduction

From a fresh clone with the documented Python dependencies:

```bash
./scripts/run-human-operator-v4-tests.sh
./scripts/replay-human-operator-v4-results.sh
./scripts/analyze-human-operator-v4-results.sh
./scripts/generate-human-operator-v4-figures.sh
./scripts/validate-human-operator-v4-pdfs.sh
./scripts/build-human-operator-v4-report.sh
```

The all-derived-artifact rebuild command is:

```bash
./scripts/rebuild-human-operator-v4-results.sh
```

It does not rerun development, Qwen, validation, training, or holdout episodes.
The formal development runner is restartable, but rerunning frozen evidence is
not required to reproduce derived artifacts.

## 13. Claims, limitations, and readiness

Supported only as **development evidence**:

- independent-agent actionability in all three simulators;
- causal usefulness of bounded simulated-operator intervention in humanitarian
  and utility restoration;
- same-information incremental causal-ranking value from entropy/disagreement
  in those two applications under this matched design;
- disappearance of that value under globally public information; and
- a commercial exact-null boundary result.

Unsupported:

- confirmatory out-of-sample benefit;
- learned-policy stability or superiority;
- a holdout Pareto improvement;
- universal thermodynamic benefit;
- safe abstention under observed low confidence;
- real-human usability, workload, trust, safety, or operational validity;
- literal physical thermodynamics; or
- operational-security conclusions about real utilities.

AIJ readiness is **insufficient**. The repository now supplies a defensible
simulator, causal measurement design, dashboard, and development boundary, but
the prospective coordination gate failed and no validation or holdout exists.
Any successor study must be a new protocol/version; it may not reinterpret this
4.43% result as passing the frozen 5% threshold.
