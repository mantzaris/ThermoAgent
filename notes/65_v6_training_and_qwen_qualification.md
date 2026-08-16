# V6 decentralized training and Qwen qualification

## Sequential decentralized PPO

The frozen training matrix completed on 2026-08-16 UTC before any seed-level
results were opened. It contains five prespecified methods and five independent
training seeds (`66201` through `66205`) per method, for 25 completed runs and
zero failed or selectively removed runs. Each run used 200 training episodes
and 60 evaluation episodes, yielding 6,500 episodes and 156,000 decentralized
agent decision epochs in total.

The implementation uses role-specific local observations and action masks,
sequential trajectories, discounted returns, generalized advantage estimation,
PPO clipping, and entropy regularization. It is accurately described as
sequential decentralized PPO, not as the contextual actor-critic used in V5.

No seed collapsed to a universal action. Minimum evaluation action diversity
was four to six actions, depending on method. Frozen aggregate results were:

| Method | Mean reward | Mean autonomous harm rate | Between-seed harm SD | Minimum action diversity |
|---|---:|---:|---:|---:|
| KPI only | -0.02174 | 0.39270 | 0.08307 | 6 |
| Predictive uncertainty | -0.12597 | 0.35350 | 0.07233 | 6 |
| Shannon/Jensen-Shannon | -0.14026 | 0.39063 | 0.05921 | 5 |
| Generalized Tsallis/Gini | -0.17616 | 0.39391 | 0.09338 | 4 |
| Combined generalized entropic | -0.03269 | 0.35906 | 0.09382 | 5 |

The combined method has higher mean reward than the frozen
predictive-uncertainty comparator, but its between-seed harm-rate standard
deviation `0.09382` exceeds the prospectively frozen Gate 9 maximum `0.08`.
No seed is excluded and the threshold is not revised. Gate 9 therefore fails
on learning stability even though all runs completed and action diversity is
nontrivial.

All PPO training and evaluation operational messages, thermodynamic-sketch
messages, bytes, and wall time are persisted in the seed manifest. Training
does not use Qwen calls or tokens.

## Real-Qwen qualification

The frozen 150-episode qualification was launched only after the entire PPO
matrix closed. It used the pinned `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4 quantization, BF16 compute,
separate agent contexts, private observations/beliefs/memories, typed
role-authorized tools, and at most one prospectively allowed structured-output
repair.

Coverage was 60 humanitarian, 60 utility-restoration, and 30 commercial
episodes, split across private-fragmented and public-shared information and
five disrupted regimes. The 150 episodes contain 2,700 agent decisions and
2,755 calls, using 3,096,925 prompt tokens and 232,235 generated tokens. Model
generation took 1,936.05 seconds and the load-inclusive run took 1,986.59
seconds.

| Application | Decisions | First-pass valid | Valid after repair | Physical actions | Service-reaching | Harmful / physical | Mean effect per decision | Distinct actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Commercial boundary | 540 | 98.33% | 100% | 1 | 0 | 1 / 1 | -0.04350 | 4 |
| Humanitarian | 1,080 | 99.07% | 100% | 0 | 0 | 0 / 0 | -0.03106 | 3 |
| Utility restoration | 1,080 | 96.67% | 100% | 91 | 26 | 52 / 91 | -0.03454 | 4 |

The repair-validity result does not establish action quality. Humanitarian
agents issued no physical actions. Utility agents did act, but 57.14% of those
physical actions were harmful and their mean causal effect was `-0.06586` per
physical action. Commercial evidence consists of only one physical action,
which was harmful. No application produced a human escalation. These failures
are retained and Gate 4 fails; the qualification is behavioral development
evidence, not a positive autonomous-agent claim.

Post-outcome diagnostic calibration, added without changing frozen gate
decisions, reports all-decision Brier scores of `0.6212`, `0.6231`, and
`0.5825` and expected calibration errors of `0.7725`, `0.7785`, and `0.7210`
for commercial, humanitarian, and utility restoration respectively. Regret in
the package is explicitly measured against `no_action`; a best-authorized-
action regret model was not prospectively specified and is not claimed.
