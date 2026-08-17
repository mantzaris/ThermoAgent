# V7 formal development

## Execution

The single frozen formal-development batch used execution source commit
`39b463ca3607f5e55679c551ce4e5d0034b5b7b8`, source checksum
`ded1b83c41513ba6b052f2874f89a3d294999ed9af3fd28917d1ee4465043840`,
and protocol SHA-256
`760e9d019140dc0a1edf16af76f0d0a393e09d3680a3ece2499e84a8b4d0fff5`.
No threshold, feature block, panel, or success criterion changed after formal
outcomes were available.

- Reference stage: 100 panels, 44,560 agent decisions, zero failures.
- Matched dynamic stage: 200 episodes (two controllers on 100 panels), 89,120
  agent decisions, zero failures.
- Communication stage: 48 episodes, 20,160 agent decisions, zero failures.
- Formal total: 348 episodes and 153,840 decentralized decision records.

The reference stage ran from `2026-08-16T15:11:17Z` to `20:20:27Z`, the
dynamic stage from `20:21:14Z` to `2026-08-17T00:23:20Z`, and the
communication stage from `00:23:30Z` to `00:56:10Z`. These were CPU stages;
they used no GPU, LLM calls, or generated tokens.

## Primary results

H1 failed. The pooled coupling-by-fragmentation interaction was `0.006094`
(95% cluster-bootstrap CI `[-0.026463, 0.045763]`), below the frozen `0.02`
practical threshold and with an interval spanning zero.

H2 failed in both applications:

- Humanitarian: harm-rate reduction `-0.005821` (95% CI
  `[-0.014668, 0.003038]`); relative service degradation `-0.000851`
  (95% CI `[-0.009909, 0.007848]`); causal-utility gain `-5.0805`
  (95% CI `[-9.2359, -0.8485]`).
- Utility restoration: harm-rate reduction `0.008630` (95% CI
  `[0.001598, 0.016835]`); the direction was positive but far below the
  frozen `0.04` target. Relative service degradation was `0.009480`, with an
  upper confidence bound of `0.057082`, failing the `0.02` noninferiority
  margin. Causal-utility gain was `-0.3327` (95% CI
  `[-1.0319, 0.3440]`).

The strongest non-entropic cross-fitted ranking was already strong:
humanitarian AP/AUC `0.9190/0.8475`; utility AP/AUC `0.9761/0.8278`.
Generalized-entropic AP/AUC was `0.9197/0.8456` and `0.9776/0.8353`,
respectively. These small diagnostic differences did not translate into the
prospectively required dynamic harm, service, or utility effects.

H3 passed only as a communication-monitoring ablation. Event-triggered
sketches reduced all messages by `37.63%` humanitarian and `40.40%` utility,
and all bytes by `37.97%` and `40.55%`. Maximum distributed-estimation MAE was
`0.04564` and `0.04965`. Both ablation arms used the same always-act
operational controller; exact zero harm difference therefore does not show
that entropy improved selective safety.

## Scientific interpretation

V7 established a more coupled and structurally distinct simulator but did not
establish complexity-dependent control value for generalized entropy. This is
a prospective development-stage no-go, not a confirmatory null result. The
communication result supports efficient distributed monitoring under the
tested abstraction, with the operational-causality limitation stated above.
