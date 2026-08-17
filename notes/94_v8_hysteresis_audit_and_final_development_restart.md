# V8 hysteresis audit and complete final-development restart

During a source audit on 2026-08-17, before protocol freeze and before any
aggregate result from the final-development batch existed, I found that
`LocalBeliefScheduler.mark_transmitted` left a generalized trigger armed.
Consequently, the configured `tau_off` branch could not be reached in normal
execution. Cooldown and reference resetting still constrained traffic, but
this was not the explicitly required two-threshold hysteretic controller.

The last supervisor update before cancellation recorded 106/240 completed
arms and zero failures. The atomic episode manifests show that two additional
workers completed during shutdown: 108 complete episodes and two partial run
directories are preserved. All raw artifacts, registry data, configuration,
and supervisor state were retained under
`negative_results/development_final_pre_hysteresis_invalidated/` and
`raw/development_final_pre_hysteresis_invalidated/`. No comparative outcome
was aggregated or inspected before the decision.

The repair makes a successful transmission enter the off state. A sender
re-arms only after its score falls to or below `tau_off`; partition recovery
and the maximum-silence deadline remain explicit overrides. Unit tests now
exercise off-state entry, re-arming, an equal-Shannon-entropy mode change,
cooldown, and maximum silence.

The replacement batch uses all 48 independent development panels and all six
prospectively declared candidates (288 arms). It does not resume only missing
arms. It uses `dynamic_delta` event ledgers because full V7 initialization
events are deterministic scaffolding and their XZ compression—not simulation
or analysis—dominated runtime. Delta ledgers retain all V8 communication,
policy, action, resource, service, cascade, commitment, disruption, audit, and
metric events and are covered by exact replay tests.

A separate interface audit during this pre-freeze batch found that
`distributed_estimate` returned evaluator scoring fields in the same mapping
from which a policy received its authorized local estimate. The deterministic
development policy ignored the mapping, and the learned feature function had
an explicit local-field whitelist, so this did not change development
numerics. The formal source now applies an allowlist before every policy call;
evaluator pooled belief, true mode, and error fields never enter the policy
interface. A focused test checks the absence of every such field.
