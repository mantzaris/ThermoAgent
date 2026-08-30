# Methodology

## Scientific scope

The final study asks whether statistical-mechanics observables provide a useful
reduced description of a finite network of state-separated, locally informed
LLM-agent instances. Each persistent identity owns a categorical belief,
categorical action, confidence, commitment state, bounded private memory,
private field, inbox, outbox, and typed local authority. A random-sequential
scheduler selects update opportunities and delivers permitted messages; it
does not choose substantive model decisions.

The complete implemented state is denoted by `Xi_t`: all agent states and
memories, inboxes and outboxes, private fields, graph and delivery state,
quench phase, and specified randomness source. The recorded microscopic
projection is `Y_t = phi(Xi_t)`, and rolling collective observables form
`Z_t = psi(Y_{t-w+1:t})`. The augmented process is stochastic under the fixed
model and quench schedule. Neither the microscopic projection nor the rolling
macrostate is assumed to be Markov. Omitted private memory can act as a hidden
slow coordinate, so identical visible belief-action configurations can have
different future transition laws.

## Evidence stages

The repository preserves four scientifically distinct stages under semantic
names:

1. **Discovery** established measurable local response and exploratory
   collective trends in Qwen-agent networks.
2. **Replication** prospectively tested the principal collective-order,
   noise, and memory directions with new Qwen graph/environment clusters.
3. **Corrected quench** prospectively studied field reversal, network
   partition, message corruption, and restoration. A delayed audit corrected
   recovery-threshold fitting so the held-out cluster never enters nominal
   geometry. It also reclassified a structurally nonnegative historical
   recovery statistic as invalid directional evidence; no recorded trajectory
   was changed.
4. **Cross-model** used Qwen and Granite with six independent graph/environment
   clusters per model and four matched conditions per cluster: nominal
   Markovized, field Markovized, field persistent-memory, and field
   scrambled-history placebo. This is the final confirmatory layer used in the
   manuscript.

The scrambled-history arm is a deterministic, past-only, format- and
length-matched synthetic-history placebo. It contains no genuine trajectory
content, future information, donor-agent state, or peer-private state. It is
not a permutation of genuine private histories.

## Final design and inference

The cross-model study has 48 trajectories: two model families, six clusters per
model, and four matched conditions. Each trajectory contains 16 agents and 45
sweeps, with 16 attempted random-sequential updates per sweep. The periods are
15 baseline sweeps, 15 field-reversal or matched nominal sweeps, and 15 restored
sweeps. The complete model-by-graph/environment trajectory cluster is the
inferential unit.

Primary uncertainty uses cluster bootstrap intervals. Directional tests use
exact cluster-level sign flips, with the prospectively specified alpha
allocation and Holm adjustment. Model-specific heterogeneity remains visible;
the pooled memory contrast is not described as separate confirmation within
both model families.

## Statistical-mechanics observables

The paper reports belief and action magnetization, belief-action overlap,
disagreement, configuration entropy, entropy rate, mutual information, total
correlation, susceptibility, connected graph-distance correlation, truncated
integrated autocorrelation time, and Binder cumulants. The symmetric-layer
Hamiltonian is an **effective reference energy** for compatibility with a
chosen interaction model. It is not literal physical energy. The Binder
cumulant is a finite-size distribution-shape diagnostic, not evidence of a
critical point.

Path-reversal block divergence is evaluated with a shuffled finite-sample
floor. Adjusted arm-level values can be negative and are never truncated.
Positive contrasts therefore mean that one condition has a larger adjusted
coarse-grained time-asymmetry statistic than its matched control; they do not
establish positive absolute entropy production or physical dissipation.

## Interpretation boundaries

- Decoding temperature is a sampling control, not physical temperature.
- Effective reference energy is descriptive, not physical internal energy.
- Pathwise irreversibility is a coarse-grained projection statistic, not exact
  stochastic-thermodynamic entropy production.
- One size (`N=16`) and one reciprocal modular topology establish no
  thermodynamic limit, universal exponent, critical point, or phase transition.
- No human, operational-benefit, controller-performance, or application
  superiority claim is supported.
