# Statistical-mechanics methodology

The three role-normalized dimensions are shortage/backlog pressure, capacity
impairment, and commitment/communication strain. Each is binned into three
levels, giving 27 states. Thresholds are fit from nominal scripted training
episodes only. Pooled, role-conditioned-shrinkage, one-period, and three-period
alternatives were compared before any treatment evaluation. The fixed choice is
pooled occupancy with a one-period window and Laplace `alpha=0.1`.

The implementation computes normalized Gibbs entropy, a fixed-priority
operational energy, and `Delta F = T KL(p || q)` with
`T=0.1896119180636096`. This scale minimized nominal-occupancy KL over a fixed
120-point log grid from 0.05 to 2.0. The
energy weights `(backlog=.35, delay=.20, shortfall=.30, commitment=.15)` follow
domain priorities and are not selected against final method performance.
Sensitivity analysis uses three alternatives fixed before the main run:
balanced `(0.25, 0.25, 0.25, 0.25)`, backlog-and-service-heavy
`(0.40, 0.10, 0.40, 0.10)`, and delay-and-commitment-heavy
`(0.15, 0.35, 0.15, 0.35)`, in `(backlog, delay, shortfall, commitment)` order.
Each alternative produces both evaluator-only operational energy and a
separately induced healthy-reference free-energy gap. No alternative changes
an agent observation or replaces the primary construct after outcomes are seen.

Because the operational macrostate has three axes but the fixed energy has four
domain-priority terms, each state's bin centers are mapped prospectively as
follows: shortage pressure supplies the backlog and service-shortfall proxies;
capacity impairment and strain jointly supply the delay proxy
(`0.55 * impairment + 0.45 * strain`); and the strain center supplies the
commitment proxy. This is a coarse operational model, not a claim that these
quantities are physically interchangeable.

The final scripted monitor-selection data produced a +0.062971 operational-entropy
shift but a -0.030783 free-energy shift after moderate disruption. This negative
direction is retained. Monitoring analysis therefore reports the raw one-sided
free-energy detector and a pre-main two-sided absolute deviation from the
nominal median, alongside entropy and operational energy. It does not redefine
`Delta F` or claim high entropy/free energy is intrinsically harmful.

Local surprisal uses a role-conditioned nominal healthy reference. For each
role, its Laplace-smoothed nominal occupancy receives weight 0.5 and the fixed
Gibbs ensemble receives shrinkage weight 0.5; this limits variance for sparse
roles without making all role baselines identical. The fit uses the same
nominal training seeds only and is fixed before v3/main. Interaction entropy is
computed separately over recent directed interaction weights.

Each agent sends only a smoothed one-hot macrostate sketch over the currently
available communication graph. Metropolis weights support stable average
consensus. Tests empirically verify convergence on a connected graph and
persistent error across a partition. The exact distribution is evaluator-only,
except in the named global-oracle ablation.

Sketch exchange uses a separately accounted monitoring channel so mandatory
consensus traffic does not consume an agent's domain-negotiation authority.
It is not treated as free: every directed edge-round transmission and compact
serialized 27-state payload byte is included in combined communication metrics.

The shuffled control is causal: an agent receives another identity's
prior-period distributed monitor vector. It preserves dimensions and scale
while breaking identity and current-event alignment, with no future leakage.

Terms such as energy and temperature are statistical-mechanics-inspired
operational constructs, not literal thermodynamic quantities.

## Final empirical assessment

Across the frozen evaluation, exact operational entropy was the strongest
monitor (average precision `0.934`, ROC AUC `0.863`, nominal false-alarm rate
`0.030`). Operational energy was also discriminative (AP `0.885`, AUC `0.800`)
but poorly calibrated at its nominal threshold. The free-energy gap retained
the pilot sign mismatch: primary disrupted-minus-nominal means were `-0.0108`
commercial and `-0.0046` humanitarian, with AP `0.577`, AUC `0.393`, and false-
alarm rate `0.826` when high values were treated as alarms. Alternative fixed
energy weights did not repair the free-energy direction.

Agent-local distributed error rose monotonically with communication damage.
Reliable entropy/free-energy MAE was `0.00524/0.00327` commercial and
`0.00072/0.00045` humanitarian; partition values were `0.03976/0.02485` and
`0.06280/0.03924`. Consensus RMSE strongly tracked estimator error. This
supports the distributed-estimation implementation and entropy detector, but
not the free-energy alarm or a claim that feeding these signals to the policy
improves logistics.
