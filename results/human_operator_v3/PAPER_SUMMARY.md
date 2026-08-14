# Paper-oriented summary

## Working title

**ThermoHITL: Mechanism-Gated Thermodynamic Triage for Human-on-the-Loop
Oversight of Decentralized Logistics Agents**

The more positive title “Thermodynamic Triage for Human-on-the-Loop Oversight”
should not be used without a subtitle or framing that communicates the
development-stage cross-application failure.

## Provisional abstract

Autonomous multi-organization logistics systems may require supervisory help,
but indiscriminate human review consumes scarce attention. We implement
ThermoHITL, an adaptive-autonomy architecture in which independent agents
exchange privacy-preserving operational energy and entropy sketches, request
bounded simulated-operator assistance, and respond to typed interventions. To
avoid promoting a monitoring-only mechanism, we preregister six development
gates covering engineering, material actionability, coordination necessity,
human causal usefulness, incremental thermodynamic information, and trigger
timing. In two synthetic commercial and abstract humanitarian logistics
applications, 183 tests and exact replay of 817 event ledgers show zero replay
mismatches and a maximum conservation residual of 6.82e-13. A corrected frozen
Qwen2.5-7B planner achieves 97.89% first-pass structured validity, 100% validity
after one repair, and 84.21% accepted-action arrival at demand. Fixed
communication reduces development loss by 48.45% and 52.54%, while bounded
KPI-triggered simulated supervision reduces loss by 1.59% and 2.80% in the two
applications. Counterfactual branches verify complete alert-to-material-to-loss
chains. However, thermodynamic features fail the preregistered same-information
incremental-value criterion commercially (ΔAP -0.0147; utility +2.86%) despite
passing humanitarianly (ΔAP +0.0649; utility +211.84%). We therefore stop before
validation, reinforcement-learning training, or holdout evaluation. The result
is a reproducible mechanism-gated platform and an application boundary, not
confirmatory evidence that thermodynamic triage improves supervisory control.

## Verified contributions

1. An independently acting multi-agent oversight architecture with private
   state, link-respecting sketches, bounded requests, capacity-constrained
   attention, typed intervention authority, and returned autonomy.
2. A functional operator dashboard whose exact schema-limited payload is hashed
   and used by the tested simulated-operator policy.
3. A replayable counterfactual branch mechanism that restores simulator, agent,
   RNG, view, and workload state and audits the full intervention chain.
4. Development-only environments in which communication and bounded human
   authority are causally actionable in both applications.
5. A prospective six-gate protocol that prevented an unsupported positive
   validation/holdout claim when cross-application thermodynamic information
   value failed.

## Primary numerical results

- Tests: 183 passed.
- Exact replay: 817/817; zero mismatches/privacy failures/nonfinite values.
- Maximum material-conservation residual: `6.821210263296962e-13`.
- Qwen v9 retry: 97.89% first pass; 100% after one repair; 84.21% of
  accepted material actions reached final demand.
- Fixed communication vs no communication: 48.45% commercial and 52.54%
  humanitarian development loss reduction.
- KPI-triggered bounded operator vs autonomy: 1.59% commercial and 2.80%
  humanitarian development loss reduction.
- Trigger: 100% timely activation across 30 disrupted final-candidate episodes;
  0% missed/pre-disruption/nominal false activation.
- Same-information incremental value: commercial failed; humanitarian passed;
  Gate 5 and the cross-application claim failed.
- Additional GPU estimate: 0.18943 hours; 589 calls; 1,084,097 prompt and
  32,250 generated tokens; approximately $0.064 at $0.34/hour.

## Claim disposition

- Independent oversight architecture: confirmed as engineering evidence.
- End-to-end causal intervention mechanism: confirmed on development probes.
- Trigger feasibility: supported on development only.
- Cross-application thermodynamic incremental value: unsupported.
- ThermoHITL superiority to KPI triggering: untested.
- Non-inferiority to always-on review: untested.
- Improved loss/operator-effort Pareto frontier: untested.
- Partition robustness: untested confirmatorily.
- Actual human benefit: untested; no participants.

## Figure and table plan

Use the 20 validated PDFs in `figures/pdf/`, led by architecture, dashboard,
actionability, monitoring incremental value, trigger timing, causal funnel, and
paired development effects. The training figure must remain an explicit
`PROSPECTIVELY NOT RUN` panel. Main tables should be the gate matrix,
actionability funnel, monitoring incremental value, paired development effects,
counterfactual chain, and compute/failure accounting. Do not elevate exploratory
loss/effort points to a confirmatory Pareto claim.

## Limitations and recommendation

V3 has no validation, independent RL seeds, or locked holdout; operators are
simulated; the real planner sample is small; the partition diagnostic is
incomplete; and the commercial information-value gate failed. The current
material is suitable for an engineering artifact or a workshop/boundary paper.
It is **not ready for the intended positive Elsevier Artificial Intelligence
journal submission**. The next study must be separately versioned and must not
reuse a seen holdout.

See [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) for the 24–27 page outline and
claim-to-evidence map.
