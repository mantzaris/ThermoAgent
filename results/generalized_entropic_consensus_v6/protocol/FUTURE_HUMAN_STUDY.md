# Future real-human study protocol (not conducted)

This is technical preparation only. No participant was recruited and no human
performance, fatigue, trust, workload, or usability result exists. Ethics/IRB
approval and application-specific safety review are prerequisites.

## Proposed randomized study

Recruit trained adult participants appropriate to the eventual task context.
Randomize scenario order and display condition within participant using a
counterbalanced Latin-square schedule. Compare KPI/action-confidence only,
Shannon/JS consensus, generalized entropy spectrum, and combined uncertainty
views at the same incident set, action menu, operator budget, and timing.
Participants would prioritize alerts, choose a bounded intervention or
abstention, and record confidence.

Primary technical outcomes would be correct incident/action prioritization and
dynamic service loss under a fixed attention budget. Secondary outcomes would
include decision time, harmful interventions, missed critical incidents,
calibration, uncertainty comprehension, trust/over-reliance, and NASA-TLX (or
an approved equivalent). The study must distinguish display value from policy
recommendation value and must include a no-recommendation display control.

## Power template

The independent human participant—not a click—is the cluster. Enter the
smallest practically meaningful paired service-loss or prioritization effect,
participant-level standard deviation, within-participant correlation,
anticipated attrition, multiplicity family, and target power before collecting
data. Use simulation for the crossed participant/scenario design. Freeze
exclusions, timeouts, missing-data handling, and stopping rules.

## Required trial log

Log anonymized participant/session ID, randomized condition and order,
scenario/panel ID, view-payload hash, action alternatives, chosen action,
response time, confidence, workload answers, operator queue, intervention
cost, dynamic outcome, technical failure, and withdrawal. Never log identity,
credentials, or unapproved sensitive data. Counterfactual evaluator fields
must remain hidden during decisions.

## Dashboard task script

1. Explain that all incidents are simulated and describe bounded authority.
2. Practice on nonstudy panels until action semantics are understood.
3. Present four concurrent incidents and a fixed time/attention budget.
4. Ask for alert priority, execute/communicate/abstain/escalate choice, bounded
   intervention, confidence, and explanation category.
5. Reveal outcomes only after the decision window.
6. Administer workload/comprehension items at fixed blocks.
7. Debrief uncertainty limits and automation bias.

The replay dashboard already hashes authorized payloads and separates agent,
operator, and evaluator views; a real study would still require accessibility,
security, consent, privacy, and usability review.
