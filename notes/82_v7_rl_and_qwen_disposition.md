# V7 RL and real-Qwen disposition

Sequential decentralized PPO and formal real-Qwen qualification were
implemented and tested as conditional stages. The frozen protocol required
H1 and H2 to pass before either expensive learned-agent stage could run.
Because both failed, neither stage was unlocked.

Consequences:

- No V7 PPO training seed was launched; there are no learning curves,
  checkpoints, seed-stability results, or PPO decision epochs to report.
- The planned Qwen model
  `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28` was not loaded for V7; V7 used
  zero LLM calls and tokens.
- Formal V7 evidence comes from persistent, private-state deterministic
  decentralized agents and grouped cross-fitted Level-2 controllers. Those
  agents satisfy the execution privacy and authority boundary, but they are
  not LLM agents and not trained PPO agents.
- The unreachable existing RunPod endpoint did not alter the study
  disposition: once H1/H2 failed, GPU execution was scientifically ineligible
  even if the Pod had been reachable.

This preserves the compute budget and prevents a negative formal mechanism
test from being followed by unplanned model search.
