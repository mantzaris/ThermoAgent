# V11 final status

V11 is a prospectively stopped LLM qualification study. It began from V10
commit `4d372f00837bf75f90882392a92feac87dbc84b2` on local branch
`evidence-grounded-llm-entropy-v11`. No V1--V10 file was modified.

Protocol `v11-qualification-1.0` (SHA-256
`7bd4082f9d085222e22d195e3ff603f0f76e27c36b347bfc4132fbb164a3d03f`)
froze all decisive qualification conditions before outcomes. The authoritative
source-tree checksum was
`1f7bcd164e4f07f033def27a8236ed0995413b4c12dcd6203549ca08d48d395e`.

The 864-request Qwen qualification found a positive average continuous belief
effect, but failed frozen monotonicity and transition-diversity components.
Accordingly, the formal decentralized LLM network, entropy-production,
irreversibility, controls, and Markov-adequacy stages were not run. This status
is final for V11; no threshold will be changed and no qualification row will be
selectively rerun.

The supported claim is limited: calibrated delivered evidence changed Qwen's
reported log odds on average in two simulated semantic framings. V11 does not
support a claim of Bayesian integration, calibrated transition probabilities,
LLM-network entropy production, nonreciprocity-induced irreversibility,
first-order Markov adequacy, operational utility, or human benefit.

Repository-facing code, aggregates, vector figures, and manuscript sources are
uncommitted for manual review. Raw model interactions and invalidated pilots
remain outside Git. No V11 file has been staged, committed, or pushed.

Final verification passed 69 focused V10/V11 tests and all 518 repository tests
under pytest importlib collection, with zero failures, errors, or skips. The
default pytest prepend mode has a known collection-name collision between
frozen V9 and V10 `test_agents.py` files; no test was excluded from the
successful importlib-mode run. Five repository-facing PDFs (four figures and
the eight-page manuscript) open, render, expose extractable text, and contain
embedded fonts. Every page was manually inspected at native and 300-DPI size.

The final RunPod check found zero V11/Qwen research processes, zero tmux
sessions, no CUDA compute process, 1 MiB GPU memory use, and 0% GPU utilization.
It is safe to stop the existing Pod, but it must not be deleted.
