# V15 final scientific status

## Disposition

V15 completed the prospectively frozen two-family experiment. The unit of inference is the complete graph/environment trajectory cluster. Forty-eight trajectories (six clusters per model, four matched arms per cluster) contain 34,560 attempted local decisions. All four frozen hypotheses meet their criteria, with important model and estimator heterogeneity retained below.

## Formal effects

- H1, Granite field minus nominal maximum post-quench distance: 42.26334 distance units (95% cluster-bootstrap interval 24.31644 to 59.30340), exact one-sided sign-flip `p=0.015625`, supported at allocated alpha 0.02.
- H2, persistent minus Markovized adjusted path-reversal divergence: 0.05438 nats per attempted update (0.03459 to 0.07548), exact `p=0.0004883`, Holm `p=0.0009766`, supported.
- H3, persistent minus scrambled-history adjusted path-reversal divergence: 0.04845 nats per attempted update (0.02190 to 0.07526), exact and Holm `p=0.003418`, supported.
- H4, fixed recovery sweeps 31-35 minus sweeps 41-45: 52.54060 distance units (35.40627 to 68.67316), exact `p=0.0002441`, Holm `p=0.0007324`, supported.

H2 and H3 are heterogeneous. Persistent-minus-Markovized means are 0.03041 for Qwen (5/6 positive) and 0.07835 for Granite (6/6). Persistent-minus-scrambled means are 0.00689 for Qwen (4/6; model-specific exact `p=0.21875`) and 0.09001 for Granite (6/6; `p=0.015625`). The pooled tests were prospectively specified; the decompositions prevent a homogeneous or universal claim.

## Recovery and estimator boundaries

All six Qwen field-Markovized trajectories re-enter their training-only nominal threshold six sweeps after restoration. One of six Granite trajectories re-enters within the observation horizon; the others show positive fixed-window decline but retain final distances above threshold. Thus H4 supports finite-horizon return toward the restored regime, not universal complete recovery.

The path-divergence scale depends on block length and pseudocount. Most frozen sensitivity cells remain positive, but block length four with pseudocount one gives pooled H3 of 0.00024 nats/update and a slightly negative Granite component. Conditional-memory information is not monotonically larger under persistent memory. The result is a paired coarse-grained temporal-asymmetry finding at the frozen primary estimator, not exact thermodynamic entropy production.

## V14 correction

No V14 raw trajectory changed. Audit version 1.1 excludes the held-out cluster from threshold fitting, archives the original reports, reclassifies the structurally invalid historical H3 sign test as non-inferential, completes 10,000 cluster-preserving full-pipeline permutations, recomputes 3/5/7-sweep geometries, deletes every macrostate observable individually, and adds marginal-preserving information-estimator null floors. All six corrected V14 field paths re-enter the training-only threshold exactly six sweeps after restoration.

## Effective-model boundary

The V13-fitted kinetic surrogate was not refitted to V14 or V15 quench paths. It generally overstates the direct Qwen shared-coordinate response and shifts peak timing. The direct field-minus-nominal peak difference is 0.142 in the shared five-coordinate geometry versus 1.467 for the surrogate; corresponding energy-entropy route-area differences are 0.043 and 0.695. This is evidence that the low-dimensional closure is incomplete, not evidence against the direct experiment.

## Accounting and integrity

Formal generation used 34,565 calls, 20,908,194 prompt tokens, 2,893,967 generated tokens, and 18.9366 GPU-hours. Successful model pilots add 256 decisions and 0.1408 hours. Retained rejected Mistral engineering attempts add 129 decision requests, 222 calls, and 0.1156 hours. Total measured generation is 19.1929 hours, with estimated incremental RTX 4090 cost USD 6.53 to 13.24. Replay covers all 48 trajectories and 34,560 decision rows with zero mismatches.

The external artifact root is `/workspace/ThermoAgent-v15-artifacts`. Its complete tree has 35,057 files, 185,550,119 bytes, and SHA-256 `928a7441d3f243d1e8f498e6932a1af7dae579babbd6de95e0317989af835090`. Raw transcripts and trajectory tables remain outside Git.

## Supported and prohibited interpretations

Supported: state-separated, locally informed model instances form a measurable finite interacting process; field reversal causes a reproducible within-model macrostate departure; genuine history increases the frozen coarse-grained path-divergence measure relative to paired controls; fixed-window restoration distance declines; and the effective kinetic closure misses identifiable direct response features.

Unsupported or prohibited: universality across models; model-homogeneous memory effects; complete Granite recovery within 15 sweeps; exact LLM entropy production; literal physical energy or temperature; a thermodynamic-limit phase transition; task-performance, controller, human, or application-benefit claims.

## Final verification

- Version-focused V10-V15 regression suites: 177/177 tests passed.
- Complete repository suite: 626/626 tests passed in 400.44 seconds; no failures or skips. The warnings were scikit-learn robust-covariance numerical diagnostics already exercised by sensitivity tests.
- V15 replay: 48/48 trajectories and 34,560/34,560 attempted updates regenerated with zero mismatches.
- V14 package verification passed directly. The V15 clean-package audit passed with zero missing indexed files, zero checksum mismatches, no forbidden or oversized repository artifacts, an unchanged frozen protocol, and an exact reconstruction of the legacy execution-source hash. In a clean checkout the legacy direct source-equality bit is expected to be false because its ignored cache is intentionally absent; `verification_clean.json` requires all other package checks and the documented provenance reconstruction to pass.
- PDF QA: 13 V15 PDFs and 28 pages passed automated font, opening, rendering, and text-extraction checks; all 12 figures and all 16 manuscript pages passed original-resolution manual inspection. The corrected V14 set likewise passed for 29 PDFs and 46 pages.
- V15 repository package: 7,333,931 bytes. The net V14-audit plus V15 working-tree addition is 23,905,837 bytes (22.80 MiB), below the 25 MiB ceiling; its size is dominated by font-embedded V14 PDFs and the prespecified alternative-window/permutation aggregate tables.
- The last successful remote idle audit found zero CUDA compute processes, zero V15 workers, and zero tmux sessions. A final read-only SSH recheck returned `connection refused`, so current Pod process state could not be refreshed; the endpoint appears stopped or unreachable. No local experiment, Python, CUDA, tmux, or LaTeX process remains. On the last successful evidence the Pod was safe to stop, but not delete.

## Source-checksum correction

The final completion audit found that the frozen V15 source enumerator excluded `__pycache__` directories but not one ignored, root-level 224-byte `.pyc` file. The file was never intended for Git and had no scientific semantics, but its path and digest entered the legacy frozen source hash. The audit reconstructs the legacy hash `ec9f262...` exactly from the semantic source plus the retained cache digest and defines the cache-free semantic-source hash `f8d4fa54...`. The cache remains ignored and unstaged. `results/collective_agent_statmech_v15/reproducibility/source_checksum_audit.json`, `verify_source_checksum.py`, and the generated `verification_clean.json` preserve this distinction without altering the protocol, formal outcomes, or frozen raw artifacts.
