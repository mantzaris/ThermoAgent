# JSTAT V16 manuscript-structure audit

**Date:** 26 August 2026  
**Stage:** structural research and planning only  
**Manuscript inspected:** `paper/jstat_v15/main.tex`, `main.pdf`, `supplement.tex`, `supplement.pdf`, `results_macros.tex`, and `references.bib`  
**Scientific boundary:** all V15 results, figures, equations, numerical statements, uncertainty summaries, caveats, and conclusions are frozen  
**Working branch:** `jstat-paper-structure-v16`, created locally from the author-approved current tip `fbac5029a431ffc1e82befef072601aadeed2232` of `origin/jstat-scientific-audit-v15`  

The originally stated expected commit, `b8c639364ea8d0f7232caafa1af84f633171d7f2`, is the parent of the later author-pushed commit `fbac5029` (`copy`). The author explicitly instructed this pass to continue from the newer remote tip. No manuscript source was modified, compiled, committed, or pushed in this pass.

## 1. Executive recommendation

Adopt an argument-driven JSTAT structure with **nine main numbered sections and four article appendices**, rather than retaining the present fourteen narrow top-level sections or forcing a generic Introduction–Methodology–Data–Results–Discussion template.

The recommended main sequence is:

1. Introduction
2. State-separated LLM-agent networks and quench protocols
3. Statistical-mechanical observables and inferential design
4. Evidence hierarchy, correction, and frozen study roles
5. Quench response, restoration, and finite-size organization
6. Memory and projected temporal asymmetry
7. Reduced dynamical descriptions and closure limits
8. Discussion
9. Conclusions

This sequence follows the paper's actual scientific logic:

- define the interacting stochastic system and its information boundaries;
- define the reduced observables and what they do and do not mean;
- distinguish discovery, corrected historical evidence, and prospective V15 confirmation;
- present the quench and memory results as two connected empirical arguments;
- test the limits of reduced closure;
- interpret the results only after the evidential hierarchy is clear.

The present two-page supplement should be **eliminated as a separate file**. Its finite-window persistence and Binder material directly qualifies claims already made in the main text. It should become **Appendix C of the article**, with a short main-text summary and explicit article cross-reference. This keeps the diagnostic in the Version of Record and in the default peer-review object.

The four frozen figure assets that are catalogued but absent from both current PDFs—Figures 5, 8, 11, and 12—should also be placed in article appendices, not silently left outside the review narrative. No figure should be redesigned or regenerated.

This recommendation distinguishes three different kinds of evidence:

- **Formal publisher requirements:** common TeX is acceptable; the `iopjournal` class is optional; integral material belongs in the article; figures and tables belong near their first citations; a data-availability statement is required by current IOP policy.
- **Observed JSTAT conventions:** recent papers use flexible, topic-driven sequences; seven of the eight papers examined use article appendices for proofs, robustness, numerical detail, or implementation; Discussion and Conclusion arrangements vary.
- **Editorial recommendation for ThermoAgent:** retain separate Discussion and Conclusions because the interpretive safeguards are too substantial to compress into the take-home section, even though that exact pairing is not a journal mandate.

## 2. Research basis and source handling

The comparison sample contains the five author-specified examples plus three additional recent JSTAT research papers selected for close topical relevance. Section sequences were read from a published IOP PDF or an author-posted arXiv version associated with the published DOI. Generic summaries were not used to infer structure.

The IOP Version of Record was directly inspected for `ae7e6a`. For the remaining papers, DOI/publisher metadata were checked and the corresponding author-posted full text or TeX source was inspected. This distinction matters because typography can differ between an author version and the Version of Record, but the section hierarchy is still directly observable in the paper itself.

Research cutoff: 26 August 2026.

## 3. Formal JSTAT/IOP requirements found

| Formal point | Official evidence | Consequence for ThermoAgent |
|---|---|---|
| A proprietary journal class is not mandatory. | IOP states that any common TeX variant is acceptable and that its class file is not essential: [LaTeX template guidance](https://publishingsupport.iopscience.iop.org/questions/latex-template/). | Do not convert to `iopjournal` during the structural pass. The current standard `article` class is acceptable for review preparation. |
| Submission formatting is flexible, but reviewer readability matters. | IOP says authors may choose the format, recommends at least 12-point type, and requires English: [article format guidance](https://publishingsupport.iopscience.iop.org/questions/article-format/). | The current 12-point article is structurally acceptable. Template conversion is a later production choice, not a prerequisite for reorganization. |
| Figures and tables should be embedded at the appropriate point in the text, rather than placed at the end. | [Article format guidance](https://publishingsupport.iopscience.iop.org/questions/article-format/) and the [IOP style guide](https://publishingsupport.iopscience.iop.org/questions/style-guide-journal-articles/) say figures should be close to their first citation and numbered in order. | The current PDF behavior, in which Figures 3–9 collect on pages 16–21, should be corrected in the implementation pass through section placement and float control. |
| Integral material must be part of the article. | IOP states: material integral to the article must be submitted as part of the article, not as supplementary material: [supplementary-material guidance](https://publishingsupport.iopscience.iop.org/questions/supplementary-material-and-data-in-journal-articles/). | The current persistence/Binder diagnostic belongs in an article appendix because it qualifies claims made in the article. |
| Separate supplementary material is outside the article Version of Record and is not reviewed by default. | The same [supplementary-material guidance](https://publishingsupport.iopscience.iop.org/questions/supplementary-material-and-data-in-journal-articles/) says separate files are not part of the article PDF or Version of Record, are not included in peer review by default, and are not edited or proofed by production. | A two-page prose-and-figure supplement is a weaker location for this diagnostic than an appendix. |
| If separate supplementary files are used, they receive DOIs and need short internal metadata. | IOP requires a title and description in each supplementary file, with a title no longer than 30 characters and a description no longer than 30 words; files may be up to 50 MB each and 150 MB in total with the article. | These rules become irrelevant if the prose supplement is removed. They should not be mistaken for a reason to retain it. |
| A data-availability statement is mandatory under current IOP policy. | [IOP research data policy](https://publishingsupport.iopscience.iop.org/iop-publishing-open-data-policy/) states that authors must include a data-availability statement. | Keep a concise main-article data-availability statement. Detailed artifact accounting may later move to Appendix D without weakening the statement. |

### Requirements not found

The consulted JSTAT/IOP guidance does **not** prescribe:

- an Introduction–Methods–Results–Discussion sequence;
- a mandatory standalone Methods section;
- separate Discussion and Conclusion sections;
- a maximum number of appendices;
- a requirement to use the `iopjournal` class at submission;
- a requirement to place technical robustness material in a separate supplement.

Those are editorial decisions, not publisher rules.

## 4. Comparison of recent JSTAT papers

| Paper | Year and stable source | Main section sequence | Important subsection pattern; methods/results relationship | Discussion and conclusion | Appendices and separate supplementary material | Structural relevance to ThermoAgent |
|---|---|---|---|---|---|---|
| **Why diffusion models do not memorize: the role of implicit dynamical regularization in training** | 2026; [DOI 10.1088/1742-5468/ae7e6a](https://doi.org/10.1088/1742-5468/ae7e6a); [author version](https://arxiv.org/abs/2505.17638) | 1 Introduction → 2 Generalization and memorization during training → 3 Training dynamics of a random-features network → 4 Conclusions | Empirical behavior and theoretical model are successive argument stages, not Methods then Results. Appendices A.1–A.3 cover setup, batch-size effects, and Adam; B.1–B.3 cover a Gaussian-mixture check and conditional models; C.1–C.6 contain proofs. | Conclusions only; no standalone Discussion. | **4 appendices**: numerical experiments, Gaussian-mixture transition, analytical proofs, random-feature numerics. No separate prose supplement identified; extensive technical content is in the article PDF. | Strong precedent for an article that is concept-first, then evidence, then explanation, with large technical appendices in the Version of Record. |
| **Harnessing finite-size effects to gauge aging in the 2D Ising model** | 2026; [DOI 10.1088/1742-5468/ae8439](https://doi.org/10.1088/1742-5468/ae8439); [author version](https://arxiv.org/abs/2607.11524) | 1 Introduction → 2 Model and simulational methods → 3 Data analysis → 4 Results → 5 Conclusion and Outlook | Results are organized by temperature; one branch splits into inclusion, removal, and stochastic equilibration of metastable striped states. This is the clearest conventional separation of model/method, analysis, and results in the sample. | Discussion is subsection 4.3 inside Results; separate Conclusion and Outlook follows. | **0 appendices**; no separate supplement identified. | Relevant counterexample: explicit method/result separation works when the paper has one model, one analysis path, and a compact result hierarchy. ThermoAgent has a more complex study chronology, so copying this mechanically would obscure roles. |
| **Yielding versus random organization: convex absorbing transitions in soft matter** | 2026; [DOI 10.1088/1742-5468/ae8668](https://doi.org/10.1088/1742-5468/ae8668); [author version](https://arxiv.org/abs/2606.23914) | 1 Introduction → 2 Definition of the models → 3 Similar evolution of critical behavior → 4 Theoretical frameworks → 5 Discussion | Model definitions precede a multi-observable comparative results section; theory follows the empirical comparison and explains regimes. | Standalone Discussion; no separate Conclusion. | **1 appendix** on numerical determination of critical exponents; no separate supplement identified. | Strong precedent for putting two systems under a common observable framework, presenting comparison first, and then using a reduced theory to expose commonality and failure. |
| **Generative diffusion for perceptron problems: statistical physics analysis and efficient algorithms** | 2026; [DOI 10.1088/1742-5468/ae8ceb](https://doi.org/10.1088/1742-5468/ae8ceb); [author version](https://arxiv.org/abs/2502.16292) | 1 Introduction → 2 Preliminaries on stochastic localization → 3 Asymptotic analysis with replicas → 4 Applications on perceptron models → 5 Conclusion | The applications section combines definitions, implementation, spherical results, binary results, potential selection, and an annealed algorithm. It is not split into generic Methods and Results. | Conclusion only. | **6 appendices**: AMP, replica calculation, stability distribution, spherical case, binary case, limiting behavior. No separate prose supplement identified. | Relevant to ML viewed through statistical physics: formalism stays central, applications integrate computation and result, and lengthy derivations remain in article appendices. |
| **Habitat heterogeneity and dispersal network structure as drivers of metacommunity dynamics** | 2026; [DOI 10.1088/1742-5468/ae727d](https://doi.org/10.1088/1742-5468/ae727d); [author version](https://arxiv.org/abs/2602.06640) | 1 Introduction → 2 Metapopulation models from microscopic dynamics → 3 Exact capacity results → 4 Stochastic dynamics and finite-size effects → 5 Interspecies model → 6 Coexistence/monodominance → 7 Outlook → 8 Concluding remarks | Model construction and results alternate as the scientific scale expands from one species to stochastic finite-size effects to multispecies coexistence. | Separate Outlook and Concluding remarks; no Discussion heading. | **3 appendices**: two sets of proofs and derivation of stochastic dynamics. No separate supplement identified. | Strong precedent for organizing by conceptual scale rather than by evidence type; also relevant to networks, finite-size stochasticity, and reduced descriptions. |
| **Towards neural reinforcement learning for large deviations in non-equilibrium systems with memory** | 2025; [DOI 10.1088/1742-5468/adea65](https://doi.org/10.1088/1742-5468/adea65); [author version](https://arxiv.org/abs/2501.12333) | 1 Introduction → 2 Memory-dependent stochastic processes → 3 Current fluctuations → 4 Computational framework → 5 Applications → 6 Summary and outlook | Formal stochastic background and computational machinery are separated from applications; applications are grouped by system, while robustness and implementation leave the main line. | Summary and outlook only; no separate Discussion. | **4 appendices**: hidden-Markov analytical checks, robustness/hyperparameters, recurrent-network architecture, implementation. No separate prose supplement identified. | Closest structural precedent for memory, non-Markov dynamics, ML machinery, exact observables, computational checks, and implementation detail. |
| **Dynamical and structural properties of an absorbing phase transition: a case study from granular systems** | 2025; [DOI 10.1088/1742-5468/ae23bc](https://doi.org/10.1088/1742-5468/ae23bc); [author version](https://arxiv.org/abs/2507.06083) | 1 Introduction → 2 Granular-system models → 3 Dynamical and structural properties → 4 Kinetic theory → 5 Hydrodynamics of the active state → 6 Conclusion | Realistic and effective models are paired; numerical phenomenology is followed by two levels of theory, whose own subsections combine derivation, stability, correlations, and numerical comparison. | Conclusion only. | **6 appendices**: tail argument, critical hyperuniformity, clustered fluids, tricritical behavior, transport coefficients, adiabatic slaving. The arXiv record also supplies two ancillary videos, but no separate prose/figure supplement. | Strong precedent for keeping diagnostic derivations and secondary structure inside appendices while maintaining a direct model → phenomenon → reduced-theory narrative. |
| **Coarse-graining nonequilibrium diffusions with Markov chains** | 2026; [DOI 10.1088/1742-5468/ae4f7d](https://doi.org/10.1088/1742-5468/ae4f7d); [author version](https://arxiv.org/abs/2511.05366) | 1 Introduction → 2 Stochastic processes and nonequilibrium steady states → 3 Discrete-state approximations → 4 Example processes and numerical experiments → 5 Statistical inference → 6 Discussion | Formalism, approximation, numerical examples, and inference are separate scientific stages. Solvable and unsolvable examples are nested within one results/application section. | Standalone Discussion; no separate Conclusion. | **5 appendices**: discretization, non-diagonal diffusion, variational structure, sampling, and statistical inference. No separate supplement identified. | Especially relevant to projection, hidden memory, entropy-production boundaries, discrete approximations, and estimator limitations. |

### Sample-level caution

This sample is purposive rather than a census. No single paper establishes a house rule. The value of the sample is in repeated patterns across papers that differ substantially in subject and method.

## 5. Observed JSTAT structural conventions

### Repeated patterns

1. **All eight papers begin with an Introduction, but no single post-introduction template dominates.**
2. **Scientific objects are defined before their results are interpreted.** Model, stochastic process, formalism, or computational framework normally precedes applications or empirical comparison.
3. **Methods and results are often integrated by conceptual stage.** Only the finite-size Ising paper uses an unmistakable Model/Simulational Methods → Data Analysis → Results sequence. The other papers organize by phenomenon, formalism, application, or scale.
4. **Technical appendices are normal and often extensive.** Seven of eight papers use article appendices; the counts are 4, 0, 1, 6, 3, 4, 6, and 5.
5. **Appendices carry material needed for rigorous review.** Roles include proofs, estimator derivations, robustness/hyperparameter checks, implementation, alternate models, finite-size diagnostics, and detailed numerical procedures.
6. **Discussion/Conclusion practice is flexible.** In this sample:
   - two papers have a standalone Discussion and no standalone Conclusion;
   - one has Discussion inside Results plus a separate Conclusion and Outlook;
   - five end with Conclusion, Concluding remarks, or Summary/Outlook and no Discussion section.
7. **A separate prose supplement is not the default solution for technical depth.** The papers overwhelmingly keep derivations and diagnostics in article appendices. One author record supplies ancillary videos, which are genuinely a separate medium.
8. **Reduced theory often follows direct evidence.** Several papers define models, show phenomenology, and only then test a reduced or analytical description. This is a good match to ThermoAgent's kinetic-surrogate role.
9. **Negative or limiting behavior is structurally visible.** Robustness, metastability, failed approximations, and estimator boundaries receive named subsections or appendices rather than being hidden in a concluding paragraph.

### Implications for ThermoAgent

- The manuscript should not be forced into generic IMRaD.
- The system/protocol and observable/inference blocks should nevertheless remain distinct because their epistemic roles differ.
- The V12–V15 chronology needs its own compact evidence-hierarchy section; otherwise discovery, correction, and prospective confirmation can be conflated.
- The kinetic surrogate belongs after the direct results, not inside the system definition.
- The current supplement is more consistent with JSTAT convention as an article appendix.
- Separate Discussion and Conclusions are justified here by the amount of interpretation and caveat preservation, even though they are an editorial choice rather than the modal pattern in this small sample.

## 6. Audit of the current V15 manuscript

### 6.1 Artifact inventory

| Artifact | Current structural state |
|---|---|
| `main.tex` | 836 lines; 14 numbered main sections; 2 unnumbered declarations; 3 appendix sections; 15 numbered display-equation environments; 9 embedded figure assets; no tables. |
| `main.pdf` | 21 A4 pages. Figures 1 and 2 appear near their declarations; rendered Figures 3–9 accumulate on pages 16–21 rather than near the corresponding first discussion. |
| `supplement.tex` | 50 lines; one unnumbered section and one figure; no equation definitions, labels, bibliography, or machine-linked cross-references. |
| `supplement.pdf` | 2 A4 pages. The main article calls the item “Supplementary figure S1,” but the supplement itself renders it as “Figure 1.” |
| `results_macros.tex` | 34 lines of frozen generated macros. These supply V15 counts, effects, intervals, dispositions, secondary diagnostics, and compute accounting. They must remain byte-for-byte scientifically unchanged. |
| `references.bib` | 24 bibliography entries. No structural recommendation requires adding, deleting, or changing references in this pass. |
| Frozen figure catalog | 14 vector PDF assets with source data. Assets 5, 8, 11, and 12 are not currently embedded in either manuscript PDF. |
| Tables | None in the main article or supplement. |

### 6.2 Current numbered sequence and evidential roles

The present fourteen-section sequence contains genuine conceptual stages, but several stages are too narrow to justify top-level status:

| Current top-level section | Present role | Structural diagnosis |
|---|---|---|
| 1 Introduction | Motivation, literature position, contribution and claim boundary | Genuine top-level stage; retain. |
| 2 Stochastic process and information boundaries | System definition, complete/observed state, update authority, memory conditions | Genuine major stage, but should merge with quench and arm definitions under a broader model/protocol section. |
| 3 Collective observables | Order, effective energy, correlation, persistence, Binder, entropy, dependence, macrostate | Genuine major stage; currently overpacked because secondary spatial/persistence diagnostics interrupt the primary observable sequence. Split internally, not top-level. |
| 4 Pathwise temporal asymmetry | Estimator and thermodynamic interpretation boundary | Too narrow for a top-level section; make a central subsection of observables/inference. |
| 5 Quench protocol and nominal geometry | Intervention schedule, leave-one-cluster-out distance, recovery estimand | Two roles are mixed: protocol belongs with the system; nominal geometry and recovery estimand belong with inferential design. |
| 6 Prospective V15 design | Models, arms, pairing, pilot boundary, H1–H4, exact inference | Important but partly duplicates the arm definitions and appendical inferential details. Split between model/protocol, evidence hierarchy, and inference. |
| 7 V14 correction and audit | Correction of derived analysis, invalid historical H3, delayed audits | Must remain visible in the main narrative, but detailed mechanics should move to Appendix A. |
| 8 Results: field-quench replication | H1, H4, cross-model response, recovery and threshold behavior | Genuine result stage; merge into one broader quench/recovery section with spatial and finite-window context. |
| 9 Results: memory and temporal asymmetry | H2/H3, model heterogeneity, placebo logic, estimator caveat | Genuine result stage; retain as a broader argument-driven section. |
| 10 Spatial correlation and finite-window persistence | Secondary post-reconstruction spatial, autocorrelation, and Binder evidence | Belongs partly in the quench results and partly in Appendix C. It is not a separate primary result family. |
| 11 Out-of-sample kinetic closure | V13-fitted surrogate and its direct V14 comparison | Genuine explanatory stage; combine with the next section under reduced descriptions and closure limits. |
| 12 What the reduced representation contributes | Information beyond order and V14 representation audit | Closely coupled to closure. It should precede the surrogate within the same section. |
| 13 Interpretation and limitations | Literal/effective distinction, global limitations, preserved negative results | Genuine Discussion. Expand by subsection rather than leaving as a short catch-all. |
| 14 Conclusion | Main synthesis and future-work boundary | Genuine final stage; retain as concise Conclusions. |

### 6.3 Claim-and-caveat locks

Reorganization must move the following scientific claims together with their qualifying language. The caveats are not optional transition prose.

| Current location | Scientific role or claim | Caveat that must travel with it |
|---|---|---|
| Introduction | Statistical-mechanical variables compactly describe collective LLM-agent dynamics. | The claim is descriptive/explanatory, not performance-oriented; two models do not establish universality. |
| Stochastic process and information boundaries | The augmented simulator state induces a transition process; projected visible states need not be Markov. | Persistent history as a hidden coordinate is a mechanism for projected dependence, not proof of physical dissipation. |
| Order and effective compatibility | Magnetization, overlap, reference energy, fluctuations, spatial correlation, persistence, and Binder shape are measurable coordinates. | Reference energy is not literal energy or a Gibbs Hamiltonian; energy variance is not heat capacity; finite-window persistence is not an equilibrium correlation time; Binder values at `N=16` do not show a transition. |
| Entropy and dependence | Configuration entropy and null-adjusted dependence distinguish diversity from synchronous structure. | Short-window plug-in estimators have a large floor; adjusted values remain untruncated and may be negative. |
| Pathwise temporal asymmetry | Block-reversal KL measures coarse-grained temporal asymmetry. | It is not exact entropy production; the observable projection can be non-Markov; null dispersion and Monte Carlo error of the floor are distinct. |
| Quench protocol and nominal geometry | Field reversal and restoration generate a controlled departure from a held-out-cluster-excluded nominal manifold. | “Quench” is a finite-system protocol analogy, not a quantum or thermodynamic-limit claim; distance scales are within-model and geometry-specific. |
| Prospective V15 design | H1–H4 are frozen and tested on complete graph/environment clusters. | Agents, updates, tokens, and windows are not replicates; sign flips rely on a sign-symmetry null and are not treatment-label randomization; decoding temperature is not physical temperature. |
| V14 correction and audit | The corrected threshold and delayed audits repair derived analysis while preserving raw trajectories. | The historical maximum-minus-final H3 is structurally nonnegative and non-inferential; delayed sensitivities are not a prospective V14 experiment. |
| Field-quench results | Granite replicates field response and fixed-window distance declines during restoration. | Absolute distances cannot be pooled across models; H4 is movement toward the restored regime, not complete recovery; five of six Granite trajectories remain above threshold. |
| Memory results | Persistent history increases the frozen adjusted path-divergence contrasts in the pooled V15 estimands. | Positive paired contrasts do not establish positive absolute entropy production; Qwen's content-specific contrast is not separately confirmatory; magnitudes vary strongly by estimator. |
| Spatial/persistence results | Secondary diagnostics expose spatial covariance, temporal persistence, and order-parameter shape. | These are post-reconstruction descriptive analyses; no new hypothesis family, correlation length, critical slowing, Binder crossing, phase transition, or extra replication unit is implied. |
| Kinetic closure | A V13-fitted low-dimensional surrogate captures some response features and misses others out of sample. | It was not refit to V14/V15; surrogate failure is a closure boundary, not failure of the direct experiment; CPU size sweeps are not direct-LLM finite-size scaling. |
| Reduced representation | Non-order coordinates retain condition information that magnetization alone discards. | This is not an omnibus superiority claim for entropy or the full representation. |
| Interpretation and limitations | The study supports an effective statistical-mechanical description. | No literal physical temperature/energy/free energy, universality, thermodynamic-limit transition, exact entropy production, human benefit, controller benefit, or task-performance claim. |
| Reproducibility and data availability | Frozen artifacts, replay, source hashes, and reconstruction support reproducibility. | The original external raw tree is gone; historical call-file digest identity cannot be checked; reconstruction accounting is a measured lower bound and not a scientific outcome. |

### 6.4 Material currently in the article appendices

- **Appendix A, Frozen inferential details:** exact sign flips, cluster bootstrap, multiplicity allocation, and the V14 full-pipeline representation permutation.
- **Appendix B, Estimator sensitivity:** total-correlation nulls, rolling-window rebuilding, block-reversal floor and history-depth sensitivity.
- **Appendix C, Authority and privacy checks:** causal agent authority, transition provenance, and peer-private-state protection.

These are sound appendix roles, but the planned appendix organization should align each with the scientific main section it supports and absorb detailed provenance currently overloading the main data-availability statement.

### 6.5 Material currently in the separate supplement

The supplement contains only:

- one unnumbered section, “S1. Finite-window persistence and order-parameter shape”;
- a compact reminder of trajectory-first estimation, cluster-level uncertainty, truncation sensitivities, and Binder construction;
- frozen Figure 14, combining autocorrelation curves, fixed-lag sums, Binder values, and empirical magnetization occupancies;
- explicit “not critical slowing down / not a phase transition” caveats.

It does not contain an independent derivation, separate dataset, multimedia item, or genuinely optional narrative. Its meaning depends on equations and claims in the article.

## 7. Proposed final section and subsection hierarchy

### Main article

**1. Introduction**

- Preserve motivation, literature placement, contributions, and the descriptive/non-universal claim boundary.
- End with a short roadmap keyed to the evidence hierarchy rather than a generic methods/results preview.

**2. State-separated LLM-agent networks and quench protocols**

2.1. Augmented, local, and observed state  
2.2. Information boundaries and random-sequential update dynamics  
2.3. Markovized, persistent, and scrambled-history conditions  
2.4. Field quench, restoration, and matched controls

**3. Statistical-mechanical observables and inferential design**

3.1. Order, overlap, effective compatibility, and fluctuations  
3.2. Entropy, dependence, and the rolling macrostate  
3.3. Coarse-grained pathwise temporal asymmetry  
3.4. Spatial correlation and finite-window persistence diagnostics  
3.5. Nominal geometry, recovery estimand, and exact cluster inference

**4. Evidence hierarchy, correction, and frozen study roles**

4.1. Memory discovery and prospective replication roles  
4.2. Corrected V14 quench evidence and audit boundary  
4.3. Frozen V15 cross-model confirmation and control logic

**5. Quench response, restoration, and finite-size organization**

5.1. Cross-model field response  
5.2. Restoration and threshold re-entry  
5.3. Spatial reorganization under the quench  
5.4. Finite-window persistence and order-parameter shape

Section 5.4 should contain only the current compact numerical summary and caveats; full estimator construction and Figure 14 belong in Appendix C.

**6. Memory and projected temporal asymmetry**

6.1. Persistent versus Markovized history  
6.2. Genuine versus scrambled-history control  
6.3. Model heterogeneity and estimator dependence  
6.4. Confirmatory synthesis across H1–H4

**7. Reduced dynamical descriptions and closure limits**

7.1. Information retained beyond order  
7.2. Out-of-sample kinetic surrogate  
7.3. Captured response, failure modes, and the size-context boundary

**8. Discussion**

8.1. Literal stochastic process versus effective statistical mechanics  
8.2. Projected memory as hidden history  
8.3. Finite-size, model, topology, and estimator limitations  
8.4. Negative results, scope, and future work

**9. Conclusions**

- Keep this concise and outcome-focused.
- Do not repeat the full limitation catalogue; preserve the key Granite recovery, Qwen placebo, and estimator-boundary statements.

### Unnumbered end matter retained in the article

- Reproducibility and data availability
- AI-assisted preparation declaration
- References

The data-availability and AI statements remain in the main article. Detailed administrative provenance may later be shortened and cross-referenced to Appendix D; no wording is changed in this pass.

### Article appendices

**Appendix A. Study chronology, V14 correction, and frozen inferential details**

A.1. Training-only threshold correction and historical H3 invalidation  
A.2. Delayed V14 permutation and observable-deletion audit  
A.3. Frozen H1–H4 tests, multiplicity, and cluster bootstrap

**Appendix B. Estimator, control, and closure sensitivities**

B.1. Rolling dependence and marginal-shift null construction  
B.2. Block-reversal estimator and floor sensitivity  
B.3. Scrambled-history prompt-balance control  
B.4. Reduced-model size context

**Appendix C. Finite-window persistence and Binder diagnostics**

C.1. Autocorrelation construction and truncation  
C.2. Binder trajectory-first and pooled-moment sensitivity  
C.3. Persistence, Binder, and occupancy display

**Appendix D. Implementation integrity, authority, privacy, and provenance**

D.1. Agent authority and information-boundary checks  
D.2. Arm independence, replay, and artifact integrity  
D.3. Reconstruction and technical provenance details

## 8. One-to-one mapping from current headings to proposed destinations

| Current heading | Current role | Proposed destination | Action: retain, merge, split, move, or retitle | Scientific content preserved | Transition text eventually required |
|---|---|---|---|---|---|
| Introduction | Motivation, literature, contribution, scope | Section 1 | Retain | All motivation, citations, and claim boundaries | Add only a final roadmap into Sections 2–9. |
| Stochastic process and information boundaries | Defines the system and observation map | Section 2 | Merge and retitle | Complete/observed state distinction; scheduler and authority | Bridge from motivation to the precise interacting process. |
| Local agent state | Defines `Y_t`, `Xi_t`, private state, updates, packets | Sections 2.1–2.2 | Split and retitle | All state variables, update semantics, repair behavior, fingerprints | One sentence separating state definition from transition/update mechanics. |
| Markovized, persistent, and scrambled history | Defines memory arms and hidden-history mechanism | Section 2.3 | Retain and retitle | Genuine, Markovized, and scrambled conditions; no-dissipation caveat | Transition from update rule to the controlled history interventions. |
| Collective observables | Wrapper for all reduced variables | Section 3 | Merge and retitle | Entire observable family | Opening paragraph distinguishing observables, estimators, and inference. |
| Order and effective compatibility | Order, overlap, reference energy, fluctuations, correlation, autocorrelation, Binder | Sections 3.1 and 3.4 | Split | Every equation and every physical-interpretation caveat | A short signpost that moves secondary spatial/persistence diagnostics to 3.4 without changing their status. |
| Entropy and dependence | Entropy, mutual information, total correlation, null audit | Section 3.2 | Retain and move | Raw/null/adjusted definitions and finite-sample caveat | Link from order/fluctuation coordinates to uncertainty/dependence coordinates. |
| Rolling macrostate | Defines reduced representation `Z_t` | Section 3.2 | Merge | Full coordinate vector and 3/5/7-sweep sensitivity statement | Explain that the macrostate gathers, but does not complete, the preceding observables. |
| Pathwise temporal asymmetry | Defines block-reversal KL and exact-Markov contrast | Section 3.3 | Move and retitle | Estimator, floors, sensitivity grid, current comparison, coarse-graining caveat | Link hidden history in Section 2.3 to the projected diagnostic. |
| Quench protocol and nominal geometry | Quench schedule, LOCO geometry, response descriptors, recovery contrast | Sections 2.4 and 3.5 | Split | Schedule, fitting exclusions, distance descriptors, recovery equation | One bridge from intervention mechanics to the inference built on those trajectories. |
| Prospective V15 design | Models, software pins, clusters, arms, pilot boundary, H1–H4 | Sections 2.3–2.4, 3.5, and 4.3 | Split | All frozen design facts, pairing, hypotheses, error allocation, exact-test qualification | A study-role paragraph preventing protocol definition from duplicating the evidence hierarchy. |
| V14 correction and audit | Corrected threshold, invalid H3, delayed analyses | Section 4.2 and Appendix A | Split | Raw-trajectory immutability, corrected result, invalidation, delayed-status caveat | Introduce why V14 is retained as corrected evidence but not pooled into V15 confirmation. |
| Results: field-quench replication | H1/H4, cross-model response and recovery | Sections 5.1–5.2 | Split and retitle | All effects, intervals, sign patterns, threshold re-entry, incomplete Granite recovery | Bridge from evidence hierarchy to prospective outcomes; then from response peak to restoration. |
| Results: memory and temporal asymmetry | H2/H3, placebo logic, model heterogeneity, estimator sensitivity | Sections 6.1–6.3 | Split and retitle | All pooled and model-specific estimates and all entropy-production boundaries | Distinguish the current-state comparison, content-specific placebo comparison, and sensitivity boundary. |
| Spatial correlation and finite-window persistence | Secondary connected-correlation, autocorrelation, Binder results | Sections 5.3–5.4 and Appendix C | Split | All descriptive estimates, intervals, occupancy context, and post-reconstruction status | Explicitly mark these as secondary coordinates of the same quench trajectories, not extra tests. |
| Out-of-sample kinetic closure | Direct-surrogate comparison and closure failures | Sections 7.2–7.3 and Appendix B.4 | Split and retitle | V13-only fitting, shared-coordinate comparison, failure modes, CPU size boundary | Transition from what the representation retains to whether a simple dynamics closes it. |
| What the reduced representation contributes | Explains information beyond order; V14 representation audit | Section 7.1 | Move and retitle | Complementarity of observables and the non-omnibus caveat | Introduce the reduced-description question before the surrogate test. |
| Interpretation and limitations | Literal/effective distinction, six limitations, negative boundaries | Sections 8.1–8.4 | Split and retitle | Every interpretive limit and preserved negative result | Topic sentences for interpretation, hidden history, limitations, and negative scope. |
| Conclusion | Final synthesis and future-work boundary | Section 9 | Retain and retitle | All supported conclusions and explicit boundary results | Only a concise handoff from Discussion; no new claim. |
| Reproducibility and data availability | Data statement plus extensive reconstruction/accounting detail | Main unnumbered statement and Appendix D | Split | Access conditions, hashes, replay, lower-bound accounting, lost-tree caveat, arm independence, digest correction | Later add a compact pointer from the required main statement to Appendix D. |
| AI-assisted preparation declaration | Required disclosure | Main unnumbered statement | Retain | Entire current statement | None beyond placement consistency. |
| Appendix A: Frozen inferential details | Exact tests, bootstrap, multiplicity, V14 permutation | Appendices A.2–A.3 | Split and retitle | All inferential mechanics and prospective allocation | Cross-references from Sections 3.5, 4.2, and 6.4. |
| Appendix B: Estimator sensitivity | Dependence and pathwise estimator sensitivities | Appendices B.1–B.2 | Move and retitle | All current sensitivity construction and caveats | Cross-references from Sections 3.2–3.3 and 6.3. |
| Appendix C: Authority and privacy checks | Causal authority and privacy integrity | Appendix D.1 | Move, merge, and retitle | All current authority/privacy checks | Cross-reference from Section 2.2. |
| Supplement S1: Finite-window persistence and order-parameter shape | Context for autocorrelation, Binder, occupancy, and estimator sensitivity | Appendix C | Move and retitle | Entire supplement text and Figure 14; no numerical or claim change | Replace “Supplementary figure S1” with an article-appendix reference and introduce it from Section 5.4. |

## 9. Figure-placement map for frozen Figures 1–14

“Figure N” below identifies the frozen asset/catalog number, not a guaranteed future printed number. Printed numbering should be allowed to follow the approved article order; LaTeX labels, not hard-coded numbers, must carry references.

| Frozen asset | Scientific role | Current placement and first textual reference | Proposed destination | Disposition and preservation requirement |
|---|---|---|---|---|
| **Figure 1 — augmented-state architecture** | Information boundaries and augmented-to-macrostate projection | Declared after the Introduction at `main.tex:125`; no explicit `\ref` in prose | Section 2.1 or 2.2, immediately after the complete/observed-state distinction | Retain in main text. Keep `fig:architecture`; add an explicit prose citation before the float. |
| **Figure 2 — memory discovery/replication** | Separates V12 discovery, V13 prospective replication, and V15 extension | Declared at `main.tex:347` at the end of pathwise formalism; no explicit prose `\ref` | Section 4.1 | Move. Keep `fig:memory`; cite before the display and preserve the no-retroactive-pooling caption. |
| **Figure 3 — corrected V14 quench time series** | Historical corrected quench/counter-quench phenomenology | Declared at `main.tex:380`; no explicit prose `\ref`; renders on main-PDF page 16 | Section 4.2 | Retain in main text as the compact corrected V14 evidence. Keep `fig:v14series`; add explicit first citation. |
| **Figure 4 — V14 cluster recovery** | Cluster heterogeneity and corrected training-only thresholds | Declared at `main.tex:386`; no explicit prose `\ref`; renders on page 17 | Appendix A.1 | Move to appendix to prevent the historical audit from dominating the prospective narrative. Keep `fig:v14recovery`; main Section 4.2 should point to it. |
| **Figure 5 — V14 delayed audit** | Rolling-window, null-floor, and full-pipeline permutation audit | Catalogued as supplementary; absent from both current manuscript PDFs and never cited | Appendix A.2 | Include existing frozen asset without redesign. Add a stable label such as `fig:v14audit` and a main/appendix lead-in preserving delayed-prespecification status. |
| **Figure 6 — cross-model quench** | Qwen/Granite field response in model-specific geometries | First explicit citation at `main.tex:461`; declared at line 473; currently renders as printed Figure 5 on page 17 | Section 5.1 | Retain in main text. Keep `fig:crossquench`; keep the within-model-scale caveat adjacent. |
| **Figure 7 — V15 memory controls** | Cluster-level persistent-minus-Markovized and persistent-minus-scrambled contrasts | Declared at `main.tex:527`; no explicit prose `\ref`; renders as printed Figure 6 on page 18 | Section 6.1–6.2 boundary | Retain in main text. Keep `fig:v15memory`; explicitly cite before display and preserve prompt-control limitations. |
| **Figure 8 — path-reversal sensitivity** | Block-length and pseudocount dependence of adjusted divergence | Catalogued as supplementary; absent from both PDFs and never cited | Appendix B.2 | Include existing frozen asset. Add a stable label such as `fig:path-sensitivity`; Section 6.3 should cite it. |
| **Figure 9 — confirmatory effects** | H1–H4 effects and intervals without mixing units | Declared at `main.tex:676`; no explicit prose `\ref`; currently renders as printed Figure 9 on page 21 | Section 6.4 | Move earlier to close the confirmatory result sequence. Keep `fig:effects`; explicitly cite and preserve separate axes/units. |
| **Figure 10 — direct surrogate quench** | Out-of-sample direct-versus-surrogate comparison | Declared at `main.tex:622`; no explicit prose `\ref`; currently renders as printed Figure 8 on page 20 | Section 7.2 | Retain in main text. Keep `fig:surrogate`; explicitly cite before the float and keep the “closure failure, not experiment failure” caveat. |
| **Figure 11 — surrogate size sensitivity** | Effective-model size context | Catalogued as supplementary; absent from both PDFs and never cited | Appendix B.4 | Include existing frozen asset in the article appendix. Add `fig:surrogate-size`; state prominently that this is not direct-LLM finite-size scaling. |
| **Figure 12 — memory prompt balance** | Validates approximate token-length matching for the scrambled-history control | Catalogued as supplementary; absent from both PDFs and never cited | Appendix B.3 | Include existing frozen asset. Add `fig:prompt-balance`; Section 4.3 or 6.2 should cite it without implying turn-by-turn token identity. |
| **Figure 13 — graph-distance correlations** | Spatial covariance on the reciprocal modular graph | First explicit citation at `main.tex:560`; declared at line 574; currently renders as printed Figure 7 on page 19 | Section 5.3 | Retain in main text. Keep `fig:correlations`; preserve community-only alignment and cluster-unit caveats. |
| **Figure 14 — persistence and Binder** | Autocorrelation, truncated sums, Binder values, and occupancy context | Main text calls “Supplementary figure S1” at `main.tex:585`; supplement has no label and renders it as “Figure 1” | Appendix C.3 | Move into article Appendix C. Add `fig:persistence-binder`; remove the separate-supplement naming mismatch; preserve every finite-window/no-transition caveat. |

### Figure-numbering rule for implementation

Do not attempt to preserve current printed numbers by hand. After final placement:

1. keep stable semantic labels;
2. allow LaTeX to number figures in textual order, including any appendix convention selected by the class;
3. update every prose reference through `\ref`;
4. verify that the frozen asset filenames and figure catalog remain traceable even when printed numbering changes.

## 10. Equation and cross-reference preservation checklist

### 10.1 Current equation inventory

The current article has 15 numbered display environments but only five explicit equation labels.

| Current printed equation | Role | Source | Existing label | Proposed destination | Preservation action |
|---|---|---|---|---|---|
| (1) | Observable microscopic projection `Y_t` | `main.tex:142` | None | 2.1 | Move verbatim; add a semantic label only if later referenced. |
| (2) | Complete augmented state `Xi_t` | `main.tex:148` | None | 2.1 | Move verbatim and keep directly after equation (1). |
| (3) | Different future laws under different memories | `main.tex:176` | None | 2.3 | Move with the non-Markov/no-dissipation explanation. |
| (4) | Belief order, action order, and overlap | `main.tex:196` | None | 3.1 | Move verbatim. |
| (5) | Symmetric-layer reference compatibility | `main.tex:203` | `eq:href` | 3.1 | Preserve label exactly and keep the effective-energy caveat adjacent. |
| (6) | Belief susceptibility | `main.tex:219` | None | 3.1 | Move verbatim. |
| (7) | Connected graph-distance correlation | `main.tex:226` | `eq:connected-correlation` | 3.4 | Preserve label exactly and move with Figure 13 methods. |
| (8) | Magnetization autocorrelation and truncated sum | `main.tex:241` | `eq:autocorrelation` | 3.4 | Preserve label exactly; Appendix C must reference the label, not “equation (8).” |
| (9) | Binder cumulant | `main.tex:261` | `eq:binder` | 3.4 | Preserve label exactly; Appendix C must reference the label, not “equation (9).” |
| (10) | Configuration entropy | `main.tex:280` | None | 3.2 | Move verbatim. |
| (11) | Total correlation | `main.tex:288` | None | 3.2 | Move verbatim with the shift-null caveat. |
| (12) | Rolling macrostate `Z_t` | `main.tex:305` | None | 3.2 | Move verbatim after component definitions. |
| (13) | Block-reversal KL per attempted update | `main.tex:318` | `eq:pathkl` | 3.3 | Preserve label exactly and keep floor/interpretation caveats with it. |
| (14) | Exact Markov current used only as a contrast | `main.tex:337` | None | 3.3 | Move verbatim; do not let proximity imply that it is estimated for the LLM process. |
| (15) | Fixed early-to-late restoration contrast | `main.tex:373` | None | 3.5 | Move verbatim with the statement that it may have either sign. |

### 10.2 Preservation checklist

- [ ] Move equations as complete LaTeX blocks; do not retype, simplify, reorder internally, or introduce new statistical-mechanical quantities.
- [ ] Preserve existing labels exactly: `eq:href`, `eq:connected-correlation`, `eq:autocorrelation`, `eq:binder`, and `eq:pathkl`.
- [ ] Do not preserve printed equation numbers manually; allow renumbering after reorganization.
- [ ] Replace the supplement's hard-coded “equations (8)–(9)” with label-based references after it becomes Appendix C.
- [ ] Keep the explanatory paragraphs immediately following equations (5), (8), (9), (13), (14), and (15) with those equations; they contain indispensable interpretation limits.
- [ ] Keep the full `Z_t` coordinate order unchanged because downstream nominal-geometry prose assumes that exact representation.
- [ ] Keep `results_macros.tex` unchanged and retain the temporary digit-catcode mechanism that loads its historical macro names.
- [ ] Preserve all existing figure-label strings; add semantic labels for Figures 5, 8, 11, 12, and 14 before creating references.
- [ ] Replace every unlinked “Figure …” mention with `\ref`-based text where possible.
- [ ] Add stable `sec:` labels to the approved sections/subsections before moving cross-referenced prose.
- [ ] Check for duplicate and undefined labels only during the later approved implementation/compile pass, not during this audit.
- [ ] Verify that all 24 current bibliography keys remain reachable and that no citation is lost during block movement.
- [ ] Verify that no frozen numerical macro expands differently after movement.

## 11. Recommendation for the current supplement

### Decision

**Move the entire current supplement into article Appendix C and discontinue the separate two-page supplement.**

### Assessment against the four possible dispositions

1. **Is the diagnostic needed to understand or appropriately qualify a main-text claim? — Yes.**  
   The main article reports numerical autocorrelation and Binder contrasts, distinguishes persistence from path reversal, and relies on occupancy plots to prevent misreading extreme Binder values. The figure is therefore part of the claim's qualification, not merely optional illustration.

2. **Should it become a results subsection? — Partly.**  
   Section 5.4 should retain the compact descriptive outcome and the strongest caveats. Putting the six-panel diagnostic itself in the main results would interrupt the primary H1–H4 flow and give a post-reconstruction descriptive extension excessive visual weight.

3. **Should it become a main-paper appendix? — Yes.**  
   Appendix C is the best balance: it remains in the article PDF, Version of Record, and normal peer-review object while being clearly labeled as a secondary finite-window diagnostic.

4. **Is it peripheral enough to remain separate supplementary material? — No.**  
   It is short, prose-dependent on main equations, and directly protective against overclaiming critical slowing or a phase transition. IOP's policy weighs in favor of article inclusion for integral qualification.

### Required structural safeguards

- Preserve the post-reconstruction/descriptive status.
- Preserve trajectory-first calculation and complete-cluster resampling.
- Preserve undefined zero-variance cases.
- Preserve fixed truncation and early/late/pooled sensitivity language.
- Preserve occupancy alongside Binder values.
- Preserve the explicit prohibitions on equilibrium correlation-time, critical-slowing-down, Binder-crossing, and phase-transition interpretations.
- Correct the present “Supplementary figure S1” versus rendered “Figure 1” mismatch through an appendix label.

## 12. Main text, appendices, and later administrative editing

| Material | Recommended location | Reason |
|---|---|---|
| System state, information boundaries, update rule, history arms, quench/restoration | Main Sections 2.1–2.4 | Required to understand what process was studied. |
| Core observable equations and interpretation limits | Main Sections 3.1–3.4 | Integral statistical-mechanical formalism. |
| Nominal geometry, independent unit, hypotheses, multiplicity, exact-test qualification | Main Section 3.5 with detail in Appendix A.3 | Readers need the inferential design before results; enumeration details can be appendical. |
| Discovery/replication roles and concise V14 correction | Main Section 4 | Essential to prevent retrospective pooling and misuse of invalid V14 H3. |
| Full V14 audit mechanics and Figures 4–5 | Appendix A | Important for review, but secondary to V15's prospective result narrative. |
| H1–H4 outcomes, model decomposition, incomplete recovery, spatial result, compact persistence/Binder summary | Main Sections 5–6 | Core scientific evidence and necessary boundaries. |
| Path-reversal sensitivity and Figure 8 | Appendix B.2 | Integral robustness evidence, but too detailed for the primary result flow. |
| Prompt-token balance and Figure 12 | Appendix B.3 | Validates the scrambled-history control while remaining a technical diagnostic. |
| Surrogate size context and Figure 11 | Appendix B.4 | Useful closure boundary; not direct-LLM finite-size scaling. |
| Current supplement and Figure 14 | Appendix C | Integral claim qualification best kept in the Version of Record. |
| Authority/privacy checks | Appendix D.1, cited from Section 2.2 | Establishes causal agent independence without interrupting model exposition. |
| Concise data-availability statement | Main unnumbered end matter | Required by IOP policy and needed by readers. |
| AI-assisted preparation declaration | Main unnumbered end matter | Keep visible and unchanged during the structure pass. |
| Exact model revisions and essential software environment | Main Section 4.3 or concise data statement | Needed to identify the frozen experiment. |
| Long hash lists, complete file/byte counts, orphan-call accounting, reconstruction GPU-hour comparison, bytecode-cache digest correction | Appendix D.2–D.3 or later administrative condensation | Technically important provenance, but currently overwhelms the main data statement. Move/shorten only in a later authorized administrative-editing pass. |
| Figure source CSVs, result tables, configs, tests, analysis code | Repository/data package only | Frozen source material; no manuscript-structure change should touch it. |

## 13. Structural risks and mitigations

| Risk | Where it arises | Consequence | Required mitigation |
|---|---|---|---|
| Duplicate protocol exposition | Sections 2.3–2.4, 3.5, and 4.3 | Arms, pairing, and H1–H4 could be explained three times. | Assign distinct roles: Section 2 defines conditions; Section 3 defines estimators/inference; Section 4 defines study chronology and confirmatory status. |
| Results shown before their status is clear | Current Figure 2 and V14 figures appear amid formalism/protocol | Readers may pool discovery, corrected V14, and V15 evidence. | Put the evidence-hierarchy section before direct V15 results and keep Figure 2 there. |
| V14 audit dominates the paper | Two main V14 figures plus detailed correction prose precede V15 results | The paper may read as an audit report rather than a statistical-mechanics study. | Keep one compact V14 time-series figure in Section 4.2; move cluster/audit Figures 4–5 to Appendix A. |
| Caveats detach from claims during block movement | Effective energy, path KL, recovery, Binder, surrogate | Reorganization could silently strengthen claims. | Move each claim-and-caveat block atomically using the locks in Section 6.3. |
| Spatial and persistence material is duplicated | Current observable definitions, Section 10, and supplement | Repeated exposition disrupts the quench-to-memory flow. | Definitions in 3.4, concise outcomes in 5.3–5.4, estimator/figure detail in Appendix C. |
| Printed figure numbers become misleading | Asset filenames, current source order, and appendix moves differ | Hard-coded numbers or asset names could point to the wrong display. | Use semantic labels and automatic numbering; treat asset numbers as provenance only. |
| Current floats separate evidence from text | Main PDF pages 16–21 | Reviewers encounter figures long after the arguments they support. | Insert explicit first citations, place floats after those citations, and verify rendered proximity in the later compile pass. |
| Supplement cross-reference is already inconsistent | Main says “S1”; supplement renders “Figure 1” | Ambiguous citation and fragile numbering. | Import as Appendix C with a single article label. |
| Unlabeled equations renumber silently | 10 of 15 displayed equations lack labels | Literal equation numbers can become stale. | Avoid number-based prose; add labels only where cross-reference is needed. |
| Figure 9 currently arrives after closure exposition | Confirmatory synthesis appears after Section 12 | It summarizes H1–H4 too late. | Move it to Section 6.4 before reduced-description analysis. |
| Reduced representation and kinetic surrogate are presented in reverse explanatory order | Current Sections 11 then 12 | Readers see closure failure before the purpose of the representation is stated. | Put “information retained beyond order” first, surrogate second, failure modes third. |
| Administrative provenance overwhelms the scientific ending | Current data statement spans detailed reconstruction incidents and hashes | The transition from Conclusions to appendices is diluted. | Retain a compliant concise statement in main text and later move technical detail to Appendix D; do not edit it in this pass. |
| Adding four currently unembedded frozen figures expands the article | Figures 5, 8, 11, 12 | Page count increases and captions need careful provenance. | Confine them to appendices, reuse only frozen assets and verified catalog roles, and add no new analysis or claims. |
| Separate Discussion and Conclusions can repeat | Proposed Sections 8–9 | Redundant summary could recreate narrow sections. | Discussion performs interpretation/limits; Conclusions state only the compact answer and future boundary. |

## 14. Precise implementation plan for the next pass

This plan is for a later **author-approved, structure-only implementation pass**. It must not start until the map is approved.

1. **Re-verify repository state.** Confirm the current branch, clean starting tree apart from this audit note, and unchanged hashes for `main.tex`, `supplement.tex`, `results_macros.tex`, all figure PDFs, source CSVs, and bibliography.
2. **Create a movement ledger.** Record the source line span of every current section/subsection, every equation block, every figure declaration, every result-macro use, and every caveat paragraph.
3. **Add semantic section anchors.** Introduce stable `sec:` labels for the approved hierarchy before moving blocks. Do not change scientific prose.
4. **Build the new heading skeleton in `main.tex`.** Replace only sectioning commands and insert minimal placeholder transitions. Do not convert document class or alter packages.
5. **Move model/protocol blocks atomically.** Form Sections 2.1–2.4 from current local state, information boundaries, history arms, quench schedule, and matched-arm material.
6. **Move formalism/inference blocks atomically.** Form Sections 3.1–3.5, preserving all equations, existing labels, coordinate order, and adjacent caveats.
7. **Construct the evidence hierarchy.** Move Figure 2 and discovery/replication language to 4.1; keep a concise V14 correction plus Figure 3 in 4.2; place the frozen V15 model/cluster/control roles in 4.3.
8. **Construct the direct-results sequence.** Move quench/restoration/spatial material into Section 5 and memory/placebo/heterogeneity material into Section 6. Place Figure 9 at the end of 6.4.
9. **Reorder the reduced-description argument.** Move “What the reduced representation contributes” ahead of the kinetic surrogate, then keep surrogate successes/failures and size limitations in 7.2–7.3.
10. **Expand the Discussion hierarchy without rewriting claims.** Distribute existing interpretation, six limitations, negative findings, and future-work sentences across 8.1–8.4. Keep Section 9 concise.
11. **Rebuild appendices A–D.** Move existing appendix blocks, detailed V14 correction mechanics, and technical provenance according to the map. Include frozen Figures 4, 5, 8, 11, and 12 at their approved appendix destinations without redesign or regeneration.
12. **Absorb the supplement.** Copy its complete scientific text and Figure 14 declaration into Appendix C, preserve all caveats, add `fig:persistence-binder`, and replace hard-coded equation/figure numbers with label references.
13. **Retain end statements.** Keep data availability and AI-assisted preparation in main-article end matter. Do not shorten them during the structure-only pass; merely mark paragraphs proposed for later administrative editing.
14. **Add only necessary transitions.** Each transition should state role or sequence, not introduce a result, equation, citation, interpretation, or claim. Track every newly written sentence in a separate review list.
15. **Cross-reference audit before compilation.** Check all `sec:`, `eq:`, and `fig:` labels; verify every figure has a first prose citation; remove literal “Supplementary figure S1” and hard-coded equation numbers.
16. **Compile only after movement is complete and approved for verification.** Confirm zero undefined/duplicate references, automatic figure/equation order, figure proximity, appendix numbering, and unchanged macro expansions. Compilation is explicitly outside this first audit pass.
17. **Scientific-diff verification.** Compare extracted text, equations, captions, macro expansions, and all numerical strings against V15. The allowed delta is headings, order, cross-reference syntax, and approved transition text only.
18. **Author review gate.** Present the reordered PDF and the movement/transition ledger before any prose polishing, citation work, template conversion, commit, or push.

## 15. Structural decisions requiring author approval

The audit makes recommendations, but implementation should begin only after the author approves these visible choices:

1. **Approve the nine-section hierarchy**, including a dedicated evidence-hierarchy Section 4 and separate Discussion and Conclusions.  
   **Recommendation:** approve as written.

2. **Approve the main/appendix figure split:** main-text assets 1, 2, 3, 6, 7, 9, 10, and 13; appendix assets 4, 5, 8, 11, 12, and 14.  
   **Recommendation:** approve. This preserves all 14 frozen figures while keeping the prospective narrative dominant.

3. **Approve elimination of the separate supplement and migration of its full contents to Appendix C.**  
   **Recommendation:** approve. This is the clearest consequence of IOP's integral-material policy and the diagnostic's logical dependency on main claims.

4. **Approve deferring all shortening of the data-availability/provenance statement.**  
   **Recommendation:** approve. Structure can be implemented first; administrative condensation should be a separate, explicitly reviewed pass.

No manuscript material should move until these choices are approved.
