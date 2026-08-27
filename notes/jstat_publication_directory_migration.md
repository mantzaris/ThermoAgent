# JSTAT publication-directory migration

## 1. Starting branch and commit

- Branch: `jstat-paper-structure-v16`
- Starting local and remote commit:
  `03c0f2e68264679b2a4e748881cd574e2a853769`
- `git fetch origin` confirmed that the local branch and
  `origin/jstat-paper-structure-v16` were aligned and that the tracked worktree
  was clean.
- `paper/JSTAT` did not exist before this pass. The two copy-named historical
  snapshots were present and were not modified.
- No RunPod connection, simulation, trajectory generation, scientific
  analysis, or result recalculation was performed.

## 2. Old and new manuscript paths

The single active manuscript moved with Git history from `paper/jstat_v15` to
`paper/JSTAT`. The exact tracked file movements were:

| Old path | New path |
|---|---|
| `paper/jstat_v15/main.tex` | `paper/JSTAT/main.tex` |
| `paper/jstat_v15/main.pdf` | `paper/JSTAT/main.pdf` |
| `paper/jstat_v15/references.bib` | `paper/JSTAT/references.bib` |
| `paper/jstat_v15/results_macros.tex` | `paper/JSTAT/results_macros.tex` |
| `paper/jstat_v15/render_publication_figures.py` | `paper/JSTAT/render_publication_figures.py` |

The active build script moved from `scripts/build-statmech-v15-paper.sh` to
`scripts/build-jstat-paper.sh`. The unqualified path `paper/jstat_v15` no
longer exists. Historical directories `paper/jstat_v9` through
`paper/jstat_v14`, `paper/jstat_v15 (copy)`, and
`paper/jstat_v15 (another copy)` remain in place.

## 3. Final publication-directory tree

```text
paper/JSTAT/
├── README.md
├── main.pdf
├── main.tex
├── references.bib
├── render_publication_figures.py
├── results_macros.tex
└── figures/
    ├── figure_manifest.csv
    ├── figure01_architecture.pdf
    ├── figure02_memory_evidence_stages.pdf
    ├── figure03_corrected_quench_time_series.pdf
    ├── figure04_cross_model_quench.pdf
    ├── figure05_graph_distance_correlations.pdf
    ├── figure06_memory_controls.pdf
    ├── figure07_confirmatory_effects.pdf
    ├── figure08_direct_surrogate_quench.pdf
    ├── figure09_cluster_recovery.pdf
    ├── figure10_delayed_audit.pdf
    ├── figure11_path_reversal_sensitivity.pdf
    ├── figure12_prompt_balance.pdf
    ├── figure13_surrogate_size_context.pdf
    └── figure14_persistence_binder.pdf
```

No source CSV, auxiliary LaTeX file, unused figure, or analysis output was
copied into the publication directory.

## 4. Printed-figure mapping and provenance

`paper/JSTAT/figures/figure_manifest.csv` records the complete mapping and both
hashes. Every local hash equals its canonical hash.

| Printed | Local publication file | Semantic label | Canonical repository asset | SHA-256 |
|---:|---|---|---|---|
| 1 | `figure01_architecture.pdf` | `fig:architecture` | `figure01_augmented_state_architecture.pdf` | `21cc957a2c27712b16eee58a35ee6c833109f9005241eadab1c88100521e460b` |
| 2 | `figure02_memory_evidence_stages.pdf` | `fig:memory` | `figure02_memory_discovery_replication.pdf` | `bd04b93cd999886389e524870d1b577da57b5a7afc4bd1959312b036f411b140` |
| 3 | `figure03_corrected_quench_time_series.pdf` | `fig:v14series` | `figure03_v14_quench_time_series.pdf` | `307cdf8aa2cbd9e87f80494e224a30faad86ca97a188bc70905c5e7fb0fb25c7` |
| 4 | `figure04_cross_model_quench.pdf` | `fig:crossquench` | `figure06_cross_model_quench.pdf` | `b7bd3b5367ebc9ce22271c262f40b86ad255921dd83401c1a9198b68cd1b662b` |
| 5 | `figure05_graph_distance_correlations.pdf` | `fig:correlations` | `figure13_graph_distance_correlations.pdf` | `6271b8b469a52caa8d263bc23995dccde5249b51bd0ac864395c9393c3df3f13` |
| 6 | `figure06_memory_controls.pdf` | `fig:v15memory` | `figure07_v15_memory_controls.pdf` | `9e09c0218736c903c8b6afb860db31ef81347590bcc94f72c6b6189bbefb47ed` |
| 7 | `figure07_confirmatory_effects.pdf` | `fig:effects` | `figure09_confirmatory_effects.pdf` | `8975f000f27d181ba9c99141657f6abf16fae862dba9506e059b9abde4b7714d` |
| 8 | `figure08_direct_surrogate_quench.pdf` | `fig:surrogate` | `figure10_direct_surrogate_quench.pdf` | `22b9ac164083292048c9ccf2aed5e94fb3864583aab5103de61974b244f0cde8` |
| 9 | `figure09_cluster_recovery.pdf` | `fig:v14recovery` | `figure04_v14_cluster_recovery.pdf` | `1cf7771a3a539c9aef45e4f464b4e63e85d1ebbdfbf7eab7e58a7108e2cfbe4e` |
| 10 | `figure10_delayed_audit.pdf` | `fig:v14audit` | `figure05_v14_delayed_audit.pdf` | `ab806f1aafa705e856af81417fbf4db925668218084ef81a9f877d00b56f7895` |
| 11 | `figure11_path_reversal_sensitivity.pdf` | `fig:path-sensitivity` | `figure08_path_reversal_sensitivity.pdf` | `eeefbf62ff5e51b93d757e0112042246e5e834b8a9121184876d1d07920f2dd5` |
| 12 | `figure12_prompt_balance.pdf` | `fig:prompt-balance` | `figure12_memory_prompt_balance.pdf` | `d34c5017af7eff7ea065f3cfe1730db9ffd6224c5c9b56694b8de5648c665285` |
| 13 | `figure13_surrogate_size_context.pdf` | `fig:surrogate-size` | `figure11_surrogate_size_sensitivity.pdf` | `2df1742bab61f4076300d1433c3bb5a05a7c8540f099b8b211417632f509d7d4` |
| 14 | `figure14_persistence_binder.pdf` | `fig:persistence-binder` | `figure14_persistence_and_binder.pdf` | `3e8f383bacc00c77221bf34ae90bd4ee5741cd60c3c1bef89cd5ad7c6bebcc01` |

The manifest verifier found exactly 14 local PDFs, no unexpected PDF, one
`\resultfigure` inclusion per local filename, the expected semantic labels,
matching local/canonical hashes, and no external figure root in `main.tex`.

## 5. Figure hash results

Only the three authorized canonical PDFs changed:

| Canonical figure | Starting SHA-256 | Final SHA-256 | Reason |
|---|---|---|---|
| Architecture | `9cabba5504c80d0e37f975f8c32a7d918fc0a777dd7d4b522f330b90e7999171` | `21cc957a2c27712b16eee58a35ee6c833109f9005241eadab1c88100521e460b` | Arrow routing only |
| Memory evidence | `d9c64a4b6102abac3cd702f8367a71469bdeb563864ffc3acccf46cc134143e2` | `bd04b93cd999886389e524870d1b577da57b5a7afc4bd1959312b036f411b140` | Display-label replacements only |
| Cluster recovery | `28b051e46b61bef8027a2b52b2bd7599304b11a7b1089e9c26707e352ae7e11a` | `1cf7771a3a539c9aef45e4f464b4e63e85d1ebbdfbf7eab7e58a7108e2cfbe4e` | Legend-label replacements only |

All other 11 canonical figure-PDF hashes match the starting baseline exactly.
All 14 canonical source-data CSV hashes also match the starting baseline,
including the three source tables associated with the regenerated figures.
The catalog and figure-generation metadata were refreshed only to record the
three new PDF hashes and the targeted generation time.

## 6. Figure 1 routing repair

The canonical `_architecture` implementation retains component patch
references and its existing rendered text-containment assertion. It now also
retains component dimensions and calculates anchors just outside the box
edges. Two cubic `FancyArrowPatch` paths route evaluator-bound information:

- The belief/action-packet arrow bends to the left of the typed-action box and
  lands at the left top anchor of the observable-projection box.
- The typed-action arrow uses a shorter route to a separate right top anchor.

The arrows no longer pass through or against the typed-action box, their
arrowheads do not overlap, and both heads remain fully visible above the box
z-order. `Local LLM transition`, all prior label-containment improvements, and
the scientific information-flow relationships are unchanged.

## 7. Figure 2 and Figure 9 label replacements

Printed Figure 2 maps internal source identifiers during rendering only:

| Internal source value | Printed label |
|---|---|
| `V12_discovery` | Exploratory discovery |
| `V13_replication` | Prospective replication |
| `V15_granite` | Cross-model Granite |
| `V15_qwen` | Cross-model Qwen |

Printed Figure 9 uses the stable mapping `g0` through `g5` to `Cluster 1`
through `Cluster 6`. The underlying study and cluster identifiers remain in
the canonical source data and analysis outputs.

Text extracted from all 14 publication-facing PDFs contains no reader-visible
`V12`, `V13`, `V14`, `V15`, `V14Q_g*`, or similar internal phase/version label.
Valid scientific labels and model names remain. Raster inspection confirms the
four Figure 2 labels and all six Figure 9 legend labels are readable at the
manuscript scale.

## 8. Plotted-value preservation

No estimate, interval, point position, trajectory, time coordinate, color,
ordering, recovery calculation, or source-data value changed. The source CSV
hashes for canonical Figures 1, 2, and 4 remained, respectively:

- `c16d75a74125cd3ba74942f64dc543a132d2a355e60870799fd34e1632220e87`
- `9746b9cab47c58950fac96da2f5cf1b6af29594d5a6cbd2a8795a972510164e0`
- `9cce0bfd41c28b1d3e727a597202a76c0e046cf10cba4349350fa90f571d47b2`

The complete `main.tex` diff against the starting commit consists only of the
local figure-root definition and 14 figure filenames. The Introduction and
Discussion section hashes are byte-identical to the starting source.

## 9. Active scripts, tests, and documentation

The following active interfaces were updated:

- `scripts/build-jstat-paper.sh` builds `paper/JSTAT/main.tex`, checks all 14
  local figures before starting, runs LaTeX/BibTeX through `latexmk`, leaves
  `paper/JSTAT/main.pdf`, and removes auxiliary build files.
- `scripts/verify-jstat-paper-assets.sh` checks the manifest, figure set,
  hashes, canonical provenance, one-use inclusion rule, semantic labels, and
  absence of an external figure dependency.
- `paper/JSTAT/README.md` documents the self-contained package and commands.
- `paper/README.md` identifies `paper/JSTAT` as the single active manuscript.
- The disposable-tree manuscript test now copies only `paper/JSTAT`.
- The repository inventory and PDF-QA reporting paths now point to
  `paper/JSTAT`.
- The current scientific-results README names the new paper-build command.
- `.gitignore` now applies LaTeX-intermediate exclusions to `paper/JSTAT`.

The publication renderer accepts repeatable targeted selections for canonical
Figures 1, 2, and 4. This pass invoked only those three selections; no other
figure generator was called.

## 10. Historical paths deliberately retained

Historical audit and implementation notes were not rewritten. Frozen
reproducibility manifests, the results index, and earlier PDF-QA records retain
their original `paper/jstat_v15` paths because those entries describe the
repository state in which they were recorded. The old prefix also remains in a
test allow-list solely to recognize historical paths. None is an active build
dependency.

## 11. Build and isolated-compilation results

Normal repository-root build:

```text
scripts/build-jstat-paper.sh
```

Result: passed, including BibTeX and all required pdfLaTeX passes. The final
log contained no undefined citation, undefined reference, duplicate-label, or
rerun warning. All 14 local figure files were loaded.

Self-contained isolated build:

```text
temporary_directory=$(mktemp -d /tmp/jstat-isolated-build.XXXXXX)
cp -a paper/JSTAT/. "$temporary_directory/"
(cd "$temporary_directory" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

Result: passed with only the copied contents of `paper/JSTAT`; the canonical
results tree was not available inside the temporary directory. The isolated
PDF had 28 pages, all citations and references resolved, all 14 figures loaded
from `figures/`, and no missing-file fallback text. The validated temporary
directory was then removed.

Focused workflow tests passed:

- repository inventory excludes LaTeX intermediates;
- the manuscript compiles in a disposable self-contained tree with embedded
  fonts;
- the reconstruction namespace guard recognizes the publication package and
  historical copy-named snapshots.

## 12. Scientific and artifact-preservation checks

- Displayed equation/align blocks: 15 before, 15 after.
- Publication figures: 14 before, 14 after; each local file is included once.
- Existing semantic equation and figure labels are unchanged.
- `results_macros.tex` is unchanged:
  `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679`.
- `references.bib` is unchanged:
  `dd444145a2d0248a8561f62dc992c04f494d935cc553bab22a1cea400b3a4ccd`.
- Duplicate BibTeX-key, normalized DOI, and normalized-title checks found no
  duplicate group.
- All 14 quantitative/source-layout CSV hashes are unchanged.
- All 11 unaffected canonical figure PDFs are unchanged.
- No simulation, analysis, configuration, trajectory, data, or unrelated test
  artifact changed.
- No numerical result, uncertainty interval, hypothesis disposition, claim,
  caveat, caption, Introduction text, Discussion text, Conclusion text, title,
  anonymity marker, or manuscript date changed.
- `main.tex` contains no absolute path, `../../results` path, old manuscript
  path, or other results-tree figure dependency.
- `git diff --check` passes.

## 13. PDF page count and visual inspection

- Final PDF: 28 A4 pages.
- Final PDF SHA-256 before commit:
  `384484fbd183c41f4e5bd7dd84ff4dbbdd962096524232571ab8b0b3181a7319`.
- All pages were rendered at 120 dpi and reviewed in contact sheets.
- The Figure 1 page, Figure 2 page, and Figure 9 page were additionally
  inspected at actual manuscript scale; standalone Figure 1 was inspected at
  200 and 300 dpi.
- No clipping, overlapping Figure 1 arrowhead, arrow/label intersection,
  missing glyph, caption truncation, equation overflow, duplicate figure,
  fallback placeholder, unexpected blank page, or broken appendix/reference
  flow was observed.

## 14. Files added, moved, modified, or removed

Added:

- `paper/JSTAT/README.md`
- `paper/JSTAT/figures/figure_manifest.csv`
- 14 publication-facing PDFs under `paper/JSTAT/figures/`
- `paper/README.md`
- `scripts/verify-jstat-paper-assets.sh`
- `notes/jstat_publication_directory_migration.md`

Moved with Git history:

- the five active manuscript files from `paper/jstat_v15/` to `paper/JSTAT/`;
- the active build script to `scripts/build-jstat-paper.sh`.

Modified:

- `paper/JSTAT/main.tex` and rebuilt `main.pdf`;
- the publication renderer and canonical figure generator;
- canonical Figures 1, 2, and 4, their catalog hashes, and generation metadata;
- active reporting paths, build/test paths, `.gitignore`, and the current
  results README.

Removed as an active path:

- `paper/jstat_v15/` after its tracked contents were moved.

No historical manuscript directory was removed or reorganized.

## 15. Remaining author/anonymity decision

The title page still says `Anonymous working manuscript` and retains the
existing manuscript date. Author names, affiliations, email addresses, ORCID,
review anonymity, and date remain an explicit submission-stage decision; no
metadata was inferred or changed in this pass.

## 16. Commit and push status

The exact 100%-similarity Git moves were recorded first in commit
`8b2ba3679e9a782f4c0e5a1019e53e0b1cf4e321` (`Move active JSTAT manuscript
directory`). This preserves path history for all five manuscript files and the
active build script before their publication-package updates.

This report is part of the subsequent implementation commit, so that commit's
content-derived hash and the normal-push outcome cannot be embedded
self-referentially here. They are reported in the author-facing final response.
The implementation commit message is:

```text
Organize self-contained JSTAT manuscript package
```

The normal push target is `origin/jstat-paper-structure-v16`; no force push,
rebase, reset, or merge is authorized or used.
