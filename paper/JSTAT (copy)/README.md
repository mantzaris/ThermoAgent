# Active JSTAT manuscript

This directory contains the active JSTAT manuscript. `main.tex` is the entry
point and `main.pdf` is the current compiled paper. `references.bib` contains
the bibliography, while `results_macros.tex` contains the frozen numerical
reporting macros.

The `figures/` directory contains exactly the 14 publication figures used by
`main.tex`. `figures/figure_manifest.csv` maps their publication-facing names
to the canonical repository assets.

Build from the repository root with:

```bash
scripts/build-jstat-paper.sh
```

Optionally verify local figure provenance and completeness with:

```bash
scripts/verify-jstat-paper-assets.sh
```

Compilation is self-contained within `paper/JSTAT`; canonical analysis outputs
and scientific source-data tables remain elsewhere in the repository.
