# JSTAT V14 working manuscript

This directory contains the V14 review manuscript. It uses standard 12-point
LaTeX and common packages. The official JSTAT author guidance was rechecked on
20 August 2026. It prefers LaTeX using IOP instructions but accepts other
standard formats; requires the abstract on the first page, numerical references,
and a Data/Software/Code Availability statement; and asks authors using BibTeX
to include the generated `.bbl` at submission.

Primary guidance:

- https://jstat.sissa.it/jstat/help/helpLoader.jsp?pgType=author
- https://jstat.sissa.it/jstat/help/helpLoader.jsp?pgType=submissionFAQ
- https://publishingsupport.iopscience.iop.org/journals/journal-of-statistical-mechanics-theory-and-experiment/

After regenerating the frozen candidate figures, run
`python3 paper/jstat_v14/refine_figures.py`, then build with
`scripts/build-statmech-v14-paper.sh`. The refinement reads only saved figure
source CSVs and changes annotation placement, margins, and panel layout; it is
intentionally outside the formal execution checksum. Numerical macros are
generated from compact V14 aggregate tables. Bibliographic metadata is checked
against publisher or arXiv primary records; the related-work audit includes
arXiv:2608.16578, 2608.02827, 2605.10528, 2601.05606, and 2510.22422.

The manuscript uses ``effective reference energy'' and ``coarse-grained
pathwise irreversibility'' deliberately. It does not claim literal energy,
physical temperature, exact thermodynamic entropy production, a thermodynamic-
limit transition, universal behavior across LLMs, or application performance.
