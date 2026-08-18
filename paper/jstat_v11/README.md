# V11 JSTAT working manuscript

This directory is new for V11; `paper/jstat_v10/` remains frozen. The draft
uses standard, conservative LaTeX packages and 12-point reviewer text. Current
IOP guidance says journal submissions need not mimic the final typeset design,
and the JSTAT submission site accepts standard LaTeX while preferring IOP
instructions. A data/software availability statement and numerical references
are included.

The title and empirical sections are generated only after the prospective
qualification disposition is known. Raw prompts, generations, and LaTeX QA
renders remain outside Git.

Build after repository-facing aggregates and figures exist:

```bash
./scripts/run-statmech-v11-paper.sh
```
