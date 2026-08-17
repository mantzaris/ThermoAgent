# JSTAT V10 working draft

This portable 12-point LaTeX draft follows the current JSTAT/IOP guidance that
initial submissions may use common LaTeX layouts. It is not an official IOP
class reproduction. Build it after generating the V10 aggregate results and
figures:

```bash
MPLCONFIGDIR=/tmp/thermoagent-v10-mpl \
  .venv/bin/python paper/jstat_v10/render_publication_figures.py
THERMO_V10_ARTIFACT_ROOT=/tmp/ThermoAgent-v10-artifacts \
  ./scripts/run-statmech-v10-paper.sh
```

The first command is a presentation-only rebuild from the frozen figure source
CSVs. It corrects layout and uncertainty display without changing numerical
results or any file included in the frozen scientific-source checksum.

The LLM sections report a retained development-pilot no-go. The existing Pod was
reachable through the RunPod proxy, but Qwen did not respond to delivered peer
evidence and the formal transition-kernel/trajectory study remained locked.
The draft must not be submitted as evidence of LLM entropy production in its
current state.
