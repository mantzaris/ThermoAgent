"""V13 collective statistical mechanics for independent LLM-agent networks.

V13 imports the frozen V12 typed-agent and binary-wire primitives but owns its
protocol, disruptions, analysis, and artifact namespace. Importing this module
never loads an LLM or starts an experiment.
"""

from .observables import (
    instantaneous_state,
    macrostate_code,
    rolling_state_vectors,
    total_correlation,
)
from .simulation import DISRUPTIONS, run_replication_trajectory

__all__ = [
    "DISRUPTIONS",
    "instantaneous_state",
    "macrostate_code",
    "rolling_state_vectors",
    "run_replication_trajectory",
    "total_correlation",
]
