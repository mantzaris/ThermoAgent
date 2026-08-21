"""V14 quench trajectories using the immutable V13 LLM-agent microdynamics."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from thermoagent.statmech_llm_v12.core import LatentMapping, StructuredProvider
from thermoagent.statmech_llm_v12.graphs import DeliveryGraph
from thermoagent.statmech_llm_v13.simulation import (
    DISRUPTIONS,
    V13Agent,
    build_reciprocal_graph,
    make_v13_agents,
    partition_delivery_graph,
    phase_for_update,
    run_v13_trajectory,
)


def run_v14_trajectory(
    provider: StructuredProvider,
    graph: DeliveryGraph,
    panel_seed: int,
    sweeps: int,
    coupling_strength: float,
    sampling_temperature: float,
    disruption: str,
    periods_sweeps: Sequence[int],
    metadata: Optional[Mapping[str, object]] = None,
    mapping_override: Optional[LatentMapping] = None,
) -> List[Dict[str, object]]:
    """Run the unchanged Markovized V13 agent process on a new V14 tape.

    V14 changes only graph/environment seeds and the prospective analysis. The
    pinned prompt, response parser, typed action, private-state boundary,
    message serializer, random-sequential scheduler, and quench operators are
    inherited byte-for-byte from the immutable V13 source.
    """

    prefix = dict(metadata or {})
    prefix["study_version"] = "V14"
    return run_v13_trajectory(
        provider=provider,
        graph=graph,
        panel_seed=int(panel_seed),
        sweeps=int(sweeps),
        regime="markovized",
        coupling_strength=float(coupling_strength),
        sampling_temperature=float(sampling_temperature),
        initial_condition="disordered",
        disruption=str(disruption),
        periods_sweeps=list(periods_sweeps),
        metadata=prefix,
        mapping_override=mapping_override,
    )


__all__ = [
    "DISRUPTIONS",
    "V13Agent",
    "build_reciprocal_graph",
    "make_v13_agents",
    "partition_delivery_graph",
    "phase_for_update",
    "run_v14_trajectory",
]
