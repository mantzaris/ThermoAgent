"""Statistical mechanics of nonreciprocal decentralized LLM agents.

The package separates the exact stochastic-agent reference from empirical LLM
realizations.  Nothing in the analytical simulator calls an LLM, and nothing
in the LLM boundary is permitted to inspect evaluator-global state.
"""

from .theory import (
    PerturbativeEntropyProduction,
    directed_family,
    entropy_production_by_layer,
    perturbative_entropy_production,
)

__all__ = [
    "PerturbativeEntropyProduction",
    "directed_family",
    "entropy_production_by_layer",
    "perturbative_entropy_production",
]
