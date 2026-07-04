"""User-facing object shell: Network, Solution, and the CompiledProblem builder."""

from .network import Network, Solution
from .build import build_problem, build_problem_from_connectivity, validate_network, finalize_thermo

__all__ = [
    "Network",
    "Solution",
    "build_problem",
    "build_problem_from_connectivity",
    "validate_network",
    "finalize_thermo",
]
