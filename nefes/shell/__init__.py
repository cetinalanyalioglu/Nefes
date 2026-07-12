"""User-facing object shell: Network, Solution, and the CompiledProblem builder."""

from .build import build_problem, build_problem_from_connectivity, finalize_thermo, validate_network
from .network import Network, Solution
from .params import ParameterInfo, ParameterInventory
from .study import StudyResult, parameter_study

__all__ = [
    "Network",
    "Solution",
    "build_problem",
    "build_problem_from_connectivity",
    "validate_network",
    "finalize_thermo",
    "parameter_study",
    "StudyResult",
    "ParameterInfo",
    "ParameterInventory",
]
