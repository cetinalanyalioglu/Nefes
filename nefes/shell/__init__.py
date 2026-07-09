"""User-facing object shell: Network, Solution, and the CompiledProblem builder."""

from .network import Network, Solution
from .build import build_problem, build_problem_from_connectivity, validate_network, finalize_thermo
from .study import parameter_study, StudyResult
from .params import ParameterInfo, ParameterInventory

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
