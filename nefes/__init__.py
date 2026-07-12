"""Nefes -- Network solver for reacting compressible flows and thermoacoustics.

Nefes (Turkish for "breath") models a fluid system as a directed graph and solves
for the steady mean flow and the linear perturbation behavior around it (two
acoustic characteristics plus the entropy wave), without resolving the full 3-D
field.  Reacting flows and their thermoacoustics are among the target applications.

State lives on edges ``(mdot, p, h_t)``; equations live on nodes (elements)
plus one transport equation per edge, giving a square ``3E`` system that is
invariant to the choice of edge directions.  All residual math is smooth and
complex-step-safe; the Jacobian is obtained by complex-step differentiation and
doubles as the zero-frequency perturbation operator.

The common workflow is reachable from this top-level namespace: build a
:class:`Network` from element factories in :data:`cat`, choose a gas model with
:func:`perfect_gas` or :func:`equilibrium`, :meth:`Network.solve` it, sweep with
:func:`parameter_study`, set acoustic terminations with :class:`PerturbationBC`, and
load or save cases with :func:`load_case` / :func:`save_case`.  The acoustic analyses
(``eigenmodes``, ``forced_response``, ...) live in :mod:`nefes.perturbation`.
"""

__version__ = "0.1.0"

from .config import config
from .elements import catalog as cat
from .io import load_case, load_solution, save_case, save_solution
from .perturbation import PerturbationBC
from .shell import Network, Solution, StudyResult, parameter_study
from .thermo.configure import equilibrium, perfect_gas

__all__ = [
    "Network",
    "Solution",
    "parameter_study",
    "StudyResult",
    "load_case",
    "load_solution",
    "save_case",
    "save_solution",
    "cat",
    "perfect_gas",
    "equilibrium",
    "PerturbationBC",
    "config",
]
