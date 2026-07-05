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
"""

__version__ = "0.1.0"

from .shell import Network, Solution

__all__ = ["Network", "Solution"]
