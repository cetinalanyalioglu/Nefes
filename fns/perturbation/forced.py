"""Forced perturbation response under physical boundary conditions (theory.md s12.7).

Where ``response.py`` *measures* a network -- driving every terminal with independent
unit waves to read out a transfer/scattering matrix, regardless of how the boundaries
are actually closed -- this module *solves* the network as it is physically
terminated.  Each single-port element carries a :class:`PerturbationBC`; the operator
``A(omega)`` is assembled with the terminal reflection face stamped
(``with_boundaries=True``) and the excitation right-hand side ``b(omega)`` built from
the terminals that force.  One sparse solve per frequency gives the nodal
perturbation field.

A purely-reflective, undamped, unforced network is singular at its resonances -- that
is the (deferred) stability eigenvalue problem ``det A(omega) = 0``.  A forced
response therefore needs at least one excitation terminal (or some loss); off
resonance it is well posed.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .stamps import boundary_forcing
from .characteristics import edge_transforms, basis_block_from_state
from ..solver.control import states_table


def boundary_response(prob, x_bar, freqs, *, eps=None, eps_fb=1e-6, u_floor=1e-8, isentropic=False):
    """Solve the perturbation field under each terminal's declared boundary condition.

    Parameters
    ----------
    prob : CompiledProblem
        Compiled flow network whose single-port elements carry ``PerturbationBC``s
        (``prob.node_bc``).  Terminals left at ``inherit`` keep their linearized mean
        boundary row.
    x_bar : ndarray
        Converged mean-flow state, shape ``(n_solve, E)``.
    freqs : array_like
        Frequencies (Hz) to solve at.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to :func:`build_acoustic_blocks`.
    isentropic : bool, optional
        Force isentropic perturbations (``rho' = p'/c^2``): the entropy wave is pinned to
        zero on every edge, leaving the two acoustic waves (default False).  Standard
        acoustic analysis; uses the same operator and solve path.

    Returns
    -------
    ForcedResponse
        The nodal perturbation field at every frequency.
    """
    freqs = np.asarray(freqs, dtype=float)
    omegas = 2.0 * np.pi * freqs  # operator assembly works in angular frequency (rad/s)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor, isentropic=isentropic)
    K = float(prob.tf[0]) / float(prob.tf[1])
    est = states_table(prob, x_bar)
    cals = blocks.cals
    _, L = edge_transforms(est, K, cals)

    X = np.zeros((omegas.size, int(prob.n_col)), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        A = assemble_acoustic(omega, blocks, with_boundaries=True)
        b = boundary_forcing(prob, x_bar, omega, cals)
        X[i] = spla.spsolve(A.tocsc(), b)
    return ForcedResponse(freqs=freqs, X=X, L=L, est=est, K=K, n_solve=int(prob.n_solve), cals=cals)


@dataclass
class ForcedResponse:
    """Nodal perturbation field of a physically-terminated network over a sweep."""

    freqs: np.ndarray  # (n_freq,) in Hz
    X: np.ndarray  # (n_freq, n_col) nodal perturbation vectors
    L: List[np.ndarray]  # per-edge dx_to_char (3x3) at the mean state
    est: np.ndarray  # frozen mean edge-state table
    K: float  # cp / R
    n_solve: int
    cals: Optional[list] = None  # per-edge caloric rows (reacting "network" flavor)

    @property
    def n_char(self) -> int:
        """Characteristic count per edge (3 for inert flow)."""
        return self.L[0].shape[0]

    def waves(self, edge):
        """Characteristic amplitudes ``(f, g, h)`` at ``edge``; shape ``(n_omega, n_char)``."""
        ns, nc = self.n_solve, self.n_char
        Xe = self.X[:, ns * edge : ns * edge + nc]  # (n_omega, n_char)
        return np.einsum("ij,oj->oi", self.L[edge], Xe)  # (n_omega, n_char)

    def field(self, edge, basis="network"):
        """Perturbation at ``edge`` in a variable flavor; shape ``(n_omega, n_char)``.

        ``basis`` is any of ``characteristics.BASIS_LABELS`` (default ``"network"`` --
        the solver's own ``(mdot', p', h_t')``).
        """
        w = self.waves(edge)  # (n_omega, n_char)
        cal = None if self.cals is None else self.cals[edge]
        B = basis_block_from_state(basis, self.est[:, edge], self.K, cal)
        return np.einsum("ij,oj->oi", B, w)

    def reflection_at(self, edge):
        """Local acoustic reflection ``g/f`` at ``edge``; shape ``(n_omega,)``.

        The ratio of the upstream- to downstream-running acoustic amplitude -- the
        reflection seen looking downstream at this station.
        """
        w = self.waves(edge)
        return w[:, 1] / w[:, 0]
