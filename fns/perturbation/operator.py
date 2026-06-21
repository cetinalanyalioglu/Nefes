"""The acoustic operator ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega)``.

``J_alg`` is the converged mean-flow Jacobian -- the zero-frequency acoustic
operator (theory.md s12.1) -- reused verbatim from the @njit complex-step
machinery (no new kernel).  ``M`` is the storage block (compliance/inertance),
``P`` the duct phase propagation, ``S`` the heat-release source.  In v1 only
``P`` has a producing element; ``M = 0`` and ``S`` is a no-op, but both are wired
into the assembly so a finite-volume or flame element drops in later as a
localized addition (see ``stamps.py``).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp

from ..assemble import jacobian
from .stamps import (
    DuctStamp,
    build_duct_stamps,
    build_storage,
    stamp_propagation,
    stamp_sources,
    stamp_boundaries,
)


@dataclass
class AcousticBlocks:
    """Frequency-independent blocks + frozen context, built once for a sweep."""

    J_alg: sp.csc_matrix  # complex, the converged Jacobian (zero-frequency operator)
    M: sp.csc_matrix  # complex, storage (zero unless volumes are present)
    duct_stamps: List[DuctStamp]  # per-duct P(omega) data
    prob: object  # the CompiledProblem (read-only, for source/boundary dispatch)
    x_bar: np.ndarray  # frozen mean state
    n: int
    u_floor: float = 1e-8


def build_acoustic_blocks(prob, x_bar, eps=None, eps_fb=1e-6, u_floor=1e-8):
    """Build the frozen blocks at the mean state ``x_bar`` (shape (n_solve, E)).

    ``J_alg`` is assembled with the regularizations turned down (the
    un-regularized variant of theory.md s12.6) at ``stab = 0``.  ``M`` is the
    storage block (zero in v1).  The duct phase data is precomputed here and
    restamped cheaply per frequency.
    """
    if eps is None:
        eps = 1e-4 * prob.var_scale[0]
    x_bar = np.ascontiguousarray(x_bar)
    J = jacobian(prob, x_bar, eps, eps_fb, 0.0).astype(np.complex128)
    n = J.shape[0]
    M = build_storage(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])  # cp / R
    duct_stamps = build_duct_stamps(prob, x_bar, K, u_floor)
    return AcousticBlocks(
        J_alg=J.tocsc(),
        M=M,
        duct_stamps=duct_stamps,
        prob=prob,
        x_bar=x_bar,
        n=n,
        u_floor=u_floor,
    )


def assemble_acoustic(omega, blocks: AcousticBlocks):
    """Stamp the full ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega)``.

    The cached ``J_alg`` is never mutated: a fresh LIL copy receives the i*omega*M
    scaling and the omega-dependent stamps.  At ``omega = 0`` with no ducts this
    returns exactly ``J_alg`` (the founding consistency); with ducts the phase
    rows reduce to wave-amplitude continuity -- physically equivalent to the
    steady duct rows.
    """
    A = (blocks.J_alg + 1j * omega * blocks.M).tolil()
    stamp_propagation(A, omega, blocks.duct_stamps, blocks.u_floor)
    stamp_sources(A, omega, blocks.prob, blocks.x_bar)
    stamp_boundaries(A, omega, blocks.prob, blocks.x_bar)
    return A.tocsc()
