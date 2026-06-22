"""Analytic acoustic stamps written onto ``A(omega)`` after ``J_alg + i*omega*M``.

Three faces (implementation-plan.md s8.2-8.3):

* ``stamp_propagation`` -- the **duct** phase relations ``P(omega)`` (theory.md
  s12.3), the only omega-dependent block in v1.  For each duct it replaces three
  rows (its two node rows + the head edge's transport row) with the
  characteristic phase relations, built diagonally in the wave amplitudes
  ``w = (f, g, h)`` and mapped to solution-variable rows through ``L_e``.
* ``stamp_sources`` -- the heat-release ``S(omega)`` face (reserved; no flame
  element in v1, so a no-op).
* ``stamp_boundaries`` -- terminal reflection coefficients (reserved; the v1
  scattering driver imposes incoming waves at terminals instead, so a no-op).

``build_storage`` is the storage ``M`` hook: zero in v1 (no finite-volume
element), but the home for the ``d/dt integral_V U`` block.

These run **above the @njit line** -- plain Python / SciPy.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .characteristics import dx_to_char
from .matrices import partition
from .verify import duct_nodes, verify_acoustic
from .terminals import find_terminals
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA
from ..elements.ids import ACOUSTIC_VOLUME, ACOUSTIC_FLAME


@dataclass
class DuctStamp:
    """Frozen per-duct data for the ``P(omega)`` stamp (built once per sweep)."""

    e0: int  # tail-station edge (port 0, points into the duct)
    e1: int  # head-station edge (port 1, points out of the duct)
    L0: np.ndarray  # 3x3 dx_to_char at e0's mean state
    L1: np.ndarray  # 3x3 dx_to_char at e1's mean state
    tau_p: float  # L / (u + c)
    tau_m: float  # L / (c - u)
    tau_0: float  # L / u   (inf when quiescent)
    u: float  # mean axial velocity (>= 0 along the duct axis)
    row_f: int  # duct node row holding the downstream (f) phase relation
    row_g: int  # duct node row holding the upstream (g) phase relation
    row_h: int  # head edge's transport row, repurposed for the entropy (h) phase
    cols0: tuple  # the 3 columns of e0
    cols1: tuple  # the 3 columns of e1


def build_duct_stamps(prob, x_bar, K, u_floor=1e-8):
    """Build the per-duct ``P(omega)`` data at the frozen mean state ``x_bar``.

    Runs ``verify_acoustic`` first (pinned orientation, subsonic, length > 0).
    """
    verify_acoustic(prob, x_bar)
    est = states_table(prob, x_bar)
    ns = int(prob.n_solve)
    stamps = []
    for n in duct_nodes(prob):
        base = int(prob.row_ptr[n])
        e0 = int(prob.col_edge[base])
        e1 = int(prob.col_edge[base + 1])
        length = float(prob.npar_f[int(prob.npar_fptr[n])])

        # the duct is constant-area and lossless: e0 and e1 share the mean state.
        rho = float(est[ES_RHO, e0])
        c = float(est[ES_C, e0])
        u = float(est[ES_U, e0])
        p = float(est[ES_P, e0])
        L0 = dx_to_char(rho, c, u, p, float(est[ES_AREA, e0]), K)
        L1 = dx_to_char(
            float(est[ES_RHO, e1]),
            float(est[ES_C, e1]),
            float(est[ES_U, e1]),
            float(est[ES_P, e1]),
            float(est[ES_AREA, e1]),
            K,
        )

        tau_p = length / (u + c)
        tau_m = length / (c - u)
        tau_0 = length / u if abs(u) > u_floor else np.inf

        r0 = int(prob.node_row_ptr[n])
        stamps.append(
            DuctStamp(
                e0=e0,
                e1=e1,
                L0=L0,
                L1=L1,
                tau_p=tau_p,
                tau_m=tau_m,
                tau_0=tau_0,
                u=u,
                row_f=r0,
                row_g=r0 + 1,
                # entropy phase relation lives on the *downstream* edge's transport row
                # (head for forward/quiescent flow, tail under backflow), leaving the
                # genuine-inflow edge's transport row free for the boundary entropy seat.
                row_h=int(prob.transport_row0) + (e0 if u < -u_floor else e1),
                cols0=tuple(ns * e0 + v for v in range(3)),
                cols1=tuple(ns * e1 + v for v in range(3)),
            )
        )
    return stamps


def _set_row(A, row, cols0, coeff0, cols1, coeff1):
    """Overwrite a full LIL row with two length-3 coefficient blocks."""
    A.rows[row] = []
    A.data[row] = []
    for c, v in zip(cols0, coeff0):
        A[row, c] = v
    for c, v in zip(cols1, coeff1):
        A[row, c] = v


def stamp_propagation(A, omega, duct_stamps, u_floor=1e-8):
    """Apply the duct phase relations ``P(omega)`` to LIL matrix ``A`` in place.

    For each duct (tail station ``0`` -> head station ``1``):
        f1 = Pp*f0,   g0 = Pm*g1,   h1 = P0*h0,
    with ``Pp = exp(-i w tau_+)``, ``Pm = exp(-i w tau_-)``, ``P0 = exp(-i w
    tau_0)``.  At a quiescent duct (u ~ 0) the entropy wave is stationary and
    decoupled, so ``P0 = 1``.
    """
    for st in duct_stamps:
        Pp = np.exp(-1j * omega * st.tau_p)
        Pm = np.exp(-1j * omega * st.tau_m)
        P0 = np.exp(-1j * omega * st.tau_0) if abs(st.u) > u_floor else 1.0 + 0.0j

        # Row f:  f1 - Pp*f0 = 0
        _set_row(A, st.row_f, st.cols0, -Pp * st.L0[0, :], st.cols1, st.L1[0, :])
        # Row g:  g0 - Pm*g1 = 0
        _set_row(A, st.row_g, st.cols0, st.L0[1, :], st.cols1, -Pm * st.L1[1, :])
        # Row h:  h1 - P0*h0 = 0
        _set_row(A, st.row_h, st.cols0, -P0 * st.L0[2, :], st.cols1, st.L1[2, :])


def stamp_sources(A, omega, prob, x_bar):
    """Heat-release source face ``S(omega)`` (reserved).

    No element carries ``ACOUSTIC_FLAME`` in v1, so this is a no-op; the call
    site and signature are in place for the flame stamp.
    """
    flame = [n for n in range(prob.n_nodes) if int(prob.node_acoustic_id[n]) == ACOUSTIC_FLAME]
    if flame:
        raise NotImplementedError("flame source stamp S(omega) is a reserved v1 provision")


def _terminal_closure(prob, est, K, t, bc, omega):
    """Per to-specify wave at terminal ``t``: ``(row, coeff_block, rhs)``.

    Builds the matrix closure ``w[specify] = A(omega) @ w[arriving] + b`` via
    :meth:`PerturbationBC.closure` over the mean-state wave partition
    (:func:`matrices.partition`), and maps each to-specify wave to its matrix row --
    the acoustic wave on the boundary node row, the (inflow) entropy wave on the edge's
    transport row.  The length-3 coefficient block is
    ``L_e[specify] - sum_j A[.,j] L_e[arriving_j]`` and ``rhs`` its forcing.
    """
    e = t.edge
    rho, c, u = float(est[ES_RHO, e]), float(est[ES_C, e]), float(est[ES_U, e])
    p, area = float(est[ES_P, e]), float(est[ES_AREA, e])
    m_out = (u / c) if not t.at_tail else (-u / c)  # outward-normal mean Mach
    specify, arriving = partition(u, c, "a" if t.at_tail else "b")
    freq = omega / (2.0 * np.pi)  # BC carriers (tables/callables) are in Hz; operator stays in omega
    Amat, bvec = bc.closure(freq, rho, c, u, m_out, K, specify, arriving)
    L_e = dx_to_char(rho, c, u, p, area, K)
    out = []
    for i, ch in enumerate(specify):
        row = t.row if ch in (0, 1) else int(prob.transport_row0) + e  # acoustic -> node, entropy -> transport
        coeff = L_e[ch, :].astype(np.complex128)
        for j, cha in enumerate(arriving):
            coeff = coeff - Amat[i, j] * L_e[cha, :]
        out.append((row, coeff, complex(bvec[i])))
    return out


def stamp_boundaries(A, omega, prob, x_bar):
    """Terminal closure face ``A(omega)`` (theory.md s12.4) onto LIL ``A``.

    Each single-port terminal carrying an explicit ``PerturbationBC`` (anything but
    ``inherit``) has the rows of its to-specify waves overwritten with the matrix
    closure ``w[specify] = A(omega) @ w[arriving] + b`` (``b`` built by
    :func:`boundary_forcing`).  The acoustic to-specify wave lands on the boundary node
    row; at an inflow (tail) terminal the incoming entropy wave is also seated, on that
    edge's transport row -- always a duct *tail* edge, so it never collides with the
    duct stamp's head-edge entropy phase (theory.md s6.2).  Terminals left at
    ``inherit`` keep their linearized mean boundary row from ``J_alg``.
    """
    node_bc = prob.node_bc
    if not node_bc:
        return
    est = states_table(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])
    ns = int(prob.n_solve)
    for t in find_terminals(prob):
        bc = node_bc[t.node] if t.node < len(node_bc) else None
        if bc is None or not getattr(bc, "stamps_terminal", False):
            continue
        cols = tuple(ns * t.edge + v for v in range(3))
        for row, coeff, _rhs in _terminal_closure(prob, est, K, t, bc, omega):
            _set_row(A, row, cols, coeff, (), ())


def boundary_forcing(prob, x_bar, omega):
    """Right-hand side ``b(omega)`` for the explicitly-closed terminals.

    The forcing of each to-specify wave (acoustic excitation on the node row, incoming
    entropy on the inflow-side transport row); zero everywhere else.  Mirrors the rows
    :func:`stamp_boundaries` overwrites, via the same :func:`_terminal_closure`.
    """
    b = np.zeros(prob.n_col, dtype=np.complex128)
    node_bc = prob.node_bc
    if not node_bc:
        return b
    est = states_table(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])
    for t in find_terminals(prob):
        bc = node_bc[t.node] if t.node < len(node_bc) else None
        if bc is None or not getattr(bc, "stamps_terminal", False):
            continue
        for row, _coeff, rhs in _terminal_closure(prob, est, K, t, bc, omega):
            b[row] = rhs
    return b


def build_storage(prob, x_bar):
    """Storage block ``M`` (the ``d/dt integral_V U`` term dropped at steady state).

    Zero in v1 (no finite-volume element); a volumetric element would populate
    its conservation rows here via a complex-step of a transient-flux operator.
    """
    vol = [n for n in range(prob.n_nodes) if int(prob.node_acoustic_id[n]) == ACOUSTIC_VOLUME]
    if vol:
        raise NotImplementedError("finite-volume storage M is a reserved v1 provision")
    return sp.csc_matrix((prob.n_eq, prob.n_col), dtype=np.complex128)
