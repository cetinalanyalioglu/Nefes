"""The acoustic operator ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega)``.

``J_alg`` is the converged mean-flow Jacobian -- the zero-frequency acoustic
operator (theory.md s12.1) -- reused verbatim from the @njit complex-step
machinery (no new kernel).  ``M`` is the storage block (compliance/inertance),
``P`` the duct phase propagation, ``S`` the heat-release source.  In v1 only
``P`` has a producing element; ``M = 0`` and ``S`` is a no-op, but both are wired
into the assembly so a finite-volume or flame element drops in later as a
localized addition (see ``stamps.py``).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy.sparse as sp

from ..assemble import jacobian
from ..solver.control import states_table
from .terminals import find_terminals
from .stamps import (
    DuctStamp,
    build_duct_stamps,
    build_storage,
    stamp_propagation,
    stamp_sources,
    stamp_boundaries,
    stamp_isentropic,
    _terminal_closure,
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
    # force isentropic perturbations (rho' = p'/c^2): the entropy wave is pinned to zero
    # on every edge (stamps.stamp_isentropic).  Reduces the 3-wave system to the two
    # acoustic waves -- standard acoustic analysis -- with no change in operator size.
    isentropic: bool = False
    # cached fixed-pattern assemblers keyed by with_boundaries (lazy; see _build_plan)
    _plans: dict = field(default_factory=dict, repr=False, compare=False)


def build_acoustic_blocks(prob, x_bar, eps=None, eps_fb=1e-6, u_floor=1e-8, isentropic=False):
    """Build the frozen blocks at the mean state ``x_bar`` (shape (n_solve, E)).

    ``J_alg`` is assembled with the regularizations turned down (the
    un-regularized variant of theory.md s12.6) at ``stab = 0``.  ``M`` is the
    storage block (zero in v1).  The duct phase data is precomputed here and
    restamped cheaply per frequency.

    ``isentropic`` (default False) pins the entropy characteristic to zero on every
    edge (``rho' = p'/c^2``), reducing the operator to the two acoustic waves -- the
    standard acoustic assumption -- without changing its size; see
    :func:`stamps.stamp_isentropic`.
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
        isentropic=bool(isentropic),
    )


def _assemble_reference(omega, blocks: AcousticBlocks, with_boundaries=True):
    """Reference assembly of ``A(omega)`` via LIL stamping (the trusted, slow path).

    The cached ``J_alg`` is never mutated: a fresh LIL copy receives the i*omega*M
    scaling and the omega-dependent stamps.  At ``omega = 0`` with no ducts and only
    inherited boundaries this returns exactly ``J_alg`` (the founding consistency);
    with ducts the phase rows reduce to wave-amplitude continuity -- physically
    equivalent to the steady duct rows.

    ``with_boundaries`` controls the terminal reflection face ``R(omega)``
    (``stamps.stamp_boundaries``).  The measurement driver (``response.py``) sets it
    ``False`` because it closes every terminal itself with independent prescribed
    incoming waves; the physical forced/stability drivers leave it ``True`` so each
    terminal carries its declared ``PerturbationBC``.

    This builds the matrix from scratch every call; :func:`assemble_acoustic` is the
    fast path that reuses it once to capture the (omega-independent) sparsity pattern.
    """
    A = (blocks.J_alg + 1j * omega * blocks.M).tolil()
    stamp_propagation(A, omega, blocks.duct_stamps, blocks.u_floor, skip_entropy=blocks.isentropic)
    stamp_sources(A, omega, blocks.prob, blocks.x_bar)
    if with_boundaries:
        stamp_boundaries(A, omega, blocks.prob, blocks.x_bar)
    if blocks.isentropic:
        # pin the entropy wave to zero on every edge (rho' = p'/c^2); overrides the
        # entropy rows the duct/boundary stamps wrote.  omega-independent.
        est = states_table(blocks.prob, blocks.x_bar)
        K = float(blocks.prob.tf[0]) / float(blocks.prob.tf[1])
        stamp_isentropic(A, blocks.prob, est, K)
    return A.tocsc()


# Generic non-zero frequency used to capture the (omega-independent) sparsity pattern.
_PLAN_OMEGA = 1.0


class _AssemblyPlan:
    """Fixed sparsity pattern of ``A(omega)`` plus a fast per-omega value fill.

    The *structure* of ``A(omega)`` does not depend on ``omega`` -- only the duct
    phases ``e^{-i*omega*tau}`` and any frequency-dependent terminal closure change
    *values*, never which entries exist.  So the pattern (CSC ``indptr``/``indices``)
    and the omega-independent entries (``base``) are captured once; each subsequent
    ``A(omega)`` is built by scattering the few omega-dependent values into a copy of
    ``base``, with no LIL construction and no re-sort.

    This is the assembly hot path under the contour sweep: it replaces the
    ``O(edges)`` Python-level sparse-row overwrites and the ``tolil``/``tocsc`` round
    trip of :func:`_assemble_reference` with one array copy and a handful of
    vectorized operations per node.  The duct fill is a single complex ``exp`` and a
    fancy-index assignment; the (few) boundary terminals reuse the exact closure of
    :func:`stamps._terminal_closure`.

    Attributes
    ----------
    indptr, indices : ndarray
        Canonical CSC structure shared by every ``A(omega)``.
    shape : tuple
        Operator shape.
    base : ndarray
        Complex ``data`` array holding the omega-independent entries (the non-stamped
        ``J_alg`` rows and the constant parts of the duct rows); the omega-dependent
        slots are zeroed and overwritten per call.
    phase_slots, phase_tau, phase_coeff : ndarray
        For each omega-dependent duct entry: its index in ``data``, its transit time
        ``tau``, and its constant coefficient.  The entry's value is
        ``phase_coeff * exp(-i*omega*tau)``.
    bnd : list
        ``(terminal, bc, {row: slot_indices})`` for the explicitly-closed terminals.
    """

    __slots__ = (
        "indptr",
        "indices",
        "shape",
        "base",
        "phase_slots",
        "phase_tau",
        "phase_coeff",
        "prob",
        "est",
        "K",
        "bnd",
    )

    def __init__(self, indptr, indices, shape, base, phase_slots, phase_tau, phase_coeff, prob, est, K, bnd):
        self.indptr = indptr
        self.indices = indices
        self.shape = shape
        self.base = base
        self.phase_slots = phase_slots
        self.phase_tau = phase_tau
        self.phase_coeff = phase_coeff
        self.prob = prob
        self.est = est
        self.K = K
        self.bnd = bnd

    def assemble(self, omega):
        """Build ``A(omega)`` as a CSC matrix by the fast fill."""
        data = self.base.copy()
        if self.phase_slots.size:
            # the only bulk omega-dependence: each duct phase entry = coeff * e^{-i w tau}
            data[self.phase_slots] = self.phase_coeff * np.exp(-1j * omega * self.phase_tau)
        for t, bc, rowslots in self.bnd:
            for row, coeff, _rhs in _terminal_closure(self.prob, self.est, self.K, t, bc, omega):
                slots = rowslots.get(row)
                if slots is not None:  # entropy rows are dropped under isentropic mode
                    data[slots] = coeff
        A = sp.csc_matrix((data, self.indices, self.indptr), shape=self.shape)
        A.has_sorted_indices = True  # indices/indptr come canonical and are reused unchanged
        return A


def _build_plan(blocks: AcousticBlocks, with_boundaries):
    """Capture the fixed sparsity pattern and per-omega fill data for :class:`_AssemblyPlan`."""
    prob = blocks.prob
    ns = int(prob.n_solve)

    tr0 = int(prob.transport_row0)

    # The omega-dependent duct entries, as (row, col, constant coeff, tau).  Structural
    # zeros (exactly-zero L entries) never enter the pattern, so they are skipped.  The
    # entropy (h) phase is omega-dependent only on a *flowing* duct; on a quiescent one it
    # is the stationary P0 = 1 (a constant already folded into base).  Under isentropic
    # mode the entropy rows are pinned to the constant h = 0 (in base), so the entropy phase
    # is dropped entirely.
    duct_entries = []
    for st in blocks.duct_stamps:
        flowing = abs(st.u) > blocks.u_floor
        for j in range(3):
            if st.L0[0, j] != 0.0:
                duct_entries.append((st.row_f, st.cols0[j], -st.L0[0, j], st.tau_p))
            if st.L1[1, j] != 0.0:
                duct_entries.append((st.row_g, st.cols1[j], -st.L1[1, j], st.tau_m))
            if flowing and not blocks.isentropic and st.L0[2, j] != 0.0:
                duct_entries.append((st.row_h, st.cols0[j], -st.L0[2, j], st.tau_0))

    # Boundary terminals.  A frequency-dependent closure entry can vanish at the
    # pattern-probe frequency yet be non-zero elsewhere, so every boundary slot is forced
    # into the pattern (added with value 0) rather than read off the probe assembly.
    bnd_meta, forced_rc = [], []
    est = K = None
    if with_boundaries and prob.node_bc:
        est = states_table(prob, blocks.x_bar)
        K = float(prob.tf[0]) / float(prob.tf[1])
        for t in find_terminals(prob):
            bc = prob.node_bc[t.node] if t.node < len(prob.node_bc) else None
            if bc is None or not getattr(bc, "stamps_terminal", False):
                continue
            cols = tuple(ns * t.edge + v for v in range(3))
            rows = [row for row, _c, _r in _terminal_closure(prob, est, K, t, bc, _PLAN_OMEGA)]
            if blocks.isentropic:
                # entropy (transport) rows are pinned to h = 0 in base; the boundary fill
                # must not overwrite them, so keep only the acoustic (node) closure rows.
                rows = [row for row in rows if row < tr0]
            bnd_meta.append((t, bc, [(row, cols) for row in rows]))
            forced_rc.extend((row, c) for row in rows for c in cols)

    # Canonical pattern: the reference assembly (correct base values) unioned with the
    # forced boundary slots; sum_duplicates merges the forced zeros into existing entries.
    ref = _assemble_reference(_PLAN_OMEGA, blocks, with_boundaries).tocoo()
    if forced_rc:
        rows_all = np.concatenate([ref.row, np.array([r for r, _ in forced_rc], dtype=ref.row.dtype)])
        cols_all = np.concatenate([ref.col, np.array([c for _, c in forced_rc], dtype=ref.col.dtype)])
        vals_all = np.concatenate([ref.data.astype(np.complex128), np.zeros(len(forced_rc), np.complex128)])
    else:
        rows_all, cols_all, vals_all = ref.row, ref.col, ref.data.astype(np.complex128)
    T = sp.csc_matrix((vals_all, (rows_all, cols_all)), shape=ref.shape)
    T.sum_duplicates()
    T.sort_indices()
    indptr, indices = T.indptr.copy(), T.indices.copy()
    base = T.data.astype(np.complex128).copy()

    def slot(row, col):
        # CSC: column-major, so a column's stored rows are indices[indptr[col]:indptr[col+1]].
        lo, hi = int(indptr[col]), int(indptr[col + 1])
        pos = lo + int(np.searchsorted(indices[lo:hi], row))
        if pos >= hi or indices[pos] != row:
            raise RuntimeError(f"assembly-plan slot ({row}, {col}) missing from the captured pattern")
        return pos

    phase_slots = np.empty(len(duct_entries), dtype=np.intp)
    phase_tau = np.empty(len(duct_entries), dtype=np.float64)
    phase_coeff = np.empty(len(duct_entries), dtype=np.complex128)
    for i, (row, col, coeff, tau) in enumerate(duct_entries):
        phase_slots[i], phase_tau[i], phase_coeff[i] = slot(row, col), tau, coeff
    base[phase_slots] = 0.0  # zero the omega-dependent duct slots (overwritten per call)

    bnd = []
    for t, bc, rowcols in bnd_meta:
        rowslots = {}
        for row, cols in rowcols:
            s = np.array([slot(row, c) for c in cols], dtype=np.intp)
            rowslots[row] = s
            base[s] = 0.0  # zero the boundary slots (overwritten per call)
        bnd.append((t, bc, rowslots))

    return _AssemblyPlan(indptr, indices, ref.shape, base, phase_slots, phase_tau, phase_coeff, prob, est, K, bnd)


def assemble_acoustic(omega, blocks: AcousticBlocks, with_boundaries=True):
    """Stamp the full ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega) + R(omega)``.

    Fast path over :func:`_assemble_reference`: the sparsity pattern of ``A(omega)`` is
    independent of ``omega`` (the stamps overwrite fixed rows/columns; only the duct
    phases and any frequency-dependent terminal closure change values), so it is captured
    once into a cached :class:`_AssemblyPlan` and each subsequent ``A(omega)`` is built by
    scattering the few omega-dependent values into a copy of the omega-independent entries
    -- no LIL build, no re-sort.  This is what makes the contour sweep (hundreds of
    factorizations) scale to large networks.  Results are identical to
    :func:`_assemble_reference` to round-off.

    ``with_boundaries`` controls the terminal reflection face ``R(omega)``; the plan is
    cached per value on ``blocks``.  If a storage block ``M`` is present (reserved; not in
    v1), the reference assembly is used instead.

    Parameters
    ----------
    omega : complex
        Angular frequency (rad/s), real for a forced sweep or complex on a stability
        contour.
    blocks : AcousticBlocks
        The frozen operator blocks (carries the lazily-built plan cache).
    with_boundaries : bool, optional
        Whether to stamp the terminal closures ``R(omega)`` (default True).

    Returns
    -------
    scipy.sparse.csc_matrix
        ``A(omega)``.
    """
    if blocks.M.nnz:
        # storage block present (reserved feature): omega-dependent pattern not modeled here
        return _assemble_reference(omega, blocks, with_boundaries)
    plan = blocks._plans.get(with_boundaries)
    if plan is None:
        plan = _build_plan(blocks, with_boundaries)
        blocks._plans[with_boundaries] = plan
    return plan.assemble(omega)
