"""Transfer / scattering matrices over a converged mean flow (theory.md s12.7).

The workflow is **force once, extract many**:

1. ``perturbation_response`` solves ``A(omega) x = b`` for the **full set of
   independent incoming excitations** of the network -- one per incoming
   characteristic at the terminals -- over a frequency array, and stores the
   complete perturbation fields plus the per-edge characteristic maps ``L_e``.
2. The returned ``PerturbationResponse`` reconstructs the ``N x N`` transfer
   matrix between **any** edge pair, or the scattering matrix between the forced
   terminals, with no further solve.

For a subsonic two-terminal network the incoming waves are ``{f_up, g_down}``
(acoustic) and ``h_up`` (entropy).  **Every** incoming wave is *prescribed* -- the
two acoustic ones by overwriting each terminal's boundary row, the entropy one by
overwriting the inlet edge's **transport row** (the edge view of nodal energy
conservation, theory.md s6.2) -- so nothing is ever left floating at a boundary.
A floated incoming entropy is exactly what contaminates the acoustic block: it
acquires a small amplitude that the (large) entropy->sound coupling at an area
change folds back into the acoustic waves.

The ``excite`` argument selects which wave families are *driven* with a unit
incoming amplitude; the rest stay prescribed, but to **zero**.  The default
``excite=("acoustic",)`` drives only the acoustic waves and pins the incoming
entropy to zero -- a clean, well-conditioned ``2 x 2`` acoustic response.  Adding
``"entropy"`` drives the entropy wave too for the full ``3 x 3`` perturbation
network (reacting scalars later extend the set the same way).  The system matrix
is identical across all columns, so one factorization serves every excitation.

Each individual excitation -- "drive these wave families at this boundary node" --
is handled by ``excite_perturbation``, exposed standalone so the raw perturbation
fields can be inspected directly.  ``perturbation_response`` simply drives every
forced terminal in turn, sharing the one factorization, and stacks the columns.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .characteristics import dx_to_char, basis_block_from_state
from .terminals import Terminal, find_terminals, _BOUNDARY_RIDS  # noqa: F401  (re-exported)
from . import matrices as mat
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA, ES_MDOT  # noqa: F401


def _edge_transforms(prob, x_bar, K):
    """Per-edge L_e = dx_to_char at the frozen mean state."""
    est = states_table(prob, x_bar)
    L = []
    for e in range(prob.n_edges):
        L.append(
            dx_to_char(
                float(est[ES_RHO, e]),
                float(est[ES_C, e]),
                float(est[ES_U, e]),
                float(est[ES_P, e]),
                float(est[ES_AREA, e]),
                K,
            )
        )
    return L


def _select_forcing(terms: List[Terminal], forcing: Optional[Sequence[int]]) -> List[Terminal]:
    if forcing is None:
        sel = list(terms)
    else:
        by_node = {t.node: t for t in terms}
        sel = []
        for nd in forcing:
            if nd not in by_node:
                raise ValueError(f"forcing location {nd} is not a 1-port terminal")
            sel.append(by_node[nd])
    if len(sel) != 2:
        raise ValueError(f"v1 scattering forces exactly 2 terminals; got {len(sel)} (pass `forcing=(node_a, node_b)`)")
    return sel


# Wave families and the characteristic indices each spans, in canonical order.
# Extends with reacting scalars (e.g. {"scalar:Z": (3,)}) without touching the
# driver: a family contributes one prescribed incoming wave per inlet it lives on.
_CHAR_OF_FAMILY = {"acoustic": (0, 1), "entropy": (2,)}


def _validate_excite(excite):
    if "acoustic" not in excite:
        raise ValueError("excite must include 'acoustic' (v1 always drives the acoustic waves)")
    unknown = [f for f in excite if f not in _CHAR_OF_FAMILY]
    if unknown:
        raise ValueError(f"unknown wave family/families {unknown}; choose from {sorted(_CHAR_OF_FAMILY)}")


def _excited_char_indices(families):
    """Characteristic indices spanned by the driven families, in canonical order."""
    return tuple(c for fam in ("acoustic", "entropy") if fam in families for c in _CHAR_OF_FAMILY[fam])


@dataclass(frozen=True)
class _Prescription:
    """One incoming wave to *prescribe* at a boundary (theory.md s12.7).

    The matrix ``row`` is overwritten with "incoming characteristic ``char`` of
    ``edge`` = rhs"; ``rhs`` is 1 when the wave is driven and 0 when it is pinned.
    """

    node: int  # boundary element the wave enters at
    kind: str  # wave family ("acoustic" or "entropy")
    row: int  # matrix row to overwrite
    edge: int  # edge whose characteristic is prescribed
    char: int  # characteristic index (0/1 acoustic, 2 entropy)


def _prescriptions(prob, sel) -> List[_Prescription]:
    """Every incoming wave of the forced terminals, in canonical order.

    Acoustic waves sit on each terminal's boundary row; the incoming entropy wave
    sits on the inlet edge's transport (nodal-energy) row -- always a duct *tail*
    edge, so it never collides with the duct stamp's head-edge entropy-phase
    relation (theory.md s6.2).  Entropy seats are always included (pinned to zero
    when not driven) so nothing floats at an inflow boundary.
    """
    pres = [_Prescription(t.node, "acoustic", t.row, t.edge, t.incoming) for t in sel]
    for t in sel:
        if t.at_tail:  # inflow seat carries an incoming entropy wave
            pres.append(_Prescription(t.node, "entropy", int(prob.transport_row0) + t.edge, t.edge, 2))
    return pres


@dataclass
class _ExcitationContext:
    """Frozen-mean machinery shared across every single-node excitation.

    The fully-prescribed operator ``A(omega)`` is identical for all excitations
    (every incoming wave occupies a prescribed row, so the columns differ only in
    the right-hand side), hence it is factorized **once** per frequency here and
    only back-substituted per excitation -- force once, extract many.
    """

    omegas: np.ndarray
    L: List[np.ndarray]
    est: np.ndarray
    K: float
    sel: List[Terminal]
    prescriptions: List[_Prescription]
    lus: list  # per-omega LU factorizations of the prescribed A(omega)
    n_solve: int
    n_col: int


def _build_excitation_context(prob, x_bar, omegas, forcing, *, eps, eps_fb, u_floor) -> _ExcitationContext:
    """Assemble and factorize the prescribed operator over the whole frequency array."""
    omegas = np.asarray(omegas, dtype=float)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    K = float(prob.tf[0]) / float(prob.tf[1])
    L = _edge_transforms(prob, x_bar, K)
    sel = _select_forcing(find_terminals(prob, x_bar), forcing)
    pres = _prescriptions(prob, sel)
    ns, n_col = int(prob.n_solve), int(prob.n_col)
    lus = []
    for omega in omegas:
        # The measurement driver closes *every* terminal itself (one independent
        # incoming wave per row), so the physical boundary stamp is skipped here.
        A = assemble_acoustic(omega, blocks, with_boundaries=False).tolil()
        for p in pres:  # prescribe *every* incoming wave; only the rhs distinguishes excitations
            A.rows[p.row] = []
            A.data[p.row] = []
            for v in range(3):
                A[p.row, ns * p.edge + v] = L[p.edge][p.char, v]
        lus.append(spla.splu(sp.csc_matrix(A)))
    return _ExcitationContext(omegas, L, states_table(prob, x_bar), K, sel, pres, lus, ns, n_col)


def _validate_modes(modes):
    if not modes:
        raise ValueError("modes must name at least one wave family")
    unknown = [f for f in modes if f not in _CHAR_OF_FAMILY]
    if unknown:
        raise ValueError(f"unknown wave family/families {unknown}; choose from {sorted(_CHAR_OF_FAMILY)}")


def _driven_prescriptions(ctx: _ExcitationContext, node, modes) -> List[_Prescription]:
    """The prescriptions at ``node`` that ``modes`` drives, in canonical family order."""
    if not any(t.node == node for t in ctx.sel):
        raise ValueError(f"node {node} is not a forced terminal; pass it in `forcing`")
    driven = []
    for fam in ("acoustic", "entropy"):  # canonical order, independent of how `modes` is ordered
        if fam not in modes:
            continue
        matches = [p for p in ctx.prescriptions if p.node == node and p.kind == fam]
        if not matches:
            raise ValueError(f"terminal {node} carries no incoming {fam} wave to drive")
        driven.extend(matches)
    return driven


@dataclass
class PerturbationField:
    """Perturbation fields produced by driving the waves at one boundary node.

    Returned by :func:`excite_perturbation`.  Each column of :attr:`X` is the full
    nodal perturbation vector for one driven incoming wave; :meth:`waves` projects
    any edge onto its characteristic amplitudes ``(f, g, h)``.

    Attributes
    ----------
    omegas : ndarray
        Angular frequencies (rad/s), shape ``(n_omega,)``.
    X : ndarray
        Nodal perturbation fields, shape ``(n_omega, n_driven, n_col)`` -- one
        column per driven wave at the node.
    L : list of ndarray
        Per-edge ``dx_to_char`` (3x3) maps at the frozen mean state.
    est : ndarray
        Frozen mean edge-state table.
    K : float
        ``cp / R`` of the mean gas.
    n_solve : int
        Solve-variable stride per edge in the nodal vector.
    node : int
        Boundary element that was driven.
    driven : list of tuple
        ``(kind, char)`` tag of each column of :attr:`X`, in canonical order.
    """

    omegas: np.ndarray
    X: np.ndarray
    L: List[np.ndarray]
    est: np.ndarray
    K: float
    n_solve: int
    node: int
    driven: list

    def waves(self, edge):
        """Characteristic amplitudes ``(f, g, h)`` at ``edge`` for every driven wave.

        Parameters
        ----------
        edge : int
            Edge id to project onto its characteristics.

        Returns
        -------
        ndarray
            Shape ``(n_omega, n_char, n_driven)``: column ``k`` is the full wave
            vector (all characteristics) of driven field ``k`` along ``edge``.
        """
        ns, nc = self.n_solve, self.L[edge].shape[0]
        Xe = self.X[:, :, ns * edge : ns * edge + nc]  # (n_omega, n_driven, n_char)
        return np.einsum("ij,okj->oik", self.L[edge], Xe)  # (n_omega, n_char, n_driven)


def excite_perturbation(
    prob, x_bar, omegas, node, modes=("acoustic",), *, forcing=None, eps=None, eps_fb=1e-6, u_floor=1e-8, _context=None
):
    """Solve the perturbation field for incoming waves driven at one boundary node.

    Drives the incoming wave(s) of the requested ``modes`` at terminal ``node`` to
    unit amplitude while **pinning every other** incoming wave of the network to
    zero, then back-substitutes the frozen-mean operator ``A(omega)`` over the whole
    frequency array.  This is the single-excitation building block of
    :func:`perturbation_response`, exposed standalone so the raw perturbation fields
    can be inspected directly.

    Parameters
    ----------
    prob : Problem
        Compiled flow network.
    x_bar : ndarray
        Converged mean-flow state vector.
    omegas : array_like
        Angular frequencies (rad/s) to solve at.
    node : int
        Boundary element id to drive.  Must be one of the forced 1-port terminals
        (see ``forcing``).
    modes : sequence of str, optional
        Wave families to drive at ``node`` (``"acoustic"`` and/or ``"entropy"``).
        Each family contributes one driven incoming wave per characteristic it spans
        at this terminal.  Default ``("acoustic",)``.
    forcing : tuple of int, optional
        The pair of terminal node ids whose incoming waves are prescribed (default:
        the network's two terminals).  All of their incoming waves except the driven
        ones are pinned to zero, so nothing floats at a boundary.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to ``build_acoustic_blocks``.

    Returns
    -------
    PerturbationField
        The perturbation field, one column per driven wave at ``node``.

    Raises
    ------
    ValueError
        If ``node`` is not a forced terminal, a requested family is unknown, or an
        entropy wave is requested at a terminal that carries no incoming entropy.
    """
    modes = tuple(modes)
    _validate_modes(modes)
    ctx = _context or _build_excitation_context(prob, x_bar, omegas, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    driven = _driven_prescriptions(ctx, node, modes)
    n_driven = len(driven)

    X = np.zeros((ctx.omegas.size, n_driven, ctx.n_col), dtype=np.complex128)
    for i, lu in enumerate(ctx.lus):
        b = np.zeros((ctx.n_col, n_driven), dtype=np.complex128)
        for k, p in enumerate(driven):
            b[p.row, k] = 1.0  # unit amplitude for this driven incoming wave
        X[i] = lu.solve(b).T

    return PerturbationField(
        omegas=ctx.omegas,
        X=X,
        L=ctx.L,
        est=ctx.est,
        K=ctx.K,
        n_solve=ctx.n_solve,
        node=node,
        driven=[(p.kind, p.char) for p in driven],
    )


def perturbation_response(
    prob, x_bar, omegas, forcing=None, *, excite=("acoustic",), eps=None, eps_fb=1e-6, u_floor=1e-8
):
    """Drive every forced incoming wave and store the perturbation fields.

    Solves one single-node excitation per (terminal, wave family) via
    :func:`excite_perturbation` -- sharing a single factorization of the prescribed
    operator across them -- and stacks the resulting fields into a
    :class:`PerturbationResponse`, from which transfer/scattering matrices are
    extracted without re-solving.

    Parameters
    ----------
    prob : Problem
        Compiled flow network.
    x_bar : ndarray
        Converged mean-flow state vector.
    omegas : array_like
        Angular frequencies (rad/s) to solve at.
    forcing : tuple of int, optional
        The pair of terminal node ids to force (default: the network's two
        terminals).
    excite : sequence of str, optional
        Wave families to *drive* with a unit incoming amplitude.  ``"acoustic"`` is
        mandatory; add ``"entropy"`` for the full ``3 x 3`` perturbation network.
        Families not listed stay prescribed but pinned to zero, so the boundaries
        never float.  Default ``("acoustic",)`` -- the clean, well-conditioned
        ``2 x 2`` acoustic response with the incoming entropy pinned out.
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to ``build_acoustic_blocks``.

    Returns
    -------
    PerturbationResponse
        The stacked independent perturbation fields.

    Raises
    ------
    ValueError
        If ``excite`` omits ``"acoustic"``/names an unknown family, or entropy is
        requested with no inflow terminal to seat it.
    """
    excite = tuple(excite)
    _validate_excite(excite)
    ctx = _build_excitation_context(prob, x_bar, omegas, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor)

    # One driven excitation per (terminal, family): all acoustic first, then the
    # incoming entropy at each inflow seat -> canonical column order (f, g, h).
    excitations = [(t.node, "acoustic") for t in ctx.sel]
    if "entropy" in excite:
        entropy_nodes = [t.node for t in ctx.sel if t.at_tail]
        if not entropy_nodes:
            raise ValueError("entropy excitation requested but no inflow terminal found to seat it")
        excitations += [(nd, "entropy") for nd in entropy_nodes]

    by_node = {t.node: t for t in ctx.sel}
    cols, forcing_kinds = [], []
    for nd, fam in excitations:  # the dedicated routine solves each one; the factorization is shared
        field = excite_perturbation(prob, x_bar, omegas, nd, modes=(fam,), _context=ctx)
        cols.append(field.X[:, 0, :])  # one driven wave per call -> (n_omega, n_col)
        forcing_kinds.append((fam, by_node[nd]))

    X = np.stack(cols, axis=1)  # (n_omega, n_force, n_col)
    cidx = _excited_char_indices({fam for _nd, fam in excitations})
    return PerturbationResponse(
        omegas=ctx.omegas,
        X=X,
        L=ctx.L,
        est=ctx.est,
        K=ctx.K,
        n_solve=ctx.n_solve,
        forcing=ctx.sel,
        forcing_kinds=forcing_kinds,
        cidx=cidx,
    )


@dataclass
class PerturbationResponse:
    """Stored independent perturbation fields; extracts N x N matrices on demand."""

    omegas: np.ndarray  # (n_omega,)
    X: np.ndarray  # (n_omega, n_force, n_col) -- one forced field per excitation
    L: List[np.ndarray]  # per-edge dx_to_char (3x3) at the mean state
    est: np.ndarray  # frozen mean edge-state table (for basis blocks / wave speeds)
    K: float  # cp / R
    n_solve: int
    forcing: List[Terminal]
    forcing_kinds: list
    cidx: tuple = (0, 1, 2)  # characteristic indices spanned by the driven waves

    @property
    def n(self) -> int:
        """Matrix dimension: the number of driven perturbation waves (2 or 3)."""
        return len(self.cidx)

    @property
    def n_char(self) -> int:
        """Characteristic count per edge (3 for inert flow)."""
        return self.L[0].shape[0]

    def _waves(self, edge):
        """Characteristic amplitudes (f, g, h) at ``edge`` for every (omega, case).

        Returns an array of shape (n_omega, n_char, n_force): column k is the full
        wave vector of forced field k (all characteristics, driven or not).
        """
        ns = self.n_solve
        Xe = self.X[:, :, ns * edge : ns * edge + self.n_char]  # (n_omega, n_force, n_char)
        return np.einsum("ij,okj->oik", self.L[edge], Xe)  # (n_omega, n_char, n_force)

    _DIAGONAL_BASES = ("char", "riemann")  # flavors that do not mix characteristics

    def transfer_matrix(self, a, b, basis="char"):
        """Transfer matrix ``T_ba`` mapping the driven waves at ``a`` to those at ``b``.

        The dimension is ``self.n`` (2 for the default acoustic excitation, 3 with
        entropy).  Read along each edge's own arrow; ``basis`` selects the variable
        flavor (``characteristics.BASIS_LABELS``).  Shape ``(n_omega, n, n)``.
        """
        ci = list(self.cidx)
        Wa = self._waves(a)[:, ci, :]  # (n_omega, n, n_force) over driven characteristics
        Wb = self._waves(b)[:, ci, :]
        T = Wb @ np.linalg.inv(Wa)  # square and well-conditioned (every incoming wave prescribed)
        if basis == "char":
            return T
        if self.n < self.n_char and basis not in self._DIAGONAL_BASES:
            raise ValueError(
                f"flavor {basis!r} mixes characteristics, so it needs the full response; "
                f"re-run with excite=('acoustic', 'entropy', ...) or use 'char'/'riemann'"
            )
        Ba = basis_block_from_state(basis, self.est[:, a], self.K)[np.ix_(ci, ci)]
        Bb = basis_block_from_state(basis, self.est[:, b], self.K)[np.ix_(ci, ci)]
        return mat.tm_in_basis(T, Ba, Bb)

    def scattering_matrix(self, a, b, basis="char"):
        """Scattering matrix between stations ``a`` (upstream) and ``b``.

        Dimension is ``self.n`` (2 acoustic-only, 3 with entropy).  Incoming waves
        (``a``'s downstream-running + ``b``'s upstream-running) map to the outgoing
        ones; ordering follows ``matrices.scattering_labels``.  ``basis`` may only be
        a flavor diagonal in the characteristics (``char`` or ``riemann``); mixed
        flavors are undefined for a scattering matrix.
        """
        ua, ca = float(self.est[ES_U, a]), float(self.est[ES_C, a])
        ub, cb = float(self.est[ES_U, b]), float(self.est[ES_C, b])
        T = self.transfer_matrix(a, b, basis="char")
        S, _in, _out = mat.tm_to_sm(T, ua, ca, ub, cb)
        if basis == "char":
            return S
        if basis != "riemann":
            raise ValueError("scattering_matrix basis must be 'char' or 'riemann' (diagonal in the waves)")
        # riemann rescales each wave (f/c, g/c, -h/rho) -- diagonal, so scale in/out amplitudes
        incoming, outgoing = mat.scattering_labels(ua, ca, ub, cb, self.n)
        din = np.array([self._wave_scale("riemann", st, i, a, b) for (st, i) in incoming])
        dout = np.array([self._wave_scale("riemann", st, i, a, b) for (st, i) in outgoing])
        return (dout[None, :, None] * S) / din[None, None, :]

    def _wave_scale(self, basis, station, i, a, b):
        """Diagonal scale factor of characteristic ``i`` at the chosen station."""
        e = a if station == "a" else b
        B = basis_block_from_state(basis, self.est[:, e], self.K)
        return B[i, i]

    def scattering_labels(self, a, b):
        """Ordered (station, char-index) tags of the SM's incoming and outgoing waves."""
        ua, ca = float(self.est[ES_U, a]), float(self.est[ES_C, a])
        ub, cb = float(self.est[ES_U, b]), float(self.est[ES_C, b])
        return mat.scattering_labels(ua, ca, ub, cb, self.n)

    # -- notebook plotting (edge-aware labels) ------------------------------

    def _basis_labels(self, basis):
        """Per-variable symbols for ``basis``, trimmed to this response's dimension."""
        from .characteristics import BASIS_LABELS

        syms = BASIS_LABELS.get(basis)
        return tuple(syms[: self.n]) if syms else None

    def plot_transfer_matrix(self, a, b, freqs=None, *, basis="char", **kwargs):
        """Plot the transfer matrix ``T_ba`` in ``basis``, with edge-subscripted labels.

        Convenience wrapper that *converts* the matrix to ``basis`` (via
        :meth:`transfer_matrix`) and labels it to match, supplying the station edges so
        each entry reads ``var_a -> var_b`` (e.g. ``f₁ -> f₂``) instead of the
        ambiguous ``f -> f``.  Unlike the free :func:`fns.plotting.plot_transfer_matrix`,
        the ``basis`` here genuinely changes the matrix values, not just the labels.

        Parameters
        ----------
        a, b : int
            Upstream / downstream edge ids; the matrix maps the waves at ``a`` to
            those at ``b``.
        freqs : array_like, optional
            x-axis values (default: ``self.omegas`` in rad/s).  Pass
            ``self.omegas / (2*np.pi)`` to plot against frequency in Hz.
        basis : str, optional
            Variable flavor (``characteristics.BASIS_LABELS``; e.g. ``"char"``,
            ``"primitive"``, ``"network"``).  Default ``"char"``.
        **kwargs
            Forwarded to :func:`fns.plotting.plot_transfer_matrix`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_transfer_matrix as _plot

        T = self.transfer_matrix(a, b, basis=basis)
        x = self.omegas if freqs is None else freqs
        return _plot(T, x, labels=self._basis_labels(basis), edges=(a, b), **kwargs)

    def plot_scattering_matrix(self, a, b, freqs=None, *, basis="char", **kwargs):
        """Plot the scattering matrix between ``a`` and ``b`` with station-tagged labels.

        Convenience wrapper that *converts* the matrix to ``basis`` (via
        :meth:`scattering_matrix`) and labels it to match, supplying both the station
        edges and the incoming/outgoing wave partition so each entry is titled by its
        own station-subscripted waves (e.g. ``f₁ -> g₁`` for a reflection at edge ``a``).

        Parameters
        ----------
        a, b : int
            Upstream / downstream edge ids of the cut.
        freqs : array_like, optional
            x-axis values (default: ``self.omegas`` in rad/s).
        basis : str, optional
            Wave flavor (``"char"`` or ``"riemann"`` -- diagonal in the waves).
            Default ``"char"``.
        **kwargs
            Forwarded to :func:`fns.plotting.plot_scattering_matrix`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_scattering_matrix as _plot

        S = self.scattering_matrix(a, b, basis=basis)
        x = self.omegas if freqs is None else freqs
        return _plot(
            S, x, labels=self._basis_labels(basis), edges=(a, b), partition=self.scattering_labels(a, b), **kwargs
        )

    # -- acoustics-only convenience (entropy dropped) -----------------------

    def _acoustic_cols(self):
        return [k for k, (kind, _) in enumerate(self.forcing_kinds) if kind == "acoustic"]

    def acoustic_transfer_matrix(self, a, b):
        """The 2x2 acoustic ``(f, g)`` transfer matrix (incoming entropy = 0).

        Reconstructed from the two acoustic forcings alone -- well conditioned even
        at a quiescent mean state, where the *full* 3x3 degenerates because the
        entropy wave stops convecting (``tau_0 -> inf``).  The acoustic columns pin
        the incoming entropy to zero, so this is the clean acoustic block whether or
        not the entropy wave is also driven (no entropy-noise contamination).
        """
        cols = self._acoustic_cols()
        if len(cols) != 2:
            raise ValueError(f"expected 2 acoustic forcings, found {len(cols)}")
        Wa = self._waves(a)[:, :2, :][:, :, cols]  # (n_omega, 2, 2)
        Wb = self._waves(b)[:, :2, :][:, :, cols]
        return Wb @ np.linalg.inv(Wa)

    def acoustic_scattering_matrix(self, a, b):
        """2x2 acoustic scattering matrix, incoming ``(f_a, g_b)`` -> outgoing ``(g_a, f_b)``.

        Ordering matches the full :meth:`scattering_matrix` (reflection at ``a``
        first, transmission to ``b`` second); ``tm_fg_to_sm2`` yields the classic
        ``(f_b, g_a)`` ordering, so the two rows are swapped.
        """
        return mat.tm_fg_to_sm2(self.acoustic_transfer_matrix(a, b))[:, ::-1, :]


# -- back-compatibility aliases (pre-reframe names) -------------------------

acoustic_response = perturbation_response
AcousticResponse = PerturbationResponse
