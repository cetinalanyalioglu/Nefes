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
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .characteristics import dx_to_char, basis_block_from_state
from . import matrices as mat
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA, ES_MDOT
from ..elements.ids import MASS_FLOW_INLET, PT_INLET, P_OUTLET

_BOUNDARY_RIDS = (MASS_FLOW_INLET, PT_INLET, P_OUTLET)


@dataclass
class Terminal:
    """A 1-port boundary edge where an incoming wave can be injected/read."""

    node: int  # the boundary element
    rid: int  # its residual id (one of _BOUNDARY_RIDS)
    edge: int  # the single incident edge
    at_tail: bool  # True if the boundary is the edge's tail (wave enters as f)
    row: int  # the boundary element's single equation row
    incoming: int  # acoustic wave index injected here: 0 (f) if at_tail else 1 (g)
    outgoing: int  # the reflected/transmitted acoustic wave index read here
    inflowing: bool  # True if the mean flow *enters* the domain here (carries entropy in)


def find_terminals(prob, x_bar=None) -> List[Terminal]:
    """All 1-port boundary terminals of the network (edges at a boundary node).

    When ``x_bar`` is given, ``inflowing`` is set from the mean flow direction so
    the incoming entropy excitation can be placed at genuine inlets.
    """
    est = states_table(prob, x_bar) if x_bar is not None else None
    terms = []
    for n in range(prob.n_nodes):
        rid = int(prob.node_rid[n])
        if rid not in _BOUNDARY_RIDS:
            continue
        base = int(prob.row_ptr[n])
        deg = int(prob.row_ptr[n + 1]) - base
        if deg != 1:
            raise ValueError(f"boundary node {n} has degree {deg}; a 1-port must have one edge")
        edge = int(prob.col_edge[base])
        at_tail = int(prob.tail_node[edge]) == n
        incoming = 0 if at_tail else 1
        inflowing = False
        if est is not None:
            mdot = float(est[ES_MDOT, edge])
            inflowing = (mdot > 0.0) if at_tail else (mdot < 0.0)
        terms.append(
            Terminal(
                node=n,
                rid=rid,
                edge=edge,
                at_tail=at_tail,
                row=int(prob.node_row_ptr[n]),
                incoming=incoming,
                outgoing=1 - incoming,
                inflowing=inflowing,
            )
        )
    return terms


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


def _boundary_excitations(prob, sel, entropy_terms, force_entropy):
    """Build the list of incoming-wave excitations to *prescribe* at the boundaries.

    Each entry is ``(kind, terminal, row, char, driven)``: the boundary ``row`` is
    overwritten with "incoming characteristic ``char`` of ``terminal.edge`` = rhs",
    and ``rhs`` is 1 if this wave is ``driven`` else 0.  Acoustic waves sit on the
    terminal's boundary row; the incoming entropy wave sits on the inlet edge's
    transport (nodal-energy) row -- always a duct *tail* edge, so it never collides
    with the duct stamp's head-edge entropy-phase relation (theory.md s6.2).
    Un-driven waves are still prescribed (to zero) -- nothing is left floating.
    """
    exc = [("acoustic", t, t.row, t.incoming, True) for t in sel]
    for t in entropy_terms:
        exc.append(("entropy", t, int(prob.transport_row0) + t.edge, 2, force_entropy))
    return exc


def perturbation_response(
    prob, x_bar, omegas, forcing=None, *, excite=("acoustic",), eps=None, eps_fb=1e-6, u_floor=1e-8
):
    """Drive the selected incoming waves and store the perturbation fields.

    ``omegas`` is the user frequency array; ``forcing`` is the pair of terminal
    node ids (default: the network's two terminals).  ``excite`` is the tuple of
    wave families to *drive* (``"acoustic"`` always; add ``"entropy"`` for the full
    ``3 x 3``).  Every incoming wave is prescribed -- driven ones to unit amplitude,
    the rest to zero -- so the boundaries never float.  Returns a
    ``PerturbationResponse`` from which transfer/scattering matrices are extracted
    without re-solving.
    """
    excite = tuple(excite)
    _validate_excite(excite)
    omegas = np.asarray(omegas, dtype=float)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    K = float(prob.tf[0]) / float(prob.tf[1])
    L = _edge_transforms(prob, x_bar, K)
    terms = find_terminals(prob, x_bar)
    sel = _select_forcing(terms, forcing)
    entropy_terms = [t for t in sel if t.at_tail]  # inlets carry an incoming entropy wave
    force_entropy = "entropy" in excite
    if force_entropy and not entropy_terms:
        raise ValueError("entropy excitation requested but no inflow terminal found to seat it")

    ns = int(prob.n_solve)
    n = int(prob.n_col)
    exc = _boundary_excitations(prob, sel, entropy_terms, force_entropy)
    driven = [e for e in exc if e[4]]  # the columns we solve for (unit-amplitude waves)
    forcing_kinds = [(kind, t) for (kind, t, _row, _char, _drv) in driven]
    n_force = len(driven)

    X = np.zeros((omegas.size, n_force, n), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        A = assemble_acoustic(omega, blocks).tolil()
        for _kind, t, row, char, _drv in exc:  # prescribe *every* incoming wave (driven or pinned)
            A.rows[row] = []
            A.data[row] = []
            for v in range(3):
                A[row, ns * t.edge + v] = L[t.edge][char, v]
        b = np.zeros((n, n_force), dtype=np.complex128)
        for k, (_kind, _t, row, _char, _drv) in enumerate(driven):  # unit amplitude for the driven column
            b[row, k] = 1.0
        lu = spla.splu(sp.csc_matrix(A))
        X[i] = lu.solve(b).T

    est = states_table(prob, x_bar)
    cidx = _excited_char_indices({kind for (kind, _t) in forcing_kinds})
    return PerturbationResponse(
        omegas=omegas, X=X, L=L, est=est, K=K, n_solve=ns, forcing=sel, forcing_kinds=forcing_kinds, cidx=cidx
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
