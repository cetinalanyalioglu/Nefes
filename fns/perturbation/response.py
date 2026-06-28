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

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operator import build_acoustic_blocks, assemble_acoustic
from .stamps import _terminal_closure, _set_row
from .characteristics import dx_to_char, basis_block_from_state
from .modeshape import build_geometry, reconstruct_field, NetworkGeometry
from .terminals import Terminal, find_terminals, _BOUNDARY_RIDS  # noqa: F401  (re-exported)
from . import matrices as mat
from ..solver.control import states_table
from ..derive import ES_RHO, ES_C, ES_U, ES_P, ES_AREA, ES_MDOT  # noqa: F401


class TransferMatrixWarning(UserWarning):
    """A fitted transfer/scattering matrix may not be a genuine 2-port descriptor.

    Raised (as a warning, never an exception) by :meth:`PerturbationResponse.transfer_matrix`
    and :meth:`~PerturbationResponse.scattering_matrix` when the two edges straddle an internal
    branch point -- so no transfer matrix exists and the returned matrix is only a least-squares
    best fit -- or when the response is too under-determined to verify either way.  Silence it
    with :func:`warnings.filterwarnings` once you have understood the diagnostic.
    """


def _edge_transforms(prob, x_bar, K, cals=None):
    """Per-edge L_e = dx_to_char at the frozen mean state.

    ``cals`` (optional): per-edge caloric rows (:func:`characteristics.edge_caloric`);
    when given, edge ``e`` uses ``cals[e]`` instead of the perfect-gas ``K`` form.
    """
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
                None if cals is None else cals[e],
            )
        )
    return L


def _select_forcing(terms: List[Terminal], forcing: Optional[Sequence[int]]) -> List[Terminal]:
    """The terminals to *drive* (default: all).  Every terminal is neutralized regardless."""
    if forcing is None:
        sel = list(terms)
    else:
        by_node = {t.node: t for t in terms}
        sel = []
        for nd in forcing:
            if nd not in by_node:
                raise ValueError(f"forcing location {nd} is not a 1-port terminal")
            sel.append(by_node[nd])
    if len(sel) < 2:
        raise ValueError(f"scattering needs at least 2 driven terminals; got {len(sel)}")
    return sel


def _frozen_bc(prob, node):
    """The explicit (stamping) ``PerturbationBC`` at ``node``, or None if it inherits / is unset."""
    node_bc = prob.node_bc or ()
    bc = node_bc[node] if node < len(node_bc) else None
    return bc if (bc is not None and getattr(bc, "stamps_terminal", False)) else None


def _resolve_frozen(prob, freeze):
    """Resolve the ``freeze`` node specifiers to a set of terminal node ids.

    ``freeze`` is a sequence whose elements are node ids (``int``) or element names
    (``str``).  Each must name an **existing** node that is a 1-port boundary terminal.

    Raises
    ------
    ValueError
        If an entry references a node that does not exist, names no element, is an
        ambiguous name (carried by more than one node), or is not a 1-port terminal.
    TypeError
        If an entry is neither an int node id nor a str node name.
    """
    if not freeze:
        return frozenset()
    n_nodes = int(prob.n_nodes)
    names = tuple(getattr(prob, "node_names", ()) or ())
    term_nodes = {t.node for t in find_terminals(prob)}
    frozen = set()
    for spec in freeze:
        if isinstance(spec, str):
            matches = [i for i, nm in enumerate(names) if nm == spec]
            if not matches:
                raise ValueError(f"freeze references node name {spec!r}, which no element carries")
            if len(matches) > 1:
                raise ValueError(f"freeze node name {spec!r} is ambiguous: nodes {matches} all carry it; use the id")
            node = matches[0]
        elif isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
            node = int(spec)
            if not 0 <= node < n_nodes:
                raise ValueError(f"freeze references node {node}, which does not exist (need 0 <= node < {n_nodes})")
        else:
            raise TypeError(f"freeze entries must be int node ids or str node names; got {spec!r}")
        if node not in term_nodes:
            raise ValueError(f"freeze references node {spec!r} (node {node}), which is not a 1-port terminal")
        frozen.add(node)
    return frozenset(frozen)


# Wave families and the characteristic indices each spans, in canonical order.
# Extends with reacting scalars (e.g. {"scalar:Z": (3,)}) without touching the
# driver: a family contributes one prescribed incoming wave per inlet it lives on.
_CHAR_OF_FAMILY = {"acoustic": (0, 1), "entropy": (2,)}

# Wave symbol per characteristic index, for multiport scattering labels.
_CHAR_SYM = ("f", "g", "h")

# LaTeX special characters to escape when an element name is dropped into a
# ``\text{}`` subscript (so an arbitrary label cannot break the MathJax string).
_TEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_text(s) -> str:
    """Escape ``s`` for use inside a LaTeX ``\\text{}`` group."""
    return "".join(_TEX_ESCAPE.get(ch, ch) for ch in str(s))


def _reject_unsupported_families(families, scalar_names=()):
    """Reject any requested wave family the v1 driver cannot honor.

    A name that is a transported reacting scalar (in ``scalar_names``) is a *deferred*
    capability, so it raises :class:`NotImplementedError` -- driving a scalar wave needs the
    compositional (scalar -> acoustic) scattering closure, which is not wired yet; read the
    convected scalar response from :meth:`fns.perturbation.ForcedResponse.waves` instead.  Any
    other unrecognized name is a typo and raises :class:`ValueError`.
    """
    scalars = tuple(scalar_names or ())
    deferred = sorted({f for f in families if f in scalars})
    if deferred:
        raise NotImplementedError(
            f"reacting-scalar wave families {deferred} have no port in the scattering measurement yet "
            "(scalar scattering matrices are deferred); only 'acoustic' and 'entropy' are measured here. "
            "To *drive* a scalar, seat it at an inflow with PerturbationBC.<inlet>(driven=(...)) and read "
            "the field from ForcedResponse.waves()."
        )
    unknown = sorted({f for f in families if f not in _CHAR_OF_FAMILY})
    if unknown:
        raise ValueError(f"unknown wave family/families {unknown}; choose from {sorted(_CHAR_OF_FAMILY)}")


def _validate_excite(excite, scalar_names=()):
    if "acoustic" not in excite:
        raise ValueError("excite must include 'acoustic' (v1 always drives the acoustic waves)")
    _reject_unsupported_families(excite, scalar_names)


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


def _seats_entropy(t, est, u_floor):
    """Whether an incoming entropy wave enters the domain at terminal ``t``.

    Decided by the mean flow direction (not the element type): entropy is incoming
    wherever the convected wave propagates *into* the domain.  :func:`matrices.partition`
    settles this from the wave-speed sign, including the genuine inlet/outlet of a
    reversed-flow boundary and the quiescent fallback (entropy pinned downstream at
    ``|u| < u_floor``, so a quiescent *tail* seats it).
    """
    u, c = float(est[ES_U, t.edge]), float(est[ES_C, t.edge])
    incoming, _ = mat.partition(u, c, "a" if t.at_tail else "b", u_floor=u_floor)
    return 2 in incoming


def _prescriptions(prob, terms, est, u_floor) -> List[_Prescription]:
    """Every incoming wave of *all* terminals, in canonical order.

    Acoustic waves sit on each terminal's boundary row; the incoming entropy wave sits
    on the **genuine-inflow** edge's transport (nodal-energy) row.  For forward flow that
    is a duct *tail* edge; under boundary flow reversal it is the reversed terminal's edge
    (a duct *head*), which is free precisely because the duct stamp puts its entropy-phase
    relation on the *downstream* edge (theory.md s6.2).  Entropy seats are always included
    (pinned to zero when not driven) so nothing floats at an inflow boundary.

    This runs over the network's *full* terminal set so that **every** terminal is turned
    into a pure source (its incoming wave prescribed); the subset that is actually *driven*
    with a unit amplitude is chosen later from ``forcing``.  Neutralizing every boundary --
    not just the driven ones -- is what makes the measured matrices boundary-independent: a
    terminal left at its (reflecting) inherited mean BC would close a spurious cavity and
    inject resonances.
    """
    pres = [_Prescription(t.node, "acoustic", t.row, t.edge, t.incoming) for t in terms]
    for t in terms:
        if _seats_entropy(t, est, u_floor):  # the genuine-inflow terminal carries incoming entropy
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

    freqs: np.ndarray  # excitation frequencies (Hz)
    L: List[np.ndarray]
    est: np.ndarray
    K: float
    sel: List[Terminal]  # the (open) terminals driven with a unit amplitude (subset of all)
    terminals: List[Terminal]  # the open terminals (neutralized into ports); frozen ones excluded
    prescriptions: List[_Prescription]  # every open terminal's incoming wave (all neutralized)
    lus: list  # per-omega LU factorizations of the prescribed A(omega)
    n_solve: int
    n_col: int
    u_floor: float  # speed below which a station is treated as quiescent
    cals: Optional[list] = None  # per-edge caloric rows (reacting "network" flavor)
    frozen: frozenset = frozenset()  # terminal node ids kept at their physical BC (not measured)


def _build_excitation_context(
    prob, x_bar, freqs, forcing, *, eps, eps_fb, u_floor, frozen=frozenset()
) -> _ExcitationContext:
    """Assemble and factorize the prescribed operator over the whole frequency array.

    ``frozen`` (terminal node ids) are **kept at their physical boundary condition**
    rather than neutralized into measurement ports: each is dropped from the driven /
    prescribed set and instead keeps its closure -- an explicit ``PerturbationBC`` stamped
    via :func:`stamps._terminal_closure`, or the inherited linearized mean BC already in
    ``J_alg`` (a plain ``wall`` -> ``mdot' = 0`` -> ``R = +1``).  This folds an
    interior branch terminated by a wall (a closed stub / side resonator) into the
    operator, so the network reduces to its genuine open ports.
    """
    freqs = np.asarray(freqs, dtype=float)
    omegas = 2.0 * np.pi * freqs  # operator assembly works in angular frequency (rad/s)
    blocks = build_acoustic_blocks(prob, x_bar, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    K = float(prob.tf[0]) / float(prob.tf[1])
    est = states_table(prob, x_bar)
    cals = blocks.cals
    L = _edge_transforms(prob, x_bar, K, cals)
    all_terms = find_terminals(prob, x_bar)
    open_terms = [t for t in all_terms if t.node not in frozen]  # the ports we measure
    sel = _select_forcing(open_terms, forcing)
    pres = _prescriptions(prob, open_terms, est, u_floor)  # neutralize every *open* terminal into a pure source
    # frozen terminals carrying an explicit (stamping) BC: their omega-dependent closure must
    # be re-applied each frequency; an inherited (wall / mass-flow / pt) terminal keeps its
    # J_alg row untouched, so it needs no stamp.
    frozen_closured = [t for t in all_terms if t.node in frozen and _frozen_bc(prob, t.node) is not None]
    ns, n_col = int(prob.n_solve), int(prob.n_col)
    lus = []
    for omega in omegas:
        # The measurement driver closes every *open* terminal itself (one independent
        # incoming wave per row), so the physical boundary stamp is skipped here -- except
        # for the frozen terminals, whose explicit closure is stamped back in below.
        A = assemble_acoustic(omega, blocks, with_boundaries=False).tolil()
        for t in frozen_closured:  # keep this terminal closed by its own physical BC
            cal = None if not cals else cals[t.edge]
            for row, cols, coeff, _rhs in _terminal_closure(prob, est, K, t, prob.node_bc[t.node], omega, cal):
                _set_row(A, row, cols, coeff, (), ())
        for p in pres:  # prescribe every open incoming wave; only the rhs distinguishes excitations
            A.rows[p.row] = []
            A.data[p.row] = []
            for v in range(3):
                A[p.row, ns * p.edge + v] = L[p.edge][p.char, v]
        lus.append(spla.splu(sp.csc_matrix(A)))
    return _ExcitationContext(
        freqs, L, est, K, sel, open_terms, pres, lus, ns, n_col, float(u_floor), cals, frozenset(frozen)
    )


def _validate_modes(modes, scalar_names=()):
    if not modes:
        raise ValueError("modes must name at least one wave family")
    _reject_unsupported_families(modes, scalar_names)


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
        if fam == "entropy":  # entropy does not convect at a quiescent station (tau_0 -> inf)
            for p in matches:
                if abs(float(ctx.est[ES_U, p.edge])) < ctx.u_floor:
                    raise ValueError(
                        f"entropy excitation is undefined at quiescent terminal {node} (mean u ~ 0): "
                        "the entropy wave does not convect, so it has no response; use excite=('acoustic',)"
                    )
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
    freqs : ndarray
        Excitation frequencies (Hz), shape ``(n_freq,)``.
    X : ndarray
        Nodal perturbation fields, shape ``(n_freq, n_driven, n_col)`` -- one
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

    freqs: np.ndarray
    X: np.ndarray
    L: List[np.ndarray]
    est: np.ndarray
    K: float
    n_solve: int
    node: int
    driven: list

    def __repr__(self) -> str:
        """One-line summary: driven node, the waves driven there, and the sweep extent."""
        f = np.asarray(self.freqs, dtype=float)
        n = f.size
        span = "empty" if n == 0 else (f"f = {f[0]:.1f} Hz" if n == 1 else f"f in [{f.min():.1f}, {f.max():.1f}] Hz")
        waves = ", ".join(f"{kind}:{char}" for kind, char in self.driven) or "none"
        return f"PerturbationField: node {self.node} driving [{waves}], {n} frequenc{'y' if n == 1 else 'ies'} ({span})"

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
    prob, x_bar, freqs, node, modes=("acoustic",), *, forcing=None, eps=None, eps_fb=1e-6, u_floor=1e-8, _context=None
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
    freqs : array_like
        Frequencies (Hz) to solve at.
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
    _validate_modes(modes, getattr(prob, "scalar_names", ()))
    ctx = _context or _build_excitation_context(prob, x_bar, freqs, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor)
    driven = _driven_prescriptions(ctx, node, modes)
    n_driven = len(driven)

    X = np.zeros((ctx.freqs.size, n_driven, ctx.n_col), dtype=np.complex128)
    for i, lu in enumerate(ctx.lus):
        b = np.zeros((ctx.n_col, n_driven), dtype=np.complex128)
        for k, p in enumerate(driven):
            b[p.row, k] = 1.0  # unit amplitude for this driven incoming wave
        X[i] = lu.solve(b).T

    return PerturbationField(
        freqs=ctx.freqs,
        X=X,
        L=ctx.L,
        est=ctx.est,
        K=ctx.K,
        n_solve=ctx.n_solve,
        node=node,
        driven=[(p.kind, p.char) for p in driven],
    )


def perturbation_response(
    prob, x_bar, freqs, forcing=None, *, excite=("acoustic",), freeze=(), eps=None, eps_fb=1e-6, u_floor=1e-8
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
    freqs : array_like
        Frequencies (Hz) to solve at.
    forcing : tuple of int, optional
        The pair of terminal node ids to force (default: every open terminal).
    excite : sequence of str, optional
        Wave families to *drive* with a unit incoming amplitude.  ``"acoustic"`` is
        mandatory; add ``"entropy"`` for the full ``3 x 3`` perturbation network.
        Families not listed stay prescribed but pinned to zero, so the boundaries
        never float.  Default ``("acoustic",)`` -- the clean, well-conditioned
        acoustic response with the incoming entropy pinned out.
    freeze : sequence of int or str, optional
        Terminals whose **physical boundary condition is kept** during the measurement
        instead of being neutralized into a port.  Each entry is a node id (``int``) or
        an element name (``str``) and must name an existing 1-port terminal.  A frozen
        terminal stays closed by its own BC -- a ``wall`` reflects (``R = +1``), an
        explicit ``PerturbationBC`` applies its closure -- so an interior branch
        terminated by a wall (a closed stub / side resonator) is folded into the operator
        and the network reduces to its genuine open ports (a branched two-port then reads
        out directly via :meth:`~PerturbationResponse.acoustic_scattering_matrix`, with no
        multiport condensation).  Frozen terminals are not driven and are dropped from the
        measured terminal set.  Default ``()`` neutralizes every terminal (the
        boundary-independent multiport).  Caveat: a *lossless* frozen termination
        reintroduces real-axis resonances, so the response is ill-conditioned at those
        frequencies (add a small wall loss, or step off the exact pole).
    eps, eps_fb, u_floor : float, optional
        Operator-assembly regularizers forwarded to ``build_acoustic_blocks``.

    Returns
    -------
    PerturbationResponse
        The stacked independent perturbation fields.

    Raises
    ------
    ValueError
        If ``excite`` omits ``"acoustic"``/names an unknown family, entropy is requested
        with no inflow terminal to seat it, or a ``freeze`` entry names a missing node or
        a non-terminal node.
    """
    excite = tuple(excite)
    _validate_excite(excite, getattr(prob, "scalar_names", ()))
    frozen = _resolve_frozen(prob, freeze)
    ctx = _build_excitation_context(prob, x_bar, freqs, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor, frozen=frozen)

    # One driven excitation per (terminal, family): all acoustic first, then the incoming
    # entropy at each genuine-inflow seat -> canonical column order (f, g, h).
    excitations = [(t.node, "acoustic") for t in ctx.sel]
    if "entropy" in excite:
        entropy_nodes = [t.node for t in ctx.sel if _seats_entropy(t, ctx.est, ctx.u_floor)]
        if not entropy_nodes:
            raise ValueError("entropy excitation requested but no inflow terminal found to seat it")
        excitations += [(nd, "entropy") for nd in entropy_nodes]

    by_node = {t.node: t for t in ctx.sel}
    cols, forcing_kinds = [], []
    for nd, fam in excitations:  # the dedicated routine solves each one; the factorization is shared
        field = excite_perturbation(prob, x_bar, freqs, nd, modes=(fam,), _context=ctx)
        cols.append(field.X[:, 0, :])  # one driven wave per call -> (n_omega, n_col)
        forcing_kinds.append((fam, by_node[nd]))

    X = np.stack(cols, axis=1)  # (n_omega, n_force, n_col)
    cidx = _excited_char_indices({fam for _nd, fam in excitations})
    return PerturbationResponse(
        freqs=ctx.freqs,
        X=X,
        L=ctx.L,
        est=ctx.est,
        K=ctx.K,
        n_solve=ctx.n_solve,
        forcing=ctx.sel,
        forcing_kinds=forcing_kinds,
        cidx=cidx,
        terminals=ctx.terminals,
        node_names=tuple(getattr(prob, "node_names", ()) or ()),
        cals=ctx.cals,
        geometry=build_geometry(prob),
        frozen=tuple(sorted(ctx.frozen)),
    )


@dataclass
class PerturbationResponse:
    """Stored independent perturbation fields; extracts N x N matrices on demand."""

    freqs: np.ndarray  # (n_freq,) excitation frequencies in Hz
    X: np.ndarray  # (n_freq, n_force, n_col) -- one forced field per excitation
    L: List[np.ndarray]  # per-edge dx_to_char (3x3) at the mean state
    est: np.ndarray  # frozen mean edge-state table (for basis blocks / wave speeds)
    K: float  # cp / R
    n_solve: int
    forcing: List[Terminal]
    forcing_kinds: list
    cidx: tuple = (0, 1, 2)  # characteristic indices spanned by the driven waves
    terminals: Optional[List[Terminal]] = None  # the open (measured) terminals -- frozen ones excluded
    node_names: tuple = ()  # per-node element label (for plot labels); empty -> id only
    cals: Optional[list] = None  # per-edge caloric rows (reacting "network" flavor)
    geometry: Optional[NetworkGeometry] = None  # topology + duct lengths for spatial reconstruction
    frozen: tuple = ()  # terminal node ids kept at their physical BC (not measured)

    @property
    def n(self) -> int:
        """Matrix dimension: the number of driven perturbation waves (2 or 3)."""
        return len(self.cidx)

    @property
    def n_char(self) -> int:
        """Characteristic count per edge (3 for inert flow)."""
        return self.L[0].shape[0]

    def _term_label(self, node) -> str:
        """A terminal's ``id:name`` label (bare id when nameless), for the repr."""
        name = self.node_names[node] if node < len(self.node_names) else ""
        return f"{node}:{name}" if name else f"{node}"

    def __repr__(self) -> str:
        """One-line summary: forcing count, terminals driven, sweep extent, and any frozen-BC terminals."""
        f = np.asarray(self.freqs, dtype=float)
        nf = f.size
        span = "empty" if nf == 0 else (f"f = {f[0]:.1f} Hz" if nf == 1 else f"f in [{f.min():.1f}, {f.max():.1f}] Hz")
        n_term = len(self.terminals) if self.terminals else len({t.node for t in self.forcing})
        frozen = ""
        if self.frozen:
            frozen = ", BC frozen at " + ", ".join(self._term_label(n) for n in self.frozen)
        return (
            f"PerturbationResponse: {len(self.forcing)} forcing(s) on {n_term} terminal(s), "
            f"{nf} frequenc{'y' if nf == 1 else 'ies'} ({span}){frozen}"
        )

    def _waves(self, edge):
        """Characteristic amplitudes (f, g, h) at ``edge`` for every (omega, case).

        Returns an array of shape (n_omega, n_char, n_force): column k is the full
        wave vector of forced field k (all characteristics, driven or not).
        """
        ns = self.n_solve
        Xe = self.X[:, :, ns * edge : ns * edge + self.n_char]  # (n_omega, n_force, n_char)
        return np.einsum("ij,okj->oik", self.L[edge], Xe)  # (n_omega, n_char, n_force)

    _DIAGONAL_BASES = ("char", "riemann")  # flavors that do not mix characteristics

    @staticmethod
    def _seriality_residual(T, Wa, Wb):
        """Max-over-frequency relative residual ``||T Wa - Wb|| / ||Wb||``.

        Measures how badly a *single* fixed map ``T`` fails to reproduce ``b`` from ``a``
        across **all** forced excitations.  Near zero iff the wave state at ``a`` is sufficient
        to determine that at ``b`` -- i.e. the edges are in series.  A large value means they
        straddle an internal branch point, where no transfer matrix exists and
        ``T = Wb @ pinv(Wa)`` is only a least-squares best fit.
        """
        resid = np.linalg.norm(T @ Wa - Wb, axis=(1, 2))
        scale = np.linalg.norm(Wb, axis=(1, 2))
        rel = resid / np.where(scale > 0.0, scale, 1.0)
        return float(np.max(rel))

    def _underdetermined(self, n, n_force):
        """Whether the seriality residual is structurally blind for this response.

        The residual can only *test* seriality when the forcing over-determines the fit:
        more independent excitations than the matrix dimension, driven at **every** terminal.
        When ``n_force <= n`` the fit is exact by construction (residual ~ 0 regardless of
        topology); when a terminal is left undriven the test cannot see a branch dependence
        on it.  Either way only matters on a genuinely multi-terminal network (a single-path
        chain has no branch point to miss).
        """
        if not self.terminals or len(self.terminals) <= 2:
            return False  # single-path / 2-terminal: a transfer matrix always exists
        all_driven = {t.node for t in self.forcing} == {t.node for t in self.terminals}
        return n_force <= n or not all_driven

    def _warn_seriality(self, a, b, n, n_force, max_rel, tol=1e-6):
        """Warn -- never raise -- when the fitted ``T`` may not be a genuine transfer matrix.

        Two distinct failure modes, in order of certainty:

        * ``max_rel > tol`` -- a definitive branch straddle: no transfer matrix exists and the
          returned ``T`` is a least-squares best fit only.
        * otherwise, an under-determined multi-terminal response -- the residual is structurally
          ``~0`` and cannot confirm seriality, so the matrix is returned unverified.

        The caller always gets the best-fit matrix back; this only flags how much to trust it.
        """
        if max_rel > tol:
            warnings.warn(
                f"no transfer matrix exists between edges {a} and {b}: they straddle an internal "
                f"branch point (max relative residual {max_rel:.2e}), so the wave state at one edge "
                "is not sufficient to determine the other. The returned matrix is a least-squares best "
                "fit, not a physical transfer matrix; use multiport_scattering_matrix() for the rigorous "
                "descriptor, or source attribution to break down what reaches one edge from each terminal.",
                TransferMatrixWarning,
                stacklevel=3,
            )
        elif self._underdetermined(n, n_force):
            ndriven = len({t.node for t in self.forcing})
            warnings.warn(
                f"cannot verify a transfer matrix exists between edges {a} and {b}: the response is "
                f"under-determined ({n_force} forcing(s) for a {n}-wave matrix, {ndriven} of "
                f"{len(self.terminals)} terminals driven), so the seriality residual is structurally ~0 "
                "and cannot detect a branch point. Re-run perturbation_response with forcing=None (drive "
                "every terminal) to validate. The returned matrix is the best fit for this forcing.",
                TransferMatrixWarning,
                stacklevel=3,
            )

    def transfer_residual(self, a, b):
        """Seriality residual of the fitted transfer matrix between edges ``a`` and ``b``.

        The max-over-frequency relative residual of ``T = Wb @ pinv(Wa)`` (see
        :meth:`_seriality_residual`).  Near zero means a genuine transfer matrix exists (the
        edges are in series); a large value means they straddle an internal branch point and
        :meth:`transfer_matrix` returns only a least-squares best fit.  Computed in the
        characteristic basis, so it is independent of the ``basis`` the matrix is later
        expressed in.  Beware: when the response is under-determined
        (:meth:`_underdetermined`) this is structurally ``~0`` and does **not** certify
        seriality.

        Returns
        -------
        float
        """
        ci = list(self.cidx)
        Wa = self._waves(a)[:, ci, :]
        Wb = self._waves(b)[:, ci, :]
        return self._seriality_residual(Wb @ np.linalg.pinv(Wa), Wa, Wb)

    def transfer_matrix(self, a, b, basis="char"):
        """Transfer matrix ``T_ba`` mapping the driven waves at ``a`` to those at ``b``.

        The dimension is ``self.n`` (2 for the default acoustic excitation, 3 with
        entropy).  Read along each edge's own arrow; ``basis`` selects the variable
        flavor (``characteristics.BASIS_LABELS``).  Shape ``(n_omega, n, n)``.

        Always returns a matrix.  If ``a`` and ``b`` lie on opposite sides of an internal
        branch point no transfer matrix exists, so the returned ``T`` is only the
        least-squares best fit and a :class:`TransferMatrixWarning` is emitted -- inspect
        :meth:`transfer_residual` and prefer :meth:`multiport_scattering_matrix`.  A second
        warning flavor fires when the response is too under-determined to tell either way.

        Warns
        -----
        TransferMatrixWarning
            If ``a`` and ``b`` straddle an internal branch point, or the response is
            under-determined so seriality cannot be verified.
        """
        ci = list(self.cidx)
        Wa = self._waves(a)[:, ci, :]  # (n_omega, n, n_force) over driven characteristics
        Wb = self._waves(b)[:, ci, :]
        T = Wb @ np.linalg.pinv(Wa)  # pinv: >= n_force forcings (= n for a 2-terminal net)
        self._warn_seriality(a, b, self.n, Wa.shape[2], self._seriality_residual(T, Wa, Wb))
        if basis == "char":
            return T
        if self.n < self.n_char and basis not in self._DIAGONAL_BASES:
            raise ValueError(
                f"flavor {basis!r} mixes characteristics, so it needs the full response; "
                f"re-run with excite=('acoustic', 'entropy', ...) or use 'char'/'riemann'"
            )
        cal_a = None if self.cals is None else self.cals[a]
        cal_b = None if self.cals is None else self.cals[b]
        Ba = basis_block_from_state(basis, self.est[:, a], self.K, cal_a)[np.ix_(ci, ci)]
        Bb = basis_block_from_state(basis, self.est[:, b], self.K, cal_b)[np.ix_(ci, ci)]
        return mat.tm_in_basis(T, Ba, Bb)

    def scattering_matrix(self, a, b, basis="char"):
        """Scattering matrix between stations ``a`` (upstream) and ``b``.

        Dimension is ``self.n`` (2 acoustic-only, 3 with entropy).  Incoming waves
        (``a``'s downstream-running + ``b``'s upstream-running) map to the outgoing
        ones; ordering follows ``matrices.scattering_labels``.  ``basis`` may only be
        a flavor diagonal in the characteristics (``char`` or ``riemann``); mixed
        flavors are undefined for a scattering matrix.

        Warns
        -----
        TransferMatrixWarning
            Via :meth:`transfer_matrix`, if ``a`` and ``b`` straddle an internal branch
            point or the response is under-determined (see :meth:`transfer_residual`).
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
        cal = None if self.cals is None else self.cals[e]
        B = basis_block_from_state(basis, self.est[:, e], self.K, cal)
        return B[i, i]

    def scattering_labels(self, a, b):
        """Ordered (station, char-index) tags of the SM's incoming and outgoing waves."""
        ua, ca = float(self.est[ES_U, a]), float(self.est[ES_C, a])
        ub, cb = float(self.est[ES_U, b]), float(self.est[ES_C, b])
        return mat.scattering_labels(ua, ca, ub, cb, self.n)

    # -- multiport (whole-network terminal) scattering matrix ---------------

    def _multiport_io(self):
        """Ordered ``(node, edge, char)`` tags of the multiport incoming and outgoing waves.

        Incoming follows the driver's excitation order (the matrix columns); outgoing is
        terminal-major from :func:`matrices.multiport_partition`, restricted to the driven
        characteristic set ``cidx`` (the matrix rows).
        """
        if not self.terminals:
            raise ValueError("multiport scattering needs the full terminal set; build via perturbation_response")
        driven_nodes = {t.node for t in self.forcing}
        all_nodes = {t.node for t in self.terminals}
        if driven_nodes != all_nodes:
            raise ValueError(
                "multiport scattering describes the whole network, so every terminal must be driven. "
                f"This response drove only {sorted(driven_nodes)} of {sorted(all_nodes)} -- re-run "
                "perturbation_response with forcing=None (the default) to drive them all."
            )
        ci = set(self.cidx)
        incoming = [(t.node, t.edge, (t.incoming if fam == "acoustic" else 2)) for fam, t in self.forcing_kinds]
        stations = [
            (float(self.est[ES_U, t.edge]), float(self.est[ES_C, t.edge]), "a" if t.at_tail else "b")
            for t in self.terminals
        ]
        _inc, out = mat.multiport_partition(stations, self.n_char)
        outgoing = [(self.terminals[k].node, self.terminals[k].edge, ch) for (k, ch) in out if ch in ci]
        return incoming, outgoing

    def multiport_scattering_matrix(self):
        """Whole-network scattering matrix mapping every terminal's incoming wave to every outgoing one.

        The rigorous boundary-independent descriptor of a network with more than two
        terminals (where pairwise edge transfer matrices across a branch do not exist).
        Generally **rectangular**: with entropy the incoming set is ``#terminals +
        #inflow-terminals`` and the outgoing set ``#terminals + #outflow-terminals``
        (acoustic-only it is the square ``N x N``, ``N = #terminals``).  Columns follow the
        excitation order, rows are terminal-major; both are tagged by
        :meth:`multiport_scattering_labels`.

        Returns
        -------
        ndarray
            Shape ``(n_omega, n_outgoing, n_incoming)``.

        Raises
        ------
        ValueError
            If the network was not driven at every terminal (rebuild with ``forcing=None``).
        """
        incoming, outgoing = self._multiport_io()
        S = np.zeros((self.freqs.size, len(outgoing), len(incoming)), dtype=np.complex128)
        for r, (_node, edge, ch) in enumerate(outgoing):
            S[:, r, :] = self._waves(edge)[:, ch, :]  # outgoing amplitude per driven (unit-incoming) case
        return S

    def _node_tag(self, node):
        """LaTeX subscript for a terminal node: its id, plus its element name when known.

        Edges are referred to by id alone (edge names are not meaningful), but node names are
        unique and meaningful, so a terminal reads ``0:\\text{MassFlowInlet1}`` -- the id (for
        cross-referencing ``forcing``/code) and the label (for meaning).  Falls back to the bare
        id when the problem carries no names.  The name rides a ``\\text{}`` group so it renders
        upright and an arbitrary label cannot break the MathJax string.
        """
        name = self.node_names[node] if node < len(self.node_names) else ""
        return f"{node}:\\text{{{_tex_text(name)}}}" if name else f"{node}"

    def _wave_at_node(self, char, node):
        """LaTeX wave-symbol fragment for characteristic ``char`` at terminal ``node``.

        e.g. ``f_{0:\\text{inlet}}`` -- a fragment (no ``$``); the plotting layer wraps it.
        """
        return f"{_CHAR_SYM[char]}_{{{self._node_tag(node)}}}"

    def multiport_scattering_labels(self):
        """Per-wave symbols for the multiport columns (incoming) and rows (outgoing).

        Each wave is its characteristic symbol (``f``/``g``/``h``) subscripted by the **node id**
        of the terminal it lives on (e.g. ``f<sub>0</sub>``).  The bare id keeps the matrix labels
        uncrowded -- a multiport entry reads ``f₀ → g₀`` -- while still uniquely identifying the
        terminal (cross-reference :attr:`node_names` for the element name).
        """
        incoming, outgoing = self._multiport_io()

        def sym(node, _edge, ch):
            return f"{_CHAR_SYM[ch]}_{{{node}}}"

        return [sym(*w) for w in incoming], [sym(*w) for w in outgoing]

    # -- source attribution (where the wave at an edge comes from) -----------

    def _source_char(self, fam, t):
        """The characteristic index a source drives: a terminal's incoming acoustic wave, or entropy."""
        return t.incoming if fam == "acoustic" else 2

    def contributions(self, edge, *, incoming=None):
        """Break the wave at ``edge`` into the contribution of each terminal's incoming wave.

        Every driven excitation is a unit incoming wave at one terminal with all others zero, so
        by linearity the perturbation field at ``edge`` is the **exact superposition** of one
        contribution per source.  This is the physically honest "where does what I see at this
        edge come from" decomposition: each term is a genuine one-way path gain (a multiport
        scattering entry generalized to an internal edge), free of the common-driver confounding
        that makes a least-squares transfer matrix overstate one edge's influence on another (the
        residual measures *predictability*, this measures *contribution*).

        Parameters
        ----------
        edge : int
            Edge whose wave is decomposed.
        incoming : array_like of complex, optional
            Complex amplitude assigned to each *source*, as a 1-D array with one entry per source
            in the order of :meth:`contribution_labels` (the excitation-column order).  A source is
            the incoming characteristic wave entering at one terminal -- ``f`` driven into the
            inlet, ``g`` driven into an outlet, etc. -- expressed in characteristic units.  Use it
            to set the operating scenario you care about: e.g. ``[1, 0, 0]`` for unit forcing at
            the first terminal and silence elsewhere, or the actual/measured incoming amplitudes
            at each terminal.  The contribution of source ``k`` is then its unit response scaled by
            ``incoming[k]``, and the columns sum to the total field for that scenario.  Default
            (``None``): unit amplitude on every source, so each column is the bare per-source
            transfer function (gain) and the columns are directly comparable.

        Returns
        -------
        ndarray
            Shape ``(n_omega, n, n_source)``: entry ``[:, c, k]`` is the amplitude of driven
            characteristic ``c`` (see :attr:`cidx`) at ``edge`` produced by source ``k``.

        Raises
        ------
        ValueError
            If ``edge`` is out of range or ``incoming`` has the wrong length.
        """
        if not 0 <= edge < len(self.L):
            raise ValueError(f"edge {edge} out of range [0, {len(self.L)})")
        W = self._waves(edge)[:, list(self.cidx), :]  # (n_omega, n, n_source)
        if incoming is not None:
            w = np.asarray(incoming, dtype=np.complex128)
            if w.shape != (W.shape[2],):
                raise ValueError(f"incoming must give one amplitude per source ({W.shape[2]}); got {w.shape}")
            W = W * w[None, None, :]
        return W

    def contribution_labels(self, edge):
        """Labels for :meth:`contributions` at ``edge``: ``(output_labels, source_labels)``.

        ``output_labels`` are the driven characteristics at ``edge`` (rows, ``f``/``g``/``h``
        subscripted by the edge **id** -- edge names are not meaningful); ``source_labels`` are the
        incoming wave each excitation drives, subscripted by its terminal node **id and name**
        (columns), so a plotted entry reads ``source → output`` (e.g. ``g₇:Outlet1 → f₅``: the
        incoming wave at terminal Outlet1 contributing to ``f`` at edge 5).
        """
        outputs = [f"{_CHAR_SYM[c]}_{{{edge}}}" for c in self.cidx]
        sources = [self._wave_at_node(self._source_char(fam, t), t.node) for fam, t in self.forcing_kinds]
        return outputs, sources

    # -- notebook plotting (edge-aware labels) ------------------------------

    def _basis_labels(self, basis):
        """Per-variable symbols for ``basis``, trimmed to this response's dimension."""
        from .characteristics import BASIS_LABELS

        syms = BASIS_LABELS.get(basis)
        return tuple(syms[: self.n]) if syms else None

    def _residual_title(self, a, b, title):
        """Append the seriality residual to ``title`` so it is visible on the plot.

        A genuine in-series matrix reads ``residual ~ 0``; a branch straddle shows the
        large residual that makes the matrix a best-fit only.  ``(under-determined)`` flags
        the case where the residual cannot be trusted (see :meth:`_underdetermined`).
        """
        r = self.transfer_residual(a, b)
        note = f"max residual = {r:.1e}"
        if self._underdetermined(self.n, self.X.shape[1]):
            note += " (under-determined)"
        return note if title is None else f"{title} — {note}"

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
            x-axis values (default: ``self.freqs``, in Hz).
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
        x = self.freqs if freqs is None else freqs
        title = kwargs.pop("title", None) or f"Transfer matrix: edge {a} → edge {b}"
        title = self._residual_title(a, b, title)
        return _plot(T, x, labels=self._basis_labels(basis), edges=(a, b), title=title, **kwargs)

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
            x-axis values (default: ``self.freqs``, in Hz).
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
        x = self.freqs if freqs is None else freqs
        title = kwargs.pop("title", None) or f"Scattering matrix: edges {a}, {b}"
        title = self._residual_title(a, b, title)
        return _plot(
            S,
            x,
            labels=self._basis_labels(basis),
            edges=(a, b),
            partition=self.scattering_labels(a, b),
            title=title,
            **kwargs,
        )

    def plot_multiport_scattering_matrix(self, freqs=None, **kwargs):
        """Plot the whole-network multiport scattering matrix with terminal-tagged labels.

        Wraps :meth:`multiport_scattering_matrix` and labels every entry by its own
        terminal-subscripted waves (e.g. ``f₀ → g₀`` for the inlet reflection,
        ``f₀ → f₇`` for transmission to terminal 7).

        Parameters
        ----------
        freqs : array_like, optional
            x-axis values (default: ``self.freqs``, in Hz).
        **kwargs
            Forwarded to :func:`fns.plotting.plot_scattering_matrix`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_scattering_matrix as _plot

        S = self.multiport_scattering_matrix()
        incoming, outgoing = self.multiport_scattering_labels()
        x = self.freqs if freqs is None else freqs
        kwargs.setdefault("title", "Multiport scattering matrix")
        return _plot(S, x, row_labels=outgoing, col_labels=incoming, **kwargs)

    def plot_contributions(self, edge, freqs=None, *, incoming=None, normalize="auto", **kwargs):
        """Plot the source attribution of the wave at ``edge``: one panel per output wave.

        The honest "where does what I see at this edge come from" view.  There is **one panel per
        driven characteristic** at ``edge`` (``f``/``g``/``h``), and within each panel **one curve
        per source** (each terminal's incoming wave), magnitude over phase, so the sources can be
        compared directly against one another for that output wave -- which dominates, where they
        cross over.  Unlike a transfer matrix this is well defined across branch points: it
        decomposes a contribution rather than asserting one edge determines another.

        Parameters
        ----------
        edge : int
            Edge whose wave is decomposed.
        freqs : array_like, optional
            x-axis values (default: ``self.freqs``, in Hz).
        incoming : array_like, optional
            Per-source incoming amplitudes for a specific scenario (see :meth:`contributions`);
            default is unit amplitude on every source.
        normalize : {"auto", True, False}, optional
            Scale each panel by its **dominant source's peak magnitude** (a scalar -- never by the
            anchor's frequency curve, which would blow up at its nulls), so the leading source
            peaks at ``1`` and the rest read as honest fractions of it.  ``"auto"`` (default)
            normalizes only when ``incoming`` is ``None`` (per-source gains have no absolute scale),
            and shows absolute magnitudes once an ``incoming`` scenario fixes the amplitudes.
        **kwargs
            Forwarded to :func:`fns.plotting.plot_complex_matrix`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_complex_matrix as _plot

        C = self.contributions(edge, incoming=incoming)  # (n_omega, n, n_source)
        outputs, sources = self.contribution_labels(edge)
        if normalize == "auto":
            normalize = incoming is None
        suffix = ""
        if normalize:  # scalar per-component anchor: divide each output wave by its peak contribution
            peak = np.abs(C).max(axis=(0, 2), keepdims=True)
            C = C / np.where(peak > 0.0, peak, 1.0)
            kwargs.setdefault("mag_range", (0.0, 1.05))
            suffix = " (normalized to the dominant source per panel)"
        x = self.freqs if freqs is None else freqs
        title = kwargs.pop("title", None) or f"Edge {edge}: wave contribution by source{suffix}"
        # one overlaid series per source -> a curve per source in each output-wave panel;
        # source labels are LaTeX fragments, rendered through the global LaTeX toggle
        from ..plotting import mathify

        legend = [mathify(s) for s in sources]
        mats = [C[:, :, k, None] for k in range(C.shape[2])]
        return _plot(mats, x, names=legend, row_labels=outputs, col_labels=[""], title=title, **kwargs)

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
        if len(cols) < 2:
            raise ValueError(f"expected >= 2 acoustic forcings, found {len(cols)}")
        Wa = self._waves(a)[:, :2, :][:, :, cols]  # (n_omega, 2, n_acoustic)
        Wb = self._waves(b)[:, :2, :][:, :, cols]
        T = Wb @ np.linalg.pinv(Wa)  # pinv: >= 2 acoustic forcings on a multi-terminal net
        self._warn_seriality(a, b, 2, Wa.shape[2], self._seriality_residual(T, Wa, Wb))
        return T

    def acoustic_scattering_matrix(self, a, b):
        """2x2 acoustic scattering matrix, incoming ``(f_a, g_b)`` -> outgoing ``(g_a, f_b)``.

        Ordering matches the full :meth:`scattering_matrix` (reflection at ``a``
        first, transmission to ``b`` second); ``tm_fg_to_sm2`` yields the classic
        ``(f_b, g_a)`` ordering, so the two rows are swapped.
        """
        return mat.tm_fg_to_sm2(self.acoustic_transfer_matrix(a, b))[:, ::-1, :]

    # -- spatial field reconstruction (mode-shape animation) ----------------

    def field_along_network(self, freq, *, incoming=None, variable="p", spec=None, root=None, n_x=160):
        """Reconstruct the forced spatial field at one frequency along every root->leaf path.

        Picks the stored frequency nearest ``freq`` and superposes the driven sources by
        ``incoming``, then reconstructs the continuous perturbation field inside every duct
        (theory.md s12.3); see :func:`fns.perturbation.modeshape.reconstruct_field`.

        Parameters
        ----------
        freq : float
            Target frequency (Hz); the nearest value in :attr:`freqs` is used.
        incoming : array_like of complex, optional
            One amplitude per driven source (column of :attr:`X`), in excitation order.
            Default: unit amplitude on the first source, zero on the rest.
        variable : str, optional
            Plotted quantity (``"p"``, ``"u"``, ``"rho"``, ``"mdot"``, ``"f"``, ``"g"``,
            ``"h"``); default ``"p"``.  Ignored when ``spec`` is given.
        spec : tuple, optional
            A ``(basis_flavor, component)`` pair (e.g. from
            :func:`fns.perturbation.modeshape.resolve_specs`) selecting any basis
            component directly; overrides ``variable``.
        root : int, optional
            Developed-length origin element (default: a mean-flow inlet).
        n_x : int, optional
            Interior samples per duct (default 160).

        Returns
        -------
        list of fns.perturbation.modeshape.PathField

        Raises
        ------
        ValueError
            If the result carries no geometry, or ``incoming`` has the wrong length.
        """
        if self.geometry is None:
            raise ValueError("no network geometry stored; rebuild via perturbation_response() for spatial fields")
        n_force = self.X.shape[1]
        if incoming is None:
            w = np.zeros(n_force, dtype=np.complex128)
            w[0] = 1.0
        else:
            w = np.asarray(incoming, dtype=np.complex128)
            if w.shape != (n_force,):
                raise ValueError(f"incoming must give one amplitude per source ({n_force}); got {w.shape}")
        fi = int(np.argmin(np.abs(self.freqs - float(freq))))
        omega = 2.0 * np.pi * float(self.freqs[fi])
        return reconstruct_field(
            self.geometry,
            lambda e: self._waves(e)[fi] @ w,
            self.est,
            self.K,
            omega,
            variable=variable,
            spec=spec,
            root=root,
            n_x=n_x,
            cals=self.cals,
        )

    def intensity_along_network(self, freq, *, incoming=None, energy_density=False, root=None, n_x=160):
        """Acoustic intensity (or energy density) along the developed length, at one frequency.

        The spatial companion of :meth:`field_along_network`: instead of a primitive
        variable it reconstructs the **Myers acoustic intensity** ``I(x)`` [W/m^2]
        (downstream positive) -- or the energy density ``e(x)`` [J/m^3] when
        ``energy_density=True`` -- along every root->leaf path, so one can read where
        acoustic power flows and where it is produced or absorbed.

        Parameters
        ----------
        freq : float
            Target frequency (Hz); the nearest value in :attr:`freqs` is used.
        incoming : array_like of complex, optional
            One amplitude per driven source (column of :attr:`X`); default unit on the
            first source.
        energy_density : bool, optional
            Return the energy density instead of the intensity (default ``False``).
        root : int, optional
            Developed-length origin element (default: a mean-flow inlet).
        n_x : int, optional
            Interior samples per duct (default 160).

        Returns
        -------
        list of fns.perturbation.modeshape.PathField
            ``values`` is the real intensity (or energy density).
        """
        from .power import intensity_along_network as _intensity

        if self.geometry is None:
            raise ValueError("no network geometry stored; rebuild via perturbation_response() for spatial fields")
        n_force = self.X.shape[1]
        if incoming is None:
            w = np.zeros(n_force, dtype=np.complex128)
            w[0] = 1.0
        else:
            w = np.asarray(incoming, dtype=np.complex128)
            if w.shape != (n_force,):
                raise ValueError(f"incoming must give one amplitude per source ({n_force}); got {w.shape}")
        fi = int(np.argmin(np.abs(self.freqs - float(freq))))
        omega = 2.0 * np.pi * float(self.freqs[fi])
        return _intensity(
            self.geometry,
            lambda e: self._waves(e)[fi] @ w,
            self.est,
            omega,
            energy_density=energy_density,
            root=root,
            n_x=n_x,
        )

    def animate_field(
        self,
        freq,
        *,
        incoming=None,
        variable="p",
        basis=None,
        root=None,
        n_x=160,
        n_frames=60,
        normalize=True,
        envelope=True,
        **layout,
    ):
        """Animate the forced spatial field at one frequency over a phase cycle (slider + play).

        Pass a list of ``variable`` names, or a ``basis`` flavor (which expands to its three
        components), to overlay several quantities; all share the single forcing frequency,
        so they are phase-locked.

        Parameters
        ----------
        freq : float
            Target frequency (Hz); the nearest value in :attr:`freqs` is used.
        incoming : array_like of complex, optional
            Per-source amplitudes (see :meth:`field_along_network`); default unit on the
            first source.
        variable : str or sequence of str, optional
            Plotted quantity, or several to overlay (see :meth:`field_along_network`);
            default ``"p"``.  Ignored when ``basis`` is given.
        basis : str, optional
            A flavor from :data:`fns.perturbation.characteristics.BASIS_LABELS`; overlays its
            three components and overrides ``variable``.
        root : int, optional
            Developed-length origin element (default: a mean-flow inlet).
        n_x, n_frames : int, optional
            Interior samples per duct (default 160) and phase frames per cycle (default 60).
        normalize : bool, optional
            Scale each overlaid quantity's peak magnitude to 1 (default True).
        envelope : bool, optional
            Shade the ``+/- |psi(x)|`` span behind each animated line (default True).
        **layout
            Forwarded to ``Figure.update_layout``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import animate_mode_shape as _animate, AnimSeries, mathify
        from .modeshape import resolve_specs

        fi = int(np.argmin(np.abs(self.freqs - float(freq))))
        used = float(self.freqs[fi])
        specs = resolve_specs(variable, basis)
        multi_quantity = len(specs) > 1

        series = []
        for label, flavor, comp in specs:
            fields = self.field_along_network(freq, incoming=incoming, spec=(flavor, comp), root=root, n_x=n_x)
            series.append(AnimSeries(path_fields=fields, label=label if multi_quantity else ""))

        norm_note = ", normalized" if normalize else ""
        if len(specs) == 1:
            y_title = f"{mathify(specs[0][0])}  (Re{norm_note})"
        elif basis is not None:
            y_title = f"{basis} basis  (Re{norm_note})"
        else:
            y_title = f"amplitude  (Re{norm_note})"

        return _animate(
            series,
            y_title=y_title,
            title=layout.pop("title", None) or f"Forced response: f = {used:.4g} Hz",
            n_frames=n_frames,
            normalize=normalize,
            envelope=envelope,
            **layout,
        )


# -- back-compatibility aliases (pre-reframe names) -------------------------

acoustic_response = perturbation_response
AcousticResponse = PerturbationResponse
