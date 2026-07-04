"""Identify an element's dynamic response from a measured network transfer matrix.

Given a converged network model in which one element is marked *unknown* and a transfer
matrix ``M_meas`` measured between two of its edges, recover the unknown -- the element's
2-port transfer matrix (:func:`identify_transfer_matrix`) or the transfer function(s) of a
flame / mass-source feedback (:func:`identify_transfer_function`).

Method (theory.md s12.7)
------------------------
Both unknowns enter the perturbation operator **linearly**, as a low-rank update of the
known, passive operator ``A0(omega)``::

    source  :  A(G)  = A0 + P diag(G) Q^T      (a flame's S(omega) feedback, rank <= K terms)
    2-port  :  A(X)  = A0 - P (X - I) Q^T       (a transfer-matrix element, rank <= N)

With the terminals neutralized into ports (the measurement driver's operator), the network
response is ``x(u) = A(u)^{-1} F`` for a set of independent excitations ``F``.  Projecting
onto the two measured edges gives the port wave matrices ``W_a(u), W_b(u)``, and the measured
transfer matrix imposes ``W_b(u) = M_meas W_a(u)``.  Substituting the Woodbury form of
``A(u)^{-1}`` turns this into a **linear** system per frequency for the unknown's modal
coordinates, solved by least squares -- so one factorization of ``A0`` recovers the unknown,
for a cascade or a branched network alike.  The system's conditioning is the identifiability
diagnostic: it is rank-deficient exactly when the measurement cannot separate the unknowns
(e.g. collinear reference fluctuations for a multi-input flame).

The recovered response is returned as a real-frequency table plus, by default, its rational
continuation (:class:`~nefes.perturbation.continuation.RationalFit`) so it drops straight back
into the element / dynamic source and the stability eigensolver.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List

import numpy as np

from ..matrix import TransferMatrix
from ..operator.characteristics import edge_caloric
from ..operator.stamps import build_source_stamps, build_tm_stamps
from ..response.response import (
    _build_excitation_context,
    _driven_prescriptions,
    _excited_char_indices,
    _seats_entropy,
    _N_FIXED_CHAR,
)
from ...elements.dynamic_source import DynamicSource, DynamicResponseTerm, Constant, TransferFunction, Tabulated
from ..continuation import RationalFit

# ==========================================================================
# results
# ==========================================================================


@dataclass
class TransferMatrixIdentification:
    """Result of :func:`identify_transfer_matrix`."""

    transfer_matrix: TransferMatrix  # recovered 2-port (characteristic basis)
    freqs: np.ndarray  # identification frequencies [Hz]
    conditioning: np.ndarray  # per-frequency condition number of the de-embed system
    node: int

    def __repr__(self):
        return f"TransferMatrixIdentification(node={self.node}, N={self.transfer_matrix.n}, {self.freqs.size} pts)"


@dataclass
class TransferFunctionIdentification:
    """Result of :func:`identify_transfer_function`."""

    transfer_functions: List[TransferFunction]  # one per unknown term (continued if requested)
    values: np.ndarray  # (K, n_f) raw recovered samples
    terms: List[tuple]  # (ref_edge, quantity, gain) per term
    freqs: np.ndarray
    conditioning: np.ndarray  # per-frequency condition number of the linear system
    residual: np.ndarray  # per-frequency relative least-squares residual (consistency)
    node: int

    def __repr__(self):
        labels = ", ".join(f"{q}@e{e}" for e, q, _g in self.terms)
        return f"TransferFunctionIdentification(node={self.node}, terms=[{labels}], {self.freqs.size} pts)"


# ==========================================================================
# shared setup
# ==========================================================================


def _extraction(L_edge, edge, ci, ns, n_col):
    """Row operator ``E`` (len(ci) x n_col) so ``E @ x`` are the characteristics ``ci`` at ``edge``.

    Mirrors ``response._waves``: the acoustic+entropy block is the 3x3 ``dx_to_char`` map on the
    edge's first three solve columns; a transported scalar passes through on its own column.
    """
    E = np.zeros((len(ci), n_col), dtype=np.complex128)
    for r, c in enumerate(ci):
        if c < _N_FIXED_CHAR:
            E[r, ns * edge : ns * edge + _N_FIXED_CHAR] = L_edge[c, :]
        else:
            E[r, ns * edge + c] = 1.0
    return E


def _forcing(ctx, excite):
    """Independent excitations ``F`` (n_col x K_force): a unit incoming wave per driven port wave.

    Mirrors ``perturbation_response``: acoustic at every driven terminal, then each convected
    family (entropy, scalars) only at its genuine-inflow seats -- so the column set matches the
    transfer matrix's forcings exactly.
    """
    excitations = [(t.node, "acoustic") for t in ctx.sel]
    inflow_seats = [t.node for t in ctx.sel if _seats_entropy(t, ctx.est, ctx.u_floor)]
    for fam in ("entropy",) + tuple(ctx.scalar_names):
        if fam in excite:
            excitations += [(nd, fam) for nd in inflow_seats]
    driven = [p for nd, fam in excitations for p in _driven_prescriptions(ctx, nd, (fam,))]
    F = np.zeros((ctx.n_col, len(driven)), dtype=np.complex128)
    for k, p in enumerate(driven):
        F[p.row, k] = 1.0
    return F


def _measured_char(measured, freqs):
    """The measured transfer matrix in the characteristic basis, sampled at ``freqs`` (n_f, N, N)."""
    m = measured if measured.basis == "char" else measured.to_basis("char")
    return np.asarray(m(freqs), dtype=np.complex128)


def _with_node(descriptors, n_nodes, node, value):
    """A length-``n_nodes`` tuple copy of ``descriptors`` with entry ``node`` set to ``value``."""
    out = list(descriptors) if descriptors else [None] * n_nodes
    if len(out) < n_nodes:
        out += [None] * (n_nodes - len(out))
    out[node] = value
    return tuple(out)


# ==========================================================================
# transfer-matrix (blackbox 2-port) identification
# ==========================================================================


def identify_transfer_matrix(
    prob,
    x_bar,
    measured,
    *,
    node,
    a,
    b,
    excite=None,
    isentropic=False,
    forcing=None,
    freeze=(),
    continue_=True,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    **fit_kwargs,
):
    """Recover the 2-port transfer matrix of any interior element from a network measurement.

    Works for a marked :func:`~nefes.elements.catalog.transfer_matrix_element` **and** for a
    genuine element whose acoustic 2-port you want -- e.g. an :func:`~nefes.elements.catalog.equilibrium_flame`.
    The element keeps its own mean-flow kernel (heat addition, dilatation, area change); only
    its perturbation acoustic rows are treated as the unknown block, so the recovered matrix is
    the element's full linear 2-port (a flame's active response folds into it).

    Parameters
    ----------
    prob : CompiledProblem
        The network.  ``node`` is any interior 2-port element; any dynamic-source feedback it
        carries is folded into the recovered matrix (the identification runs it silent).
    x_bar : ndarray
        Converged mean-flow state.
    measured : TransferMatrix
        Measured transfer matrix ``w_b = M_meas w_a`` between edges ``a`` and ``b``; its grid
        sets the identification frequencies and its dimension ``N`` the matrix recovered.
    node : int
        The element's node id.
    a, b : int
        Edge ids the measured matrix spans (``a`` upstream of the element, ``b`` downstream).
    excite : sequence of str, optional
        Driven wave families (default: ``("acoustic", "entropy")`` for ``N=3``, else
        ``("acoustic",)``).
    isentropic : bool, optional
        Pin entropy to zero everywhere -- the **acoustics-only** identification (``N=2``).  Use
        this when the measured matrix is a purely acoustic 2-port and entropy generation should
        not contaminate the recovery.  The measured matrix must have been taken (or synthesized)
        under the same assumption.  Default False (full acoustic+entropy).
    forcing, freeze : optional
        Passed to the measurement driver (which terminals to drive / keep physical).
    continue_ : bool, optional
        Also fit a rational continuation of the recovered matrix (default True).
    **fit_kwargs
        Forwarded to :meth:`TransferMatrix.continue_` (e.g. ``rtol``, ``delay``).

    Returns
    -------
    TransferMatrixIdentification
    """
    freqs = np.asarray(measured.freqs, dtype=float)
    N = measured.n
    if excite is None:
        excite = ("acoustic", "entropy") if N == 3 else ("acoustic",)
    if isentropic and N != 2:
        raise ValueError("isentropic (acoustics-only) identification recovers a 2x2 matrix; measured must be N=2")

    # A0 := the operator with the element's acoustics set to a reference (identity) transfer
    # matrix, its own mean-flow kernel intact.  Any active feedback on the node is silenced so
    # it folds into the recovered 2-port rather than being double-counted.
    ident = TransferMatrix(freqs, np.broadcast_to(np.eye(N), (freqs.size, N, N)).copy(), basis="char")
    ntm = _with_node(prob.node_transfer_matrix, prob.n_nodes, node, ident)
    nds = _with_node(prob.node_dynamic_source, prob.n_nodes, node, None)
    prob0 = dataclasses.replace(prob, node_transfer_matrix=ntm, node_dynamic_source=nds)

    ctx = _build_excitation_context(
        prob0, x_bar, freqs, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor, frozen=freeze, isentropic=isentropic
    )
    cals = edge_caloric(prob0, x_bar)
    stamps = build_tm_stamps(prob0, x_bar, ctx.K, u_floor, cals)
    st = next((s for s in stamps if s.node == node), None)
    if st is None:
        raise ValueError(f"node {node} is not an interior 2-port element whose transfer matrix can be identified")
    ci = _excited_char_indices(excite, ctx.scalar_names)
    if len(ci) != N:
        raise ValueError(f"excite spans {len(ci)} characteristics but the matrix is N={N}; adjust `excite`")

    F = _forcing(ctx, excite)
    P, Q = _tm_pq(st, N, ctx.n_solve, ctx.n_col)
    Ea = _extraction(ctx.L[a], a, ci, ctx.n_solve, ctx.n_col)
    Eb = _extraction(ctx.L[b], b, ci, ctx.n_solve, ctx.n_col)
    M = _measured_char(measured, freqs)

    Xrec = np.empty((freqs.size, N, N), dtype=np.complex128)
    cond = np.empty(freqs.size)
    for i, lu in enumerate(ctx.lus):
        Xrec[i], cond[i] = _deembed_tm(lu, F, P, Q, Ea, Eb, M[i])
    # attach the element's own face states so the result converts flavor / form freely; the
    # recovered matrix relates the element edges (arrow port 0 = e_up -> port 1 = e_down).
    ports = (_port_state(ctx.est, st.e_up), _port_state(ctx.est, st.e_down))
    tm = TransferMatrix(freqs, Xrec, basis="char", ports=ports, K=ctx.K)
    if continue_:
        tm = tm.continue_(**fit_kwargs)
    return TransferMatrixIdentification(transfer_matrix=tm, freqs=freqs, conditioning=cond, node=node)


def _port_state(est, edge):
    """A :class:`~nefes.perturbation.matrix.PortState` from the mean edge-state table."""
    from ..matrix import PortState
    from ...assembly.recover import ES_RHO, ES_C, ES_U, ES_P, ES_AREA

    return PortState(
        float(est[ES_RHO, edge]),
        float(est[ES_C, edge]),
        float(est[ES_U, edge]),
        float(est[ES_P, edge]),
        float(est[ES_AREA, edge]),
    )


def _tm_pq(st, N, ns, n_col):
    """Low-rank factors of the transfer-matrix update ``A(X) = A0 - P (X - I) Q^T``."""
    P = np.zeros((n_col, N), dtype=np.complex128)
    Q = np.zeros((n_col, N), dtype=np.complex128)
    for i in range(N):
        P[st.rows[i], i] = 1.0
    for j in range(N):
        for k, c in enumerate(st.up_cols):
            Q[c, j] = st.L_up[j, k]
    return P, Q


def _deembed_tm(lu, F, P, Q, Ea, Eb, M):
    """Recover ``X`` at one frequency from ``W_b = M W_a`` (Woodbury; ``A0`` carries ``X=I``)."""
    X0 = lu.solve(F)  # (n_col, Kf)
    Pg = lu.solve(P)  # (n_col, N)
    C = Q.T @ Pg  # (N, N)
    alpha = Q.T @ X0  # (N, Kf)
    Ga, Gb = Ea @ Pg, Eb @ Pg  # (N, N)
    Wa0, Wb0 = Ea @ X0, Eb @ X0  # (N, Kf)
    # W_a(D) = Wa0 + Ga m, W_b(D) = Wb0 + Gb m, with m = (D^{-1} - C)^{-1} alpha, D = X - I.
    Kc = Gb - M @ Ga  # (N, N)
    rhs = M @ Wa0 - Wb0  # (N, Kf)
    m = np.linalg.lstsq(Kc, rhs, rcond=None)[0]  # (N, Kf)
    Dinv = C + alpha @ np.linalg.pinv(m)  # D^{-1} = C + alpha m^+
    D = np.linalg.inv(Dinv)
    X = D + np.eye(D.shape[0])
    return X, np.linalg.cond(Kc)


# ==========================================================================
# transfer-function (flame / mass-source feedback) identification
# ==========================================================================


def identify_transfer_function(
    prob,
    x_bar,
    measured,
    *,
    node,
    a,
    b,
    excite=("acoustic", "entropy"),
    isentropic=False,
    forcing=None,
    freeze=(),
    continue_=True,
    eps=None,
    eps_fb=1e-6,
    u_floor=1e-8,
    **fit_kwargs,
):
    """Recover the transfer function(s) of a marked flame / mass-source feedback.

    The element at ``node`` carries a
    :class:`~nefes.elements.dynamic_source.DynamicSource` whose terms declare the reference
    edges and quantities the response is written against (see
    :func:`~nefes.perturbation.identify.unknown_dynamic_source`); each term's transfer
    function is recovered from the measured transfer matrix between edges ``a`` and ``b``.

    A single measured matrix separates the terms only when its excitations render the
    reference fluctuations linearly independent -- otherwise the per-frequency linear system
    is rank-deficient; :attr:`TransferFunctionIdentification.conditioning` reports this, and
    multiple terms generally need multiple loading conditions.

    Parameters
    ----------
    prob, x_bar, measured, node, a, b, excite, forcing, freeze, continue_
        As for :func:`identify_transfer_matrix`; ``excite`` defaults to the full
        acoustic+entropy set (more channels -> better conditioning).
    isentropic : bool, optional
        Pin entropy to zero everywhere (acoustics-only); the measured matrix must then be
        acoustic (``N=2``) and ``excite`` acoustic.  Default False.
    **fit_kwargs
        Forwarded to :class:`~nefes.perturbation.continuation.RationalFit`.

    Returns
    -------
    TransferFunctionIdentification
    """
    if isentropic:
        excite = tuple(f for f in excite if f == "acoustic") or ("acoustic",)
    freqs = np.asarray(measured.freqs, dtype=float)
    desc = prob.node_dynamic_source[node]
    if desc is None:
        raise ValueError(f"node {node} carries no dynamic source to identify")
    terms_spec = [(int(t.ref_edge), t.quantity, float(t.gain)) for t in desc.terms]

    # A0 := the operator with the source silent (transfer = 0); its stamp structure is unchanged.
    off = DynamicSource(
        terms=[DynamicResponseTerm(Constant(0.0), e, q, g) for (e, q, g) in terms_spec],
        target=desc.target,
        q_mean=desc.q_mean,
    )
    nds = _with_node(prob.node_dynamic_source, prob.n_nodes, node, off)
    prob0 = dataclasses.replace(prob, node_dynamic_source=nds)

    ctx = _build_excitation_context(
        prob0, x_bar, freqs, forcing, eps=eps, eps_fb=eps_fb, u_floor=u_floor, frozen=freeze, isentropic=isentropic
    )
    cals = edge_caloric(prob0, x_bar)
    stamps, _ = build_source_stamps(prob0, x_bar, ctx.K, u_floor, cals)
    st = next((s for s in stamps if s.node == node), None)
    if st is None:
        raise ValueError(f"node {node} carries no dynamic-source feedback")
    K = len(st.terms)
    ci = _excited_char_indices(excite, ctx.scalar_names)

    F = _forcing(ctx, excite)
    p, Q = _source_pq(st, ctx.n_col)  # shared row vector p, per-term column vectors Q
    Ea = _extraction(ctx.L[a], a, ci, ctx.n_solve, ctx.n_col)
    Eb = _extraction(ctx.L[b], b, ci, ctx.n_solve, ctx.n_col)
    M = _measured_char(measured, freqs)

    G = np.empty((K, freqs.size), dtype=np.complex128)
    cond = np.empty(freqs.size)
    resid = np.empty(freqs.size)
    for i, lu in enumerate(ctx.lus):
        G[:, i], cond[i], resid[i] = _deembed_source(lu, F, p, Q, Ea, Eb, M[i])

    tfs: List[TransferFunction] = []
    for k in range(K):
        if continue_:
            tfs.append(RationalFit(freqs, G[k], **fit_kwargs))
        else:
            tfs.append(Tabulated(freqs, G[k]))
    return TransferFunctionIdentification(
        transfer_functions=tfs,
        values=G,
        terms=terms_spec,
        freqs=freqs,
        conditioning=cond,
        residual=resid,
        node=node,
    )


def _source_pq(st, n_col):
    """The shared row vector ``p`` (target rows x factors) and per-term columns ``Q`` (coeff)."""
    p = np.zeros(n_col, dtype=np.complex128)
    for r, fr in zip(st.rows, st.factors):
        p[int(r)] = fr
    Q = np.zeros((n_col, len(st.terms)), dtype=np.complex128)
    for k, term in enumerate(st.terms):
        Q[term.cols, k] = term.coeff
    return p, Q


def _deembed_source(lu, F, p, Q, Ea, Eb, M):
    """Recover the term gains ``G`` at one frequency from ``W_b = M W_a`` (rank-1 shared update)."""
    X0 = lu.solve(F)  # (n_col, Kf)
    g_p = lu.solve(p)  # (n_col,)
    d = Q.T @ g_p  # (K,)  d_k = q_k^T A0^{-1} p
    row = Q.T @ X0  # (K, Kf)  row_k = q_k^T X0
    va, vb = Ea @ g_p, Eb @ g_p  # (N,)
    Wa0, Wb0 = Ea @ X0, Eb @ X0  # (N, Kf)
    R0 = Wb0 - M @ Wa0  # (N, Kf)
    w = vb - M @ va  # (N,)
    # condition: R0 + sum_k G_k (d_k R0 - outer(w, row_k)) = 0
    K = d.size
    A_sys = np.stack([(d[k] * R0 - np.outer(w, row[k])).ravel() for k in range(K)], axis=1)  # (N*Kf, K)
    rhs = -R0.ravel()
    G, *_ = np.linalg.lstsq(A_sys, rhs, rcond=None)
    sv = np.linalg.svd(A_sys, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    res = float(np.linalg.norm(A_sys @ G - rhs) / max(np.linalg.norm(rhs), 1e-30))
    return G, cond, res
