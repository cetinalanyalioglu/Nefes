"""Residual and Jacobian assembly.

Two routines build the nonlinear system the Newton solve drives to zero.

``assemble_residual`` fills the residual vector by walking two families of rows:
the node balances (mass / momentum via the CSR node-edge map) and the edge
transport rows (upwinded advection of ``h_t`` and composition along each edge).

``jacobian_fill`` builds the sparse Jacobian by complex-step differentiation.
Each band-1 unknown lives on a single edge, so perturbing it only re-recovers
that edge's state and only disturbs the residual rows that read it: the two
endpoint node blocks and the transport rows of the edges sharing those nodes.
Each column recomputes just those rows and scatters ``Im/h`` into the fixed CSC
pattern, keeping assembly cost proportional to the number of nonzeros.
"""

import numpy as np
import scipy.sparse as sp
from numba import njit

from .derive import recover_all, recover_edge, NS_EST, ES_MDOT
from .smooth import smooth_step
from ..elements.kernels import node_residual, node_donor
from ..thermo.api import PERFECT_GAS

# Complex step perturbation size
CS_H = 1e-30


@njit(cache=True)
def assemble_residual(
    edge_model,
    tf,
    ti,
    n_elem,
    x,
    area,
    row_ptr,
    col_edge,
    orient,
    tail_node,
    head_node,
    node_rid,
    npar_f,
    npar_fptr,
    node_row_ptr,
    transport_row0,
    marker_row,
    marker_s,
    eps,
    node_eps,
    eps_fb,
    kappa,
    est,
    R,
    nj_cache,
):
    """Fill the full residual vector R (length n_eq) and the est table."""
    N = node_rid.shape[0]
    E = x.shape[1]
    # refresh every edge's recovered (rho, u, T, c, ...) state into the est table
    recover_all(edge_model, tf, ti, x, area, n_elem, marker_row, est, nj_cache)

    # node balance rows: each node emits its mass / momentum(-pressure) block
    for n in range(N):
        eps_n = node_eps[n] if node_eps[n] >= 0.0 else eps  # per-element smoothing override
        node_residual(
            n, node_rid[n], row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps_n, eps_fb, kappa, est, R, node_row_ptr
        )

    # advected scalars: band-1 rows 2.. (s=0 is h_t, s>=1 are composition Z_el, marker_s is the marker)
    n_scalars = x.shape[0] - 2
    mdot_e = est[ES_MDOT]
    Hd = R[:N] * 0.0  # per-node donor value of the current scalar (reused each s)
    for s in range(n_scalars):
        phi_e = x[2 + s]
        # donor value seen at each node (the flux-weighted upstream mix of scalar s)
        for n in range(N):
            Hd[n] = node_donor(
                n, node_rid[n], s, marker_s, row_ptr, col_edge, orient, npar_f, npar_fptr, tf, eps, mdot_e, phi_e
            )
        # transport residual: edge scalar minus its upwind-donor value (0 at solution)
        for e in range(E):
            theta = smooth_step(est[ES_MDOT, e], eps)  # ~1 forward flow, ~0 reversed
            phi_up = theta * Hd[tail_node[e]] + (1.0 - theta) * Hd[head_node[e]]
            R[transport_row0 + s * E + e] = phi_e[e] - phi_up


@njit(cache=True)
def _find_slot(c, row, indptr, indices):
    """Binary-search the CSC data index of entry ``(row, c)`` in the fixed pattern."""
    lo = indptr[c]
    hi = indptr[c + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < row:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def jacobian_fill(
    edge_model,
    tf,
    ti,
    n_elem,
    x,
    area,
    row_ptr,
    col_edge,
    orient,
    tail_node,
    head_node,
    node_rid,
    npar_f,
    npar_fptr,
    node_row_ptr,
    transport_row0,
    marker_row,
    marker_s,
    n_eq,
    indptr,
    indices,
    eps,
    node_eps,
    eps_fb,
    kappa,
    Jdata,
    nj_cache,
):
    """Fill the CSC ``Jdata`` array against the fixed (indptr, indices) pattern."""
    n_solve = x.shape[0]
    E = x.shape[1]
    n_scalars = n_solve - 2  # advected scalars: h_t (s=0) + composition Z_el (s>=1) + marker
    H = CS_H
    has_marker = marker_row >= 0

    xc = x.astype(np.complex128)
    est = np.zeros((NS_EST, E), dtype=np.complex128)
    recover_all(edge_model, tf, ti, xc, area, n_elem, marker_row, est, nj_cache)
    Rc = np.zeros(n_eq, dtype=np.complex128)

    # one complex-step column per (edge e, band-1 unknown v): seed x[v, e] += i*H,
    # re-recover only edge e, then read Im/H off just the rows that edge feeds.
    for e in range(E):
        nt = tail_node[e]
        nh = head_node[e]
        eps_nt = node_eps[nt] if node_eps[nt] >= 0.0 else eps  # per-element smoothing override
        eps_nh = node_eps[nh] if node_eps[nh] >= 0.0 else eps
        for v in range(n_solve):
            c = n_solve * e + v
            xc[v, e] = x[v, e] + 1j * H
            marker = xc[marker_row, e] if has_marker else xc[2, e] * 0.0
            recover_edge(
                edge_model[e],
                tf,
                ti,
                xc[0, e],
                xc[1, e],
                xc[2, e],
                area[e],
                xc[3 : 3 + n_elem, e],
                marker,
                est[:, e],
                nj_cache[e],
            )

            # (a) the two endpoint node-equation blocks
            node_residual(
                nt,
                node_rid[nt],
                row_ptr,
                col_edge,
                orient,
                npar_f,
                npar_fptr,
                tf,
                eps_nt,
                eps_fb,
                kappa,
                est,
                Rc,
                node_row_ptr,
            )
            for r in range(node_row_ptr[nt], node_row_ptr[nt + 1]):
                Jdata[_find_slot(c, r, indptr, indices)] = Rc[r].imag / H
            if nh != nt:
                node_residual(
                    nh,
                    node_rid[nh],
                    row_ptr,
                    col_edge,
                    orient,
                    npar_f,
                    npar_fptr,
                    tf,
                    eps_nh,
                    eps_fb,
                    kappa,
                    est,
                    Rc,
                    node_row_ptr,
                )
                for r in range(node_row_ptr[nh], node_row_ptr[nh + 1]):
                    Jdata[_find_slot(c, r, indptr, indices)] = Rc[r].imag / H

            # (b) transport rows of every edge incident to nt or nh (donor coupling),
            #     one set per advected scalar s
            mdot_e = est[ES_MDOT]
            for nd in (nt, nh):
                for k in range(row_ptr[nd], row_ptr[nd + 1]):
                    e2 = col_edge[k]
                    theta = smooth_step(est[ES_MDOT, e2], eps)
                    for s in range(n_scalars):
                        phi_e = xc[2 + s]
                        d_t = node_donor(
                            tail_node[e2],
                            node_rid[tail_node[e2]],
                            s,
                            marker_s,
                            row_ptr,
                            col_edge,
                            orient,
                            npar_f,
                            npar_fptr,
                            tf,
                            eps,
                            mdot_e,
                            phi_e,
                        )
                        d_h = node_donor(
                            head_node[e2],
                            node_rid[head_node[e2]],
                            s,
                            marker_s,
                            row_ptr,
                            col_edge,
                            orient,
                            npar_f,
                            npar_fptr,
                            tf,
                            eps,
                            mdot_e,
                            phi_e,
                        )
                        val = phi_e[e2] - (theta * d_t + (1.0 - theta) * d_h)
                        Jdata[_find_slot(c, transport_row0 + s * E + e2, indptr, indices)] = val.imag / H

            # restore
            xc[v, e] = x[v, e]
            marker = xc[marker_row, e] if has_marker else xc[2, e] * 0.0
            recover_edge(
                edge_model[e],
                tf,
                ti,
                xc[0, e],
                xc[1, e],
                xc[2, e],
                area[e],
                xc[3 : 3 + n_elem, e],
                marker,
                est[:, e],
                nj_cache[e],
            )


# --------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------


def _resolve_node_eps(prob):
    """Per-element eps overrides as a dense float64[N] (< 0 -> follow global eps)."""
    if prob.node_eps is not None:
        return prob.node_eps
    return np.full(prob.n_nodes, -1.0, dtype=np.float64)


def _marker_row(prob):
    """Band-1 row of the transported burnt marker (``-1`` when the network carries none)."""
    return int(getattr(prob, "marker_row", -1))


def _marker_s(prob):
    """Advected-scalar index of the burnt marker (``-1`` when absent); band-1 row is ``2 + s``."""
    mr = int(getattr(prob, "marker_row", -1))
    return mr - 2 if mr >= 0 else -1


def _nj_cache_off(prob):
    """An ``(E, 0)`` cache: disables the equilibrium warm start (the robust uniform guess)."""
    return np.zeros((prob.n_edges, 0), dtype=np.float64)


def _nj_cache_jacobian(prob):
    """Fresh per-edge equilibrium warm-start cache ``(E, Ns)`` for one Jacobian assembly.

    The Jacobian re-solves each edge's equilibrium once per (variable, edge) complex-step
    column.  Seeding those from the freshly-recovered base composition *and* temperature (the
    perturbation is infinitesimal) cuts each to a couple of Newton steps.  Each row is
    ``Np + 1`` wide -- ``Np`` product (gas-phase) moles plus the temperature.  Reacting models
    only; the perfect gas gets the no-op ``(E, 0)``.  A fresh array per call -- no stale
    cross-iterate state to risk diverging the cold first solve.
    """
    width = int(prob.ti[6]) + 1 if prob.model_id != PERFECT_GAS and np.size(prob.ti) > 6 else 0
    return np.zeros((prob.n_edges, width), dtype=np.float64)


def residual(prob, x2d, eps, eps_fb, kappa=0.0):
    """Assemble the residual vector (R) for state ``x2d`` of shape (n_solve, E)."""
    R = np.zeros(prob.n_eq, dtype=x2d.dtype)
    est = np.zeros((NS_EST, prob.n_edges), dtype=x2d.dtype)
    assemble_residual(
        prob.edge_model,
        prob.tf,
        prob.ti,
        prob.n_elem,
        x2d,
        prob.area,
        prob.row_ptr,
        prob.col_edge,
        prob.orient,
        prob.tail_node,
        prob.head_node,
        prob.node_rid,
        prob.npar_f,
        prob.npar_fptr,
        prob.node_row_ptr,
        prob.transport_row0,
        _marker_row(prob),
        _marker_s(prob),
        eps,
        _resolve_node_eps(prob),
        eps_fb,
        kappa,
        est,
        R,
        _nj_cache_off(prob),
    )
    return R


def jacobian(prob, x2d, eps, eps_fb, kappa=0.0):
    """Assemble the sparse Jacobian (scipy CSC) for state ``x2d``."""
    Jdata = np.zeros(len(prob.indices), dtype=np.float64)
    jacobian_fill(
        prob.edge_model,
        prob.tf,
        prob.ti,
        prob.n_elem,
        np.ascontiguousarray(x2d),
        prob.area,
        prob.row_ptr,
        prob.col_edge,
        prob.orient,
        prob.tail_node,
        prob.head_node,
        prob.node_rid,
        prob.npar_f,
        prob.npar_fptr,
        prob.node_row_ptr,
        prob.transport_row0,
        _marker_row(prob),
        _marker_s(prob),
        prob.n_eq,
        prob.indptr,
        prob.indices,
        eps,
        _resolve_node_eps(prob),
        eps_fb,
        kappa,
        Jdata,
        _nj_cache_jacobian(prob),
    )
    return sp.csc_matrix((Jdata, prob.indices, prob.indptr), shape=(prob.n_eq, prob.n_col))


def jacobian_dense(prob, x2d, eps, eps_fb, kappa=0.0, h=CS_H):
    """Reference dense complex-step Jacobian (full re-eval per column)."""
    n, E = prob.n_solve, prob.n_edges
    J = np.zeros((prob.n_eq, n * E))
    xc = x2d.astype(np.complex128)
    for e in range(E):
        for v in range(n):
            xc[v, e] = x2d[v, e] + 1j * h
            R = residual(prob, xc, eps, eps_fb, kappa)
            J[:, n * e + v] = R.imag / h
            xc[v, e] = x2d[v, e]
    return J
