"""Newton control loop with nondimensionalization, globalization, and homotopy.

The solve runs a short vanishing-friction homotopy: stages ``stab in
(0.1, 0.01, 0.0)`` each warm-started from the previous, with the smoothing width
``eps = max(0.3*stab, 1e-4) * mdot_ref``.  Each stage is a damped Newton: a
sparse-LU step with backtracking line search on the scaled residual norm, and a
Levenberg-Marquardt fallback when the LU step stalls or the Jacobian is singular.
The final stage solves the exact equations (stab = 0).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..assemble import residual, jacobian
from ..elements.ids import MASS_FLOW_INLET, PT_INLET, MASS_SOURCE
from ..thermo.api import EQ_KERNEL, PERFECT_GAS
from .linear import newton_step, lm_step, scaled_system, col_scale

EPS_FB = 1e-5


def flatten(x2d):
    return np.ascontiguousarray(x2d).T.ravel()


def unflatten(flat, n_edges, n_solve=3):
    return np.ascontiguousarray(flat.reshape(n_edges, n_solve).T)


def initial_guess(prob, mdot0=None, p0=None, h0=None, z0=None):
    """Initial state; a small co-directional ``mdot`` by default.

    Each of ``mdot0``, ``p0``, ``h0`` may be a scalar (all edges) or an array of
    length ``n_edges`` (per-edge).  ``z0`` (composition rows ``3..n_solve``) may be
    ``None`` (zeros), a scalar, a ``(n_elem,)`` vector (same composition on every
    edge), or a ``(n_elem, n_edges)`` array (per-edge composition).  A reacting
    network usually needs **per-edge** ``h0``/``z0``: an unburnt air edge and a
    burnt edge sit at very different ``(h_t, Z)``, so a single uniform guess can
    leave a frozen ``h -> T`` inversion or an equilibrium solve far from any root.
    """
    mdot_ref, p_ref, h_ref = prob.var_scale[0], prob.var_scale[1], prob.var_scale[2]
    x = np.zeros((prob.n_solve, prob.n_edges))
    x[0, :] = 0.05 * mdot_ref if mdot0 is None else mdot0
    x[1, :] = p_ref if p0 is None else p0
    x[2, :] = h_ref if h0 is None else h0
    if prob.n_solve > 3:
        if z0 is None:
            x[3:, :] = 0.0
        else:
            z = np.asarray(z0, dtype=np.float64)
            if z.ndim == 2:
                x[3:, :] = z  # (n_elem, n_edges) per-edge composition
            else:
                x[3:, :] = z.reshape(-1, 1)  # (n_elem,) same composition on every edge
    return x


def auto_initial_guess(prob, mdot0=None, p0=None):
    """Physically-seeded per-edge state by propagating the feeds through the graph.

    A reacting network spans a wide enthalpy range -- an unburnt air edge sits at
    ``h_t ~ +1.9e3`` J/kg while CH4-laden / burnt edges sit at ``~ -2.2e5`` J/kg
    (the formation-inclusive datum, D-1) -- so a single uniform guess strands the
    frozen ``h -> T`` inversion or the burnt equilibrium far from any root.  This
    builds each edge's seed by **mass-weighted mixing along the network**, with no
    case-specific tuning:

    1. propagate the edge mass flow (inlets/sources inject; junctions sum; splitters
       divide), giving each edge a conserved ``mdot`` estimate;
    2. blend the advected scalars -- total enthalpy ``h_t`` and the feed-stream
       mixture fractions ``xi`` -- mass-weighted by that ``mdot``.

    Because every feed enters with its own ``h_t`` (formation + sensible at its
    injection ``T``) and ``xi = e_stream``, the blend lands each edge at exactly the
    adiabatic-mixing ``(h_t, xi)``: the burnt edge inherits the fuel+air mixture
    enthalpy automatically, and the equilibrium solve turns it into the flame
    temperature.  Conserved scalars mix linearly, so this is the right basin
    regardless of how negative the formation enthalpies are.
    """
    N = prob.n_nodes
    E = prob.n_edges
    n_solve = prob.n_solve
    nscal = n_solve - 2  # advected scalars: h_t (0) + mixture fractions (1..K)
    mdot_ref, p_ref, h_ref = prob.var_scale[0], prob.var_scale[1], prob.var_scale[2]
    md = float(mdot_ref if mdot0 is None else mdot0)

    tail = np.asarray(prob.tail_node)
    head = np.asarray(prob.head_node)
    rid = np.asarray(prob.node_rid)
    npar_f = np.asarray(prob.npar_f)
    ptr = np.asarray(prob.npar_fptr)

    # per-node injected (mdot, [h_t, xi...]) for the stream-introducing elements
    inj_mdot = np.zeros(N)
    inj_scal = np.zeros((N, nscal))
    has_inj = np.zeros(N)
    for n in range(N):
        pb = ptr[n]
        r = rid[n]
        if r == MASS_FLOW_INLET:
            inj_mdot[n] = npar_f[pb + 0]
            inj_scal[n] = npar_f[pb + 1 : pb + 1 + nscal]
            has_inj[n] = 1.0
        elif r == PT_INLET:
            inj_mdot[n] = md  # mdot is unknown at a pt-inlet; weight by the reference
            inj_scal[n] = npar_f[pb + 1 : pb + 1 + nscal]
            has_inj[n] = 1.0
        elif r == MASS_SOURCE:
            inj_mdot[n] = npar_f[pb + 0]
            inj_scal[n] = npar_f[pb + 2 : pb + 2 + nscal]
            has_inj[n] = 1.0

    out_count = np.zeros(N)
    np.add.at(out_count, tail, 1.0)
    max_sweeps = min(N + 5, 1000)

    # (1) propagate the edge mass flow
    edge_mdot = np.full(E, md)
    has_out = out_count[tail] > 0
    for _ in range(max_sweeps):
        node_in = inj_mdot * has_inj
        np.add.at(node_in, head, edge_mdot)
        new = edge_mdot.copy()
        new[has_out] = (node_in[tail] / np.maximum(out_count[tail], 1.0))[has_out]
        if np.allclose(new, edge_mdot, rtol=1e-12, atol=1e-12 * md):
            edge_mdot = new
            break
        edge_mdot = new

    # (2) blend the advected scalars, mass-weighted by the propagated mdot
    default = np.zeros(nscal)
    default[0] = h_ref
    inj_nodes = np.nonzero(has_inj)[0]
    if inj_nodes.size:
        default = inj_scal[inj_nodes[0]].copy()
    edge_scal = np.tile(default, (E, 1))
    inj_w = inj_mdot * has_inj
    for _ in range(max_sweeps):
        node_num = (inj_w[:, None] * inj_scal).copy()
        node_den = inj_w.copy()
        np.add.at(node_den, head, edge_mdot)
        np.add.at(node_num, head, edge_mdot[:, None] * edge_scal)
        good = node_den > 0.0
        node_mix = np.zeros((N, nscal))
        node_mix[good] = node_num[good] / node_den[good][:, None]
        new = edge_scal.copy()
        valid = has_out & good[tail]
        new[valid] = node_mix[tail][valid]
        if np.allclose(new, edge_scal, rtol=1e-12, atol=1e-12):
            edge_scal = new
            break
        edge_scal = new

    x = np.zeros((n_solve, E))
    x[0, :] = edge_mdot
    x[1, :] = float(p_ref if p0 is None else p0)
    x[2, :] = edge_scal[:, 0]
    if nscal > 1:
        x[3:, :] = edge_scal[:, 1:].T
    return x


@dataclass
class SolveResult:
    x: np.ndarray  # converged state, shape (3, E)
    converged: bool
    iterations: int
    residual_norm: float
    history: List[float] = field(default_factory=list)


@dataclass
class _Reporter:
    """Newton-progress printer (see ``solve``'s ``verbose``/``progress_interval``).

    ``level`` 0 is silent; 1 prints a one-line summary per homotopy stage; 2 also
    prints the scaled residual norm every ``interval`` iterations within a stage.
    """

    level: int = 0
    interval: int = 1

    def stage_start(self, stab, eps):
        if self.level >= 2:
            print(f"[stab={stab:<5g} eps={eps:.2e}]")

    def iteration(self, it, norm):
        if self.level >= 2 and (it % self.interval == 0):
            print(f"  it {it:3d}   ||R_hat||={norm:.3e}")

    def stage_end(self, stab, it, norm, converged):
        if self.level >= 1:
            print(f"stab={stab:<5g} -> {it:3d} iters, ||R_hat||={norm:.3e}, converged={converged}")


def _merit(prob, x2d, eps, stab, res_scale):
    """Scaled residual 2-norm; +inf if the state is non-physical.

    Pressure must stay positive for every model.  The transported scalar in row 2
    is the absolute total enthalpy ``h_t``: for a perfect gas ``h_t = cp*Tt > 0``,
    but with the reacting backend's formation-inclusive datum (D-1) ``h_t`` is
    legitimately **negative** for fuels with a negative formation enthalpy (e.g.
    CH4), so the positivity guard applies only to the perfect gas.  Physicality of
    a reacting state is then enforced by the closure (a failed ``h -> T`` inversion
    raises and is caught below).
    """
    if np.any(x2d[1, :] <= 0.0):
        return np.inf, None
    if prob.model_id == PERFECT_GAS and np.any(x2d[2, :] <= 0.0):
        return np.inf, None
    try:
        R = residual(prob, x2d, eps, EPS_FB, stab)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(R)):
        return np.inf, None
    R_hat = R / res_scale
    return float(np.linalg.norm(R_hat)), R


def _solve_stage(prob, x2d, eps, stab, tol, max_iter, history, reporter=None):
    res_scale = prob.res_scale
    vcol = col_scale(prob.var_scale, prob.n_edges)
    lam = 1e-3
    norm, R = _merit(prob, x2d, eps, stab, res_scale)
    for it in range(max_iter):
        history.append(norm)
        if reporter is not None:
            reporter.iteration(it, norm)
        if norm < tol:
            return x2d, True, it, norm
        J = jacobian(prob, x2d, eps, EPS_FB, stab)
        J_hat, R_hat = scaled_system(J, R, vcol, res_scale)

        dy = newton_step(J_hat, R_hat)
        accepted = False
        if dy is not None:
            dx = unflatten(vcol * dy, prob.n_edges, prob.n_solve)
            alpha = 1.0
            for _ in range(30):
                x_try = x2d + alpha * dx
                n_try, R_try = _merit(prob, x_try, eps, stab, res_scale)
                if n_try < (1.0 - 1e-4 * alpha) * norm:
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * 0.5, 1e-12)
                    break
                alpha *= 0.5

        if not accepted:
            # Levenberg-Marquardt fallback with adaptive damping.
            for _ in range(40):
                dy = lm_step(J_hat, R_hat, lam)
                x_try = x2d + unflatten(vcol * dy, prob.n_edges, prob.n_solve)
                n_try, R_try = _merit(prob, x_try, eps, stab, res_scale)
                if n_try < norm:
                    x2d, norm, R = x_try, n_try, R_try
                    accepted = True
                    lam = max(lam * 0.5, 1e-12)
                    break
                lam = min(lam * 4.0, 1e8)
            if not accepted:
                return x2d, False, it, norm
    return x2d, norm < tol, max_iter, norm


def solve(prob, x0=None, tol=1e-10, max_iter=80, stab_stages=(0.1, 0.01, 0.0), verbose=0, progress_interval=1):
    """Solve the steady mean flow.  Returns a SolveResult (state shape (3, E)).

    Parameters
    ----------
    prob : Problem
        Compiled flow network.
    x0 : ndarray, optional
        Initial state, shape ``(3, E)`` (default: a uniform co-directional guess).
    tol : float, optional
        Convergence tolerance on the scaled residual 2-norm.
    max_iter : int, optional
        Maximum Newton iterations per homotopy stage.
    stab_stages : sequence of float, optional
        Vanishing-friction homotopy schedule, warm-started in order.
    verbose : int or bool, optional
        Progress verbosity.  ``0``/``False`` is silent; ``1``/``True`` prints a
        one-line summary per homotopy stage; ``2`` additionally prints the scaled
        residual norm every ``progress_interval`` iterations within each stage.
    progress_interval : int, optional
        Iteration stride for the per-iteration prints at ``verbose >= 2``.

    Returns
    -------
    SolveResult
        The converged state and solve diagnostics.
    """
    mdot_ref = prob.var_scale[0]
    if x0 is not None:
        x2d = np.array(x0, dtype=np.float64)
    elif prob.model_id == EQ_KERNEL:
        # reacting networks need a physically-seeded per-edge guess (wide h_t range)
        x2d = auto_initial_guess(prob)
    else:
        x2d = initial_guess(prob)
    reporter = _Reporter(level=int(verbose), interval=max(1, int(progress_interval)))
    history: List[float] = []
    total_it = 0
    converged = False
    norm = np.inf
    for stab in stab_stages:
        eps = max(0.3 * stab, 1e-4) * mdot_ref
        reporter.stage_start(stab, eps)
        x2d, converged, it, norm = _solve_stage(prob, x2d, eps, stab, tol, max_iter, history, reporter)
        total_it += it
        reporter.stage_end(stab, it, norm, converged)
        if not converged and stab == 0.0:
            break
    return SolveResult(x=x2d, converged=converged, iterations=total_it, residual_norm=norm, history=history)


def states_table(prob, x2d):
    """Recover the full edge-state table (NS_EST, E) for diagnostics/output."""
    from ..derive import recover_all, NS_EST

    est = np.zeros((NS_EST, prob.n_edges))
    recover_all(prob.edge_model, prob.tf, prob.ti, np.ascontiguousarray(x2d), prob.area, prob.n_elem, est)
    return est
