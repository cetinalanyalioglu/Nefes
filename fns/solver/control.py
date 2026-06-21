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
from .linear import newton_step, lm_step, scaled_system, col_scale

EPS_FB = 1e-5


def flatten(x2d):
    return np.ascontiguousarray(x2d).T.ravel()


def unflatten(flat, n_edges):
    return np.ascontiguousarray(flat.reshape(n_edges, 3).T)


def initial_guess(prob, mdot0=None, p0=None, h0=None):
    """Uniform initial state; a small co-directional mdot by default."""
    mdot_ref, p_ref, h_ref = prob.var_scale
    x = np.zeros((3, prob.n_edges))
    x[0, :] = 0.05 * mdot_ref if mdot0 is None else mdot0
    x[1, :] = p_ref if p0 is None else p0
    x[2, :] = h_ref if h0 is None else h0
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
    """Scaled residual 2-norm; +inf if the state is non-physical."""
    if np.any(x2d[1, :] <= 0.0) or np.any(x2d[2, :] <= 0.0):
        return np.inf, None
    try:
        R = residual(prob, x2d, eps, EPS_FB, stab)
    except Exception:
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
            dx = unflatten(vcol * dy, prob.n_edges)
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
                x_try = x2d + unflatten(vcol * dy, prob.n_edges)
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
    x2d = initial_guess(prob) if x0 is None else np.array(x0, dtype=np.float64)
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
    recover_all(prob.model_id, prob.tf, prob.ti, np.ascontiguousarray(x2d), prob.area, prob.n_elem, est)
    return est
