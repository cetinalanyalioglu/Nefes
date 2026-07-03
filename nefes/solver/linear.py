"""Linear-step backends for the Newton control loop (scipy sparse).

Default: a direct sparse LU on the scaled Jacobian (the un-squared system, which
the acoustic layer also needs).  Fallback: a Levenberg-Marquardt step on the
normal equations for the near-singular symmetric-split states the prototype
flagged (it squares conditioning and densifies, so it is a fallback only).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def newton_step(J_hat, R_hat):
    """Solve ``J_hat dy = -R_hat`` by sparse LU.  Returns dy or None on failure."""
    try:
        lu = spla.splu(sp.csc_matrix(J_hat))
        return lu.solve(-R_hat)
    except (RuntimeError, ValueError):
        return None


def lm_step(J_hat, R_hat, lam):
    """Levenberg-Marquardt step: ``(J^T J + lam I) dy = -J^T R``."""
    Jc = sp.csc_matrix(J_hat)
    n = Jc.shape[1]
    A = (Jc.T @ Jc + lam * sp.identity(n, format="csc")).tocsc()
    b = -(Jc.T @ R_hat)
    return spla.spsolve(A, b)


def scaled_system(J, R, var_scale_col, res_scale):
    """Return the nondimensionalized ``(J_hat, R_hat)``.

    ``R_hat = R / res_scale``;  ``J_hat = diag(1/res_scale) J diag(var_scale)``.
    """
    R_hat = R / res_scale
    Dr = sp.diags(1.0 / res_scale)
    Dc = sp.diags(var_scale_col)
    J_hat = (Dr @ sp.csc_matrix(J) @ Dc).tocsc()
    return J_hat, R_hat


def col_scale(var_scale, n_edges):
    """Per-column variable scale for the flattened system: ``var_scale`` tiled per edge.

    Matches the edge-major column ordering of :func:`unflatten` and the Jacobian, so a
    single vector rescales every unknown of every edge.
    """
    return np.tile(var_scale, n_edges)


def unflatten(flat, n_edges, n_solve=3):
    """Reshape a flat Newton solution vector back to a ``(n_solve, n_edges)`` state.

    The solver stacks the per-edge unknowns edge-major (each edge's ``n_solve`` variables
    contiguous) to form the column vector the sparse solve operates on; this is the inverse
    reshape, matching the column ordering of :func:`col_scale` and the assembled Jacobian.
    """
    return np.ascontiguousarray(flat.reshape(n_edges, n_solve).T)
