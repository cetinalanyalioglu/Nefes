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
    return np.tile(var_scale, n_edges)
