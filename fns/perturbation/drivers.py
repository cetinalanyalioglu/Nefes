"""Acoustic analysis drivers (real-only Python + SciPy, off the @njit line).

Two of the three theory.md s12.7 analyses are provided for v1: ``modes_from_det``
(the nonlinear eigenproblem ``det A(omega) = 0`` by determinant minimization) and
``scattering_2port`` (a 2-port transfer/scattering helper used by the duct model).
"""

# CA: Do we need this file at all?

import numpy as np
from scipy.optimize import minimize_scalar


def modes_from_det(det_func, omega_grid, tol=1e-6):
    """Find real eigenfrequencies where ``|det A(omega)|`` dips to zero.

    Scans ``omega_grid`` for local minima of ``|det_func(omega)|`` and refines
    each by 1-D minimization.  Returns the refined frequencies (ascending).
    """
    vals = np.array([abs(det_func(w)) for w in omega_grid])
    roots = []
    for i in range(1, len(omega_grid) - 1):
        if vals[i] <= vals[i - 1] and vals[i] < vals[i + 1]:
            a, b = omega_grid[i - 1], omega_grid[i + 1]
            res = minimize_scalar(lambda w: abs(det_func(w)), bounds=(a, b), method="bounded")
            scale = max(abs(det_func(omega_grid[0])), 1.0)
            if abs(det_func(res.x)) < tol * scale or res.fun < vals[i]:
                roots.append(res.x)
    return np.array(sorted(roots))


def scattering_2port(c, length, omega, u=0.0):
    """Acoustic transfer of a uniform duct: phase delays of the two acoustic waves.

    Returns ``diag(exp(-i*omega*tau_+), exp(-i*omega*tau_-))`` mapping the
    incoming wave amplitudes (downstream f at the tail, upstream g at the head)
    to the outgoing ones.  Lossless: both entries have unit modulus.
    """
    tau_p = length / (u + c)
    tau_m = length / (c - u)
    return np.array([[np.exp(-1j * omega * tau_p), 0.0], [0.0, np.exp(-1j * omega * tau_m)]])
