"""Characteristic-variable maps and perturbation-variable *flavors* at a mean
edge state (theory.md s9.1-9.2, s12.2).

The 1-D Euler system carries **three** perturbation characteristics -- the same
count as the mean-flow unknowns -- so the perturbation network is genuinely
``N = 3`` (and grows with reacting scalars).  The characteristic amplitudes
``w = (f, g, h)`` are

    u'   = f - g
    p'   = rho*c*(f + g)
    rho' = h + p'/c^2

f: downstream acoustic wave (speed u+c); g: upstream (u-c); h: entropy/convected
wave (speed u).  ``char_to_dx`` is the per-edge block ``T_e`` mapping ``w`` to the
perturbations of the network unknowns ``(d_mdot, d_p, d_h_t)``; ``dx_to_char`` is
``L_e = T_e^-1``.  At the converged mean state these blocks turn the algebraic
Jacobian into the zero-frequency perturbation jump conditions.

A transfer/scattering matrix is the *same* information in different coordinates;
the **flavors** are inter-convertible 3x3 rescalings of ``w``.  ``basis_matrix``
returns the per-edge block ``B`` with ``v_basis = B @ w`` for each named flavor,
and ``BASIS_LABELS`` gives its per-component symbols (for plot labelling):

* ``"char"``        -- ``(f, g, h)``                 the Riemann/entropy amplitudes
* ``"primitive"``   -- ``(rho', u', p')``            primitive fluctuations
* ``"network"``     -- ``(mdot', p', h_t')``         the solver's own unknowns
* ``"riemann"``     -- ``(P+, P-, sigma)``           De Domenico (f/c, g/c, -h/rho)
* ``"pu_entropy"``  -- ``(p'/(rho c), u', s'/cp)``   normalized acoustic + entropy
* ``"pu_rho"``      -- ``(p'/(rho c), u', rho')``    normalized acoustic + density
"""

import numpy as np

from ..derive import ES_RHO, ES_U, ES_P, ES_C, ES_AREA


def char_to_dq(rho, c):
    """R: (d_rho, d_u, d_p) = R @ (f, g, h)."""
    return np.array(
        [
            [rho / c, rho / c, 1.0],
            [1.0, -1.0, 0.0],
            [rho * c, rho * c, 0.0],
        ]
    )


def dq_to_dx(rho, u, p, area, K):
    """(d_mdot, d_p, d_h_t) from (d_rho, d_u, d_p) for a calorically perfect gas.

    mdot = rho*u*A ;  h_t = (cp/R) p/rho + u^2/2.
    """
    return np.array(
        [
            [u * area, rho * area, 0.0],
            [0.0, 0.0, 1.0],
            [-K * p / rho**2, u, K / rho],
        ]
    )


def char_to_dx(rho, c, u, p, area, K):
    """T_e: (d_mdot, d_p, d_h_t) = T_e @ (f, g, h)."""
    return dq_to_dx(rho, u, p, area, K) @ char_to_dq(rho, c)


def dx_to_char(rho, c, u, p, area, K):
    """L_e = T_e^-1: (f, g, h) = L_e @ (d_mdot, d_p, d_h_t)."""
    return np.linalg.inv(char_to_dx(rho, c, u, p, area, K))


def edge_transforms(est, K):
    """Per-edge (T_e, L_e) lists from the mean edge-state table ``est``."""
    E = est.shape[1]
    Ts, Ls = [], []
    for e in range(E):
        rho = est[ES_RHO, e]
        c = est[ES_C, e]
        u = est[ES_U, e]
        p = est[ES_P, e]
        area = est[ES_AREA, e]
        T = char_to_dx(rho, c, u, p, area, K)
        Ts.append(T)
        Ls.append(np.linalg.inv(T))
    return Ts, Ls


# --------------------------------------------------------------------------
# Perturbation-variable flavors: B with  v_basis = B @ w,  w = (f, g, h).
# --------------------------------------------------------------------------

BASIS_LABELS = {
    "char": ("f", "g", "h"),
    "primitive": ("ρ'", "u'", "p'"),
    "network": ("ṁ'", "p'", "h_t'"),
    "riemann": ("P+", "P−", "σ"),
    "pu_entropy": ("p'/ρc", "u'", "s'/c_p"),
    "pu_rho": ("p'/ρc", "u'", "ρ'"),
}


def basis_matrix(basis, rho, c, u, p, area, K):
    """3x3 block ``B`` mapping characteristic amplitudes ``w=(f,g,h)`` to a flavor.

    ``v_basis = B @ w``.  All flavors are non-singular rescalings of ``w`` at any
    physical state, so any transfer matrix converts between flavors by a
    similarity built from these blocks (see ``matrices.tm_in_basis``).
    """
    if basis == "char":
        return np.eye(3)
    if basis == "primitive":  # (rho', u', p')
        return char_to_dq(rho, c)
    if basis == "network":  # (d_mdot, d_p, d_h_t)
        return char_to_dx(rho, c, u, p, area, K)
    if basis == "riemann":  # (P+, P-, sigma) = (f/c, g/c, -h/rho)
        return np.array([[1.0 / c, 0.0, 0.0], [0.0, 1.0 / c, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_entropy":  # (p'/(rho c), u', s'/cp);  s'/cp = -h/rho
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_rho":  # (p'/(rho c), u', rho')
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [rho / c, rho / c, 1.0]])
    raise ValueError(f"unknown perturbation-variable flavor {basis!r}; choose from {sorted(BASIS_LABELS)}")


def basis_block_from_state(basis, est_col, K):
    """``basis_matrix`` from one column ``est[:, e]`` of the mean edge-state table."""
    return basis_matrix(
        basis,
        float(est_col[ES_RHO]),
        float(est_col[ES_C]),
        float(est_col[ES_U]),
        float(est_col[ES_P]),
        float(est_col[ES_AREA]),
        K,
    )
