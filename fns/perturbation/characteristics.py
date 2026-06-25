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
* ``"primitive"``   -- ``(p'/rho c, u', rho' c/rho)``  velocity-normalized primitives
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


def dq_to_dx(rho, u, p, area, K, cal=None):
    """(d_mdot, d_p, d_h_t) from (d_rho, d_u, d_p); ``mdot = rho*u*A``.

    The total-enthalpy (third) row is the gas's **caloric coupling**.  When ``cal``
    is ``None`` the calorically perfect form ``h_t = (cp/R) p/rho + u^2/2`` is used,
    i.e. the row ``(-K p/rho^2, u, K/rho)`` with ``K = cp/R``.  For a reacting /
    variable-``gamma`` edge pass ``cal = (a, m, b)`` -- the row ``dh_t = a*d_rho +
    m*d_u + b*d_p`` taken from a complex step of the converged closure
    (:func:`edge_caloric`); this is what keeps the perturbation maps consistent with
    the mean-flow Jacobian (the perfect-gas ``K`` is wrong for the reacting backend,
    and is invisible to the passive spectrum but load-bearing once a dynamic source
    references a velocity/density fluctuation -- theory.md s12.2).
    """
    if cal is None:
        a, m, b = -K * p / rho**2, u, K / rho
    else:
        a, m, b = cal
    return np.array(
        [
            [u * area, rho * area, 0.0],
            [0.0, 0.0, 1.0],
            [a, m, b],
        ]
    )


def char_to_dx(rho, c, u, p, area, K, cal=None):
    """T_e: (d_mdot, d_p, d_h_t) = T_e @ (f, g, h).  ``cal``: see :func:`dq_to_dx`."""
    return dq_to_dx(rho, u, p, area, K, cal) @ char_to_dq(rho, c)


def dx_to_char(rho, c, u, p, area, K, cal=None):
    """L_e = T_e^-1: (f, g, h) = L_e @ (d_mdot, d_p, d_h_t).  ``cal``: see :func:`dq_to_dx`."""
    return np.linalg.inv(char_to_dx(rho, c, u, p, area, K, cal))


def edge_transforms(est, K, cals=None):
    """Per-edge (T_e, L_e) lists from the mean edge-state table ``est``.

    ``cals`` (optional): per-edge caloric rows from :func:`edge_caloric`; when given,
    edge ``e`` uses ``cals[e]`` instead of the perfect-gas ``K`` form.
    """
    E = est.shape[1]
    Ts, Ls = [], []
    for e in range(E):
        rho = est[ES_RHO, e]
        c = est[ES_C, e]
        u = est[ES_U, e]
        p = est[ES_P, e]
        area = est[ES_AREA, e]
        cal = None if cals is None else cals[e]
        T = char_to_dx(rho, c, u, p, area, K, cal)
        Ts.append(T)
        Ls.append(np.linalg.inv(T))
    return Ts, Ls


def edge_caloric(prob, x_bar):
    """Per-edge caloric row ``(a, m, b)`` of :func:`dq_to_dx` (``dh_t = a*d_rho + m*d_u + b*d_p``).

    A calorically perfect edge keeps the kinetic-energy term (``m = u``) with the
    constant ``K = cp/R`` of the gas; the reacting closures drop it (``m = 0``, the
    MVP ``h ~ h_t``) and take the caloric derivatives ``(dh/drho)_p`` and
    ``(dh/dp)_rho`` from a complex step of the equilibrium/frozen state at the frozen
    mean -- the *same* closure the converged Jacobian ``J_alg`` was built from, so the
    perturbation characteristic maps stay consistent with the mean-flow operator
    (theory.md s12.2).

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network (carries ``edge_model``, ``tf``, ``ti``).
    x_bar : ndarray
        Converged mean-flow solve state, shape ``(n_solve, E)``.

    Returns
    -------
    list of tuple
        ``E`` rows ``(a, m, b)``, one per edge.
    """
    from ..solver.control import states_table
    from ..thermo.api import thermo_state, PERFECT_GAS

    est = states_table(prob, x_bar)
    K = float(prob.tf[0]) / float(prob.tf[1])
    n_elem = int(prob.n_elem)
    x_bar = np.ascontiguousarray(x_bar)
    rows = []
    for e in range(int(prob.n_edges)):
        mid = int(prob.edge_model[e])
        rho = float(est[ES_RHO, e])
        u = float(est[ES_U, e])
        p = float(est[ES_P, e])
        if mid == PERFECT_GAS:
            rows.append((-K * p / rho**2, u, K / rho))
            continue
        # reacting: rho = rho(xi, h_t, p) (KE dropped); invert by complex step.
        xi = np.ascontiguousarray(x_bar[3 : 3 + n_elem, e]).astype(np.complex128)
        ht = complex(x_bar[2, e])
        d = 1e-30
        drho_dh = thermo_state(mid, prob.tf, prob.ti, xi, ht + 1j * d, complex(p))[1].imag / d
        drho_dp = thermo_state(mid, prob.tf, prob.ti, xi, ht + 0j, p + 1j * d)[1].imag / d
        a = 1.0 / drho_dh  # (dh/drho)_p
        b = -drho_dp / drho_dh  # (dh/dp)_rho
        rows.append((a, 0.0, b))
    return rows


# --------------------------------------------------------------------------
# Perturbation-variable flavors: B with  v_basis = B @ w,  w = (f, g, h).
# --------------------------------------------------------------------------

# Per-component symbols as LaTeX (MathJax) fragments, for plot labelling.  The
# plotting layer wraps and subscripts these (so they must group cleanly under a
# subscript); they are not meant for plain-text display.
BASIS_LABELS = {
    "char": ("f", "g", "h"),
    "primitive": (r"p'/\rho c", r"u'", r"\rho' c/\rho"),
    "network": (r"\dot{m}'", r"p'", r"h_t'"),
    "riemann": (r"P^+", r"P^-", r"\sigma"),
    "pu_entropy": (r"p'/\rho c", r"u'", r"s'/c_p"),
    "pu_rho": (r"p'/\rho c", r"u'", r"\rho'"),
}


def basis_matrix(basis, rho, c, u, p, area, K, cal=None):
    """3x3 block ``B`` mapping characteristic amplitudes ``w=(f,g,h)`` to a flavor.

    ``v_basis = B @ w``.  All flavors are non-singular rescalings of ``w`` at any
    physical state, so any transfer matrix converts between flavors by a
    similarity built from these blocks (see ``matrices.tm_in_basis``).  ``cal`` (see
    :func:`dq_to_dx`) supplies the reacting caloric coupling for the ``network`` flavor.
    """
    if basis == "char":
        return np.eye(3)
    if basis == "primitive":  # velocity-normalized (p'/(rho c), u', rho' c/rho)
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, c / rho]])
    if basis == "network":  # (d_mdot, d_p, d_h_t)
        return char_to_dx(rho, c, u, p, area, K, cal)
    if basis == "riemann":  # (P+, P-, sigma) = (f/c, g/c, -h/rho)
        return np.array([[1.0 / c, 0.0, 0.0], [0.0, 1.0 / c, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_entropy":  # (p'/(rho c), u', s'/cp);  s'/cp = -h/rho
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_rho":  # (p'/(rho c), u', rho')
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [rho / c, rho / c, 1.0]])
    raise ValueError(f"unknown perturbation-variable flavor {basis!r}; choose from {sorted(BASIS_LABELS)}")


def basis_block_from_state(basis, est_col, K, cal=None):
    """``basis_matrix`` from one column ``est[:, e]`` of the mean edge-state table."""
    return basis_matrix(
        basis,
        float(est_col[ES_RHO]),
        float(est_col[ES_C]),
        float(est_col[ES_U]),
        float(est_col[ES_P]),
        float(est_col[ES_AREA]),
        K,
        cal,
    )
