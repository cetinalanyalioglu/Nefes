"""Characteristic-variable maps and perturbation-variable *flavors* at a mean
edge state.

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

from ...assembly.recover import ES_RHO, ES_U, ES_C, ES_AREA, ES_DHDRHO, ES_DHDP


def caloric_row(est_col):
    """The caloric coupling row ``(a, u, b)`` at one edge, read from the state table.

    ``dh_t = a d_rho + u d_u + b d_p`` with ``a = (dh/drho)_p`` and ``b = (dh/dp)_rho``
    the edge's own thermo-model partials (columns ``ES_DHDRHO``/``ES_DHDP``, filled by
    :func:`~nefes.assembly.recover.enrich_caloric`); the middle term is the kinematic ``u``.
    """
    return (float(est_col[ES_DHDRHO]), float(est_col[ES_U]), float(est_col[ES_DHDP]))


def char_to_dq(rho, c):
    """R: (d_rho, d_u, d_p) = R @ (f, g, h)."""
    return np.array(
        [
            [rho / c, rho / c, 1.0],
            [1.0, -1.0, 0.0],
            [rho * c, rho * c, 0.0],
        ]
    )


def dq_to_dx(rho, u, area, cal):
    """(d_mdot, d_p, d_h_t) from (d_rho, d_u, d_p); ``mdot = rho*u*A``.

    The total-enthalpy (third) row is the gas's **caloric coupling** ``cal = (a, u, b)``:
    the row ``dh_t = a*d_rho + u*d_u + b*d_p`` with the edge's own thermo-model partials
    ``a = (dh/drho)_p`` and ``b = (dh/dp)_rho`` (:func:`caloric_row`, from the edge-state
    table).  Reading it per edge keeps the perturbation maps consistent with the mean-flow
    Jacobian for every gas model -- perfect gas and reacting alike.
    """
    a, m, b = cal
    return np.array(
        [
            [u * area, rho * area, 0.0],
            [0.0, 0.0, 1.0],
            [a, m, b],
        ]
    )


def char_to_dx(rho, c, u, area, cal):
    """T_e: (d_mdot, d_p, d_h_t) = T_e @ (f, g, h).  ``cal``: see :func:`dq_to_dx`."""
    return dq_to_dx(rho, u, area, cal) @ char_to_dq(rho, c)


def dx_to_char(rho, c, u, area, cal):
    """L_e = T_e^-1: (f, g, h) = L_e @ (d_mdot, d_p, d_h_t).  ``cal``: see :func:`dq_to_dx`."""
    return np.linalg.inv(char_to_dx(rho, c, u, area, cal))


def edge_transforms(est):
    """Per-edge (T_e, L_e) lists from the edge-state table ``est`` (``caloric=True``).

    The caloric coupling of each edge is read straight from the table
    (:func:`caloric_row`), so both lists are consistent with the mean-flow Jacobian for
    every gas model.
    """
    E = est.shape[1]
    Ts, Ls = [], []
    for e in range(E):
        T = char_to_dx(est[ES_RHO, e], est[ES_C, e], est[ES_U, e], est[ES_AREA, e], caloric_row(est[:, e]))
        Ts.append(T)
        Ls.append(np.linalg.inv(T))
    return Ts, Ls


def edge_caloric(prob, x_bar):
    """Per-edge caloric row ``(a, u, b)`` of :func:`dq_to_dx` from the edge-state dataset.

    Recovers the state with the caloric columns filled and reads each edge's row
    (:func:`caloric_row`): ``a = (dh/drho)_p`` and ``b = (dh/dp)_rho`` come from that edge's
    own thermo model (:func:`~nefes.assembly.recover.enrich_caloric`), so the row is
    consistent with the mean-flow Jacobian for perfect gas and reacting alike.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled network.
    x_bar : ndarray
        Converged mean-flow solve state, shape ``(n_solve, E)``.

    Returns
    -------
    list of tuple
        ``E`` rows ``(a, u, b)``, one per edge.
    """
    from ...solver.report import states_table

    est = states_table(prob, x_bar, caloric=True)
    return [caloric_row(est[:, e]) for e in range(int(prob.n_edges))]


# --------------------------------------------------------------------------
# Perturbation-variable flavors: B with  v_basis = B @ w,  w = (f, g, h).
# --------------------------------------------------------------------------

# Symbols of the three fixed characteristic amplitudes ``w = (f, g, h)`` -- the canonical
# names reused by the response/eigenmode readouts and by ``BASIS_LABELS["char"]``.
CHAR_SYMBOLS = ("f", "g", "h")

# Per-component symbols as LaTeX (MathJax) fragments, for plot labelling.  The
# plotting layer wraps and subscripts these (so they must group cleanly under a
# subscript); they are not meant for plain-text display.
BASIS_LABELS = {
    "char": CHAR_SYMBOLS,
    "primitive": (r"p'/\rho c", r"u'", r"\rho' c/\rho"),
    "network": (r"\dot{m}'", r"p'", r"h_t'"),
    "riemann": (r"P^+", r"P^-", r"\sigma"),
    "pu_entropy": (r"p'/\rho c", r"u'", r"s'/c_p"),
    "pu_rho": (r"p'/\rho c", r"u'", r"\rho'"),
}

# Friendly single-variable names -> (basis flavor, component) into ``BASIS_LABELS``, for
# selecting one quantity across flavors (e.g. ``"p"``, ``"u"``, ``"rho"``).  Its display
# symbol is ``BASIS_LABELS[flavor][component]``.
VARIABLE_SPEC = {
    "p": ("network", 1),
    "u": ("primitive", 1),
    "rho": ("pu_rho", 2),
    "mdot": ("network", 0),
    "f": ("char", 0),
    "g": ("char", 1),
    "h": ("char", 2),
}


def basis_matrix(basis, rho, c, u, area, cal=None):
    """3x3 block ``B`` mapping characteristic amplitudes ``w=(f,g,h)`` to a flavor.

    ``v_basis = B @ w``.  All flavors are non-singular rescalings of ``w`` at any
    physical state, so any transfer matrix converts between flavors by a similarity built
    from these blocks (see ``matrices.tm_in_basis``).  Only the ``network`` flavor needs the
    caloric coupling ``cal = (a, u, b)`` (see :func:`dq_to_dx`); the others ignore it.
    """
    if basis == "char":
        return np.eye(3)
    if basis == "primitive":  # velocity-normalized (p'/(rho c), u', rho' c/rho)
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, c / rho]])
    if basis == "network":  # (d_mdot, d_p, d_h_t)
        if cal is None:
            raise ValueError("the 'network' flavor needs the caloric row (a, u, b); pass cal or a caloric est column")
        return char_to_dx(rho, c, u, area, cal)
    if basis == "riemann":  # (P+, P-, sigma) = (f/c, g/c, -h/rho)
        return np.array([[1.0 / c, 0.0, 0.0], [0.0, 1.0 / c, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_entropy":  # (p'/(rho c), u', s'/cp);  s'/cp = -h/rho
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, -1.0 / rho]])
    if basis == "pu_rho":  # (p'/(rho c), u', rho')
        return np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [rho / c, rho / c, 1.0]])
    raise ValueError(f"unknown perturbation-variable flavor {basis!r}; choose from {sorted(BASIS_LABELS)}")


def basis_block_from_state(basis, est_col, cal=None):
    """``basis_matrix`` from one column ``est[:, e]`` of the edge-state table.

    The caloric row defaults to the column's own caloric entries (:func:`caloric_row`), so a
    ``caloric=True`` state column carries everything the ``network`` flavor needs.
    """
    return basis_matrix(
        basis,
        float(est_col[ES_RHO]),
        float(est_col[ES_C]),
        float(est_col[ES_U]),
        float(est_col[ES_AREA]),
        caloric_row(est_col) if cal is None else cal,
    )
