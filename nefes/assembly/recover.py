"""Edge-state recovery: the refresh DAG ``unknowns -> closure -> thermo -> flow``.

``recover_edge`` maps one edge's band-1 unknowns ``(mdot, p, h_t [, Z_el])`` and
its ``area`` to the full recovered state used by element residuals.  It is
dtype-generic and complex-step-safe: on the seeded (complex) path every derived
quantity is recomputed inline, never read back from a cache (a stale real value
would drop the imaginary seed).

The recovered state is packed into a fixed-width "edge-state table" column
``est[:, e]`` with the slot layout below.  The state proper (slots ``0..ES_CP``)
is filled inline by ``recover_edge``; the two caloric-derivative slots
(``ES_DHDRHO``/``ES_DHDP``) are filled separately by :func:`enrich_caloric` at a
converged (real) state, because a reacting edge takes them from a complex step of
its closure -- which cannot ride the already-seeded residual/Jacobian recovery.
"""

import numpy as np
from numba import njit

from ..thermo.api import EQ_FROZEN, EQ_KERNEL, EQ_MARKER, PERFECT_GAS, thermo_state, thermo_total_pressure
from ..thermo.edge_state import eq_frozen_state_ke, eq_kernel_state_ke_warm, eq_marker_state_ke_warm
from ..thermo.kernel import RU
from .closure import closure_solve

# edge-state table (est) slot layout
ES_MDOT = 0
ES_P = 1
ES_HT = 2
ES_RHO = 3
ES_U = 4
ES_T = 5
ES_C = 6
ES_M = 7
ES_PT = 8
ES_AREA = 9
ES_W = 10  # mixture molar mass [kg/mol] (rho * R_u * T / p)
ES_CP = 11  # mixture specific heat [J/(kg K)], consistent with the local sound speed
ES_DHDRHO = 12  # caloric partial (dh/drho)_p [J m^3/kg^2] (filled by enrich_caloric)
ES_DHDP = 13  # caloric partial (dh/dp)_rho [m^3/kg] (filled by enrich_caloric)
NS_EST = 14


@njit(cache=True)
def recover_edge(model_id, tf, ti, mdot, p, ht, area, Z_el, marker, out, nj_io):
    """Recover one edge's full state into ``out[0:NS_EST]`` (dtype-generic).

    For the reacting models the equilibrium/frozen solve is the dominant cost and
    yields ``(T, rho, c, W)`` in one shot.  The kinetic-energy coupling
    ``h = h_t - u^2/2`` (``u = mdot/(rho A)``) is carried by an outer bracketed root
    on the static enthalpy wrapped around that solve (``eq_*_state_ke_*``), mirroring
    the perfect gas's density root, so every model recovers the exact static state.

    ``marker`` is the transported burnt-marker scalar; only the ``EQ_MARKER`` model reads
    it, to gate the frozen/equilibrium blend (the other models ignore it).

    ``nj_io`` is the per-edge equilibrium warm-start cache (a moles vector): the
    ``EQ_KERNEL`` / ``EQ_MARKER`` solve seeds from it and writes the converged composition
    back, so a nearby re-solve (the next Newton iterate or a complex-step Jacobian column)
    converges fast.  It only hints the solver -- the equilibrium is unique -- and is
    ignored (any size) by the perfect-gas and frozen branches.
    """
    if model_id == PERFECT_GAS:
        rho, h = closure_solve(model_id, tf, ti, mdot, p, ht, Z_el, area)
        T, _rho2, c, W = thermo_state(model_id, tf, ti, Z_el, h, p)
    elif model_id == EQ_KERNEL:
        # reacting (equilibrium): warm-started solve with the outer kinetic-energy root
        T, rho, c, W = eq_kernel_state_ke_warm(tf, ti, Z_el, mdot, p, ht, area, nj_io)
    elif model_id == EQ_MARKER:
        # reacting (marker-gated): blend frozen (b=0) and equilibrium (b=1), each with its own KE root
        T, rho, c, W = eq_marker_state_ke_warm(tf, ti, Z_el, marker, mdot, p, ht, area, nj_io)
    else:
        # reacting (frozen): frozen solve with the outer kinetic-energy root
        T, rho, c, W = eq_frozen_state_ke(tf, ti, Z_el, mdot, p, ht, area, nj_io)
    u = mdot / (rho * area)
    M = u / c
    pt = thermo_total_pressure(model_id, tf, M, p, T, c, W)
    out[ES_MDOT] = mdot
    out[ES_P] = p
    out[ES_HT] = ht
    out[ES_RHO] = rho
    out[ES_U] = u
    out[ES_T] = T
    out[ES_C] = c
    out[ES_M] = M
    out[ES_PT] = pt
    out[ES_AREA] = area
    # Mixture molar mass and specific heat, derived from the recovered state (cheap,
    # complex-step-safe).  gamma = c^2 rho / p is the local isentropic exponent (the
    # equilibrium one on an EQ_KERNEL edge, the frozen one on an EQ_FROZEN edge, since
    # ``c`` carries that flavor); cp = c^2 / (T (gamma - 1)) = gamma R / (gamma - 1) is the
    # specific heat consistent with it (exact for a perfect gas).  W = rho R_u T / p.
    gamma = c * c * rho / p
    out[ES_W] = rho * RU * T / p
    out[ES_CP] = c * c / (T * (gamma - 1.0))


@njit(cache=True)
def recover_all(edge_model, tf, ti, x, area, n_elem, marker_row, est, nj_cache):
    """Recover every edge state into ``est[NS_EST, E]`` (per-edge thermo model).

    ``edge_model[e]`` selects the thermo model for edge ``e`` -- so a frozen
    (unburnt) approach edge and an equilibrium (burnt) edge can coexist in one
    network, with the flame element bridging them.  ``marker_row`` is the band-1 row of
    the transported burnt marker (``< 0`` when the network carries none -- perfect gas, or
    a hard-closure reacting network); it gates the ``EQ_MARKER`` blend per edge.  ``nj_cache``
    is the per-edge equilibrium warm-start cache, shape ``(E, Ns + 1)`` (moles + temperature,
    or ``(E, 0)`` to disable); row ``e`` seeds and stores edge ``e``'s converged state.
    """
    E = x.shape[1]
    has_marker = marker_row >= 0
    for e in range(E):
        Z_el = x[3 : 3 + n_elem, e]
        marker = x[marker_row, e] if has_marker else x[2, e] * 0.0
        recover_edge(edge_model[e], tf, ti, x[0, e], x[1, e], x[2, e], area[e], Z_el, marker, est[:, e], nj_cache[e])


def enrich_caloric(edge_model, tf, ti, x, est, n_elem, marker_row):
    """Fill the caloric-derivative columns ``ES_DHDRHO``/``ES_DHDP`` of ``est`` per edge.

    The perturbation network needs each edge's caloric coupling ``dh_t = a d_rho + u d_u +
    b d_p`` with ``a = (dh/drho)_p`` and ``b = (dh/dp)_rho`` -- the two model-dependent
    partials of static enthalpy (``m = u`` is kinematic and already ``ES_U``).  Each edge's
    own thermo model supplies them: a perfect-gas edge from the closed form ``(-K p/rho^2,
    K/rho)`` with ``K = cp/R``; a reacting edge from a complex step of its converged closure
    at the static enthalpy ``h = h_t - u^2/2`` (the state ``J_alg`` linearizes about), the
    equilibrium composition free to shift.

    Runs on a **real** (converged) state: it complex-steps ``thermo_state`` internally, so it
    cannot be folded into the already-seeded residual/Jacobian recovery.  ``est`` must already
    carry the recovered state (:func:`recover_all`); this only writes the two caloric slots.

    Parameters
    ----------
    edge_model : ndarray
        Per-edge thermo-model id (as :func:`recover_all`).
    tf, ti : ndarray
        Thermo float/int config.
    x : ndarray
        Converged solve state, shape ``(n_solve, E)``.
    est : ndarray
        Edge-state table ``(NS_EST, E)`` from :func:`recover_all`; modified in place.
    n_elem : int
        Number of transported elemental scalars.
    marker_row : int
        Band-1 row of the burnt marker (``< 0`` if none); gates the ``EQ_MARKER`` blend.
    """
    K = float(tf[0]) / float(tf[1])  # perfect-gas cp/R (that model's own caloric constant)
    E = est.shape[1]
    has_marker = marker_row >= 0
    d = 1e-30
    for e in range(E):
        mid = int(edge_model[e])
        rho = float(est[ES_RHO, e])
        u = float(est[ES_U, e])
        p = float(est[ES_P, e])
        if mid == PERFECT_GAS:
            est[ES_DHDRHO, e] = -K * p / rho**2
            est[ES_DHDP, e] = K / rho
            continue
        # A marker-gated edge is bimodal at convergence (marker ~ 0 or 1); its caloric
        # derivatives equal the dominant pure closure's (the gate depends on the marker, not
        # on rho/h/p).  Map EQ_MARKER to that closure for the step.
        eff = mid
        if mid == EQ_MARKER:
            b_mark = float(x[marker_row, e]) if has_marker else 0.0
            eff = EQ_KERNEL if b_mark >= 0.5 else EQ_FROZEN
        # reacting: rho = rho(xi, h, p) at the *static* enthalpy h = h_t - u^2/2; invert
        # (dh/drho)_p, (dh/dp)_rho there by complex step of the converged closure.
        xi = np.ascontiguousarray(x[3 : 3 + n_elem, e]).astype(np.complex128)
        h_static = complex(float(x[2, e]) - 0.5 * u * u)
        drho_dh = thermo_state(eff, tf, ti, xi, h_static + 1j * d, complex(p))[1].imag / d
        drho_dp = thermo_state(eff, tf, ti, xi, h_static + 0j, p + 1j * d)[1].imag / d
        est[ES_DHDRHO, e] = 1.0 / drho_dh
        est[ES_DHDP, e] = -drho_dp / drho_dh
