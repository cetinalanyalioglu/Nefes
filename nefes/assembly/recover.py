"""Edge-state recovery: the refresh DAG ``unknowns -> closure -> thermo -> flow``.

``recover_edge`` maps one edge's band-1 unknowns ``(mdot, p, h_t [, Z_el])`` and
its ``area`` to the full recovered state used by element residuals.  It is
dtype-generic and complex-step-safe: on the seeded (complex) path every derived
quantity is recomputed inline, never read back from a cache (a stale real value
would drop the imaginary seed).

The recovered state is packed into a fixed-width "edge-state table" column
``est[:, e]`` with the slot layout below.
"""

from numba import njit

from .closure import closure_solve
from ..thermo.api import thermo_state, thermo_total_pressure, PERFECT_GAS, EQ_KERNEL, EQ_MARKER
from ..thermo.equilibrium import eq_kernel_state_ke_warm, eq_marker_state_ke_warm, eq_frozen_state_ke
from ..thermo._chem import RU

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
NS_EST = 12


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
    pt = thermo_total_pressure(model_id, tf, ti, Z_el, M, p, T, c, W)
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
