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
from .thermo.api import thermo_state, thermo_total_pressure, PERFECT_GAS, EQ_KERNEL
from .thermo.equilibrium import eq_kernel_state_warm

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
NS_EST = 10


@njit(cache=True)
def recover_edge(model_id, tf, ti, mdot, p, ht, area, Z_el, out, nj_io):
    """Recover one edge's full state into ``out[0:NS_EST]`` (dtype-generic).

    For the reacting models the equilibrium/frozen solve is the dominant cost and
    yields ``(T, rho, c, W)`` in one shot; the closure's static density is the same
    solve, so it is folded in here (``h = h_t``, the MVP kinetic-energy drop) instead
    of running the equilibrium a second time through ``closure_solve``.  The perfect
    gas keeps its two distinct cheap steps (the density root-find carries the exact
    ``h = h_t - u^2/2`` kinetic-energy coupling).

    ``nj_io`` is the per-edge equilibrium warm-start cache (a moles vector): the
    ``EQ_KERNEL`` solve seeds from it and writes the converged composition back, so a
    nearby re-solve (the next Newton iterate or a complex-step Jacobian column)
    converges fast.  It only hints the solver -- the equilibrium is unique -- and is
    ignored (any size) by the perfect-gas and frozen branches.
    """
    if model_id == PERFECT_GAS:
        rho, h = closure_solve(model_id, tf, ti, mdot, p, ht, Z_el, area)
        T, _rho2, c, W = thermo_state(model_id, tf, ti, Z_el, h, p)
    elif model_id == EQ_KERNEL:
        # reacting (equilibrium): one warm-started solve gives rho, T, c, W; KE dropped (h = h_t)
        T, rho, c, W = eq_kernel_state_warm(tf, ti, Z_el, ht, p, nj_io)
    else:
        # reacting (frozen): one solve gives rho, T, c, W; KE dropped (h = h_t)
        T, rho, c, W = thermo_state(model_id, tf, ti, Z_el, ht, p)
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


@njit(cache=True)
def recover_all(edge_model, tf, ti, x, area, n_elem, est, nj_cache):
    """Recover every edge state into ``est[NS_EST, E]`` (per-edge thermo model).

    ``edge_model[e]`` selects the thermo model for edge ``e`` -- so a frozen
    (unburnt) approach edge and an equilibrium (burnt) edge can coexist in one
    network, with the flame element bridging them.  ``nj_cache`` is the per-edge
    equilibrium warm-start cache, shape ``(E, Ns + 1)`` (moles + temperature, or
    ``(E, 0)`` to disable); row ``e`` seeds and stores edge ``e``'s converged state.
    """
    E = x.shape[1]
    for e in range(E):
        Z_el = x[3 : 3 + n_elem, e]
        recover_edge(edge_model[e], tf, ti, x[0, e], x[1, e], x[2, e], area[e], Z_el, est[:, e], nj_cache[e])
