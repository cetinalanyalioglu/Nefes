"""Closure adapter: the AD-3 thermo boundary used by the solver.

Maps the band-1 unknowns ``(mdot, p, h_t, comp)`` plus the edge ``area`` to the
density ``rho`` and static enthalpy ``h``, resolving the kinetic-energy coupling
``h = h_t - u^2/2`` with ``u = mdot/(rho*A)``.  ``comp`` is the transported
composition vector -- the feed-stream **mixture fractions** ``xi`` for the
reacting model (the equilibrium kernels map them to elements / feed-species moles
internally); it is unused by the perfect gas.

For a perfect gas this collapses to the single density root (the kinetic-energy
term is already inside ``F(rho)``), so there is one root-find, not a nested fixed
point.  The reacting backends carry the same ``h = h_t - u^2/2`` coupling through an
outer bracketed root on the static enthalpy (``eq_*_state_ke_*``); both return
``(rho, h)`` with the IFT-spliced complex-step seed, keeping the boundary
backend-agnostic.
"""

import numpy as np
from numba import njit

from ..thermo.api import PERFECT_GAS, EQ_KERNEL, EQ_FROZEN
from ..thermo.equilibrium import eq_frozen_state_ke, eq_kernel_state_ke_warm
from ..thermo.perfect_gas import pg_solve_density


@njit(cache=True)
def closure_solve(model_id, tf, ti, mdot, p, h_t, comp, area):
    """Return ``(rho, h)`` for the edge state.  Dtype-generic, complex-safe.

    ``comp`` is the transported composition vector: the feed-stream mixture
    fractions ``xi`` for the reacting model (mapped to elements / feed-species moles
    inside the equilibrium kernels), ignored by the perfect gas.
    """
    if model_id == PERFECT_GAS:
        K = tf[0] / tf[1]  # cp / R
        m = mdot / area
        rho = pg_solve_density(m, p, h_t, K)
        u = m / rho
        h = h_t - 0.5 * u * u
        return rho, h
    if model_id == EQ_KERNEL:
        # equilibrium density at the KE-coupled static enthalpy (cold-start, no warm cache here)
        _T, rho, _c, _W = eq_kernel_state_ke_warm(tf, ti, comp, mdot, p, h_t, area, np.zeros(0))
        u = mdot / (rho * area)
        return rho, h_t - 0.5 * u * u
    if model_id == EQ_FROZEN:
        _T, rho, _c, _W = eq_frozen_state_ke(tf, ti, comp, mdot, p, h_t, area, np.zeros(0))
        u = mdot / (rho * area)
        return rho, h_t - 0.5 * u * u
    raise ValueError("unknown thermo model_id")
