"""Closure adapter: the AD-3 thermo boundary used by the solver.

Maps the band-1 unknowns ``(mdot, p, h_t, comp)`` plus the edge ``area`` to the
density ``rho`` and static enthalpy ``h``, resolving the kinetic-energy coupling
``h = h_t - u^2/2`` with ``u = mdot/(rho*A)``.  ``comp`` is the transported
composition vector -- the feed-stream **mixture fractions** ``xi`` for the
reacting model (the equilibrium kernels map them to elements / feed-species moles
internally); it is unused by the perfect gas.

For a perfect gas this collapses to the single density root (the kinetic-energy
term is already inside ``F(rho)``), so there is one root-find, not a nested fixed
point.  A future opaque-thermo backend (equilibrium/table) implements the same
interface with an outer fixed point; both return ``(rho, h)`` with the
IFT-spliced complex-step seed, keeping the boundary backend-agnostic.
"""

from numba import njit

from .thermo.api import PERFECT_GAS, EQ_KERNEL, EQ_FROZEN
from .thermo.equilibrium import eq_frozen_state, eq_kernel_state
from .thermo.perfect_gas import pg_solve_density


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
        # MVP: drop the kinetic-energy coupling (h ~ h_t); O(M^2) at low Mach.
        # The static density is the equilibrium density at the transported h_t.
        _T, rho, _c, _W = eq_kernel_state(tf, ti, comp, h_t, p)
        return rho, h_t
    if model_id == EQ_FROZEN:
        _T, rho, _c, _W = eq_frozen_state(tf, ti, comp, h_t, p)
        return rho, h_t
    raise ValueError("unknown thermo model_id")
