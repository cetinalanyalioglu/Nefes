"""Closure adapter: the perfect-gas thermo boundary used by the solver.

Maps the band-1 unknowns ``(mdot, p, h_t)`` plus the edge ``area`` to the density
``rho`` and static enthalpy ``h``, resolving the kinetic-energy coupling
``h = h_t - u^2/2`` with ``u = mdot/(rho*A)``.

For a perfect gas this is a single density root -- the kinetic-energy term is already
folded into ``F(rho)``, so there is one root-find, not a nested fixed point -- and the
returned ``(rho, h)`` carries the IFT-spliced complex-step seed.  The reacting models
recover their static state directly through the equilibrium/frozen kernels
(``eq_*_state_ke_*`` in :mod:`nefes.thermo.edge_state`) and do not pass through this
adapter.
"""

from numba import njit

from ..thermo.api import PERFECT_GAS
from ..thermo.perfect_gas import pg_solve_density


@njit(cache=True)
def closure_solve(model_id, tf, ti, mdot, p, h_t, comp, area):
    """Return ``(rho, h)`` for a perfect-gas edge state.  Dtype-generic, complex-safe.

    ``comp`` is accepted for signature parity with the recovery kernels but is unused
    by the perfect gas.
    """
    if model_id == PERFECT_GAS:
        K = tf[0] / tf[1]  # cp / R
        m = mdot / area
        rho = pg_solve_density(m, p, h_t, K)
        u = m / rho
        h = h_t - 0.5 * u * u
        return rho, h
    raise ValueError("closure_solve handles only the perfect gas; reacting edges recover via eq_*_state_ke_*")
