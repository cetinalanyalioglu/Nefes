"""Acoustic-power diagnostics: energy-flux physics and the boundary energy budget.

Covers the primitives (group-speed transport, energy-neutral reflection bounds) and
the mode-level :func:`boundary_power` budget, whose net must share a sign with the
growth rate (the global energy law ``2 sigma E = sum boundary power``).
"""

import warnings

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.perturbation import (
    eigenmodes,
    acoustic_intensity,
    acoustic_energy_density,
    passive_reflection_bound,
    boundary_power,
)
from fns.perturbation.boundary_bc import PerturbationBC
from fns.solver import solve
from fns.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


# --------------------------------------------------------------------------
# Physics primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [0.0, 0.15, 0.4])
def test_energy_transports_at_group_speed(M):
    """A pure ``f`` wave carries energy at ``u + c``; a pure ``g`` wave at ``u - c``."""
    rho, c = 1.2, 340.0
    fwd = acoustic_intensity(rho, c, M, 1.3, 0.0) / acoustic_energy_density(rho, M, 1.3, 0.0)
    bwd = acoustic_intensity(rho, c, M, 0.0, 0.7) / acoustic_energy_density(rho, M, 0.0, 0.7)
    assert fwd == pytest.approx(c * (1.0 + M))
    assert bwd == pytest.approx(c * (M - 1.0))


def test_energy_density_nonnegative_and_phase_invariant():
    """Energy density is non-negative and depends only on wave magnitudes."""
    rho, M = 1.0, 0.25
    e0 = acoustic_energy_density(rho, M, 1.0, 0.5)
    e1 = acoustic_energy_density(rho, M, 1.0j, -0.5)  # same |f|, |g|, different phase
    assert e0 > 0.0
    assert e1 == pytest.approx(e0)


@pytest.mark.parametrize("M", [0.05, 0.2, 0.45])
def test_passive_bound_is_the_zero_flux_reflection(M):
    """At ``|R| =`` the passive bound, the net acoustic flux into the domain vanishes.

    Outlet bound ``(1+M)/(1-M)`` (incident ``f``, reflected ``g = R f``) and inlet
    bound ``(1-M)/(1+M)`` (incident ``g``, reflected ``f = R g``) are exactly the
    energy-neutral reflectors.
    """
    rho, c, amp = 1.2, 340.0, 0.9
    R_out = passive_reflection_bound(M, "outlet")
    # outlet: domain power in = -flux; flux must be zero
    assert acoustic_intensity(rho, c, M, amp, R_out * amp) == pytest.approx(0.0, abs=1e-9)
    R_in = passive_reflection_bound(M, "inlet")
    assert acoustic_intensity(rho, c, M, R_in * amp, amp) == pytest.approx(0.0, abs=1e-9)
    assert R_out == pytest.approx((1.0 + M) / (1.0 - M))
    assert R_in == pytest.approx((1.0 - M) / (1.0 + M))


def test_above_passive_bound_is_a_source():
    """An outlet/inlet reflecting harder than its neutral bound injects power."""
    rho, c, M = 1.2, 340.0, 0.3
    R = passive_reflection_bound(M, "outlet") * 1.1  # over-reflecting outlet
    flux = acoustic_intensity(rho, c, M, 1.0, R)  # incident f, reflected g = R f
    assert -flux > 0.0  # power into the domain (outlet: into-domain = -downstream flux)


def test_constant_mass_flow_is_the_neutral_outlet():
    """``constant_mass_flow`` reflection equals the energy-neutral outlet bound."""
    rho, c, M = 1.0, 340.0, 0.2
    R = PerturbationBC.constant_mass_flow().reflection_coefficient(0.0, rho, c, M)
    assert abs(R) == pytest.approx(passive_reflection_bound(M, "outlet"))


# --------------------------------------------------------------------------
# Mode-level boundary energy budget
# --------------------------------------------------------------------------


def _rig(inlet_R):
    """Plenum -> splitter -> choked nozzle + metered bleed (the notebook rig)."""
    els = [
        cat.total_pressure_inlet(2.5e5, 300.0, name="reservoir", perturbation_bc=PerturbationBC.reflection(inlet_R)),
        cat.duct(0.6, name="feed"),
        cat.splitter(name="manifold"),
        cat.duct(0.4, name="core"),
        cat.choked_nozzle_outlet(0.015, name="nozzle"),
        cat.duct(0.5, name="bleedpipe"),
        cat.mass_flow_outlet(2.0, name="bleed"),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.05), (2, 3, 0.03), (3, 4, 0.03), (2, 5, 0.02), (5, 6, 0.02)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=6.0, p_ref=1.5e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _spec(prob, res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eigenmodes(prob, res.x, freq_band=(50.0, 500.0), growth_band=(-150.0, 150.0), isentropic=True)


def test_boundary_power_sign_matches_growth_every_mode():
    """The net boundary power and the growth rate share a sign for every mode.

    This is the global energy law ``dE/dt = 2 sigma E = net boundary power`` (E > 0),
    independently cross-checking the contour eigenvalues.
    """
    for inlet_R in (0.8, 0.0):
        spec = _spec(*_rig(inlet_R))
        assert spec.n_modes >= 1
        for i in range(spec.n_modes):
            bp = spec.boundary_power(i)
            assert bp.sign_consistent
            assert np.sign(bp.net) * np.sign(bp.growth_rate) >= 0.0


def test_active_inlet_drives_instability_via_boundary_power():
    """The R=0.8 inlet is the source; the choked nozzle the sink; the bleed neutral."""
    spec = _spec(*_rig(0.8))
    unstable = [i for i in range(spec.n_modes) if spec.unstable[i]]
    assert unstable, "expected the over-reflecting inlet to drive a growing mode"
    bp = spec.boundary_power(unstable[0])
    assert bp.net > 0.0  # net energy fed into the domain
    by_name = {e["name"]: e for e in bp.entries}
    assert by_name["reservoir"]["power_in"] > 0.0  # active inlet feeds energy
    assert by_name["reservoir"]["reflection"] > by_name["reservoir"]["passive_bound"]
    assert by_name["nozzle"]["power_in"] < 0.0  # choked nozzle absorbs
    assert abs(by_name["bleed"]["fraction"]) < 1e-6  # mass-flow outlet is energy-neutral


def test_passive_inlet_is_globally_dissipative():
    """An anechoic inlet leaves only sinks -> every mode decays, net power < 0."""
    spec = _spec(*_rig(0.0))
    for i in range(spec.n_modes):
        bp = spec.boundary_power(i)
        assert bp.growth_rate < 0.0
        assert bp.net < 0.0
        assert by_name_power_in(bp, "reservoir") <= 0.0


def by_name_power_in(bp, name):
    return next(e["power_in"] for e in bp.entries if e["name"] == name)


def test_boundary_power_requires_terminals():
    """A bare result without terminals raises a helpful error."""
    spec = _spec(*_rig(0.0))
    spec.terminals = None
    with pytest.raises(ValueError, match="terminals"):
        boundary_power(spec, 0)
