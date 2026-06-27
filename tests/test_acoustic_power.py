"""Acoustic-power diagnostics: energy-flux physics and the boundary energy budget.

Covers the primitives (group-speed transport, energy-neutral reflection bounds) and
the mode-level :func:`boundary_power` budget, whose net must share a sign with the
growth rate (the global energy law ``2 sigma E = sum boundary power``).
"""

import warnings

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.elements.dynamic_source import n_tau_flame
from fns.perturbation import (
    eigenmodes,
    forced_response,
    acoustic_intensity,
    acoustic_energy_density,
    passive_reflection_bound,
    boundary_power,
    acoustic_flux_spectrum,
    compact_power_spectrum,
    duct_energy_spectrum,
    forced_power_balance,
    modal_energy_balance,
    find_terminals,
)
from fns.perturbation.boundary_bc import PerturbationBC
from fns.perturbation.modeshape import build_geometry
from fns.shell import Network
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


# --------------------------------------------------------------------------
# Forced-sweep power balance (a real-frequency drive, not a single eigenmode)
# --------------------------------------------------------------------------


def _driven_tube(n, tau, drive=True):
    """A Rijke tube inlet -> duct -> n-tau flame -> duct -> open end (driven for forced sweeps)."""
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=1.0e5, T_ref=300.0, mdot_ref=0.006)
    bc = PerturbationBC.mean_flow_open_end(driven=("acoustic",) if drive else ())
    i_in = net.add(cat.mass_flow_inlet(0.006, 300.0))
    i_cold = net.add(cat.duct(0.6))
    i_flame = net.add(cat.heat_release_flame(0.006 * CP * 400.0))
    i_hot = net.add(cat.duct(0.4))
    i_out = net.add(cat.pressure_outlet(1.0e5, perturbation_bc=bc))
    net.connect(i_in, i_cold, 0.01)
    ref = net.connect(i_cold, i_flame, 0.01)
    net.connect(i_flame, i_hot, 0.01)
    net.connect(i_hot, i_out, 0.01)
    net.set_dynamic_source(i_flame, n_tau_flame(n, tau, ref_edge=ref))
    sol = net.solve()
    assert sol.converged
    return sol


def test_intensity_and_density_are_array_safe():
    """The Myers primitives accept per-frequency wave arrays element-wise."""
    rho, c, M = 1.2, 340.0, 0.1
    f = np.array([1.0 + 0.5j, 0.3 - 0.2j, 0.0])
    g = np.array([0.2 + 0.1j, -0.4j, 0.7])
    intensity = acoustic_intensity(rho, c, M, f, g)
    density = acoustic_energy_density(rho, M, f, g)
    assert intensity.shape == f.shape and density.shape == f.shape
    for k in range(f.size):
        assert intensity[k] == pytest.approx(acoustic_intensity(rho, c, M, f[k], g[k]))
        assert density[k] == pytest.approx(acoustic_energy_density(rho, M, f[k], g[k]))
    assert np.all(density >= 0.0)  # energy density is non-negative for subsonic flow


def test_forced_power_balance_matches_its_components():
    """The one-call balance equals its building blocks, and stores non-negative energy."""
    sol = _driven_tube(0.0, 0.0)
    freqs = np.linspace(60.0, 320.0, 80)
    fr = forced_response(sol.problem, sol.x, freqs, isentropic=True)
    bal = forced_power_balance(fr, sol.problem)
    # energy is the duct integral; the interior generation is the flame node's flux jump
    energy = duct_energy_spectrum(fr, build_geometry(sol.problem).ducts)
    generation = compact_power_spectrum(fr, sol.problem, 2)  # node 2 is the flame

    # net boundary flux is the signed face flux summed over the terminals
    def _into_domain(t):
        face = acoustic_flux_spectrum(fr, t.edge)
        return face if t.at_tail else -face

    total_flux = sum(_into_domain(t) for t in find_terminals(sol.problem))
    assert np.array_equal(bal.freqs, freqs)
    assert np.allclose(bal.energy, energy)
    assert np.allclose(bal.generation, generation)
    assert np.allclose(bal.net_boundary_flux, total_flux)
    assert np.all(bal.energy >= 0.0)


def test_generation_is_the_flame_flux_jump():
    """The flame's produced power is the jump in acoustic flux across it -- what the balance sums."""
    sol = _driven_tube(0.8, 4.0e-3)
    freqs = np.linspace(80.0, 200.0, 60)
    fr = forced_response(sol.problem, sol.x, freqs, isentropic=True)
    flame = 2  # inlet, cold duct, FLAME, hot duct, outlet -> node 2
    jump = acoustic_flux_spectrum(fr, 2) - acoustic_flux_spectrum(fr, 1)  # downstream(out) - upstream(in)
    assert np.allclose(compact_power_spectrum(fr, sol.problem, flame), jump)
    assert np.allclose(forced_power_balance(fr, sol.problem).generation, jump)


def test_energy_budget_closes():
    """generation + net_boundary_flux ~ 0: a steady forced state neither stores nor loses energy.

    The reflecting ends carry essentially no net flux on their own, so whatever the flame exchanges
    with the field must cross the (driven) open end -- the two terms are equal and opposite to within
    the tiny numerical / mean-flow dissipation.
    """
    sol = _driven_tube(0.8, 4.0e-3)
    freqs = np.linspace(80.0, 200.0, 150)
    bal = forced_power_balance(forced_response(sol.problem, sol.x, freqs, isentropic=True), sol.problem)
    assert np.abs(bal.residual).max() < 1e-6 * np.abs(bal.generation).max()


def test_passive_flame_absorbs_while_active_flame_pumps():
    """The energy fingerprint of self-excitation, read off the flame's acoustic power production.

    The inert flame's mean density jump makes it a passive scatterer that only *absorbs* acoustic
    power (the drive feeds energy in), whereas the n-tau flame *produces* acoustic power near its
    unstable mode -- and since the real ends reflect, that excess can only leave through the driven
    open end (net boundary flux turns negative, the mirror of the generation).
    """
    freqs = np.linspace(80.0, 200.0, 150)
    sol_p, sol_a = _driven_tube(0.0, 0.0), _driven_tube(0.8, 4.0e-3)
    pas = forced_power_balance(forced_response(sol_p.problem, sol_p.x, freqs, isentropic=True), sol_p.problem)
    act = forced_power_balance(forced_response(sol_a.problem, sol_a.x, freqs, isentropic=True), sol_a.problem)
    # inert flame never produces acoustic power, and the passive boundaries only absorb (net flux in)
    assert pas.generation.max() <= 1e-6 * np.abs(pas.generation).max()
    assert pas.net_boundary_flux.min() >= -1e-6 * np.abs(pas.net_boundary_flux).max()
    # active flame pumps the field near its mode, and that power radiates out the driven end
    assert act.generation.max() > 0.0
    assert act.net_boundary_flux.min() < 0.0


def test_boundary_split_keeps_the_excitation_out_of_the_reflector_flux():
    """The reflectors carry ~no net flux; the excitation source mirrors the flame generation."""
    sol = _driven_tube(0.8, 4.0e-3)
    freqs = np.linspace(80.0, 200.0, 150)
    bal = forced_power_balance(forced_response(sol.problem, sol.x, freqs, isentropic=True), sol.problem)
    gscale = np.abs(bal.generation).max()
    # the (near) energy-neutral ends carry only their O(M) leak -- a few percent of the budget at most
    assert np.abs(bal.boundary_reflection).max() < 5e-2 * gscale
    # the drive sinks the flame's generation: boundary_source ~ -generation
    assert np.allclose(bal.boundary_source, -bal.generation, atol=5e-2 * gscale)
    # and the buckets reconstruct the total boundary flux exactly
    assert np.allclose(bal.net_boundary_flux, bal.boundary_reflection + bal.boundary_source)


def test_modal_energy_balance_recovers_growth_rate():
    """The node-wise energy budget reproduces each eigenmode's growth rate (sign and value)."""
    sol = _driven_tube(0.8, 4.0e-3, drive=False)  # eigenmodes use the passive (undriven) ends
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eigenmodes(sol.problem, sol.x, freq_band=(40.0, 320.0), growth_band=(-200.0, 200.0), isentropic=True)
    assert res.n_modes >= 1
    for m in range(res.n_modes):
        eb = res.energy_balance(m)
        assert eb.consistent
        assert eb.growth_rate_energy == pytest.approx(eb.growth_rate, rel=1e-3, abs=1e-2)
        # an unstable mode is the flame's generation trapped by the (near-neutral) ends
        if eb.growth_rate > 0.0:
            assert eb.generation > 0.0

    # the convenience function and the method agree
    direct = modal_energy_balance(res, 0)
    assert direct.growth_rate_energy == pytest.approx(res.energy_balance(0).growth_rate_energy)
