"""Fuel-flow modulation: the dynamic mass source and the equivalence-ratio instability.

A fuel injector (:func:`~fns.elements.catalog.mass_source`) carrying a dynamic ``S(omega)``
on its injected mass flow models a fluctuating fuel feed.  The perturbation stamp must
modulate every source term the element carries -- mass, momentum (via the injection
velocity), the injected enthalpy, **and** the injected composition -- so the fuel pulse
sheds a convected mixture-fraction wave.  That wave convects to a downstream (static)
reacting flame, which burns it into an unsteady heat release: the classic *equivalence-ratio*
combustion-instability mechanism.

The tests anchor it:

1. The stamp factors equal ``mdot_src * dR/d(mdot_src)`` on every modulated row (mass,
   momentum, energy, each composition scalar) -- a finite-difference identity.
2. A forced fuel fluctuation generates a mixture-fraction wave at the injector (none
   upstream) that convects losslessly to the flame.
3. With a total-pressure (air) inlet and a mass-flow outlet (both inherited), the fuel-flow
   coupling destabilizes a mode -- and only because the convected composition wave is
   retained (``isentropic`` freezes it -> stable).

Run in the ``fns`` env (numba); needs the thermolib H2/air data.
"""

import os
import warnings

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.elements.dynamic_source import mass_flow_response, n_tau, n_tau_lowpass
from fns.perturbation.boundary_bc import PerturbationBC
from fns.perturbation import open_loop_response, forced_response
from fns.perturbation.stamps import build_source_stamps
from fns.perturbation.characteristics import edge_caloric
from fns.assemble import residual
from fns.solver import solve
from fns.solver.control import states_table
from fns.thermo.api import EQ_FROZEN, EQ_KERNEL
from fns.thermo.configure import equilibrium
from fns.derive import ES_T, ES_U

AREA = 0.02
MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")
AIR = {"O2": 0.21, "N2": 0.79}
# node 0 inlet(air) | 1 duct | 2 fuel injector | 3 duct (convective lag) | 4 flame | 5 duct | 6 outlet
# edge 0 air | 1 | 2 injector-out | 3 flame-approach | 4 flame-out | 5
E_AIR, E_INJ_OUT, E_APPROACH, E_FLAME_OUT = 0, 2, 3, 4
N_INJECTOR = 2


def _air_enthalpy_datum():
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_native(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    Y[idx["O2"]], Y[idx["N2"]] = 0.21, 0.79
    Y /= Y.sum()
    return gas, gas.enthalpy_mass(Y, 300.0)


def _rig(La=0.4, Lb=0.4, Lc=0.5, mdot_air=0.4, mdot_fuel=0.006, pt=1.2e5, ds=None, inlet_excite=False):
    """Air -> duct -> H2 injector (optional dynamic fuel feed) -> duct -> reacting flame -> duct -> outlet."""
    gas, h_air = _air_enthalpy_datum()
    inlet_bc = PerturbationBC.anechoic(driven=("acoustic",)) if inlet_excite else PerturbationBC.inherit()
    els = [
        cat.total_pressure_inlet(pt, 300.0, composition=AIR, basis="mole", perturbation_bc=inlet_bc),
        cat.duct(La),
        cat.mass_source(mdot_fuel, 300.0, composition={"H2": 1.0}, basis="mole", dynamic_source=ds),
        cat.duct(Lb),
        cat.equilibrium_flame(),
        cat.duct(Lc),
        cat.mass_flow_outlet(mdot_air + mdot_fuel, perturbation_bc=PerturbationBC.inherit()),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA), (4, 5, AREA), (5, 6, AREA)]
    edge_models = [EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL]
    prob = cat.build_problem(
        equilibrium(gas.mech), els, edges, mdot_ref=mdot_air, p_ref=1e5, h_ref=h_air, edge_models=edge_models
    )
    return prob


def _converged(prob):
    res = solve(prob)
    assert res.converged, "mean solve failed"
    return res.x


# --------------------------------------------------------------------------
# 1. The stamp linearizes every injected source term (finite-difference identity)
# --------------------------------------------------------------------------


def test_fuel_stamp_matches_finite_difference():
    """Each modulated row's factor equals ``mdot_src * dR/d(mdot_src)`` (mass, momentum, energy, composition).

    The injected mass flow is bumped as a *static* parameter and the residual re-evaluated at
    the same frozen state, giving ``dR/d(mdot_src)`` directly; the dynamic-source stamp must
    reproduce it on every row it writes -- proof the fuel pulse perturbs all of the injector's
    source terms, not just mass + momentum.
    """
    mdot_fuel = 0.006
    ds = mass_flow_response(n_tau(0.7, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(mdot_fuel=mdot_fuel, ds=ds)
    x = _converged(prob)

    cals = edge_caloric(prob, x)
    K = float(prob.tf[0]) / float(prob.tf[1])
    stamps, _ = build_source_stamps(prob, x, K, cals=cals)
    inj = next(s for s in stamps if s.node == N_INJECTOR)
    assert len(inj.rows) >= 4  # mass, momentum, energy, and at least one composition scalar

    eps = 1e-4 * prob.var_scale[0]
    R0 = residual(prob, x, eps, 1e-6, 0.0)
    R1 = residual(_rig(mdot_fuel=mdot_fuel + 1e-6), x, eps, 1e-6, 0.0)
    dR = (R1 - R0) / 1e-6  # dR/d(mdot_src) at fixed state
    for r, fac in zip(inj.rows, inj.factors):
        assert fac / mdot_fuel == pytest.approx(dR[r], abs=1e-3, rel=1e-3)


# --------------------------------------------------------------------------
# 2. A fuel pulse sheds a convected mixture-fraction wave
# --------------------------------------------------------------------------


def test_fuel_pulse_generates_a_convected_mixture_wave():
    """A fuel-fraction wave appears at the injector (none upstream) and convects to the flame.

    Forcing the inlet drives the velocity-coupled fuel feed; the injected-stream fraction
    fluctuates only *downstream* of the injector and rides the convected wave to the flame --
    magnitude conserved, phase = the convective lag ``e^{-i omega L/u}`` (the equivalence-ratio
    transport that sets the instability's phase).
    """
    ds = mass_flow_response(n_tau(1.0, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds, inlet_excite=True)
    x = _converged(prob)
    est = states_table(prob, x)
    assert est[ES_T, E_FLAME_OUT] > 1200.0  # flame ignited

    ns = int(prob.n_solve)
    s_fuel = ns - 1  # last composition scalar = the injected ("source") stream fraction
    Lb, u = 0.4, float(est[ES_U, E_INJ_OUT])  # the injector -> flame duct and its mean speed
    freqs = np.array([150.0, 280.0, 400.0])
    omega = 2.0 * np.pi * freqs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forced_response(prob, x, freqs, isentropic=False)
    z_air = fr.X[:, ns * E_AIR + s_fuel]
    z_inj = fr.X[:, ns * E_INJ_OUT + s_fuel]
    z_flame = fr.X[:, ns * E_APPROACH + s_fuel]
    assert np.all(np.abs(z_air) < 1e-12)  # no fuel-fraction fluctuation upstream of the injector
    assert np.all(np.abs(z_inj) > 1e-4)  # the injector sheds the wave
    assert np.allclose(np.abs(z_inj), np.abs(z_flame), rtol=1e-6)  # convection is lossless
    assert np.allclose(z_flame, z_inj * np.exp(-1j * omega * Lb / u), rtol=1e-6)  # convective phase lag


def test_forced_response_surfaces_scalar_waves():
    """``ForcedResponse.waves()`` returns the convected scalar wave alongside ``(f, g, h)``.

    Wave-family parity for reacting scalars: the per-edge wave vector now carries one entry per
    transported scalar (named by ``prob.scalar_names``), surfaced directly from the network
    column -- the scalar is already its own convected wave -- so the same field accessor serves
    entropy and the scalars, no hand-indexing of ``X`` required.
    """
    ds = mass_flow_response(n_tau(1.0, 2e-3), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds, inlet_excite=True)
    x = _converged(prob)
    ns = int(prob.n_solve)
    assert ns > 3 and len(prob.scalar_names) == ns - 3  # genuinely reacting (scalars present)
    freqs = np.array([150.0, 280.0, 400.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forced_response(prob, x, freqs, isentropic=False)
    # as many waves as unknowns, labelled (f, g, h) then the feed-stream names
    assert fr.n_char == ns
    assert fr.wave_labels[:3] == ("f", "g", "h")
    assert tuple(fr.wave_labels[3:]) == tuple(prob.scalar_names)
    # the surfaced scalar wave is exactly the raw network column (identity), for every scalar/edge
    for s in range(3, ns):
        for e in (E_AIR, E_INJ_OUT, E_APPROACH):
            assert np.allclose(fr.waves(e)[:, s], fr.X[:, ns * e + s])
    # genuinely nonzero (the fuel pulse sheds a mixture-fraction wave) and acoustics untouched
    assert np.max(np.abs(fr.waves(E_INJ_OUT)[:, ns - 1])) > 1e-4
    assert np.allclose(fr.reflection_at(E_AIR), fr.waves(E_AIR)[:, 1] / fr.waves(E_AIR)[:, 0])


# --------------------------------------------------------------------------
# 3. The equivalence-ratio instability (fuel flow -> heat release -> acoustics)
# --------------------------------------------------------------------------


def test_fuel_flow_coupling_destabilizes_a_mode():
    """The fuel-flow FTF drives a mode unstable -- and only with the convected composition wave.

    Nyquist on the fuel-injector return ratio: keeping the convected mixture wave
    (``isentropic=False``) the fuel-flow coupling encircles the critical point (an unstable
    mode); freezing it (``isentropic=True``) removes the convective lag and the encirclement,
    so the equivalence-ratio transport is the essential link.
    """
    ds = mass_flow_response(n_tau_lowpass(2.0, 1e-3, 400.0), ref_edge=E_AIR, quantity="u")
    prob = _rig(ds=ds)
    x = _converged(prob)
    freqs = np.linspace(0.0, 1400.0, 900)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = open_loop_response(prob, x, freqs, isentropic=False)
        iso = open_loop_response(prob, x, freqs, isentropic=True)
    assert full.closed and iso.closed
    assert full.n_unstable >= 1  # the fuel-flow coupling destabilizes a mode
    assert iso.n_unstable == 0  # frozen composition wave: no equivalence-ratio instability
    assert any(150.0 < c["freq_hz"] < 700.0 for c in full.crossings(tol=0.35))
