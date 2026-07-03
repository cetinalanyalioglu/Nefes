"""The generic mass-source element (e.g. a fuel injector).

A 2-port inline injection that adds mass, momentum and energy (and composition,
for the reacting model) with the proper source terms -- conserving each quantity
across the element.  Ignition is NOT its job: a downstream flame burns the mixture
the source has prepared.
"""

import os

import numpy as np
import pytest

from nefes.assembly.derive import ES_HT, ES_MDOT, ES_P, ES_RHO, ES_T, ES_U
from nefes.elements import catalog as cat
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium, perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
A = 0.1

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data")
THERMO_INP = os.path.join(DATA, "thermo.inp")


# --------------------------------------------------------------------------
# Perfect-gas mass source: mass / energy / momentum source balances
# --------------------------------------------------------------------------
def test_perfect_gas_mass_source_balances():
    mdot_in, Tt_in = 10.0, 300.0
    mdot_src, Tt_src = 2.0, 600.0
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [
        cat.mass_flow_inlet(mdot_in, Tt_in),
        cat.mass_source(mdot_src, Tt_src, None, u_inj=0.0, name="src"),
        cat.pressure_outlet(1.0e5),
    ]
    prob = cat.build_problem(cfg, els, [(0, 1, A), (1, 2, A)], mdot_ref=mdot_in, p_ref=1.0e5, h_ref=CP * Tt_in)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)

    # mass: the outflow exceeds the inflow by the injected mass-flow
    assert est[ES_MDOT, 0] == pytest.approx(mdot_in, rel=1e-6)
    assert est[ES_MDOT, 1] == pytest.approx(mdot_in + mdot_src, rel=1e-6)

    # energy: mass-weighted enthalpy mixing -> the mixed total temperature
    Tt_mix = (mdot_in * Tt_in + mdot_src * Tt_src) / (mdot_in + mdot_src)
    assert est[ES_HT, 1] / CP == pytest.approx(Tt_mix, rel=1e-6)

    # momentum: u_inj = 0 (normal injection) keeps the momentum flux rho u^2 + p
    # continuous across the constant-area element
    imp0 = est[ES_RHO, 0] * est[ES_U, 0] ** 2 + est[ES_P, 0]
    imp1 = est[ES_RHO, 1] * est[ES_U, 1] ** 2 + est[ES_P, 1]
    assert imp1 == pytest.approx(imp0, rel=1e-6)


def test_perfect_gas_mass_source_injection_momentum():
    """A nonzero injection velocity adds axial momentum: rho u^2 + p is no longer
    continuous, the jump equals the injected momentum flux per area."""
    mdot_in, mdot_src, u_inj = 10.0, 3.0, 120.0
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [
        cat.mass_flow_inlet(mdot_in, 300.0),
        cat.mass_source(mdot_src, 300.0, None, u_inj=u_inj, name="src"),
        cat.pressure_outlet(1.0e5),
    ]
    prob = cat.build_problem(cfg, els, [(0, 1, A), (1, 2, A)], mdot_ref=mdot_in, p_ref=1.0e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    imp0 = est[ES_RHO, 0] * est[ES_U, 0] ** 2 + est[ES_P, 0]
    imp1 = est[ES_RHO, 1] * est[ES_U, 1] ** 2 + est[ES_P, 1]
    # downstream momentum flux = upstream + injected momentum per area
    assert imp1 - imp0 == pytest.approx(mdot_src * u_inj / A, rel=1e-5)


# --------------------------------------------------------------------------
# Reacting: fuel injected into air, then burnt by a downstream flame
# --------------------------------------------------------------------------
def _ch4_air_lib():
    from thermolib import ThermoInp

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    species = ["O2", "N2", "CH4", "CO2", "H2O", "CO", "OH", "H", "O", "NO", "H2"]
    return ThermoInp(THERMO_INP).library(species)


def _fuel_injection_network(mdot_air=1.0, mdot_fuel=0.05, Tin=300.0, p=1.0e5):
    """Build ``air -> fuel-source -> flame -> outlet``.

    The two feed streams (air at the inlet, CH4 at the source) are discovered at
    build time; the per-edge ``(mdot, h_t, xi)`` seed comes from the solver's graph
    propagation, so no hand-built initial guess is needed even though the air edge
    sits at ``h_t ~ +1.9e3`` J/kg and the CH4-laden / burnt edges at ``~ -2.2e5``
    J/kg (CH4's negative formation enthalpy).
    """
    from nefes.chem.composition import enthalpy_mass, resolve_composition
    from thermolib import Thermo

    lib = _ch4_air_lib()
    gas = Thermo(lib)
    air = {"O2": 0.21, "N2": 0.79}
    fuel = {"CH4": 1.0}
    Yair, _ = resolve_composition(lib, air, basis="mole")
    Yfuel, _ = resolve_composition(lib, fuel, basis="mole")
    h_air = enthalpy_mass(lib, Yair, Tin)
    # enthalpy is mass-specific, so the mixed total enthalpy is the mass-weighted mean
    h_mix = (mdot_air * h_air + mdot_fuel * enthalpy_mass(lib, Yfuel, Tin)) / (mdot_air + mdot_fuel)

    cfg = equilibrium(lib)
    els = [
        cat.mass_flow_inlet(mdot_air, Tin, composition=air, basis="mole", name="air"),
        cat.mass_source(mdot_fuel, Tin, composition=fuel, basis="mole", name="fuel"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=Tin, composition=air, basis="mole", name="out"),
    ]
    # air(0) -> fuel-source(1) -> flame(2) -> outlet(3)
    edges = [(0, 1, A), (1, 2, A), (2, 3, A)]
    edge_models = [EQ_FROZEN, EQ_FROZEN, EQ_KERNEL]  # air, air+fuel (unburnt), burnt
    prob = cat.build_problem(cfg, els, edges, mdot_ref=mdot_air, p_ref=p, h_ref=abs(h_mix), edge_models=edge_models)
    return lib, gas, prob, h_mix


def test_fuel_injection_then_burn():
    lib, gas, prob, h_mix = _fuel_injection_network()
    res = solve(prob)  # auto-seeded
    assert res.converged
    est = states_table(prob, res.x)

    mdot_air, mdot_fuel = 1.0, 0.05
    # mass added across the source; the flame conserves it
    assert est[ES_MDOT, 0] == pytest.approx(mdot_air, rel=1e-6)
    assert est[ES_MDOT, 1] == pytest.approx(mdot_air + mdot_fuel, rel=1e-6)
    assert est[ES_MDOT, 2] == pytest.approx(mdot_air + mdot_fuel, rel=1e-6)

    # the unburnt mixed edge is cold; the burnt edge is hot (the flame ignited it)
    assert est[ES_T, 1] == pytest.approx(300.0, abs=2.0)
    assert est[ES_T, 2] > 1800.0

    # h_t is conserved across both the source-mix and the flame
    assert est[ES_HT, 1] == pytest.approx(h_mix, rel=1e-5)
    assert est[ES_HT, 2] == pytest.approx(h_mix, rel=1e-5)

    # the burnt temperature matches a standalone HP-equilibrium solve at the
    # injected (air + CH4) mixture, the conserved h_t and the burnt-edge pressure
    from nefes.chem.composition import resolve_composition

    Yair, _ = resolve_composition(lib, {"O2": 0.21, "N2": 0.79}, basis="mole")
    Yfuel, _ = resolve_composition(lib, {"CH4": 1.0}, basis="mole")
    Ymix = (mdot_air * Yair + mdot_fuel * Yfuel) / (mdot_air + mdot_fuel)
    Zmix = gas.elemental_mass_fractions(Ymix)
    ref = gas.equilibrate_HP(Zmix, h_mix, est[ES_P, 2], T_guess=2000.0)
    assert ref.converged
    assert est[ES_T, 2] == pytest.approx(ref.T, rel=2e-3)


def test_fuel_injection_jacobian_pattern():
    """Sparse complex-step Jacobian == dense reference for a network with a mass
    source feeding a reacting flame (validates the source donor/residual fill)."""
    from nefes.assembly.assemble import jacobian, jacobian_dense

    lib, gas, prob, h_mix = _fuel_injection_network()
    res = solve(prob)
    assert res.converged
    eps, eps_fb = 1e-4, 1e-5
    Js = jacobian(prob, res.x, eps, eps_fb).toarray()
    Jd = jacobian_dense(prob, res.x, eps, eps_fb)
    np.testing.assert_allclose(Js, Jd, atol=1e-6, rtol=1e-6)
