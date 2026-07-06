"""Flame elements: the perfect-gas heat-release flame and the reacting
equilibrium flame.

The heat-release flame conserves mass and total pressure while raising the
through-flow's total enthalpy by ``Q_dot / mdot``.  The equilibrium flame is its
reacting counterpart: frozen unburnt reactants enter, equilibrium products leave,
with the flame temperature emerging from an HP-equilibrium solve at the conserved
``(Z, h_t, p)`` -- "ignition" by a per-edge frozen->equilibrium closure switch.
"""

import os

import numpy as np
import pytest

from nefes.assembly.assemble import jacobian, jacobian_dense
from nefes.assembly.recover import ES_CP, ES_HT, ES_MDOT, ES_P, ES_RHO, ES_T, ES_U
from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium, perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
A = 0.05

MECH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")


def _flame_network(mdot, Tt, Qdot, p_out=1.0e5):
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [cat.mass_flow_inlet(mdot, Tt), cat.heat_release_flame(Qdot), cat.pressure_outlet(p_out)]
    edges = [(0, 1, A), (1, 2, A)]
    return build_problem(cfg, els, edges, mdot_ref=mdot, p_ref=p_out, h_ref=CP * Tt)


@pytest.mark.parametrize("Qdot", [2.5e5, 1.0e6, 3.0e6])
def test_heat_release_flame_energy_balance(Qdot):
    mdot, Tt = 10.0, 300.0
    prob = _flame_network(mdot, Tt, Qdot)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)

    # mass is conserved across the flame
    assert est[ES_MDOT, 0] == pytest.approx(mdot, rel=1e-6)
    assert est[ES_MDOT, 1] == pytest.approx(mdot, rel=1e-6)

    # the inlet edge carries h_t = cp*Tt; the flame adds Qdot/mdot
    assert est[ES_HT, 0] == pytest.approx(CP * Tt, rel=1e-5)
    assert est[ES_HT, 1] - est[ES_HT, 0] == pytest.approx(Qdot / mdot, rel=1e-5)

    # total temperature rise and a genuine static-temperature rise (mean flow moves)
    Tt_out = est[ES_HT, 1] / CP
    assert Tt_out == pytest.approx(Tt + Qdot / (mdot * CP), rel=1e-5)
    assert est[ES_T, 1] > est[ES_T, 0] + 1.0

    # constant-area momentum balance: rho u^2 + p is continuous across the flame
    imp0 = est[ES_RHO, 0] * est[ES_U, 0] ** 2 + est[ES_P, 0]
    imp1 = est[ES_RHO, 1] * est[ES_U, 1] ** 2 + est[ES_P, 1]
    assert imp1 == pytest.approx(imp0, rel=1e-6)
    # heat addition at constant area drops the static pressure (Rayleigh)
    assert est[ES_P, 1] < est[ES_P, 0]


def test_zero_heat_release_is_a_passthrough():
    """Q_dot = 0 must recover a plain (duct-like) passthrough: no h_t jump."""
    mdot, Tt = 12.0, 300.0
    prob = _flame_network(mdot, Tt, 0.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    assert est[ES_HT, 1] == pytest.approx(est[ES_HT, 0], rel=1e-10)
    assert est[ES_T, 1] == pytest.approx(est[ES_T, 0], rel=1e-8)


# --------------------------------------------------------------------------
# Reacting equilibrium flame
# --------------------------------------------------------------------------


def _h2_air_reactant():
    """Stoichiometric H2/air reactant: (Thermo, species mass fractions, elemental Z)."""
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_cantera(MECH_PATH)
    gas = Thermo(lib)
    idx = lib.species_index
    moles = np.zeros(lib.n_species)
    moles[idx["H2"]] = 1.0
    moles[idx["O2"]] = 0.5
    moles[idx["N2"]] = 0.5 * 3.76
    Y = moles * lib.molar_masses
    Y /= Y.sum()
    return gas, Y, gas.elemental_mass_fractions(Y)


def _equilibrium_flame_network(mdot=1.0, Aedge=0.05, Tin=300.0, p=101325.0):
    th, Y, Z = _h2_air_reactant()
    h_react = th.enthalpy_mass(Y, Tin)
    # stoichiometric H2/air as a named species mixture (mole basis); a single
    # premixed feed stream -> one transported mixture fraction, discovered at build
    fuel_air = {"H2": 1.0, "O2": 0.5, "N2": 0.5 * 3.76}
    cfg = equilibrium(th.mech)
    els = [
        cat.mass_flow_inlet(mdot, Tin, composition=fuel_air, basis="mole", name="fuel-air"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=Tin, composition=fuel_air, basis="mole", name="out"),
    ]
    edges = [(0, 1, Aedge), (1, 2, Aedge)]
    edge_models = [EQ_FROZEN, EQ_KERNEL]  # unburnt approach, burnt products
    prob = build_problem(cfg, els, edges, mdot_ref=mdot, p_ref=p, h_ref=h_react, edge_models=edge_models)
    return th, Z, h_react, p, prob


def test_equilibrium_flame_ignites():
    th, Z, h_react, p, prob = _equilibrium_flame_network()
    res = solve(prob)  # auto-seeded: no hand-built per-edge guess
    assert res.converged

    est = states_table(prob, res.x)
    # mass conserved; momentum (rho u^2 + p) conserved; total enthalpy conserved
    assert est[ES_MDOT, 0] == pytest.approx(est[ES_MDOT, 1], rel=1e-9)
    imp0 = est[ES_RHO, 0] * est[ES_U, 0] ** 2 + est[ES_P, 0]
    imp1 = est[ES_RHO, 1] * est[ES_U, 1] ** 2 + est[ES_P, 1]
    assert imp1 == pytest.approx(imp0, rel=1e-7)
    assert est[ES_HT, 0] == pytest.approx(est[ES_HT, 1], rel=1e-9)

    # the approach edge is unburnt, the product edge is burnt (hot).  With the
    # kinetic-energy coupling the recovered T is the *static* temperature: the cold
    # approach sits just below its 300 K stagnation value by u^2/(2 cp).
    T0_static = 300.0 - 0.5 * est[ES_U, 0] ** 2 / est[ES_CP, 0]
    assert est[ES_T, 0] == pytest.approx(T0_static, abs=1e-2)
    assert est[ES_T, 0] < 300.0  # KE drop present
    assert est[ES_T, 1] > 2000.0
    # the burnt *static* temperature matches a standalone HP-equilibrium solve (== Cantera)
    # at the *static* enthalpy h = h_t - u^2/2 (the KE-coupled closure).
    h_static_1 = est[ES_HT, 1] - 0.5 * est[ES_U, 1] ** 2
    ref = th.equilibrate_HP(Z, h_static_1, est[ES_P, 1])
    assert ref.converged
    assert est[ES_T, 1] == pytest.approx(ref.T, rel=1e-4)
    # dilatation: the gas expands strongly across the flame
    assert est[ES_RHO, 0] / est[ES_RHO, 1] > 5.0


def test_equilibrium_flame_jacobian_pattern():
    """Sparse complex-step Jacobian must match the dense reference for the reacting
    backend -- validating the CSC pattern + fill through the equilibrium closure."""
    th, Z, h_react, p, prob = _equilibrium_flame_network()
    res = solve(prob)
    assert res.converged
    eps, eps_fb = 1e-4, 1e-5
    Js = jacobian(prob, res.x, eps, eps_fb).toarray()
    Jd = jacobian_dense(prob, res.x, eps, eps_fb)
    assert Js.shape == Jd.shape
    np.testing.assert_allclose(Js, Jd, atol=1e-6, rtol=1e-6)
