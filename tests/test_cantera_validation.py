"""Validation oracle: compare the equilibrium engine to Cantera when available.

Every test here is skipped (not failed) if Cantera is absent, via the ``cantera``
fixture's ``importorskip``.
"""

import numpy as np
import pytest

from nefes.thermo import Mechanism, Thermo


@pytest.fixture
def setup(cantera):
    ct = cantera
    ctg = ct.Solution("h2o2.yaml")
    mech = Mechanism.from_cantera(ctg)
    gas = Thermo(mech)
    return ct, mech, gas, ctg


def test_species_thermo_exact(setup):
    ct, mech, gas, ctg = setup
    for T in (350.0, 800.0, 1500.0, 3000.0):
        ctg.TP = T, ctg.reference_pressure
        assert np.allclose(mech.cp_R(T), ctg.standard_cp_R, atol=1e-9)
        assert np.allclose(mech.h_RT(T), ctg.standard_enthalpies_RT, atol=1e-8)
        assert np.allclose(mech.s_R(T), ctg.standard_entropies_R, atol=1e-8)


def test_mixture_properties_exact(setup):
    ct, mech, gas, ctg = setup
    comp = {"H2": 0.05, "O2": 0.2, "H2O": 0.1, "N2": 0.65}
    T, p = 1250.0, 2e5
    ctg.TPY = T, p, comp
    Y = ctg.Y.copy()
    pr = gas.properties(Y, T, p)
    assert np.isclose(pr.rho, ctg.density, rtol=1e-12)
    assert np.isclose(pr.cp, ctg.cp_mass, rtol=1e-12)
    assert np.isclose(pr.h, ctg.enthalpy_mass, rtol=1e-10)
    assert np.isclose(pr.s, ctg.entropy_mass, rtol=1e-12)
    assert np.isclose(pr.gamma, ctg.cp_mass / ctg.cv_mass, rtol=1e-12)


def test_tp_equilibrium_composition(setup):
    ct, mech, gas, ctg = setup
    comp = {"H2": 2, "O2": 1, "N2": 3.76}
    T, p = 2500.0, 101325.0
    ctg.TPX = 300.0, p, comp
    Z = gas.elemental_mass_fractions(ctg.Y)
    ctg.TPX = T, p, comp
    ctg.equilibrate("TP")
    res = gas.equilibrate_TP(Z, T, p)
    assert np.allclose(res.X, ctg.X, atol=1e-7)


def test_hp_adiabatic_flame_temperature(setup):
    ct, mech, gas, ctg = setup
    comp = {"H2": 2, "O2": 1, "N2": 3.76}
    p = 101325.0
    ctg.TPX = 300.0, p, comp
    h0 = ctg.enthalpy_mass
    Z = gas.elemental_mass_fractions(ctg.Y)
    ctg.equilibrate("HP")
    res = gas.equilibrate_HP(Z, h0, p, T_guess=2000.0)
    assert np.isclose(res.T, ctg.T, rtol=1e-8)
    assert np.allclose(res.X, ctg.X, atol=1e-6)


def test_equilibrium_sound_speed(setup):
    ct, mech, gas, ctg = setup
    comp = {"H2": 2, "O2": 1, "N2": 3.76}
    p = 101325.0
    ctg.TPX = 2400.0, p, comp
    Z = gas.elemental_mass_fractions(ctg.Y)
    ctg.equilibrate("TP")
    s0, rho0, p0 = ctg.entropy_mass, ctg.density, ctg.P
    dp = p0 * 1e-5
    ctg.SP = s0, p0 + dp
    ctg.equilibrate("SP")
    a_eq_ct = np.sqrt(dp / (ctg.density - rho0))
    res = gas.equilibrate_TP(Z, 2400.0, p)
    assert np.isclose(res.a_equilibrium, a_eq_ct, rtol=1e-4)


def test_methane_air_flame_via_gri30(setup):
    # Carbon chemistry: import GRI-Mech 3.0 offline and validate HP flame.
    ct, _, _, _ = setup
    ctg = ct.Solution("gri30.yaml")
    mech = Mechanism.from_cantera(ctg)
    gas = Thermo(mech)
    p = 101325.0
    ctg.TPX = 300.0, p, {"CH4": 1, "O2": 2, "N2": 7.52}
    h0 = ctg.enthalpy_mass
    Z = gas.elemental_mass_fractions(ctg.Y)
    ctg.equilibrate("HP")
    res = gas.equilibrate_HP(Z, h0, p, T_guess=2000.0)
    assert res.converged
    assert np.isclose(res.T, ctg.T, rtol=2e-4)
    # Major products agree.
    for sp in ("CO2", "H2O", "CO", "O2", "N2"):
        j = mech.species_index[sp]
        assert np.isclose(res.X[j], ctg.X[ctg.species_index(sp)], atol=2e-3)


def test_kc_matches_cantera(setup):
    # Shared-Gibbs equilibrium constants match Cantera's Kc.
    ct, mech, gas, ctg = setup
    T = 1800.0
    ctg.TP = T, ctg.reference_pressure
    Kc = gas.equilibrium_constants_Kc(T)
    # Cantera Kc is per kmol/m^3 units; convert our mol/m^3 by net stoich.
    kc_ct = ctg.equilibrium_constants
    # Compare the first several elementary reactions on a log scale.
    n = min(len(Kc), len(kc_ct), 10)
    # Our Kc is in mol/m^3 units; Cantera's is in kmol/m^3. A concentration in
    # kmol/m^3 is (mol/m^3)/1000, so Kc_ct = Kc_ours * (1e-3)^dnu.
    for i in range(n):
        rxn = mech.reactions[i]
        dnu = sum(rxn.products.values()) - sum(rxn.reactants.values())
        ours = Kc[i] * (1e-3) ** dnu
        assert np.isclose(np.log(ours), np.log(kc_ct[i]), atol=1e-6)


def test_net_rates_is_design_hook(cantera_mech):
    gas = Thermo(cantera_mech)
    Y = np.zeros(cantera_mech.n_species)
    Y[cantera_mech.species_index["H2O"]] = 1.0
    with pytest.raises(NotImplementedError):
        gas.net_rates(Y, 1500.0, 101325.0)
