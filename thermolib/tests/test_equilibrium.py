"""Chemical equilibrium (Backend D) -- R-A4.1, R-A4.2, R-A4.3, R-A3.4.

These tests are Cantera-free: they check internal consistency
(conservation, realizability, HP<->TP round-trips, differentiation).
"""

import numpy as np
import pytest

from thermolib import Thermo


@pytest.fixture
def gas(native_mech):
    return Thermo(native_mech)


@pytest.fixture
def Z_stoich(gas, native_mech):
    # Elemental composition of a stoichiometric H2/O2 + N2 dilution mixture.
    Y = np.zeros(native_mech.n_species)
    idx = native_mech.species_index
    # 2 H2 + O2 -> 2 H2O, plus N2 diluent (mole-based -> set via X then convert)
    X = np.zeros(native_mech.n_species)
    X[idx["H2"]] = 2.0
    X[idx["O2"]] = 1.0
    X[idx["N2"]] = 3.76
    X /= X.sum()
    W = native_mech.molar_masses
    Y = X * W / np.sum(X * W)
    return gas.elemental_mass_fractions(Y)


def test_tp_equilibrium_realizable_and_conserved(gas, native_mech, Z_stoich):
    res = gas.equilibrate_TP(Z_stoich, T=2400.0, p=101325.0)
    assert res.converged
    assert np.isclose(np.sum(res.Y), 1.0)
    assert np.all(res.Y >= -1e-12)
    # Element conservation: recovered elemental composition matches input.
    Z_out = gas.elemental_mass_fractions(res.Y)
    assert np.allclose(Z_out, Z_stoich, atol=1e-10)


def test_hp_equilibrium_conserves_enthalpy(gas, native_mech, Z_stoich):
    # Pick a target enthalpy from a known (T, composition); equilibrate at HP
    # and confirm the resulting mixture enthalpy equals the target (R-A4.1).
    p = 101325.0
    res_tp = gas.equilibrate_TP(Z_stoich, T=2200.0, p=p)
    h_target = res_tp.properties.h
    res_hp = gas.equilibrate_HP(Z_stoich, h_target, p, T_guess=1800.0)
    assert res_hp.converged
    assert np.isclose(res_hp.properties.h, h_target, rtol=1e-9)
    # HP should recover the same temperature/composition as the TP point.
    assert np.isclose(res_hp.T, 2200.0, rtol=1e-6)
    assert np.allclose(res_hp.Y, res_tp.Y, atol=1e-8)


def test_equilibrium_sound_speed_below_frozen(gas, Z_stoich):
    # Reacting equilibrium is softer: a_eq <= a_frozen.
    res = gas.equilibrate_TP(Z_stoich, T=2600.0, p=101325.0)
    assert res.a_equilibrium < res.a_frozen
    assert res.a_equilibrium > 0


def test_hp_complex_step_sensitivities(gas, Z_stoich):
    # dT/dh and dT/dp via complex step match finite differences (R-A4.2).
    p = 101325.0
    res = gas.equilibrate_TP(Z_stoich, T=2300.0, p=p)
    h0 = res.properties.h
    eps = 1e-200

    dTdh_cs = gas.equilibrate_HP(Z_stoich, h0 + 1j * eps, p).T.imag / eps
    d = h0 * 1e-6
    dTdh_fd = (gas.equilibrate_HP(Z_stoich, h0 + d, p).T - gas.equilibrate_HP(Z_stoich, h0 - d, p).T) / (2 * d)
    assert np.isclose(dTdh_cs, dTdh_fd, rtol=1e-5)

    dTdp_cs = gas.equilibrate_HP(Z_stoich, h0, p + 1j * 1e-200).T.imag / 1e-200
    dpr = p * 1e-6
    dTdp_fd = (gas.equilibrate_HP(Z_stoich, h0, p + dpr).T - gas.equilibrate_HP(Z_stoich, h0, p - dpr).T) / (2 * dpr)
    assert np.isclose(dTdp_cs, dTdp_fd, rtol=1e-5)


def test_pressure_is_ordinary_input(gas, native_mech, Z_stoich):
    # R-A4.1a: pressure is an ordinary input; higher p suppresses dissociation,
    # raising the adiabatic flame temperature for a fixed reactant enthalpy.
    p = 101325.0
    idx = native_mech.species_index
    X = np.zeros(native_mech.n_species)
    X[idx["H2"]] = 2.0
    X[idx["O2"]] = 1.0
    X[idx["N2"]] = 3.76
    W = native_mech.molar_masses
    Y_react = X * W / np.sum(X * W)
    # Frozen reactant enthalpy at 300 K (the energy that the flame conserves).
    h0 = gas.properties(Y_react, 300.0, p).h
    T_lo = gas.equilibrate_HP(Z_stoich, h0, 0.5 * p).T
    T_hi = gas.equilibrate_HP(Z_stoich, h0, 5.0 * p).T
    assert T_hi > T_lo


def test_backend_selection(native_mech):
    assert Thermo(native_mech, backend="kernel").backend.name == "kernel"
    with pytest.raises(NotImplementedError):
        Thermo(native_mech, backend="table")
    with pytest.raises(ValueError):
        Thermo(native_mech, backend="nonsense")
