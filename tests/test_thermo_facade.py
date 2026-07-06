"""Species and mixture thermodynamics."""

import numpy as np
import pytest

from nefes.thermo import Thermo
from nefes.thermo.constants import R_UNIVERSAL


@pytest.fixture
def gas(cantera_mech):
    return Thermo(cantera_mech, backend="kernel")


def _mix(mech):
    Y = np.zeros(mech.n_species)
    idx = mech.species_index
    Y[idx["H2"]] = 0.05
    Y[idx["O2"]] = 0.20
    Y[idx["H2O"]] = 0.10
    Y[idx["N2"]] = 0.65
    return Y


def test_species_thermo_consistency(cantera_mech):
    # g/RT = h/RT - s/R, by construction.
    T = 1100.0
    assert np.allclose(cantera_mech.g_RT(T), cantera_mech.h_RT(T) - cantera_mech.s_R(T))


def test_frozen_sound_speed_identity(gas, cantera_mech):
    Y = _mix(cantera_mech)
    T, p = 1300.0, 2e5
    pr = gas.properties(Y, T, p)
    # a_frozen^2 = gamma * p / rho.
    assert np.isclose(pr.a_frozen**2, pr.gamma * p / pr.rho, rtol=1e-12)
    # cp - cv = R/W.
    assert np.isclose(pr.cp - pr.cv, R_UNIVERSAL / pr.W, rtol=1e-12)


def test_properties_complex_step_in_T(gas, cantera_mech):
    Y = _mix(cantera_mech)
    p = 1.5e5
    T = 1000.0
    eps = 1e-200
    # dh/dT should equal cp.
    h_cs = gas.properties(Y, T + 1j * eps, p).h.imag / eps
    cp = gas.properties(Y, T, p).cp
    assert np.isclose(h_cs, cp, rtol=1e-10)


def test_properties_complex_step_in_composition(gas, cantera_mech):
    Y = _mix(cantera_mech).astype(complex)
    idx = cantera_mech.species_index
    T, p = 1000.0, 1e5
    eps = 1e-200
    Yp = Y.copy()
    Yp[idx["H2"]] += 1j * eps
    dh = gas.properties(Yp, T, p).h.imag / eps
    # finite-difference cross check
    d = 1e-7
    Y1 = Y.real.copy()
    Y1[idx["H2"]] += d
    Y2 = Y.real.copy()
    Y2[idx["H2"]] -= d
    dh_fd = (gas.properties(Y1, T, p).h - gas.properties(Y2, T, p).h) / (2 * d)
    assert np.isclose(dh, dh_fd, rtol=1e-5)


def test_elemental_mass_fractions_sum_to_one(gas, cantera_mech):
    Y = _mix(cantera_mech)
    Z = gas.elemental_mass_fractions(Y)
    assert np.isclose(np.sum(Z), 1.0)
    assert np.all(Z >= 0)
