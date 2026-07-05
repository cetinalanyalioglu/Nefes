"""The closure adapter (AD-3 boundary).

Checks that closure_solve returns (rho, h) consistent with the kinetic-energy
coupling and that its complex-step derivatives w.r.t. the band-1 unknowns match
the analytic implicit-function-theorem block (chained through area).
"""

import numpy as np
import pytest

from nefes.assembly.closure import closure_solve
from nefes.thermo import perfect_gas, PERFECT_GAS

CS_H = 1e-30
R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR

CFG = perfect_gas(R_AIR, GAMMA)
CASES = [
    (50.0, 101325.0, CP * 300.0, 0.2),
    (-50.0, 101325.0, CP * 300.0, 0.2),
    (220.0, 80000.0, CP * 320.0, 0.5),
]


def _call(mdot, p, h_t, area):
    return closure_solve(PERFECT_GAS, CFG.tf, CFG.ti, mdot, p, h_t, np.zeros(0), area)


@pytest.mark.parametrize("mdot, p, h_t, area", CASES)
def test_closure_kinetic_energy_consistency(mdot, p, h_t, area):
    rho, h = _call(mdot, p, h_t, area)
    u = mdot / (rho * area)
    assert h == pytest.approx(h_t - 0.5 * u * u, rel=1e-12)
    # rho must satisfy the perfect-gas EOS at the recovered static state.
    T = h / CP
    assert rho == pytest.approx(p / (R_AIR * T), rel=1e-12)
    assert isinstance(rho, float) and isinstance(h, float)


@pytest.mark.parametrize("mdot, p, h_t, area", CASES)
def test_closure_complex_step_vs_finite_difference(mdot, p, h_t, area):
    # drho/dmdot and dh/dmdot via complex step vs central difference.
    step = 1e-2
    rho_p, h_p = _call(mdot + step, p, h_t, area)
    rho_m, h_m = _call(mdot - step, p, h_t, area)
    fd_rho = (rho_p - rho_m) / (2 * step)
    fd_h = (h_p - h_m) / (2 * step)

    cs_rho, cs_h = _call(complex(mdot, CS_H), p, h_t, area)
    assert cs_rho.imag / CS_H == pytest.approx(fd_rho, rel=1e-5)
    assert cs_h.imag / CS_H == pytest.approx(fd_h, rel=1e-5)


def test_closure_analytic_drho_dp():
    mdot, p, h_t, area = 220.0, 80000.0, CP * 320.0, 0.5
    m = mdot / area
    rho, _ = _call(mdot, p, h_t, area)
    H = h_t - m * m / (2.0 * rho * rho)
    f_rho = 1.0 + p * K * m * m / (rho**3 * H * H)
    f_p = -K / H
    analytic = -f_p / f_rho
    cs = _call(mdot, complex(p, CS_H), h_t, area)[0].imag / CS_H
    assert cs == pytest.approx(analytic, rel=1e-10)
