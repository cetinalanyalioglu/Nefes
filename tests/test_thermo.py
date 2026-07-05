"""Density recovery and its complex-step Jacobian.

This is the load-bearing test of the project.  If the IFT-spliced complex-step
derivative through the @njit density solve does not match the analytic
sensitivity, the whole complex-step-Jacobian premise fails.

Checks:
  * the real root reproduces an independent brentq root of F(rho),
  * complex-step d(rho)/d{m,p,h_t} through the njit kernel matches the analytic
    implicit-function-theorem block to ~1e-10 (and a finite-difference check),
  * thermo_update reproduces a perfect-gas state,
  * recovery round-trips a constructed physical state (rho, T, c, M, p_t, T_t).
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from nefes.thermo import perfect_gas, thermo_update, PERFECT_GAS
from nefes.thermo.perfect_gas import pg_solve_density

CS_H = 1e-30
R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR  # = gamma/(gamma-1)

# (m, p, h_t) grid: forward / reversed / near-stagnant / high-subsonic.
STATES = [
    (50.0, 101325.0, CP * 300.0),
    (-50.0, 101325.0, CP * 300.0),
    (1e-3, 101325.0, CP * 300.0),
    (220.0, 80000.0, CP * 320.0),  # M ~ 0.7
    (-180.0, 90000.0, CP * 310.0),
]


def _F(rho, m, p, ht):
    H = ht - m * m / (2.0 * rho * rho)
    return rho - p * K / H


def _density_ref(m, p, ht):
    rho_min = abs(m) / np.sqrt(2.0 * ht) if abs(m) > 0 else 0.0
    a = max(rho_min * (1.0 + 1e-9), 1e-8)
    return brentq(lambda r: _F(r, m, p, ht), a, 1.0e7, xtol=1e-14, rtol=1e-15)


@pytest.mark.parametrize("m, p, ht", STATES)
def test_real_root_matches_brentq(m, p, ht):
    got = pg_solve_density(m, p, ht, K)
    assert got == pytest.approx(_density_ref(m, p, ht), rel=1e-12)


@pytest.mark.parametrize("m, p, ht", STATES)
def test_complex_step_jacobian_matches_ift(m, p, ht):
    rho = _density_ref(m, p, ht)
    H = ht - m * m / (2.0 * rho * rho)
    f_rho = 1.0 + p * K * m * m / (rho**3 * H * H)
    f_m = -p * K * m / (rho * rho * H * H)
    f_p = -K / H
    f_h = p * K / (H * H)
    d_dm = -f_m / f_rho
    d_dp = -f_p / f_rho
    d_dh = -f_h / f_rho

    cs_m = pg_solve_density(complex(m, CS_H), p, ht, K).imag / CS_H
    cs_p = pg_solve_density(m, complex(p, CS_H), ht, K).imag / CS_H
    cs_h = pg_solve_density(m, p, complex(ht, CS_H), K).imag / CS_H

    assert cs_m == pytest.approx(d_dm, rel=1e-10)
    assert cs_p == pytest.approx(d_dp, rel=1e-10)
    assert cs_h == pytest.approx(d_dh, rel=1e-10)


@pytest.mark.parametrize("m, p, ht", STATES)
def test_complex_step_matches_finite_difference(m, p, ht):
    base = pg_solve_density(m, p, ht, K)
    step = 1e-3
    fd = (pg_solve_density(m + step, p, ht, K) - pg_solve_density(m - step, p, ht, K)) / (2 * step)
    cs = pg_solve_density(complex(m, CS_H), p, ht, K).imag / CS_H
    assert cs == pytest.approx(fd, rel=1e-5)
    assert isinstance(base, float)


def test_thermo_update_perfect_gas():
    cfg = perfect_gas(R_AIR, GAMMA)
    out = np.zeros(4)
    h, p = CP * 305.0, 95000.0
    thermo_update(PERFECT_GAS, cfg.tf, cfg.ti, cfg.tf[:0], h, p, 0, out)
    T = h / CP
    assert out[0] == pytest.approx(T)
    assert out[1] == pytest.approx(p / (R_AIR * T))
    assert out[2] == pytest.approx(np.sqrt(GAMMA * R_AIR * T))
    assert out[3] == pytest.approx(8.314462618 / R_AIR)  # molar mass [kg/mol]


def test_round_trip_physical_state():
    cfg = perfect_gas(R_AIR, GAMMA)
    rho, u, p = 1.05, 240.0, 92000.0  # build a known physical state, area = 1
    T = p / (rho * R_AIR)
    c = np.sqrt(GAMMA * R_AIR * T)
    M = u / c
    h_t = CP * T + 0.5 * u * u
    pt = p * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))
    Tt = h_t / CP

    rho_rec = pg_solve_density(rho * u, p, h_t, K)
    assert rho_rec == pytest.approx(rho, rel=1e-12)

    h = h_t - 0.5 * (rho * u / rho_rec) ** 2
    out = np.zeros(4)
    thermo_update(PERFECT_GAS, cfg.tf, cfg.ti, cfg.tf[:0], h, p, 0, out)
    T_rec, c_rec = out[0], out[2]
    u_rec = rho * u / rho_rec
    M_rec = u_rec / c_rec
    pt_rec = p * (1.0 + 0.5 * (GAMMA - 1.0) * M_rec**2) ** (GAMMA / (GAMMA - 1.0))
    Tt_rec = h_t / CP

    assert T_rec == pytest.approx(T, rel=1e-12)
    assert c_rec == pytest.approx(c, rel=1e-12)
    assert M_rec == pytest.approx(M, rel=1e-12)
    assert pt_rec == pytest.approx(pt, rel=1e-12)
    assert Tt_rec == pytest.approx(Tt, rel=1e-12)
