"""Perturbation-network core: characteristic maps, A(0)=J_alg, and duct phases."""

import numpy as np
import pytest

from nefes.thermo.configure import perfect_gas
from nefes.elements import catalog as cat
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.assembly.derive import ES_MDOT
from nefes.perturbation import (
    char_to_dx,
    dx_to_char,
    build_acoustic_blocks,
    assemble_acoustic,
    duct_modes,
    scattering_2port,
)
from nefes.perturbation.operator.characteristics import char_to_dq, basis_matrix

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR


def test_characteristic_maps_are_inverse():
    T = char_to_dx(1.2, 340.0, 50.0, 1.0e5, 0.10, K)
    L = dx_to_char(1.2, 340.0, 50.0, 1.0e5, 0.10, K)
    assert np.allclose(T @ L, np.eye(3), atol=1e-10)


def test_characteristic_amplitude_relations():
    rho, c = 1.2, 340.0
    R = char_to_dq(rho, c)
    for w in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.3, -0.4, 0.7])):
        f, g, h = w
        drho, du, dp = R @ w
        assert du == pytest.approx(f - g)
        assert dp == pytest.approx(rho * c * (f + g))
        assert drho == pytest.approx(h + dp / c**2)


def test_primitive_basis_is_velocity_normalized():
    # primitive flavor is (p'/(rho c), u', rho' c/rho) -- all in velocity units.
    rho, c, u, p, area = 1.2, 340.0, 50.0, 1.0e5, 0.10
    B = basis_matrix("primitive", rho, c, u, p, area, K)
    R = char_to_dq(rho, c)  # (rho', u', p') from (f, g, h)
    for w in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.3, -0.4, 0.7])):
        drho, du, dp = R @ w
        v1, v2, v3 = B @ w
        assert v1 == pytest.approx(dp / (rho * c))
        assert v2 == pytest.approx(du)
        assert v3 == pytest.approx(drho * c / rho)


def _nozzle(pt, Tt, p_out):
    cfg = perfect_gas(R_AIR, GAMMA)
    net = [
        cat.total_pressure_inlet(pt, Tt),
        cat.isentropic_area_change(),
        cat.pressure_outlet(p_out, Tt_backflow=Tt),
    ]
    edges = [(0, 1, 0.10), (1, 2, 0.06)]
    return cat.build_problem(cfg, net, edges, 10.0, 101325.0, CP * Tt)


def test_zero_frequency_operator_equals_jacobian():
    # A(0) = J_alg: the converged Jacobian IS the zero-frequency acoustic operator.
    prob = _nozzle(120000.0, 300.0, 101325.0)
    res = solve(prob)
    assert res.converged
    blocks = build_acoustic_blocks(prob, res.x)
    A0 = assemble_acoustic(0.0, blocks)
    diff = A0 - blocks.J_alg
    assert abs(diff).sum() == 0.0
    # the acoustic operator is finite and (away from resonance) nonsingular
    A = blocks.J_alg.toarray()
    assert np.all(np.isfinite(A))
    with np.errstate(divide="ignore", invalid="ignore"):  # benign LAPACK flags on complex det
        assert abs(np.linalg.det(A)) > 0.0


def test_quiescent_mean_assembles_cleanly():
    # pt_in = p_out -> mdot ~ 0 (the M_bar = 0 singular point, theory s12.6).
    prob = _nozzle(101325.0, 300.0, 101325.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    # quiescent to within the smoothing width (eps = 1e-3 * mdot_ref): Mach ~ 1e-6
    assert np.max(np.abs(est[ES_MDOT])) < 1e-2
    blocks = build_acoustic_blocks(prob, res.x)
    assert np.all(np.isfinite(blocks.J_alg.toarray()))


@pytest.mark.parametrize("c, L", [(340.0, 1.0), (300.0, 0.5)])
def test_closed_closed_duct_eigenfrequencies(c, L):
    modes = duct_modes(c, L, n_modes=3)
    expected = np.array([n * np.pi * c / L for n in (1, 2, 3)])
    assert np.allclose(modes, expected, rtol=2e-3)


def test_duct_scattering_is_lossless_phase():
    c, L, omega = 340.0, 1.2, 800.0
    S = scattering_2port(c, L, omega)
    assert abs(S[0, 0]) == pytest.approx(1.0)  # lossless transmission
    assert np.angle(S[0, 0]) == pytest.approx(-omega * L / c + 2 * np.pi * round(omega * L / c / (2 * np.pi)), abs=1e-9)
    # transmission phase delay tau = L/c
    assert S[0, 0] == pytest.approx(np.exp(-1j * omega * L / c))
