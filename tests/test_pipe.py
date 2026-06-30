"""The length-bearing pipe atom: Darcy-Weisbach friction + the duct acoustic phase.

The ``DUCT (+) LOSS`` unification (Greyvenstein-Laurie): one element that drops total
pressure ``pt0 - pt1 = K * 1/2 rho u^2`` with ``K = f*L/D`` on the mean flow *and* carries
its length for the acoustic phase -- so it is the right atom for the distributed Fanno pipe
and tapered-duct chains (the Class-2 composites).  Its complex-step safety lives in
``test_complex_step_safety.py`` (the PIPE probe).
"""

import numpy as np
import pytest

from fns.thermo.configure import perfect_gas
from fns.elements import catalog as cat
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_PT, ES_M, ES_T
from fns.perturbation import perturbation_response

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
P0, T0 = 101325.0, 300.0
L, D, F = 2.0, 0.05, 0.02
AREA = np.pi * D**2 / 4.0
K = F * L / D


def _solve(els, edges, mdot=0.3):
    prob = cat.build_problem(CFG, els, edges, mdot, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_pipe_friction_matches_loss_with_k_fl_over_d():
    # the pipe's mean total-pressure drop equals a loss(K = f*L/D) on the same flow
    pp, xp = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.pipe(L, D, F), cat.pressure_outlet(P0, T0)], [(0, 1, AREA), (1, 2, AREA)]
    )
    pl, xl = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.loss(K), cat.duct(L), cat.pressure_outlet(P0, T0)],
        [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA)],
    )
    ep, el = states_table(pp, xp), states_table(pl, xl)
    assert (ep[ES_PT, 0] - ep[ES_PT, 1]) == pytest.approx(el[ES_PT, 0] - el[ES_PT, 2], rel=1e-9)


def test_pipe_carries_the_acoustic_phase():
    # a real pipe propagates waves: its transmission phase rotates with frequency (a
    # lengthless loss would not), tracking a duct of the same length.
    freqs = np.linspace(50.0, 800.0, 200)

    def trans_phase(prob, x):
        resp = perturbation_response(prob, x, freqs)
        return np.unwrap(np.angle(resp.acoustic_scattering_matrix(0, 1)[:, 1, 0]))

    pp, xp = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.pipe(L, D, F), cat.pressure_outlet(P0, T0)], [(0, 1, AREA), (1, 2, AREA)]
    )
    assert np.ptp(trans_phase(pp, xp)) > 1.0  # a genuine propagation delay, not a flat (compact) phase


def test_pipe_zero_friction_is_a_duct():
    # f = 0 -> K = 0 -> the pipe is exactly a lossless duct (mean flow + acoustic phase)
    freqs = np.linspace(50.0, 800.0, 150)
    pp, xp = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.pipe(L, D, 0.0), cat.pressure_outlet(P0, T0)], [(0, 1, AREA), (1, 2, AREA)]
    )
    pd, xd = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.duct(L), cat.pressure_outlet(P0, T0)], [(0, 1, AREA), (1, 2, AREA)]
    )
    ep, ed = states_table(pp, xp), states_table(pd, xd)
    assert ep[ES_PT, 0] == pytest.approx(ed[ES_PT, 0], rel=1e-10)
    assert ep[ES_PT, 1] == pytest.approx(ed[ES_PT, 1], rel=1e-10)  # no total-pressure drop
    sp = perturbation_response(pp, xp, freqs).acoustic_scattering_matrix(0, 1)
    sd = perturbation_response(pd, xd, freqs).acoustic_scattering_matrix(0, 1)
    assert np.allclose(sp, sd, atol=1e-9)


def test_pipe_mach_rises_under_friction():
    # constant-area adiabatic friction is Fanno flow: total temperature is conserved while
    # the subsonic Mach rises toward the exit (the gradient a single lumped pipe captures
    # to leading order; the fanno_pipe chain resolves it).
    pp, xp = _solve(
        [cat.mass_flow_inlet(0.3, T0), cat.pipe(L, D, F), cat.pressure_outlet(P0, T0)], [(0, 1, AREA), (1, 2, AREA)]
    )
    e = states_table(pp, xp)
    Tt0 = e[ES_T, 0] * (1.0 + 0.5 * (GAMMA - 1.0) * e[ES_M, 0] ** 2)
    Tt1 = e[ES_T, 1] * (1.0 + 0.5 * (GAMMA - 1.0) * e[ES_M, 1] ** 2)
    assert Tt1 == pytest.approx(Tt0, rel=1e-6)  # adiabatic: total temperature conserved
    assert e[ES_M, 1] > e[ES_M, 0]  # the friction accelerates the subsonic flow


def test_pipe_is_constant_area():
    # the pipe is constant-area (like a duct): wiring unequal port areas is rejected
    with pytest.raises(ValueError, match="area"):
        _solve(
            [cat.mass_flow_inlet(0.3, T0), cat.pipe(L, D, F), cat.pressure_outlet(P0, T0)],
            [(0, 1, AREA), (1, 2, 2.0 * AREA)],
        )


def test_pipe_factory_validation():
    with pytest.raises(ValueError, match="positive"):
        cat.pipe(0.0, D, F)
    with pytest.raises(ValueError, match="positive"):
        cat.pipe(L, 0.0, F)
    with pytest.raises(ValueError, match="positive"):
        cat.pipe(L, D, -0.1)
