"""Stress test of the artificial-resistance continuation on large networks.

The solver warm-starts through ``kappa in (0.1, 0.01, 0.0)`` with a vanishing
smoothing width.  These tests scale the element count far past the handful in
``test_solver`` -- long serial chains, many parallel branches, and a long run of
sudden area changes (whose momentum<->contraction smooth switch is the most
continuation-sensitive kernel) -- and assert the continuation still lands a converged,
physical (subsonic, positive, mass-conserving) steady state, including from a
quiescent (zero-flow) cold start.
"""

import numpy as np
import pytest

from nefes.assembly.recover import ES_M, ES_MDOT, ES_P, ES_T
from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.control import initial_guess
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
P_REF = 101325.0


def _assert_physical(prob, res):
    """Converged, subsonic, strictly-positive thermodynamic state."""
    assert res.converged
    assert res.residual_norm < 1e-9
    est = states_table(prob, res.x)
    assert np.all(est[ES_P] > 0.0)
    assert np.all(est[ES_T] > 0.0)
    assert np.all(np.abs(est[ES_M]) < 1.0)  # subsonic scope
    assert np.all(np.isfinite(res.x))
    return est


# -- long serial chain ------------------------------------------------------


def _long_chain(n_blocks, area=0.30, K=0.6, mdot=10.0):
    """``mass_flow_inlet -> [loss, duct] * n_blocks -> pressure_outlet`` (constant area)."""
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [cat.mass_flow_inlet(mdot, 300.0)]
    for _ in range(n_blocks):
        elements += [cat.loss(K), cat.duct()]
    elements += [cat.pressure_outlet(P_REF)]
    edges = [(i, i + 1, area) for i in range(len(elements) - 1)]
    return build_problem(cfg, elements, edges, mdot_ref=mdot, p_ref=P_REF, h_ref=CP * 300.0), mdot


def test_long_serial_chain_converges():
    prob, mdot = _long_chain(n_blocks=30)  # 62 elements, 61 edges
    res = solve(prob)
    est = _assert_physical(prob, res)
    # series chain: every edge carries the inlet mass flow exactly
    assert np.allclose(est[ES_MDOT], mdot, rtol=1e-6)
    # dissipative: static pressure is non-increasing downstream (flat across the
    # lossless ducts, dropping across the losses) with a net fall to the outlet
    assert np.all(np.diff(est[ES_P]) <= 1e-6 * P_REF)
    assert est[ES_P, 0] > est[ES_P, -1] + 1e3


def test_long_serial_chain_cold_start():
    # The continuation must reach the same state from a dead-stop (mdot = 0) guess.
    prob, mdot = _long_chain(n_blocks=30)
    res = solve(prob, x0=initial_guess(prob, mdot0=0.0))
    est = _assert_physical(prob, res)
    assert np.allclose(est[ES_MDOT], mdot, rtol=1e-6)


# -- wide parallel manifold -------------------------------------------------


def _parallel_branches(n_branch, area=0.20, mdot=15.0):
    """``inlet -> splitter -> [loss_i] * n_branch -> junction -> outlet`` with spread loss K."""
    cfg = perfect_gas(R_AIR, GAMMA)
    j = 2 + n_branch  # junction node index
    elements = [cat.mass_flow_inlet(mdot, 300.0), cat.splitter()]
    Ks = [0.5 + 0.4 * i for i in range(n_branch)]  # strictly increasing loss
    elements += [cat.loss(K) for K in Ks]
    elements += [cat.junction(), cat.pressure_outlet(P_REF)]
    edges = [(0, 1, area)]
    for i in range(n_branch):
        edges += [(1, 2 + i, area), (2 + i, j, area)]
    edges += [(j, j + 1, area)]
    return build_problem(cfg, elements, edges, mdot_ref=mdot, p_ref=P_REF, h_ref=CP * 300.0), mdot, Ks


def test_many_parallel_branches_converge():
    prob, mdot, Ks = _parallel_branches(n_branch=16)  # 20 elements, 33 edges
    res = solve(prob)
    est = _assert_physical(prob, res)

    # branch mdots are edges 1, 3, 5, ... (the splitter->loss legs)
    branch_mdot = np.array([est[ES_MDOT, 1 + 2 * i] for i in range(len(Ks))])
    assert branch_mdot.sum() == pytest.approx(mdot, rel=1e-6)  # mass conserved
    assert np.all(branch_mdot > 0.0)  # no recirculation
    # higher-loss branches carry less flow (strictly decreasing in K)
    assert np.all(np.diff(branch_mdot) < 0.0)


# -- long run of sudden area changes (the continuation-sensitive kernel) ---------


def _sac_chain(n_sac, A0=0.25, A1=0.18, mdot=12.0):
    """``inlet -> sudden_area_change * n_sac -> outlet`` with edge area alternating A0/A1."""
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [cat.mass_flow_inlet(mdot, 300.0)]
    elements += [cat.sudden_area_change(cc=0.8) for _ in range(n_sac)]
    elements += [cat.pressure_outlet(P_REF)]
    edges = [(i, i + 1, A0 if i % 2 == 0 else A1) for i in range(len(elements) - 1)]
    return build_problem(cfg, elements, edges, mdot_ref=mdot, p_ref=P_REF, h_ref=CP * 300.0), mdot


def test_sudden_area_change_chain_converges():
    prob, mdot = _sac_chain(n_sac=20)  # 22 elements, 21 edges
    res = solve(prob)
    est = _assert_physical(prob, res)
    assert np.allclose(est[ES_MDOT], mdot, rtol=1e-6)  # series: mass conserved


def test_sudden_area_change_chain_cold_start():
    prob, mdot = _sac_chain(n_sac=20)
    res = solve(prob, x0=initial_guess(prob, mdot0=0.0))
    est = _assert_physical(prob, res)
    assert np.allclose(est[ES_MDOT], mdot, rtol=1e-6)
