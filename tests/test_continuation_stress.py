"""Stress test of the artificial-resistance continuation on large networks.

The solver warm-starts through ``kappa in (0.1, 0.01, 0.0)`` with a vanishing
smoothing width.  These tests scale the element count far past the handful in
``test_solver`` -- long serial chains, many parallel branches, and a long run of
sudden area changes (whose momentum<->contraction smooth switch is the most
continuation-sensitive kernel) -- and assert the continuation still lands a converged,
physical (subsonic, positive, mass-conserving) steady state, including from a
quiescent (zero-flow) cold start.
"""

import warnings

import numpy as np
import pytest

import nefes
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
    elements = [cat.mass_flow_inlet(mdot, 300.0), cat.junction()]
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


# -- heat-release flame: the h_t jump the default seed must anticipate ------------


def _heated_tube(Qdot, mdot=0.5, Tt=300.0, area=0.01, throat=0.006):
    """``inlet -> duct -> heat_release_flame -> duct -> choked_nozzle`` (perfect gas)."""
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.mass_flow_inlet(mdot, Tt),
        cat.duct(0.3),
        cat.heat_release_flame(Qdot),
        cat.duct(0.6),
        cat.choked_nozzle_outlet(throat),
    ]
    edges = [(i, i + 1, area) for i in range(len(elements) - 1)]
    return build_problem(cfg, elements, edges, mdot_ref=mdot, p_ref=P_REF, h_ref=CP * Tt), mdot


@pytest.mark.parametrize("Qdot", [1.0e4, 2.0e5, 8.0e5])
def test_heat_release_flame_converges_from_default_seed(Qdot):
    # The flame raises h_t by Qdot/|mdot|, so a uniform (unburnt) seed puts the default solve
    # on the steep 1/mdot wall of the energy row and it stalls; the seed must carry the rise.
    prob, mdot = _heated_tube(Qdot)
    res = solve(prob)  # no x0: exercises the default seeding path
    est = _assert_physical(prob, res)
    assert np.allclose(est[ES_MDOT], mdot, rtol=1e-6)  # series: mass conserved
    # the flame's heat lands as a total-enthalpy rise across it: Qdot = mdot * (ht_out - ht_in)
    ht = est[ES_T] * CP + 0.5 * (est[ES_MDOT] / (est[ES_P] / (R_AIR * est[ES_T]) * 0.01)) ** 2
    assert ht[2] - ht[1] == pytest.approx(Qdot / mdot, rel=1e-6)


def test_heat_release_flame_default_seed_matches_ramped_solve():
    # Ramping Qdot by hand (warm-starting each step from the last) is the workaround the
    # default seed removes: both paths must land on the same state.
    prob, _ = _heated_tube(2.0e5)
    direct = solve(prob)
    x = None
    for Q in (2.0e4, 5.0e4, 1.0e5, 2.0e5):  # the manual ramp
        ramped = solve(_heated_tube(Q)[0], x0=x)
        assert ramped.converged
        x = ramped.x
    assert np.allclose(ramped.x, direct.x, rtol=1e-8)


# -- the flame seed's divisor: a flow the seed must estimate when no inlet prescribes it ----


def _pt_fed_flame(Qdot, mdot_ref=None):
    """``reservoir -> duct -> heat_release_flame -> duct -> choked_nozzle``.

    A total-pressure inlet leaves the mass flow to the solve, so the seed must estimate the flow
    it divides ``Qdot`` by.  ``mdot_ref=None`` takes the reference the network derives itself.
    """
    kw = {} if mdot_ref is None else {"mdot_ref": mdot_ref}
    return nefes.Network(
        nodes=[
            cat.total_pressure_inlet(3.0e5, 300.0, name="reservoir"),
            cat.duct(0.3, name="cold"),
            cat.heat_release_flame(Qdot, name="burner"),
            cat.duct(0.6, name="hot"),
            cat.choked_nozzle_outlet(0.006, name="nozzle"),
        ],
        edges=[(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01), (3, 4, 0.01)],
        p_ref=P_REF,
        T_ref=300.0,
        **kw,
    )


@pytest.mark.parametrize("Qdot", [1.0e5, 1.0e6, 2.0e6])
def test_pt_inlet_flame_converges_from_default_seed(Qdot):
    # The flow is an outcome here, so the seed divides Qdot by the derived reference rather than a
    # prescribed flow.  Heat release throttles the true flow and the cold reference cannot see it,
    # but the resulting seed error is small enough that the default path still converges.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no diagnostic is due on the default path
        sol = _pt_fed_flame(Qdot).solve()
    assert sol.converged
    assert sol.field("M").max() < 1.0
    assert np.all(sol.field("T") > 0.0)


def test_cold_flame_seed_is_reported():
    # An mdot_ref far above the true flow (~2.7 kg/s) makes the seed divide Qdot by a flow an
    # order of magnitude too large, so it seeds the flame cold.  That is the one case the seed
    # cannot absorb, and it must say so rather than fail mutely.
    with pytest.warns(UserWarning, match=r"burner.*seeded its total-enthalpy rise"):
        _pt_fed_flame(1.0e6, mdot_ref=50.0).solve()


def test_flame_seed_report_names_the_reference_to_blame():
    with pytest.warns(UserWarning, match=r"mdot_ref"):
        _pt_fed_flame(1.0e6, mdot_ref=50.0).solve()


def test_prescribed_inflow_flame_seed_is_not_reported():
    # A mass-flow inlet pins the flow, so the seed's divisor is exact and no diagnostic is due --
    # even with an mdot_ref just as wrong as the reported case above.
    net = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(0.5, 300.0, name="inlet"),
            cat.duct(0.3, name="cold"),
            cat.heat_release_flame(2.0e5, name="burner"),
            cat.duct(0.6, name="hot"),
            cat.choked_nozzle_outlet(0.006, name="nozzle"),
        ],
        edges=[(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01), (3, 4, 0.01)],
        p_ref=P_REF,
        T_ref=300.0,
        mdot_ref=50.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol = net.solve()
    assert sol.converged
