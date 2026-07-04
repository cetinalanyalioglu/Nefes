"""Phase 4 validation: the mean-flow solver vs analytic 1-D compressible flow."""

import numpy as np
import pytest

from nefes.thermo.configure import perfect_gas
from nefes.elements import catalog as cat
from nefes.solver import solve
from nefes.solver.control import initial_guess
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_MDOT, ES_P, ES_M, ES_PT, ES_HT

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)


def _nozzle(pt, Tt, p_out, A0=0.10, A1=0.05, mdot_ref=10.0):
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.total_pressure_inlet(pt, Tt),
        cat.isentropic_area_change(),
        cat.pressure_outlet(p_out, Tt_backflow=Tt),
    ]
    edges = [(0, 1, A0), (1, 2, A1)]
    return cat.build_problem(cfg, elements, edges, mdot_ref, 101325.0, CP * Tt)


def _isentropic_exit(pt, Tt, p, A):
    pr = p / pt
    M = np.sqrt((pr ** (-(GAMMA - 1.0) / GAMMA) - 1.0) * 2.0 / (GAMMA - 1.0))
    T = Tt * pr ** ((GAMMA - 1.0) / GAMMA)
    c = np.sqrt(GAMMA * R_AIR * T)
    u = M * c
    rho = p / (R_AIR * T)
    return M, rho * u * A


def test_subsonic_nozzle_matches_isentropic():
    pt, Tt, p_out = 120000.0, 300.0, 101325.0
    prob = _nozzle(pt, Tt, p_out)
    res = solve(prob)
    assert res.converged
    assert res.residual_norm < 1e-9

    est = states_table(prob, res.x)
    M_exit, mdot_exit = est[ES_M, 1], est[ES_MDOT, 1]
    M_ref, mdot_ref = _isentropic_exit(pt, Tt, p_out, 0.05)

    assert M_exit == pytest.approx(M_ref, rel=1e-4)
    assert mdot_exit == pytest.approx(mdot_ref, rel=1e-4)
    assert est[ES_P, 1] == pytest.approx(p_out, rel=1e-6)  # subsonic: exit p = p_spec
    # isentropic: total pressure uniform
    assert est[ES_PT, 0] == pytest.approx(pt, rel=1e-5)
    assert est[ES_PT, 1] == pytest.approx(pt, rel=1e-5)


def test_choked_nozzle_saturates_mass_flow():
    pt, Tt = 120000.0, 300.0
    A1 = 0.05
    flux_star = np.sqrt(GAMMA / R_AIR) * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
    mdot_max = pt / np.sqrt(Tt) * flux_star * A1

    # Back pressure well below critical (0.528 pt = 63360) -> exit chokes.
    prob = _nozzle(pt, Tt, 50000.0, mdot_ref=14.0)
    res = solve(prob)
    assert res.converged

    est = states_table(prob, res.x)
    assert est[ES_M, 1] == pytest.approx(1.0, abs=5e-3)  # exit sonic
    assert est[ES_MDOT, 1] == pytest.approx(mdot_max, rel=5e-3)
    # underexpanded: exit static pressure detaches UPWARD from the spec
    assert est[ES_P, 1] > 50000.0 * 1.05


def test_critical_pressure_ratio_is_the_knee():
    # Just above critical: unchoked (mdot < max). Just below: choked (mdot = max).
    pt, Tt, A1 = 120000.0, 300.0, 0.05
    flux_star = np.sqrt(GAMMA / R_AIR) * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
    mdot_max = pt / np.sqrt(Tt) * flux_star * A1

    prob_hi = _nozzle(pt, Tt, 0.70 * pt, mdot_ref=14.0)  # 0.70 > 0.528
    prob_lo = _nozzle(pt, Tt, 0.40 * pt, mdot_ref=14.0)  # 0.40 < 0.528
    m_hi = states_table(prob_hi, solve(prob_hi).x)[ES_MDOT, 1]
    m_lo = states_table(prob_lo, solve(prob_lo).x)[ES_MDOT, 1]
    assert m_hi < 0.99 * mdot_max
    assert m_lo == pytest.approx(mdot_max, rel=5e-3)


def test_quiescent_cold_start_converges():
    # Start from (near) zero flow; the continuation must still find the solution.
    prob = _nozzle(120000.0, 300.0, 101325.0)
    x0 = initial_guess(prob, mdot0=0.0)
    res = solve(prob, x0=x0)
    assert res.converged
    assert res.residual_norm < 1e-9


def test_branch_network_conserves_mass():
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.mass_flow_inlet(20.0, 300.0),
        cat.splitter(),
        cat.loss(1.5),
        cat.loss(4.0),
        cat.junction(),
        cat.pressure_outlet(101325.0),
    ]
    edges = [
        (0, 1, 0.20),
        (1, 2, 0.12),
        (1, 3, 0.12),
        (2, 4, 0.12),
        (3, 4, 0.12),
        (4, 5, 0.20),
    ]
    prob = cat.build_problem(cfg, elements, edges, 20.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged

    est = states_table(prob, res.x)
    mdot = est[ES_MDOT]
    assert mdot[0] == pytest.approx(20.0, rel=1e-6)  # inlet spec
    assert mdot[0] == pytest.approx(mdot[5], rel=1e-6)  # in == out
    assert mdot[1] + mdot[2] == pytest.approx(mdot[0], rel=1e-6)  # split sums
    # the lower-loss branch (K=1.5) carries more flow than the K=4 branch
    assert mdot[1] > mdot[2]


def test_edge_direction_invariance():
    # Flipping the middle edge's reference direction flips its mdot sign only.
    pt, Tt, p_out = 120000.0, 300.0, 101325.0
    prob = _nozzle(pt, Tt, p_out)
    res = solve(prob)
    est = states_table(prob, res.x)

    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.total_pressure_inlet(pt, Tt),
        cat.isentropic_area_change(),
        cat.pressure_outlet(p_out, Tt_backflow=Tt),
    ]
    edges = [(0, 1, 0.10), (2, 1, 0.05)]  # exit edge reversed: tail=outlet, head=iac
    prob_f = cat.build_problem(cfg, elements, edges, 10.0, 101325.0, CP * Tt)
    res_f = solve(prob_f)
    est_f = states_table(prob_f, res_f.x)

    assert res_f.converged
    # exit edge mdot sign flips, magnitude and static state are identical
    assert est_f[ES_MDOT, 1] == pytest.approx(-est[ES_MDOT, 1], rel=1e-5)
    assert est_f[ES_P, 1] == pytest.approx(est[ES_P, 1], rel=1e-5)
    assert est_f[ES_HT, 1] == pytest.approx(est[ES_HT, 1], rel=1e-5)


# -- progress reporting -----------------------------------------------------


def test_verbose_silent_by_default(capsys):
    prob = _nozzle(120000.0, 300.0, 101325.0)
    solve(prob)
    assert capsys.readouterr().out == ""


def test_verbose_level1_prints_one_line_per_stage(capsys):
    prob = _nozzle(120000.0, 300.0, 101325.0)
    solve(prob, verbose=1)  # one summary line per continuation stage
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 3  # default kappa_stages = (0.1, 0.01, 0.0)
    assert all(ln.startswith("kappa=") and "converged=True" in ln for ln in lines)


def _iter_rows(out):
    """Per-iteration data rows of the verbose=2 group table (lead with the iter index)."""
    return [ln for ln in out.splitlines() if ln.split() and ln.split()[0].isdigit()]


def test_verbose_level2_prints_per_iteration(capsys):
    prob = _nozzle(120000.0, 300.0, 101325.0)
    solve(prob, verbose=2)
    out = capsys.readouterr().out
    assert "||R_hat||=" in out  # the gross-residual stage summary is still printed
    assert "mass" in out and "pressure" in out and "energy" in out  # equation-kind header
    assert len(_iter_rows(out)) > 3  # more detail than the per-stage summary


def test_progress_interval_thins_iteration_prints(capsys):
    prob = _nozzle(120000.0, 300.0, 101325.0)
    solve(prob, verbose=2, progress_interval=1)
    every = len(_iter_rows(capsys.readouterr().out))
    solve(prob, verbose=2, progress_interval=100)  # only iteration 0 of each stage
    sparse = len(_iter_rows(capsys.readouterr().out))
    assert sparse < every
