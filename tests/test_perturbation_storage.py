"""Storage block ``M`` and the finite-volume cavity (the operator's compliance face).

Covers (theory.md s12.5, scratch/helmholtz-resonator-plan.md Phase 1-3):

* the cavity is a wall to the mean flow (``mdot = 0``);
* its compliance populates ``M`` with the single entry ``V/c^2`` on the mass row;
* the fast fixed-pattern assembler reproduces the reference assembly with ``M`` present
  (``A = J_alg + i*omega*M + P + R``), and ``A(0) = J_alg``;
* a side-branch neck+cavity resonates at ``f0 = c*sqrt(A_n/(V*l))/2pi`` -- the Helmholtz
  frequency emerges from the neck inertance (the duct) and the cavity compliance (``M``).
"""

import numpy as np
import pytest

from fns.thermo.configure import perfect_gas
from fns.elements import catalog as cat
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_MDOT, ES_C
from fns.perturbation import build_acoustic_blocks, assemble_acoustic, perturbation_response
from fns.perturbation.operator import _assemble_reference

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
P0, TT = 101325.0, 300.0


def _cavity_on_a_duct(volume=1.0e-3, area=1.0e-3, l_neck=0.02):
    """inlet -> short neck duct -> cavity (a quiescent dead-end resonator)."""
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [cat.total_pressure_inlet(P0, TT), cat.duct(l_neck), cat.cavity(volume)]
    edges = [(0, 1, area), (1, 2, area)]
    prob = cat.build_problem(cfg, els, edges, 1.0, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_cavity_is_a_wall_to_the_mean_flow():
    # the cavity blocks mean flow exactly like a wall: mdot = 0 on its edge.
    prob, x = _cavity_on_a_duct()
    est = states_table(prob, x)
    assert np.max(np.abs(est[ES_MDOT])) < 1e-8


def test_cavity_storage_is_the_compliance():
    # M carries a single entry: V/c^2 on the cavity mass row, the cavity-edge pressure column.
    V = 1.5e-3
    prob, x = _cavity_on_a_duct(volume=V)
    blocks = build_acoustic_blocks(prob, x)
    assert blocks.M.nnz == 1
    est = states_table(prob, x)
    c = float(est[ES_C, 1])  # cavity edge
    r0 = int(prob.node_row_ptr[2])  # cavity node mass row
    pcol = int(prob.n_solve) * 1 + 1  # cavity edge (e=1), pressure variable (v=1)
    Mco = blocks.M.tocoo()
    assert (int(Mco.row[0]), int(Mco.col[0])) == (r0, pcol)
    assert Mco.data[0] == pytest.approx(V / c**2, rel=1e-12)


def test_no_storage_element_gives_empty_M():
    # a network without a cavity has an all-zero storage block.
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [cat.total_pressure_inlet(P0, TT), cat.duct(0.3), cat.pressure_outlet(P0, Tt_backflow=TT)]
    prob = cat.build_problem(cfg, els, [(0, 1, 1e-3), (1, 2, 1e-3)], 1.0, P0, CP * TT)
    res = solve(prob)
    blocks = build_acoustic_blocks(prob, res.x)
    assert blocks.M.nnz == 0


def test_storage_vanishes_at_zero_frequency():
    # the storage adds i*omega*M onto the cavity mass row; no duct/boundary stamp touches
    # that row, so at omega = 0 the row is exactly J_alg, and at omega > 0 it differs.
    prob, x = _cavity_on_a_duct()
    blocks = build_acoustic_blocks(prob, x)
    J = blocks.J_alg.tocsr()
    r0 = int(prob.node_row_ptr[2])  # cavity mass row
    A0 = assemble_acoustic(0.0, blocks).tocsr()
    assert abs(A0[r0] - J[r0]).sum() == pytest.approx(0.0, abs=1e-12)
    Aw = assemble_acoustic(2.0 * np.pi * 200.0, blocks).tocsr()
    assert abs(Aw[r0] - J[r0]).sum() > 0.0


@pytest.mark.parametrize("with_boundaries", [True, False])
def test_storage_fast_path_matches_reference(with_boundaries):
    # the fixed-pattern fast assembler reproduces the slow reference with M present.
    prob, x = _cavity_on_a_duct()
    blocks = build_acoustic_blocks(prob, x)
    assert blocks.M.nnz  # storage is genuinely exercised
    for f in (37.0, 220.0, 750.0, 1900.0):
        w = 2.0 * np.pi * f
        fast = assemble_acoustic(w, blocks, with_boundaries=with_boundaries).toarray()
        ref = _assemble_reference(w, blocks, with_boundaries=with_boundaries).toarray()
        assert np.max(np.abs(fast - ref)) < 1e-9 * (1.0 + np.max(np.abs(ref)))


def _side_branch_hr(volume, neck_area, l_neck, main_area=3.0e-3, l_main=0.05):
    """inlet - duct - junction - duct - outlet, with junction - neck - cavity."""
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [
        cat.total_pressure_inlet(P0, TT),
        cat.duct(l_main),
        cat.junction(),
        cat.duct(l_main),
        cat.pressure_outlet(P0, Tt_backflow=TT),
        cat.duct(l_neck),
        cat.cavity(volume),
    ]
    edges = [
        (0, 1, main_area),
        (1, 2, main_area),
        (2, 3, main_area),
        (3, 4, main_area),
        (2, 5, neck_area),
        (5, 6, neck_area),
    ]
    prob = cat.build_problem(cfg, els, edges, 1.0, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _tl_peak_frequency(prob, x, freqs):
    resp = perturbation_response(prob, x, freqs)
    tau = resp.acoustic_scattering_matrix(0, 3)[:, 1, 0]  # transmission inlet(e0) -> outlet(e3)
    tl = -20.0 * np.log10(np.abs(tau))
    return float(freqs[int(np.argmax(tl))]), float(np.max(tl))


def test_side_branch_helmholtz_resonance_frequency():
    # the neck inertance (duct) + cavity compliance (M) resonate at f0 = c*sqrt(A_n/(V*l))/2pi,
    # shorting the junction -> a transmission-loss peak there.
    V, AN, LN = 1.0e-3, 5.0e-4, 0.02
    prob, x = _side_branch_hr(V, AN, LN)
    c = float(states_table(prob, x)[ES_C, 3])
    f0 = c * np.sqrt(AN / (V * LN)) / (2.0 * np.pi)
    freqs = np.linspace(50.0, 1100.0, 1100)
    f_peak, tl_peak = _tl_peak_frequency(prob, x, freqs)
    # within the lumped-model compactness tolerance (no end correction on the neck yet)
    assert f_peak == pytest.approx(f0, rel=0.03)
    assert tl_peak > 20.0  # a genuine, sharp reactive resonance


def test_resonance_scales_with_cavity_volume():
    # f0 ~ 1/sqrt(V): quadrupling the cavity halves the resonance frequency.
    AN, LN = 5.0e-4, 0.02
    freqs = np.linspace(40.0, 1100.0, 1300)
    prob1, x1 = _side_branch_hr(1.0e-3, AN, LN)
    prob4, x4 = _side_branch_hr(4.0e-3, AN, LN)
    f1, _ = _tl_peak_frequency(prob1, x1, freqs)
    f4, _ = _tl_peak_frequency(prob4, x4, freqs)
    assert f1 / f4 == pytest.approx(2.0, rel=0.05)
