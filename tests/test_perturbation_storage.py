"""Storage block ``M`` and the finite-volume cavity (the operator's compliance face).

Covers:

* the cavity is a wall to the mean flow (``mdot = 0``);
* its compliance populates ``M`` with the single entry ``V/c^2`` on the mass row;
* the fast fixed-pattern assembler reproduces the reference assembly with ``M`` present
  (``A = J_alg + i*omega*M + P + R``), and ``A(0) = J_alg``;
* a side-branch neck+cavity resonates at ``f0 = c*sqrt(A_n/(V*l))/2pi`` -- the Helmholtz
  frequency emerges from the neck inertance (the duct) and the cavity compliance (``M``).
"""

import numpy as np
import pytest

from nefes.thermo.configure import perfect_gas
from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_MDOT, ES_C
from nefes.perturbation import build_acoustic_blocks, assemble_acoustic, perturbation_response
from nefes.perturbation.operator.operator import _assemble_reference

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
P0, TT = 101325.0, 300.0


def _cavity_on_a_duct(volume=1.0e-3, area=1.0e-3, l_neck=0.02):
    """inlet -> short neck duct -> cavity (a quiescent dead-end resonator)."""
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [cat.total_pressure_inlet(P0, TT), cat.duct(l_neck), cat.cavity(volume)]
    edges = [(0, 1, area), (1, 2, area)]
    prob = build_problem(cfg, els, edges, 1.0, P0, CP * TT)
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
    prob = build_problem(cfg, els, [(0, 1, 1e-3), (1, 2, 1e-3)], 1.0, P0, CP * TT)
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
    prob = build_problem(cfg, els, edges, 1.0, P0, CP * TT)
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


# ---------------------------------------------------------------------------
# Generalized storage: per-port compliance + series inertance on the inline
# pressure elements, and the chamber-volume compliance on the manifolds.
# ---------------------------------------------------------------------------


def _M_dict(blocks):
    """The storage block as ``{(row, col): value}`` for direct entry checks."""
    Mco = blocks.M.tocoo()
    return {(int(r), int(c)): complex(v) for r, c, v in zip(Mco.row, Mco.col, Mco.data)}


def _quiescent_inline(el, area=1.0e-3):
    """inlet -> (el under test) -> outlet, a still uniform medium at P0/TT."""
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [cat.total_pressure_inlet(P0, TT), el, cat.pressure_outlet(P0, Tt_backflow=TT)]
    prob = build_problem(cfg, els, [(0, 1, area), (1, 2, area)], 1.0, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_inline_compliance_and_inertance_entries():
    # a linear_resistance carrying storage lengths populates M with two per-port
    # compliance entries (mass row, pressure cols) and one series-inertance entry
    # (pressure row, through mdot col); values match the closed forms.
    lu, ld, ec, A = 0.013, 0.007, 0.004, 1.0e-3
    el = cat.linear_resistance(0.0, l_up=lu, l_down=ld, end_correction=ec)
    prob, x = _quiescent_inline(el, area=A)
    est = states_table(prob, x)
    c = float(est[ES_C, 0])
    ns = int(prob.n_solve)
    r_mass = int(prob.node_row_ptr[1])
    r_press = r_mass + 1
    M = _M_dict(build_acoustic_blocks(prob, x))
    # compliance: l_i * A_i / c_i^2 on each port's pressure column
    assert M[(r_mass, ns * 0 + 1)] == pytest.approx(lu * A / c**2, rel=1e-12)
    assert M[(r_mass, ns * 1 + 1)] == pytest.approx(ld * A / c**2, rel=1e-12)
    # inertance: |L_eff| / A_throat on the through mdot column (sign = port-0 orientation)
    L_eff = lu + ld + ec
    assert abs(M[(r_press, ns * 0 + 0)]) == pytest.approx(L_eff / A, rel=1e-12)
    assert len(M) == 3


def test_lengths_are_inert_in_the_mean_flow():
    # the storage lengths are acoustic metadata only: the converged mean state is identical
    # with and without them (a flowing loss element, so the mean is non-trivial).
    cfg = perfect_gas(R_AIR, GAMMA)

    def solve_with(el):
        els = [cat.mass_flow_inlet(0.4, TT), el, cat.pressure_outlet(P0, Tt_backflow=TT)]
        prob = build_problem(cfg, els, [(0, 1, 2e-3), (1, 2, 2e-3)], 0.4, P0, CP * TT)
        res = solve(prob)
        assert res.converged
        return res.x

    x_plain = solve_with(cat.loss(2.5))
    x_stored = solve_with(cat.loss(2.5, l_up=0.05, l_down=0.05, end_correction=0.01))
    assert np.max(np.abs(x_plain - x_stored)) < 1e-10


def test_manifold_volume_is_a_compliance():
    # a junction with a chamber volume carries one compliance entry V/c^2 on its mass row,
    # at the common (port-0) pressure column -- the cavity rule with through-flow.
    cfg = perfect_gas(R_AIR, GAMMA)
    V = 2.0e-3
    els = [
        cat.total_pressure_inlet(P0, TT),
        cat.duct(0.05),
        cat.junction(volume=V),
        cat.duct(0.05),
        cat.pressure_outlet(P0, Tt_backflow=TT),
    ]
    edges = [(0, 1, 2e-3), (1, 2, 2e-3), (2, 3, 2e-3), (3, 4, 2e-3)]
    prob = build_problem(cfg, els, edges, 1.0, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    blocks = build_acoustic_blocks(prob, res.x)
    assert blocks.M.nnz == 1
    est = states_table(prob, res.x)
    e0 = int(prob.col_edge[int(prob.row_ptr[2])])  # junction's first port edge
    c = float(est[ES_C, e0])
    r0 = int(prob.node_row_ptr[2])
    M = _M_dict(blocks)
    assert M[(r0, int(prob.n_solve) * e0 + 1)] == pytest.approx(V / c**2, rel=1e-12)


def _hr_with_inline_neck(volume, neck_area, l_neck, main_area=3.0e-3, l_main=0.05):
    """Side-branch HR whose neck is an inline inertance (linear_resistance), not a duct.

    junction - [linear_resistance with l_eff = l_neck] - cavity.  The half-length split
    l_up=l_down=l_neck/2 reproduces a neck of geometric length l_neck (inertance l_neck +
    the matching compliance), the dual of _side_branch_hr's neck *duct*.
    """
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [
        cat.total_pressure_inlet(P0, TT),
        cat.duct(l_main),
        cat.junction(),
        cat.duct(l_main),
        cat.pressure_outlet(P0, Tt_backflow=TT),
        cat.linear_resistance(0.0, l_up=l_neck / 2, l_down=l_neck / 2),
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
    prob = build_problem(cfg, els, edges, 1.0, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_inline_inertance_matches_neck_duct_resonance():
    # the duct-equivalence: a neck modeled as an inline inertance (M)
    # resonates at the same Helmholtz frequency as the neck modeled as a duct (P).
    V, AN, LN = 1.0e-3, 5.0e-4, 0.02
    freqs = np.linspace(50.0, 1100.0, 1100)
    p_duct, x_duct = _side_branch_hr(V, AN, LN)
    p_inl, x_inl = _hr_with_inline_neck(V, AN, LN)
    f_duct, tl_duct = _tl_peak_frequency(p_duct, x_duct, freqs)
    f_inl, tl_inl = _tl_peak_frequency(p_inl, x_inl, freqs)
    assert f_inl == pytest.approx(f_duct, rel=0.01)
    assert tl_inl > 20.0  # a genuine reactive resonance, not a numerical artifact


def test_inline_inertance_sign_is_orientation_invariant():
    # flipping the neck edges (so the port-0 orientation s0 flips) must not move the
    # resonance: the s0 factor in the inertance stamp keeps it arrow-independent.
    V, AN, LN = 1.0e-3, 5.0e-4, 0.02
    freqs = np.linspace(50.0, 1100.0, 1100)
    cfg = perfect_gas(R_AIR, GAMMA)
    main_area, l_main = 3.0e-3, 0.05

    def build(flip):
        els = [
            cat.total_pressure_inlet(P0, TT),
            cat.duct(l_main),
            cat.junction(),
            cat.duct(l_main),
            cat.pressure_outlet(P0, Tt_backflow=TT),
            cat.linear_resistance(0.0, l_up=LN / 2, l_down=LN / 2),
            cat.cavity(V),
        ]
        # the two neck edges, wired forward or reversed
        neck = [(5, 2, AN), (6, 5, AN)] if flip else [(2, 5, AN), (5, 6, AN)]
        edges = [(0, 1, main_area), (1, 2, main_area), (2, 3, main_area), (3, 4, main_area)] + neck
        prob = build_problem(cfg, els, edges, 1.0, P0, CP * TT)
        res = solve(prob)
        assert res.converged
        return prob, res.x

    p0, x0 = build(False)
    p1, x1 = build(True)
    f0, _ = _tl_peak_frequency(p0, x0, freqs)
    f1, _ = _tl_peak_frequency(p1, x1, freqs)
    assert f1 == pytest.approx(f0, rel=0.005)


@pytest.mark.parametrize("with_boundaries", [True, False])
def test_inline_storage_fast_path_matches_reference(with_boundaries):
    # the fixed-pattern fast assembler reproduces the slow reference with the inline
    # (compliance + inertance) storage present on a flowing element.
    cfg = perfect_gas(R_AIR, GAMMA)
    els = [
        cat.mass_flow_inlet(0.5, TT),
        cat.isentropic_area_change(l_up=0.03, l_down=0.02, end_correction=0.005),
        cat.pressure_outlet(P0, Tt_backflow=TT),
    ]
    prob = build_problem(cfg, els, [(0, 1, 3e-3), (1, 2, 1.5e-3)], 0.5, P0, CP * TT)
    res = solve(prob)
    assert res.converged
    blocks = build_acoustic_blocks(prob, res.x)
    assert blocks.M.nnz == 3  # two compliance + one inertance
    for f in (60.0, 340.0, 900.0, 2100.0):
        w = 2.0 * np.pi * f
        fast = assemble_acoustic(w, blocks, with_boundaries=with_boundaries).toarray()
        ref = _assemble_reference(w, blocks, with_boundaries=with_boundaries).toarray()
        assert np.max(np.abs(fast - ref)) < 1e-9 * (1.0 + np.max(np.abs(ref)))
