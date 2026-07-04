"""Forced-split splitter: a flow divider that pins outflow fractions of the inflow.

The element (``cat.forced_splitter``) keeps one inflow at port 0 and ``N`` outflows;
``N - 1`` of the outflows are flow-controlled (their mass rate is a fixed fraction of
the inflow) and the last carries the remainder while keeping total-pressure continuity
with the inflow.  Every residual row is linear in the flow state, so it is
complex-step-exact and inherited unchanged by the perturbation operator.

Checks: the mean split is exact and independent of the outflow back-pressures (the
control-valve idealization), build/creation validation fires, and the element is
compatible with the perturbation network (a finite multiport scattering matrix).
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_MDOT, ES_PT
from nefes.perturbation import perturbation_response

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
H_REF = CP * 300.0


def _forced_split_network(fractions, out_pressures, mdot_in=30.0, area=0.25):
    """mass-flow inlet -> forced splitter -> one pressure outlet per outflow branch.

    The area keeps the inflow comfortably subsonic (M ~ 0.3); the floating-pressure
    divider is harder to converge near choke than a pressure-matched splitter.
    """
    els = [cat.mass_flow_inlet(mdot_in, 300.0), cat.forced_splitter(fractions, name="divider")]
    edges = [(0, 1, area)]  # e0: inlet -> splitter (port 0, the inflow)
    for k, p_out in enumerate(out_pressures):
        els.append(cat.pressure_outlet(p_out, 300.0))
        edges.append((1, 2 + k, area))  # splitter -> outlet k  (port k + 1)
    return cat.build_problem(CFG, els, edges, mdot_in, 101325.0, H_REF)


def test_mean_split_is_exact_and_backpressure_independent():
    # Two controlled branches (0.3, 0.5) + a remainder branch (0.2).  The outlet
    # pressures differ widely, yet the forced split must hold exactly: the
    # controlled branches are valves, not pressure-matched legs.
    fractions = [0.3, 0.5]
    out_p = [1.00e5, 0.90e5, 1.05e5]  # deliberately unequal back-pressures
    prob = _forced_split_network(fractions, out_p, mdot_in=30.0)
    res = solve(prob)
    assert res.converged

    st = states_table(prob, res.x)
    mdot = st[ES_MDOT]
    # e0 inflow; e1, e2 controlled; e3 remainder
    assert mdot[0] == pytest.approx(30.0, rel=1e-9)
    assert mdot[1] == pytest.approx(0.3 * 30.0, rel=1e-8)
    assert mdot[2] == pytest.approx(0.5 * 30.0, rel=1e-8)
    assert mdot[3] == pytest.approx(0.2 * 30.0, rel=1e-8)  # remainder = 1 - 0.3 - 0.5

    # the remainder branch (last outflow) keeps total-pressure continuity with the
    # inflow; the controlled branches' total pressures float to their back-pressures.
    pt = st[ES_PT]
    assert pt[3] == pytest.approx(pt[0], rel=1e-9)
    assert abs(pt[1] - pt[0]) > 1.0  # a controlled branch does NOT pressure-match


def test_split_tracks_fraction_choice():
    # A different fraction set on a 2-outflow divider: 0.65 controlled, 0.35 remainder.
    prob = _forced_split_network([0.65], [1.0e5, 1.0e5], mdot_in=22.0)
    res = solve(prob)
    assert res.converged
    mdot = states_table(prob, res.x)[ES_MDOT]
    assert mdot[1] == pytest.approx(0.65 * 22.0, rel=1e-8)
    assert mdot[2] == pytest.approx(0.35 * 22.0, rel=1e-8)


def test_mass_is_conserved():
    prob = _forced_split_network([0.2, 0.2, 0.2], [1.0e5] * 4, mdot_in=40.0, area=0.35)
    res = solve(prob)
    assert res.converged
    mdot = states_table(prob, res.x)[ES_MDOT]
    # outflows e1..e4 sum to the inflow e0
    assert mdot[1:].sum() == pytest.approx(mdot[0], rel=1e-9)
    assert mdot[4] == pytest.approx(0.4 * 40.0, rel=1e-8)  # remainder = 1 - 3*0.2


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_creation_rejects_bad_fractions():
    with pytest.raises(ValueError, match="at least one split fraction"):
        cat.forced_splitter([])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        cat.forced_splitter([1.2])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        cat.forced_splitter([-0.1])
    with pytest.raises(ValueError, match="sum to < 1"):
        cat.forced_splitter([0.5, 0.5])  # remainder would be zero


def test_build_rejects_port_count_mismatch():
    # 1 fraction declares 2 outflows (3 ports); wiring 3 outflows (4 ports) needs 2.
    with pytest.raises(ValueError, match="needs 2 split fraction"):
        _forced_split_network([0.3], [1.0e5, 1.0e5, 1.0e5])

    # fewer than 3 ports (1 inflow + 1 outflow) is not a meaningful split
    els = [cat.mass_flow_inlet(10.0, 300.0), cat.forced_splitter([0.3]), cat.pressure_outlet(1.0e5)]
    with pytest.raises(ValueError, match="needs >= 3 ports"):
        cat.build_problem(CFG, els, [(0, 1, 0.1), (1, 2, 0.1)], 10.0, 101325.0, H_REF)


# --------------------------------------------------------------------------- #
# Perturbation-network compatibility
# --------------------------------------------------------------------------- #
def _forced_tree(pt_in=1.10e5, p_out=1.01325e5, a=0.05, a_branch=0.03):
    """inlet -> duct -> forced splitter -> two outlets (3 terminals, a tree)."""
    els = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.duct(1.0),
        cat.forced_splitter([0.6], name="divider"),
        cat.pressure_outlet(p_out, 300.0),
        cat.pressure_outlet(p_out, 300.0),
    ]
    edges = [(0, 1, a), (1, 2, a), (2, 3, a_branch), (2, 4, a_branch)]
    prob = cat.build_problem(CFG, els, edges, 10.0, 101325.0, H_REF)
    res = solve(prob)
    assert res.converged
    return prob, res


def test_perturbation_multiport_is_finite():
    # The forced splitter is interior and carries no special acoustic stamp, so the
    # perturbation operator inherits its (linear) mean rows through J_alg.  A 3-terminal
    # tree must yield a bounded, grid-stable multiport scattering matrix.
    prob, res = _forced_tree()
    om = np.linspace(20.0, 1500.0, 200)
    r = perturbation_response(prob, res.x, om)
    S = r.multiport_scattering_matrix()
    assert S.shape == (om.size, 3, 3)
    assert np.all(np.isfinite(S))

    S_fine = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 800)).multiport_scattering_matrix()
    assert np.abs(S).max() < 5.0  # no resonant blow-up on the tree
    assert abs(np.abs(S).max() - np.abs(S_fine).max()) < 0.1 * np.abs(S_fine).max()


def test_perturbation_inherits_forced_fraction_constraint():
    # The mean fraction row  si*mdot_i' - beta*(-s0*mdot_0') = 0  is inherited verbatim
    # by the zero-frequency operator J_alg, so the controlled branch's mass-flow
    # perturbation is locked to beta times the inflow's.  Read it straight off J_alg.
    from nefes.perturbation import build_acoustic_blocks

    prob, res = _forced_tree()  # forced_splitter([0.6]) is node 2; edges: 1 (inflow), 2/3 (outflows)
    blocks = build_acoustic_blocks(prob, res.x)
    J = blocks.J_alg.tocsr()
    ns = prob.n_solve
    # the divider node's controlled-fraction row = node_row_ptr[2] + 1 (row 0 is mass balance)
    row = int(prob.node_row_ptr[2]) + 1
    coeffs = np.asarray(J[row].todense()).ravel()
    # inflow edge is 1, controlled outflow edge is 2 (port 1); mdot column is n_solve*e + 0
    c_in = coeffs[ns * 1 + 0]
    c_out = coeffs[ns * 2 + 0]
    # row is  (+-)mdot_2 -/+ 0.6*mdot_1 = 0  ->  the ratio of the two mdot coefficients is 0.6
    assert abs(c_in) == pytest.approx(0.6 * abs(c_out), rel=1e-9)
