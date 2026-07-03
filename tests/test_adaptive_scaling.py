"""Adaptive, measured-from-state residual / variable scaling.

The Newton solve nondimensionalizes by a characteristic scale per row / variable.  These
were fixed boundary references; now (``adaptive_scale``, default on) the mass and enthalpy
scales are re-measured from the realized inflow at each homotopy stage, so the user need not
supply ``mdot_ref`` / ``h_ref`` (auto-derived seeds, hidden but overridable).  The quiescent
``mdot = 0`` case must keep working -- it falls back to the seed scales.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.shell.network import Network
from nefes.solver import solve
from nefes.assembly.scaling import compose_scales, measure_inflow_scales

CFG = perfect_gas(287.0, 1.4)
CP = 1.4 * 287.0 / 0.4


def _duct_prob(pt=1.2e5, p_out=1.0e5, mdot_ref=10.0, area=0.1):
    els = [cat.total_pressure_inlet(pt, 300.0), cat.duct(0.5), cat.pressure_outlet(p_out, 300.0)]
    return cat.build_problem(CFG, els, [(0, 1, area), (1, 2, area)], mdot_ref, 101325.0, CP * 300.0)


# --------------------------------------------------------------------------- #
# Scale composition and measurement
# --------------------------------------------------------------------------- #
def test_compose_scales_reproduces_compiled_references():
    prob = _duct_prob()
    deg = np.diff(prob.row_ptr)
    res, var = compose_scales(prob.node_rid, deg, prob.n_edges, prob.n_elem, *prob.var_scale[:3])
    assert np.allclose(res, prob.res_scale)
    assert np.allclose(var, prob.var_scale)


def test_measure_inflow_scales_reads_the_realized_flow():
    els = [cat.mass_flow_inlet(7.0, 300.0), cat.duct(), cat.pressure_outlet(1.0e5, 300.0)]
    prob = cat.build_problem(CFG, els, [(0, 1, 0.1), (1, 2, 0.1)], 7.0, 101325.0, CP * 300.0)
    res = solve(prob)
    mass, h = measure_inflow_scales(prob, res.x, prob.var_scale[0], prob.var_scale[2])
    assert mass == pytest.approx(7.0, rel=1e-6)  # the inlet mass flow
    assert h == pytest.approx(CP * 300.0, rel=1e-6)  # mean inlet total enthalpy


# --------------------------------------------------------------------------- #
# Adaptive vs fixed
# --------------------------------------------------------------------------- #
def test_adaptive_and_fixed_reach_the_same_solution():
    prob = _duct_prob(area=0.05)
    ra = solve(prob, adaptive_scale=True)
    rf = solve(prob, adaptive_scale=False)
    assert ra.converged and rf.converged
    assert np.allclose(ra.x, rf.x, rtol=1e-8)


def test_adaptive_default_handles_overestimated_seed():
    # a deliberately large compiled mdot_ref (1000) vs a true flow ~ few kg/s: adaptive
    # re-measures and still converges to the right state.
    prob = _duct_prob(mdot_ref=1000.0)
    res = solve(prob)  # adaptive on by default
    assert res.converged
    # the realized inflow is far below the seed, and the solve found it anyway
    assert abs(float(res.x[0, 0])) < 100.0


# --------------------------------------------------------------------------- #
# Quiescent mdot = 0 guardrail
# --------------------------------------------------------------------------- #
def test_quiescent_mass_inlet_still_solves():
    # an explicit zero-mass-flow inlet: the measured inflow is 0, so the adaptive scales must
    # fall back to the seed (never divide the norm by a vanishing scale).
    net = Network(gas=CFG, p_ref=1e5, T_ref=300.0)
    a = net.add(cat.mass_flow_inlet(0.0, 300.0))
    d = net.add(cat.duct(0.5))
    o = net.add(cat.pressure_outlet(1.0e5, 300.0))
    net.connect(a, d, 0.1)
    net.connect(d, o, 0.1)
    sol = net.solve()
    assert sol.converged
    assert np.allclose(sol.field("mdot"), 0.0, atol=1e-6)

    # measure_inflow_scales returns the seed unchanged at zero inflow
    mass, h = measure_inflow_scales(sol.problem, sol.x, 5.0, 1234.0)
    assert mass == 5.0 and h == 1234.0


# --------------------------------------------------------------------------- #
# Hidden, auto-derived seed references
# --------------------------------------------------------------------------- #
def test_seed_sums_mass_inlets():
    net = Network(gas=CFG, p_ref=1e5, T_ref=300.0)
    i1 = net.add(cat.mass_flow_inlet(3.0, 300.0))
    i2 = net.add(cat.mass_flow_inlet(5.0, 300.0))
    j = net.add(cat.junction())
    o = net.add(cat.pressure_outlet(1.0e5, 300.0))
    for a, b in [(i1, j), (i2, j), (j, o)]:
        net.connect(a, b, 0.1)
    assert net._seed_mdot() == pytest.approx(8.0)  # total specified inflow


def test_seed_uses_dp_estimate_when_pressure_driven():
    net = Network(gas=CFG, p_ref=1e5, T_ref=300.0)
    i = net.add(cat.total_pressure_inlet(1.2e5, 300.0))
    d = net.add(cat.duct())
    o = net.add(cat.pressure_outlet(1.0e5, 300.0))
    net.connect(i, d, 0.1)
    net.connect(d, o, 0.1)
    rho = 1e5 / (287.0 * 300.0)
    expected = 0.1 * np.sqrt(2.0 * rho * (1.2e5 - 1.0e5))
    assert net._seed_mdot() == pytest.approx(expected, rel=1e-6)


def test_refs_remain_overridable():
    net = Network(gas=CFG, p_ref=1e5, T_ref=300.0, mdot_ref=42.0, h_ref=9999.0)
    assert net._seed_mdot() == 42.0
    assert net._seed_h() == 9999.0
