"""Mass-flow and compact choked-nozzle outlet elements (mean flow + acoustic closure).

The two outflow boundaries added alongside the static-pressure outlet:

* ``mass_flow_outlet`` pins the outflow rate; its inherited acoustic row is ``mdot' = 0``
  (the constant-mass-flow termination).
* ``choked_nozzle_outlet`` lumps a sonic throat of area ``A*`` just downstream -- the
  outflow is the critical mass flux for the interior stagnation state, the application
  plane stays **subsonic** (no M=1 degeneracy), and its inherited linearization is the
  compact choked-nozzle (Marble--Candel) reflection, entropy coupling included.
"""

import numpy as np
import pytest

from fns.thermo.configure import perfect_gas
from fns.elements import catalog as cat
from fns.solver import solve
from fns.solver.control import states_table
from fns.perturbation.operator import build_acoustic_blocks, assemble_acoustic
from fns.perturbation.characteristics import char_to_dx
from fns.perturbation.boundary_bc import PerturbationBC
from fns.derive import ES_MDOT, ES_P, ES_M, ES_PT, ES_RHO, ES_C, ES_U, ES_AREA

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR


def _area_mach(M, g):
    """Isentropic area ratio ``A/A*`` for subsonic Mach ``M`` (quasi-1D)."""
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def _crit_mass_flux(pt, Tt, A_star, g=GAMMA, Rg=R_AIR):
    """Choked-throat critical mass flow for total conditions ``(pt, Tt)`` and throat ``A*``."""
    return A_star * pt / np.sqrt(Tt) * np.sqrt(g / Rg) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


# --------------------------------------------------------------------------
# mass_flow_outlet
# --------------------------------------------------------------------------


def test_mass_flow_outlet_pins_outflow():
    """The outflow equals the prescribed rate; the static pressure floats."""
    mdot_spec = 8.0
    els = [cat.total_pressure_inlet(1.2e5, 300.0), cat.duct(0.5), cat.mass_flow_outlet(mdot_spec)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 8.0, 1e5, CP * 300.0)
    res = solve(prob)  # default guess
    assert res.converged
    est = states_table(prob, res.x)
    assert est[ES_MDOT, 1] == pytest.approx(mdot_spec, rel=1e-6)
    assert 0.0 < est[ES_M, 1] < 1.0  # subsonic outflow


def test_mass_flow_outlet_acoustic_row_is_mdot_zero():
    """Inherited (perturbation_bc=None) the acoustic outlet row enforces mdot' = 0."""
    els = [
        cat.total_pressure_inlet(1.2e5, 300.0, perturbation_bc=PerturbationBC.open_end()),
        cat.duct(0.5),
        cat.mass_flow_outlet(8.0),
    ]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 8.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged
    blocks = build_acoustic_blocks(prob, res.x)
    ns, e = int(prob.n_solve), 1
    r0 = int(prob.node_row_ptr[2])
    row = np.asarray(blocks.J_alg[r0, ns * e : ns * e + 3].todense()).ravel()
    # row acts only on mdot' (coefficients on p' and h_t' vanish)
    assert abs(row[0]) > 1e-6
    assert abs(row[1]) < 1e-9 * abs(row[0])
    assert abs(row[2]) < 1e-9 * abs(row[0])
    # and the operator assembles (the constant-mass-flow termination is well-posed)
    A = assemble_acoustic(2 * np.pi * 200.0, blocks, with_boundaries=True)
    assert A.nnz > 0


# --------------------------------------------------------------------------
# choked_nozzle_outlet
# --------------------------------------------------------------------------


def test_choked_nozzle_outlet_critical_mass_flux():
    """The outflow is the choked-throat critical mass flux; the approach stays subsonic."""
    pt, Tt, A_out, A_star = 1.2e5, 300.0, 0.05, 0.03
    els = [cat.total_pressure_inlet(pt, Tt), cat.duct(0.5), cat.choked_nozzle_outlet(A_star)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, A_out), (1, 2, A_out)], 10.0, 1e5, CP * Tt)
    res = solve(prob)  # default guess
    assert res.converged
    est = states_table(prob, res.x)
    M = float(est[ES_M, 1])
    assert 0.0 < M < 1.0  # subsonic application plane -> no M=1 degeneracy
    # outflow matches the independent critical-mass-flux formula
    assert est[ES_MDOT, 1] == pytest.approx(_crit_mass_flux(pt, Tt, A_star), rel=1e-4)
    # the approach Mach is the one consistent with the throat via the area-Mach relation
    assert _area_mach(M, GAMMA) == pytest.approx(A_out / A_star, rel=1e-4)
    # choked exhaust: the static pressure detaches below the (here) total pressure
    assert est[ES_P, 1] < float(est[ES_PT, 1])


def test_choked_nozzle_outlet_rejects_throat_not_smaller_than_outlet():
    """A* >= A_out has no subsonic choked approach -> rejected at build (not a CD/supersonic nozzle)."""
    els = [cat.total_pressure_inlet(1.6e5, 300.0), cat.duct(0.5), cat.choked_nozzle_outlet(0.05)]
    with pytest.raises(ValueError, match="smaller than"):
        cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 8.0, 1e5, CP * 300.0)


@pytest.mark.parametrize("pt", [1.05e5, 2.0e5, 5.0e5])
def test_choked_nozzle_outlet_imposes_critical_flux_at_any_total_pressure(pt):
    """The element asserts choking: it imposes the critical mass flux at *any* total pressure.

    There is no back-pressure to unchoke against, so the outflow simply scales with the
    interior total pressure (mdot ~ pt) -- robust at low pt, never failing to "choke".
    """
    A_out, A_star = 0.05, 0.02
    els = [cat.total_pressure_inlet(pt, 300.0), cat.duct(0.4), cat.choked_nozzle_outlet(A_star)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, A_out), (1, 2, A_out)], 5.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    assert 0.0 < est[ES_M, 1] < 1.0  # approach stays subsonic at every pt
    assert est[ES_MDOT, 1] == pytest.approx(_crit_mass_flux(pt, 300.0, A_star), rel=1e-4)


def test_choked_nozzle_outlet_acoustic_is_non_degenerate():
    """The compact (subsonic-approach) choked outlet assembles cleanly -- unlike a resolved M=1."""
    els = [
        cat.total_pressure_inlet(1.2e5, 300.0, perturbation_bc=PerturbationBC.open_end()),
        cat.duct(0.5),
        cat.choked_nozzle_outlet(0.03),
    ]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 10.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged
    blocks = build_acoustic_blocks(prob, res.x)
    A = assemble_acoustic(2 * np.pi * 250.0, blocks, with_boundaries=True)  # would raise at exactly M=1
    assert A.nnz > 0


def test_mass_flow_outlet_linearizes_to_constant_mass_flow():
    """The inherited mass-flow-outlet row IS the constant-mass-flow (mdot'=0) reflection.

    And ``PerturbationBC.constant_mass_flow()`` reproduces the same ``R`` / ``R_s`` -- so the
    standalone closure equals what the element inherits (theory: mdot' = 0 -> g = R f + R_s h,
    R = (1+M)/(1-M), R_s = c M / (rho (1-M))).
    """
    els = [
        cat.total_pressure_inlet(1.6e5, 300.0, perturbation_bc=PerturbationBC.open_end()),
        cat.duct(0.5),
        cat.mass_flow_outlet(6.0),
    ]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 6.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    blocks = build_acoustic_blocks(prob, res.x)
    ns, e = int(prob.n_solve), 1
    r0 = int(prob.node_row_ptr[2])
    row = np.asarray(blocks.J_alg[r0, ns * e : ns * e + 3].todense()).ravel()
    rho, c, u, p, area = (float(est[k, e]) for k in (ES_RHO, ES_C, ES_U, ES_P, ES_AREA))
    M = float(est[ES_M, e])
    cf, cg, ch = row @ char_to_dx(rho, c, u, p, area, K)
    R_inherit, Rs_inherit = -cf / cg, -ch / cg
    R_cmf, Rs_cmf = (1.0 + M) / (1.0 - M), c * M / (rho * (1.0 - M))
    assert R_inherit == pytest.approx(R_cmf, rel=1e-9)
    assert Rs_inherit == pytest.approx(Rs_cmf, rel=1e-7)
    # the standalone BC computes the same closure coefficients
    bc = PerturbationBC.constant_mass_flow()
    assert bc.reflection_coefficient(0.0, rho, c, M) == pytest.approx(R_cmf, rel=1e-12)
    assert bc.entropy_coupling_coefficient(0.0, rho, c, M) == pytest.approx(Rs_cmf, rel=1e-12)


def test_mass_flow_outlet_rejects_nonpositive_rate():
    """The outflow-only mass-flow outlet refuses a non-positive (ingesting) rate."""
    with pytest.raises(ValueError, match="outflow"):
        cat.mass_flow_outlet(-1.0)
    with pytest.raises(ValueError, match="outflow"):
        cat.mass_flow_outlet(0.0)


def test_choked_nozzle_outlet_rejects_nonpositive_area():
    with pytest.raises(ValueError, match="throat area"):
        cat.choked_nozzle_outlet(0.0)


# --------------------------------------------------------------------------
# Boundary-set well-posedness (pressure-gauge reference)
# --------------------------------------------------------------------------


def test_all_mass_flow_boundaries_rejected_no_pressure_reference():
    """mass-flow in + mass-flow out (the user's example) has no pressure gauge -> rejected."""
    els = [cat.mass_flow_inlet(5.0, 300.0), cat.duct(0.5), cat.mass_flow_outlet(5.0)]
    with pytest.raises(ValueError, match="pressure reference"):
        cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 5.0, 1e5, CP * 300.0)


def test_mass_flow_inlet_and_wall_rejected():
    """mass-flow in feeding a wall is also all-flow-pinned (no pressure reference)."""
    els = [cat.mass_flow_inlet(5.0, 300.0), cat.duct(0.5), cat.wall()]
    with pytest.raises(ValueError, match="pressure reference"):
        cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 5.0, 1e5, CP * 300.0)


def test_choked_outlet_supplies_the_pressure_reference():
    """mass-flow in + choked out is well-posed: the choke ties the pressure to the flow."""
    els = [cat.mass_flow_inlet(5.0, 300.0), cat.duct(0.5), cat.choked_nozzle_outlet(0.03)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 5.0, 1e5, CP * 300.0)
    res = solve(prob)  # builds (passes validation) and converges
    assert res.converged
    assert states_table(prob, res.x)[ES_MDOT, 1] == pytest.approx(5.0, rel=1e-6)


def test_pt_inlet_and_mass_flow_outlet_is_well_posed():
    """A pressure inlet + a mass-flow outlet is a valid pair (pressure pinned, flow pinned)."""
    els = [cat.total_pressure_inlet(1.6e5, 300.0), cat.duct(0.5), cat.mass_flow_outlet(6.0)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 6.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged


def test_pt_inlet_linearizes_to_convective_open_end():
    """The inherited total-pressure inlet row IS the energy-neutral convective open end.

    A stagnation reservoir fixes ``p_t`` and ``T_t``, so it seeds no entropy wave
    (``h_in = 0``); its single ``J_alg`` row then reduces to a pure acoustic reflection
    ``f = R g`` with ``R = -c_g/c_f``.  That equals ``-(1-M)/(1+M)`` -- identical to
    ``mean_flow_open_end`` and sitting exactly on the inlet energy-neutral bound
    ``|R| = (1-M)/(1+M)`` (so the reservoir does no net acoustic work).
    """
    els = [cat.total_pressure_inlet(2.5e5, 300.0), cat.duct(0.5), cat.choked_nozzle_outlet(0.03)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, 0.05), (1, 2, 0.05)], 10.0, 1e5, CP * 300.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    blocks = build_acoustic_blocks(prob, res.x)
    ns, e = int(prob.n_solve), 0  # inlet edge
    r0 = int(prob.node_row_ptr[0])  # inlet node row
    row = np.asarray(blocks.J_alg[r0, ns * e : ns * e + 3].todense()).ravel()
    rho, c, u, p, area = (float(est[k, e]) for k in (ES_RHO, ES_C, ES_U, ES_P, ES_AREA))
    M = float(est[ES_M, e])
    # inlet row in characteristic coordinates: cf f + cg g + ch h = 0; with h_in = 0 -> f = -(cg/cf) g
    cf, cg, ch = row @ char_to_dx(rho, c, u, p, area, K)
    R_inherit = -cg / cf
    R_open = -(1.0 - M) / (1.0 + M)
    assert R_inherit == pytest.approx(R_open, abs=1e-9)
    # it is the energy-neutral inlet reflection (does no net acoustic work)
    assert abs(R_inherit) == pytest.approx((1.0 - M) / (1.0 + M), rel=1e-9)
    # the standalone mean_flow_open_end closure reproduces the same R
    bc = PerturbationBC.mean_flow_open_end()
    assert bc.reflection_coefficient(0.0, rho, c, M) == pytest.approx(R_open, rel=1e-12)


def test_choked_nozzle_outlet_linearizes_to_marble_candel():
    """The inherited choked-outlet row IS the compact choked-nozzle (Marble--Candel) reflection.

    Steady-jump-linearized = acoustic-jump: the critical-mass-flux residual's complex-step
    linearization reproduces the Marble--Candel reflection ``R`` *and* the entropy -> acoustic
    coupling ``R_s`` at the (subsonic) approach Mach -- to round-off, no hand-coded BC.
    """
    pt, Tt, A_out, A_star = 1.2e5, 300.0, 0.05, 0.03
    els = [
        cat.total_pressure_inlet(pt, Tt, perturbation_bc=PerturbationBC.open_end()),
        cat.duct(0.5),
        cat.choked_nozzle_outlet(A_star),
    ]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, [(0, 1, A_out), (1, 2, A_out)], 10.0, 1e5, CP * Tt)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    blocks = build_acoustic_blocks(prob, res.x)
    ns, e = int(prob.n_solve), 1
    r0 = int(prob.node_row_ptr[2])
    row = np.asarray(blocks.J_alg[r0, ns * e : ns * e + 3].todense()).ravel()
    rho, c, u, p, area = (float(est[k, e]) for k in (ES_RHO, ES_C, ES_U, ES_P, ES_AREA))
    M = float(est[ES_M, e])
    # outlet row in characteristic coordinates: cf f + cg g + ch h = 0  ->  g = R f + R_s h
    cf, cg, ch = row @ char_to_dx(rho, c, u, p, area, K)
    R_inherit, Rs_inherit = -cf / cg, -ch / cg
    gm1 = GAMMA - 1.0
    R_mc = (2.0 - gm1 * M) / (2.0 + gm1 * M)
    Rs_mc = (c / rho) * M / (2.0 + gm1 * M)
    assert R_inherit == pytest.approx(R_mc, abs=1e-9)
    assert Rs_inherit == pytest.approx(Rs_mc, rel=1e-7)
