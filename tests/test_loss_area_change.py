"""Physics of the concentrated loss straddling an area change.

The loss element drops total pressure by ``K`` dynamic heads referenced to one
chosen port (``catalog.loss(ref_port=...)``).  When its two ports carry different
areas the static state on each is reconstructed from that port's own area, so the
loss rides on top of an isentropic (Pt-preserving) area change.  These tests pin
the realised Pt drop to ``K * 1/2 rho u^2`` at the reference port and check that
the two ``ref_port`` choices differ exactly by the ratio of the ports' dynamic
heads.
"""

import pytest

from nefes.thermo.configure import perfect_gas
from nefes.elements import catalog as cat
from nefes.solver import solve
from nefes.solver.control import states_table
from nefes.assembly.derive import ES_MDOT, ES_RHO, ES_PT, ES_AREA

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
PT, TT, P_OUT = 130000.0, 300.0, 101325.0


def _solve_loss(K, A0, A1, ref_port):
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.total_pressure_inlet(PT, TT),
        cat.loss(K, ref_port=ref_port),
        cat.pressure_outlet(P_OUT, Tt_backflow=TT),
    ]
    prob = cat.build_problem(cfg, elements, [(0, 1, A0), (1, 2, A1)], 10.0, P_OUT, CP * TT)
    res = solve(prob)
    assert res.converged and res.residual_norm < 1e-9
    return states_table(prob, res.x)


@pytest.mark.parametrize("ref_port", [0, 1])
def test_pt_drop_matches_reference_dynamic_head(ref_port):
    K, A0, A1 = 1.8, 0.10, 0.06
    est = _solve_loss(K, A0, A1, ref_port)
    # dynamic head at the reference port, from that port's own velocity
    rho_avg = 0.5 * (est[ES_RHO, 0].real + est[ES_RHO, 1].real)
    ref_edge = 0 if ref_port == 0 else 1
    u_ref = abs(est[ES_MDOT, ref_edge].real) / (rho_avg * est[ES_AREA, ref_edge].real)
    q_ref = 0.5 * rho_avg * u_ref * u_ref
    pt_drop = est[ES_PT, 0].real - est[ES_PT, 1].real
    assert pt_drop == pytest.approx(K * q_ref, rel=1e-6)


def test_ref_port_choice_changes_the_drop():
    # the smaller downstream port carries the higher velocity, so referencing K to
    # it gives a strictly larger Pt drop than referencing the larger upstream port.
    # (The two solves run at different mass flow -- a bigger loss throttles the
    # flow -- so the ratio is not a clean area power; only the ordering is robust.)
    K, A0, A1 = 1.2, 0.10, 0.06
    drop0 = _solve_loss(K, A0, A1, 0)
    drop1 = _solve_loss(K, A0, A1, 1)
    d0 = drop0[ES_PT, 0].real - drop0[ES_PT, 1].real
    d1 = drop1[ES_PT, 0].real - drop1[ES_PT, 1].real
    assert d1 > d0


def test_equal_area_ref_port_is_immaterial():
    K, A = 2.5, 0.08
    e0 = _solve_loss(K, A, A, 0)
    e1 = _solve_loss(K, A, A, 1)
    d0 = e0[ES_PT, 0].real - e0[ES_PT, 1].real
    d1 = e1[ES_PT, 0].real - e1[ES_PT, 1].real
    assert d0 == pytest.approx(d1, rel=1e-9)


def test_reverse_flow_reverses_drop():
    # drive flow backwards by putting the high pressure at the nominal outlet
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.total_pressure_inlet(P_OUT, TT),
        cat.loss(2.0, ref_port=0),
        cat.pressure_outlet(PT, Tt_backflow=TT),
    ]
    prob = cat.build_problem(cfg, elements, [(0, 1, 0.10), (1, 2, 0.06)], 10.0, P_OUT, CP * TT)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    # flow runs e1 -> e0, so total pressure rises from port 0 to port 1
    assert est[ES_MDOT, 0].real < 0.0
    assert est[ES_PT, 1].real > est[ES_PT, 0].real


def test_ref_port_validation():
    with pytest.raises(ValueError, match="ref_port"):
        cat.loss(1.0, ref_port=2)
