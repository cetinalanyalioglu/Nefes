"""Mean-flow checks for the sudden-area-change reverse (contraction) loss.

``sudden_area_change`` carries a vena-contracta contraction coefficient ``cc``:
reverse (large -> small) flow loses total pressure ``K_c * (1/2 rho u^2)_small``
with ``K_c = (1/cc - 1)^2``, referenced to the downstream (small) side.
``cc = 1`` (default) is loss-free and must reproduce the historical
total-pressure-continuous contraction.  Forward (expanding) flow is unaffected.

The ``1/2 rho u^2`` head is the incompressible reduction of the Borda momentum
balance, so the loss is exact only to ``O(M^2)``; these tests run at modest Mach.
"""

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_RHO, ES_U, ES_M, ES_PT

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)


def _contraction(cc, A_large=0.09, A_small=0.05, pt_in=120000.0, p_out=101325.0):
    """Solve inlet -> sudden contraction -> outlet; return the converged state table.

    Edge 0 (area ``A_large``) feeds the element and edge 1 (area ``A_small``) leaves
    it, so forward flow contracts large -> small and the small (downstream) port is
    edge 1.
    """
    net = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.sudden_area_change(cc=cc),
        cat.pressure_outlet(p_out, 300.0),
    ]
    prob = cat.build_problem(CFG, net, [(0, 1, A_large), (1, 2, A_small)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    assert est[ES_M, 1] > est[ES_M, 0]  # contraction accelerates the flow
    assert est[ES_M, 1] < 0.95  # subsonic (v1 scope)
    return est


def test_contraction_loss_matches_Kc():
    cc = 0.62
    est = _contraction(cc)
    K_c = (1.0 / cc - 1.0) ** 2
    q_small = 0.5 * est[ES_RHO, 1] * est[ES_U, 1] ** 2  # downstream (small) dynamic head
    dp_t = est[ES_PT, 0] - est[ES_PT, 1]  # large (up) -> small (down)
    assert dp_t > 0.0
    assert np.isclose(dp_t, K_c * q_small, rtol=1e-6)


def test_contraction_lossless_default_conserves_pt():
    # cc = 1 (default): the reverse branch is exact total-pressure continuity, i.e.
    # the historical behaviour, recovered byte-for-byte (K_c = 0).
    est = _contraction(1.0)
    assert np.isclose(est[ES_PT, 0], est[ES_PT, 1], rtol=1e-7)


def test_contraction_loss_grows_as_cc_drops():
    # a tighter contraction (smaller cc) dissipates more total pressure.
    losses = []
    for cc in (1.0, 0.8, 0.62):
        est = _contraction(cc)
        losses.append(est[ES_PT, 0] - est[ES_PT, 1])
    assert losses[0] < losses[1] < losses[2]
    assert losses[0] < 1e-6 * losses[2]  # cc = 1 is lossless to solver accuracy


def test_expansion_unaffected_by_cc():
    # forward EXPANSION (small -> large) runs the Borda momentum branch; cc tags the
    # inactive contraction branch, so the converged expansion must not move with cc.
    def expand(cc):
        net = [
            cat.total_pressure_inlet(115000.0, 300.0),
            cat.sudden_area_change(cc=cc),
            cat.pressure_outlet(101325.0, 300.0),
        ]
        prob = cat.build_problem(CFG, net, [(0, 1, 0.05), (1, 2, 0.09)], 10.0, 101325.0, CP * 300.0)
        res = solve(prob)
        assert res.converged
        return states_table(prob, res.x)

    base = expand(1.0)
    lossy = expand(0.5)
    assert base[ES_M, 0] > base[ES_M, 1]  # genuinely expanding (decelerates)
    # cc only leaks through the (1 - xi) ~ (eps/mdot)^2 saturated switch weight.
    assert np.isclose(base[ES_PT, 1], lossy[ES_PT, 1], rtol=1e-6)
    assert np.isclose(base[ES_M, 1], lossy[ES_M, 1], rtol=1e-6)


@pytest.mark.parametrize("cc", [-0.1, 0.0, 1.5])
def test_invalid_cc_rejected(cc):
    with pytest.raises(ValueError, match="contraction coefficient"):
        cat.sudden_area_change(cc=cc)
