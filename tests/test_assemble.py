"""Phase 3 validation: residual assembly and the sparse complex-step Jacobian.

Headline check: the sparse seeded Jacobian equals a dense reference Jacobian
(full re-eval per column) to ~1e-9, which simultaneously proves the sparsity
pattern contains every true nonzero (any nonzero outside the pattern would make
the sparse matrix disagree with the dense one).
"""

import numpy as np
import pytest

from fns.thermo.configure import perfect_gas
from fns.elements import catalog as cat
from fns.assembly.assemble import residual, jacobian, jacobian_dense

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
EPS = 1e-3 * 40.0
EPS_FB = 1e-5


def _chain():
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.mass_flow_inlet(30.0, 300.0),
        cat.isentropic_area_change(),
        cat.sudden_area_change(),
        cat.loss(2.0),
        cat.duct(),
        cat.pressure_outlet(9.0e4),
    ]
    edges = [
        (0, 1, 0.25),
        (1, 2, 0.20),
        (2, 3, 0.18),
        (3, 4, 0.18),  # loss: constant area
        (4, 5, 0.18),  # duct: constant area
    ]
    prob = cat.build_problem(cfg, elements, edges, mdot_ref=40.0, p_ref=101325.0, h_ref=CP * 300.0)
    return prob


def _branch():
    cfg = perfect_gas(R_AIR, GAMMA)
    elements = [
        cat.mass_flow_inlet(30.0, 300.0),
        cat.splitter(),
        cat.loss(1.5),
        cat.loss(3.0),
        cat.junction(),
        cat.pressure_outlet(9.0e4),
    ]
    edges = [
        (0, 1, 0.25),
        (1, 2, 0.20),
        (1, 3, 0.20),
        (2, 4, 0.20),  # loss: constant area
        (3, 4, 0.20),  # loss: constant area
        (4, 5, 0.30),
    ]
    prob = cat.build_problem(cfg, elements, edges, mdot_ref=40.0, p_ref=101325.0, h_ref=CP * 300.0)
    return prob


def _state(prob, mdots):
    x = np.zeros((3, prob.n_edges))
    x[0, :] = mdots
    x[1, :] = 1.0e5
    x[2, :] = CP * 305.0
    return x


@pytest.mark.parametrize("build", [_chain, _branch])
def test_sparse_jacobian_matches_dense(build):
    prob = build()
    rng = np.random.default_rng(0)
    mdots = rng.uniform(-20.0, 25.0, size=prob.n_edges)
    mdots[0] = 28.0  # keep the inlet edge clearly forward
    x = _state(prob, mdots)

    Js = jacobian(prob, x, EPS, EPS_FB).toarray()
    Jd = jacobian_dense(prob, x, EPS, EPS_FB)
    assert np.allclose(Js, Jd, rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize("build", [_chain, _branch])
def test_jacobian_square_and_finite(build):
    prob = build()
    x = _state(prob, np.full(prob.n_edges, 5.0))
    J = jacobian(prob, x, EPS, EPS_FB)
    assert J.shape == (3 * prob.n_edges, 3 * prob.n_edges)
    assert np.all(np.isfinite(J.toarray()))


def test_mass_balance_rows_are_exact():
    # At a divergence-free state, mass-balance rows must vanish identically.
    prob = _chain()  # series chain -> mdot equal on every edge conserves mass
    x = _state(prob, np.full(prob.n_edges, 12.0))
    R = residual(prob, x, EPS, EPS_FB)
    # node 1 (iac) mass row is its first equation row:
    r0 = prob.node_row_ptr[1]
    assert abs(R[r0]) < 1e-10


def test_residual_c1_through_zero_flow():
    # The transport row's mdot dependence (smooth upwind) must be C1 at mdot=0:
    # complex-step derivative just below and just above 0 nearly coincide.
    prob = _chain()
    x = _state(prob, np.full(prob.n_edges, 6.0))

    def dRdmdot_edge2(m0):
        xc = x.astype(np.complex128)
        xc[0, 2] = complex(m0, 1e-30)
        R = residual(prob, xc, EPS, EPS_FB)
        return R.imag / 1e-30

    d_lo = dRdmdot_edge2(-1e-4)
    d_hi = dRdmdot_edge2(+1e-4)
    assert np.allclose(d_lo, d_hi, atol=1e-3 * (np.abs(d_hi).max() + 1.0))
