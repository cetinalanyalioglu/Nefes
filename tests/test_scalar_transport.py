"""Phase-3 validation: the composition / passive-scalar transport framework.

A passive scalar rides the perfect-gas backend (which ignores ``Z_el``), so the
mean flow is unchanged while one extra conserved scalar is advected per edge.  We
check mass-weighted mixing at a junction, realizability (the donor mix is a convex
combination), and that the sparse Jacobian pattern/fill still matches the dense
complex-step reference once a second advected scalar is present.
"""

import numpy as np
import pytest

from fns.assemble import jacobian, jacobian_dense
from fns.elements import catalog as cat
from fns.solver import solve
from fns.solver.control import initial_guess
from fns.thermo.configure import perfect_gas_passive_scalars

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
EPS, EPS_FB = 1e-4, 1e-5
A = 0.1


def _mixing_network(tracerA, tracerB, mdotA=2.0, mdotB=1.0):
    """Two mass-flow inlets carrying different tracer values, mixed at a junction."""
    cfg = perfect_gas_passive_scalars(1, R_AIR, GAMMA, names=["tracer"])
    Tt = 300.0
    elements = [
        cat.mass_flow_inlet(mdotA, Tt, composition=[tracerA], name="inA"),
        cat.mass_flow_inlet(mdotB, Tt, composition=[tracerB], name="inB"),
        cat.junction(name="mix"),
        cat.pressure_outlet(101325.0, Tt_backflow=Tt, name="out"),
    ]
    edges = [(0, 2, A), (1, 2, A), (2, 3, A)]
    return cat.build_problem(cfg, elements, edges, mdot_ref=3.0, p_ref=101325.0, h_ref=CP * Tt)


def test_problem_has_extra_scalar():
    prob = _mixing_network(1.0, 0.0)
    assert prob.n_solve == 4  # mdot, p, h_t, tracer
    assert prob.n_elem == 1
    # one transport row per advected scalar (h_t, tracer) per edge
    assert prob.n_eq == prob.transport_row0 + 2 * prob.n_edges


def test_passive_tracer_mixes_mass_weighted():
    prob = _mixing_network(tracerA=1.0, tracerB=0.0, mdotA=2.0, mdotB=1.0)
    res = solve(prob)
    assert res.converged
    tr = res.x[3]  # tracer row, per edge
    # feed edges carry their inlet values; the mixed edge is the mass-weighted mean
    assert tr[0] == pytest.approx(1.0, abs=1e-6)
    assert tr[1] == pytest.approx(0.0, abs=1e-6)
    assert tr[2] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_passive_tracer_does_not_perturb_mean_flow():
    """The passive scalar must not change the (mdot, p, h_t) solution."""
    prob_tr = _mixing_network(0.7, 0.2)
    res_tr = solve(prob_tr)
    # same network without the scalar
    from fns.thermo.configure import perfect_gas

    Tt = 300.0
    elements = [
        cat.mass_flow_inlet(2.0, Tt, name="inA"),
        cat.mass_flow_inlet(1.0, Tt, name="inB"),
        cat.junction(name="mix"),
        cat.pressure_outlet(101325.0, Tt_backflow=Tt, name="out"),
    ]
    edges = [(0, 2, A), (1, 2, A), (2, 3, A)]
    prob_pg = cat.build_problem(perfect_gas(R_AIR, GAMMA), elements, edges, 3.0, 101325.0, CP * Tt)
    res_pg = solve(prob_pg)
    assert res_tr.converged and res_pg.converged
    np.testing.assert_allclose(res_tr.x[:3], res_pg.x[:3], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("a,b", [(1.0, 0.0), (0.3, 0.7), (0.5, 0.5)])
def test_passive_tracer_realizable(a, b):
    prob = _mixing_network(a, b)
    res = solve(prob)
    assert res.converged
    tr = res.x[3]
    assert np.all(tr >= -1e-9)
    assert np.all(tr <= 1.0 + 1e-9)
    # the mixed value lies between the two feeds (a small O(eps^2) smoothing bias)
    assert min(a, b) - 1e-6 <= tr[2] <= max(a, b) + 1e-6


def test_sparse_jacobian_matches_dense_with_scalar():
    """The CSC pattern + complex-step fill must match the dense reference for n_scalars=2."""
    prob = _mixing_network(0.6, 0.1)
    res = solve(prob)
    assert res.converged
    x = res.x
    Js = jacobian(prob, x, EPS, EPS_FB).toarray()
    Jd = jacobian_dense(prob, x, EPS, EPS_FB)
    assert Js.shape == Jd.shape
    np.testing.assert_allclose(Js, Jd, atol=1e-7, rtol=1e-6)


def test_initial_guess_shape():
    prob = _mixing_network(1.0, 0.0)
    x0 = initial_guess(prob)
    assert x0.shape == (4, prob.n_edges)
