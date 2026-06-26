"""The dynamic source face ``S(omega)`` of the perturbation operator (theory.md s12.4).

A mass source or a flame may carry a frequency-domain feedback: its injected mass /
heat release fluctuates with the unsteady flow elsewhere (a flame transfer function).
These tests pin the *specification* (transfer functions, the descriptor) and the
*operator stamping* (which rows the feedback lands on, with what sign and magnitude),
independently of the eigensolver -- the end-to-end physics is validated against the
analytical Rijke tube in ``test_rijke_stability.py``.
"""

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.elements.dynamic_source import (
    NTau,
    Constant,
    n_tau,
    tabulated,
    constant,
    as_transfer,
    DynamicSource,
    FlameResponseTerm,
    n_tau_flame,
    heat_release_response,
    mass_flow_response,
)
from fns.perturbation.boundary_bc import PerturbationBC
from fns.perturbation.characteristics import dq_to_dx
from fns.perturbation.operator import build_acoustic_blocks, assemble_acoustic, _assemble_reference
from fns.perturbation import eigenmodes
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_MDOT, ES_U, ES_RHO, ES_P, ES_AREA
from fns.thermo.configure import perfect_gas

R, GAMMA = 287.0, 1.4
CP = GAMMA * R / (GAMMA - 1.0)
A = 0.01


# ==========================================================================
# Transfer functions
# ==========================================================================


def test_ntau_value_and_phase():
    f = np.array([0.0, 50.0, 137.0])
    F = n_tau(0.7, 2.5e-3)(f)
    assert np.allclose(F, 0.7 * np.exp(-2j * np.pi * f * 2.5e-3))
    # under e^{+i w t} the response lags: phase decreases with frequency
    assert np.angle(F[2]) < np.angle(F[1]) < 0.0


def test_ntau_complex_frequency_is_analytic():
    # the stability eigenproblem evaluates at complex frequency -> must not error
    w = 2 * np.pi * (120.0 - 8.0j)
    F = NTau(1.0, 3e-3)(w / (2 * np.pi))
    assert np.isfinite(F).all()
    assert NTau(1.0, 3e-3).analytic
    assert NTau(1.0, 3e-3).max_delay == pytest.approx(3e-3)


def test_constant_broadcasts():
    F = constant(0.5 + 0.2j)(np.zeros(4))
    assert F.shape == (4,) and np.allclose(F, 0.5 + 0.2j)


def test_tabulated_recovers_samples_and_rejects_complex():
    f = np.linspace(10.0, 500.0, 25)
    vals = (1.0 + 0.3j) * np.exp(-2j * np.pi * f * 1e-3)  # a smooth curve
    tf = tabulated(f, vals)
    assert not tf.analytic
    # interpolation reproduces the samples to round-off
    assert np.allclose(tf(f), vals, atol=1e-6)
    # a complex frequency is not interpolatable -> explicit error
    with pytest.raises(ValueError, match="complex frequency"):
        tf(np.array([120.0 - 5.0j]))


def test_as_transfer_coercions():
    assert isinstance(as_transfer((0.5, 1e-3)), NTau)
    assert isinstance(as_transfer(0.9), Constant)
    assert isinstance(as_transfer(NTau(1.0, 1e-3)), NTau)
    wrapped = as_transfer(lambda f: np.ones_like(np.asarray(f, complex)))
    assert not wrapped.analytic  # a bare callable is treated as non-analytic


# ==========================================================================
# Descriptor + builders
# ==========================================================================


def test_descriptor_validation():
    with pytest.raises(ValueError, match="at least one"):
        DynamicSource(terms=[])
    with pytest.raises(ValueError, match="target"):
        DynamicSource(terms=[FlameResponseTerm((1.0, 1e-3), 0)], target="bogus")
    with pytest.raises(ValueError, match="quantity"):
        FlameResponseTerm((1.0, 1e-3), 0, quantity="w")


def test_descriptor_analytic_and_max_delay():
    ds = DynamicSource(terms=[FlameResponseTerm(n_tau(1.0, 2e-3), 0), FlameResponseTerm(n_tau(0.5, 5e-3), 1)])
    assert ds.analytic
    assert ds.max_delay == pytest.approx(5e-3)
    ds2 = DynamicSource(terms=[FlameResponseTerm(tabulated([1.0, 2.0], [1.0, 1.0]), 0)])
    assert not ds2.analytic


def test_builders_shapes():
    ds = n_tau_flame(0.8, 3e-3, ref_edge=1)
    assert ds.target == "Qdot" and len(ds.terms) == 1 and ds.terms[0].ref_edge == 1
    assert mass_flow_response((0.5, 1e-3), 2).target == "mdot"
    assert heat_release_response(0.9, 0, quantity="p").terms[0].quantity == "p"


# ==========================================================================
# Operator stamping: row placement, sign, magnitude
# ==========================================================================


def _flame_network(dynamic_source=None, mdot=0.02, Qdot=8.0e3):
    """inlet -> duct -> heat-release flame -> duct -> outlet (edges 0..3)."""
    els = [
        cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(0.5),
        cat.heat_release_flame(Qdot, dynamic_source=dynamic_source),
        cat.duct(0.5),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A)]
    prob = cat.build_problem(perfect_gas(R, GAMMA), els, edges, mdot_ref=mdot, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _u_functional(est, e, K):
    """Coefficients of u' = vec . (mdot', p', h_t') at edge e, and the mean u."""
    rho, u, p, area = est[ES_RHO, e], est[ES_U, e], est[ES_P, e], est[ES_AREA, e]
    vec = np.linalg.inv(dq_to_dx(rho, u, p, area, K))[1, :]
    return vec, float(u)


def test_heat_release_lands_on_downstream_energy_row_with_correct_sign():
    n_gain, ref = 0.9, 1  # drive on the edge just upstream of the flame
    prob, x = _flame_network()
    prob_s, xs = _flame_network(dynamic_source=n_tau_flame(n_gain, 2e-3, ref_edge=ref))
    np.testing.assert_allclose(x, xs, rtol=1e-10)  # the source must not touch the mean flow

    K = float(prob.tf[0]) / float(prob.tf[1])
    b0 = build_acoustic_blocks(prob, x)
    b1 = build_acoustic_blocks(prob_s, xs)

    # downstream (product) edge of the flame is edge 2; its energy transport row is the seat
    assert set(b1.flame_edges) == {2}
    e_down = 2
    row = int(prob.transport_row0) + e_down

    # isolate the source contribution at omega = 0 (F = n, real)
    D = (assemble_acoustic(0.0, b1) - assemble_acoustic(0.0, b0)).tocoo()
    touched = sorted(set(D.row.tolist()))
    assert touched == [row], f"source must touch only the downstream energy row, got {touched}"

    # magnitude + sign: residual gains -delta * n * (vec / u_bar) . x_ref
    est = states_table(prob, x)
    mdot_mag = float(est[ES_MDOT, e_down])  # forward flow, oriented outflow > 0
    delta = 8.0e3 / mdot_mag
    vec, ubar = _u_functional(est, ref, K)
    ns = int(prob.n_solve)
    Dcsr = D.tocsr()
    for v in range(3):
        got = Dcsr[row, ns * ref + v]
        expected = -delta * n_gain * (vec[v] / ubar)
        assert got == pytest.approx(expected, rel=1e-8, abs=1e-9)


def test_heat_release_is_frequency_dependent():
    prob, x = _flame_network(dynamic_source=n_tau_flame(0.9, 3e-3, ref_edge=1))
    blocks = build_acoustic_blocks(prob, x)
    assert blocks.has_sources
    row = int(prob.transport_row0) + 2
    A1 = assemble_acoustic(2 * np.pi * 100.0, blocks)
    A2 = assemble_acoustic(2 * np.pi * 200.0, blocks)
    # the n-tau phase makes the source row genuinely omega-dependent (not frozen)
    assert abs((A1 - A2)[row]).sum() > 1e-6
    # the fast assembler handles S(omega) directly (the transfer functions are accumulated
    # onto the source slots per frequency, no fallback); it matches the reference assembly
    # to round-off relative to the operator's magnitude (the source row carries ~1e7 entries).
    ref = _assemble_reference(2 * np.pi * 137.0, blocks)
    fast = assemble_acoustic(2 * np.pi * 137.0, blocks)
    assert abs((ref - fast)).max() < 1e-12 * abs(ref).max()


def test_q_mean_override_scales_the_coupling():
    # the auto-derived Q_bar for a heat-release flame is its Qdot (= 8 kW); an explicit
    # override of twice that must exactly double the source coupling.
    prob0, x0 = _flame_network()
    base = assemble_acoustic(0.0, build_acoustic_blocks(prob0, x0))

    p_auto, xa = _flame_network(n_tau_flame(1.0, 1e-3, 1))
    p_2x, xb = _flame_network(n_tau_flame(1.0, 1e-3, 1, q_mean=2.0 * 8.0e3))
    Da = assemble_acoustic(0.0, build_acoustic_blocks(p_auto, xa)) - base
    Db = assemble_acoustic(0.0, build_acoustic_blocks(p_2x, xb)) - base
    assert abs(Db).max() == pytest.approx(2.0 * abs(Da).max(), rel=1e-8)


def test_isentropic_keeps_the_flame_edge_physical():
    prob, x = _flame_network(dynamic_source=n_tau_flame(0.9, 2e-3, ref_edge=1))
    blocks = build_acoustic_blocks(prob, x, isentropic=True)
    A = assemble_acoustic(2 * np.pi * 120.0, blocks)
    tr0 = int(prob.transport_row0)
    # a pinned (isentropic) edge has an entropy-only row: its h_t (energy) column carries
    # the entropy coefficient, and the row does not couple to the flame's heat release.
    # The active-flame downstream edge (2) is exempt -> its row still carries the source.
    pinned_edge, flame_edge = 0, 2
    ns = int(prob.n_solve)
    # the flame energy row couples to the *reference* edge (1); a pinned row never does
    assert abs(A[tr0 + flame_edge, ns * 1 + 0]) > 0.0
    assert abs(A[tr0 + pinned_edge, ns * 1 + 0]) == 0.0


# ==========================================================================
# Mass-source feedback lands on the node rows (mass + momentum)
# ==========================================================================


def _mass_source_network(dynamic_source=None, mdot_in=0.05, mdot_src=0.01, u_inj=30.0):
    els = [
        cat.mass_flow_inlet(mdot_in, 300.0, perturbation_bc=PerturbationBC.hard_wall()),
        cat.duct(0.5),
        cat.mass_source(mdot_src, 300.0, None, u_inj=u_inj, dynamic_source=dynamic_source),
        cat.duct(0.5),
        cat.pressure_outlet(1.0e5, perturbation_bc=PerturbationBC.open_end()),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A)]
    prob = cat.build_problem(perfect_gas(R, GAMMA), els, edges, mdot_ref=mdot_in, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_mass_source_feedback_on_node_rows():
    mdot_src, u_inj, n_gain, ref = 0.01, 30.0, 0.6, 1
    ds = mass_flow_response(n_tau(n_gain, 1e-3), ref_edge=ref)
    prob, x = _mass_source_network()
    prob_s, xs = _mass_source_network(dynamic_source=ds, mdot_src=mdot_src, u_inj=u_inj)
    K = float(prob.tf[0]) / float(prob.tf[1])
    b0, b1 = build_acoustic_blocks(prob, x), build_acoustic_blocks(prob_s, xs)
    # the injected enthalpy modulates the outflow entropy row, so the outflow edge is an active source edge
    (e_out,) = tuple(b1.flame_edges)

    # the feedback lands on the mass-source node rows r0 (mass) and r0+1 (momentum) plus the outflow
    # energy transport row (the fuel pulse drags the convected total enthalpy at the outflow)
    n_src = 2
    r0 = int(prob.node_row_ptr[n_src])
    tr0 = int(prob.transport_row0)
    D = (assemble_acoustic(0.0, b1) - assemble_acoustic(0.0, b0)).tocoo()
    assert sorted(set(D.row.tolist())) == sorted([r0, r0 + 1, tr0 + e_out])

    est = states_table(prob, x)
    vec, ubar = _u_functional(est, ref, K)
    a0 = float(est[ES_AREA, int(prob.col_edge[int(prob.row_ptr[n_src])])])
    ns = int(prob.n_solve)
    # outflow energy-row factor -mdot_src (h_t,src - h_t,out) / mdot_out, evaluated at the perturbed mean state
    ests = states_table(prob_s, xs)
    pb = int(prob_s.npar_fptr[n_src])
    factor_e = -mdot_src * (float(prob_s.npar_f[pb + 2]) - float(xs[2, e_out])) / float(ests[ES_MDOT, e_out])
    Dcsr = D.tocsr()
    for v in range(3):
        coeff = n_gain * vec[v] / ubar
        assert Dcsr[r0, ns * ref + v] == pytest.approx(-mdot_src * coeff, rel=1e-8, abs=1e-12)
        assert Dcsr[r0 + 1, ns * ref + v] == pytest.approx(-mdot_src * u_inj / a0 * coeff, rel=1e-8, abs=1e-12)
        assert Dcsr[tr0 + e_out, ns * ref + v] == pytest.approx(factor_e * coeff, rel=1e-8, abs=1e-12)


# ==========================================================================
# Stability driver guards
# ==========================================================================


def test_stability_rejects_nonanalytic_transfer():
    f = np.linspace(10.0, 400.0, 20)
    ds = heat_release_response(tabulated(f, np.ones_like(f, dtype=complex)), ref_edge=1)
    prob, x = _flame_network(dynamic_source=ds)
    with pytest.raises(ValueError, match="analytically continuable"):
        eigenmodes(prob, x, freq_band=(50.0, 200.0), isentropic=True)
