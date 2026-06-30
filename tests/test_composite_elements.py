"""Composite elements: build-time expansion + the Class-1 macro recipes.

A composite expands to atomic elements + internal edges at build time; the solver and
perturbation layers never see it.  The headline guarantee is *round-trip identity*: an
``orifice`` composite solves byte-for-byte like a hand-placed ``iac + sac`` on the same
boundary edges (the De Domenico entropy_generator reference), with the user's declared
edge ids preserved (internals appended at the tail).
"""

import warnings

import numpy as np
import pytest

from fns.thermo.configure import perfect_gas
from fns.elements import catalog as cat
from fns.elements.composite import CompositeElementSpec, expand_composites, validate_composite, is_composite
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_M, ES_P, ES_RHO, ES_U
from fns.perturbation import perturbation_response
from fns.shell import Network

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
A1, AT, A2 = 3.0e-3, 1.0e-3, 3.0e-3
PT, P0, T0 = 140000.0, 101325.0, 300.0


def _orifice_reference():
    """Hand-built De Domenico orifice: inlet -> iac -> sac -> outlet (the notebook reference)."""
    els = [
        cat.total_pressure_inlet(PT, T0),
        cat.isentropic_area_change(),
        cat.sudden_area_change(),
        cat.pressure_outlet(P0, T0),
    ]
    prob = cat.build_problem(CFG, els, [(0, 1, A1), (1, 2, AT), (2, 3, A2)], 1.0, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _orifice_composite(throat=AT):
    """The same orifice as one composite element: inlet -> orifice -> outlet."""
    els = [cat.total_pressure_inlet(PT, T0), cat.orifice(throat), cat.pressure_outlet(P0, T0)]
    prob = cat.build_problem(CFG, els, [(0, 1, A1), (1, 2, A2)], 1.0, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _scatter(prob, x, a, b, f=120.0):
    return perturbation_response(prob, x, np.array([f])).acoustic_scattering_matrix(a, b)[0]


# -- expansion mechanics ---------------------------------------------------------------------------


def test_no_composite_is_zero_overhead():
    # a composite-free element list passes through expand_composites unchanged (map None)
    els = [cat.total_pressure_inlet(PT, T0), cat.duct(0.1), cat.pressure_outlet(P0, T0)]
    edges = [(0, 1, A1), (1, 2, A1)]
    out_els, out_edges, cmap = expand_composites(els, edges)
    assert out_els is els and out_edges is edges and cmap is None
    prob = cat.build_problem(CFG, els, edges, 1.0, P0, CP * T0)
    assert prob.composite_map is None


def test_expansion_appends_and_preserves_user_ids():
    # the orifice keeps its node id (1); the second sub-element and the throat edge append
    els = [cat.total_pressure_inlet(PT, T0), cat.orifice(AT), cat.pressure_outlet(P0, T0)]
    edges = [(0, 1, A1), (1, 2, A2)]
    out_els, out_edges, cmap = expand_composites(els, edges)
    assert len(out_els) == 4 and len(out_edges) == 3  # +1 node, +1 internal edge
    # user nodes 0, 2 unchanged; orifice's iac keeps slot 1, sac appends at 3
    assert cmap.user_node_to_expanded == ((0,), (1, 3), (2,))
    assert cmap.internal_nodes == frozenset({3}) and cmap.internal_edges == frozenset({2})
    # user edges keep their endpoints/areas; only the orifice endpoint is rewired
    assert out_edges[0] == (0, 1, A1)  # inlet -> iac (unchanged)
    assert out_edges[1] == (3, 2, A2)  # sac -> outlet (tail rewired to the downstream sub)
    assert out_edges[2] == (1, 3, AT)  # internal iac -> sac at the throat area


# -- round-trip identity (the must-pass test) ------------------------------------------------------


def test_orifice_roundtrip_mean_flow():
    pr, xr = _orifice_reference()
    pc, xc = _orifice_composite()
    er, ec = states_table(pr, xr), states_table(pc, xc)
    # the user edges (inlet = 0, outlet = 1) match the reference inlet (0) and outlet (2)
    for q in (ES_M, ES_P, ES_RHO, ES_U):
        assert ec[q, 0] == pytest.approx(er[q, 0], rel=1e-7, abs=1e-7)  # inlet
        assert ec[q, 1] == pytest.approx(er[q, 2], rel=1e-7, abs=1e-7)  # outlet
    # the throat (composite internal edge 2) matches the reference throat (edge 1)
    assert ec[ES_M, 2] == pytest.approx(er[ES_M, 1], rel=1e-7)


def test_orifice_roundtrip_scattering_matrix():
    pr, xr = _orifice_reference()
    pc, xc = _orifice_composite()
    # edge-indexed: reference inlet(0) -> outlet(2) vs composite inlet(0) -> outlet(1)
    assert np.allclose(_scatter(pr, xr, 0, 2), _scatter(pc, xc, 0, 1), atol=1e-6)


def test_lossy_nozzle_orifice_limit_matches_orifice():
    # beta = AT/A2 -> the orifice (maximum loss)
    els = [cat.total_pressure_inlet(PT, T0), cat.lossy_nozzle(AT, AT / A2, A2), cat.pressure_outlet(P0, T0)]
    pn = cat.build_problem(CFG, els, [(0, 1, A1), (1, 2, A2)], 1.0, P0, CP * T0)
    rn = solve(pn)
    assert rn.converged
    pc, xc = _orifice_composite()
    assert np.allclose(_scatter(pn, rn.x, 0, 1), _scatter(pc, xc, 0, 1), atol=1e-6)


def test_lossy_nozzle_lossless_limit_conserves_total_pressure():
    # beta = 1 -> the lossless (isentropic) nozzle: total pressure is conserved across it
    # (the Borda re-expansion is A2 -> A2, its loss term vanishes), unlike the lossy orifice.
    # Use a mild throat (PT 110 kPa, AT = 2/3 A2) so the con-di flow stays comfortably subsonic.
    from fns.derive import ES_PT

    pti, at = 110000.0, 2.0e-3

    def pt_drop(el):
        prob = cat.build_problem(
            CFG,
            [cat.total_pressure_inlet(pti, T0), el, cat.pressure_outlet(P0, T0)],
            [(0, 1, A1), (1, 2, A2)],
            1.0,
            P0,
            CP * T0,
        )
        res = solve(prob)
        assert res.converged
        est = states_table(prob, res.x)
        return float(est[ES_PT, 0] - est[ES_PT, 1])  # inlet - outlet total pressure

    assert pt_drop(cat.lossy_nozzle(at, 1.0, A2)) == pytest.approx(0.0, abs=1e-3 * pti)  # lossless
    assert pt_drop(cat.lossy_nozzle(at, at / A2, A2)) > 0.01 * pti  # the orifice limit loses head


def test_lossy_nozzle_rejects_out_of_range_beta():
    with pytest.raises(ValueError, match="beta must lie"):
        cat.lossy_nozzle(AT, 1.5, A2)
    with pytest.raises(ValueError, match="beta must lie"):
        cat.lossy_nozzle(AT, 0.01, A2)


# -- helmholtz_resonator (a side-branch composite) -------------------------------------------------


def _hr_peak(build, freqs):
    net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
    build(net)
    sol = net.solve()
    assert sol.converged
    resp = perturbation_response(sol.problem, sol.x, freqs)
    with warnings.catch_warnings():  # inlet/outlet straddle the tee branch point (expected)
        warnings.simplefilter("ignore")
        tau = resp.acoustic_scattering_matrix(0, 3)[:, 1, 0]
    tl = -20.0 * np.log10(np.abs(tau))
    return float(freqs[int(np.argmax(tl))]), float(np.max(tl)), sol


def test_helmholtz_resonator_matches_hand_built():
    V, AN, LN, AM, LM = 1.0e-3, 5.0e-4, 0.02, 3.0e-3, 0.05
    freqs = np.linspace(50.0, 1100.0, 1100)

    def hand(net):
        i, d1, tee = net.add(cat.total_pressure_inlet(P0, T0)), net.add(cat.duct(LM)), net.add(cat.junction())
        d2, o = net.add(cat.duct(LM)), net.add(cat.pressure_outlet(P0, T0))
        nk, cv = net.add(cat.duct(LN)), net.add(cat.cavity(V))
        net.connect(i, d1, AM), net.connect(d1, tee, AM), net.connect(tee, d2, AM), net.connect(d2, o, AM)
        net.connect(tee, nk, AN), net.connect(nk, cv, AN)

    def comp(net):
        i, d1, hr = (
            net.add(cat.total_pressure_inlet(P0, T0)),
            net.add(cat.duct(LM)),
            net.add(cat.helmholtz_resonator(V, LN, AN)),
        )
        d2, o = net.add(cat.duct(LM)), net.add(cat.pressure_outlet(P0, T0))
        net.connect(i, d1, AM), net.connect(d1, hr, AM), net.connect(hr, d2, AM), net.connect(d2, o, AM)

    f_hand, tl_hand, _ = _hr_peak(hand, freqs)
    f_comp, tl_comp, sol = _hr_peak(comp, freqs)
    assert f_comp == pytest.approx(f_hand, abs=2.0)
    assert tl_comp == pytest.approx(tl_hand, rel=0.05)
    assert sol.problem.composite_map is not None


# -- projection / view helpers ---------------------------------------------------------------------


def test_composite_view_reads_the_throat():
    net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
    i = net.add(cat.total_pressure_inlet(PT, T0))
    orf = net.add(cat.orifice(AT, name="orifice"))
    o = net.add(cat.pressure_outlet(P0, T0))
    net.connect(i, orf, A1)
    net.connect(orf, o, A2)
    sol = net.solve()
    assert sol.converged
    cv = sol.composite("orifice")
    assert cv.name == "orifice" and cv.kind == "orifice" and cv.node == orf
    assert cv.throat is not None
    assert cv.throat_state["area"] == pytest.approx(AT)  # the narrowest section
    assert 0.0 < cv.throat_state["M"] < 1.0  # subsonic throat
    # composites list + lookup by node id agree
    assert [c.name for c in sol.composites] == ["orifice"]
    assert sol.composite(orf).throat == cv.throat


def test_show_internal_hides_composite_edges():
    pc, xc = _orifice_composite()
    net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
    i = net.add(cat.total_pressure_inlet(PT, T0))
    orf = net.add(cat.orifice(AT))
    o = net.add(cat.pressure_outlet(P0, T0))
    net.connect(i, orf, A1)
    net.connect(orf, o, A2)
    sol = net.solve()
    assert sol.table().shape[1] == 3  # all edges (incl. the throat)
    assert sol.table(show_internal=False).shape[1] == 2  # only the user-facing edges


# -- validation ------------------------------------------------------------------------------------


def test_validate_composite_rejects_bad_recipes():
    # too few sub-elements
    with pytest.raises(ValueError, match=">= 2 sub-elements"):
        validate_composite(CompositeElementSpec("x", [cat.duct(0.1)], []))
    # nested composite
    with pytest.raises(ValueError, match="nested composites"):
        validate_composite(CompositeElementSpec("x", [cat.orifice(AT), cat.duct(0.1)], []))
    # internal edge out of range
    with pytest.raises(ValueError, match="out of range"):
        validate_composite(CompositeElementSpec("x", [cat.duct(0.1), cat.duct(0.1)], [(0, 5, A1)]))
    # degree mismatch: a 2-port iac wired to only its internal edge (no external) is degree 1
    bad = CompositeElementSpec(
        "x",
        [cat.isentropic_area_change(), cat.isentropic_area_change()],
        [(0, 1, AT)],
        upstream_sub=0,
        downstream_sub=0,
    )
    with pytest.raises(ValueError, match="port"):
        validate_composite(bad)


def test_is_composite_discriminates():
    assert is_composite(cat.orifice(AT))
    assert not is_composite(cat.duct(0.1))
