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

from nefes.assembly.recover import ES_AREA, ES_M, ES_P, ES_PT, ES_RHO, ES_U
from nefes.elements import catalog as cat
from nefes.elements.composite import CompositeElementSpec, expand_composites, is_composite, validate_composite
from nefes.perturbation import perturbation_response
from nefes.shell import Network
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

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
    prob = build_problem(CFG, els, [(0, 1, A1), (1, 2, AT), (2, 3, A2)], 1.0, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _orifice_composite(throat=AT):
    """The same orifice as one composite element: inlet -> orifice -> outlet."""
    els = [cat.total_pressure_inlet(PT, T0), cat.orifice(throat), cat.pressure_outlet(P0, T0)]
    prob = build_problem(CFG, els, [(0, 1, A1), (1, 2, A2)], 1.0, P0, CP * T0)
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
    prob = build_problem(CFG, els, edges, 1.0, P0, CP * T0)
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
    # user edges keep their endpoints/areas; only the orifice endpoint is rewired.  Each edge
    # carries explicit flow-aligned ports (..., tail_port, head_port): a 2-port sub-element
    # takes its inflow on port 0 and its outflow on port 1, so the throat's iac/sac are wired
    # port 0 (in) / port 1 (out) exactly as by hand.
    assert out_edges[0] == (0, 1, A1, 0, 0)  # inlet -> iac: iac inflow on port 0
    assert out_edges[1] == (3, 2, A2, 1, 0)  # sac -> outlet: sac outflow on port 1
    assert out_edges[2] == (1, 3, AT, 1, 0)  # internal iac (out, port 1) -> sac (in, port 0)


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
    els = [cat.total_pressure_inlet(PT, T0), cat.lossy_nozzle(AT, AT / A2), cat.pressure_outlet(P0, T0)]
    pn = build_problem(CFG, els, [(0, 1, A1), (1, 2, A2)], 1.0, P0, CP * T0)
    rn = solve(pn)
    assert rn.converged
    pc, xc = _orifice_composite()
    assert np.allclose(_scatter(pn, rn.x, 0, 1), _scatter(pc, xc, 0, 1), atol=1e-6)


def test_lossy_nozzle_lossless_limit_conserves_total_pressure():
    # beta = 1 -> the lossless (isentropic) nozzle: total pressure is conserved across it
    # (the Borda re-expansion is A2 -> A2, its loss term vanishes), unlike the lossy orifice.
    # Use a mild throat (PT 110 kPa, AT = 2/3 A2) so the con-di flow stays comfortably subsonic.
    from nefes.assembly.recover import ES_PT

    pti, at = 110000.0, 2.0e-3

    def pt_drop(el):
        prob = build_problem(
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

    assert pt_drop(cat.lossy_nozzle(at, 1.0)) == pytest.approx(0.0, abs=1e-3 * pti)  # lossless
    assert pt_drop(cat.lossy_nozzle(at, at / A2)) > 0.01 * pti  # the orifice limit loses head


def test_lossy_nozzle_rejects_out_of_range_beta():
    # beta > 1 is rejected at the factory; beta < AT/A2 needs the outflow edge area,
    # so it is rejected at build time, when the composite reads A2 off its edge.
    with pytest.raises(ValueError, match="beta must lie"):
        cat.lossy_nozzle(AT, 1.5)
    els = [cat.total_pressure_inlet(PT, T0), cat.lossy_nozzle(AT, 0.01), cat.pressure_outlet(P0, T0)]
    with pytest.raises(ValueError, match="beta must lie"):
        build_problem(CFG, els, [(0, 1, A1), (1, 2, A2)], 1.0, P0, CP * T0)


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
    cv = sol.composite("orifice")  # an explicitly chosen name is kept, even when it equals the default
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


# -- Class-2 discretization composites (fanno_pipe, tapered_duct) -----------------------------------


def _fanno(n, length=8.0, diameter=0.05, friction=0.03, mdot=0.55):
    area = np.pi * diameter**2 / 4.0
    els = [cat.mass_flow_inlet(mdot, T0), cat.fanno_pipe(length, diameter, friction, n), cat.pressure_outlet(P0, T0)]
    prob = build_problem(CFG, els, [(0, 1, area), (1, 2, area)], mdot, P0, CP * T0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_fanno_pipe_chain_converges_in_n():
    # the exit Mach settles as the chain is refined; N=1 is exactly the lumped pipe atom
    exits = []
    for n in (1, 4, 16, 64):
        prob, x = _fanno(n)
        assert prob.n_edges == n + 1  # N segments joined by N-1 internal edges, + 2 boundary edges
        exits.append(float(states_table(prob, x)[ES_M, 1]))
    assert abs(exits[-1] - exits[-2]) < 1e-3  # converged between N=16 and N=64
    # the lumped pipe atom is the N=1 limit
    pp = build_problem(
        CFG,
        [cat.mass_flow_inlet(0.55, T0), cat.pipe(8.0, 0.05, 0.03), cat.pressure_outlet(P0, T0)],
        [(0, 1, np.pi * 0.05**2 / 4), (1, 2, np.pi * 0.05**2 / 4)],
        0.55,
        P0,
        CP * 300.0,
    )
    sp = solve(pp)
    assert states_table(pp, sp.x)[ES_M, 1] == pytest.approx(exits[0], rel=1e-9)


def test_fanno_pipe_resolves_the_mach_rise():
    # constant-area friction accelerates the subsonic flow toward the exit (Fanno)
    prob, x = _fanno(32)
    est = states_table(prob, x)
    assert est[ES_M, 1] > est[ES_M, 0]  # exit faster than inlet
    # the interior Mach increases monotonically along the chain (a resolved gradient)
    internal = sorted(prob.composite_map.internal_edges) if prob.composite_map else []
    Ms = [est[ES_M, 0]] + [est[ES_M, e] for e in internal] + [est[ES_M, 1]]
    assert all(Ms[i + 1] >= Ms[i] - 1e-9 for i in range(len(Ms) - 1))


def _condi(n, pt_in, length=0.4, a_in=3.0e-3, a_th=1.5e-3, a_out=3.0e-3):
    half = n // 2
    areas = list(np.linspace(a_in, a_th, half + 1)) + list(np.linspace(a_th, a_out, n - half + 1))[1:]
    table = list(zip(np.linspace(0.0, length, len(areas)), areas))  # (x, A) pairs; length inferred
    net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
    i = net.add(cat.total_pressure_inlet(pt_in, T0))
    td = net.add(cat.tapered_duct(table, name="nozzle"))
    o = net.add(cat.pressure_outlet(P0, T0))
    net.connect(i, td, a_in)
    net.connect(td, o, a_out)
    sol = net.solve()
    assert sol.converged
    return sol


def test_tapered_duct_throat_is_narrowest_and_fastest():
    # a subsonic con-di: the composite's throat is the min-area edge and carries the peak Mach
    sol = _condi(8, pt_in=108000.0)
    est = sol.table()
    cv = sol.composite("nozzle")  # an explicitly chosen name is kept
    assert cv.throat is not None
    assert est[ES_AREA, cv.throat] == pytest.approx(est[ES_AREA].min())  # the narrowest edge
    assert est[ES_M, cv.throat] == pytest.approx(est[ES_M].max(), rel=1e-6)  # the fastest
    assert est[ES_M, cv.throat] < 1.0  # subsonic


def test_tapered_duct_from_callable_matches_table():
    # a callable A(x) sampled at N+1 stations builds the same chain as the explicit table
    import numpy as _np

    L, N = 0.5, 6

    def A(x):
        return 3.0e-3 - 1.5e-3 * _np.sin(_np.pi * x / L)  # a smooth contraction-expansion

    table = [(L * k / N, A(L * k / N)) for k in range(N + 1)]  # (x, A) at the same stations
    a0, aN = table[0][1], table[-1][1]

    def build(spec):
        net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
        i = net.add(cat.total_pressure_inlet(108000.0, T0))
        td = net.add(spec)
        o = net.add(cat.pressure_outlet(P0, T0))
        net.connect(i, td, a0)
        net.connect(td, o, aN)
        sol = net.solve()
        assert sol.converged
        return sol.table()

    e_call = build(cat.tapered_duct(A, length=L, n_segments=N))
    e_tab = build(cat.tapered_duct(table))  # length inferred from x
    assert np.allclose(e_call, e_tab, rtol=1e-9, atol=1e-9)


def test_tapered_duct_nonuniform_stations_give_nonuniform_ducts():
    # (x, A) with non-uniform spacing -> each segment's duct spans its own x-interval, and the
    # total propagation length is the inferred x-span (not a uniform L/N tiling)
    table = [(0.0, 3.0e-3), (0.05, 1.5e-3), (0.30, 3.0e-3)]  # a station clustered near the throat
    spec = cat.tapered_duct(table, name="nozzle")
    duct_lengths = [spec.sub_elements[2 * i + 1].fparams[0] for i in range(spec.n_sub // 2)]
    assert duct_lengths == pytest.approx([0.05, 0.25])  # the two station intervals, non-uniform
    assert sum(duct_lengths) == pytest.approx(0.30)  # total length inferred from x-span


def test_tapered_duct_rejects_inconsistent_n_segments():
    table = [(0.0, 3e-3), (0.15, 2e-3), (0.3, 3e-3)]
    with pytest.raises(ValueError, match="n_segments"):
        cat.tapered_duct(table, n_segments=5)


def test_tapered_duct_rejects_flat_area_list():
    # a flat area-only list is rejected -- the message points at the (x, A) form
    with pytest.raises(ValueError, match=r"\(x, area\) pairs"):
        cat.tapered_duct([3e-3, 2e-3, 3e-3], length=0.3)


def test_tapered_duct_length_must_match_x_span():
    table = [(0.0, 3e-3), (0.15, 2e-3), (0.3, 3e-3)]
    with pytest.raises(ValueError, match="does not match"):
        cat.tapered_duct(table, length=0.9)


def test_grid_refine_reports_convergence():
    from nefes.elements.composite import grid_refine

    def build(n):
        prob, x = _fanno(n)
        return states_table(prob, x)

    gr = grid_refine(build, 16, lambda est: {"M_exit": float(est[ES_M, 1])})
    assert gr.n_coarse == 16 and gr.n_fine == 32
    assert gr.converged(tol=1e-2) and gr.worst < 1e-2


def test_tapered_duct_scattering_matrix_converges_with_refinement():
    # The mean flow of a taper is N-exact, but its ACOUSTICS are not: a compact area change
    # reflects strongly and carries no transit phase; a refined (distributed) horn is a gradual
    # impedance match.  This pins (a) that a tapered_duct assembles acoustically at all -- its
    # internal ducts are correctly port-wired (port 0 in, port 1 out) -- and (b) that the
    # scattering matrix genuinely moves under refinement and settles.
    L, Ain, Aout = 0.4, 4.0e-3, 2.0e-3

    def area(x):
        return Ain + (Aout - Ain) * (x / L)

    def S(n, f=1500.0):
        els = [
            cat.total_pressure_inlet(1.08e5, T0),
            cat.tapered_duct(area, length=L, n_segments=n),
            cat.pressure_outlet(P0, T0),
        ]
        prob = build_problem(CFG, els, [(0, 1, Ain), (1, 2, Aout)], 4.0, P0, CP * T0)
        resp = perturbation_response(prob, solve(prob).x, np.array([f]), excite=("acoustic",))
        return np.asarray(resp.scattering_matrix(0, prob.n_edges - 1)).reshape(2, 2)

    S1, S8, S16, S32 = S(1), S(8), S(16), S(32)
    # the compact (N=1) reflection is far from the resolved horn (empirically ~0.48 vs ~0.04)
    assert abs(S1[0, 0]) > 3.0 * abs(S32[0, 0])
    # and refinement converges: the N=16 -> N=32 change is smaller than N=8 -> N=32
    assert np.linalg.norm(S16 - S32) < np.linalg.norm(S8 - S32)


def test_auto_refine_converges_on_a_fanno_pipe():
    from nefes.elements.composite import GridRefinement, auto_refine

    def build(n):
        prob, x = _fanno(n)
        return states_table(prob, x)

    ar = auto_refine(build, 4, lambda est: {"M_exit": float(est[ES_M, 1])}, tol=1e-2, max_refine=6)
    assert ar.converged
    assert ar.worst < 1e-2
    # the doubling history is exposed, oldest first, each a coarse->fine GridRefinement
    assert all(isinstance(s, GridRefinement) for s in ar.steps)
    assert ar.n_refine == len(ar.steps) and ar.n_refine >= 1
    assert ar.steps[0].n_coarse == 4 and ar.steps[0].n_fine == 8
    assert ar.n_final == 4 * 2**ar.n_refine  # finest resolution actually solved
    assert "M_exit" in ar.final


def test_auto_refine_respects_the_max_refine_cap():
    from nefes.elements.composite import auto_refine

    # a quantity that keeps drifting (never settles) must stop at the cap, not run forever
    calls = []

    def build(n):
        calls.append(n)
        return n

    def probe(n):
        return {"q": float(np.log(n))}  # log(N) never converges under doubling

    ar = auto_refine(build, 2, probe, tol=1e-3, max_refine=3)
    assert not ar.converged
    assert ar.n_refine == 3  # exactly the cap
    assert ar.n_final == 2 * 2**3
    assert calls == [2, 4, 8, 16]  # coarsest once, then one solve per doubling


def test_auto_refine_stops_as_soon_as_it_settles():
    from nefes.elements.composite import auto_refine

    # q(N) = 1 + 1/N: the relative step change halves each doubling, crossing 1% partway
    def probe(n):
        return {"q": 1.0 + 1.0 / n}

    ar = auto_refine(lambda n: n, 4, probe, tol=1e-2, max_refine=8)
    assert ar.converged
    # first step below tol is 64->128 (rel ~ 1/128 / 1.0078 ~ 0.0077 < 1e-2)
    assert ar.n_final == 128 and ar.n_refine == 5


def test_auto_refine_validates_inputs():
    from nefes.elements.composite import auto_refine

    with pytest.raises(ValueError, match="n_start must be >= 1"):
        auto_refine(lambda n: n, 0, lambda s: {"q": 1.0})
    with pytest.raises(ValueError, match="max_refine must be >= 1"):
        auto_refine(lambda n: n, 4, lambda s: {"q": 1.0}, max_refine=0)


def test_segments_for_frequency():
    # N >= P * f_max * L / c; e.g. 12 points/wavelength, 1 kHz, 1 m, 340 m/s -> ceil(35.3) = 36
    assert cat.segments_for_frequency(1.0, 340.0, 1000.0, points_per_wavelength=12) == 36
    assert cat.segments_for_frequency(0.1, 340.0, 100.0) >= 1
    with pytest.raises(ValueError):
        cat.segments_for_frequency(0.0, 340.0, 100.0)


# -- sudden contraction (vena-contracta composite, item 6) -----------------------------------------

A_BIG, A_SMALL, CC = 4.0e-3, 1.0e-3, 0.62


def _contraction(pt_in, cc=CC, eps=None):
    net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
    i = net.add(cat.total_pressure_inlet(pt_in, T0))
    sc = net.add(cat.sudden_contraction(cc=cc, name="contr", eps=eps))
    o = net.add(cat.pressure_outlet(P0, T0))
    e_in = net.connect(i, sc, A_BIG)
    e_out = net.connect(sc, o, A_SMALL)
    sol = net.solve()
    assert sol.converged
    return sol, e_in, e_out


def test_sudden_contraction_resolves_the_vena_contracta():
    # the throat is the vena contracta cc*A2, carrying the minimum static pressure of the
    # whole element (below both the upstream and the recovered downstream) -- the resolved
    # minimum a lumped cc-loss cannot report.
    sol, e_in, e_out = _contraction(130000.0)
    est = sol.table()
    cv = sol.composite("contr")
    assert cv.throat_state["area"] == pytest.approx(CC * A_SMALL)
    p_vc = est[ES_P, cv.throat]
    assert p_vc < est[ES_P, e_in] and p_vc < est[ES_P, e_out]  # the minimum static pressure
    assert p_vc == pytest.approx(est[ES_P].min())
    assert est[ES_M, cv.throat] > est[ES_M, e_out]  # fastest at the vena contracta, then re-expands
    assert est[ES_M, cv.throat] < 1.0  # subsonic


def test_sudden_contraction_loss_is_compressible():
    # the resolved loss matches the O(M^2) sudden_area_change(cc) at low Mach but diverges
    # from it as the Mach rises -- the compressible correction that is the point of item 6.
    def sac_loss(pt_in):
        net = Network(CFG, p_ref=P0, T_ref=T0, mdot_ref=1.0)
        i = net.add(cat.total_pressure_inlet(pt_in, T0))
        s = net.add(cat.sudden_area_change(cc=CC))
        o = net.add(cat.pressure_outlet(P0, T0))
        a = net.connect(i, s, A_BIG)
        b = net.connect(s, o, A_SMALL)
        sol = net.solve()
        assert sol.converged
        est = sol.table()
        return (est[ES_PT, a] - est[ES_PT, b]) / est[ES_PT, a], float(est[ES_M, b])

    def sc_loss(pt_in):
        sol, a, b = _contraction(pt_in)
        est = sol.table()
        return (est[ES_PT, a] - est[ES_PT, b]) / est[ES_PT, a], float(est[ES_M, b])

    l_lo_sac, m_lo = sac_loss(102000.0)  # low Mach
    l_lo_sc, _ = sc_loss(102000.0)
    l_hi_sac, m_hi = sac_loss(128000.0)  # higher Mach
    l_hi_sc, _ = sc_loss(128000.0)
    assert m_lo < 0.15 and m_hi > 0.4  # the sweep genuinely spans low -> high subsonic Mach
    # both losses are real and positive
    assert l_lo_sc > 0.0 and l_hi_sc > 0.0
    # near-agreement at low Mach, growing divergence at high Mach (the compressible correction)
    rel_lo = abs(l_lo_sc - l_lo_sac) / l_lo_sc
    rel_hi = abs(l_hi_sc - l_hi_sac) / l_hi_sc
    assert rel_lo < 0.12
    assert rel_hi > 0.25


def test_sudden_contraction_validation():
    with pytest.raises(ValueError, match="cc must be in"):
        cat.sudden_contraction(cc=0.0)
    with pytest.raises(ValueError, match="cc must be in"):
        cat.sudden_contraction(cc=1.5)
