"""Perturbation network: N x N transfer / scattering matrices.

The operator is ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega)``; the
implementation covers the duct phase stamp ``P`` and the force-once / extract-many
driver, with ``M = 0`` and ``S`` a no-op provision.  A subsonic two-terminal network has
three independent incoming waves -- two acoustic plus **one entropy** -- so the
matrices are genuinely ``3 x 3``.  Targets are closed-form duct phases, the
acoustic 2x2 sub-block, and internal consistency (cascade composition, unitarity).
"""

import numpy as np
import pytest

from nefes.assembly.recover import ES_AREA, ES_C, ES_RHO, ES_U
from nefes.elements import catalog as cat
from nefes.elements.ids import STAMP_DEFAULT, STAMP_DUCT, STAMP_FLAME
from nefes.graph.connectivity import build_connectivity
from nefes.perturbation import (
    TransferMatrixWarning,
    assemble_acoustic,
    build_acoustic_blocks,
    find_terminals,
    perturbation_response,
    scattering_2port,
    verify_acoustic,
)
from nefes.shell.build import build_problem, build_problem_from_connectivity
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
OM = np.linspace(50.0, 1500.0, 9)  # angular frequencies (rad/s) for the e^{-iwt} phase checks
FR = OM / (2.0 * np.pi)  # the matching Hz sweep fed to the (Hz) perturbation_response API
FULL = ("acoustic", "entropy")  # drive the entropy wave too -> full 3x3 response


def _single_duct(pt_in, p_out, L, area=0.05):
    net = [cat.total_pressure_inlet(pt_in, 300.0), cat.duct(L), cat.pressure_outlet(p_out, 300.0)]
    prob = build_problem(CFG, net, [(0, 1, area), (1, 2, area)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _cascade(pt_in, p_out, L1=0.7, L2=1.1, A1=0.05, A2=0.03):
    net = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.duct(L1),
        cat.isentropic_area_change(),
        cat.duct(L2),
        cat.pressure_outlet(p_out, 300.0),
    ]
    edges = [(0, 1, A1), (1, 2, A1), (2, 3, A2), (3, 4, A2)]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _tree_3term(pt_in=110000.0, p_out=101325.0, a_branch=0.03):
    """Inlet -> duct -> splitter -> two outlets: 3 terminals, a tree (no internal loop).

    Edges: 0 (inlet->duct), 1 (duct->splitter), 2 / 3 (the two splitter branches).  With
    equal outlet pressures the branches are symmetric, so both genuinely flow outward.
    """
    net = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.duct(1.0),
        cat.splitter(),
        cat.pressure_outlet(p_out, 300.0),
        cat.pressure_outlet(p_out, 300.0),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.05), (2, 3, a_branch), (2, 4, a_branch)]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _wall_branch(pt_in=110000.0, p_out=101325.0):
    """Inlet -> duct -> splitter -> (duct -> outlet) and a side duct -> wall (dead leg).

    The wall blocks mean flow, so the side branch (edges 4, 5) is a quiescent dead leg
    (a closed side-branch); all the mean flow takes the outlet branch.  Terminals: the
    inlet (node 0), the outlet (node 4), and the wall (node 6).
    """
    net = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.duct(0.6),
        cat.splitter(),
        cat.duct(0.7),
        cat.pressure_outlet(p_out, 300.0),
        cat.duct(0.4),
        cat.wall(),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.05), (2, 3, 0.03), (3, 4, 0.03), (2, 5, 0.03), (5, 6, 0.03)]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _diamond(pt_in=110000.0, p_out=101325.0):
    """Inlet -> duct -> splitter -> (two ducts) -> junction -> duct -> outlet.

    One inlet, one outlet, but the two parallel branches recombine -> an internal
    acoustic loop; two excitations still fully determine its 2x2 terminal matrix.
    """
    net = [
        cat.total_pressure_inlet(pt_in, 300.0),
        cat.duct(0.5),
        cat.splitter(),
        cat.duct(0.7),
        cat.duct(1.1),
        cat.junction(),
        cat.duct(0.5),
        cat.pressure_outlet(p_out, 300.0),
    ]
    edges = [
        (0, 1, 0.05),
        (1, 2, 0.05),
        (2, 3, 0.025),
        (2, 4, 0.025),
        (3, 5, 0.025),
        (4, 5, 0.025),
        (5, 6, 0.05),
        (6, 7, 0.05),
    ]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


# -- 0. acoustic-only by default; entropy drives the full N = 3 -------------


def test_default_excitation_is_acoustic_2x2():
    # the default drives only the acoustic waves and pins the incoming entropy to
    # zero -- a clean, well-conditioned 2x2 acoustic response.
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, FR)
    assert resp.X.shape[1] == 2  # f@inlet, g@outlet
    assert resp.n == 2 and resp.cidx == (0, 1)
    assert resp.transfer_matrix(0, 1).shape == (OM.size, 2, 2)
    assert resp.scattering_matrix(0, 1).shape == (OM.size, 2, 2)


def test_entropy_excitation_gives_full_3x3():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, FR, excite=FULL)
    assert resp.X.shape[1] == 3  # f@inlet, g@outlet, h@inlet
    assert resp.n == 3 and resp.cidx == (0, 1, 2)
    assert resp.transfer_matrix(0, 1).shape == (OM.size, 3, 3)
    assert resp.scattering_matrix(0, 1).shape == (OM.size, 3, 3)


def test_unknown_family_rejected():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    with pytest.raises(ValueError, match="unknown wave family"):
        perturbation_response(prob, res.x, FR, excite=("acoustic", "vortical"))
    with pytest.raises(ValueError, match="must include 'acoustic'"):
        perturbation_response(prob, res.x, FR, excite=("entropy",))


def test_duct_entropy_phase_and_decoupling():
    # the entropy wave is convected at u (tau_0 = L/u) and does NOT couple to the
    # acoustics on a uniform duct: the 3x3 char TM is diagonal in (f, g, h).
    L = 1.0
    prob, res = _single_duct(110000.0, 101325.0, L)
    u = states_table(prob, res.x)[ES_U, 0]
    resp = perturbation_response(prob, res.x, FR, excite=FULL)
    T = resp.transfer_matrix(0, 1)
    assert np.allclose(T[:, 2, 2], np.exp(-1j * OM * L / u), atol=1e-7)  # entropy phase
    for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:  # acoustic <-> entropy blocks vanish
        assert np.allclose(T[:, i, j], 0.0, atol=1e-7)


# -- 1. quiescent single duct, transmission phase ---------------------------


def test_quiescent_duct_transmission_phase():
    L = 1.0
    prob, res = _single_duct(101325.0, 101325.0, L)
    c = states_table(prob, res.x)[ES_C, 0]
    resp = perturbation_response(prob, res.x, FR)
    trans = resp.transfer_matrix(0, 1)[:, 0, 0]  # f -> f
    assert np.allclose(np.abs(trans), 1.0, atol=1e-6)  # lossless
    assert np.allclose(trans, np.exp(-1j * OM * L / c), atol=1e-4)
    standalone = np.array([scattering_2port(c, L, w)[0, 0] for w in OM])
    assert np.allclose(trans, standalone, atol=1e-4)


# -- 2. duct with mean flow, tau_+ phase (orientation/sign sentinel) --------


def test_meanflow_duct_tau_plus_phase():
    L = 1.0
    prob, res = _single_duct(110000.0, 101325.0, L)
    est = states_table(prob, res.x)
    c, u = est[ES_C, 0], est[ES_U, 0]
    assert u > 1.0  # genuinely flowing
    tau_p = L / (u + c)
    resp = perturbation_response(prob, res.x, FR)
    trans = resp.transfer_matrix(0, 1)[:, 0, 0]
    assert np.allclose(np.abs(trans), 1.0, atol=1e-9)
    assert np.allclose(trans, np.exp(-1j * OM * tau_p), atol=1e-9)  # wrong signs give tau_-


# -- 3. re-extraction without re-solving ------------------------------------


def test_reextraction_without_resolve():
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, FR, excite=FULL)
    T01 = resp.transfer_matrix(0, 1)
    T03 = resp.transfer_matrix(0, 3)
    S13 = resp.scattering_matrix(1, 3)
    assert T01.shape == (OM.size, 3, 3)
    assert T03.shape == (OM.size, 3, 3)
    assert S13.shape == (OM.size, 3, 3)
    est = states_table(prob, res.x)
    c, u = est[ES_C, 0], est[ES_U, 0]
    diag = np.array([np.diag([np.exp(-1j * w * 0.7 / (u + c)), np.exp(1j * w * 0.7 / (c - u))]) for w in OM])
    assert np.allclose(resp.acoustic_transfer_matrix(0, 1), diag, atol=1e-9)


# -- 4. lossless unitarity (acoustic 2x2) -----------------------------------


def test_quiescent_scattering_unitary():
    prob, res = _single_duct(101325.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, FR)
    S = resp.acoustic_scattering_matrix(0, 1)
    for i in range(OM.size):
        assert np.allclose(S[i].conj().T @ S[i], np.eye(2), atol=1e-6)
        assert abs(abs(np.linalg.det(S[i])) - 1.0) < 1e-6


# -- 5. duct length is inert metadata ---------------------------------------


def test_duct_length_inert_in_mean_flow():
    p_a, _ = _single_duct(110000.0, 101325.0, 0.5)
    p_b, _ = _single_duct(110000.0, 101325.0, 2.0)
    ra = solve(p_a)
    rb = solve(p_b)
    assert np.array_equal(states_table(p_a, ra.x), states_table(p_b, rb.x))


# -- 6. terminal detection + verifier ---------------------------------------


def test_terminal_detection():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    terms = find_terminals(prob, res.x)
    assert len(terms) == 2
    by_node = {t.node: t for t in terms}
    assert by_node[0].at_tail and by_node[0].incoming == 0  # inlet injects f
    assert by_node[0].inflowing  # mean flow enters -> carries an incoming entropy wave
    assert (not by_node[2].at_tail) and by_node[2].incoming == 1  # outlet injects g
    assert not by_node[2].inflowing


def test_three_terminals_multiport():
    """A 3-terminal tree: every terminal is neutralized, so the multiport SM is well posed.

    Acoustic-only it is the square ``3 x 3`` (one incoming/outgoing wave per terminal);
    with entropy it is the rectangular ``5 x 4`` (incoming ``f@in, g@out, g@out, h@in``;
    outgoing ``g@in, f@out, h@out, f@out, h@out``).  A tree with all terminals anechoic has
    no resonant poles, so the magnitude is bounded and its peak does not grow under grid
    refinement -- the spurious comb from a stray reflecting terminal is gone.
    """
    prob, res = _tree_3term()
    om = np.linspace(20.0, 1500.0, 400)
    r = perturbation_response(prob, res.x, om)  # acoustic-only; default forcing drives all 3
    S = r.multiport_scattering_matrix()
    inc, out = r.multiport_scattering_labels()
    assert S.shape == (om.size, 3, 3)
    assert len(inc) == 3 and len(out) == 3

    S_fine = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 1600))
    S_fine = S_fine.multiport_scattering_matrix()
    assert np.abs(S).max() < 1.5  # lossless tree, anechoic everywhere -> no blow-up
    assert abs(np.abs(S).max() - np.abs(S_fine).max()) < 0.05 * np.abs(S_fine).max()  # no hidden sharp peak

    rf = perturbation_response(prob, res.x, om[:5], excite=FULL)  # rectangular with entropy
    assert rf.multiport_scattering_matrix().shape == (5, 5, 4)


def test_branched_entropy_excitation_convects_in_series():
    """Entropy excitation on a branched net convects through the in-series inlet duct.

    On the tree's inlet duct (edges 0->1) the 3x3 transfer matrix must show the entropy wave
    convecting with phase ``exp(-i w L / u)`` and staying decoupled from the acoustics (a pure
    duct), exactly as on a standalone duct -- the branch downstream does not change that.
    """
    prob, res = _tree_3term()
    L = 1.0  # the inlet duct length in _tree_3term
    u = float(states_table(prob, res.x)[ES_U, 0])
    T = perturbation_response(prob, res.x, FR, excite=FULL).transfer_matrix(0, 1)  # in-series, 3x3
    assert T.shape == (OM.size, 3, 3)
    assert np.allclose(T[:, 2, 2], np.exp(-1j * OM * L / u), atol=1e-7)  # entropy convection phase
    for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:  # acoustic <-> entropy decoupled on a pure duct
        assert np.allclose(T[:, i, j], 0.0, atol=1e-7)


def test_transfer_matrix_in_series_well_defined():
    """On a 3-terminal net the in-series duct transfer matrix is unique (pinv over 3 forcings)."""
    prob, res = _tree_3term()
    r = perturbation_response(prob, res.x, FR)  # 3 forcings -> Wa is (2, 3), pinv path
    T = r.transfer_matrix(0, 1)  # edges 0, 1 are the two ends of the inlet duct (in series)
    assert T.shape == (OM.size, 2, 2)
    assert np.allclose(T, r.acoustic_transfer_matrix(0, 1), atol=1e-9)  # consistent reconstruction


def test_transfer_matrix_across_branch_warns_and_returns_best_fit():
    """Edges across the splitter have no transfer matrix -> warn (don't raise), return best fit.

    With all three terminals driven the seriality residual is a valid test, so the branch
    straddle is flagged by a large residual and a clear warning -- but the user still gets the
    least-squares matrix back to inspect / plot.
    """
    prob, res = _tree_3term()
    r = perturbation_response(prob, res.x, FR)  # all 3 terminals driven -> residual test valid
    with pytest.warns(TransferMatrixWarning, match="straddle an internal branch"):
        T = r.transfer_matrix(2, 3)  # the two splitter branches
    assert T.shape == (OM.size, 2, 2)
    assert r.transfer_residual(2, 3) > 1e-3  # the large residual is what flags the non-seriality
    assert r.transfer_residual(0, 1) < 1e-6  # an in-series pair stays clean (no false positive)


def test_transfer_matrix_underdetermined_warns_unverifiable():
    """Forcing a 2-terminal subset of a 3-terminal net cannot test seriality -> unverified warning.

    Here ``n_force == n``, so the fit is exact by construction and the residual is ~0 even across
    the branch: the warning must call out that seriality is *unverifiable*, not that it is clean.
    """
    prob, res = _tree_3term()
    nodes = sorted(t.node for t in find_terminals(prob, res.x))
    r = perturbation_response(prob, res.x, FR, forcing=(nodes[0], nodes[1]))  # drive only 2 of 3
    with pytest.warns(TransferMatrixWarning, match="under-determined"):
        T = r.transfer_matrix(2, 3)  # across the branch, but residual is structurally ~0
    assert T.shape == (OM.size, 2, 2)
    assert r.transfer_residual(2, 3) < 1e-6  # blind: looks clean despite straddling the branch


def test_source_attribution_breaks_down_wave_at_edge():
    """contributions(edge) is the exact per-terminal-source decomposition of the wave there."""
    prob, res = _tree_3term()
    r = perturbation_response(prob, res.x, FR)  # 3 terminals -> 3 sources, acoustic n=2
    C = r.contributions(3)  # edge 3 lives in branch B
    assert C.shape == (OM.size, 2, 3)
    outputs, sources = r.contribution_labels(3)
    assert len(outputs) == 2 and len(sources) == 3
    assert all("_{" in s for s in outputs + sources)  # LaTeX subscript fragments
    # a unit selector isolates exactly one source and zeroes the rest (pure superposition)
    sel = r.contributions(3, incoming=[1.0, 0.0, 0.0])
    assert np.allclose(sel[:, :, 0], C[:, :, 0]) and np.allclose(sel[:, :, 1:], 0.0)
    with pytest.raises(ValueError, match="one amplitude per source"):
        r.contributions(3, incoming=[1.0, 0.0])
    with pytest.raises(ValueError, match="out of range"):
        r.contributions(99)


def test_source_attribution_consistent_with_in_series_transfer():
    """For in-series edges each source's contribution at b is T_ba times its contribution at a."""
    prob, res = _cascade(110000.0, 101325.0)
    r = perturbation_response(prob, res.x, FR)
    T = r.transfer_matrix(0, 1)  # in series -> exact (residual ~ 0)
    Ca, Cb = r.contributions(0), r.contributions(1)
    assert np.allclose(np.einsum("oij,ojk->oik", T, Ca), Cb, atol=1e-8)


def test_plot_contributions_overlays_sources_per_output_panel():
    prob, res = _tree_3term()
    r = perturbation_response(prob, res.x, FR)
    _, sources = r.contribution_labels(3)

    fig = r.plot_contributions(3)  # normalize defaults on (no incoming)
    titles = {a.text for a in fig.layout.annotations if a.text}
    assert {"$f_{3}$", "$g_{3}$"} <= titles  # one panel per output wave at edge 3
    assert {f"${s}$" for s in sources} <= {d.name for d in fig.data}  # one overlaid curve per source (legend)
    assert tuple(fig.layout.yaxis.range) == (0.0, 1.05)  # normalized magnitude axis

    fig_abs = r.plot_contributions(3, incoming=[1.0, 1.0, 1.0])  # absolute -> auto-scaled, not (0, 1.05)
    assert tuple(fig_abs.layout.yaxis.range) != (0.0, 1.05)


def test_branched_single_in_out_two_excitations():
    """A 1-inlet/1-outlet net with a recombining internal loop is fully set by 2 excitations."""
    prob, res = _diamond()
    assert len(find_terminals(prob, res.x)) == 2
    om = np.linspace(20.0, 1500.0, 400)
    r = perturbation_response(prob, res.x, om)
    S = r.multiport_scattering_matrix()
    assert S.shape == (om.size, 2, 2)
    assert np.abs(S).max() < 1.5  # lossless + anechoic: bounded despite the internal loop
    assert r.transfer_matrix(0, 1).shape == (om.size, 2, 2)  # in-series duct still well defined


def test_wall_terminated_branch_multiport():
    """A branch closed by a wall is a quiescent dead leg, yet the multiport SM stays well posed.

    Acoustically the wall is just another anechoic port (in the measurement convention), so the
    3 x 3 acoustic multiport is bounded and refinement-stable.  The dead-leg entropy is decoupled
    (no convection at ``u = 0``) and does **not** contaminate the acoustic block.  With entropy the
    quiescent wall carries no convected-wave port -- it contributes only its acoustic reflection,
    never an ``h`` wave -- so the rectangular SM has no entropy row/column at the wall terminal.
    """
    prob, res = _wall_branch()
    est = states_table(prob, res.x)
    assert abs(float(est[ES_U, 5])) < 1e-9  # the side branch behind the wall is a quiescent dead leg

    om = np.linspace(20.0, 1500.0, 400)
    r = perturbation_response(prob, res.x, om)  # acoustic-only; all 3 terminals driven
    Sa = r.multiport_scattering_matrix()
    inc, out = r.multiport_scattering_labels()
    assert Sa.shape == (om.size, 3, 3) and len(inc) == 3 and len(out) == 3
    assert np.abs(Sa).max() < 1.5  # lossless, anechoic everywhere -> bounded (no quarter-wave pole)

    Sf_fine = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 3200))
    Sf_fine = Sf_fine.multiport_scattering_matrix()
    assert abs(np.abs(Sa).max() - np.abs(Sf_fine).max()) < 0.05 * np.abs(Sf_fine).max()  # no hidden peak

    # driving the entropy wave too must leave the acoustic block bit-for-bit unchanged: the
    # quiescent dead-leg entropy is decoupled and never folds back into the acoustics.
    rf = perturbation_response(prob, res.x, om, excite=FULL)
    Sfull = rf.multiport_scattering_matrix()
    finc, fout = rf.multiport_scattering_labels()
    acoustic_sub = Sfull[:, [[0], [1], [3]], [0, 1, 2]]  # rows g@n0,f@n4,f@n6 x cols f@n0,g@n4,g@n6
    assert np.allclose(Sa, acoustic_sub, atol=1e-10)

    # the quiescent wall (node 6) carries no entropy port: no h wave at the wall in either set.
    assert "h_{6}" not in finc and "h_{6}" not in fout
    assert finc == [  # entropy enters only at the flowing inlet
        "f_{0}",
        "g_{4}",
        "g_{6}",
        "h_{0}",
    ]
    assert fout == [  # entropy leaves only at the flowing outlet
        "g_{0}",
        "f_{4}",
        "h_{4}",
        "f_{6}",
    ]
    assert Sfull.shape == (om.size, 4, 4) and np.isfinite(Sfull).all()


def test_entropy_quiescent_rejected():
    """Entropy excitation is undefined at a quiescent terminal (no convection); acoustic is fine."""
    net = [cat.mass_flow_inlet(0.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    prob = build_problem(CFG, net, [(0, 1, 0.05), (1, 2, 0.05)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    with pytest.raises(ValueError, match="quiescent"):
        perturbation_response(prob, res.x, FR, excite=FULL)
    assert perturbation_response(prob, res.x, FR).transfer_matrix(0, 1).shape == (OM.size, 2, 2)


def test_mass_flow_inlet_rejects_reverse_flow():
    """A mass-flow inlet is inflow-only: a negative (reversing/suction) mass rate is rejected.

    Reverse flow at a boundary is modelled by a ``pressure_outlet`` that ingests (backflow);
    the genuine inlet/outlet read from the mean flow under reversal is still exercised by
    :func:`test_branched_reversal_well_posed`.  A quiescent (``mdot = 0``) inlet stays valid.
    """
    with pytest.raises(ValueError, match="inflow boundary"):
        cat.mass_flow_inlet(-0.05, 300.0)
    # zero (quiescent) and positive (injecting) remain valid constructions
    cat.mass_flow_inlet(0.0, 300.0)
    cat.mass_flow_inlet(0.05, 300.0)


def test_branched_reversal_well_posed():
    """A 3-terminal net with one reversed outlet stays well posed (no floating-entropy blow-up)."""
    net = [
        cat.mass_flow_inlet(0.05, 300.0),
        cat.duct(0.6),
        cat.splitter(),
        cat.duct(0.7),
        cat.pressure_outlet(101325.0, 300.0),
        cat.duct(0.9),
        cat.pressure_outlet(101300.0, 300.0),  # lower pressure -> backflow in the other branch
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.05), (2, 3, 0.03), (3, 4, 0.03), (2, 5, 0.03), (5, 6, 0.03)]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    assert float(states_table(prob, res.x)[ES_U, 2]) < 0.0  # the (101325) branch reverses
    Sc = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 400)).multiport_scattering_matrix()
    Sf = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 3200)).multiport_scattering_matrix()
    # bounded and refinement-stable -> no spurious resonance from a floating incoming entropy
    assert np.abs(Sc).max() < 1.5
    assert abs(np.abs(Sc).max() - np.abs(Sf).max()) < 0.05 * np.abs(Sf).max()

    # entropy excitation is also well posed here: entropy enters at *both* genuine inlets
    # (the forward inlet and the reversed branch) and leaves at the single genuine outlet.
    rf = perturbation_response(prob, res.x, np.linspace(20.0, 1500.0, 400), excite=FULL)
    Srect = rf.multiport_scattering_matrix()
    inc, out = rf.multiport_scattering_labels()
    assert Srect.shape == (400, 4, 5) and np.abs(Srect).max() < 1.5
    assert sum(s.startswith("h_{") for s in inc) == 2  # two genuine inlets carry incoming entropy
    assert sum(s.startswith("h_{") for s in out) == 1  # one genuine outlet carries outgoing entropy


def test_reverse_listed_duct_is_auto_flow_aligned():
    # Kind-aware port assignment claims each edge a direction-matching port, so port 0 of a
    # duct is its inflow regardless of the order edges were listed: a reverse-listed duct is
    # auto-corrected and passes the acoustic flow-alignment verifier.
    net = [cat.total_pressure_inlet(110000.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    edges = [(1, 2, 0.05), (0, 1, 0.05)]  # outgoing edge listed first
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    verify_acoustic(prob, res.x)  # no raise: port 0 points into the duct


def test_verifier_rejects_reverse_wired_duct():
    # Explicitly pinned ports bypass kind-aware assignment: pin the duct's port 0 to its
    # outgoing edge so it points OUT, and the acoustic verifier still rejects it.
    net = [cat.total_pressure_inlet(110000.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    endpoints = [(0, 0, 1, 1), (1, 0, 2, 0)]  # duct port 0 = outgoing edge (orient +1: points OUT)
    conn = build_connectivity(3, endpoints)
    area = np.array([0.05, 0.05])
    prob = build_problem_from_connectivity(CFG, net, conn, area, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    with pytest.raises(ValueError, match="flow-aligned"):
        verify_acoustic(prob, res.x)


def test_verifier_rejects_supersonic_duct():
    prob, res = _single_duct(500000.0, 50000.0, 1.0, area=0.05)  # huge pressure ratio -> sonic throat
    est = states_table(prob, res.x)
    if np.max(np.abs(est[ES_U] / est[ES_C])) < 1.0:
        pytest.skip("mean flow stayed subsonic; cannot exercise supersonic guard here")
    with pytest.raises(ValueError, match="supersonic|>= 1"):
        verify_acoustic(prob, res.x)


# -- 7. A(0) consistency (duct network) -------------------------------------


def test_zero_frequency_duct_is_continuity():
    prob, res = _single_duct(101325.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, np.array([0.0]), excite=FULL)
    T = resp.transfer_matrix(0, 1)[0]
    assert np.allclose(T, np.eye(3), atol=1e-6)  # DC limit: lossless pass-through, all 3 waves


# -- 8. four-term provision shape -------------------------------------------


def test_storage_block_zero_and_shape():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    blocks = build_acoustic_blocks(prob, res.x)
    assert blocks.M.shape == (prob.n_eq, prob.n_col)
    assert blocks.M.nnz == 0  # no finite-volume element in this network


def test_acoustic_stamp_provisions():
    assert cat.duct(1.0).acoustic_stamp == STAMP_DUCT
    assert cat.isentropic_area_change().acoustic_stamp == STAMP_DEFAULT
    assert cat.total_pressure_inlet(1e5, 300.0).acoustic_stamp == STAMP_DEFAULT


def test_source_block_keys_off_the_descriptor_not_the_acoustic_stamp():
    """The S(omega) block is driven by an attached DynamicSource, not by ``acoustic_stamp``.

    A network with no dynamic-source descriptor has an inert source block (no stamps,
    assembly succeeds); merely tagging an element ``STAMP_FLAME`` does nothing.  The
    active-source behaviour is exercised in ``test_dynamic_source`` /
    ``test_rijke_stability``.
    """
    net = [cat.total_pressure_inlet(110000.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    net[1].acoustic_stamp = STAMP_FLAME  # a bare tag, with no DynamicSource attached
    prob = build_problem(CFG, net, [(0, 1, 0.05), (1, 2, 0.05)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged  # acoustic_stamp never touches the mean-flow residual
    blocks = build_acoustic_blocks(prob, res.x)
    assert not blocks.has_sources  # no descriptor -> inert source block
    assemble_acoustic(100.0, blocks)  # assembles without error


# -- 9-11. multi-element networks (ducts joined by an area change) -----------


@pytest.mark.parametrize("pt_in", [101325.0, 110000.0])
def test_cascade_composition(pt_in):
    # full 3x3 composition: a transfer-matrix chain multiplies, entropy included.
    prob, res = _cascade(pt_in, 101325.0)
    resp = perturbation_response(prob, res.x, FR, excite=FULL)
    T01 = resp.transfer_matrix(0, 1)
    T12 = resp.transfer_matrix(1, 2)
    T23 = resp.transfer_matrix(2, 3)
    T03 = resp.transfer_matrix(0, 3)
    assert np.allclose(T03, T23 @ T12 @ T01, atol=1e-8)


def test_cascade_quiescent_unitary():
    # Across an area change the raw (f, g) amplitudes are not power-conserving:
    # acoustic power ~ rho*c*A*(|f|^2 - |g|^2).  The *power-normalized* acoustic
    # scattering matrix (waves scaled by sqrt(rho*c*A)) is unitary for a lossless
    # quiescent cascade.
    prob, res = _cascade(101325.0, 101325.0)
    est = states_table(prob, res.x)

    def scale(e):
        return np.sqrt(est[ES_RHO, e] * est[ES_C, e] * est[ES_AREA, e])

    sa, sb = scale(0), scale(3)
    D = np.diag([sa, sb])
    Dinv = np.diag([1.0 / sa, 1.0 / sb])
    resp = perturbation_response(prob, res.x, FR)
    S = resp.acoustic_scattering_matrix(0, 3)
    for i in range(OM.size):  # exact unitarity only at u=0; residual Mach ~1e-6 leaks ~1e-6
        Sn = D @ S[i] @ Dinv  # power-normalized
        assert np.allclose(Sn.conj().T @ Sn, np.eye(2), atol=1e-4)


def test_cascade_embedded_duct_phases():
    prob, res = _cascade(110000.0, 101325.0, L1=0.7, L2=1.1)
    est = states_table(prob, res.x)
    resp = perturbation_response(prob, res.x, FR)
    for (a, b), e, Ld in (((0, 1), 0, 0.7), ((2, 3), 2, 1.1)):
        c, u = est[ES_C, e], est[ES_U, e]
        T = resp.acoustic_transfer_matrix(a, b)
        diag = np.array([np.diag([np.exp(-1j * w * Ld / (u + c)), np.exp(1j * w * Ld / (c - u))]) for w in OM])
        assert np.allclose(T, diag, atol=1e-8)


# -- edge-aware plotting convenience ----------------------------------------


def test_response_plot_methods_label_entries_by_edge():
    # the f -> f bug: the free plotter cannot see the edges, so the response
    # methods must inject them and produce f_a -> f_b titles.
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, FR, excite=FULL)

    figT = resp.plot_transfer_matrix(1, 2)
    titlesT = {a.text for a in figT.layout.annotations}
    assert r"${f}_{1} \to {f}_{2}$" in titlesT  # input edge 1 -> output edge 2
    assert r"$f \to f$" not in titlesT  # the ambiguous bare form is gone

    figS = resp.plot_scattering_matrix(1, 2)
    titlesS = {a.text for a in figS.layout.annotations}
    # every scattering label carries a station (edge 1 or 2) subscript
    assert titlesS and all(("_{1}" in t or "_{2}" in t) for t in titlesS if t)


def test_response_plot_methods_default_to_hz_axis():
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, FR)
    assert np.allclose(resp.freqs, FR)  # the response stores its sweep in Hz
    fig = resp.plot_transfer_matrix(0, 1)  # default x-axis is resp.freqs (Hz)
    xs = np.asarray(fig.data[0].x)
    assert np.allclose(xs, FR)
    # an explicit frequency axis is still honored
    fig2 = resp.plot_transfer_matrix(0, 1, FR / 1000.0)  # e.g. kHz
    assert np.allclose(np.asarray(fig2.data[0].x), FR / 1000.0)


def test_response_plot_basis_converts_and_relabels_consistently():
    # the response-method basis genuinely re-expresses the matrix AND names it to
    # match -- no label-only mismatch like the (removed) free-function basis knob.
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, FR, excite=FULL)

    figP = resp.plot_transfer_matrix(1, 2, basis="primitive")
    titlesP = {a.text for a in figP.layout.annotations}
    assert r"${u'}_{1} \to {u'}_{2}$" in titlesP

    # the basis really changed the numbers, not just the labels
    char = resp.transfer_matrix(1, 2, basis="char")
    prim = resp.transfer_matrix(1, 2, basis="primitive")
    assert not np.allclose(char, prim)


# -- freeze: keep a terminal's physical BC during measurement -----------------


def _freeze_io_index(io, node):
    """Position of ``node`` in a multiport (node, edge, char) incoming/outgoing list."""
    return next(i for i, (nd, _e, _c) in enumerate(io) if nd == node)


def test_freeze_wall_equivalent_to_multiport_condensation():
    """Freezing the wall reads out the closed-stub two-port directly -- the same answer the
    rigorous multiport-then-condense route gives by re-closing the (anechoic) wall port with
    ``R = +1``.  This is the whole premise: an interior wall branch reduces to a true 2-port."""
    prob, res = _wall_branch()
    f_eq = np.array([40.0, 80.0, 120.0, 160.0])  # off the stub resonance (~215 Hz): both paths well posed
    e_in, e_out, wall_node = 0, 3, 6  # inlet edge, outlet edge, the wall terminal

    # frozen: the wall stays a hard wall, so inlet->outlet is a genuine 2-port read directly
    resp_fr = perturbation_response(prob, res.x, f_eq, freeze=[wall_node])
    tau_frozen = resp_fr.acoustic_scattering_matrix(e_in, e_out)[:, 1, 0]  # transmission f_out / f_in

    # condensed: measure the open 3-port (wall neutralized to anechoic), re-close it with R=+1
    resp_open = perturbation_response(prob, res.x, f_eq)
    S = resp_open.multiport_scattering_matrix()
    inc, out = resp_open._multiport_io()
    i = _freeze_io_index(inc, 0)  # inlet incoming column
    o = _freeze_io_index(out, 4)  # outlet outgoing row
    wi = _freeze_io_index(inc, wall_node)  # wall incoming column
    wo = _freeze_io_index(out, wall_node)  # wall outgoing row
    a_w = S[:, wo, i] / (1.0 - S[:, wo, wi])  # close the wall port: a_w = R*(S_wi + S_ww a_w), R=+1
    tau_cond = S[:, o, i] + S[:, o, wi] * a_w

    assert np.allclose(tau_frozen, tau_cond, rtol=1e-6, atol=1e-9)


def test_freeze_makes_branched_net_a_clean_two_port():
    """With the wall open there are 3 terminals, so edges across the splitter straddle a branch
    (warns); freezing the wall leaves 2 ports, so the inlet->outlet transfer matrix is exact."""
    prob, res = _wall_branch()

    resp_open = perturbation_response(prob, res.x, FR)  # 3 terminals -> straddle
    with pytest.warns(TransferMatrixWarning):
        resp_open.transfer_matrix(0, 3)

    resp_fr = perturbation_response(prob, res.x, FR, freeze=[6])  # wall frozen -> 2 ports
    assert len(resp_fr.terminals) == 2 and {t.node for t in resp_fr.terminals} == {0, 4}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", TransferMatrixWarning)  # must NOT warn now
        resp_fr.transfer_matrix(0, 3)
    assert resp_fr.transfer_residual(0, 3) < 1e-8  # genuinely in series


def test_freeze_changes_transmission_vs_open_dead_leg():
    """A closed stub (frozen wall) is acoustically different from an anechoic dead leg
    (the default neutralized wall): the transmission must change."""
    import warnings

    prob, res = _wall_branch()
    f_eq = np.array([120.0, 160.0])
    with warnings.catch_warnings():  # the open 3-terminal read straddles the branch (expected)
        warnings.simplefilter("ignore", TransferMatrixWarning)
        tau_open = perturbation_response(prob, res.x, f_eq).acoustic_scattering_matrix(0, 3)[:, 1, 0]
    tau_frozen = perturbation_response(prob, res.x, f_eq, freeze=[6]).acoustic_scattering_matrix(0, 3)[:, 1, 0]
    assert not np.allclose(tau_open, tau_frozen, rtol=1e-3)


def test_freeze_accepts_node_name_and_validates():
    """``freeze`` takes node ids or element names; bad references raise with a clear message."""
    prob, res = _wall_branch()

    by_id = perturbation_response(prob, res.x, FR, freeze=[6])
    by_name = perturbation_response(prob, res.x, FR, freeze=["wall-1"])  # the wall's element name
    assert by_id.frozen == by_name.frozen == (6,)

    with pytest.raises(ValueError, match="does not exist"):
        perturbation_response(prob, res.x, FR, freeze=[99])
    with pytest.raises(ValueError, match="no element carries"):
        perturbation_response(prob, res.x, FR, freeze=["nope"])
    with pytest.raises(ValueError, match="not a 1-port terminal"):
        perturbation_response(prob, res.x, FR, freeze=[1])  # node 1 is a duct
    with pytest.raises(TypeError, match="int node ids or str node names"):
        perturbation_response(prob, res.x, FR, freeze=[1.5])


def test_repr_omits_matrix_dims_and_reports_frozen():
    prob, res = _wall_branch()

    r_open = perturbation_response(prob, res.x, FR)
    s_open = repr(r_open)
    assert "matrices" not in s_open and "x2" not in s_open and "2x" not in s_open
    assert "frozen" not in s_open and "forcing(s)" in s_open and "terminal(s)" in s_open

    r_fr = perturbation_response(prob, res.x, FR, freeze=["wall-1"])
    s_fr = repr(r_fr)
    assert "BC frozen at 6:wall-1" in s_fr
    assert "matrices" not in s_fr
