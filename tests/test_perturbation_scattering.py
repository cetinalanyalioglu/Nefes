"""Perturbation network: N x N transfer / scattering matrices (theory.md s12.7 (i)).

The operator is ``A(omega) = J_alg + i*omega*M + P(omega) + S(omega)``; v1
implements the duct phase stamp ``P`` and the force-once / extract-many driver,
with ``M = 0`` and ``S`` a no-op provision.  A subsonic two-terminal network has
three independent incoming waves -- two acoustic plus **one entropy** -- so the
matrices are genuinely ``3 x 3``.  Targets are closed-form duct phases, the
acoustic 2x2 sub-block, and internal consistency (cascade composition, unitarity).
"""

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.elements.ids import ACOUSTIC_DUCT, ACOUSTIC_DEFAULT, ACOUSTIC_FLAME
from fns.thermo.configure import perfect_gas
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_C, ES_U, ES_RHO, ES_AREA
from fns.perturbation import (
    perturbation_response,
    find_terminals,
    build_acoustic_blocks,
    assemble_acoustic,
    verify_acoustic,
    scattering_2port,
)

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
OM = np.linspace(50.0, 1500.0, 9)
FULL = ("acoustic", "entropy")  # drive the entropy wave too -> full 3x3 response


def _single_duct(pt_in, p_out, L, area=0.05):
    net = [cat.total_pressure_inlet(pt_in, 300.0), cat.duct(L), cat.pressure_outlet(p_out, 300.0)]
    prob = cat.build_problem(CFG, net, [(0, 1, area), (1, 2, area)], 10.0, 101325.0, CP * 300.0)
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
    prob = cat.build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


# -- 0. acoustic-only by default; entropy drives the full N = 3 -------------


def test_default_excitation_is_acoustic_2x2():
    # the default drives only the acoustic waves and pins the incoming entropy to
    # zero -- a clean, well-conditioned 2x2 acoustic response.
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, OM)
    assert resp.X.shape[1] == 2  # f@inlet, g@outlet
    assert resp.n == 2 and resp.cidx == (0, 1)
    assert resp.transfer_matrix(0, 1).shape == (OM.size, 2, 2)
    assert resp.scattering_matrix(0, 1).shape == (OM.size, 2, 2)


def test_entropy_excitation_gives_full_3x3():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    resp = perturbation_response(prob, res.x, OM, excite=FULL)
    assert resp.X.shape[1] == 3  # f@inlet, g@outlet, h@inlet
    assert resp.n == 3 and resp.cidx == (0, 1, 2)
    assert resp.transfer_matrix(0, 1).shape == (OM.size, 3, 3)
    assert resp.scattering_matrix(0, 1).shape == (OM.size, 3, 3)


def test_unknown_family_rejected():
    prob, res = _single_duct(110000.0, 101325.0, 1.0)
    with pytest.raises(ValueError, match="unknown wave family"):
        perturbation_response(prob, res.x, OM, excite=("acoustic", "vortical"))
    with pytest.raises(ValueError, match="must include 'acoustic'"):
        perturbation_response(prob, res.x, OM, excite=("entropy",))


def test_duct_entropy_phase_and_decoupling():
    # the entropy wave is convected at u (tau_0 = L/u) and does NOT couple to the
    # acoustics on a uniform duct: the 3x3 char TM is diagonal in (f, g, h).
    L = 1.0
    prob, res = _single_duct(110000.0, 101325.0, L)
    u = states_table(prob, res.x)[ES_U, 0]
    resp = perturbation_response(prob, res.x, OM, excite=FULL)
    T = resp.transfer_matrix(0, 1)
    assert np.allclose(T[:, 2, 2], np.exp(-1j * OM * L / u), atol=1e-7)  # entropy phase
    for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:  # acoustic <-> entropy blocks vanish
        assert np.allclose(T[:, i, j], 0.0, atol=1e-7)


# -- 1. quiescent single duct, transmission phase ---------------------------


def test_quiescent_duct_transmission_phase():
    L = 1.0
    prob, res = _single_duct(101325.0, 101325.0, L)
    c = states_table(prob, res.x)[ES_C, 0]
    resp = perturbation_response(prob, res.x, OM)
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
    resp = perturbation_response(prob, res.x, OM)
    trans = resp.transfer_matrix(0, 1)[:, 0, 0]
    assert np.allclose(np.abs(trans), 1.0, atol=1e-9)
    assert np.allclose(trans, np.exp(-1j * OM * tau_p), atol=1e-9)  # wrong signs give tau_-


# -- 3. re-extraction without re-solving ------------------------------------


def test_reextraction_without_resolve():
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, OM, excite=FULL)
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
    resp = perturbation_response(prob, res.x, OM)
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


def test_three_terminals_rejected():
    net = [
        cat.total_pressure_inlet(110000.0, 300.0),
        cat.splitter(),
        cat.pressure_outlet(101325.0, 300.0),
        cat.pressure_outlet(101325.0, 300.0),
    ]
    edges = [(0, 1, 0.05), (1, 2, 0.03), (1, 3, 0.03)]
    prob = cat.build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    with pytest.raises(ValueError, match="exactly 2 terminals"):
        perturbation_response(prob, res.x, OM)


def test_verifier_rejects_reverse_wired_duct():
    net = [cat.total_pressure_inlet(110000.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    edges = [(1, 2, 0.05), (0, 1, 0.05)]  # outgoing edge first -> port 0 points OUT: banned
    prob = cat.build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
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
    assert blocks.M.nnz == 0  # no finite-volume element in v1


def test_acoustic_id_provisions():
    assert cat.duct(1.0).acoustic_id == ACOUSTIC_DUCT
    assert cat.isentropic_area_change().acoustic_id == ACOUSTIC_DEFAULT
    assert cat.total_pressure_inlet(1e5, 300.0).acoustic_id == ACOUSTIC_DEFAULT


def test_flame_face_is_wired_but_unimplemented():
    net = [cat.total_pressure_inlet(110000.0, 300.0), cat.duct(1.0), cat.pressure_outlet(101325.0, 300.0)]
    net[1].acoustic_id = ACOUSTIC_FLAME  # pretend the duct is a flame
    prob = cat.build_problem(CFG, net, [(0, 1, 0.05), (1, 2, 0.05)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged  # acoustic_id never touches the mean-flow residual
    blocks = build_acoustic_blocks(prob, res.x)
    with pytest.raises(NotImplementedError, match="flame"):
        assemble_acoustic(100.0, blocks)


# -- 9-11. multi-element networks (ducts joined by an area change) -----------


@pytest.mark.parametrize("pt_in", [101325.0, 110000.0])
def test_cascade_composition(pt_in):
    # full 3x3 composition: a transfer-matrix chain multiplies, entropy included.
    prob, res = _cascade(pt_in, 101325.0)
    resp = perturbation_response(prob, res.x, OM, excite=FULL)
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
    resp = perturbation_response(prob, res.x, OM)
    S = resp.acoustic_scattering_matrix(0, 3)
    for i in range(OM.size):  # exact unitarity only at u=0; residual Mach ~1e-6 leaks ~1e-6
        Sn = D @ S[i] @ Dinv  # power-normalized
        assert np.allclose(Sn.conj().T @ Sn, np.eye(2), atol=1e-4)


def test_cascade_embedded_duct_phases():
    prob, res = _cascade(110000.0, 101325.0, L1=0.7, L2=1.1)
    est = states_table(prob, res.x)
    resp = perturbation_response(prob, res.x, OM)
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
    resp = perturbation_response(prob, res.x, OM, excite=FULL)

    figT = resp.plot_transfer_matrix(1, 2)
    titlesT = {a.text for a in figT.layout.annotations}
    assert "f<sub>1</sub>→f<sub>2</sub>" in titlesT  # input edge 1 -> output edge 2
    assert "f→f" not in titlesT  # the ambiguous bare form is gone

    figS = resp.plot_scattering_matrix(1, 2)
    titlesS = {a.text for a in figS.layout.annotations}
    # every scattering label carries a station (edge 1 or 2) subscript
    assert titlesS and all(("<sub>1</sub>" in t or "<sub>2</sub>" in t) for t in titlesS if t)


def test_response_plot_methods_accept_hz_axis():
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, OM)
    fig = resp.plot_transfer_matrix(0, 1, resp.omegas / (2.0 * np.pi))  # x in Hz
    xs = np.asarray(fig.data[0].x)
    assert np.allclose(xs, OM / (2.0 * np.pi))


def test_response_plot_basis_converts_and_relabels_consistently():
    # the response-method basis genuinely re-expresses the matrix AND names it to
    # match -- no label-only mismatch like the (removed) free-function basis knob.
    prob, res = _cascade(110000.0, 101325.0)
    resp = perturbation_response(prob, res.x, OM, excite=FULL)

    figP = resp.plot_transfer_matrix(1, 2, basis="primitive")
    titlesP = {a.text for a in figP.layout.annotations}
    assert "u'<sub>1</sub>→u'<sub>2</sub>" in titlesP

    # the basis really changed the numbers, not just the labels
    char = resp.transfer_matrix(1, 2, basis="char")
    prim = resp.transfer_matrix(1, 2, basis="primitive")
    assert not np.allclose(char, prim)
