"""Identification: recover an element's dynamic response from a measured network transfer
matrix (theory.md s12.7).

Plant-recover throughout -- synthesize the measured matrix from a known response with the
forward machinery, then de-embed and compare:

* a blackbox 2-port's full transfer matrix (cascade, and a branched 3-terminal network);
* a single-input flame transfer function (n-tau);
* a multi-input flame (velocity *and* pressure sensitivity separated from one measurement);
* graceful degradation under measurement noise, with the conditioning diagnostic reported.
"""

import numpy as np

from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.elements.dynamic_source import n_tau, n_tau_flame, DynamicSource, DynamicResponseTerm
from nefes.thermo.configure import perfect_gas
from nefes.solver import solve
from nefes.perturbation import perturbation_response, TransferMatrix
from nefes.perturbation.identify import (
    identify_transfer_matrix,
    identify_transfer_function,
    UnknownTransferMatrix,
    unknown_dynamic_source,
)

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
FREQS = np.linspace(80.0, 1200.0, 21)
FULL = ("acoustic", "entropy")


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---------------------------------------------------------------------------
# blackbox transfer-matrix identification
# ---------------------------------------------------------------------------


def _cascade(mid, A1=0.05, A2=0.03):
    net = [
        cat.total_pressure_inlet(120000.0, 300.0),
        cat.duct(0.7),
        mid,
        cat.duct(1.1),
        cat.pressure_outlet(101325.0, 300.0),
    ]
    edges = [(0, 1, A1), (1, 2, A1), (2, 3, A2), (3, 4, A2)]
    prob = build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_identify_transfer_matrix_cascade():
    pa, xa = _cascade(cat.isentropic_area_change())
    ref = perturbation_response(pa, xa, FREQS, excite=FULL)
    T_true = ref.transfer_matrix(1, 2, basis="char")  # ground truth
    M_meas = ref.transfer_matrix(0, 3, basis="char")  # network transfer matrix

    pu, xu = _cascade(cat.transfer_matrix_element(tm=UnknownTransferMatrix(n=3)))
    out = identify_transfer_matrix(pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, continue_=False)
    assert _rel(out.transfer_matrix.data, T_true) < 1e-6
    assert out.conditioning.max() < 1e3  # well-conditioned cascade


def _branched(mid, A=0.02):
    net = [
        cat.mass_flow_inlet(0.03, 300.0),
        cat.duct(0.6),
        mid,
        cat.splitter(),
        cat.duct(0.7),
        cat.pressure_outlet(1.0e5),
        cat.duct(1.0),
        cat.pressure_outlet(1.0e5),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A), (4, 5, A), (3, 6, A), (6, 7, A)]
    prob = build_problem(CFG, net, edges, mdot_ref=0.03, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_identify_transfer_matrix_branched_3_terminal():
    # a splitter downstream of the element -> a 3-terminal network; a=inlet edge, b=element
    # outlet edge are serial (the branch is folded into the operator A0).
    pb, xb = _branched(cat.isentropic_area_change())
    ref = perturbation_response(pb, xb, FREQS, excite=FULL)
    T_true = ref.transfer_matrix(1, 2, basis="char")
    M_meas = ref.transfer_matrix(0, 2, basis="char")

    pu, xu = _branched(cat.transfer_matrix_element(tm=UnknownTransferMatrix(n=3)))
    out = identify_transfer_matrix(pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=2, continue_=False)
    assert _rel(out.transfer_matrix.data, T_true) < 1e-5


def test_identify_transfer_matrix_continuation_is_analytic():
    pa, xa = _cascade(cat.isentropic_area_change())
    ref = perturbation_response(pa, xa, FREQS, excite=FULL)
    M_meas = ref.transfer_matrix(0, 3, basis="char")
    pu, xu = _cascade(cat.transfer_matrix_element(tm=UnknownTransferMatrix(n=3)))
    out = identify_transfer_matrix(pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, rtol=1e-10)
    assert out.transfer_matrix.analytic  # continued -> usable at complex frequency
    assert out.transfer_matrix(np.array([500.0 - 20.0j])).shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# flame / mass-source transfer-function identification
# ---------------------------------------------------------------------------


def _flame(ds, mdot=0.02, Qdot=8.0e3, A=0.01):
    net = [
        cat.mass_flow_inlet(mdot, 300.0),
        cat.duct(0.5),
        cat.heat_release_flame(Qdot, dynamic_source=ds),
        cat.duct(0.9),
        cat.pressure_outlet(1.0e5),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A)]
    prob = build_problem(CFG, net, edges, mdot_ref=mdot, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def test_identify_single_input_ftf():
    Fu = n_tau(0.8, 2.5e-3)
    pf, xf = _flame(n_tau_flame(0.8, 2.5e-3, ref_edge=1, quantity="u"))
    M_meas = perturbation_response(pf, xf, FREQS, excite=FULL).transfer_matrix(0, 3, basis="char")

    pu, xu = _flame(unknown_dynamic_source([(1, "u")]))
    out = identify_transfer_function(pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, continue_=False)
    assert out.terms == [(1, "u", 1.0)]
    assert _rel(out.values[0], Fu(FREQS)) < 1e-6
    assert out.residual.max() < 1e-8  # the single-input model is consistent with the data


def test_identify_multi_input_ftf():
    # a flame responding to BOTH velocity and pressure at edge 1 -- separated from one measured
    # matrix because the excitations decorrelate u' and p' (finite conditioning, not singular).
    Fu, Fp = n_tau(0.8, 2.5e-3), n_tau(0.35, 1.1e-3)
    plant = DynamicSource(terms=[DynamicResponseTerm(Fu, 1, "u"), DynamicResponseTerm(Fp, 1, "p")])
    pf, xf = _flame(plant)
    M_meas = perturbation_response(pf, xf, FREQS, excite=FULL).transfer_matrix(0, 3, basis="char")

    pu, xu = _flame(unknown_dynamic_source([(1, "u"), (1, "p")]))
    out = identify_transfer_function(pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, continue_=False)
    assert _rel(out.values[0], Fu(FREQS)) < 1e-4
    assert _rel(out.values[1], Fp(FREQS)) < 1e-4
    assert np.isfinite(out.conditioning).all()


def test_identify_acoustic_only_isentropic():
    # the acoustics-only (isentropic) identification pins entropy out and recovers the clean
    # acoustic 2-port -- consistent with an isentropic measurement (the classic flame TM).
    pf, xf = _flame(n_tau_flame(0.8, 2.5e-3, ref_edge=1, quantity="u"))
    ref = perturbation_response(pf, xf, FREQS, excite=("acoustic",), isentropic=True)
    T_true = ref.transfer_matrix(1, 2, basis="char")  # 2x2 acoustic flame TM
    M_meas = ref.transfer_matrix(0, 3, basis="char")
    out = identify_transfer_matrix(
        pf, xf, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, isentropic=True, continue_=False
    )
    assert out.transfer_matrix.n == 2
    assert _rel(out.transfer_matrix.data, T_true) < 1e-6


def test_identify_ftf_acoustic_only_isentropic():
    Fu = n_tau(0.8, 2.5e-3)
    pf, xf = _flame(n_tau_flame(0.8, 2.5e-3, ref_edge=1, quantity="u"))
    M_meas = perturbation_response(pf, xf, FREQS, excite=("acoustic",), isentropic=True).transfer_matrix(
        0, 3, basis="char"
    )
    pu, xu = _flame(unknown_dynamic_source([(1, "u")]))
    out = identify_transfer_function(
        pu, xu, TransferMatrix(FREQS, M_meas), node=2, a=0, b=3, isentropic=True, continue_=False
    )
    assert _rel(out.values[0], Fu(FREQS)) < 1e-6


def test_identify_noise_degrades_gracefully():
    Fu = n_tau(0.8, 2.5e-3)
    pf, xf = _flame(n_tau_flame(0.8, 2.5e-3, ref_edge=1, quantity="u"))
    M_meas = perturbation_response(pf, xf, FREQS, excite=FULL).transfer_matrix(0, 3, basis="char")
    rng = np.random.default_rng(1)
    noise = 1e-3 * np.max(np.abs(M_meas)) * (rng.standard_normal(M_meas.shape) + 1j * rng.standard_normal(M_meas.shape))

    pu, xu = _flame(unknown_dynamic_source([(1, "u")]))
    out = identify_transfer_function(pu, xu, TransferMatrix(FREQS, M_meas + noise), node=2, a=0, b=3, continue_=False)
    # 0.1% measurement noise -> ~0.1-1% identification error (no blow-up)
    assert np.median(np.abs(out.values[0] - Fu(FREQS))) / np.max(np.abs(Fu(FREQS))) < 5e-2
