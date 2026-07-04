"""The TRANSFER_MATRIX element: mean flow == isentropic area change; perturbation acoustics
are a user-supplied transfer matrix.

Targets:
* the mean flow is bit-identical to an :func:`isentropic_area_change`;
* loading the element with an element's OWN extracted transfer matrix reproduces its
  scattering matrix (the de-embed round-trip) for N=3 and N=2;
* the fast assembler equals the slow reference stamp at real and complex omega (stability).

(The mean-flow kernel's complex-step safety is covered by tests/test_complex_step_safety.py.)
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.perturbation import perturbation_response, TransferMatrix, PortState
from nefes.perturbation.operator.operator import build_acoustic_blocks, assemble_acoustic, _assemble_reference
from nefes.assembly.recover import ES_RHO, ES_C, ES_U, ES_P, ES_AREA

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
CFG = perfect_gas(R_AIR, GAMMA)
FREQS = np.linspace(50.0, 1500.0, 13)
FULL = ("acoustic", "entropy")
A1, A2 = 0.05, 0.03


def _build(mid):
    net = [
        cat.total_pressure_inlet(120000.0, 300.0),
        cat.duct(0.7),
        mid,
        cat.duct(1.1),
        cat.pressure_outlet(101325.0, 300.0),
    ]
    edges = [(0, 1, A1), (1, 2, A1), (2, 3, A2), (3, 4, A2)]
    prob = cat.build_problem(CFG, net, edges, 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


def _port(est, e):
    return PortState(est[ES_RHO, e], est[ES_C, e], est[ES_U, e], est[ES_P, e], est[ES_AREA, e])


def test_mean_flow_matches_isentropic_area_change():
    pa, xa = _build(cat.isentropic_area_change())
    pb, xb = _build(cat.transfer_matrix_element())
    assert np.max(np.abs(states_table(pa, xa) - states_table(pb, xb))) < 1e-10


@pytest.mark.parametrize("N,excite", [(3, FULL), (2, ("acoustic",))])
def test_loading_own_tm_reproduces_scattering(N, excite):
    # extract the element's own N x N transfer matrix, load it into a TM element, and confirm
    # the network scattering is reproduced -- the identification de-embed round-trip.
    pa, xa = _build(cat.isentropic_area_change())
    ref = perturbation_response(pa, xa, FREQS, excite=excite)
    T = ref.transfer_matrix(1, 2, basis="char")
    assert T.shape[1] == N
    S_ref = ref.scattering_matrix(0, 3, basis="char")

    est = states_table(pa, xa)
    tm = TransferMatrix(FREQS, T, basis="char", ports=(_port(est, 1), _port(est, 2)))
    pb, xb = _build(cat.transfer_matrix_element(tm=tm))
    S_tst = perturbation_response(pb, xb, FREQS, excite=excite).scattering_matrix(0, 3, basis="char")

    assert np.max(np.abs(S_ref - S_tst)) / np.max(np.abs(S_ref)) < 1e-8


@pytest.mark.parametrize("N", [3, 2])
def test_fast_plan_equals_reference_real_and_complex(N):
    pa, xa = _build(cat.isentropic_area_change())
    ref = perturbation_response(pa, xa, FREQS, excite=FULL)
    est = states_table(pa, xa)
    T = ref.transfer_matrix(1, 2, basis="char")
    if N == 2:
        T = T[:, :2, :2]
    tm = TransferMatrix(FREQS, T, basis="char", ports=(_port(est, 1), _port(est, 2))).continue_(rtol=1e-11)
    pb, xb = _build(cat.transfer_matrix_element(tm=tm))
    blocks = build_acoustic_blocks(pb, xb)
    for w in (2 * np.pi * 300.0, 2 * np.pi * (700.0 - 40.0j)):  # real + complex (unstable) omega
        A_fast = assemble_acoustic(w, blocks, with_boundaries=True).toarray()
        A_slow = _assemble_reference(w, blocks, with_boundaries=True).toarray()
        assert np.max(np.abs(A_fast - A_slow)) < 1e-9


def test_rejects_non_2port_and_bad_dimension():
    from nefes.perturbation.operator.stamps import build_tm_stamps

    # a 4x4 "transfer matrix" is neither acoustic (2) nor full (3) -> rejected at build
    est_freqs = FREQS
    bad = TransferMatrix(est_freqs, np.tile(np.eye(4), (est_freqs.size, 1, 1)))
    pb, xb = _build(cat.transfer_matrix_element(tm=bad))
    with pytest.raises(ValueError):
        build_tm_stamps(pb, xb, CP / R_AIR)
