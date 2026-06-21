"""Replication of De Domenico, Rolland & Hochgreb (2019), JSV 440, 212-230,
"A generalised model for acoustic and entropic transfer function of nozzles with
losses" (``scratch/De Domenico et al. - 2019 ...pdf``).

That model is **compact** (low frequency), so our ``omega -> 0`` jumps are the
exact regime.  A lossy nozzle is four stations ``A1 -> AT -> Aj -> A2``: isentropic
contraction ``A1->AT`` and isentropic expansion ``AT->Aj`` (total pressure
conserved), then a momentum-conserving Borda jump ``Aj->A2`` (their Eq. 5/6) whose
loss is set by ``beta = Aj/A2`` (``beta_min = AT/A2`` -> orifice plate; ``beta = 1``
-> isentropic nozzle).  We build these from ``isentropic_area_change`` +
``sudden_area_change`` and check the assembled compact scattering matrix in the
De Domenico ``(P+, P-, sigma)`` flavor (== our ``riemann`` basis, Eqs. 9-11)
against an **independent** composition of the analytic jumps.

The sudden element is a smooth Borda<->isentropic switch; its mean flow is exact
Borda, but the frozen perturbation picks up an ``O(eps)`` switch-derivative bias
(TODO.md).  With a sharp per-element ``eps`` (the flow is one-directional here) the
perturbation recovers the exact Borda jump -- which is what these tests assert.
"""

import numpy as np

from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.solver import solve
from fns.solver.control import states_table
from fns.derive import ES_RHO, ES_U, ES_P, ES_C, ES_M, ES_PT
from fns.perturbation import perturbation_response
from fns.perturbation.characteristics import char_to_dq, basis_matrix

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR
CFG = perfect_gas(R_AIR, GAMMA)
ZERO = np.array([0.0])
FULL = ("acoustic", "entropy")
MDOT_REF = 3.0
SHARP = 1e-7 * MDOT_REF  # sudden-element eps: one-directional flow -> exact Borda perturbation


# -- independent analytic compact jumps (primitive vars, complex-step safe) --


def _derived(y):
    rho, u, p = y
    c = (GAMMA * p / rho) ** 0.5
    ht = K * p / rho + 0.5 * u * u
    M = u / c
    pt = p * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))
    return ht, pt


def _r_isen(y0, y1, A0, A1):
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    ht0, pt0 = _derived(y0)
    ht1, pt1 = _derived(y1)
    return np.array([rho1 * u1 * A1 - rho0 * u0 * A0, ht1 - ht0, pt0 - pt1])


def _r_borda(y0, y1, A0, A1):
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    ht0, _ = _derived(y0)
    ht1, _ = _derived(y1)
    md0, md1 = rho0 * u0 * A0, rho1 * u1 * A1
    p_side = p0 if A0 <= A1 else p1  # small-port pressure on the area step (De Domenico Eq. 5)
    mom = (md1 * u1 + p1 * A1) - (md0 * u0 + p0 * A0) - p_side * (A1 - A0)
    return np.array([md1 - md0, ht1 - ht0, mom])


def _jump_tm(rfun, y0, y1, *args):
    """Primitive transfer matrix dy1 = T dy0 by complex-stepping the jump residual."""
    y0 = np.asarray(y0, complex)
    y1 = np.asarray(y1, complex)
    h = 1e-30
    Ja = np.zeros((3, 3))
    Jb = np.zeros((3, 3))
    for k in range(3):
        yp = y0.copy()
        yp[k] += 1j * h
        Ja[:, k] = rfun(yp, y1, *args).imag / h
        yp = y1.copy()
        yp[k] += 1j * h
        Jb[:, k] = rfun(y0, yp, *args).imag / h
    return -np.linalg.solve(Jb, Ja)


def _prim(est, e):
    return np.array([est[ES_RHO, e], est[ES_U, e], est[ES_P, e]])


# -- network builders: the three De Domenico limit cases ---------------------


def _build(mids, areas, mdot, sac_eps=SHARP):
    """Mass-flow-driven 2-terminal nozzle from interior element specs + edge areas.

    Mass-flow drive gives direct, choke-safe control of the throat Mach (an
    isentropic diffuser otherwise chokes the throat at a low pressure ratio).
    """
    net = [cat.mass_flow_inlet(mdot, 300.0)] + mids + [cat.pressure_outlet(101325.0, 300.0)]
    edges = [(i, i + 1, areas[i]) for i in range(len(areas))]
    prob = cat.build_problem(CFG, net, edges, MDOT_REF, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _orifice(A1, AT, A2, mdot):
    """beta = beta_min: isentropic contraction then Borda expansion (Aj = AT)."""
    return _build([cat.isentropic_area_change(), cat.sudden_area_change(eps=SHARP)], [A1, AT, A2], mdot)


def _nonisentropic(A1, AT, Aj, A2, mdot):
    """beta_min < beta < 1: contraction, isentropic expansion to Aj, Borda Aj->A2."""
    mids = [cat.isentropic_area_change(), cat.isentropic_area_change(), cat.sudden_area_change(eps=SHARP)]
    return _build(mids, [A1, AT, Aj, A2], mdot)


def _isentropic_nozzle(A1, AT, A2, mdot):
    """beta = 1: contraction then isentropic expansion (no loss)."""
    return _build([cat.isentropic_area_change(), cat.isentropic_area_change()], [A1, AT, A2], mdot)


def _riemann_scattering(prob, res, a, b):
    return perturbation_response(prob, res.x, ZERO, excite=FULL).scattering_matrix(a, b, basis="riemann")[0]


def _to_riemann(T_char, est, a, b):
    """Convert a characteristic transfer matrix to the (P+, P-, sigma) flavor."""
    Ba = basis_matrix("riemann", est[ES_RHO, a], est[ES_C, a], est[ES_U, a], est[ES_P, a], 1.0, K)
    Bb = basis_matrix("riemann", est[ES_RHO, b], est[ES_C, b], est[ES_U, b], est[ES_P, b], 1.0, K)
    return Bb @ T_char @ np.linalg.inv(Ba)


# -- 1. orifice plate: compact scattering == independent Borda composition ----


def test_orifice_matches_independent_borda_composition():
    A1 = A2 = 0.020
    AT = 0.008
    prob, res = _orifice(A1, AT, A2, mdot=2.0)
    est = states_table(prob, res.x)
    assert 0.3 < est[ES_M, 1] < 0.95  # subsonic, genuinely accelerated throat
    assert est[ES_PT, 1] - est[ES_PT, 2] > 0.0  # a real total-pressure loss across the jet

    # independent reference: isentropic contraction o Borda expansion, in primitive vars
    T_ref_prim = _jump_tm(_r_borda, _prim(est, 1), _prim(est, 2), AT, A2) @ _jump_tm(
        _r_isen, _prim(est, 0), _prim(est, 1), A1, AT
    )
    # our assembled compact char TM, converted to primitive, must match
    T_char = perturbation_response(prob, res.x, ZERO, excite=FULL).transfer_matrix(0, 2)[0]
    R0 = char_to_dq(est[ES_RHO, 0], est[ES_C, 0])
    R2 = char_to_dq(est[ES_RHO, 2], est[ES_C, 2])
    T_ours_prim = (R2 @ T_char @ np.linalg.inv(R0)).real
    assert np.allclose(T_ours_prim, T_ref_prim, rtol=1e-4, atol=1e-4 * np.max(np.abs(T_ref_prim)))


# -- 2. non-isentropic nozzle (intermediate beta): three-jump composition -----


def test_nonisentropic_nozzle_matches_composition():
    A1 = A2 = 0.020
    AT = 0.010
    Aj = 0.014  # beta = Aj/A2 = 0.70, between orifice and isentropic
    prob, res = _nonisentropic(A1, AT, Aj, A2, mdot=2.0)
    est = states_table(prob, res.x)
    assert est[ES_PT, 2] - est[ES_PT, 3] > 0.0  # loss only on the Borda jump Aj->A2
    assert abs(est[ES_PT, 0] - est[ES_PT, 1]) < 1e-3 * est[ES_PT, 0]  # 0->1->2 isentropic

    T_ref = (
        _jump_tm(_r_borda, _prim(est, 2), _prim(est, 3), Aj, A2)
        @ _jump_tm(_r_isen, _prim(est, 1), _prim(est, 2), AT, Aj)
        @ _jump_tm(_r_isen, _prim(est, 0), _prim(est, 1), A1, AT)
    )
    T_char = perturbation_response(prob, res.x, ZERO, excite=FULL).transfer_matrix(0, 3)[0]
    R0 = char_to_dq(est[ES_RHO, 0], est[ES_C, 0])
    R3 = char_to_dq(est[ES_RHO, 3], est[ES_C, 3])
    T_ours = (R3 @ T_char @ np.linalg.inv(R0)).real
    assert np.allclose(T_ours, T_ref, rtol=1e-4, atol=1e-4 * np.max(np.abs(T_ref)))


# -- 3. isentropic-nozzle limit (beta = 1): R+ = 0, T+ = 1 for A1 = A2 --------


def test_isentropic_nozzle_unit_transmission_zero_reflection():
    # De Domenico p.219: under the compact approximation, a fully isentropic nozzle
    # with A1 = A2 has acoustic transmission 1 and reflection 0.
    A1 = A2 = 0.020
    AT = 0.010
    prob, res = _isentropic_nozzle(A1, AT, A2, mdot=2.0)
    est = states_table(prob, res.x)
    assert abs(est[ES_PT, 0] - est[ES_PT, 2]) < 1e-6 * est[ES_PT, 0]  # no loss anywhere
    assert np.allclose(_prim(est, 0), _prim(est, 2), rtol=1e-6)  # identical up/down states

    S = _riemann_scattering(prob, res, 0, 2)  # incoming [P+_a, sigma_a, P-_b] -> [P-_a, P+_b, sigma_b]
    R_plus = S[0, 0]  # P-_1 / P+_1
    T_plus = S[1, 0]  # P+_2 / P+_1  (gamma*p2/gamma*p1 = 1 since p1 = p2)
    assert abs(R_plus) < 1e-6
    assert abs(T_plus - 1.0) < 1e-6


# -- 4. the De Domenico (P+, P-, sigma) flavor == our riemann basis -----------


def test_scattering_riemann_equals_dedomenico_normalisation():
    A1 = A2 = 0.020
    AT = 0.008
    prob, res = _orifice(A1, AT, A2, mdot=2.0)
    est = states_table(prob, res.x)
    # build the SM from the char SM by an explicit (P+,P-,sigma) rescale and compare
    resp = perturbation_response(prob, res.x, ZERO, excite=FULL)
    S_riemann = resp.scattering_matrix(0, 2, basis="riemann")[0]
    S_char = resp.scattering_matrix(0, 2, basis="char")[0]
    # incoming/outgoing wave indices (a-downstream f & sigma; b-upstream g)
    inc = [("a", 0), ("a", 2), ("b", 1)]
    out = [("a", 1), ("b", 0), ("b", 2)]

    def scale(station, i):
        e = 0 if station == "a" else 2
        B = basis_matrix("riemann", est[ES_RHO, e], est[ES_C, e], est[ES_U, e], est[ES_P, e], 1.0, K)
        return B[i, i]

    din = np.array([scale(s, i) for (s, i) in inc])
    dout = np.array([scale(s, i) for (s, i) in out])
    assert np.allclose(S_riemann, (dout[:, None] * S_char) / din[None, :], atol=1e-9)


# -- 5. per-element eps sharpens the raw TM; the scattering is eps-robust -----


def test_sudden_eps_sharpens_tm_but_scattering_is_robust():
    # The smooth momentum<->isentropic switch biases the *raw* transfer matrix by
    # O(eps) (in the large entropy-coupling entries); a sharp per-element eps
    # removes it.  The physical *scattering* coefficients (R+, T+, S_R, S_T) are
    # eps-robust to ~1e-6 either way -- so the De Domenico validation holds at the
    # default smoothing; the sharp eps is a refinement, not a requirement.
    D1, DT = 0.0426, 0.0066  # Cambridge geometry, A1/AT ~ 42 (a strong, high-loss orifice)
    A1 = A2 = np.pi / 4 * D1**2
    AT = np.pi / 4 * DT**2

    def extract(eps_sac):
        net = [
            cat.total_pressure_inlet(140000.0, 300.0),
            cat.isentropic_area_change(),
            cat.sudden_area_change(eps=eps_sac),
            cat.pressure_outlet(101325.0, 300.0),
        ]
        prob = cat.build_problem(CFG, net, [(0, 1, A1), (1, 2, AT), (2, 3, A2)], 1.0, 101325.0, CP * 300.0)
        res = solve(prob)
        assert res.converged
        est = states_table(prob, res.x)
        assert est[ES_M, 1] < 1.0  # subsonic throat (v1)
        resp = perturbation_response(prob, res.x, ZERO, excite=FULL)
        Ra = char_to_dq(est[ES_RHO, 1], est[ES_C, 1])
        Rb = char_to_dq(est[ES_RHO, 2], est[ES_C, 2])
        T_prim = (Rb @ resp.transfer_matrix(1, 2)[0] @ np.linalg.inv(Ra)).real
        T_ref = _jump_tm(_r_borda, _prim(est, 1), _prim(est, 2), AT, A2)
        tm_dev = np.max(np.abs(T_prim - T_ref)) / np.max(np.abs(T_ref))
        return tm_dev, resp.scattering_matrix(0, 2, basis="riemann")[0]

    tm_default, S_default = extract(None)
    tm_sharp, S_sharp = extract(1e-7)  # absolute eps in mdot units (mdot_throat ~ 0.3 >> 1e-7)
    assert tm_default > 1e-5  # the raw TM carries a real O(eps) switch bias
    assert tm_sharp < tm_default / 50.0  # the per-element eps sharpens it away
    assert np.max(np.abs(S_default - S_sharp)) < 1e-4  # scattering coefficients are eps-robust
