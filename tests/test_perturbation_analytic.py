"""Analytic verification of the perturbation network (theory.md s12.1-12.2).

The compact elements of s7 are *instantaneous algebraic* jumps, so the zero-
frequency perturbation matrix the solver assembles (the converged complex-step
Jacobian, transformed by the characteristic maps) *is* the linearised jump
condition.  This module proves that against **independent** analytic references,
element by element:

1. **Closed form.**  The compact isentropic area change must reproduce the
   Marble--Candel / De Domenico compact-nozzle transfer matrix -- including the
   entropy -> acoustic coupling (entropy noise) -- to solver accuracy.

2. **Independent re-derivation.**  For every interior element, a standalone
   *primitive-variable* statement of the exact physical jump (mass, total
   enthalpy, and the element's pressure/momentum law), linearised by an
   independent complex step, must match the assembled matrix.  This shares no
   code with the @njit kernels or the assembly path.

3. **Structure.**  In the network basis ``(mdot', p', h_t')`` the mass row is
   ``[1,0,0]`` and the enthalpy row ``[0,0,1]`` for every adiabatic element:
   mass and **nodal energy** are conserved across the jump exactly.  This is the
   perturbation-network realisation of the edge total-enthalpy transport rows
   being a *view* of nodal energy conservation (theory.md s6.2).
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.solver import solve
from nefes.solver.control import states_table
from nefes.assembly.derive import ES_RHO, ES_U, ES_P, ES_C, ES_M
from nefes.perturbation import perturbation_response
from nefes.perturbation.operator.characteristics import char_to_dq, basis_block_from_state

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
K = CP / R_AIR  # cp / R
CFG = perfect_gas(R_AIR, GAMMA)
ZERO = np.array([0.0])  # omega -> 0: the algebraic (compact) limit
FULL = ("acoustic", "entropy")  # drive the entropy wave too -> full 3x3 response


# --------------------------------------------------------------------------
# Standalone physical jump residuals in primitive variables y = (rho', u', p').
# Dtype-generic (complex-step safe): only sqrt / powers / arithmetic.
# --------------------------------------------------------------------------


def _derived(y):
    """(c, M, p_t, h_t) from primitive (rho, u, p)."""
    rho, u, p = y
    c = (GAMMA * p / rho) ** 0.5
    M = u / c
    pt = p * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))
    ht = K * p / rho + 0.5 * u * u
    return c, M, pt, ht


def _r_isentropic(y0, y1, A0, A1):
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    _, _, pt0, ht0 = _derived(y0)
    _, _, pt1, ht1 = _derived(y1)
    return np.array([rho1 * u1 * A1 - rho0 * u0 * A0, ht1 - ht0, pt0 - pt1])  # mass, energy, [p_t]=0


def _r_sudden(y0, y1, A0, A1):
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    _, _, _, ht0 = _derived(y0)
    _, _, _, ht1 = _derived(y1)
    mdot0, mdot1 = rho0 * u0 * A0, rho1 * u1 * A1
    p_side = p0 if A0 <= A1 else p1  # side wall sees the small-port (here upstream) pressure
    mom = (mdot1 * u1 + p1 * A1) - (mdot0 * u0 + p0 * A0) - p_side * (A1 - A0)
    return np.array([mdot1 - mdot0, ht1 - ht0, mom])


def _r_sudden_contraction(y0, y1, A0, A1, cc):
    """Reverse (large -> small) sudden contraction: vena-contracta loss law.

    Active when ``A0 > A1`` (side 1 is the small / downstream port): mass, energy,
    and a downstream-referenced total-pressure loss ``K_c * (1/2 rho u^2)_small``
    with ``K_c = (1/cc - 1)^2``.  ``cc = 1`` collapses to total-pressure continuity.
    """
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    _, _, pt0, ht0 = _derived(y0)
    _, _, pt1, ht1 = _derived(y1)
    K_c = (1.0 / cc - 1.0) ** 2
    q_small = 0.5 * rho1 * u1 * u1  # small side = downstream = side 1 (A1 < A0)
    return np.array([rho1 * u1 * A1 - rho0 * u0 * A0, ht1 - ht0, (pt0 - pt1) - K_c * q_small])


def _r_loss(y0, y1, A0, A1, K_loss):
    rho0, u0, p0 = y0
    rho1, u1, p1 = y1
    _, _, pt0, ht0 = _derived(y0)
    _, _, pt1, ht1 = _derived(y1)
    rho_avg = 0.5 * (rho0 + rho1)
    u_ref = (rho0 * u0 * A0) / (rho_avg * A0)  # forward flow: q = 1/2 rho_avg u_ref^2
    q = 0.5 * rho_avg * u_ref * u_ref
    return np.array([rho1 * u1 * A1 - rho0 * u0 * A0, ht1 - ht0, pt0 - pt1 - K_loss * q])


def _jump_tm(rfun, y0, y1, *args):
    """Primitive transfer matrix ``dy1 = T dy0`` by complex-stepping the jump residual."""
    y0 = np.asarray(y0, dtype=complex)
    y1 = np.asarray(y1, dtype=complex)
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
    return -np.linalg.solve(Jb, Ja)  # r = 0 => Jb dy1 + Ja dy0 = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _two_port(mid, A0, A1, pt_in=120000.0, p_out=101325.0):
    net = [cat.total_pressure_inlet(pt_in, 300.0), mid, cat.pressure_outlet(p_out, 300.0)]
    prob = cat.build_problem(CFG, net, [(0, 1, A0), (1, 2, A1)], 10.0, 101325.0, CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res


def _assembled_prim_tm(prob, res):
    """Assembled omega->0 transfer matrix between edges 0 and 1, in primitive vars."""
    est = states_table(prob, res.x)
    R0 = char_to_dq(est[ES_RHO, 0], est[ES_C, 0])
    R1 = char_to_dq(est[ES_RHO, 1], est[ES_C, 1])
    T_char = perturbation_response(prob, res.x, ZERO, excite=FULL).transfer_matrix(0, 1)[0]
    return (R1 @ T_char @ np.linalg.inv(R0)).real, est


def _prim(est, e):
    return np.array([est[ES_RHO, e], est[ES_U, e], est[ES_P, e]])


# --------------------------------------------------------------------------
# 1. Closed-form anchor: the compact isentropic area change is Marble--Candel.
# --------------------------------------------------------------------------


def _C_isentropic(M):
    """Conserved-quantity functional (mdot'/mdot, T_t'/T_t, sigma) <- (p/gp, u/c, sigma)."""
    D = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    return np.array(
        [
            [1.0, 1.0 / M, -1.0],
            [(GAMMA - 1.0) / D, (GAMMA - 1.0) * M / D, 1.0 / D],
            [0.0, 0.0, 1.0],
        ]
    )


def test_isentropic_area_change_matches_marble_candel():
    prob, res = _two_port(cat.isentropic_area_change(), 0.10, 0.06, pt_in=130000.0)
    T_prim, est = _assembled_prim_tm(prob, res)
    M0, M1 = est[ES_M, 0], est[ES_M, 1]
    assert 0.05 < M0 < M1 < 0.95  # genuinely flowing & subsonic, with acceleration

    # analytic transfer matrix in the De Domenico basis (p/gp, u/c, sigma)
    T_dd_analytic = np.linalg.inv(_C_isentropic(M1)) @ _C_isentropic(M0)

    # convert the assembled primitive matrix to the same basis and compare
    def B_dd(e):  # (p/gp, u/c, sigma) <- (rho', u', p')
        rho, c, p = est[ES_RHO, e], est[ES_C, e], est[ES_P, e]
        return np.array([[0.0, 0.0, 1.0 / (GAMMA * p)], [0.0, 1.0 / c, 0.0], [-1.0 / rho, 0.0, 1.0 / (GAMMA * p)]])

    T_dd_assembled = (B_dd(1) @ T_prim @ np.linalg.inv(B_dd(0))).real
    assert np.allclose(T_dd_assembled, T_dd_analytic, atol=1e-6)

    # the entropy -> acoustic coupling (entropy noise) is genuinely present, not a
    # decoupled trivial case: the sigma column drives p and u fluctuations.
    assert np.linalg.norm(T_dd_analytic[:2, 2]) > 0.1


# --------------------------------------------------------------------------
# 2. Independent re-derivation: primitive physical jump == assembled matrix.
# --------------------------------------------------------------------------


def test_isentropic_area_change_independent_jump():
    prob, res = _two_port(cat.isentropic_area_change(), 0.10, 0.06, pt_in=130000.0)
    T_prim, est = _assembled_prim_tm(prob, res)
    T_ref = _jump_tm(_r_isentropic, _prim(est, 0), _prim(est, 1), 0.10, 0.06)
    assert np.allclose(T_prim, T_ref, rtol=1e-6, atol=1e-6)


def test_sudden_area_change_independent_jump():
    # a sudden EXPANSION with forward flow -> the momentum (jet) branch is active
    prob, res = _two_port(cat.sudden_area_change(), 0.05, 0.09, pt_in=115000.0)
    T_prim, est = _assembled_prim_tm(prob, res)
    assert est[ES_M, 0] > est[ES_M, 1]  # accelerated into the throat, expands after
    T_ref = _jump_tm(_r_sudden, _prim(est, 0), _prim(est, 1), 0.05, 0.09)
    assert np.allclose(T_prim, T_ref, rtol=1e-5, atol=1e-5 * np.max(np.abs(T_ref)))


def test_sudden_contraction_independent_jump():
    # a sudden CONTRACTION with forward flow (large -> small): the vena-contracta
    # loss branch is active and cc < 1 makes the jump genuinely lossy.  A sharp eps
    # (1e-7 * mdot_ref) saturates the regime switch so the assembled jump is the
    # clean loss-law linearisation, with no switch-derivative leak from the inactive
    # momentum branch.
    cc = 0.62
    prob, res = _two_port(cat.sudden_area_change(cc=cc, eps=1e-7 * 10.0), 0.09, 0.05, pt_in=120000.0)
    T_prim, est = _assembled_prim_tm(prob, res)
    assert est[ES_M, 1] > est[ES_M, 0]  # contracting: accelerates into the small pipe
    T_ref = _jump_tm(_r_sudden_contraction, _prim(est, 0), _prim(est, 1), 0.09, 0.05, cc)
    assert np.allclose(T_prim, T_ref, rtol=1e-4, atol=1e-4 * np.max(np.abs(T_ref)))


def test_loss_element_independent_jump():
    K_loss = 2.5
    # the concentrated loss is a constant-area element (its K is referenced to the
    # inlet dynamic head); the jump stays non-trivial through the total-pressure drop.
    prob, res = _two_port(cat.loss(K_loss), 0.08, 0.08, pt_in=120000.0)
    T_prim, est = _assembled_prim_tm(prob, res)
    T_ref = _jump_tm(_r_loss, _prim(est, 0), _prim(est, 1), 0.08, 0.08, K_loss)
    # the loss kernel smooths |u| with width eps ~ 1e-4*mdot_ref; away from u=0 the
    # smoothing is saturated, so the match is tight.
    assert np.allclose(T_prim, T_ref, rtol=1e-4, atol=1e-4 * np.max(np.abs(T_ref)))


# --------------------------------------------------------------------------
# 3. Structure: mass + nodal-energy continuity across every adiabatic element.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mid, A0, A1",
    [
        (cat.isentropic_area_change(), 0.08, 0.05),
        (cat.sudden_area_change(), 0.05, 0.08),
        (cat.sudden_area_change(cc=0.62), 0.09, 0.05),  # lossy contraction: mass+energy still continuous
        (cat.loss(2.5), 0.08, 0.08),  # loss is constant-area (equal-area rule)
        (cat.duct(1.0), 0.06, 0.06),  # length inert at omega -> 0 (phases -> 1)
    ],
)
def test_mass_and_nodal_energy_continuity(mid, A0, A1):
    prob, res = _two_port(mid, A0, A1)
    T_net = perturbation_response(prob, res.x, ZERO, excite=FULL).transfer_matrix(0, 1, basis="network")[0].real
    assert np.allclose(T_net[0], [1.0, 0.0, 0.0], atol=1e-7)  # mdot' continuity
    assert np.allclose(T_net[2], [0.0, 0.0, 1.0], atol=1e-7)  # h_t' (nodal energy) continuity


# --------------------------------------------------------------------------
# 4. Basis round-trips and TM <-> SM consistency on a real element.
# --------------------------------------------------------------------------


def test_basis_and_scattering_roundtrips():
    from nefes.perturbation import matrices as mat

    prob, res = _two_port(cat.isentropic_area_change(), 0.10, 0.06, pt_in=130000.0)
    est = states_table(prob, res.x)
    resp = perturbation_response(prob, res.x, np.linspace(100.0, 1200.0, 5), excite=FULL)
    ua, ca = est[ES_U, 0], est[ES_C, 0]
    ub, cb = est[ES_U, 1], est[ES_C, 1]
    T = resp.transfer_matrix(0, 1)  # char
    # TM -> SM -> TM round-trips
    S, _in, _out = mat.tm_to_sm(T, ua, ca, ub, cb)
    assert np.allclose(mat.sm_to_tm(S, ua, ca, ub, cb), T, atol=1e-9)
    # SM from the stored fields agrees with SM converted from the TM
    assert np.allclose(resp.scattering_matrix(0, 1), S, atol=1e-9)
    # a flavor change is an invertible similarity: primitive then back == char
    Tp = resp.transfer_matrix(0, 1, basis="primitive")
    Ba = basis_block_from_state("primitive", est[:, 0], K)
    Bb = basis_block_from_state("primitive", est[:, 1], K)
    assert np.allclose(mat.tm_in_basis(Tp, np.linalg.inv(Ba), np.linalg.inv(Bb)), T, atol=1e-9)
