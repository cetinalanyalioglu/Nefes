"""Verification of the perturbation boundary conditions (theory.md s12.4).

Every terminal BC is a reflection relation ``w_incoming - R(omega) w_outgoing = b``.
The physical, non-circular check is a **duct terminated by the BC**: drive an acoustic
wave at one end and read the input reflection at that end, which transmission-line
theory fixes as

    Gamma_in(omega) = R_term * exp(-i omega (tau_+ + tau_-)),

with ``tau_+ = L/(u + c)`` and ``tau_- = L/(c - u)`` the duct round-trip delays.  This
exercises the BC stamp, the duct propagation, and the forced-response driver together,
and reproduces the analytic reflection coefficient of each closure.  The wall element
is checked both for its mean flow (``mdot = 0``) and for terminating a duct as an
acoustic hard wall.  A separate block unit-tests :class:`PerturbationBC` itself.
"""

import os
import warnings

import numpy as np
import pytest
import yaml

from nefes.shell import Network
from nefes.elements import catalog as cat
from nefes.elements.ids import WALL
from nefes.io import load_case
from nefes.thermo.configure import perfect_gas, perfect_gas_passive_scalars
from nefes.assembly.recover import ES_U, ES_C, ES_RHO, ES_P
from nefes.perturbation import PerturbationBC, forced_response, CompositionalNoiseWarning
from nefes.perturbation.operator.characteristics import char_to_dq

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")

R_AIR, GAMMA = 287.0, 1.4
CFG = perfect_gas(R_AIR, GAMMA)
OMEGAS = np.linspace(80.0, 3200.0, 9)  # angular frequencies (rad/s) for the e^{-iwt} phase checks
FREQS = OMEGAS / (2.0 * np.pi)  # the matching Hz sweep fed to the (Hz) forced_response API
LDUCT = 0.5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _duct_case(inlet_bc, outlet_bc, *, pt_in=104000.0, p_out=101325.0, L=LDUCT, area=0.05):
    """[total-pressure inlet] -- duct(L) -- [pressure outlet], with the given BCs."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(pt_in, 300.0, perturbation_bc=inlet_bc))
    net.add(cat.duct(L))
    net.add(cat.pressure_outlet(p_out, 300.0, perturbation_bc=outlet_bc))
    net.connect(0, 1, area)
    net.connect(1, 2, area)
    sol = net.solve()
    assert sol.converged
    return net, sol


def _uc(sol, e=0):
    est = sol.table()
    return float(est[ES_U, e]), float(est[ES_C, e])


def _roundtrip(L, u, c, omegas):
    return np.exp(-1j * omegas * (L / (u + c) + L / (c - u)))


# --------------------------------------------------------------------------
# 1. Terminated-duct input reflection: one analytic R per closure.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outlet_bc, Rval",
    [
        (PerturbationBC.hard_wall(), 1.0),
        (PerturbationBC.open_end(), -1.0),
        (PerturbationBC.anechoic(), 0.0),
        (PerturbationBC.reflection(0.5 - 0.3j), 0.5 - 0.3j),
    ],
)
def test_terminated_duct_reflection(outlet_bc, Rval):
    _, sol = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), outlet_bc)
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    expected = Rval * _roundtrip(LDUCT, u, c, OMEGAS)
    assert np.allclose(fr.reflection_at(0), expected, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("zeta", [2.0, 0.5, 1.5 - 0.4j])
def test_impedance_specific_and_absolute(zeta):
    R = (zeta - 1.0) / (zeta + 1.0)
    # specific impedance: R independent of rho*c
    _, sol = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.impedance(zeta, specific=True))
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    assert np.allclose(fr.reflection_at(0), R * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)

    # absolute impedance Z = zeta * (rho c) read at the outlet edge -> same R
    est = sol.table()
    rho_c = float(est[ES_RHO, 1]) * float(est[ES_C, 1])
    _, sol2 = _duct_case(
        PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.impedance(zeta * rho_c, specific=False)
    )
    u2, c2 = _uc(sol2)
    fr2 = forced_response(sol2.problem, sol2.x, FREQS)
    assert np.allclose(fr2.reflection_at(0), R * _roundtrip(LDUCT, u2, c2, OMEGAS), atol=1e-9)


def test_mean_flow_open_end():
    _, sol = _duct_case(
        PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.mean_flow_open_end(), pt_in=115000.0
    )
    u, c = _uc(sol)
    M = u / c
    R = -(1.0 - M) / (1.0 + M)
    assert M > 0.2 and abs(R + 1.0) > 0.05  # genuinely mean-flow-corrected, not the ideal open end
    fr = forced_response(sol.problem, sol.x, FREQS)
    assert np.allclose(fr.reflection_at(0), R * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)


def test_frequency_dependent_reflection_table():
    f_tab = np.linspace(0.0, 6000.0, 13)  # reflection table in Hz
    R_tab = 0.2 + 0.1j + (f_tab / 6000.0) * (0.5 - 0.2j)  # ramps with frequency
    _, sol = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.reflection((f_tab, R_tab)))
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    R_at = np.interp(FREQS, f_tab, R_tab.real) + 1j * np.interp(FREQS, f_tab, R_tab.imag)
    assert np.allclose(fr.reflection_at(0), R_at * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)


def test_frequency_dependent_reflection_callable():
    # A reflection coefficient given as a callable R(freq_hz) -- a frequency-domain
    # boundary model (here a single-pole low-pass with a phase lag).  The terminated-duct
    # input reflection must track R(f) * roundtrip at every frequency.
    def R_of_f(f):
        return (0.6 / (1.0 + 1j * f / 800.0)) * np.exp(-1j * 2.0 * np.pi * f * 2.0e-4)

    _, sol = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.reflection(R_of_f))
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    R_at = np.array([R_of_f(f) for f in FREQS])
    assert np.allclose(fr.reflection_at(0), R_at * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-9)


def test_excitation_pins_incoming_wave_and_propagates():
    amp = 0.7 - 0.2j
    _, sol = _duct_case(
        PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": amp}), PerturbationBC.anechoic()
    )
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    # the excitation pins the incoming downstream wave f at the inlet edge
    assert np.allclose(fr.waves(0)[:, 0], amp, atol=1e-9)
    # anechoic outlet: no reflected (upstream) wave returns
    assert np.allclose(fr.waves(1)[:, 1], 0.0, atol=1e-9)
    # f propagates down the duct with the downstream phase exp(-i w tau_+)
    assert np.allclose(fr.waves(1)[:, 0], amp * np.exp(-1j * OMEGAS * (LDUCT / (u + c))), atol=1e-9)


@pytest.mark.parametrize(
    "inlet_bc",
    [
        PerturbationBC.anechoic(
            driven=("entropy",), amplitudes={"entropy": 0.6 - 0.2j}
        ),  # entropy-family excitation seat
        PerturbationBC.anechoic(entropy_in=0.6 - 0.2j),  # incoming entropy on a non-forcing BC
    ],
)
def test_entropy_seat_convects_through_flowing_duct(inlet_bc):
    # Seat an incoming entropy wave at the (flowing) inlet and read it at the outlet.
    # Both entropy paths -- the entropy-family excitation and the entropy_in carrier --
    # land on the inlet edge's transport row with an acoustically anechoic closure
    # (R = 0). With no area change entropy stays decoupled from sound and convects at
    # the mean speed u, so h_outlet = h_inlet * exp(-i w L/u); f and g vanish.
    amp = 0.6 - 0.2j
    _, sol = _duct_case(inlet_bc, PerturbationBC.anechoic(), pt_in=115000.0)
    u, c = _uc(sol)
    assert u > 1.0  # genuinely flowing, so the entropy wave convects
    fr = forced_response(sol.problem, sol.x, FREQS)
    assert np.allclose(fr.waves(0)[:, 2], amp, atol=1e-9)  # entropy seated at the inlet edge
    assert np.allclose(fr.waves(1)[:, 2], amp * np.exp(-1j * OMEGAS * (LDUCT / u)), atol=1e-9)  # convected out
    assert np.allclose(fr.waves(0)[:, 0], 0.0, atol=1e-9)  # no incoming acoustic f at the inlet
    assert np.allclose(fr.waves(1)[:, 1], 0.0, atol=1e-9)  # anechoic outlet: no returning g


def test_inherited_pressure_outlet_is_pressure_release():
    # An outlet left at 'inherit' keeps its linearized mean BC; for a subsonic
    # pressure outlet that is p' = 0 -- the ideal open end R = -1 (theory.md s12.4,
    # "continuity with the steady solution").
    _, sol = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), PerturbationBC.inherit())
    u, c = _uc(sol)
    fr = forced_response(sol.problem, sol.x, FREQS)
    assert np.allclose(fr.reflection_at(0), -1.0 * _roundtrip(LDUCT, u, c, OMEGAS), atol=1e-8)


# --------------------------------------------------------------------------
# 1b. Linearity: the perturbation response scales/superposes with the excitation.
# --------------------------------------------------------------------------


def test_forced_response_scales_linearly_with_amplitude():
    """Scaling the excitation scales the whole field by the same (complex) factor.

    The perturbation problem is a linear system ``A(omega) x = b``; the operator is
    fixed by the mean state and the closures (here a constant reflection, so the same
    for every amplitude), and only ``b`` carries the excitation.  So the field must be
    exactly proportional to the excitation amplitude -- the defining test of linearity.
    """
    outlet = PerturbationBC.reflection(0.4 + 0.1j)
    _, s1 = _duct_case(PerturbationBC.anechoic(driven=("acoustic",)), outlet)
    X1 = forced_response(s1.problem, s1.x, FREQS).X
    assert np.linalg.norm(X1) > 0.0  # genuinely excited
    for scale in (2.0, 0.25, -3.0 + 1.5j):
        _, s = _duct_case(PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": scale}), outlet)
        X = forced_response(s.problem, s.x, FREQS).X
        assert np.allclose(X, scale * X1, atol=1e-10, rtol=1e-8)


def test_forced_response_obeys_superposition():
    """Independent excitations add: ``f(acoustic a, 0) + f(0, entropy b) == f(a, b)``.

    With a clean (``R = 0``) excitation seat the operator is identical across the three
    cases and only the right-hand side differs, so linearity makes the combined response
    the exact sum of the individual ones.
    """
    a, b = 0.7 - 0.2j, 0.4 + 0.5j
    outlet = PerturbationBC.reflection(0.3 - 0.2j)

    def field(amp, ent):  # flowing inlet so the seated entropy wave convects
        _, sol = _duct_case(
            PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": amp}, entropy_in=ent),
            outlet,
            pt_in=115000.0,
        )
        return forced_response(sol.problem, sol.x, FREQS).X

    Xa, Xe, Xae = field(a, 0.0), field(0.0, b), field(a, b)
    assert np.allclose(Xa + Xe, Xae, atol=1e-10, rtol=1e-8)
    assert not np.allclose(Xa, Xae)  # both excitations genuinely contribute


# --------------------------------------------------------------------------
# 1c. Compact choked-nozzle outlet (Marble--Candel): entropy -> acoustic coupling.
# --------------------------------------------------------------------------


def _reduced_massflow(y):
    """Choked reduced mass flow ``~ rho*u*sqrt(Tt)/pt`` (throat area constant)."""
    rho, u, p = y
    c = (GAMMA * p / rho) ** 0.5
    M = u / c
    D = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    Tt = (p / (rho * R_AIR)) * D
    pt = p * D ** (GAMMA / (GAMMA - 1.0))
    return rho * u * Tt**0.5 / pt


def _choked_reflection(rho, u, p):
    """Independent ``(R, R_s)`` for a compact choked outlet.

    Complex-steps ``delta(reduced mass flow) = 0`` and projects onto the characteristics
    (``g = R f + R_s h``).  Shares no code with the BC implementation.
    """
    y = np.array([rho, u, p], dtype=complex)
    h = 1e-30
    Lp = np.zeros(3)
    for k in range(3):
        yp = y.copy()
        yp[k] += 1j * h
        Lp[k] = _reduced_massflow(yp).imag / h
    c = (GAMMA * p / rho) ** 0.5
    coef = Lp @ char_to_dq(rho, c)  # c_f f + c_g g + c_h h = 0
    return -coef[0] / coef[1], -coef[2] / coef[1]  # g = R f + R_s h


def test_choked_nozzle_outlet_marble_candel():
    # Drive acoustic f and seat entropy h at the (flowing) inlet; the compact choked
    # outlet must reflect g = R f + R_s h with the Marble--Candel coefficients.
    inlet = PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": 0.5}, entropy_in=0.7 - 0.2j)
    _, sol = _duct_case(inlet, PerturbationBC.choked_nozzle(), pt_in=135000.0)
    est = sol.table()
    rho, p = float(est[ES_RHO, 1]), float(est[ES_P, 1])
    u, c = _uc(sol, 1)
    M = u / c
    assert 0.1 < M < 0.9  # genuinely flowing, subsonic
    R, R_s = _choked_reflection(rho, u, p)
    assert R == pytest.approx((2 - (GAMMA - 1) * M) / (2 + (GAMMA - 1) * M), rel=1e-6)  # literature R(M)
    fr = forced_response(sol.problem, sol.x, FREQS)
    f1, g1, h1 = fr.waves(1)[:, 0], fr.waves(1)[:, 1], fr.waves(1)[:, 2]
    assert np.allclose(g1, R * f1 + R_s * h1, atol=1e-7, rtol=1e-6)  # BC encodes the coupling
    assert np.max(np.abs(R_s * h1)) > 0.05 * np.max(np.abs(g1))  # entropy noise genuinely active


def test_choked_nozzle_coefficients_and_limits():
    bc = PerturbationBC.choked_nozzle()
    rho, c, K = 1.2, 340.0, GAMMA / (GAMMA - 1.0)  # K = cp/R
    gm1 = GAMMA - 1.0
    # M -> 0: hard wall, no entropy noise
    assert bc.reflection_coefficient(0.0, rho, c, 0.0, K) == pytest.approx(1.0)
    assert bc.entropy_coupling_coefficient(0.0, rho, c, 0.0, K) == pytest.approx(0.0)
    # finite M: the Marble--Candel closed forms
    for M in (0.2, 0.5, 0.8):
        assert bc.reflection_coefficient(0.0, rho, c, M, K) == pytest.approx((2 - gm1 * M) / (2 + gm1 * M))
        assert bc.entropy_coupling_coefficient(0.0, rho, c, M, K) == pytest.approx((c / rho) * M / (2 + gm1 * M))
    # choked nozzle only terminates an outlet (entropy must be an arriving wave)
    with pytest.raises(ValueError):
        bc.closure(0.0, rho, c, 0.0, 0.0, K, specify=(0, 2), arriving=(1,))


def test_choked_nozzle_effective_gamma_from_state():
    # The effective gamma is taken from the state (rho c^2 / p) when p is given -- backend-correct --
    # and equals the perfect-gas K result exactly when p is the perfect-gas pressure.  A *different* p
    # (a non-1.4 effective gamma, as a reacting mixture would have) shifts R, proving p is honoured.
    bc = PerturbationBC.choked_nozzle()
    rho, c, M = 1.2, 340.0, 0.4
    K = GAMMA / (GAMMA - 1.0)
    p_pg = rho * c * c / GAMMA  # the pressure consistent with gamma = 1.4
    assert bc.reflection_coefficient(0.0, rho, c, M, p=p_pg) == pytest.approx(
        bc.reflection_coefficient(0.0, rho, c, M, K)
    )
    g_eff = 1.3  # a different effective gamma -> a different (correct-for-that-gas) reflection
    p_eff = rho * c * c / g_eff
    R_eff = bc.reflection_coefficient(0.0, rho, c, M, K, p=p_eff)  # p overrides K
    assert R_eff == pytest.approx((2 - (g_eff - 1) * M) / (2 + (g_eff - 1) * M))
    assert R_eff != pytest.approx((2 - (GAMMA - 1) * M) / (2 + (GAMMA - 1) * M))


def test_generic_outlet_entropy_coupling():
    # The generic off-diagonal carrier: g = R f + R_s h with user-set constants.
    Rv, Rsv = 0.3 - 0.1j, 0.45 + 0.2j
    inlet = PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": 0.5}, entropy_in=0.6 - 0.3j)
    outlet = PerturbationBC.reflection(Rv, entropy_coupling=Rsv)
    _, sol = _duct_case(inlet, outlet, pt_in=130000.0)
    u, _ = _uc(sol, 1)
    assert u > 1.0  # flowing, so an entropy wave reaches the outlet
    fr = forced_response(sol.problem, sol.x, FREQS)
    f1, g1, h1 = fr.waves(1)[:, 0], fr.waves(1)[:, 1], fr.waves(1)[:, 2]
    assert np.allclose(g1, Rv * f1 + Rsv * h1, atol=1e-9)


# --------------------------------------------------------------------------
# 2. The wall element: mean flow blocked; acoustically a hard wall.
# --------------------------------------------------------------------------


def test_wall_blocks_mean_flow_dead_end():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(105000.0, 300.0))
    net.add(cat.junction())
    net.add(cat.pressure_outlet(101325.0, 300.0))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)  # feed
    net.connect(1, 2, 0.05)  # main
    net.connect(1, 3, 0.05)  # dead-end into the wall
    sol = net.solve()
    assert sol.converged
    main, dead = sol.edge(1), sol.edge(2)
    assert abs(dead["mdot"]) < 1e-9  # no mass crosses the wall
    assert dead["p"] == pytest.approx(main["p"], rel=1e-6)  # junction common static pressure
    assert dead["h_t"] == pytest.approx(main["h_t"], rel=1e-6)  # enthalpy-transparent donor


def test_wall_terminated_duct_is_hard_wall():
    # A wall closes the duct -> mean flow is blocked (M = 0, a quiescent duct); the
    # wall's default closure must reflect as a hard wall, R = +1.
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(LDUCT))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    u, c = _uc(sol)
    assert abs(u) < 1e-9  # quiescent
    fr = forced_response(sol.problem, sol.x, FREQS)
    assert np.allclose(fr.reflection_at(0), np.exp(-2j * OMEGAS * LDUCT / c), atol=1e-9)


# --------------------------------------------------------------------------
# 3. PerturbationBC unit tests (reflection map, impedance, tables, forcing).
# --------------------------------------------------------------------------


def test_bc_reflection_presets():
    rho, c = 1.2, 340.0
    assert PerturbationBC.inherit().reflection_coefficient(0.0, rho, c, 0.0) is None
    assert PerturbationBC.hard_wall().reflection_coefficient(0.0, rho, c, 0.0) == 1.0
    assert PerturbationBC.open_end().reflection_coefficient(0.0, rho, c, 0.0) == -1.0
    assert PerturbationBC.anechoic().reflection_coefficient(0.0, rho, c, 0.0) == 0.0
    for M in (0.0, 0.3, 0.7):
        R = PerturbationBC.mean_flow_open_end().reflection_coefficient(0.0, rho, c, M)
        assert R == pytest.approx(-(1.0 - M) / (1.0 + M))


def test_bc_impedance_to_reflection():
    rho, c = 1.2, 340.0
    zc = rho * c
    for Z in (2.0 * zc, zc, (1.5 + 0.3j) * zc):
        R = PerturbationBC.impedance(Z).reflection_coefficient(0.0, rho, c, 0.0)
        assert R == pytest.approx((Z - zc) / (Z + zc))
    # specific impedance, and the rigid / pressure-release limits
    assert PerturbationBC.impedance(2.0, specific=True).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1 / 3)
    assert PerturbationBC.impedance(1e12 * zc).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert PerturbationBC.impedance(0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(-1.0)


def test_bc_impedance_polar():
    # the UI closure: specific magnitude + phase (deg)
    rho, c = 1.2, 340.0
    assert PerturbationBC.impedance_polar(2.0, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(1 / 3)
    # magnitude 1, phase 0 -> matched (anechoic)
    assert PerturbationBC.impedance_polar(1.0, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(0.0)
    # large magnitude -> rigid wall; phase 90 deg (zeta = i) -> |R| = 1
    assert PerturbationBC.impedance_polar(1e9, 0.0).reflection_coefficient(0.0, rho, c, 0.0) == pytest.approx(
        1, abs=1e-6
    )
    assert abs(PerturbationBC.impedance_polar(1.0, 90.0).reflection_coefficient(0.0, rho, c, 0.0)) == pytest.approx(1.0)


def test_bc_table_interpolation_in_freq():
    f = np.array([0.0, 100.0, 200.0])  # Hz
    val = np.array([0.0 + 0j, 1.0 + 0j, 1.0 + 2j])
    bc = PerturbationBC.reflection((f, val))
    assert bc.reflection_coefficient(50.0, 1.0, 1.0, 0.0) == pytest.approx(0.5 + 0j)
    assert bc.reflection_coefficient(150.0, 1.0, 1.0, 0.0) == pytest.approx(1.0 + 1j)


def test_bc_forcing_and_entropy():
    # An explicit acoustic amplitude drives the acoustic row only.
    bc = PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"acoustic": 0.7 - 0.2j})
    assert bc.forcing(0.0) == pytest.approx(0.7 - 0.2j)
    assert bc.entropy_forcing(0.0) == 0.0
    # A marked-but-unspecified family drives a unit wave.
    bce = PerturbationBC.anechoic(driven=("entropy",))
    assert bce.forcing(0.0) == 0.0
    assert bce.entropy_forcing(0.0) == pytest.approx(1.0)
    # The entropy_in carrier and a driven entropy wave add on the entropy row.
    assert PerturbationBC.anechoic(entropy_in=0.3 + 0.1j).entropy_forcing(0.0) == pytest.approx(0.3 + 0.1j)
    both = PerturbationBC.anechoic(entropy_in=0.3 + 0.1j, driven=("entropy",), amplitudes={"entropy": 0.5})
    assert both.entropy_forcing(0.0) == pytest.approx(0.8 + 0.1j)


def test_bc_rejects_unknown_kind_and_bad_driven():
    with pytest.raises(ValueError):
        PerturbationBC(kind="nonsense")
    # driven families must be strings (a scalar name is resolved later, at stamp time, against
    # prob.scalar_names -- so an arbitrary string is accepted here and validated then).
    with pytest.raises(TypeError):
        PerturbationBC.anechoic(driven=(3,))
    # An amplitude for a family that is not driven is a mistake.
    with pytest.raises(ValueError):
        PerturbationBC.anechoic(driven=("acoustic",), amplitudes={"entropy": 1.0})


def test_driven_is_orthogonal_to_reflection():
    # The headline composability: a drive rides on top of ANY reflection.  mean_flow_open_end
    # keeps its convective-neutral R while injecting a unit incoming acoustic wave.
    rho, c, M = 1.2, 340.0, 0.3
    bc = PerturbationBC.mean_flow_open_end(driven=("acoustic",))
    assert bc.reflection_coefficient(0.0, rho, c, M) == pytest.approx(-(1.0 - M) / (1.0 + M))
    assert bc.forcing(0.0) == pytest.approx(1.0)
    # an explicit amplitude overrides the unit default; the reflection is untouched
    bc2 = PerturbationBC.reflection(0.4, driven=("acoustic",), amplitudes={"acoustic": 0.25 - 0.1j})
    assert bc2.reflection_coefficient(0.0, rho, c, M) == pytest.approx(0.4)
    assert bc2.forcing(0.0) == pytest.approx(0.25 - 0.1j)


def test_cannot_drive_entropy_at_an_outlet():
    # Entropy is an arriving (outgoing) wave at an outlet, so driving it there is rejected
    # at closure time (specify = (g,), arriving = (f, h)).
    bc = PerturbationBC.anechoic(driven=("entropy",))
    with pytest.raises(ValueError):
        bc.closure(0.0, 1.2, 340.0, 100.0, 0.3, None, specify=(1,), arriving=(0, 2))


# --------------------------------------------------------------------------
# 3b. Driving a reacting/passive scalar wave at an inflow terminal (decoupled seat).
# --------------------------------------------------------------------------


def _scalar_duct(inlet_bc, outlet_bc=None, *, L=0.7, mdot=2.0):
    """[mass-flow inlet] -- duct(L) -- [pressure outlet] on a 2-passive-scalar perfect gas."""
    cfg = perfect_gas_passive_scalars(2, names=["Z1", "Z2"])
    net = Network(cfg, p_ref=1.0e5, T_ref=300.0, mdot_ref=mdot)
    net.add(cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=inlet_bc))
    net.add(cat.duct(L))
    net.add(cat.pressure_outlet(1.0e5, perturbation_bc=outlet_bc or PerturbationBC.anechoic()))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    return sol


def test_driven_scalar_wave_seats_and_convects():
    # Seat a scalar wave Z1 at the inflow; it must appear on the inlet edge and convect to the
    # outlet at the mean speed u (phase e^{-i w L/u}), decoupled from sound, with the undriven
    # scalar Z2 staying zero.  No compact-nozzle closure here, so the (narrowed) compositional-noise
    # warning must NOT fire -- a driven scalar through inherited/anechoic terminals drops nothing.
    amp, L = 0.4 - 0.2j, 0.7
    sol = _scalar_duct(PerturbationBC.anechoic(driven=("Z1",), amplitudes={"Z1": amp}), L=L)
    u = float(sol.table()[ES_U, 0])
    assert u > 1.0  # genuinely flowing
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompositionalNoiseWarning)  # a spurious warning would fail here
        fr = forced_response(sol.problem, sol.x, FREQS, isentropic=False)
    iz = fr.wave_labels.index("Z1")
    assert np.allclose(fr.waves(0)[:, iz], amp, atol=1e-9)  # seated at the inlet edge
    assert np.allclose(fr.waves(1)[:, iz], amp * np.exp(-1j * OMEGAS * (L / u)), atol=1e-6)  # convected out
    assert np.allclose(fr.waves(0)[:, fr.wave_labels.index("Z2")], 0.0, atol=1e-12)  # undriven scalar


def test_compact_nozzle_closure_warns_in_reacting_flow():
    # The narrowed warning: a *hand-written* compact-nozzle closure on a scalar-carrying flow drops
    # the composition -> acoustic noise (R_xi), so forced_response warns once.  (Driving a scalar is
    # incidental -- the gap is the closure, which discards composition whether or not a scalar is
    # driven; here the mere presence of transported scalars + the compact closure triggers it.)
    sol = _scalar_duct(PerturbationBC.inherit(), PerturbationBC.choked_nozzle())
    with pytest.warns(CompositionalNoiseWarning, match="compact nozzle closure"):
        forced_response(sol.problem, sol.x, FREQS, isentropic=False)


def test_driving_scalar_rejected_at_outlet():
    # A scalar is arriving (outgoing) at the outlet, so seating one there is rejected on assembly.
    sol = _scalar_duct(PerturbationBC.inherit(), PerturbationBC.anechoic(driven=("Z1",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionalNoiseWarning)
        with pytest.raises(ValueError, match="genuine inflow"):
            forced_response(sol.problem, sol.x, FREQS, isentropic=False)


def test_driving_unknown_scalar_name_is_rejected():
    # A driven family that is neither acoustic/entropy nor a transported scalar errors at stamp
    # time (it cannot be checked at BC construction), listing what the network does transport.
    sol = _scalar_duct(PerturbationBC.anechoic(driven=("Zbogus",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionalNoiseWarning)
        with pytest.raises(ValueError, match="unknown scalar wave family"):
            forced_response(sol.problem, sol.x, FREQS, isentropic=False)


# --------------------------------------------------------------------------
# 4. UI/YAML loader: BC attributes parse into PerturbationBCs on the elements.
# --------------------------------------------------------------------------


def _branched_terminations_case():
    # reservoir --(feed)--> tee --(main)--> duct --> liner (impedance)
    #                          \--(branch)--> duct --> wall (rigid; stagnant)
    def node(nid, typ, **attrs):
        return {"id": nid, "type": typ, "attributes": attrs}

    def edge(eid, src, tgt, idx, area):
        return {
            "id": eid,
            "source": src,
            "target": tgt,
            "sourceHandle": f"{src}-port-0",
            "targetHandle": f"{tgt}-port-0",
            "attributes": {"index": idx, "area": area},
        }

    return {
        "version": "2.0.0",
        "model": {
            "id": "fns-flow-network",
            "globalAttributes": {
                "gasConstant": 287.0,
                "heatCapacityRatio": 1.4,
                "referencePressure": 101325.0,
                "referenceTemperature": 300.0,
                "referenceMassFlow": 5.0,
            },
            "nodes": [
                node(
                    "res",
                    "TotalPressureInlet",
                    index=0,
                    label="reservoir",
                    totalPressure=110000.0,
                    totalTemperature=300.0,
                    boundaryType="impedance",
                    impedanceMagnitude=1.0,
                    impedancePhase=0.0,
                ),
                node("tee", "JunctionStaticP", index=1, label="tee"),
                node("main", "Duct", index=2, label="main-duct", length=0.60),
                node(
                    "liner",
                    "PressureOutlet",
                    index=3,
                    label="liner",
                    pressure=101325.0,
                    backflowTotalTemperature=300.0,
                    boundaryType="impedance",
                    impedanceMagnitude=2.0,
                    impedancePhase=0.0,
                ),
                node("branch", "Duct", index=4, label="branch-duct", length=0.25),
                node("wend", "Wall", index=5, label="resonator-end", boundaryType="rigid"),
            ],
            "edges": [
                edge("e0", "res", "tee", 0, 0.020),
                edge("e1", "tee", "main", 1, 0.020),
                edge("e2", "main", "liner", 2, 0.020),
                edge("e3", "tee", "branch", 3, 0.010),
                edge("e4", "branch", "wend", 4, 0.010),
            ],
        },
    }


def test_loader_parses_boundary_conditions_and_runs(tmp_path):
    net = _load_case_dict(_branched_terminations_case(), tmp_path)
    kinds = {el.name: (None if el.perturbation_bc is None else el.perturbation_bc.kind) for el in net._elements}
    assert kinds["liner"] == "impedance"  # specific-impedance liner
    assert kinds["resonator-end"] == "hard_wall"  # rigid wall
    assert kinds["reservoir"] == "impedance"  # matched default (magnitude 1)
    assert kinds["tee"] is None  # interior element -> no BC
    sol = net.solve()
    assert sol.converged
    assert abs(sol.edge(3)["mdot"]) < 1e-9  # branch behind the wall: no mean flow
    fr = forced_response(sol.problem, sol.x, np.linspace(100.0, 2000.0, 4))
    assert fr.X.shape == (4, 3 * net.compile().n_edges)


def test_loader_default_inherit_keeps_old_cases():
    # boundary nodes with no acoustic fields default to inherit (None)
    net = load_case(os.path.join(_EXAMPLES, "converging_nozzle.yaml"))
    assert all(el.perturbation_bc is None for el in net._elements)


def _ui_case(outlet_attrs):
    return {
        "version": "2.0.0",
        "model": {
            "id": "fns-flow-network",
            "globalAttributes": {
                "gasConstant": 287.0,
                "heatCapacityRatio": 1.4,
                "referencePressure": 101325.0,
                "referenceTemperature": 300.0,
                "referenceMassFlow": 5.0,
            },
            "nodes": [
                {
                    "id": "in",
                    "type": "TotalPressureInlet",
                    "attributes": {"index": 0, "label": "in", "totalPressure": 104000.0, "totalTemperature": 300.0},
                },
                {"id": "d", "type": "Duct", "attributes": {"index": 1, "label": "d", "length": 0.5}},
                {
                    "id": "out",
                    "type": "PressureOutlet",
                    "attributes": dict({"index": 2, "label": "out", "pressure": 101325.0}, **outlet_attrs),
                },
            ],
            "edges": [
                {
                    "id": "e0",
                    "source": "in",
                    "target": "d",
                    "sourceHandle": "in-port-0",
                    "targetHandle": "d-port-0",
                    "attributes": {"index": 0, "area": 0.05},
                },
                {
                    "id": "e1",
                    "source": "d",
                    "target": "out",
                    "sourceHandle": "d-port-1",
                    "targetHandle": "out-port-0",
                    "attributes": {"index": 1, "area": 0.05},
                },
            ],
        },
    }


def _load_case_dict(case, tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case))
    return load_case(str(path))


def test_loader_rigid_maps_to_hard_wall(tmp_path):
    net = _load_case_dict(_ui_case({"boundaryType": "rigid"}), tmp_path)
    assert net._elements[2].perturbation_bc.kind == "hard_wall"


def test_loader_open_maps_to_open_end(tmp_path):
    # boundaryType 'open' -> ideal pressure-release end (p'=0, R=-1)
    net = _load_case_dict(_ui_case({"boundaryType": "open"}), tmp_path)
    bc = net._elements[2].perturbation_bc
    assert bc.kind == "open_end"
    assert bc.reflection_coefficient(0.0, 1.2, 340.0, 0.0) == pytest.approx(-1.0)


def test_loader_impedance_polar(tmp_path):
    # specific impedance magnitude 2, phase 0 -> zeta = 2 -> R = (2-1)/(2+1) = 1/3
    net = _load_case_dict(
        _ui_case({"boundaryType": "impedance", "impedanceMagnitude": 2.0, "impedancePhase": 0.0}),
        tmp_path,
    )
    bc = net._elements[2].perturbation_bc
    assert bc.kind == "impedance" and bc.specific
    assert bc.reflection_coefficient(0.0, 1.2, 340.0, 0.0) == pytest.approx(1.0 / 3.0)


def test_loader_rejects_unknown_boundary_type(tmp_path):
    with pytest.raises(ValueError):
        _load_case_dict(_ui_case({"boundaryType": "bogus"}), tmp_path)


def test_loader_no_acoustic_fields_is_inherit(tmp_path):
    net = _load_case_dict(_ui_case({}), tmp_path)
    assert net._elements[2].perturbation_bc is None


def test_loader_builds_wall_element(tmp_path):
    case = _ui_case({})
    case["model"]["nodes"][2] = {
        "id": "out",
        "type": "Wall",
        "attributes": {"index": 2, "label": "wall", "boundaryType": "rigid"},
    }
    net = _load_case_dict(case, tmp_path)
    wall = net._elements[2]
    assert wall.residual_id == WALL
    assert wall.perturbation_bc.kind == "hard_wall"
