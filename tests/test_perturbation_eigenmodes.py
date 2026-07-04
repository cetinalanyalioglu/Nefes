"""Linear-stability eigenmodes: the nonlinear eigenproblem ``det A(omega) = 0`` (theory.md s12.7 (ii)).

The verification is layered against closed form:

* The Beyn contour solver itself, on the 4x4 duct oracle (``DuctAcoustics``), for
  both the lossless (real) and lossy (complex) acoustic dispersion.
* The full network operator, reproducing the same analytic duct modes -- closed,
  open, lossy, and with mean flow -- so the assembled ``A(omega)`` and the oracle
  agree.
* Cross-driver consistency: a mode's frequency coincides with the resonance peak of
  the *forced* response (``forced_response``) on the same network -- the two
  s12.7 analyses share one operator, so they must.

Sign convention: under the operator's ``e^{+i*omega*t}`` time dependence a passive
lossy resonator decays for ``Im(omega) > 0``, so the growth rate is ``-Im(omega)``
and a mode is unstable iff ``Im(omega) < 0``.  This is pinned directly: a lossy duct
must come out with ``Im(omega) > 0`` (decaying).
"""

import warnings

import numpy as np
import pytest

from nefes.shell import Network
from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.assembly.recover import ES_U, ES_C
from nefes.perturbation import (
    PerturbationBC,
    eigenmodes,
    EigenmodeResult,
    EigenmodeWarning,
    forced_response,
    DuctAcoustics,
)
from nefes.perturbation.stability.contour import ellipse_contour, circle_contour, beyn, winding_count, lu_logdet_phase

R_AIR, GAMMA = 287.0, 1.4
CFG = perfect_gas(R_AIR, GAMMA)
LDUCT = 0.5


# --------------------------------------------------------------------------
# Analytic duct dispersion: e^{-i w (tau_+ + tau_-)} = R0 R1.
# --------------------------------------------------------------------------


def _analytic_duct_modes(u, c, L, R0, R1, krange):
    """Complex acoustic eigenfrequencies of a uniform duct with end reflections.

    Standard reflection convention (reflected = R x incident at each end), so the
    round-trip loop gain is ``R0 R1 e^{-i w (tau_+ + tau_-)} = 1``, i.e. ``e^{-i w T}
    = 1/(R0 R1)`` with ``T = tau_+ + tau_-``.  Hence ``omega_k = (-i Ln(R0 R1) - 2 pi
    k) / T``; a passive end (``|R| < 1``) gives ``Im(omega) > 0`` (decaying).
    """
    T = L / (u + c) + L / (c - u)
    rr = complex(R0 * R1)
    return np.array([(-1j * np.log(rr) - 2.0 * np.pi * k) / T for k in krange])


def _match(found, expected, rtol):
    """Every ``expected`` value has a ``found`` value within ``rtol`` (relative)."""
    found = np.asarray(found)
    for w in expected:
        assert np.any(np.abs(found - w) <= rtol * abs(w)), f"missing mode near {w}; found {found}"


# --------------------------------------------------------------------------
# 1. The Beyn contour solver on the 4x4 duct oracle.
# --------------------------------------------------------------------------


def test_beyn_oracle_closed_closed_real_modes():
    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT  # first closed-closed mode (R0 = R1 = 1)
    cont = circle_contour(w1 + 0j, 0.4 * w1, 96)
    lam, vecs, info = beyn(lambda z, B: np.linalg.solve(da.system(z, 1.0, 1.0), B), 4, cont, n_probe=4)
    inside = np.array([z for z in lam if cont.inside(z)])
    _match(inside, [w1], rtol=1e-6)
    # a lossless closed resonator is marginally stable: real eigenvalue
    z = inside[np.argmin(np.abs(inside - w1))]
    assert abs(z.imag) < 1e-6 * w1


def test_beyn_oracle_lossy_complex_modes_and_sign():
    c = 340.0
    R0, R1 = 1.0, 0.5  # one lossy end
    da = DuctAcoustics(c, LDUCT, 0.0)
    expected = _analytic_duct_modes(0.0, c, LDUCT, R0, R1, [-1])[0]
    cont = ellipse_contour(expected.real + 0j, 0.4 * expected.real, 0.5 * abs(expected.real), 128)
    lam, _, _ = beyn(lambda z, B: np.linalg.solve(da.system(z, R0, R1), B), 4, cont, n_probe=4)
    inside = np.array([z for z in lam if cont.inside(z)])
    _match(inside, [expected], rtol=1e-5)
    z = inside[np.argmin(np.abs(inside - expected))]
    assert z.imag > 0.0  # lossy -> decaying -> Im(omega) > 0 -> growth rate -Im(omega) < 0


def test_beyn_finds_multiple_modes_and_grows_probe():
    # Two modes inside one contour, started with too few probes: the rank saturates
    # and the solver must grow the probe block to resolve both (Beyn needs l > k).
    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT
    expected = [w1, 2.0 * w1]  # modes n = 1, 2
    center = 1.5 * w1
    cont = ellipse_contour(center + 0j, 0.9 * w1, 0.3 * w1, 160)  # encloses w1, 2 w1; excludes 3 w1
    lam, _, info = beyn(lambda z, B: np.linalg.solve(da.system(z, 1.0, 1.0), B), 4, cont, n_probe=1)
    inside = np.array([z for z in lam if cont.inside(z)])
    _match(inside, expected, rtol=1e-5)
    assert info["rank"] >= 2 and info["n_probe"] >= 2  # probe block grew from 1


# --------------------------------------------------------------------------
# Network builders.
# --------------------------------------------------------------------------


def _duct_net(inlet_bc, outlet_elem, *, pt_in=101325.0, p_out=101325.0, L=LDUCT, area=0.05, mdot_ref=5.0):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=mdot_ref)
    net.add(cat.total_pressure_inlet(pt_in, 300.0, perturbation_bc=inlet_bc))
    net.add(cat.duct(L))
    net.add(outlet_elem)
    net.connect(0, 1, area)
    net.connect(1, 2, area)
    sol = net.solve()
    assert sol.converged, "mean-flow solve did not converge"
    return net, sol


def _uc(sol, e=0):
    est = sol.table()
    return float(est[ES_U, e]), float(est[ES_C, e])


# --------------------------------------------------------------------------
# 2. Full network reproduces the analytic duct modes.
# --------------------------------------------------------------------------


def test_network_closed_closed_quiescent():
    # hard-wall acoustic inlet + wall outlet -> R0 = R1 = 1, no mean flow.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    assert abs(u) < 1e-9  # quiescent
    f1 = c / (2.0 * LDUCT)  # omega_1 = pi c / L
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 2.5 * f1))
    assert isinstance(res, EigenmodeResult)
    _match(res.freqs, [f1, 2.0 * f1], rtol=1e-4)
    i = int(np.argmin(np.abs(res.freqs - f1)))
    assert abs(res.growth_rates[i]) < 1e-3 * (2 * np.pi * f1)  # lossless -> marginal
    assert res.residuals[i] < 1e-8


def test_network_resolves_multiple_modes_wide_band():
    # a wide band holding several harmonics: the driver must tile it into sub-contours
    # (a single contour over symmetric modes is rank-deficient and misses them).
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 3.5 * f1))
    _match(res.freqs, [f1, 2.0 * f1, 3.0 * f1], rtol=1e-4)  # all three harmonics resolved
    assert np.all(res.residuals < 1e-8)


def test_network_open_closed_quarter_wave():
    # ideal open inlet (R = -1) + wall (R = +1) -> quarter-wave: f = (2k+1) c / (4L).
    _, sol = _duct_net(PerturbationBC.open_end(), cat.wall())
    u, c = _uc(sol)
    assert abs(u) < 1e-9
    f_qw = c / (4.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f_qw, 4.0 * f_qw))
    _match(res.freqs, [f_qw, 3.0 * f_qw], rtol=1e-4)


def test_network_matches_oracle_lossy_quiescent():
    # hard-wall inlet + reflecting outlet at equal pressure -> (near-)quiescent, R0 = 1, R1 = 0.6.
    R1 = 0.6
    _, sol = _duct_net(
        PerturbationBC.hard_wall(),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(R1)),
    )
    u, c = _uc(sol)
    assert abs(u) / c < 1e-5  # essentially quiescent
    expected = _analytic_duct_modes(u, c, LDUCT, 1.0, R1, [-1, -2])
    fband = (0.3 * abs(expected[0].real) / (2 * np.pi), 1.3 * abs(expected[1].real) / (2 * np.pi))
    res = eigenmodes(sol.problem, sol.x, fband)
    # network operator modes match the analytic dispersion (standard reflection convention)
    _match(res.omega, expected, rtol=1e-4)
    assert np.all(res.growth_rates < 0.0)  # passive losses -> every mode decays


def test_network_mean_flow_shifts_modes():
    # reflecting ends with a driving pressure ratio -> subsonic mean flow; modes shift.
    R = 0.7
    _, sol = _duct_net(
        PerturbationBC.reflection(R),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(R)),
        pt_in=140000.0,
    )
    u, c = _uc(sol)
    assert u > 1.0 and u / c < 0.95  # genuinely flowing, subsonic
    expected = _analytic_duct_modes(u, c, LDUCT, R, R, [-1, -2])
    fband = (0.3 * abs(expected[0].real) / (2 * np.pi), 1.3 * abs(expected[1].real) / (2 * np.pi))
    res = eigenmodes(sol.problem, sol.x, fband)
    _match(res.omega, expected, rtol=2e-4)
    # convective shift: the fundamental sits at f_1 = c/(2L) (1 - M^2), below the quiescent c/(2L)
    M = u / c
    f1_found = res.freqs[np.argmin(np.abs(res.omega - expected[0]))]
    assert f1_found == pytest.approx(c / (2.0 * LDUCT) * (1.0 - M**2), rel=2e-3)
    assert f1_found < c / (2.0 * LDUCT)


# --------------------------------------------------------------------------
# 3. Cross-driver consistency: a mode sits at the forced-response resonance.
# --------------------------------------------------------------------------


def test_eigenmode_coincides_with_forced_resonance():
    # high-Q cavity: reflective, driven inlet + reflective outlet, quiescent.
    R = 0.9
    inlet = PerturbationBC.reflection(R, driven=("acoustic",))
    _, sol = _duct_net(inlet, cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(R)))
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.6 * f1, 1.4 * f1))
    i = int(np.argmin(np.abs(res.freqs - f1)))
    f_mode = res.freqs[i]
    assert res.growth_rates[i] < 0.0  # R<1 -> damped

    # sweep the forced response across the mode and find the resonance peak
    freqs = np.linspace(0.85 * f_mode, 1.15 * f_mode, 601)
    fr = forced_response(sol.problem, sol.x, freqs)
    amp = np.linalg.norm(fr.X, axis=1)  # response magnitude per frequency
    f_peak = freqs[int(np.argmax(amp))]
    # the forced resonance lands on the eigenmode frequency (within the sweep resolution)
    assert abs(f_peak - f_mode) < 0.01 * f_mode


# --------------------------------------------------------------------------
# 4. Result object + plotting surface.
# --------------------------------------------------------------------------


def test_eigenmode_result_api_and_plotting():
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 1.5 * f1))
    assert len(res) == res.n_modes >= 1
    # summary fields
    rows = res.summary()
    assert set(rows[0]) == {"freq_hz", "growth_rate", "damping_ratio", "unstable", "residual"}
    # mode shape projects to every edge in the requested basis
    shape = res.mode_shape(0, basis="primitive")
    assert shape.shape == (sol.problem.n_edges, 3)
    waves = res.mode_waves(0, 0)
    assert waves.shape == (3,)
    # plotting returns figures (no rendering)
    fig = res.plot_spectrum()
    assert fig.data  # has at least one trace
    fig2 = res.plot_mode(0)
    assert fig2.data


def test_eigenmode_result_repr_and_html():
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 2.5 * f1))
    assert res.n_modes >= 1

    text = repr(res)
    # header reports the mode count, the unstable tally, the certification, and the search band
    assert f"{res.n_modes} mode" in text
    assert f"{int(np.count_nonzero(res.unstable))} unstable" in text
    assert ("certified" in text) or ("uncertified" in text) or ("incomplete" in text)
    assert "Hz" in text and "growth" in text
    # every mode's frequency appears in the table (sorted, original index preserved)
    for f in res.freqs:
        assert f"{f:.3f}" in text

    html = res._repr_html_()
    assert html.startswith("<div") and "<table" in html and "</table>" in html
    assert "EigenmodeResult" in html
    assert html.count("<tr") == res.n_modes + 1  # one header row + one per mode

    # empty result is handled (no table, just the header)
    empty = EigenmodeResult(
        omega=np.empty(0, complex),
        modes=np.empty((0, res.modes.shape[1]), complex),
        residuals=np.empty(0),
        L=res.L,
        est=res.est,
        K=res.K,
        n_solve=res.n_solve,
        n_edges=res.n_edges,
        contour=res.contour,
        expected=0,
    )
    assert "0 modes" in repr(empty)
    assert "<table" not in empty._repr_html_()


# --------------------------------------------------------------------------
# 5. Fixed-pattern assembly fast path == reference assembly.
# --------------------------------------------------------------------------


def test_assembly_fast_path_matches_reference():
    # the cached fixed-pattern A(omega) must equal the LIL reference to round-off, across
    # a flowing duct (entropy phase live) and a quiescent one (entropy decoupled), with and
    # without terminal closures, at real and complex omega.
    from nefes.perturbation.operator.operator import build_acoustic_blocks, assemble_acoustic, _assemble_reference

    nets = [
        # flowing: reflecting ends, driven -> mean flow, so the entropy phase is omega-dependent
        _duct_net(
            PerturbationBC.reflection(0.7),
            cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.7)),
            pt_in=140000.0,
        )[1],
        # quiescent: hard wall + wall -> entropy stationary (P0 = 1)
        _duct_net(PerturbationBC.hard_wall(), cat.wall())[1],
    ]
    omegas = (0.0, 137.0, 2000.0 + 0j, 850.0 - 30.0j, 1500.0 + 12.0j)
    for sol in nets:
        blocks = build_acoustic_blocks(sol.problem, sol.x)
        for wb in (True, False):
            for omega in omegas:
                ref = _assemble_reference(omega, blocks, wb).tocsc()
                fast = assemble_acoustic(omega, blocks, wb)
                scale = max(abs(ref).max() if ref.nnz else 1.0, 1.0)
                assert abs((ref - fast)).max() <= 1e-9 * scale, f"mismatch wb={wb} omega={omega}"


# --------------------------------------------------------------------------
# 6. Completeness certificate: the argument principle counts the modes in a region.
# --------------------------------------------------------------------------


def test_lu_logdet_phase_matches_dense_determinant():
    # arg(det A) from a SuperLU factorization (phases + permutation parity) must equal
    # the dense determinant's phase, modulo 2 pi -- the basis of the winding count.
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    rng = np.random.default_rng(3)
    A = rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))
    phase = lu_logdet_phase(spla.splu(sp.csc_matrix(A)))
    expected = np.angle(np.linalg.det(A))
    d = (phase - expected + np.pi) % (2.0 * np.pi) - np.pi  # difference wrapped into (-pi, pi]
    assert abs(d) < 1e-9


def test_winding_count_oracle_counts_modes():
    # the argument principle on the 4x4 duct oracle: count = number of modes enclosed.
    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT

    def det_phase(z):
        return float(np.angle(da.det(z, 1.0, 1.0)))

    n1, info1 = winding_count(det_phase, circle_contour(w1 + 0j, 0.4 * w1, 256))
    assert n1 == 1 and info1["round_error"] < 1e-6  # one mode (w1)
    n2, _ = winding_count(det_phase, ellipse_contour(1.5 * w1 + 0j, 0.9 * w1, 0.3 * w1, 256))
    assert n2 == 2  # two modes (w1, 2 w1); excludes 3 w1
    n0, _ = winding_count(det_phase, circle_contour(0.4 * w1 + 0j, 0.2 * w1, 128))
    assert n0 == 0  # empty region below the first mode


def test_eigenmodes_certified_count_matches():
    # the driver certifies completeness: argument-principle count == modes resolved.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res = eigenmodes(sol.problem, sol.x, (0.5 * f1, 3.5 * f1))
    assert res.expected == res.n_modes == 3
    assert res.certified


def test_eigenmodes_certify_toggle():
    # certify=False leaves the count unset and certified False; certify=True fills it in.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    res_off = eigenmodes(sol.problem, sol.x, (0.5 * f1, 2.5 * f1), certify=False)
    assert res_off.expected is None and res_off.certified is False
    res_on = eigenmodes(sol.problem, sol.x, (0.5 * f1, 2.5 * f1), certify=True)
    assert res_on.expected == res_on.n_modes == 2 and res_on.certified


def test_certification_recovers_with_explicit_contour_and_tiny_probe():
    # a single wide contour over three symmetric modes with a deliberately tiny probe:
    # the completeness certificate (3) drives adaptive re-tiling until all three appear.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    w1 = 2.0 * np.pi * f1
    cont = ellipse_contour(2.0 * w1 + 0j, 1.3 * w1, 0.25 * w1, 192)  # encloses w1, 2 w1, 3 w1
    res = eigenmodes(sol.problem, sol.x, contour=cont, n_probe=1)
    assert res.expected == 3
    assert res.n_modes == 3 and res.certified
    _match(res.freqs, [f1, 2.0 * f1, 3.0 * f1], rtol=1e-4)


# --------------------------------------------------------------------------
# 7. Isentropic option (rho' = p'/c^2): the entropy wave is pinned to zero.
# --------------------------------------------------------------------------


def test_isentropic_fast_path_matches_reference():
    # the fixed-pattern fast path must equal the reference assembly with isentropic on too.
    from nefes.perturbation.operator.operator import build_acoustic_blocks, assemble_acoustic, _assemble_reference

    _, sol = _duct_net(
        PerturbationBC.reflection(0.7),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.7)),
        pt_in=140000.0,
    )
    blocks = build_acoustic_blocks(sol.problem, sol.x, isentropic=True)
    for wb in (True, False):
        for omega in (0.0, 137.0, 2000.0 + 0j, 850.0 - 30.0j):
            ref = _assemble_reference(omega, blocks, wb).tocsc()
            fast = assemble_acoustic(omega, blocks, wb)
            scale = max(abs(ref).max() if ref.nnz else 1.0, 1.0)
            assert abs((ref - fast)).max() <= 1e-9 * scale, f"mismatch wb={wb} omega={omega}"


def test_isentropic_preserves_acoustic_modes():
    # a flowing duct: isentropic leaves the acoustic spectrum unchanged (still
    # f1 = c/2L (1 - M^2)) -- pinning the entropy wave does not touch the acoustic waves.
    _, sol = _duct_net(
        PerturbationBC.reflection(0.7),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.7)),
        pt_in=140000.0,
    )
    u, c = _uc(sol)
    M = u / c
    f0 = c / (2 * LDUCT)
    full = eigenmodes(sol.problem, sol.x, (0.4 * f0, 1.3 * f0))
    isen = eigenmodes(sol.problem, sol.x, (0.4 * f0, 1.3 * f0), isentropic=True)
    assert isen.certified
    fa = f0 * (1 - M**2)  # convective-shifted fundamental
    assert isen.freqs[np.argmin(np.abs(isen.freqs - fa))] == pytest.approx(fa, rel=2e-3)
    # every isentropic mode is one of the full-model acoustic modes
    for f in isen.freqs:
        assert np.any(np.abs(full.freqs - f) < 1e-4 * max(f, 1.0))


def test_isentropic_zeroes_entropy_and_removes_convective_modes():
    # a sudden area change with mean flow generates entropy/convective modes; isentropic
    # drops them (a strict subset remains) and every isentropic mode carries zero entropy.
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=2.0)
    A1, A2 = 0.01, 0.05
    net.add(cat.total_pressure_inlet(112000.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(0.3))
    net.add(cat.sudden_area_change(name="exp"))
    net.add(cat.duct(0.4))
    net.add(cat.sudden_area_change(name="con"))
    net.add(cat.duct(0.3))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.5)))
    for a, b, ar in [(0, 1, A1), (1, 2, A1), (2, 3, A2), (3, 4, A2), (4, 5, A1), (5, 6, A1)]:
        net.connect(a, b, ar)
    sol = net.solve()
    assert sol.converged

    band = (50.0, 700.0)
    with warnings.catch_warnings():
        # the full (entropy-laden) spectrum is dense and may graze the band edge -- that
        # clutter is exactly what isentropic removes; only its mode count is needed here.
        warnings.simplefilter("ignore", EigenmodeWarning)
        full = eigenmodes(sol.problem, sol.x, band)
    isen = eigenmodes(sol.problem, sol.x, band, isentropic=True)
    assert isen.certified
    assert isen.n_modes < full.n_modes  # entropy / convective modes removed
    # every isentropic mode has a vanishing entropy characteristic (h) on every edge
    for i in range(isen.n_modes):
        shape = isen.mode_shape(i, basis="char")
        assert np.max(np.abs(shape[:, 2])) < 1e-10 * np.max(np.abs(shape[:, :2]))


def test_eigenmodes_warns_without_frequency_dependence():
    # no duct -> A(omega) has no frequency dependence; the search is ill-posed.
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(104000.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.reflection(0.5)))
    net.connect(0, 1, 0.05)
    sol = net.solve()
    assert sol.converged
    with pytest.warns(EigenmodeWarning):
        eigenmodes(sol.problem, sol.x, (100.0, 500.0))
