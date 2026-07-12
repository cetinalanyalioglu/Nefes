"""Linear-stability eigenmodes: the nonlinear eigenproblem ``det A(omega) = 0``.

The verification is layered against closed form:

* The Beyn contour solver itself, on the 4x4 duct oracle (``DuctAcoustics``), for
  both the lossless (real) and lossy (complex) acoustic dispersion.
* The full network operator, reproducing the same analytic duct modes -- closed,
  open, lossy, and with mean flow -- so the assembled ``A(omega)`` and the oracle
  agree.
* Cross-driver consistency: a mode's frequency coincides with the resonance peak of
  the *forced* response (``forced_response``) on the same network -- the two
  analyses share one operator, so they must.

Sign convention: under the operator's ``e^{+i*omega*t}`` time dependence a passive
lossy resonator decays for ``Im(omega) > 0``, so the growth rate is ``-Im(omega)``
and a mode is unstable iff ``Im(omega) < 0``.  This is pinned directly: a lossy duct
must come out with ``Im(omega) > 0`` (decaying).
"""

import warnings

import numpy as np
import pytest

from nefes.assembly.recover import ES_C, ES_U
from nefes.elements import catalog as cat
from nefes.perturbation import (
    DuctAcoustics,
    EigenmodeResult,
    EigenmodeWarning,
    PerturbationBC,
    eigenmodes,
    forced_response,
)
from nefes.perturbation.stability.contour import beyn, circle_contour, ellipse_contour, lu_logdet_phase, winding_count
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

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
    from nefes.perturbation.operator.operator import _assemble_reference, assemble_acoustic, build_acoustic_blocks

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


# --------------------------------------------------------------------------
# 7. The rank of the moment matrix comes from the argument principle, not from its
#    own singular values, and the residual is measured on the equilibrated operator.
# --------------------------------------------------------------------------


def test_beyn_moment_rank_is_ambiguous_on_an_empty_contour():
    # On a contour enclosing an eigenvalue the moment's singular values show a gap of many
    # decades at the true rank.  On an eigenvalue-free contour A_0 is analytically zero, its
    # singular values are pure quadrature error, and no threshold on them can see rank 0:
    # the relative test svd_tol * s[0] reports the full probe width.  This is why the driver
    # supplies the rank instead of letting beyn infer it.
    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT

    def solve(z, B):
        return np.linalg.solve(da.system(z, 1.0, 1.0), B)

    populated = circle_contour(w1 + 0j, 0.4 * w1, 128)
    _, _, info = beyn(solve, 4, populated, n_probe=4)
    s = info["svals"]
    assert info["rank"] == 1
    assert s[0] / s[1] > 1e6  # a clean gap at the true rank

    empty = circle_contour(0.4 * w1 + 0j, 0.2 * w1, 128)
    _, _, info = beyn(solve, 4, empty, n_probe=4)
    s = info["svals"]
    assert info["rank"] == 4  # the relative threshold sees full rank where the true rank is 0
    assert s[0] / s[-1] < 1e3  # no gap anywhere: the singular values are all quadrature noise
    assert s[0] < 1e-9  # ... and all of them are at the noise floor


def test_beyn_rank_override_pins_the_mode_count():
    # Handing beyn the argument-principle count makes it exact on both contours above.
    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT

    def solve(z, B):
        return np.linalg.solve(da.system(z, 1.0, 1.0), B)

    def det_phase(z):
        return float(np.angle(da.det(z, 1.0, 1.0)))

    empty = circle_contour(0.4 * w1 + 0j, 0.2 * w1, 128)
    k, _ = winding_count(det_phase, empty)
    lam, vecs, info = beyn(solve, 4, empty, rank=k)
    assert k == 0 and lam.size == 0 and vecs.shape == (4, 0) and info["rank"] == 0

    populated = ellipse_contour(1.5 * w1 + 0j, 0.9 * w1, 0.3 * w1, 256)
    k, _ = winding_count(det_phase, populated)
    lam, _, info = beyn(solve, 4, populated, rank=k)
    assert k == 2 and info["rank"] == 2
    _match(lam, [w1, 2.0 * w1], rtol=1e-5)  # the two modes, and nothing else


def test_equilibrated_residual_separates_a_mode_from_an_arbitrary_point():
    # A network operator mixes rows in incompatible units, so cond(A) is huge at every omega
    # and ||A x|| / max|A| is tiny even where there is no mode.  Equilibrating first restores
    # the separation.  The diagonal below stands in for that unit disparity.
    import scipy.sparse as sp

    from nefes.perturbation.stability.eigenmodes import _ResidualScale

    c = 340.0
    da = DuctAcoustics(c, LDUCT, 0.0)
    w1 = np.pi * c / LDUCT
    D = np.diag([1.0, 1e6, 1e-3, 1e8])  # rows in wildly different units

    def A_of(z):
        return sp.csr_matrix(D @ da.system(complex(z), 1.0, 1.0))

    def raw_residual(z, v):
        A = A_of(z)
        return float(np.linalg.norm(A @ v)) / float(np.abs(A.data).max())

    def min_singular_vector(z):
        return np.linalg.svd(A_of(z).toarray())[2][-1].conj()

    residual = _ResidualScale(A_of, w1)

    # away from any mode, the raw residual of the near-null vector passes any sane cutoff...
    for z in (0.4 * w1, 0.7 * w1):
        v = min_singular_vector(z)
        assert raw_residual(z, v) < 1e-9  # indistinguishable from a converged mode
        assert residual(A_of, z, v) > 1e-6  # ... but the equilibrated residual rejects it

    # on the mode, both agree that the residual is at round-off
    v = min_singular_vector(w1)
    assert residual(A_of, w1, v) < 1e-12


def test_subcontours_cover_the_counted_region():
    # The certificate counts eigenvalues inside `bound`, and modes are accepted inside `bound`;
    # the tiles must therefore cover it.  Side-by-side ellipses of the region's own half-height
    # pinch at every seam, leaving a mode of large growth rate there counted but unreachable.
    from nefes.perturbation.stability.eigenmodes import _band_subcontours, _tile, build_operator

    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    _, blocks, _, _ = build_operator(sol.problem, sol.x)
    band = (50.0, 2000.0)
    subs, bound, _, geom = _band_subcontours(band, None, 64, blocks, None)

    # the region is exactly the band the caller asked for -- no hidden margin, so "counted
    # inside" and "kept inside" are the same set
    c_re, rx = np.pi * (band[0] + band[1]), np.pi * (band[1] - band[0])
    assert bound.center.real == pytest.approx(c_re) and bound.rx == pytest.approx(rx)

    assert len(subs) > 1, "the band must tile into several sub-contours for this to bite"
    for t in np.linspace(0.0, 2.0 * np.pi, 181):
        for r in (0.35, 0.7, 0.95, 0.999):  # rings of the region ellipse, out to its rim
            z = bound.center + r * (bound.rx * np.cos(t) + 1j * bound.ry * np.sin(t))
            assert any(s.inside(z) for s in subs), f"z={z} lies in no sub-contour"

    # ... and re-tiling finer (the adaptive refinement path) keeps covering it
    c_re, c_im, rx, ry = geom
    for s in _tile(c_re, c_im, rx, ry, 4 * len(subs), 64):
        assert s.ry > ry, "a tile must be taller than the region it helps cover"


def test_eigenmodes_finds_nothing_between_harmonics():
    # A band that encloses no mode must come back empty and certified so, not populated with
    # the quadrature noise of an eigenvalue-free contour.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)  # modes sit at f1, 2 f1, 3 f1, ...
    with warnings.catch_warnings():
        warnings.simplefilter("error", EigenmodeWarning)  # and without complaining
        res = eigenmodes(sol.problem, sol.x, (1.2 * f1, 1.8 * f1))
    assert res.n_modes == 0 and res.expected == 0 and res.certified


def _broken_loop_net():
    """A duct with one reflecting end and one anechoic end: no round trip, so no free mode.

    The small area leaves the assembled operator with ``cond(A) ~ 1e9``, the regime a real
    network lives in, where ``||A x|| / max|A|`` is ``~1e-9`` at *every* frequency and cannot
    tell a mode from an arbitrary point.
    """
    return _duct_net(
        PerturbationBC.reflection(0.2),
        cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.anechoic()),
        area=1e-4,
        mdot_ref=1e-3,
    )[1]


def test_no_modes_survive_an_eigenvalue_free_region():
    # The spectrum is empty, and the driver must say so at every band, on an operator whose
    # conditioning makes the raw residual useless as a filter.  Inferring the rank from the
    # moment's singular values instead invents a handful of modes per band, each with a
    # convincingly small raw residual, and each at a different frequency for a different band.
    sol = _broken_loop_net()
    for band in [(50.0, 900.0), (50.0, 1200.0), (50.0, 1800.0), (50.0, 2400.0)]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", EigenmodeWarning)
            res = eigenmodes(sol.problem, sol.x, band, isentropic=True)
        assert res.expected == 0, f"the region {band} Hz should enclose no eigenvalue"
        assert res.n_modes == 0, f"invented {res.n_modes} mode(s) at {np.sort(res.freqs)} Hz in {band}"
        assert res.certified


def test_eigenmodes_are_insensitive_to_the_band_edge():
    # The tiling is derived from the band, so moving its upper edge re-cuts the sub-contours and
    # changes which of them enclose a mode.  What is found below the edge must not care.
    _, sol = _duct_net(PerturbationBC.hard_wall(), cat.wall())
    u, c = _uc(sol)
    f1 = c / (2.0 * LDUCT)
    common = None
    for f_hi in (2.2, 2.4, 2.6, 2.8):  # every edge lies between the 2nd and 3rd harmonic
        res = eigenmodes(sol.problem, sol.x, (0.5 * f1, f_hi * f1))
        assert res.certified, f"uncertified at f_hi = {f_hi} f1"
        found = np.sort(res.freqs)
        assert found.size == 2
        if common is None:
            common = found
        assert np.allclose(found, common, rtol=1e-6)
    _match(common, [f1, 2.0 * f1], rtol=1e-4)


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
    from nefes.perturbation.operator.operator import _assemble_reference, assemble_acoustic, build_acoustic_blocks

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


def test_passive_temperature_jump_keeps_the_zero_mach_velocity_continuity():
    # A closed-open tube split by a compact steady heat source.  In the zero-Mach limit the
    # jump carries continuous p' and u' (the entropy spot the flame sheds holds the density
    # balance), so the resonances solve
    #     xi cos(w tau_1) cos(w tau_2) + sin(w tau_1) sin(w tau_2) = 0,
    # with xi = rho_1 c_1 / (rho_2 c_2).  The isentropic reduction must reproduce these:
    # pinning the entropy on the flame's downstream edge instead would silently turn the
    # jump into mass-flux continuity (u' jumping by rho_1/rho_2) and detune every mode.
    theta = 3.0
    t_cold = 300.0
    l_cold, l_hot, area = 0.3, 0.5, 2.0e-3
    c_cold = np.sqrt(GAMMA * R_AIR * t_cold)
    c_hot = c_cold * np.sqrt(1.0 + theta)
    mdot = 1.2 * 1.0e-3 * c_cold * area  # cold-side Mach about 1e-3
    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    qdot = mdot * cp * theta * t_cold

    net = Network(CFG, p_ref=101325.0, T_ref=t_cold, mdot_ref=mdot)
    net.add(cat.mass_flow_inlet(mdot, t_cold, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(l_cold))
    net.add(cat.heat_release_flame(qdot))
    net.add(cat.duct(l_hot))
    net.add(cat.pressure_outlet(101325.0, perturbation_bc=PerturbationBC.open_end()))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        net.connect(a, b, area)
    sol = net.solve()
    assert sol.converged

    # first two analytic roots by bisection on the pole-free numerator
    xi = np.sqrt(1.0 + theta)
    tau1, tau2 = l_cold / c_cold, l_hot / c_hot

    def numerator(f):
        w = 2.0 * np.pi * f
        return xi * np.cos(w * tau1) * np.cos(w * tau2) + np.sin(w * tau1) * np.sin(w * tau2)

    roots = []
    grid = np.arange(10.0, 900.0, 1.0)
    vals = numerator(grid)
    for i in np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][:2]:
        lo, hi = grid[i], grid[i + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sign(numerator(mid)) == np.sign(numerator(lo)):
                lo = mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))

    res = eigenmodes(sol.problem, sol.x, (10.0, 900.0), isentropic=True)
    _match(res.freqs, roots, rtol=3e-3)
    # the ideal modes are neutral; only the Mach-order terms may damp them
    assert np.all(np.abs(res.growth_rates) < 2.0 * np.pi * 2.0)


def test_low_mach_choked_nozzle_warns_not_crashes():
    """A very low-Mach choked-nozzle chamber ill-conditions the argument-principle count.

    The entropy characteristic (convected at u = M*c -> 0) degenerates the operator's determinant
    at very low mean-flow Mach, so a sub-contour count comes out negative.  The solver must not
    crash (it used to raise ``rank must be non-negative``) and must not falsely certify a "no modes"
    result: it warns and leaves ``certified`` False.  The modes are real -- ``isentropic=True``
    recovers them -- so the honest outcome is uncertified, not a silent empty answer.
    """
    import warnings

    import numpy as np

    import nefes
    from nefes.elements import catalog as cat

    sol = nefes.Network(
        nodes=[
            cat.mass_flow_inlet(2.0, 300.0, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            cat.duct(0.5),
            cat.choked_nozzle_outlet(1e-3, name="throat"),
        ],
        edges=[(0, 1, 0.02), (1, 2, 0.02)],
    ).solve()
    assert sol.edge(0)["M"] < 0.05  # genuinely low Mach

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sol.eigenmodes(freq_band=(50.0, 2000.0))  # must not raise
    assert not res.certified  # never a false "no modes, certified"
    assert any("ill-conditioned" in str(w.message) for w in caught)

    # The acoustic modes are real: the isentropic path recovers them, certified.
    iso = sol.eigenmodes(freq_band=(50.0, 2000.0), isentropic=True)
    assert iso.n_modes >= 4 and iso.certified
    assert np.all(np.array(iso.freqs) > 300.0)
