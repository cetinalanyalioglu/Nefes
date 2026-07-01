"""Analytic continuation of tabulated frequency-response data (TODO "to implement").

A measured / tabulated transfer function or reflection coefficient lives on a real
frequency grid; the stability eigenproblem searches the *complex* plane.  ``rational_fit``
(AAA barycentric rational, :mod:`nefes.elements.continuation`) bridges the two: the fit
reproduces the data on the grid yet continues off the real axis, so the same object drives
both the real-axis Nyquist sweep and the contour eigensolver.

These tests anchor the framework four ways:

1. **Continuation fidelity** -- a fit of a *known* analytic response (n-tau, low-pass,
   a rational reflection) reproduces it off the real axis, not just on the grid.
2. **Protocol** -- the fit is analytic and drops into a flame FTF, a fuel-modulation
   mass response, and a boundary reflection coefficient (the tabulated-data surfaces).
3. **End-to-end stability** -- a Rijke tube fed *both* its reflection coefficients and its
   flame transfer function as tabulated data reproduces the closed-form eigenmodes (and the
   instability) once the tables are continued; the raw tables drive the Nyquist count.
4. **Guard rails** -- a raw table refuses a complex frequency with a pointed error, and the
   pole-region diagnostics flag an untrustworthy fit.

Run in the ``nefes`` env (numba).
"""

import warnings

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.elements.continuation import rational_fit, continuation_warning
from nefes.elements.dynamic_source import (
    NTau,
    NTauLowpass,
    tabulated,
    heat_release_response,
    mass_flow_response,
)
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.perturbation import eigenmodes, nyquist_stability
from nefes.solver import solve
from nefes.thermo.configure import perfect_gas

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)
AREA, L1, L2 = 0.01, 0.6, 0.4


# ==========================================================================
# 1. Continuation fidelity -- off the real axis, not just on the grid
# ==========================================================================


@pytest.mark.parametrize(
    "F, delay",
    [
        (NTau(0.8, 3.0e-3), "auto"),
        (NTauLowpass(0.7, 4.0e-3, 350.0), "auto"),
        (NTauLowpass(0.7, 4.0e-3, 350.0), None),
        (lambda f: 0.85 / (1.0 + 2.0 * 0.18 * (1j * f / 420.0) + (1j * f / 420.0) ** 2), None),
    ],
)
def test_continuation_recovers_known_response_off_axis(F, delay):
    """The fit matches the closed form at complex frequencies away from the data grid."""
    fg = np.linspace(0.0, 600.0, 160)
    data = np.asarray([complex(F(f)) for f in fg])
    fit = rational_fit(fg, data, delay=delay)

    # on the grid: reproduces the samples to ~machine precision
    assert fit.max_error() < 1e-9
    # off the real axis (a lightly-damped mode sits here): still matches the closed form
    probes = [137.0 - 30.0 / (2 * np.pi) * 1j, 250.0 + 50.0 / (2 * np.pi) * 1j, 410.0 - 5.0j]
    for z in probes:
        assert abs(complex(fit(z)) - complex(F(z))) < 1e-6


def test_explicit_delay_collapses_the_fit():
    """Peeling off the exact transport lag leaves a low-order rational (the bare low-pass)."""
    F = NTauLowpass(0.8, 4.0e-3, 400.0)
    fg = np.linspace(0.0, 600.0, 200)
    fit = rational_fit(fg, F(fg), delay=4.0e-3)
    # n e^{-i w tau} / (1 + i f/fc) with the delay removed is degree-1 rational: 2 support points
    assert fit.n_terms <= 3
    assert fit.delay == pytest.approx(4.0e-3)


def test_auto_delay_estimates_the_lag():
    """``delay='auto'`` recovers the dominant transport lag from the phase slope."""
    fg = np.linspace(0.0, 800.0, 200)
    fit = rational_fit(fg, NTau(0.9, 2.5e-3)(fg), delay="auto")
    assert fit.delay == pytest.approx(2.5e-3, rel=0.1)


def test_vector_and_scalar_evaluation_shapes():
    """``__call__`` is shape-preserving: scalar in -> scalar out, array in -> array out."""
    fg = np.linspace(10.0, 500.0, 80)
    fit = rational_fit(fg, NTau(0.5, 1e-3)(fg))
    assert np.asarray(fit(123.0)).shape == ()
    assert fit(fg).shape == fg.shape
    # works through the complex(...) coercion the operator uses
    assert isinstance(complex(np.asarray(fit(80.0 + 1.0j)).reshape(-1)[0]), complex)


# ==========================================================================
# 2. Protocol -- analytic, and usable on every tabulated-data surface
# ==========================================================================


def test_fit_is_analytic_and_feeds_dynamic_sources():
    """A continuation is analytic, so it makes both a flame FTF and a fuel-modulation response."""
    fg = np.linspace(0.0, 500.0, 100)
    fit = rational_fit(fg, NTauLowpass(0.6, 3e-3, 300.0)(fg), delay="auto")
    assert fit.analytic is True
    assert heat_release_response(fit, ref_edge=1).analytic is True
    assert mass_flow_response(fit, ref_edge=1).analytic is True


def test_fit_serves_as_reflection_coefficient_real_and_complex():
    """A continued reflection coefficient evaluates at real *and* complex frequency."""

    def Rcf(f):  # near-open termination with a small end delay
        return -0.92 * np.exp(-1j * 2 * np.pi * f * 4e-4)

    fg = np.linspace(0.0, 1000.0, 120)
    fit = rational_fit(fg, np.asarray([complex(Rcf(f)) for f in fg]))
    bc = PerturbationBC.reflection(fit)
    rho, c, M = 1.2, 340.0, 0.0
    # real frequency (forced / Nyquist path)
    assert abs(bc.reflection_coefficient(250.0, rho, c, M) - complex(Rcf(250.0))) < 1e-6
    # complex frequency (the eigensolver path the raw table cannot take)
    z = 250.0 - 20.0 / (2 * np.pi) * 1j
    assert abs(bc.reflection_coefficient(z, rho, c, M) - complex(Rcf(z))) < 1e-5


# ==========================================================================
# 3. End-to-end Rijke stability: tabulated reflections + tabulated FTF
# ==========================================================================


def _rijke(ftf, bc_in, bc_out, *, mdot=0.005, dT=400.0):
    """Cold-air Rijke tube: inlet(bc_in) -> duct -> low-pass flame(ftf) -> duct -> outlet(bc_out)."""
    els = [
        cat.mass_flow_inlet(mdot, 300.0, perturbation_bc=bc_in),
        cat.duct(L1),
        cat.heat_release_flame(mdot * CP * dT, dynamic_source=heat_release_response(ftf, ref_edge=1)),
        cat.duct(L2),
        cat.pressure_outlet(1.0e5, perturbation_bc=bc_out),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA)]
    prob = cat.build_problem(perfect_gas(R_AIR, GAMMA), els, edges, mdot_ref=mdot, p_ref=1e5, h_ref=CP * 300.0)
    res = solve(prob)
    assert res.converged
    return prob, res.x


# Reference closed-form curves (all analytic).  The boundaries are near-rigid / near-open with a
# small end delay so |R| ~ 1 (energy-neutral, the Rijke instability survives) yet the reflection
# is genuinely frequency dependent -- a non-trivial continuation.
def _RIN(f):  # near-rigid inlet with a small end delay
    return 1.0 * np.exp(-1j * 2 * np.pi * f * 5e-5)


def _ROUT(f):  # near-open outlet with a small end delay
    return -1.0 * np.exp(-1j * 2 * np.pi * f * 5e-5)


_N, _TAU, _FC = 0.85, 4.0e-3, 400.0


def _modes(prob, x, band=(40.0, 320.0)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = eigenmodes(prob, x, freq_band=band, growth_band=(-250.0, 250.0), isentropic=True)
    return r


def test_tabulated_inputs_reproduce_closed_form_eigenmodes():
    """Feeding the reflections *and* the FTF as tabulated data (then continuing them) reproduces
    the closed-form spectrum -- including the self-excited instability."""
    fg = np.linspace(0.0, 2000.0, 400)

    # reference: closed-form callables straight in
    ftf_cf = NTauLowpass(_N, _TAU, _FC)
    prob_cf, x_cf = _rijke(ftf_cf, PerturbationBC.reflection(_RIN), PerturbationBC.reflection(_ROUT))
    res_cf = _modes(prob_cf, x_cf)

    # tabulated: sample each curve onto a grid, continue with rational_fit, feed those in
    ftf_tab = rational_fit(fg, ftf_cf(fg), delay="auto")
    bc_in = PerturbationBC.reflection(rational_fit(fg, np.asarray([complex(_RIN(f)) for f in fg])))
    bc_out = PerturbationBC.reflection(rational_fit(fg, np.asarray([complex(_ROUT(f)) for f in fg])))
    prob_tab, x_tab = _rijke(ftf_tab, bc_in, bc_out)
    res_tab = _modes(prob_tab, x_tab)

    # the mean flow is untouched by the perturbation inputs
    assert np.allclose(x_cf, x_tab, rtol=1e-8, atol=1e-8)

    # the self-excited (most unstable) mode matches in frequency and growth
    f_cf, g_cf = max(zip(res_cf.freqs, res_cf.growth_rates), key=lambda m: m[1])
    f_tab, g_tab = max(zip(res_tab.freqs, res_tab.growth_rates), key=lambda m: m[1])
    assert g_cf > 0.0 and g_tab > 0.0  # the instability is present both ways
    assert f_tab == pytest.approx(f_cf, rel=3e-3)
    assert g_tab == pytest.approx(g_cf, abs=max(2.0, 0.05 * abs(g_cf)))


def test_constant_reflection_table_reproduces_named_closures():
    """A constant +1 / -1 reflection table continues to the rigid-wall / open-end closures."""
    fg = np.linspace(0.0, 1500.0, 50)
    ftf = NTauLowpass(_N, _TAU, _FC)

    prob_named, x_named = _rijke(ftf, PerturbationBC.hard_wall(), PerturbationBC.open_end())
    bc_in = PerturbationBC.reflection(rational_fit(fg, np.ones_like(fg, dtype=complex)))
    bc_out = PerturbationBC.reflection(rational_fit(fg, -np.ones_like(fg, dtype=complex)))
    prob_tab, x_tab = _rijke(ftf, bc_in, bc_out)

    rn = _modes(prob_named, x_named)
    rt = _modes(prob_tab, x_tab)
    f_n, g_n = max(zip(rn.freqs, rn.growth_rates), key=lambda m: m[1])
    f_t, g_t = max(zip(rt.freqs, rt.growth_rates), key=lambda m: m[1])
    assert f_t == pytest.approx(f_n, rel=3e-3)
    assert g_t == pytest.approx(g_n, abs=max(2.0, 0.05 * abs(g_n)))


def test_nyquist_on_raw_tables_agrees_with_eigensolver_verdict():
    """The Nyquist driver runs on the *raw* tables (no continuation needed -- real frequency
    only) and agrees with the continued eigensolver that the tube is unstable.

    A lossless (``|R| = 1``) tube has a non-passive ``A_0``, so the Nyquist tally is not the
    absolute count; the robust, method-independent claim is the *verdict* (unstable) plus the
    unstable mode living in the swept band -- which is what the showcase demonstrates."""
    fg = np.linspace(0.0, 2500.0, 600)
    ftf_cf = NTauLowpass(_N, _TAU, _FC)

    # raw tabulated FTF straight into the Nyquist sweep (real frequency, no continuation)
    ftf_raw = tabulated(fg, np.asarray([complex(ftf_cf(f)) for f in fg]))
    prob_raw, x_raw = _rijke(ftf_raw, PerturbationBC.hard_wall(), PerturbationBC.open_end())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lossless ends -> non-passive A_0, count is a tally (expected)
        ny = nyquist_stability(prob_raw, x_raw, fg, isentropic=True)
        n_unstable_ny = ny.n_unstable
    assert n_unstable_ny >= 1  # raw tables -> Nyquist detects the instability

    # continued eigensolver: same case, finds the unstable mode in the swept band
    ftf_tab = rational_fit(fg, ftf_cf(fg), delay="auto")
    prob_tab, x_tab = _rijke(ftf_tab, PerturbationBC.hard_wall(), PerturbationBC.open_end())
    res = _modes(prob_tab, x_tab, band=(40.0, 320.0))
    f_u, g_u = max(zip(res.freqs, res.growth_rates), key=lambda m: m[1])
    assert g_u > 0.0  # the eigensolver agrees: unstable
    assert fg[0] <= f_u <= fg[-1]  # the unstable mode sits in the band Nyquist swept


# dissipative terminations (|R| < 1) so the flame-off operator A_0 is passive -- then the Nyquist
# encirclement count is the *absolute* unstable-mode number, and a rational fit of D(omega) along the
# real axis recovers each mode off-axis (the showcase notebook's "Nyquist gives freq + growth" claim).
def _RIN_LOSSY(f):
    return 0.85 * np.exp(-1j * 2 * np.pi * f * 2.0e-5)


def _ROUT_LOSSY(f):
    s = 1j * f / 950.0
    return -0.6 / (1.0 + 2.0 * 0.5 * s + s * s)


def test_nyquist_mode_estimates_recover_eigenmode_freq_and_growth():
    """On a cleanly-damped tube, the real-axis Nyquist sweep recovers the growing mode's frequency
    *and* growth (not just the count): a rational fit of the scalar determinant D(omega) on the real
    axis -- fed the smooth continuations -- has a zero at the eigensolver's complex eigenfrequency.

    This is the notebook's headline: Nyquist yields the same (freq, growth) the eigensolver does,
    from a real-axis sweep with no off-axis evaluation of the operator."""
    fg = np.linspace(0.0, 1800.0, 1000)
    ftf_cf = NTauLowpass(1.6, _TAU, 250.0)

    # continuations of every input (real-axis-exact and smooth, so D(omega) is smooth to fit)
    ftf_fit = rational_fit(fg, ftf_cf(fg), delay="auto")
    bc_in = PerturbationBC.reflection(rational_fit(fg, np.asarray([complex(_RIN_LOSSY(f)) for f in fg]), delay="auto"))
    bc_out = PerturbationBC.reflection(rational_fit(fg, np.asarray([complex(_ROUT_LOSSY(f)) for f in fg])))
    prob, x = _rijke(ftf_fit, bc_in, bc_out)

    # the eigensolver's most-unstable mode (the reference)
    res = _modes(prob, x, band=(40.0, 320.0))
    f_eig, g_eig = max(zip(res.freqs, res.growth_rates), key=lambda m: m[1])
    assert g_eig > 0.0

    # the real-axis Nyquist sweep on the same continued inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ny = nyquist_stability(prob, x, fg, isentropic=True)
        passive_ok = ny.passive_assumption_ok
        n_unstable = ny.n_unstable
        ests = ny.mode_estimates(unstable_only=True)

    # A_0 passive -> the count is the absolute number, and there is exactly one growing mode
    assert passive_ok
    assert n_unstable == 1
    assert len(ests) == 1

    # and that single off-axis estimate matches the eigenmode in frequency and growth
    m = ests[0]
    assert m["freq_hz"] == pytest.approx(f_eig, rel=2e-3)
    assert m["growth_rate"] == pytest.approx(g_eig, rel=0.1, abs=2.0)


def test_nyquist_mode_estimates_swarm_on_raw_piecewise_linear_tables():
    """The off-axis fit needs a *smooth* D(omega): raw piecewise-linear tables give a robust count
    but pollute mode_estimates with a pole/zero doublet at every table kink -- so the continuations
    (not the raw tables) are the right input for level-3 mode location."""
    fg = np.linspace(0.0, 1800.0, 1000)
    f_tab = np.linspace(0.0, 4000.0, 70)  # a coarse "measured" grid -> kinks under interpolation
    ftf_cf = NTauLowpass(1.6, _TAU, 250.0)

    raw_ftf = tabulated(f_tab, np.asarray([complex(ftf_cf(f)) for f in f_tab]))
    raw_in = PerturbationBC.reflection((f_tab, np.asarray([complex(_RIN_LOSSY(f)) for f in f_tab])))
    raw_out = PerturbationBC.reflection((f_tab, np.asarray([complex(_ROUT_LOSSY(f)) for f in f_tab])))
    prob, x = _rijke(raw_ftf, raw_in, raw_out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ny = nyquist_stability(prob, x, fg, isentropic=True)
        n_unstable = ny.n_unstable
        ests = ny.mode_estimates(unstable_only=True)

    # the count is still robust (the winding ignores the small kinks)
    assert n_unstable == 1
    # but the fit-based mode estimates are polluted by the kinks (many spurious "modes")
    assert len(ests) > 5
    # the true mode is nonetheless among them
    assert any(abs(m["freq_hz"] - 127.3) < 3.0 and m["growth_rate"] > 80.0 for m in ests)


# ==========================================================================
# 4. Guard rails -- diagnostics and pointed errors
# ==========================================================================


def test_raw_table_refuses_a_complex_frequency():
    """A raw (freqs, values) reflection table cannot be continued -- it errors, helpfully."""
    fg = np.linspace(0.0, 1000.0, 11)
    bc = PerturbationBC.reflection((fg, 0.5 * np.ones_like(fg, dtype=complex)))
    # real frequency is fine (the Nyquist / forced path)
    assert abs(bc.reflection_coefficient(300.0, 1.2, 340.0, 0.0) - 0.5) < 1e-12
    # complex frequency raises a message pointing at rational_fit
    with pytest.raises(TypeError, match="rational_fit"):
        bc.reflection_coefficient(300.0 - 5.0j, 1.2, 340.0, 0.0)


def test_tabulated_ftf_is_rejected_by_eigensolver():
    """A non-analytic (real-grid) FTF is still refused by the eigensolver (continue it first)."""
    fg = np.linspace(0.0, 2000.0, 300)
    ftf_raw = tabulated(fg, np.asarray([complex(NTauLowpass(_N, _TAU, _FC)(f)) for f in fg]))
    prob, x = _rijke(ftf_raw, PerturbationBC.hard_wall(), PerturbationBC.open_end())
    with pytest.raises(ValueError, match="analytically continuable"):
        eigenmodes(prob, x, freq_band=(40.0, 320.0), growth_band=(-200.0, 200.0), isentropic=True)


def test_pole_region_diagnostics():
    """``poles_in_region`` flags poles inside a search window; a clean fit leaves it empty."""
    fg = np.linspace(0.0, 600.0, 160)
    fit = rational_fit(fg, NTauLowpass(0.8, 4e-3, 400.0)(fg), delay=4e-3)
    # the bare low-pass pole sits at f = i fc -> growth = -2 pi fc ~ -2513 1/s (well below any modest band)
    assert fit.poles_in_region((40.0, 320.0), (-250.0, 250.0)).size == 0
    # a window dragged down to enclose the low-pass pole catches it
    caught = fit.poles_in_region((-50.0, 50.0), (-3000.0, -2000.0))
    assert caught.size >= 1
    # the warning helper mirrors the check
    with pytest.warns(RuntimeWarning, match="inside the search window"):
        continuation_warning(fit, (-50.0, 50.0), (-3000.0, -2000.0))


def test_input_validation():
    """Bad inputs are rejected up front."""
    with pytest.raises(ValueError, match="at least two"):
        rational_fit([100.0], [1.0 + 0j])
    with pytest.raises(ValueError, match="equal length"):
        rational_fit([1.0, 2.0, 3.0], [1.0 + 0j, 2.0 + 0j])
    with pytest.raises(ValueError, match="distinct"):
        rational_fit([1.0, 1.0, 2.0], [1.0 + 0j, 1.0 + 0j, 2.0 + 0j])
    with pytest.raises(ValueError, match="finite"):
        rational_fit([1.0, 2.0, np.inf], [1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    with pytest.raises(ValueError, match="delay"):
        rational_fit([1.0, 2.0], [1.0 + 0j, 2.0 + 0j], delay="bogus")


def test_plot_helpers_build_figures():
    """The plotting helpers return figures and focus the pole map on the data band."""
    from nefes.plotting import plot_fit, plot_pole_map

    fg = np.linspace(0.0, 4000.0, 90)
    fit = rational_fit(fg, NTauLowpass(1.2, 4e-3, 400.0)(fg), delay="auto")

    f1 = plot_fit(fit, extend=0.1, phase="deg")
    assert len(f1.data) == 4  # data + continuation, magnitude + phase

    f2 = plot_pole_map(fit, freq_band=(40.0, 320.0), growth_band=(-300.0, 300.0))
    assert len(f2.data) == 2  # poles + zeros
    # the default view focuses on the data band, not the far-field poles at tens of kHz
    assert f2.layout.yaxis.range[0] > -1e4 and f2.layout.yaxis.range[1] < 1e4
