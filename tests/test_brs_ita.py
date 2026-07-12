"""Benchmark: acoustic and intrinsic thermoacoustic modes of the TUM BRS test rig.

T. Emmert, S. Bomberg, S. Jaensch and W. Polifke, "Acoustic and intrinsic thermoacoustic
modes of a premixed combustor", Proceedings of the Combustion Institute 36 (2017)
3835-3842, doi:10.1016/j.proci.2016.08.002.  The rig is the perfectly premixed swirl
burner of T. Komarek and W. Polifke, J. Eng. Gas Turbines Power 132(6), 061503 (2010),
doi:10.1115/1.4000127.

The network of the reference's Table 1 is a closed end, a plenum, an area contraction, a
swirler tube, an area expansion, a compact flame, a combustion chamber and an open end:

    closed end (R = +1)                       inlet Mach 0.0011, c = 343 m/s, rho = 1.2
    plenum        0.17 m                      area ratio 29.76 into the swirler tube
    swirler tube  0.18 m                      area ratio 0.13 into the chamber
    flame         theta = (T_d - T_u)/T_u = 5.59,   xi = rho_u c_u / (rho_d c_d) = 2.57
    chamber       0.70 m                      open end (R = -1)

Those entries fix the whole mean state, and they are mutually consistent for a calorically
perfect gas: ``xi = sqrt(1 + theta)``, and the implied chamber cross-section is the
90 x 90 mm square of the rig.  The one quantity Table 1 does not give in numbers is the
flame transfer function, which it shows only as a figure; its curves are digitized in
``examples/thermoacoustics/data/brs_ftf.csv`` and turned back into the finite impulse
response the reference says they came from.

The published eigenvalues live in ``examples/thermoacoustics/data/brs_published_*.csv``,
read off the vector twins of the paper's figures in the first author's dissertation, so
the comparison targets carry no pixel quantization.  The reference reports three dominant
(least decaying) modes near 42, 111 and 315 Hz, two acoustic and one intrinsic, all
stable; two pure acoustic modes; a family of pure intrinsic (ITA) modes with the dominant
one near 105 Hz; and the paradox that lowering the outlet reflection stabilizes the
acoustic modes while *destabilizing* the intrinsic one.

Two conventions of the reference matter for reading the numbers.  Growth rates are quoted
in hertz (sigma / 2 pi, with s = sigma + j omega), and a mode counts as dominant when its
growth exceeds the -25 Hz line of its Fig. 4.  And the reference's scalar equations write
the flame coupling as ``theta * F``, which holds only for a heat-release fluctuation
normalized by the velocity at the flame's own (chamber) side; the published FTF is
normalized by the burner-mouth velocity, so the physically consistent coupling carries the
area ratio: ``theta * alpha_2 * F``.  Solving the reference's Eq. (9) as printed with its
own flame response yields an unstable pure-ITA mode, contradicting its own figure; with
the ``alpha_2`` bridge the published pure-ITA spectrum is recovered.

Reading the published FTF figure to one pixel leaves 0.135 rad of phase uncertainty, which
moves the intrinsic mode by about +/- 4 Hz; the frequency tolerances below are set by
that, not by the network model, whose Mach-number terms shift the modes by less than a
hertz.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from nefes.assembly.recover import ES_T, ES_U
from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import finite_impulse_response, fit_impulse_response, heat_release_response
from nefes.perturbation import eigenmodes
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.shell.network import Network
from nefes.thermo.configure import perfect_gas

DATA = Path(__file__).resolve().parents[1] / "examples" / "thermoacoustics" / "data"
FTF_CSV = DATA / "brs_ftf.csv"
PUBLISHED_CSV = DATA / "brs_published_eigenvalues.csv"
SWEEP_CSV = DATA / "brs_published_reflection_sweep.csv"

R_AIR, GAMMA = 287.05, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)

# Table 1 of the reference.
C_COLD = 343.0  # speed of sound upstream [m/s]
RHO_COLD = 1.2  # density upstream [kg/m^3]
MACH_INLET = 0.0011  # Mach number at the plenum inlet
THETA = 5.59  # normalized temperature jump across the flame
L_PLENUM, L_SWIRLER, L_CHAMBER = 0.17, 0.18, 0.70
AREA_RATIO_1, AREA_RATIO_2 = 29.76, 0.13
R_PLENUM = 0.10  # plenum radius [m], from the reference's cut-on estimate
XI = math.sqrt(1.0 + THETA)  # ratio of specific impedances of a perfect gas (quoted 2.57)

T_COLD = C_COLD**2 / (GAMMA * R_AIR)
P_MEAN = RHO_COLD * R_AIR * T_COLD
T_HOT = (1.0 + THETA) * T_COLD
A_PLENUM = math.pi * R_PLENUM**2
A_SWIRLER = A_PLENUM / AREA_RATIO_1
A_CHAMBER = A_SWIRLER / AREA_RATIO_2
MDOT = RHO_COLD * MACH_INLET * C_COLD * A_PLENUM
QDOT = MDOT * CP * (T_HOT - T_COLD)

# The impulse response is reconstructed with its resolvable-frequency limit at the edge of
# the digitized band, so the fit carries no frequency content the figure cannot constrain.
FIR_DT, FIR_DURATION, FIR_SMOOTHING = 1.0e-3, 20.0e-3, 1.0e-4

# The reference's dominance criterion: the dashed line of its Fig. 4 sits at -25 Hz.
DOMINANT_GROWTH_HZ = -25.0

FREQ_BAND, GROWTH_BAND = (10.0, 480.0), (-160.0, 60.0)

TWO_PI = 2.0 * math.pi


def _digitized_ftf():
    f, gain, phase = np.loadtxt(FTF_CSV, delimiter=",", unpack=True)
    return f, gain * np.exp(1j * phase)


def _impulse_response():
    """Smoothed impulse response behind the digitized frequency response."""
    f, F = _digitized_ftf()
    return fit_impulse_response(f, F, duration=FIR_DURATION, dt=FIR_DT, smoothing=FIR_SMOOTHING).h


def published(system):
    """Published eigenvalues of one subsystem as ``[(frequency [Hz], growth [Hz])]``."""
    rows = np.genfromtxt(PUBLISHED_CSV, delimiter=",", comments="#", skip_header=9, dtype=None, encoding=None)
    return sorted((float(f), float(g)) for label, g, f in rows if label == system and f > 1.0)


def brs_rig(*, active=True, bc_outlet=None, mdot_scale=1.0):
    """Build and solve the BRS network of the reference's Table 1.

    Parameters
    ----------
    active : bool, optional
        Attach the flame transfer function (default ``True``).  With ``False`` the flame is
        a steady heat source and the network is acoustically passive, which is the
        reference's "pure acoustic" system.
    bc_outlet : PerturbationBC, optional
        Outlet closure (default a pressure-release open end, ``R = -1``).
    mdot_scale : float, optional
        Scale the mass flow and heat power together, holding the temperature ratio.  Used
        to show that the Mach-number terms do not carry the comparison.

    Returns
    -------
    Solution
    """
    mdot, qdot = MDOT * mdot_scale, QDOT * mdot_scale
    bc_outlet = bc_outlet or PerturbationBC.open_end()

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=P_MEAN, T_ref=T_COLD, mdot_ref=mdot)
    i_inlet = net.add(cat.mass_flow_inlet(mdot, T_COLD, perturbation_bc=PerturbationBC.hard_wall()))
    i_plenum = net.add(cat.duct(L_PLENUM, name="plenum"))
    i_contract = net.add(cat.isentropic_area_change(name="area-1"))
    i_swirler = net.add(cat.duct(L_SWIRLER, name="swirler"))
    i_expand = net.add(cat.isentropic_area_change(name="area-2"))
    i_flame = net.add(cat.heat_release_flame(qdot))
    i_chamber = net.add(cat.duct(L_CHAMBER, name="chamber"))
    i_outlet = net.add(cat.pressure_outlet(P_MEAN, perturbation_bc=bc_outlet))

    net.connect(i_inlet, i_plenum, A_PLENUM)
    net.connect(i_plenum, i_contract, A_PLENUM)
    net.connect(i_contract, i_swirler, A_SWIRLER)
    e_ref = net.connect(i_swirler, i_expand, A_SWIRLER)  # the burner mouth: the FTF reference
    net.connect(i_expand, i_flame, A_CHAMBER)
    net.connect(i_flame, i_chamber, A_CHAMBER)
    net.connect(i_chamber, i_outlet, A_CHAMBER)

    if active:
        ftf = finite_impulse_response(_impulse_response(), FIR_DT)
        net.set_dynamic_source(i_flame, heat_release_response(ftf, ref_edge=e_ref))

    sol = net.solve()
    assert sol.converged
    return sol


def ita_rig():
    """The reference's pure ITA system: the burner mouth and flame between anechoic ends."""
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=P_MEAN, T_ref=T_COLD, mdot_ref=MDOT)
    i_inlet = net.add(cat.mass_flow_inlet(MDOT, T_COLD, perturbation_bc=PerturbationBC.anechoic()))
    i_expand = net.add(cat.isentropic_area_change(name="area-2"))
    i_flame = net.add(cat.heat_release_flame(QDOT))
    i_outlet = net.add(cat.pressure_outlet(P_MEAN, perturbation_bc=PerturbationBC.anechoic()))
    e_ref = net.connect(i_inlet, i_expand, A_SWIRLER)
    net.connect(i_expand, i_flame, A_CHAMBER)
    net.connect(i_flame, i_outlet, A_CHAMBER)
    ftf = finite_impulse_response(_impulse_response(), FIR_DT)
    net.set_dynamic_source(i_flame, heat_release_response(ftf, ref_edge=e_ref))
    sol = net.solve()
    assert sol.converged
    return sol


def spectrum(sol, growth_band=GROWTH_BAND, freq_band=FREQ_BAND):
    """Modes in the search box as ``[(frequency [Hz], growth rate [1/s])]``, low to high."""
    res = eigenmodes(sol.problem, sol.x, freq_band=freq_band, growth_band=growth_band, isentropic=True)
    return sorted(zip([float(v) for v in res.freqs], [float(v) for v in res.growth_rates]))


def dominant(modes):
    """The reference calls a mode dominant when its growth rate exceeds the -25 Hz line."""
    return [(f, g) for f, g in modes if g / TWO_PI > DOMINANT_GROWTH_HZ]


def nearest(modes, freq):
    return min(modes, key=lambda m: abs(m[0] - freq))


def pure_ita_root(theta_eff, seed=(-20.0, 100.0)):
    """Newton root of the pure ITA relation ``alpha_2 + xi + theta_eff F(s) = 0``.

    Returns ``(frequency [Hz], growth [Hz])`` in the reference's convention
    ``s = sigma + j omega``.
    """
    h = _impulse_response()
    lags = np.arange(h.size) * FIR_DT

    def relation(s):
        return AREA_RATIO_2 + XI + theta_eff * np.sum(h * np.exp(-s * lags))

    s = TWO_PI * (seed[0] + 1j * seed[1])
    for _ in range(60):
        step = 1e-4 * (1.0 + abs(s))
        d = (relation(s + step) - relation(s - step)) / (2.0 * step)
        ds = relation(s) / d
        s -= ds
        if abs(ds) < 1e-10 * (1.0 + abs(s)):
            break
    return s.imag / TWO_PI, s.real / TWO_PI


@pytest.fixture(scope="module")
def active_solution():
    return brs_rig()


@pytest.fixture(scope="module")
def active_modes(active_solution):
    return spectrum(active_solution)


@pytest.fixture(scope="module")
def passive_modes():
    return spectrum(brs_rig(active=False))


# ==========================================================================
# The mean state, which Table 1 fixes completely
# ==========================================================================


def test_table_1_is_self_consistent_for_a_perfect_gas():
    # the reference quotes xi and theta independently; for a calorically perfect gas at
    # constant pressure they are not independent, and the quoted pair agrees
    assert XI == pytest.approx(2.57, abs=5e-3)
    # the area ratios and the plenum radius imply the 90 x 90 mm square chamber of the rig
    assert math.sqrt(A_CHAMBER) == pytest.approx(0.090, abs=2e-4)
    # rho = 1.2 with c = 343 m/s puts the rig at atmospheric pressure
    assert P_MEAN == pytest.approx(101325.0, rel=0.01)


def test_mean_flow_reproduces_the_reported_operating_point(active_solution):
    est = active_solution.table()
    t_cold, t_hot = est[ES_T, 1], est[ES_T, 6]
    assert t_cold == pytest.approx(T_COLD, rel=1e-4)
    assert t_hot == pytest.approx(T_HOT, rel=1e-4)
    assert (t_hot - t_cold) / t_cold == pytest.approx(THETA, rel=1e-3)
    inlet_mach = est[ES_U, 1] / math.sqrt(GAMMA * R_AIR * t_cold)
    assert inlet_mach == pytest.approx(MACH_INLET, rel=1e-3)


def test_the_chamber_quarter_wave_sets_the_highest_dominant_mode():
    # a hot chamber closed at the flame and open at the exit rings at c_hot / (4 L)
    quarter_wave = C_COLD * XI / (4.0 * L_CHAMBER)
    highest = max(f for f, g in published("full") if g > DOMINANT_GROWTH_HZ)
    assert quarter_wave == pytest.approx(highest, rel=0.01)


# ==========================================================================
# The flame response, recovered from the digitized figure
# ==========================================================================


def test_the_impulse_response_has_the_shape_of_a_swirl_flame():
    h = _impulse_response()
    lags = np.arange(h.size) * FIR_DT
    peak, trough = int(np.argmax(h)), int(np.argmin(h))
    # a positive lobe from the axial-velocity response, followed by a negative swirl lobe
    assert h[peak] > 0.5 and 3.0e-3 < lags[peak] < 7.0e-3
    assert h[trough] < -0.05 and lags[trough] > lags[peak]
    # the zero-frequency gain is the sum of the coefficients, and matches the figure
    assert h.sum() == pytest.approx(1.3, abs=0.1)
    # the mean lag is the burner's convective timescale, a few milliseconds
    assert 3.5e-3 < (lags * h).sum() / h.sum() < 5.5e-3


def test_the_reconstructed_response_reproduces_the_digitized_curve():
    f, F = _digitized_ftf()
    fit = finite_impulse_response(_impulse_response(), FIR_DT)(f.astype(complex))
    residual = np.abs(fit - F)
    # one pixel of phase is 0.135 rad; binning the trace to 5 Hz leaves about 0.06 of noise,
    # and the reconstruction sits at that floor rather than below it
    assert math.sqrt((residual**2).mean()) < 0.07
    assert abs(F[0]) == pytest.approx(1.179, abs=0.01)
    assert np.abs(F).max() == pytest.approx(1.77, abs=0.02)


def test_the_digitized_curve_agrees_with_the_reference_impulse_response():
    # the dissertation publishes the impulse response behind the same FTF as a stem plot;
    # its frequency response is an independent record of the same quantity, so the two
    # digitizations validate each other
    coeffs = np.loadtxt(DATA / "brs_impulse_response.csv", delimiter=",", skiprows=10, usecols=3)
    f, F = _digitized_ftf()
    lags = np.arange(coeffs.size) * 2.5e-4
    resp = (coeffs * np.exp(-2j * np.pi * np.outer(f, np.arange(coeffs.size)) * 2.5e-4)).sum(axis=1)
    # the two gain records agree far inside one printed pixel
    gain_rms = math.sqrt(float(((np.abs(resp) - np.abs(F)) ** 2).mean()))
    assert gain_rms < 0.02
    # the phase records agree within about two printed pixels (one pixel is 0.135 rad),
    # with no systematic drift; that residual is what moves the intrinsic mode a few hertz
    dphi = np.unwrap(np.angle(resp)) - np.unwrap(np.angle(F))
    assert float(np.abs(dphi).max()) < 0.45
    # and the peak sits at the characteristic delay the dissertation itself quotes (4.8 ms)
    assert lags[int(np.argmax(coeffs))] == pytest.approx(4.8e-3, abs=0.5e-3)


# ==========================================================================
# Figure 4: the full, pure acoustic, and pure ITA spectra
# ==========================================================================


def test_the_passive_network_matches_the_published_acoustic_modes(passive_modes):
    # the reference expects "only two acoustic modes ... in the frequency range": the
    # Helmholtz mode of plenum and swirler tube, and the chamber quarter wave
    assert len(passive_modes) == 2
    for (f, g), (f_pub, g_pub) in zip(passive_modes, published("acoustic")):
        assert f == pytest.approx(f_pub, abs=0.5)
        # the ideal modes are neutral; the reference's small positive growth at the quarter
        # wave is its own discretization, ours is damping from the retained Mach terms
        assert abs(g / TWO_PI) < 3.5
        assert abs(g_pub) < 3.5


def test_the_flame_adds_one_mode_that_the_passive_network_does_not_have(active_modes, passive_modes):
    assert len(active_modes) > len(passive_modes)
    # the extra mode near 106 Hz has no passive counterpart anywhere near it
    intrinsic = nearest(active_modes, 111.0)
    assert min(abs(intrinsic[0] - f) for f, _ in passive_modes) > 50.0


def test_three_dominant_modes_match_the_published_frequencies(active_modes):
    # the reference's three dominant modes: the Helmholtz mode, the intrinsic mode, and
    # the chamber quarter wave; the frequency residual is the one-pixel digitization of
    # the flame response
    pub = [(f, g) for f, g in published("full") if g > DOMINANT_GROWTH_HZ]
    assert len(pub) == 3
    got = dominant(active_modes)
    for f_pub, _ in pub:
        f = nearest(got, f_pub)[0]
        assert f == pytest.approx(f_pub, rel=0.06)
    assert all(g < 0.0 for _, g in got), "the reference reports the rig as stable"


def test_the_robust_growth_rates_match_the_published_values(active_modes):
    # the Helmholtz and intrinsic growth rates are insensitive to the flame-response
    # digitization and land on the published values
    for f_pub, g_pub in [(42.42, -8.16), (111.04, -2.02)]:
        f, g = nearest(active_modes, f_pub)
        assert g / TWO_PI == pytest.approx(g_pub, abs=2.5)


def test_the_near_degenerate_pair_matches_in_its_mean(active_modes):
    # near 315 Hz the quarter wave and the second intrinsic mode form an avoided crossing;
    # its splitting is hypersensitive to the flame response (the reference's own exact
    # impulse response splits it differently than its published figure), but the pair's
    # mean is robust and is compared here
    pair = sorted(active_modes, key=lambda m: abs(m[0] - 315.0))[:2]
    pub_pair = [(314.28, -26.58), (315.58, -6.34)]
    f_mean = sum(f for f, _ in pair) / 2.0
    g_mean = sum(g for _, g in pair) / (2.0 * TWO_PI)
    assert f_mean == pytest.approx(sum(f for f, _ in pub_pair) / 2.0, rel=0.02)
    assert g_mean == pytest.approx(sum(g for _, g in pub_pair) / 2.0, abs=4.0)


def test_the_intrinsic_mode_is_the_least_damped(active_modes):
    intrinsic = nearest(active_modes, 111.0)
    assert intrinsic == max(active_modes, key=lambda m: m[1])
    # "dominant and marginally stable": within a hertz of neutral in growth-rate units
    assert -2.0 < intrinsic[1] / TWO_PI < 0.0


def test_the_mach_number_terms_do_not_carry_the_comparison(active_modes):
    # the reference's jump conditions are written at zero Mach; ours are not, so check that
    # shrinking the mean flow twentyfold leaves the modes where they are
    slow = spectrum(brs_rig(mdot_scale=0.05), growth_band=(-180.0, 60.0))
    assert len(slow) == len(active_modes)
    for (f_slow, _), (f, _) in zip(slow, active_modes):
        assert f_slow == pytest.approx(f, abs=0.8)


# ==========================================================================
# The pure ITA system and the reference's dispersion relation
# ==========================================================================


def test_the_pure_ita_relation_needs_the_normalization_bridge():
    # the reference's Eq. (9), alpha + xi + theta F = 0, holds for a heat release
    # normalized by the flame-side velocity; with the burner-mouth FTF it shows an
    # unstable pure ITA mode, contradicting the reference's own (all stable) figure
    f_hz, g_hz = pure_ita_root(theta_eff=THETA, seed=(30.0, 110.0))
    assert g_hz > 10.0
    # the area ratio alpha_2 = u_bar(flame side) / u_bar(burner mouth) closes the gap:
    # with it, the dominant pure ITA mode lands on the published square
    f_hz, g_hz = pure_ita_root(theta_eff=THETA * AREA_RATIO_2)
    f_pub, g_pub = nearest(published("ita"), 105.0)
    assert f_hz == pytest.approx(f_pub, abs=3.0)
    assert g_hz == pytest.approx(g_pub, abs=1.5)


def test_the_pure_ita_network_matches_the_dispersion_relation():
    # the same eigenvalue must come out of the network operator with anechoic ends;
    # this ties the assembled jump conditions to the reference's scalar relation
    sol = ita_rig()
    modes = spectrum(sol, freq_band=(60.0, 130.0), growth_band=(-250.0, 0.0))
    assert len(modes) == 1
    f_net, g_net = modes[0]
    f_rel, g_rel = pure_ita_root(theta_eff=THETA * AREA_RATIO_2)
    assert f_net == pytest.approx(f_rel, abs=1.5)
    assert g_net / TWO_PI == pytest.approx(g_rel, abs=1.5)
    # and lands on the published dominant pure ITA mode
    f_pub, g_pub = nearest(published("ita"), 105.0)
    assert f_net == pytest.approx(f_pub, abs=3.5)
    assert g_net / TWO_PI == pytest.approx(g_pub, abs=2.0)


# ==========================================================================
# Figure 7: acoustic losses at the outlet destabilize the intrinsic mode
# ==========================================================================


def test_reducing_the_outlet_reflection_destabilizes_the_intrinsic_mode(active_modes):
    # the reference's closing result: reflections bleed energy out of the intrinsic
    # feedback loop, so removing them releases it
    intrinsic_full = max(active_modes, key=lambda m: m[1])
    leaky = spectrum(brs_rig(bc_outlet=PerturbationBC.reflection(-0.6)), growth_band=(-160.0, 200.0))
    intrinsic_leaky = max(leaky, key=lambda m: m[1])
    assert intrinsic_full[1] < 0.0
    assert intrinsic_leaky[1] > 0.0


def test_reducing_the_outlet_reflection_stabilizes_the_acoustic_mode(active_modes):
    acoustic_full = nearest(active_modes, 42.4)
    leaky = spectrum(brs_rig(bc_outlet=PerturbationBC.reflection(-0.8)))
    acoustic_leaky = nearest(leaky, acoustic_full[0])
    assert acoustic_leaky[1] < acoustic_full[1]


def test_the_anechoic_outlet_endpoint_matches_the_published_track():
    # the published sweep ends, at a non-reflective outlet, with the intrinsic mode
    # unstable at (+23.0 Hz, 88.5 Hz); the frequency residual is again the flame-response
    # digitization
    sol = brs_rig(bc_outlet=PerturbationBC.anechoic())
    modes = spectrum(sol, freq_band=(60.0, 130.0), growth_band=(-50.0, 250.0))
    assert len(modes) == 1
    f, g = modes[0]
    rows = np.genfromtxt(SWEEP_CSV, delimiter=",", comments="#", skip_header=11, dtype=None, encoding=None)
    ends = [(float(ff), float(gg)) for label, gg, ff in rows if label == "at-zero-reflection" and ff > 50.0]
    f_pub, g_pub = min(ends, key=lambda m: abs(m[0] - f))
    assert g / TWO_PI == pytest.approx(g_pub, abs=1.5)
    assert f == pytest.approx(f_pub, abs=5.5)
