"""Benchmark: the EM2C laboratory combustor, cross-checked against OSCILOS.

J. Li, D. Yang, C. Luzzato and A. S. Morgans, "Open Source Combustion Instability Low Order
Simulator (OSCILOS-Long) Technical Report," Department of Mechanical Engineering, Imperial
College London (2017), Sec. 5.4.1; report and code at https://www.oscilos.com and
https://github.com/MorgansLab/OSCILOS_long.  The configuration is a stable setting of the
swirl-stabilized combustor of P. Palies, D. Durox, T. Schuller and S. Candel, Combustion and
Flame 158(10), 1980-1991 (2011), doi:10.1016/j.combustflame.2011.02.012, reduced to three
cylinders: a plenum, an injection unit and a combustion chamber.  The flame model is the
delayed second-order low-pass response of A. P. Dowling, J. Fluid Mech. 346, 271-290 (1997),
doi:10.1017/S0022112097006484.

The case is fully specified in the report's text, so nothing has to be read off a plot:

    lengths   117.3 / 117 / 100 mm       radii  32.5 / 10.585 / 35 mm
    p_1 = 1 bar, T_1 = 300 K, T_3 = 1600 K, mean velocity 4.13 m/s at the injector exit
    rigid inlet (R = +1), open outlet (R = -1)
    FTF   F(f) = wc^2 / (s^2 + 2 zeta wc s + wc^2) exp(-tau s),  s = i 2 pi f,
          f_c = 200 Hz, zeta = 0.5, tau = 2 ms

and OSCILOS reports the dominant mode at ``152.6 Hz`` with growth rate ``-19.1 1/s``.

Two conventions of that reference matter for a like-for-like comparison.  Its mean flow runs
on a calorically perfect gas with ``gamma = 1.4`` and ``R = R_u / W_air`` on *both* sides of
the flame (its "constant gamma" option), which is exactly the Nefes ``perfect_gas`` model; and
the flame sits at an abrupt area increase, treated as a Borda-Carnot expansion followed by
constant-area heat addition, which is ``sudden_area_change`` followed by ``heat_release_flame``.

Nefes returns 153.4 Hz and -19.0 1/s: 0.5 % in frequency, 0.7 % in growth rate.  The residual
is the size of the inlet-temperature ambiguity in the report itself -- its cold-tube case
quotes ``T = 300 K`` while the sound speed it uses, 343.25 m/s, is the value at 293.15 K, and
running this case at 293.15 K instead moves the mode to 151.7 Hz / -20.6 1/s, bracketing the
published pair.
"""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.elements.dynamic_source import heat_release_response, n_tau_lowpass2
from nefes.perturbation import eigenmodes
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.shell.network import Network
from nefes.thermo.configure import perfect_gas
from nefes.assembly.recover import ES_T, ES_U

# OSCILOS's air: R = R_u / W_air, and its "constant gamma" option holds gamma = 1.4 everywhere.
R_AIR = 8.3145 / 28.96512 * 1000.0
GAMMA = 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)

# Geometry: plenum / injection unit / combustion chamber.
LENGTHS = (0.1173, 0.117, 0.100)
RADII = (0.0325, 0.010585, 0.035)
AREAS = tuple(np.pi * r**2 for r in RADII)

# Operating point.
P_MEAN = 1.0e5  # mean pressure [Pa]
T_COLD = 300.0  # reactant temperature [K]
T_BURNT = 1600.0  # burnt-gas temperature [K]
U_INJECTOR = 4.13  # mean velocity at the injector exit [m/s]

# Flame transfer function: a second-order low-pass with a pure time lag, unit gain at f = 0.
FTF_GAIN, FTF_TAU, FTF_FC, FTF_ZETA = 1.0, 2.0e-3, 200.0, 0.5

# The published eigenvalue of the dominant mode.
OSCILOS_FREQ_HZ = 152.6
OSCILOS_GROWTH = -19.1

# Search box around the dominant mode.
FREQ_BAND = (60.0, 320.0)
GROWTH_BAND = (-160.0, 160.0)


def em2c_combustor(*, T_cold=T_COLD, active=True, expansion="borda"):
    """Build and solve the three-cylinder EM2C combustor.

    Parameters
    ----------
    T_cold : float, optional
        Reactant temperature [K] at the plenum inlet.
    active : bool, optional
        Attach the flame transfer function (default ``True``).  With ``False`` the flame
        is a steady heat source and the network is acoustically passive.
    expansion : {"borda", "isentropic"}, optional
        Model of the abrupt area increase at the chamber inlet.  ``"borda"`` (default) is
        the dissipative sudden expansion OSCILOS uses; ``"isentropic"`` replaces it with a
        lossless area change and so removes the only interior sink of acoustic energy.

    Returns
    -------
    Solution
        The converged mean flow, carrying ``.problem`` and ``.x``.
    """
    rho_injector = P_MEAN / (R_AIR * T_cold)
    mdot = rho_injector * U_INJECTOR * AREAS[1]
    qdot = mdot * CP * (T_BURNT - T_cold)

    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=P_MEAN, T_ref=T_cold, mdot_ref=mdot)
    i_inlet = net.add(cat.mass_flow_inlet(mdot, T_cold, perturbation_bc=PerturbationBC.hard_wall()))
    i_plenum = net.add(cat.duct(LENGTHS[0], name="plenum"))
    i_contract = net.add(cat.isentropic_area_change(name="contraction"))
    i_injector = net.add(cat.duct(LENGTHS[1], name="injector"))
    i_expand = net.add(
        cat.sudden_area_change(name="dump") if expansion == "borda" else cat.isentropic_area_change(name="dump")
    )
    i_flame = net.add(cat.heat_release_flame(qdot))
    i_chamber = net.add(cat.duct(LENGTHS[2], name="chamber"))
    i_outlet = net.add(cat.pressure_outlet(P_MEAN, perturbation_bc=PerturbationBC.open_end()))

    net.connect(i_inlet, i_plenum, AREAS[0])
    net.connect(i_plenum, i_contract, AREAS[0])
    net.connect(i_contract, i_injector, AREAS[1])
    e_ref = net.connect(i_injector, i_expand, AREAS[1])
    net.connect(i_expand, i_flame, AREAS[2])
    net.connect(i_flame, i_chamber, AREAS[2])
    net.connect(i_chamber, i_outlet, AREAS[2])

    if active:
        ftf = n_tau_lowpass2(FTF_GAIN, FTF_TAU, FTF_FC, FTF_ZETA)
        net.set_dynamic_source(i_flame, heat_release_response(ftf, ref_edge=e_ref))

    sol = net.solve()
    assert sol.converged
    return sol, e_ref


def dominant_mode(sol, *, isentropic=True):
    """The single mode of the search box, as ``(frequency [Hz], growth rate [1/s])``."""
    res = eigenmodes(sol.problem, sol.x, freq_band=FREQ_BAND, growth_band=GROWTH_BAND, isentropic=isentropic)
    assert len(res) == 1, f"expected one mode in the box, found {len(res)}"
    return float(res.freqs[0]), float(res.growth_rates[0])


@pytest.fixture(scope="module")
def active_solution():
    return em2c_combustor()[0]


def test_mean_flow_matches_the_reported_operating_point(active_solution):
    est = active_solution.table()
    # the injector exit is the edge feeding the dump plane; the chamber runs at the burnt temperature
    assert est[ES_U, 3] == pytest.approx(U_INJECTOR, rel=1e-3)
    assert est[ES_T, 1] == pytest.approx(T_COLD, rel=1e-4)
    assert est[ES_T, 6] == pytest.approx(T_BURNT, rel=1e-4)


def test_dominant_mode_matches_oscilos(active_solution):
    freq, growth = dominant_mode(active_solution)
    assert freq == pytest.approx(OSCILOS_FREQ_HZ, rel=0.01)
    assert growth == pytest.approx(OSCILOS_GROWTH, rel=0.02)
    assert growth < 0.0  # the reported configuration is stable


def test_inlet_temperature_ambiguity_brackets_the_published_mode(active_solution):
    # the report states 300 K but its cold-tube sound speed is the 293.15 K value; the published
    # eigenvalue lies between the two readings, so the residual disagreement is its own ambiguity
    hot = dominant_mode(active_solution)
    cold = dominant_mode(em2c_combustor(T_cold=293.15)[0])
    assert cold[0] < OSCILOS_FREQ_HZ < hot[0]
    assert cold[1] < OSCILOS_GROWTH < hot[1]


def test_the_entropy_wave_is_a_spectator_at_an_open_end(active_solution):
    # the flame sheds an entropy wave that convects out of the pressure-release outlet without
    # generating sound, so the full operator and the isentropic reduction share the eigenvalue
    full = dominant_mode(active_solution, isentropic=False)
    isen = dominant_mode(active_solution, isentropic=True)
    # agreement to the eigensolver's polish tolerance, far below any modelling difference
    assert full[0] == pytest.approx(isen[0], abs=1e-3)
    assert full[1] == pytest.approx(isen[1], abs=1e-3)


def test_the_dump_plane_carries_the_passive_damping():
    # with the flame inert the only interior loss is the Borda-Carnot expansion; removing it
    # (a lossless area change) leaves an energy-neutral network, so the mode stops decaying
    _, borda = dominant_mode(em2c_combustor(active=False)[0])
    _, lossless = dominant_mode(em2c_combustor(active=False, expansion="isentropic")[0])
    assert borda < -10.0
    assert abs(lossless) < 1.0


def test_the_flame_pulls_the_frequency_up_and_adds_damping(active_solution):
    passive_freq, passive_growth = dominant_mode(em2c_combustor(active=False)[0])
    active_freq, active_growth = dominant_mode(active_solution)
    # near 150 Hz the transfer function is close to antiphase, so the flame damps rather than drives
    assert active_freq > passive_freq
    assert active_growth < passive_growth
