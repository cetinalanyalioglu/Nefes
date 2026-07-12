"""Eigenmode spectrum plot with the search contour overlay."""

import warnings

import numpy as np

from nefes.assembly.recover import ES_C
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC, eigenmodes
from nefes.perturbation.stability.contour import ellipse_contour
from nefes.plotting import plot_spectrum
from nefes.plotting.spectrum import _contour_to_fg
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)
LDUCT = 0.5


def _contour_trace_names(fig):
    return [t for t in fig.data if t.name == "search contour"]


def test_contour_maps_omega_to_frequency_growth():
    # An ellipse centred at omega = 2*pi*500 - i*0 maps to f = 500 Hz, growth band +/- ry.
    c = ellipse_contour(2.0 * np.pi * 500.0 + 0.0j, 2.0 * np.pi * 100.0, 50.0, 64)
    x, y = _contour_to_fg(c)
    assert np.isclose(x.max(), 600.0, atol=1.0) and np.isclose(x.min(), 400.0, atol=1.0)
    assert np.isclose(y.max(), 50.0, atol=1.0) and np.isclose(y.min(), -50.0, atol=1.0)


def test_plot_spectrum_draws_contour_when_given():
    freqs = np.array([120.0, 480.0])
    growth = np.array([10.0, -5.0])
    c = ellipse_contour(2.0 * np.pi * 300.0 + 0.0j, 2.0 * np.pi * 250.0, 40.0, 32)
    fig_no = plot_spectrum(freqs, growth)
    fig_yes = plot_spectrum(freqs, growth, contour=c)
    assert not _contour_trace_names(fig_no)
    assert len(_contour_trace_names(fig_yes)) == 1


def test_plot_spectrum_accepts_a_list_of_contours():
    c1 = ellipse_contour(2.0 * np.pi * 200.0 + 0.0j, 2.0 * np.pi * 100.0, 40.0, 16)
    c2 = ellipse_contour(2.0 * np.pi * 600.0 + 0.0j, 2.0 * np.pi * 100.0, 40.0, 16)
    fig = plot_spectrum(np.array([200.0]), np.array([1.0]), contour=[c1, c2])
    traces = _contour_trace_names(fig)
    assert len(traces) == 2
    # exactly one carries the legend entry
    assert sum(bool(t.showlegend) for t in traces) == 1


def test_eigenmode_result_plot_spectrum_overlays_its_contour():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(LDUCT))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    c = float(sol.table()[ES_C, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eigenmodes(sol.problem, sol.x, (0.4 * c / (2 * LDUCT), 1.6 * c / (2 * LDUCT)))
    assert res.n_modes >= 1 and res.contour is not None
    fig = res.plot_spectrum()
    assert len(_contour_trace_names(fig)) == 1
    # the eigenvalue markers are present too (stable and/or unstable traces)
    assert any(t.name in ("stable / decaying", "unstable (growing)") for t in fig.data)
