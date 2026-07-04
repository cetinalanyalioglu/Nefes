"""Higher-order-mode cut-on frequencies (plane-wave validity ceiling)."""

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.thermo.configure import perfect_gas
from nefes.perturbation import cuton_frequency, duct_cuton_frequencies, ALPHA_CIRCULAR

CFG = perfect_gas(R=287.0, gamma=1.4)
CP = 1.4 * 287.0 / 0.4


def test_circular_cuton_matches_closed_form():
    # First cut-on of a hard-walled circular duct: f = 1.8412 c / (pi d).
    c = 340.0
    d = 0.1
    area = np.pi * (d / 2.0) ** 2
    expected = ALPHA_CIRCULAR * c / (np.pi * d)
    assert cuton_frequency(area, c, 0.0, "circular") == pytest.approx(expected, rel=1e-12)


def test_square_cuton_matches_closed_form():
    c = 340.0
    side = 0.08
    area = side * side
    assert cuton_frequency(area, c, 0.0, "square") == pytest.approx(c / (2.0 * side), rel=1e-12)


def test_rectangular_cuton_uses_the_larger_side():
    c = 340.0
    a, b = 0.12, 0.03  # larger, smaller side
    area, aspect = a * b, a / b
    # first cut-on is a half wavelength across the larger side a
    assert cuton_frequency(area, c, 0.0, "rectangular", aspect) == pytest.approx(c / (2.0 * a), rel=1e-12)
    # aspect 1 collapses back onto the square section
    assert cuton_frequency(area, c, 0.0, "rectangular", 1.0) == pytest.approx(
        cuton_frequency(area, c, 0.0, "square"), rel=1e-12
    )
    # an elongated (aspect > 1) duct cuts on lower than the equal-area square
    assert cuton_frequency(area, c, 0.0, "rectangular", aspect) < cuton_frequency(area, c, 0.0, "square")


def test_rectangular_requires_valid_aspect():
    with pytest.raises(ValueError, match="aspect"):
        cuton_frequency(0.01, 340.0, section="rectangular", aspect=0.5)


def test_flow_lowers_the_ceiling():
    area, c = 0.01, 340.0
    f0 = cuton_frequency(area, c, 0.0)
    f_flow = cuton_frequency(area, c, 0.3)
    assert f_flow == pytest.approx(f0 * np.sqrt(1.0 - 0.09), rel=1e-12)
    assert f_flow < f0


def test_wider_duct_cuts_on_lower():
    c = 340.0
    narrow = cuton_frequency(0.001, c)
    wide = cuton_frequency(0.01, c)
    assert wide < narrow  # larger area -> lower cut-on


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cuton_frequency(-1.0, 340.0)
    with pytest.raises(ValueError):
        cuton_frequency(0.01, 0.0)
    with pytest.raises(ValueError, match="section"):
        cuton_frequency(0.01, 340.0, section="hexagon")


def _stepped_duct_problem(area_wide):
    """A line with a wide middle section (via area changes), so the middle sets the ceiling."""
    a = 0.01
    net = [
        cat.mass_flow_inlet(2.0, 300.0),  # 0
        cat.isentropic_area_change(),  # 1  (a -> wide)
        cat.duct(0.5),  # 2  (the wide section)
        cat.isentropic_area_change(),  # 3  (wide -> a)
        cat.pressure_outlet(101325.0),  # 4
    ]
    edges = [(0, 1, a), (1, 2, area_wide), (2, 3, area_wide), (3, 4, a)]
    return build_problem(CFG, net, edges, mdot_ref=2.0, p_ref=101325.0, h_ref=CP * 300.0)


def test_report_over_a_solved_network():
    prob = _stepped_duct_problem(area_wide=0.04)  # wide edges 1 and 2
    res = solve(prob)
    assert res.converged
    report = duct_cuton_frequencies(prob, res.x)
    assert len(report.ducts) == 4
    # A wide edge (1 or 2) sets the network ceiling, below the narrow edges.
    assert report.limiting.edge in (1, 2)
    assert report.f_cuton == pytest.approx(report.limiting.f_cuton)
    assert report.f_cuton < min(report.ducts[0].f_cuton, report.ducts[3].f_cuton)
    # The repr is a readable table mentioning the ceiling.
    text = repr(report)
    assert "validity ceiling" in text and "Hz" in text
    # Each duct's with-flow ceiling does not exceed its quiescent one.
    for d in report.ducts:
        assert d.f_cuton <= d.f_cuton_quiescent + 1e-9


def test_solution_convenience_method():
    from nefes.shell import Network

    net = Network(gas=CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=2.0, h_ref=CP * 300.0)
    i = net.add(cat.mass_flow_inlet(2.0, 300.0))
    d = net.add(cat.duct(0.5))
    o = net.add(cat.pressure_outlet(101325.0))
    net.connect(i, d, 0.02, name="approach")
    net.connect(d, o, 0.02, name="exit")
    sol = net.solve()
    rep = sol.cuton_report()
    assert len(rep.ducts) == 2
    assert rep.ducts[0].name == "approach"
    assert np.isfinite(rep.f_cuton)

    # a rectangular section is threaded through and reported, lowering the ceiling vs the square default
    rect = sol.cuton_report(section="rectangular", aspect=3.0)
    assert rect.aspect == 3.0 and "rectangular" in repr(rect) and "aspect 3" in repr(rect)
    assert rect.f_cuton < sol.cuton_report(section="square").f_cuton
    # the report renders as HTML (Jupyter) without error, naming the limiting duct
    html = rep._repr_html_()
    assert "Cut-on report" in html and "<table" in html
