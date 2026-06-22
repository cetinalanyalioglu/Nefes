"""Network input verification (catalog.validate_network).

The compiler rejects ill-posed networks up front: non-positive areas, wrong port
arity, and -- the headline check -- an area change across an element that does
not permit one.  Area changes are physically carried by the dedicated
``isentropic_area_change`` / ``sudden_area_change`` elements; the constant-area
elements (duct, concentrated loss) must share one area across both ports.
"""

import numpy as np
import pytest

from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)
CP = 1.4 * 287.0 / 0.4
H_REF = CP * 300.0


def _build(mid, A0, A1, name="mid"):
    """Inlet -> mid -> outlet with the two interior edges at areas A0, A1."""
    net = [cat.total_pressure_inlet(130000.0, 300.0), mid, cat.pressure_outlet(101325.0, 300.0)]
    return cat.build_problem(CFG, net, [(0, 1, A0), (1, 2, A1)], 10.0, 101325.0, H_REF)


# -- area-change consistency ------------------------------------------------


def test_duct_area_change_rejected():
    with pytest.raises(ValueError, match="area change"):
        _build(cat.duct(0.5, name="tailpipe"), 0.02, 0.01)


def test_loss_area_change_rejected():
    with pytest.raises(ValueError, match="area change"):
        _build(cat.loss(2.5, name="valve"), 0.08, 0.05)


def test_error_message_names_the_element():
    with pytest.raises(ValueError, match="tailpipe"):
        _build(cat.duct(0.5, name="tailpipe"), 0.02, 0.01)


def test_isentropic_area_change_allows_jump():
    prob = _build(cat.isentropic_area_change(), 0.10, 0.06)
    assert prob.n_nodes == 3


def test_sudden_area_change_allows_jump():
    prob = _build(cat.sudden_area_change(), 0.05, 0.09)
    assert prob.n_nodes == 3


def test_equal_area_duct_and_loss_ok():
    assert _build(cat.duct(1.0), 0.06, 0.06).n_nodes == 3
    assert _build(cat.loss(2.5), 0.08, 0.08).n_nodes == 3


def test_junction_and_splitter_allow_different_areas():
    # inlet + two outlets fed by a 3-port junction whose legs differ in area
    net = [
        cat.total_pressure_inlet(200000.0, 300.0),
        cat.junction(),
        cat.pressure_outlet(120000.0, 300.0),
        cat.pressure_outlet(110000.0, 300.0),
    ]
    edges = [(0, 1, 0.03), (1, 2, 0.01), (1, 3, 0.02)]
    prob = cat.build_problem(CFG, net, edges, 10.0, 101325.0, H_REF)
    assert prob.n_nodes == 4


# -- positive areas ---------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.01, np.inf, np.nan])
def test_nonpositive_or_nonfinite_area_rejected(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        _build(cat.isentropic_area_change(), 0.10, bad)


# -- port arity -------------------------------------------------------------


def test_two_port_element_with_one_edge_rejected():
    # an isentropic area change wired with a single incident edge
    net = [cat.total_pressure_inlet(130000.0, 300.0), cat.isentropic_area_change()]
    with pytest.raises(ValueError, match="expects 2 port"):
        cat.build_problem(CFG, net, [(0, 1, 0.02)], 10.0, 101325.0, H_REF)


def test_boundary_with_two_edges_rejected():
    # a pressure outlet (single-port) wired to two edges
    net = [cat.total_pressure_inlet(200000.0, 300.0), cat.junction(), cat.pressure_outlet(120000.0, 300.0)]
    edges = [(0, 1, 0.02), (1, 2, 0.02), (2, 1, 0.02)]
    with pytest.raises(ValueError, match="expects 1 port"):
        cat.build_problem(CFG, net, edges, 10.0, 101325.0, H_REF)


def test_junction_needs_two_ports():
    net = [cat.total_pressure_inlet(130000.0, 300.0), cat.junction()]
    with pytest.raises(ValueError, match=">= 2 port"):
        cat.build_problem(CFG, net, [(0, 1, 0.02)], 10.0, 101325.0, H_REF)
