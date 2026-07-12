"""Network topology diagram (`plot_network_topology` / `Network.plot` / `Solution.plot`)."""

import pytest

from nefes.elements import catalog as cat
from nefes.plotting import plot_network_topology
from nefes.plotting.topology import _layers, _positions
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(R=287.0, gamma=1.4)


def _linear_net():
    net = Network(gas=CFG)
    i = net.add(cat.mass_flow_inlet(2.0, 300.0))
    d = net.add(cat.duct(0.5))
    o = net.add(cat.pressure_outlet(101325.0))
    net.connect(i, d, 0.02, name="approach")
    net.connect(d, o, 0.02, name="exit")
    return net


def _branched_net():
    """inlet -> splitter -> two ducts -> junction -> outlet."""
    net = Network(gas=CFG)
    i = net.add(cat.mass_flow_inlet(2.0, 300.0))
    s = net.add(cat.splitter())
    d1 = net.add(cat.duct(0.5))
    d2 = net.add(cat.duct(0.5))
    j = net.add(cat.junction())
    o = net.add(cat.pressure_outlet(101325.0))
    net.connect(i, s, 0.02)
    net.connect(s, d1, 0.01)
    net.connect(s, d2, 0.01)
    net.connect(d1, j, 0.01)
    net.connect(d2, j, 0.01)
    net.connect(j, o, 0.02)
    return net


def test_layers_are_longest_path_ranks():
    # 0 -> 1 -> 2 with a skip 0 -> 2: node 2 must sit past node 1 (longest path).
    layer = _layers(3, [(0, 1), (1, 2), (0, 2)])
    assert layer == [0, 1, 2]


def test_layers_terminate_on_a_cycle():
    layer = _layers(3, [(0, 1), (1, 2), (2, 0)])  # a 3-cycle
    assert len(layer) == 3 and all(0 <= v <= 2 for v in layer)


def test_positions_spread_within_a_layer():
    # A splitter feeding two parallel nodes: the two share a layer, centered on 0.
    x, y = _positions(4, [(0, 1), (1, 2), (1, 3)])
    assert x[2] == x[3]  # same layer (x rank)
    assert y[2] != y[3]  # spread vertically
    assert pytest.approx(y[2] + y[3], abs=1e-12) == 0.0  # centered


def test_linear_figure_has_nodes_and_edge_arrows():
    fig = plot_network_topology(_linear_net())
    # one node trace per role: inlet, interior (duct), outlet -> 3 roles + 1 edge-midpoint trace
    node_traces = [t for t in fig.data if t.name in ("inlet", "outlet", "interior", "source", "flame", "wall")]
    assert len(node_traces) == 3
    # every directed edge becomes an arrow annotation
    assert len(fig.layout.annotations) == 2
    # node labels carry "index: name"
    all_text = [s for t in node_traces for s in (t.text or ())]
    assert any(s.startswith("0:") for s in all_text)


def test_branched_topology_layers_branch_and_merge():
    net = _branched_net()
    fig = net.plot()
    assert len(fig.layout.annotations) == 6  # six directed edges
    # the two parallel ducts (nodes 2, 3) land on the same layer
    x, _y = _positions(len(net._elements), [(t, h) for (t, h, _a) in net._edges])
    assert x[2] == x[3]
    # the merge node (junction) sits past both ducts
    assert x[4] > x[2]


def test_empty_network_is_safe():
    fig = plot_network_topology(Network(gas=CFG))
    assert fig is not None and len(fig.layout.annotations) == 0


# -- solution overlay (color_by / width_by) ---------------------------------


def _solved_linear_net():
    net = Network(gas=CFG, p_ref=101325.0, T_ref=300.0)
    i = net.add(cat.total_pressure_inlet(150000.0, 320.0))
    d = net.add(cat.duct(0.5))
    o = net.add(cat.pressure_outlet(101325.0))
    net.connect(i, d, 0.02, name="approach")
    net.connect(d, o, 0.02, name="exit")
    sol = net.solve()
    assert sol.converged
    return net, sol


def test_color_by_adds_colorbar_and_tints_edges():
    net, sol = _solved_linear_net()
    fig = net.plot(solution=sol, color_by="M")
    # the edge-midpoint trace carries the colored field + a colorbar
    edge_trace = next(t for t in fig.data if t.name == "edges")
    assert edge_trace.marker.showscale is True
    assert tuple(edge_trace.marker.color) == pytest.approx(tuple(sol.field("M")))
    # each edge arrow is individually colored (not the default grey)
    arrow_colors = {ann.arrowcolor for ann in fig.layout.annotations}
    assert "#9aa5b1" not in arrow_colors
    # the default title names the overlaid field
    assert "Mach" in fig.layout.title.text


def test_width_by_scales_arrow_widths():
    net, sol = _solved_linear_net()
    fig = net.plot(solution=sol, width_by="mdot")
    widths = [ann.arrowwidth for ann in fig.layout.annotations]
    assert all(w > 0 for w in widths)  # all scaled from |mdot|


def test_overlay_requires_solution():
    net, _ = _solved_linear_net()
    with pytest.raises(ValueError, match="need a converged"):
        net.plot(color_by="T")


def test_unknown_field_rejected():
    net, sol = _solved_linear_net()
    with pytest.raises(ValueError, match="not a known edge field"):
        net.plot(solution=sol, color_by="entropy")


def test_width_by_area_needs_no_solution():
    # area lives on the network, so a geometry-weighted width works with no solve
    net = _branched_net()
    fig = plot_network_topology(net, width_by="area")
    widths = [round(ann.arrowwidth, 6) for ann in fig.layout.annotations]
    assert all(w > 0 for w in widths)
    assert len(set(widths)) > 1  # the wider inlet/outlet edges get thicker arrows than the split legs


def test_network_plot_defaults_to_area_width():
    net = _branched_net()
    # Network.plot() defaults width_by="area" -> identical to asking for it explicitly
    got = [a.arrowwidth for a in net.plot().layout.annotations]
    ref = [a.arrowwidth for a in plot_network_topology(net, width_by="area").layout.annotations]
    assert got == ref
    # the default is overridable: width_by=None restores uniform arrows
    assert [a.arrowwidth for a in net.plot(width_by=None).layout.annotations] == pytest.approx([1.4] * 6)


def test_network_plot_area_default_is_safe_on_empty_network():
    fig = Network(gas=CFG).plot()  # the area-width default must not choke on an edgeless network
    assert fig is not None and len(fig.layout.annotations) == 0


def test_no_solution_is_structural_view():
    # the default (no solution) path is unchanged: grey arrows, index labels, no colorbar
    fig = plot_network_topology(_linear_net())
    assert all(ann.arrowcolor == "#9aa5b1" for ann in fig.layout.annotations)
    edge_trace = next(t for t in fig.data if t.name == "edges")
    assert edge_trace.marker.showscale in (None, False)
    assert fig.layout.title.text == "Network topology"


def test_solution_plot_shares_backend_and_overlays():
    # Solution.plot draws the same diagram as Network.plot, with this solution attached
    net, sol = _solved_linear_net()
    fig = sol.plot(color_by="M", width_by="mdot")
    edge_trace = next(t for t in fig.data if t.name == "edges")
    assert edge_trace.marker.showscale is True
    assert tuple(edge_trace.marker.color) == pytest.approx(tuple(sol.field("M")))
    assert "Mach" in fig.layout.title.text
    # a bare sol.plot() is the solved-network structural view (no colorbar, enriched hover)
    plain = sol.plot()
    assert next(t for t in plain.data if t.name == "edges").marker.showscale in (None, False)


def test_solution_plot_rejects_unknown_field():
    _net, sol = _solved_linear_net()
    with pytest.raises(ValueError, match="not a known edge field"):
        sol.plot(color_by="entropy")
