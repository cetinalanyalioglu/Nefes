"""Network topology diagram (`plot_network_topology` / `Network.plot_topology`)."""

import pytest

from fns.shell import Network
from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.plotting import plot_network_topology
from fns.plotting.topology import _layers, _positions

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
    fig = net.plot_topology()
    assert len(fig.layout.annotations) == 6  # six directed edges
    # the two parallel ducts (nodes 2, 3) land on the same layer
    x, _y = _positions(len(net._elements), [(t, h) for (t, h, _a) in net._edges])
    assert x[2] == x[3]
    # the merge node (junction) sits past both ducts
    assert x[4] > x[2]


def test_empty_network_is_safe():
    fig = plot_network_topology(Network(gas=CFG))
    assert fig is not None and len(fig.layout.annotations) == 0
