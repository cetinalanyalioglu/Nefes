"""One-shot Network construction from node / edge lists.

The Network can be built incrementally (``add`` / ``connect``) or fully specified at
construction with ``nodes`` and ``edges`` -- the convenient form that supersedes the
lower-level ``build_problem``.  The one-shot path must be equivalent (same ports, same
solution) and carry the per-edge thermo-model override.
"""

import os

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas, equilibrium
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.shell.network import Network
from nefes.solver import solve

CFG = perfect_gas(287.0, 1.4)
CP = 1.4 * 287.0 / 0.4
MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")


def _nodes_edges():
    nodes = [cat.total_pressure_inlet(1.2e5, 300.0), cat.isentropic_area_change(), cat.pressure_outlet(1.0e5, 300.0)]
    edges = [(0, 1, 0.1), (1, 2, 0.05)]
    return nodes, edges


def test_one_shot_matches_incremental():
    nodes, edges = _nodes_edges()
    one = Network(gas=CFG, nodes=nodes, edges=edges).solve()

    inc = Network(gas=CFG)
    for n in nodes:
        inc.add(n)
    for t, h, a in edges:
        inc.connect(t, h, a)
    inc = inc.solve()

    assert one.converged and inc.converged
    assert np.allclose(one.x, inc.x)


def test_one_shot_matches_build_problem():
    nodes, edges = _nodes_edges()
    net = Network(gas=CFG, nodes=nodes, edges=edges)
    one = net.solve()
    prob = cat.build_problem(CFG, nodes, edges, net.mdot_ref, net.p_ref, net.h_ref)
    func = solve(prob)
    assert np.allclose(one.x, func.x)


def test_ports_auto_assigned_in_attachment_order():
    # a 3-branch splitter: port 0 is the inflow (first attached), then the outflows.
    nodes = [
        cat.total_pressure_inlet(1.1e5, 300.0),
        cat.splitter(),
        cat.pressure_outlet(1.0e5, 300.0),
        cat.pressure_outlet(1.0e5, 300.0),
    ]
    edges = [(0, 1, 0.2), (1, 2, 0.1), (1, 3, 0.1)]  # wider inflow keeps it comfortably subsonic
    sol = Network(gas=CFG, nodes=nodes, edges=edges).solve()
    assert sol.converged
    # mass splits symmetrically across the two equal outlets
    mdot = sol.field("mdot")
    assert mdot[1] == pytest.approx(mdot[2], rel=1e-6)
    assert mdot[1] + mdot[2] == pytest.approx(mdot[0], rel=1e-9)


def test_edge_models_passthrough():
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_native(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    Y[idx["O2"]], Y[idx["N2"]] = 0.21, 0.79
    Y /= Y.sum()
    h_air = gas.enthalpy_mass(Y, 300.0)

    nodes = [
        cat.total_pressure_inlet(1.2e5, 300.0, composition={"O2": 0.21, "N2": 0.79}, name="air"),
        cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, name="H2"),
        cat.equilibrium_flame(),
        cat.mass_flow_outlet(0.406),
    ]
    edges = [(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01)]
    models = [EQ_FROZEN, EQ_FROZEN, EQ_KERNEL]
    net = Network(
        gas=equilibrium(gas.mech),
        p_ref=1e5,
        T_ref=300.0,
        mdot_ref=0.4,
        h_ref=h_air,
        nodes=nodes,
        edges=edges,
        edge_models=models,
    )
    prob = net.compile()
    assert prob.edge_model.tolist() == models
    sol = net.solve()
    assert sol.converged
    assert sol.field("T")[-1] > 1200.0  # flame ignited on the burnt edge


def test_validation():
    nodes, edges = _nodes_edges()
    with pytest.raises(ValueError, match="edge_models has 1 entries"):
        Network(gas=CFG, nodes=nodes, edges=edges, edge_models=[EQ_FROZEN])
    with pytest.raises(ValueError, match="edge_models was given without edges"):
        Network(gas=CFG, nodes=nodes, edge_models=[EQ_FROZEN])


def test_backward_compatible_empty_constructor():
    # the incremental API is unchanged: no nodes/edges -> an empty network to build up.
    net = Network(gas=CFG, p_ref=1e5, T_ref=300.0)
    assert net._elements == [] and net._edges == []
