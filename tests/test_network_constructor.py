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
from nefes.shell.build import build_problem
from nefes.shell.network import Network
from nefes.solver import solve
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium, perfect_gas

CFG = perfect_gas(287.0, 1.4)
CP = 1.4 * 287.0 / 0.4
MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data", "h2o2.yaml")


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
    prob = build_problem(CFG, nodes, edges, net._seed_mdot(), net.p_ref, net._seed_h())
    func = solve(prob)
    assert np.allclose(one.x, func.x)


def test_ports_auto_assigned_in_attachment_order():
    # a 3-branch splitter: port 0 is the inflow (first attached), then the outflows.
    nodes = [
        cat.total_pressure_inlet(1.1e5, 300.0),
        cat.junction(),
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
    from nefes.thermo import SpeciesSet, Thermo

    lib = SpeciesSet.from_cantera(MECH)
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


def test_marker_gating_matches_explicit_edge_models():
    # The automatic burnt-marker closure (edge_models left unset) reproduces the mean flow of the
    # hand-specified frozen/equilibrium split -- so the explicit knob is an escape hatch, not a
    # requirement, for a reacting network with an equilibrium flame.
    from nefes.thermo import SpeciesSet, Thermo

    gas = Thermo(SpeciesSet.from_cantera(MECH))

    def build(models):
        nodes = [
            cat.total_pressure_inlet(1.2e5, 300.0, composition={"O2": 0.21, "N2": 0.79}, name="air"),
            cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, name="H2"),
            cat.equilibrium_flame(),
            cat.mass_flow_outlet(0.406),
        ]
        edges = [(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01)]
        return Network(gas=equilibrium(gas.mech), p_ref=1e5, T_ref=300.0, nodes=nodes, edges=edges, edge_models=models)

    explicit = build([EQ_FROZEN, EQ_FROZEN, EQ_KERNEL]).solve()
    auto = build(None).solve()  # marker-gated
    assert explicit.converged and auto.converged
    assert auto.marker(2) == pytest.approx(1.0, abs=1e-6)  # burnt edge flagged by the transported marker
    for f in ("T", "M", "p", "mdot"):
        assert np.allclose(auto.field(f), explicit.field(f), rtol=1e-6), f


def test_marker_gating_handles_fresh_dilution_downstream():
    # A fresh (marker-0) stream injected downstream of the flame must not revert the diluted
    # edge to a frozen reactant: the sticky (noisy-OR) marker keeps it burnt, so the auto path
    # re-equilibrates the diluted zone and reproduces the pinned all-equilibrium closure with
    # no edge_models (the staged-combustion / exhaust-gas-recirculation case).
    from nefes.thermo import SpeciesSet, Thermo

    gas = Thermo(SpeciesSet.from_cantera(MECH))

    def build(models):
        nodes = [
            cat.total_pressure_inlet(1.2e5, 300.0, composition={"O2": 0.21, "N2": 0.79}, name="air"),
            cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, name="H2"),
            cat.equilibrium_flame(),
            cat.mass_source(0.4, 300.0, composition={"O2": 0.21, "N2": 0.79}, name="dilution"),  # fresh, marker 0
            cat.mass_flow_outlet(0.806),
        ]
        edges = [(0, 1, 0.02), (1, 2, 0.02), (2, 3, 0.02), (3, 4, 0.02)]
        return Network(gas=equilibrium(gas.mech), p_ref=1e5, T_ref=300.0, nodes=nodes, edges=edges, edge_models=models)

    explicit = build([EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL]).solve()
    auto = build(None).solve()  # marker-gated
    assert explicit.converged and auto.converged
    assert auto.marker(3) == pytest.approx(1.0, abs=1e-4)  # the fresh dilution does not un-burn the edge
    for f in ("T", "M", "p", "mdot"):
        assert np.allclose(auto.field(f), explicit.field(f), rtol=1e-5, atol=1e-5 * np.abs(explicit.field(f)).max()), f


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


def test_edge_tuples_accept_explicit_ports():
    # a 5-tuple edge pins the local ports; the pins thread through to connect() and the compiled problem
    nodes, edges3 = _nodes_edges()
    edges5 = [(0, 1, 0.1, 0, 0), (1, 2, 0.05, 1, 0)]  # (tail, head, area, tail_port, head_port)
    net = Network(gas=CFG, nodes=nodes, edges=edges5)
    assert net._ports == [(0, 0), (1, 0)]
    pinned = net.solve()
    auto = Network(gas=CFG, nodes=nodes, edges=edges3).solve()
    assert pinned.converged and auto.converged
    assert np.allclose(pinned.field("M"), auto.field("M"), rtol=1e-8)


def test_from_yaml_and_from_dict_roundtrip(tmp_path):
    import yaml

    nodes, edges = _nodes_edges()
    net = Network(gas=CFG, nodes=nodes, edges=edges)
    ref = net.solve()
    path = os.path.join(str(tmp_path), "case.yaml")
    net.to_yaml(path)

    from_file = Network.from_yaml(path)
    from_dict = Network.from_dict(yaml.safe_load(open(path).read()))
    for reloaded in (from_file, from_dict):
        assert len(reloaded._elements) == len(nodes)
        assert len(reloaded._edges) == len(edges)
        again = reloaded.solve()
        assert again.converged
        assert np.allclose(again.field("M"), ref.field("M"), rtol=1e-6)


def test_problem_property_caches_and_invalidates():
    nodes, edges = _nodes_edges()
    net = Network(gas=CFG, nodes=nodes, edges=edges)
    first = net.problem
    assert first is net.problem and net._compiled is first  # cached: same object on re-access
    net.add(cat.duct(0.1))  # any topology change drops the cache
    assert net._compiled is None


def test_kind_aware_ports_put_inflow_on_port_zero():
    # Kind-aware auto-assignment claims each edge a direction-matching port: a two-port through
    # element's port 0 is its inflow (orient -1) and port 1 its outflow (orient +1), independent
    # of the order edges were attached.
    nodes = [cat.total_pressure_inlet(1.2e5, 300.0), cat.duct(), cat.pressure_outlet(1.0e5, 300.0)]
    for edges in ([(0, 1, 0.05), (1, 2, 0.05)], [(1, 2, 0.05), (0, 1, 0.05)]):  # forward / reverse-listed
        prob = build_problem(CFG, nodes, edges, 10.0, 1.0e5, CP * 300.0)
        base = int(prob.row_ptr[1])  # the duct node
        assert int(prob.orient[base]) == -1  # port 0 points into the duct
        assert int(prob.orient[base + 1]) == +1  # port 1 out of it
