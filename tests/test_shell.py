"""Phase 5 validation: the Network/Solution shell and YAML connectivity loader."""

import os

import numpy as np
import pytest

from fns.shell import Network
from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.io import load_connectivity, load_case

R_AIR, GAMMA = 287.0, 1.4
CP = GAMMA * R_AIR / (GAMMA - 1.0)

DEMO_YAML = os.path.join(
    os.path.dirname(__file__),
    "..",
    "preliminary-study",
    "docs",
    "examples",
    "ConnectivityDemonstrator.yaml",
)


def test_network_api_solves_nozzle():
    net = Network(perfect_gas(R_AIR, GAMMA), p_ref=101325.0, T_ref=300.0)
    a = net.add(cat.total_pressure_inlet(120000.0, 300.0))
    b = net.add(cat.isentropic_area_change())
    c = net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(a, b, 0.10)
    net.connect(b, c, 0.05)

    sol = net.solve()
    assert sol.converged
    exit_state = sol.edge(1)
    # subsonic exit at the spec pressure, isentropic (uniform total pressure)
    assert exit_state["p"] == pytest.approx(101325.0, rel=1e-6)
    assert exit_state["p_t"] == pytest.approx(120000.0, rel=1e-5)
    assert 0.0 < exit_state["M"] < 1.0


def test_mdot_ref_default_from_inlet():
    net = Network()
    net.add(cat.mass_flow_inlet(7.5, 300.0))
    net.add(cat.pressure_outlet(101325.0))
    net.connect(0, 1, 0.1)
    assert net.mdot_ref == pytest.approx(7.5)


def test_warm_restart_is_cheaper():
    net = Network(perfect_gas(R_AIR, GAMMA))
    net.add(cat.total_pressure_inlet(115000.0, 300.0))
    net.add(cat.isentropic_area_change())
    net.add(cat.pressure_outlet(101325.0, Tt_backflow=300.0))
    net.connect(0, 1, 0.10)
    net.connect(1, 2, 0.06)

    cold = net.solve()
    assert cold.converged
    warm = net.solve(x0=cold.x)
    assert warm.converged
    # warm start from the converged state needs far fewer iterations
    assert warm.iterations < cold.iterations


CASE_YAML = os.path.join(os.path.dirname(__file__), "..", "examples", "converging_nozzle.yaml")


SHOWCASE = os.path.join(os.path.dirname(__file__), "..", "preliminary-study", "examples", "ui_showcase")


@pytest.mark.skipif(not os.path.exists(CASE_YAML), reason="example case not present")
def test_load_case_solves():
    net = load_case(CASE_YAML)
    sol = net.solve()
    assert sol.converged
    throat = sol.edge(2)  # feed(0), pipe(1), throat(2), tailpipe(3)
    # isentropic nozzle + inert constant-area ducts: total pressure uniform
    assert throat["p_t"] == pytest.approx(200000.0, rel=1e-4)
    assert throat["p"] == pytest.approx(150000.0, rel=1e-6)  # subsonic: exit p = spec
    assert 0.0 < throat["M"] < 1.0


@pytest.mark.skipif(not os.path.exists(CASE_YAML), reason="example case not present")
def test_load_case_preserves_ports():
    # The UI export pins ports via the handles: the pipe enters the area change at
    # port 0, the throat leaves it at port 1.  The compiled connectivity must
    # reflect exactly that (port 0 = target/in side, port 1 = source/out side).
    net = load_case(CASE_YAML)
    prob = net.compile()
    # inlet(0) -feed(0)-> Duct(1) -pipe(1)-> nozzle(2) -throat(2)-> Duct(3) -tailpipe(3)-> outlet(4)
    assert list(prob.tail_node) == [0, 1, 2, 3]
    assert list(prob.head_node) == [1, 2, 3, 4]
    # node 2 (the area change) sees edge1 at its port 0 and edge2 at its port 1
    sl = slice(prob.row_ptr[2], prob.row_ptr[3])
    assert list(prob.col_edge[sl]) == [1, 2]
    assert list(prob.orient[sl]) == [-1, 1]  # edge1 incoming, edge2 outgoing


def _contraction_case(tmp_path, sac_attrs):
    """Write a minimal UI-export case: inlet -> sudden contraction -> outlet.

    Edge 0 (area 0.09) enters the element at port 0, edge 1 (area 0.05) leaves it
    at port 1, so the forward flow contracts large -> small.  ``sac_attrs`` are the
    extra attributes carried by the SuddenAreaChange node (e.g. a contraction
    coefficient), exactly as the UI would export them.
    """
    import yaml

    doc = {
        "version": "2.0.0",
        "model": {
            "id": "fns-flow-network",
            "globalAttributes": {
                "gasConstant": 287.0,
                "heatCapacityRatio": 1.4,
                "referencePressure": 101325.0,
                "referenceTemperature": 300.0,
                "referenceMassFlow": 10.0,
            },
            "nodes": [
                {
                    "id": "in",
                    "type": "TotalPressureInlet",
                    "attributes": {"label": "in", "index": 0, "totalPressure": 120000.0, "totalTemperature": 300.0},
                },
                {"id": "sac", "type": "SuddenAreaChange", "attributes": {"label": "sac", "index": 1, **sac_attrs}},
                {
                    "id": "out",
                    "type": "PressureOutlet",
                    "attributes": {"label": "out", "index": 2, "pressure": 101325.0, "backflowTotalTemperature": 300.0},
                },
            ],
            "edges": [
                {
                    "id": "e0",
                    "source": "in",
                    "target": "sac",
                    "sourceHandle": "in-port-0",
                    "targetHandle": "sac-port-0",
                    "type": "flow",
                    "attributes": {"label": "big", "index": 0, "area": 0.09},
                },
                {
                    "id": "e1",
                    "source": "sac",
                    "target": "out",
                    "sourceHandle": "sac-port-1",
                    "targetHandle": "out-port-0",
                    "type": "flow",
                    "attributes": {"label": "small", "index": 1, "area": 0.05},
                },
            ],
        },
    }
    path = tmp_path / f"contraction_{len(sac_attrs)}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_load_case_sudden_contraction_coefficient(tmp_path):
    # The UI exports the contraction coefficient as a node attribute; load_case must
    # thread it to the kernel.  With cc = 0.62 the reverse (large -> small) flow loses
    # total pressure K_c * (1/2 rho u^2)_small; omitting the attribute is loss-free.
    lossy = load_case(_contraction_case(tmp_path, {"contractionCoefficient": 0.62})).solve()
    free = load_case(_contraction_case(tmp_path, {})).solve()  # default cc = 1
    assert lossy.converged and free.converged

    up, dn = lossy.edge(0), lossy.edge(1)  # large (upstream) -> small (downstream)
    assert dn["M"] > up["M"] and dn["M"] < 1.0  # genuinely contracting, subsonic
    K_c = (1.0 / 0.62 - 1.0) ** 2
    q_small = 0.5 * dn["rho"] * dn["u"] ** 2
    assert up["p_t"] - dn["p_t"] == pytest.approx(K_c * q_small, rel=1e-5)

    # the attribute-less case defaults to the historical loss-free contraction
    assert free.edge(0)["p_t"] - free.edge(1)["p_t"] == pytest.approx(0.0, abs=1.0)


@pytest.mark.skipif(not os.path.exists(SHOWCASE), reason="UI showcase cases not present")
def test_load_multiport_showcase_conserves_mass():
    # A real UI export with splitters/junctions (multi-port elements) must load
    # with correct ports and conserve mass at the merge.
    net = load_case(os.path.join(SHOWCASE, "gas_turbine_splits.yaml"))
    sol = net.solve()
    assert sol.converged
    mdot = sol.field("mdot")
    assert np.isfinite(mdot).all()


@pytest.mark.skipif(not os.path.exists(SHOWCASE), reason="UI showcase cases not present")
def test_deferred_supersonic_raises():
    with pytest.raises(ValueError, match="deferred"):
        load_case(os.path.join(SHOWCASE, "cd_nozzle_supersonic.yaml"))


@pytest.mark.skipif(not os.path.exists(DEMO_YAML), reason="demonstrator YAML not present")
def test_demonstrator_yaml_connectivity():
    conn = load_connectivity(DEMO_YAML)
    assert conn.n_nodes == 6
    assert conn.n_edges == 7
    assert list(conn.tail_node) == [0, 1, 1, 2, 2, 3, 4]
    assert list(conn.head_node) == [1, 2, 3, 3, 4, 4, 5]
    assert list(conn.tail_port) == [0, 1, 2, 2, 1, 2, 2]
    assert list(conn.head_port) == [0, 0, 1, 0, 0, 1, 0]
    assert int(conn.row_ptr[-1]) == 2 * conn.n_edges
