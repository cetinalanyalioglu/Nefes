"""Verification of the UI-case writer (:mod:`fns.io.yaml_out`).

The writer is the symmetric counterpart of the reader: it must emit a document
the UI (and our own :func:`fns.io.load_case`) reads back into the *same* network.
The strongest check is therefore a round-trip -- dump, reload, re-solve, and
compare the converged fields -- exercised for both a network built in Python (a
synthesized layout, no provenance) and a network loaded from the UI export (ids,
handles, positions reused).  A second group checks the result ``data`` section
against the UI's binding rules: one value per element, ordered by element index.
"""

import os
import re

import numpy as np
import pytest
import yaml

from fns.shell import Network
from fns.elements import catalog as cat
from fns.thermo.configure import perfect_gas
from fns.io import load_case, save_case, dump_case, DataItem, DataSet, MetaEntry
from fns.io.yaml_out import SAVE_FILE_VERSION, _FIELD_META
from fns.assembly.derive import ES_C
from fns.perturbation import forced_response, eigenmodes, PerturbationBC

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
_HANDLE_RE = re.compile(r"^.+-port-\d+$")

CFG = perfect_gas(287.0, 1.4)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _nozzle_in_python():
    """[pt inlet] - duct - isentropic area change - duct - [p outlet]."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(200000.0, 300.0, name="reservoir"))
    net.add(cat.duct(0.5, name="feed"))
    net.add(cat.isentropic_area_change(name="nozzle"))
    net.add(cat.duct(0.3, name="tail"))
    net.add(cat.pressure_outlet(150000.0, 300.0, name="back"))
    net.connect(0, 1, 0.02, name="feed")
    net.connect(1, 2, 0.02, name="pipe")
    net.connect(2, 3, 0.01, name="throat")
    net.connect(3, 4, 0.01, name="tailpipe")
    return net


def _junction_in_python():
    """Two inlets feeding a static-pressure junction into one outlet (synthesized)."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.mass_flow_inlet(2.0, 300.0, name="in-a"))
    net.add(cat.mass_flow_inlet(3.0, 300.0, name="in-b"))
    net.add(cat.junction(name="manifold"))
    net.add(cat.pressure_outlet(101325.0, 300.0, name="out"))
    net.connect(0, 2, 0.03)
    net.connect(1, 2, 0.03)
    net.connect(2, 3, 0.05)
    return net


def _reload(net, sol, tmp_path, **kw):
    path = os.path.join(tmp_path, "case.yaml")
    save_case(net, str(path), solution=sol, **kw)
    return load_case(str(path)), yaml.safe_load(open(path).read())


def _single_outlet_net(outlet):
    """[pt inlet] - duct - <outlet>, for the round-trip of a flow-fixing outlet."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=6.0)
    net.add(cat.total_pressure_inlet(160000.0, 300.0, name="res"))
    net.add(cat.duct(0.4, name="pipe"))
    net.add(outlet)
    net.connect(0, 1, 0.02)
    net.connect(1, 2, 0.02)
    return net


# --------------------------------------------------------------------------
# 1. Round-trip preserves the physics (synthesized and provenance paths).
# --------------------------------------------------------------------------
def test_synthesized_roundtrip_resolves(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    assert sol.converged
    net2, _ = _reload(net, sol, str(tmp_path))
    sol2 = net2.solve()
    assert sol2.converged
    for f in _FIELD_META:
        assert np.allclose(sol.field(f), sol2.field(f), rtol=1e-9, atol=1e-9), f


def test_synthesized_manifold_roundtrip_resolves(tmp_path):
    net = _junction_in_python()
    sol = net.solve()
    assert sol.converged
    net2, doc = _reload(net, sol, str(tmp_path))
    # Junction carries the UI dynamic-port counts (2 in, 1 out).
    jct = next(n for n in doc["model"]["nodes"] if n["type"] == "JunctionStaticP")
    assert jct["attributes"]["leftPorts"] == 2
    assert jct["attributes"]["rightPorts"] == 1
    sol2 = net2.solve()
    assert sol2.converged
    assert np.allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    "outlet, ui_type",
    [
        (cat.mass_flow_outlet(5.0, name="out"), "MassFlowOutlet"),
        (cat.choked_nozzle_outlet(0.012, name="out"), "ChokedNozzleOutlet"),
    ],
)
def test_outlet_elements_roundtrip(tmp_path, outlet, ui_type):
    """The two new outflow boundaries survive the UI-format round-trip and re-solve."""
    net = _single_outlet_net(outlet)
    sol = net.solve()
    assert sol.converged
    net2, doc = _reload(net, sol, str(tmp_path))
    assert any(n["type"] == ui_type for n in doc["model"]["nodes"]), f"{ui_type} not serialized"
    sol2 = net2.solve()
    assert sol2.converged
    assert np.allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-9, atol=1e-9)


def test_storage_elements_roundtrip(tmp_path):
    """Cavity, LinearResistance, and the storage lengths / manifold volume survive the
    UI round-trip: the elements re-build and their storage block M is reproduced."""
    from fns.perturbation import build_acoustic_blocks

    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=1.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, name="src"))
    net.add(cat.isentropic_area_change(name="diffuser", l_up=0.03, l_down=0.02, end_correction=0.005))
    net.add(cat.linear_resistance(40.0, name="screen", l_up=0.01, end_correction=0.004))
    net.add(cat.junction(name="plenum", volume=2.0e-3, neck_length=0.015))
    net.add(cat.duct(0.02, name="neck"))
    net.add(cat.cavity(1.0e-3, name="cav"))
    net.add(cat.pressure_outlet(101325.0, 300.0, name="back"))
    net.connect(0, 1, 3.0e-3)
    net.connect(1, 2, 2.0e-3)
    net.connect(2, 3, 2.0e-3)
    net.connect(3, 6, 2.0e-3)  # plenum -> outlet
    net.connect(3, 4, 5.0e-4)  # plenum -> neck
    net.connect(4, 5, 5.0e-4)  # neck -> cavity
    sol = net.solve()
    assert sol.converged
    net2, doc = _reload(net, sol, str(tmp_path))

    types = {n["type"] for n in doc["model"]["nodes"]}
    assert {"Cavity", "LinearResistance", "JunctionStaticP", "IsentropicAreaChange"} <= types
    by_type = {n["type"]: n["attributes"] for n in doc["model"]["nodes"]}
    assert by_type["Cavity"]["volume"] == pytest.approx(1.0e-3)
    assert by_type["JunctionStaticP"]["volume"] == pytest.approx(2.0e-3)
    assert by_type["JunctionStaticP"]["neck_length"] == pytest.approx(0.015)
    assert by_type["IsentropicAreaChange"]["lengthUpstream"] == pytest.approx(0.03)
    assert by_type["IsentropicAreaChange"]["endCorrection"] == pytest.approx(0.005)
    assert by_type["LinearResistance"]["resistance"] == pytest.approx(40.0)

    sol2 = net2.solve()
    assert sol2.converged
    assert np.allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-9, atol=1e-9)
    # the storage block is identical after the round-trip (same nnz and entries)
    M1 = build_acoustic_blocks(sol.problem, sol.x).M
    M2 = build_acoustic_blocks(sol2.problem, sol2.x).M
    assert M1.nnz == M2.nnz and M1.nnz > 0
    assert np.allclose(np.sort_complex(M1.tocoo().data), np.sort_complex(M2.tocoo().data))


@pytest.mark.parametrize("name", ["converging_nozzle.yaml", "gas_turbine_large.yaml"])
def test_provenance_roundtrip_resolves(tmp_path, name):
    net = load_case(os.path.join(_EXAMPLES, name))
    sol = net.solve()
    assert sol.converged
    net2, doc = _reload(net, sol, str(tmp_path))
    sol2 = net2.solve()
    assert sol2.converged
    for f in ("mdot", "p", "T", "M", "p_t"):
        assert np.allclose(sol.field(f), sol2.field(f), rtol=1e-9, atol=1e-9), f


def test_provenance_preserves_ids_and_handles(tmp_path):
    src = os.path.join(_EXAMPLES, "converging_nozzle.yaml")
    orig = yaml.safe_load(open(src).read())
    net = load_case(src)
    _, doc = _reload(net, net.solve(), str(tmp_path))
    assert [n["id"] for n in doc["model"]["nodes"]] == [n["id"] for n in orig["model"]["nodes"]]
    by_id = {e["id"]: e for e in orig["model"]["edges"]}
    for e in doc["model"]["edges"]:
        assert e["sourceHandle"] == by_id[e["id"]]["sourceHandle"]
        assert e["targetHandle"] == by_id[e["id"]]["targetHandle"]


# --------------------------------------------------------------------------
# 2. The emitted document obeys the UI schema / binding rules.
# --------------------------------------------------------------------------
def test_ui_schema_invariants(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    _, doc = _reload(net, sol, str(tmp_path), node_data=True)

    assert doc["version"] == SAVE_FILE_VERSION
    assert doc["model"]["id"] == "fns-flow-network"

    n_nodes, n_edges = len(net._elements), len(net._edges)
    # Indices are dense 0..N-1 (values[i] binds to the element with index i).
    assert sorted(n["attributes"]["index"] for n in doc["model"]["nodes"]) == list(range(n_nodes))
    assert sorted(e["attributes"]["index"] for e in doc["model"]["edges"]) == list(range(n_edges))
    # Handles are well-formed and edge areas positive.
    for e in doc["model"]["edges"]:
        assert _HANDLE_RE.match(e["sourceHandle"]) and _HANDLE_RE.match(e["targetHandle"])
        assert e["attributes"]["area"] > 0.0
    # uiAttributes covers every node.
    assert {u["id"] for u in doc["uiAttributes"]["nodes"]} == {n["id"] for n in doc["model"]["nodes"]}
    # Every dataset item supplies exactly one value per element of its target.
    for ds in doc["data"]["datasets"]:
        for it in ds["items"]:
            want = n_nodes if it["target"] == "node" else n_edges
            assert len(it["values"]) == want, (it["name"], it["target"])


def test_edge_values_match_solution(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    _, doc = _reload(net, sol, str(tmp_path))
    items = {it["name"]: it for it in doc["data"]["datasets"][0]["items"]}
    for f, (label, _unit) in _FIELD_META.items():
        assert np.allclose(items[label]["values"], sol.field(f)), label


def test_node_data_is_incident_edge_mean(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    _, doc = _reload(net, sol, str(tmp_path), node_data=True)
    items = [it for it in doc["data"]["datasets"][0]["items"] if it["name"] == "Static pressure"]
    edge_item = next(it for it in items if it["target"] == "edge")
    node_item = next(it for it in items if it["target"] == "node")
    assert len(node_item["values"]) == len(net._elements)
    # The inlet (node 0) touches only edge 0, so its node value equals that edge value.
    assert np.isclose(node_item["values"][0], edge_item["values"][0])
    # An interior node touches two edges -> the mean of the two.
    assert np.isclose(node_item["values"][1], np.mean([edge_item["values"][0], edge_item["values"][1]]))


def test_field_selection(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    _, doc = _reload(net, sol, str(tmp_path), fields=["p", "mdot"])
    names = {it["name"] for it in doc["data"]["datasets"][0]["items"]}
    assert names == {"Static pressure", "Mass flow"}


def test_unknown_field_rejected():
    net = _nozzle_in_python()
    sol = net.solve()
    with pytest.raises(ValueError, match="unknown mean-flow field"):
        dump_case(net, solution=sol, fields=["not_a_field"])


# --------------------------------------------------------------------------
# 3. Forced-response (acoustic) frequency snapshots.
# --------------------------------------------------------------------------
def test_forced_response_snapshots(tmp_path):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(104000.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.open_end()))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    freqs = np.array([100.0, 250.0, 500.0])
    fr = forced_response(sol.problem, sol.x, freqs)

    text = dump_case(net, solution=sol, forced=fr, forced_freqs=[100.0, 500.0], forced_fields=("p", "u"))
    doc = yaml.safe_load(text)
    names = [d["name"] for d in doc["data"]["datasets"]]
    assert names == ["Mean flow", "100 Hz", "500 Hz"]

    snap = next(d for d in doc["data"]["datasets"] if d["name"] == "100 Hz")
    item_names = [it["name"] for it in snap["items"]]
    assert item_names == ["Pressure amplitude", "Pressure phase", "Velocity amplitude", "Velocity phase"]
    n_edges = len(net._edges)
    assert all(len(it["values"]) == n_edges for it in snap["items"])
    # Magnitudes equal |p'| from the forced response at that frequency.
    j = int(np.argmin(np.abs(np.asarray(fr.freqs) - 100.0)))
    expected = [abs(complex(fr.field(e, "network")[j, 1])) for e in range(n_edges)]
    p_amp = next(it for it in snap["items"] if it["name"] == "Pressure amplitude")
    assert np.allclose(p_amp["values"], expected)


def test_forced_unknown_frequency_rejected(tmp_path):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(104000.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.open_end()))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    fr = forced_response(sol.problem, sol.x, np.array([100.0, 200.0]))
    with pytest.raises(ValueError, match="not in the forced response"):
        dump_case(net, forced=fr, forced_freqs=[123.0])


def _resonator():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, perturbation_bc=PerturbationBC.hard_wall()))
    net.add(cat.duct(0.5))
    net.add(cat.wall())
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    c = float(sol.table()[ES_C, 0])
    res = eigenmodes(sol.problem, sol.x, (0.4 * c / (2 * 0.5), 1.6 * c / (2 * 0.5)))
    return net, sol, res


def test_eigenmode_snapshots(tmp_path):
    net, sol, res = _resonator()
    assert res.n_modes >= 1

    text = dump_case(net, solution=sol, eigenmodes=res, eig_fields=("p", "u"))
    doc = yaml.safe_load(text)
    names = [d["name"] for d in doc["data"]["datasets"]]
    assert names[0] == "Mean flow"
    mode_dsets = [d for d in doc["data"]["datasets"] if d["name"].startswith("Mode ")]
    assert len(mode_dsets) == res.n_modes

    ds0 = mode_dsets[0]
    item_names = [it["name"] for it in ds0["items"]]
    assert item_names == ["Pressure amplitude", "Pressure phase", "Velocity amplitude", "Velocity phase"]
    n_edges = len(net._edges)
    assert all(len(it["values"]) == n_edges for it in ds0["items"])

    info = _info_map(ds0)
    assert info["kind"]["value"] == "Eigenmode"
    assert info["frequency"]["value"] == pytest.approx(float(res.freqs[0]), rel=1e-6)
    assert info["growth_rate"]["value"] == pytest.approx(float(res.growth_rates[0]), rel=1e-6, abs=1e-9)
    assert "unstable" in info and "residual" in info
    # the per-edge pressure magnitude matches the mode shape (network basis, component 1)
    expected = [abs(complex(res.mode_shape(0, "network")[e, 1])) for e in range(n_edges)]
    p_amp = next(it for it in ds0["items"] if it["name"] == "Pressure amplitude")
    assert np.allclose(p_amp["values"], expected)


def test_eigenmode_subset_and_bad_index(tmp_path):
    net, sol, res = _resonator()
    text = dump_case(net, eigenmodes=res, eig_modes=[0])
    doc = yaml.safe_load(text)
    assert [d["name"] for d in doc["data"]["datasets"]] == [f"Mode 0: {float(res.freqs[0]):g} Hz"]
    with pytest.raises(ValueError, match="out of range"):
        dump_case(net, eigenmodes=res, eig_modes=[res.n_modes])


# --------------------------------------------------------------------------
# 4. Custom datasets and the convenience methods.
# --------------------------------------------------------------------------
def test_extra_datasets_passthrough(tmp_path):
    net = _nozzle_in_python()
    n_nodes = len(net._elements)
    nodal = DataSet("Custom nodal", [DataItem("Score", "node", list(range(n_nodes)), "-")])
    text = dump_case(net, extra_datasets=[nodal])
    doc = yaml.safe_load(text)
    ds = next(d for d in doc["data"]["datasets"] if d["name"] == "Custom nodal")
    assert ds["items"][0]["target"] == "node"
    assert ds["items"][0]["values"] == list(range(n_nodes))


def _info_map(dataset_doc):
    return {e["key"]: e for e in dataset_doc.get("info", [])}


def test_mean_flow_dataset_has_minimal_metadata(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    _, doc = _reload(net, sol, str(tmp_path))
    mean = doc["data"]["datasets"][0]
    info = _info_map(mean)
    # Minimal, self-describing entries: each carries key + label + value.
    assert info["kind"]["value"] == "Mean flow"
    assert info["converged"]["value"] is True
    assert all({"key", "label", "value"} <= set(e) for e in mean["info"])


def test_forced_snapshot_dataset_metadata(tmp_path):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(104000.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.open_end()))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    fr = forced_response(sol.problem, sol.x, np.array([100.0, 250.0]))
    doc = yaml.safe_load(dump_case(net, forced=fr, forced_freqs=[250.0]))
    snap = next(d for d in doc["data"]["datasets"] if d["name"] == "250 Hz")
    info = _info_map(snap)
    assert info["kind"]["value"] == "Forced response"
    assert info["frequency"]["value"] == 250.0
    assert info["frequency"]["unit"] == "Hz"


def test_custom_dataset_metadata_roundtrips(tmp_path):
    net = _nozzle_in_python()
    n_nodes = len(net._elements)
    ds = DataSet(
        "Custom",
        [DataItem("Score", "node", list(range(n_nodes)), "-")],
        description="A hand-authored dataset",
        info=[MetaEntry("origin", "Origin", "unit test", description="where it came from")],
    )
    doc = yaml.safe_load(dump_case(net, extra_datasets=[ds]))
    out = next(d for d in doc["data"]["datasets"] if d["name"] == "Custom")
    assert out["description"] == "A hand-authored dataset"
    entry = out["info"][0]
    assert entry == {"key": "origin", "label": "Origin", "value": "unit test", "description": "where it came from"}
    # Empty unit/description are omitted to keep the file clean.
    assert "unit" not in entry


def test_network_and_solution_save_methods(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    p_case = os.path.join(str(tmp_path), "net.yaml")
    p_sol = os.path.join(str(tmp_path), "sol.yaml")
    net.save(p_case)
    sol.save(p_sol)
    # Network.save writes no result data; Solution.save embeds it.
    assert "data" not in yaml.safe_load(open(p_case).read())
    assert "data" in yaml.safe_load(open(p_sol).read())
