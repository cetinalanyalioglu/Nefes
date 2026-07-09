"""Verification of the UI-case writer (:mod:`nefes.io.yaml_out`).

The writer is the symmetric counterpart of the reader: it must emit a document
the UI (and our own :func:`nefes.io.load_case`) reads back into the *same* network.
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

from nefes.shell import Network
from nefes.elements import catalog as cat
from nefes.thermo.configure import perfect_gas
from nefes.io import load_case, save_case, dump_case, DataItem, DataSet, MetaEntry
from nefes.io.yaml_out import SAVE_FILE_VERSION, _FIELD_META
from nefes.assembly.recover import ES_C
from nefes.perturbation import forced_response, eigenmodes, PerturbationBC

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
    from nefes.perturbation import build_acoustic_blocks

    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=1.0)
    net.add(cat.total_pressure_inlet(101325.0, 300.0, name="src"))
    net.add(cat.isentropic_area_change(name="diffuser", l_up=0.03, l_down=0.02, end_correction=0.005))
    net.add(cat.linear_resistance(40.0, name="screen", l_up=0.01, end_correction=0.004))
    net.add(cat.junction(name="plenum", volume=2.0e-3))
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


@pytest.mark.parametrize("name", ["getting-started/converging_nozzle.yaml", "flow/gas_turbine_large.yaml"])
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
    src = os.path.join(_EXAMPLES, "getting-started", "converging_nozzle.yaml")
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


def test_network_and_solution_to_yaml_methods(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    p_case = os.path.join(str(tmp_path), "net.yaml")
    p_sol = os.path.join(str(tmp_path), "sol.yaml")
    net.to_yaml(p_case)
    sol.to_yaml(p_sol)
    # Network.to_yaml writes no result data; Solution.to_yaml embeds it.
    assert "data" not in yaml.safe_load(open(p_case).read())
    assert "data" in yaml.safe_load(open(p_sol).read())


def test_save_is_an_alias_for_to_yaml(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    a, b = os.path.join(str(tmp_path), "a.yaml"), os.path.join(str(tmp_path), "b.yaml")
    net.save(a)  # alias
    net.to_yaml(b)  # canonical
    # both write a network-only case (topology, no result data)
    da, db = yaml.safe_load(open(a).read()), yaml.safe_load(open(b).read())
    assert da["model"]["nodes"] and "data" not in da
    assert da["model"]["nodes"] == db["model"]["nodes"]
    # Solution.save likewise mirrors Solution.to_yaml (embeds the mean-flow data)
    p = os.path.join(str(tmp_path), "sol.yaml")
    sol.save(p)
    assert "data" in yaml.safe_load(open(p).read())


def test_solution_to_yaml_appends_named_datasets(tmp_path):
    net = _nozzle_in_python()
    sol = net.solve()
    p = os.path.join(str(tmp_path), "multi.yaml")
    sol.to_yaml(p, dataset="Operating point A")
    sol.to_yaml(p, dataset="Operating point B")  # appended to the same file
    names = [d["name"] for d in yaml.safe_load(open(p).read())["data"]["datasets"]]
    assert "Operating point A" in names and "Operating point B" in names
    # A repeated dataset name is rejected rather than silently overwritten.
    with pytest.raises(ValueError, match="already exists"):
        sol.to_yaml(p, dataset="Operating point A")


# --------------------------------------------------------------------------
# 5. Animated datasets (frames axis) and the Nyquist summary.
# --------------------------------------------------------------------------
def _forced_net():
    net = Network(CFG, p_ref=101325.0, T_ref=300.0, mdot_ref=5.0)
    net.add(cat.total_pressure_inlet(104000.0, 300.0, perturbation_bc=PerturbationBC.anechoic(driven=("acoustic",))))
    net.add(cat.duct(0.5))
    net.add(cat.pressure_outlet(101325.0, 300.0, perturbation_bc=PerturbationBC.open_end()))
    net.connect(0, 1, 0.05)
    net.connect(1, 2, 0.05)
    sol = net.solve()
    assert sol.converged
    return net, sol


def test_forced_sweep_animated_dataset():
    net, sol = _forced_net()
    freqs = np.array([100.0, 250.0, 500.0])
    fr = forced_response(sol.problem, sol.x, freqs)
    doc = yaml.safe_load(dump_case(net, forced=fr, forced_sweep=True, forced_fields=("p", "u")))
    (ds,) = doc["data"]["datasets"]
    assert ds["name"] == "Frequency sweep"
    assert ds["frames"] == {"variable": "Frequency", "unit": "Hz", "values": freqs.tolist()}
    n_edges = len(net._edges)
    item_names = [it["name"] for it in ds["items"]]
    assert item_names == ["Pressure amplitude", "Pressure phase", "Velocity amplitude", "Velocity phase"]
    for it in ds["items"]:
        assert len(it["values"]) == len(freqs)
        assert all(len(row) == n_edges for row in it["values"])
    # Frame rows reproduce the per-frequency snapshot values.
    snap_doc = yaml.safe_load(dump_case(net, forced=fr, forced_freqs=[250.0]))
    snap = next(d for d in snap_doc["data"]["datasets"] if d["name"] == "250 Hz")
    for name in item_names:
        sweep_row = next(it for it in ds["items"] if it["name"] == name)["values"][1]
        snap_vals = next(it for it in snap["items"] if it["name"] == name)["values"]
        assert np.allclose(sweep_row, snap_vals)
    info = _info_map(ds)
    assert info["kind"]["value"] == "Forced response sweep"
    assert info["n_freqs"]["value"] == 3
    assert info["f_min"]["value"] == 100.0 and info["f_max"]["value"] == 500.0


def test_forced_sweep_frequency_subset():
    net, sol = _forced_net()
    fr = forced_response(sol.problem, sol.x, np.array([100.0, 250.0, 500.0]))
    doc = yaml.safe_load(dump_case(net, forced=fr, forced_sweep=True, forced_freqs=[100.0, 500.0]))
    (ds,) = doc["data"]["datasets"]
    assert ds["frames"]["values"] == [100.0, 500.0]
    assert all(len(it["values"]) == 2 for it in ds["items"])


def test_eigenmode_animation_dataset():
    net, sol, res = _resonator()
    doc = yaml.safe_load(
        dump_case(net, eigenmodes=res, eig_modes=[0], eig_fields=("p",), eig_animation=True, eig_frames=8)
    )
    names = [d["name"] for d in doc["data"]["datasets"]]
    f0 = float(res.freqs[0])
    assert names == [f"Mode 0: {f0:g} Hz", f"Mode 0: {f0:g} Hz animation"]
    anim = doc["data"]["datasets"][1]
    assert anim["frames"]["variable"] == "Phase"
    assert anim["frames"]["unit"] == "deg"
    assert anim["frames"]["values"] == [45.0 * k for k in range(8)]  # endpoint excluded
    (item,) = anim["items"]
    assert item["name"] == "Pressure"
    n_edges = len(net._edges)
    assert len(item["values"]) == 8
    assert all(len(row) == n_edges for row in item["values"])
    # Each frame is the instantaneous field Re{psi e^(i theta)}.
    shape = res.mode_shape(0, "network")
    for k, th in enumerate(anim["frames"]["values"]):
        expected = [(complex(shape[e, 1]) * np.exp(1j * np.radians(th))).real for e in range(n_edges)]
        assert np.allclose(item["values"][k], expected)
    info = _info_map(anim)
    assert info["kind"]["value"] == "Eigenmode animation"
    assert info["n_frames"]["value"] == 8
    assert info["frequency"]["value"] == pytest.approx(f0, rel=1e-6)


def test_eigenmode_animation_rejects_degenerate_frame_count():
    net, sol, res = _resonator()
    with pytest.raises(ValueError, match="at least 2"):
        dump_case(net, eigenmodes=res, eig_animation=True, eig_frames=1)


def test_animated_node_reduction_is_per_frame():
    net, sol, res = _resonator()
    doc = yaml.safe_load(
        dump_case(
            net, eigenmodes=res, eig_modes=[0], eig_fields=("p",), eig_animation=True, eig_frames=4, node_data=True
        )
    )
    anim = doc["data"]["datasets"][1]
    edge_item = next(it for it in anim["items"] if it["target"] == "edge")
    node_item = next(it for it in anim["items"] if it["target"] == "node")
    assert len(node_item["values"]) == 4
    # Node 1 of the resonator chain touches both edges: its value is their mean, frame by frame.
    for k in range(4):
        row = edge_item["values"][k]
        assert node_item["values"][k][1] == pytest.approx((row[0] + row[1]) / 2.0)


def test_animated_dataset_frame_mismatch_rejected():
    net = _nozzle_in_python()
    from nefes.io import FrameAxis

    n_edges = len(net._edges)
    bad = DataSet(
        "Bad",
        [DataItem("X", "edge", [[0.0] * n_edges, [1.0] * n_edges])],
        frames=FrameAxis("Phase", [0.0, 120.0, 240.0], "deg"),
    )
    with pytest.raises(ValueError, match="frame rows"):
        dump_case(net, extra_datasets=[bad])
    # Per-frame rows without a frames axis are likewise rejected.
    orphan = DataSet("Orphan", [DataItem("X", "edge", [[0.0] * n_edges])])
    with pytest.raises(ValueError, match="no frames axis"):
        dump_case(net, extra_datasets=[orphan])


def test_animated_custom_dataset_roundtrips():
    net = _nozzle_in_python()
    from nefes.io import FrameAxis

    n_edges = len(net._edges)
    rows = [[float(k + e) for e in range(n_edges)] for k in range(3)]
    ds = DataSet(
        "Sweep",
        [DataItem("X", "edge", rows, "-")],
        frames=FrameAxis("Parameter", [0.1, 0.2, 0.3]),
    )
    doc = yaml.safe_load(dump_case(net, extra_datasets=[ds]))
    out = next(d for d in doc["data"]["datasets"] if d["name"] == "Sweep")
    assert out["frames"] == {"variable": "Parameter", "values": [0.1, 0.2, 0.3]}  # empty unit omitted
    assert out["items"][0]["values"] == rows


def test_nyquist_summary_dataset():
    from nefes.perturbation import NyquistResponse

    net = _nozzle_in_python()
    freqs = np.linspace(0.0, 500.0, 64)
    # A quiet, non-encircling determinant locus: stable, closed band edge, no crossings.
    D = 1.0 + 0.05 * np.exp(1j * np.linspace(0.0, np.pi, freqs.size))
    D[-1] = 1.0
    ny = NyquistResponse(freqs=freqs, L=1.0 - D, D=D, rank=1, source_labels=("flame",))
    doc = yaml.safe_load(dump_case(net, nyquist=ny))
    (ds,) = doc["data"]["datasets"]
    assert ds["name"] == "Nyquist stability"
    assert ds["items"] == []
    info = _info_map(ds)
    assert info["kind"]["value"] == "Nyquist stability"
    assert info["stable"]["value"] is True
    assert info["n_unstable"]["value"] == 0
    assert info["margin"]["value"] == pytest.approx(float(np.min(np.abs(D))))
    assert info["f_max"]["value"] == 500.0 and info["f_max"]["unit"] == "Hz"
    assert info["crossings"]["value"] == "none"
    assert info["sources"]["value"] == "flame"
