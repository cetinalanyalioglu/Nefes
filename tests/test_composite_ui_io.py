"""UI-case round trips for composite elements, the forced splitter, and the transfer matrix.

Composites serialize as the single node the user specified (``Orifice``,
``FannoPipe``, ...) carrying only the factory parameters retained on
``CompositeElementSpec.params`` -- never their expanded internals.  The strongest
check is the full round trip: dump, reload, re-solve, and compare the converged
fields.  The reader side is additionally exercised through the explicit-port
build path (a UI export pins every port), which must expand composites exactly
like the auto-port path.
"""

import os

import numpy as np
import pytest
import yaml

from nefes.elements import catalog as cat
from nefes.elements.composite import CompositeElementSpec
from nefes.io import load_case, save_case
from nefes.io.yaml_in import _parse_area_profile, _parse_fractions
from nefes.shell import Network
from nefes.thermo.configure import perfect_gas

CFG = perfect_gas(287.0, 1.4)

TAPER_TABLE = [(0.0, 3e-3), (0.1, 2e-3), (0.2, 3e-3)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _inline_net(element, area_in=3e-3, area_out=3e-3):
    """[pt inlet] - <element> - [p outlet] with the given edge areas."""
    net = Network(CFG, p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(130000.0, 300.0, name="res"))
    net.add(element)
    net.add(cat.pressure_outlet(101325.0, 300.0, name="back"))
    net.connect(0, 1, area_in)
    net.connect(1, 2, area_out)
    return net


def _roundtrip(net, tmp_path):
    """Solve, dump, reload, re-solve; return (sol, reloaded net, reloaded sol, doc)."""
    sol = net.solve()
    assert sol.converged
    path = os.path.join(tmp_path, "case.yaml")
    save_case(net, str(path), solution=sol)
    net2 = load_case(str(path))
    sol2 = net2.solve()
    assert sol2.converged
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    return sol, net2, sol2, doc


def _node_of_type(doc, ui_type):
    nodes = [n for n in doc["model"]["nodes"] if n["type"] == ui_type]
    assert len(nodes) == 1, f"expected exactly one {ui_type} node"
    return nodes[0]


# --------------------------------------------------------------------------
# Composite round trips (writer emits the composite node, reader rebuilds it)
# --------------------------------------------------------------------------
COMPOSITES = [
    (cat.orifice(1.2e-3, name="orf"), "Orifice", {"throatArea": 1.2e-3}),
    (
        cat.lossy_nozzle(1.2e-3, 0.7, name="noz"),
        "LossyNozzle",
        {"throatArea": 1.2e-3, "beta": 0.7},
    ),
    (
        cat.sudden_contraction(cc=0.62, name="con"),
        "SuddenContraction",
        {"contractionCoefficient": 0.62},
    ),
    (
        cat.helmholtz_resonator(1e-3, 0.05, 5e-4, name="hr"),
        "HelmholtzResonator",
        {"volume": 1e-3, "neckLength": 0.05, "neckArea": 5e-4},
    ),
    (
        cat.fanno_pipe(2.0, 0.062, 0.02, 4, name="fanno"),
        "FannoPipe",
        {"length": 2.0, "diameter": 0.062, "frictionFactor": 0.02, "nSegments": 4},
    ),
    (
        cat.tapered_duct(TAPER_TABLE, name="taper"),
        "TaperedDuct",
        # station values are emitted at full round-trip precision (repr), so 0.0 stays "0.0"
        {"areaProfile": "0.0:0.003, 0.1:0.002, 0.2:0.003"},
    ),
]


@pytest.mark.parametrize("element,ui_type,attrs", COMPOSITES, ids=[c[1] for c in COMPOSITES])
def test_composite_roundtrip_resolves(tmp_path, element, ui_type, attrs):
    if ui_type == "SuddenContraction":
        net = _inline_net(element, area_in=4e-3, area_out=2e-3)
    elif ui_type == "FannoPipe":
        a = np.pi * 0.062**2 / 4.0
        net = _inline_net(element, area_in=a, area_out=a)
    else:
        net = _inline_net(element)
    sol, net2, sol2, doc = _roundtrip(net, tmp_path)

    node = _node_of_type(doc, ui_type)
    for key, val in attrs.items():
        if isinstance(val, str):
            assert node["attributes"][key] == val
        else:
            assert node["attributes"][key] == pytest.approx(val)

    # user-facing edge count is preserved (internals are never serialized)
    assert len(doc["model"]["edges"]) == 2
    np.testing.assert_allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-10)


def test_tapered_duct_area_roundtrips_exactly(tmp_path):
    """A non-round taper station area must reload bit-identical to the external edge area.

    The taper's boundary duct is constant-area: one port is the external edge, the other the
    composite's own station.  If the station were serialized lossily (``%g``) while the edge is
    full precision, the two disagree past the equal-area tolerance and the reloaded network fails
    validation.  A round-valued taper (the parametrized case above) hides this; a non-round one
    exercises it."""
    import math

    from nefes.io.yaml_in import _parse_area_profile

    a = 3e-3 * math.sqrt(2) / 2  # ~2.12132e-3: %g truncates to 6 figures, repr keeps it exact
    net = _inline_net(cat.tapered_duct([(0.0, a), (0.1, 1.5e-3), (0.2, a)], name="taper"), area_in=a, area_out=a)
    _sol, _net2, _sol2, doc = _roundtrip(net, tmp_path)  # re-solve asserts the reload validates

    prof = _node_of_type(doc, "TaperedDuct")["attributes"]["areaProfile"]
    stations = _parse_area_profile(prof)
    assert stations[0][1] == a and stations[-1][1] == a  # exact, not truncated


def test_composite_provenance_roundtrip(tmp_path):
    """save -> load -> save -> load keeps ids and the solution."""
    net = _inline_net(cat.orifice(1.2e-3, name="orf"))
    _sol, net2, sol2, _doc = _roundtrip(net, tmp_path)
    path2 = os.path.join(tmp_path, "case2.yaml")
    save_case(net2, str(path2), solution=sol2)
    net3 = load_case(str(path2))
    sol3 = net3.solve()
    np.testing.assert_allclose(sol2.field("mdot"), sol3.field("mdot"), rtol=1e-10)
    assert float(sol3.composite("orf").throat_state["M"]) > 0.5


def test_lumped_fanno_pipe_serializes_as_pipe(tmp_path):
    """n_segments=1 short-circuits to a plain pipe atom, hence a Pipe node."""
    a = np.pi * 0.05**2 / 4.0
    net = _inline_net(cat.fanno_pipe(1.0, 0.05, 0.02, 1, name="lp"), area_in=a, area_out=a)
    _sol, _net2, _sol2, doc = _roundtrip(net, tmp_path)
    _node_of_type(doc, "Pipe")


def test_unknown_composite_kind_rejected(tmp_path):
    spec = CompositeElementSpec(
        name="custom",
        sub_elements=[cat.isentropic_area_change(name="c.a"), cat.sudden_area_change(name="c.b")],
        internal_edges=[(0, 1, 1e-3)],
        kind="bespoke",
    )
    net = _inline_net(spec)
    with pytest.raises(ValueError, match="cannot be serialized"):
        save_case(net, os.path.join(tmp_path, "case.yaml"))


# --------------------------------------------------------------------------
# Forced splitter
# --------------------------------------------------------------------------
def test_forced_splitter_roundtrip(tmp_path):
    net = Network(CFG, p_ref=101325.0, T_ref=300.0)
    net.add(cat.total_pressure_inlet(130000.0, 300.0, name="res"))
    net.add(cat.forced_splitter([0.3], name="div"))
    net.add(cat.pressure_outlet(101325.0, 300.0, name="out-a"))
    net.add(cat.pressure_outlet(101325.0, 300.0, name="out-b"))
    net.connect(0, 1, 3e-3)
    net.connect(1, 2, 2e-3)
    net.connect(1, 3, 2e-3)
    sol, _net2, sol2, doc = _roundtrip(net, tmp_path)

    node = _node_of_type(doc, "ForcedSplitter")
    assert node["attributes"]["fractions"] == "0.3"
    np.testing.assert_allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-10)
    # the controlled branch carries its prescribed share of the inflow
    mdot = sol2.field("mdot")
    assert float(mdot[1]) == pytest.approx(0.3 * float(mdot[0]), rel=1e-8)


# --------------------------------------------------------------------------
# Transfer matrix element (topology round-trips; the descriptor does not)
# --------------------------------------------------------------------------
def test_transfer_matrix_roundtrip(tmp_path):
    net = _inline_net(cat.transfer_matrix_element(name="tm"))
    sol, net2, sol2, doc = _roundtrip(net, tmp_path)

    node = _node_of_type(doc, "TransferMatrix")
    # no modeled attributes beyond the writer's label/index bookkeeping
    assert set(node["attributes"]) <= {"label", "index"}
    np.testing.assert_allclose(sol.field("mdot"), sol2.field("mdot"), rtol=1e-10)
    # the reloaded element carries no descriptor; it is attached in Python
    from nefes.elements.ids import TRANSFER_MATRIX

    (spec,) = [e for e in net2._elements if e.residual_id == TRANSFER_MATRIX]
    assert spec.transfer_matrix is None


def test_transfer_matrix_descriptor_warns_on_save(tmp_path):
    net = _inline_net(cat.transfer_matrix_element(tm=object(), name="tm"))
    with pytest.warns(UserWarning, match="not serializable"):
        save_case(net, os.path.join(tmp_path, "case.yaml"))


# --------------------------------------------------------------------------
# Explicit-port build path (the loader pins every port)
# --------------------------------------------------------------------------
def test_explicit_ports_expand_composites():
    """Pinned ports at atomic endpoints survive; the composite expands identically."""

    def build(pin):
        net = Network(CFG, p_ref=101325.0, T_ref=300.0)
        net.add(cat.total_pressure_inlet(130000.0, 300.0, name="res"))
        net.add(cat.helmholtz_resonator(1e-3, 0.05, 5e-4, name="hr"))
        net.add(cat.pressure_outlet(101325.0, 300.0, name="back"))
        kw0 = {"tail_port": 0, "head_port": 0} if pin else {}
        kw1 = {"tail_port": 1, "head_port": 0} if pin else {}
        net.connect(0, 1, 3e-3, **kw0)
        net.connect(1, 2, 3e-3, **kw1)
        return net.solve()

    sol_auto, sol_pinned = build(False), build(True)
    assert sol_auto.converged and sol_pinned.converged
    np.testing.assert_allclose(sol_auto.field("mdot"), sol_pinned.field("mdot"), rtol=1e-12)


# --------------------------------------------------------------------------
# Attribute-string parsers
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("0:3e-3, 0.15:1.5e-3, 0.3:3e-3", [(0.0, 3e-3), (0.15, 1.5e-3), (0.3, 3e-3)]),
        ("0=3e-3; 0.1=2e-3", [(0.0, 3e-3), (0.1, 2e-3)]),
        ("[[0, 0.003], [0.1, 0.002]]", [(0.0, 3e-3), (0.1, 2e-3)]),
        ([(0.0, 3e-3), (0.1, 2e-3)], [(0.0, 3e-3), (0.1, 2e-3)]),
    ],
)
def test_parse_area_profile(text, expected):
    assert _parse_area_profile(text) == pytest.approx(expected)


def test_parse_area_profile_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_area_profile("0.1, 0.2, 0.3")
    with pytest.raises(ValueError):
        _parse_area_profile("")
    with pytest.raises(ValueError):
        _parse_area_profile(None)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.3, 0.2", [0.3, 0.2]),
        ("0.5", [0.5]),
        ("[0.25, 0.25]", [0.25, 0.25]),
        ([0.4], [0.4]),
    ],
)
def test_parse_fractions(text, expected):
    assert _parse_fractions(text) == pytest.approx(expected)


def test_parse_fractions_rejects_empty():
    with pytest.raises(ValueError):
        _parse_fractions("")
    with pytest.raises(ValueError):
        _parse_fractions(None)
