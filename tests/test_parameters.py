"""Tests for the named-parameter API: schema consistency, addressing, set/get, copies.

Covers the guarantees the parameter machinery makes:

* the declared schema packs ``fparams`` identically to every catalog factory (no drift);
* ``get`` after ``set`` returns the set value, including inside composites;
* validation is fail-closed (out-of-range raises, unknown addresses raise with
  suggestions, nothing is ever silently dropped);
* ``with_params`` leaves the base pristine and preserves the edge layout, so a previous
  solution warm-starts the modified copy;
* the YAML round-trip preserves values modified through ``set``.
"""

import numpy as np
import pytest

import nefes
from nefes.elements import catalog as cat
from nefes.elements.composite import is_composite
from nefes.elements.parameters import (
    COMPOSITE_PARAMS,
    ELEMENT_PARAMS,
    descriptors_for,
    pack_fparams,
)
from nefes.perturbation.operator.boundary_bc import PerturbationBC
from nefes.shell.params import ParameterInventory

# --------------------------------------------------------------------------- #
# Layer 1: schema <-> factory consistency
# --------------------------------------------------------------------------- #

# One representative factory call per atomic kind: (factory, kwargs, named slot values).
# The consistency test packs the named values through the schema and compares with the
# factory's actual fparams -- guarding the declared slot layout against drift.
ATOMIC_SAMPLES = [
    (cat.mass_flow_inlet, dict(mdot=0.31, Tt=713.0), dict(mdot=0.31, Tt=713.0)),
    (cat.total_pressure_inlet, dict(pt=2.1e5, Tt=650.0), dict(pt=2.1e5, Tt=650.0)),
    (cat.pressure_outlet, dict(p=1.3e5, Tt_backflow=310.0), dict(p=1.3e5, Tt_backflow=310.0)),
    (cat.mass_flow_outlet, dict(mdot=0.4), dict(mdot=0.4)),
    (cat.choked_nozzle_outlet, dict(throat_area=2.5e-3), dict(throat_area=2.5e-3)),
    (cat.wall, dict(), dict()),
    (cat.cavity, dict(volume=1.7e-3), dict(volume=1.7e-3)),
    (
        cat.isentropic_area_change,
        dict(l_up=0.01, l_down=0.02, end_correction=0.003),
        dict(l_up=0.01, l_down=0.02, end_correction=0.003),
    ),
    (cat.transfer_matrix_element, dict(), dict()),
    (
        cat.sudden_area_change,
        dict(cc=0.62, l_up=0.01, l_down=0.0, end_correction=0.002),
        dict(cc=0.62, l_up=0.01, l_down=0.0, end_correction=0.002),
    ),
    (
        cat.loss,
        dict(K=1.8, ref_port=1, l_up=0.004, l_down=0.0, end_correction=0.001),
        dict(K=1.8, ref_port=1, l_up=0.004, l_down=0.0, end_correction=0.001),
    ),
    (
        cat.linear_resistance,
        dict(R=4.0e4, l_up=0.002, l_down=0.002, end_correction=0.0),
        dict(R=4.0e4, l_up=0.002, l_down=0.002, end_correction=0.0),
    ),
    (cat.heat_release_flame, dict(Qdot=8.0e3), dict(Qdot=8.0e3)),
    (cat.equilibrium_flame, dict(), dict()),
    (
        cat.mass_source,
        dict(mdot=0.01, T=300.0, composition={"CH4": 1.0}, u_inj=25.0),
        dict(mdot=0.01, u_inj=25.0, T=300.0),
    ),
    (cat.junction, dict(volume=2e-3), dict(volume=2e-3)),
    (cat.splitter, dict(volume=0.0), dict(volume=0.0)),
    (cat.forced_splitter, dict(fractions=[0.3, 0.2]), dict(fractions=[0.3, 0.2])),
    (cat.duct, dict(length=0.7), dict(length=0.7)),
    (
        cat.pipe,
        dict(length=1.5, diameter=0.05, friction_factor=0.02),
        dict(length=1.5, diameter=0.05, friction_factor=0.02),
    ),
]


@pytest.mark.parametrize("factory,kwargs,named", ATOMIC_SAMPLES, ids=lambda s: getattr(s, "__name__", ""))
def test_schema_packing_matches_factory(factory, kwargs, named):
    spec = factory(**kwargs)
    packed = pack_fparams(spec.residual_id, named)
    assert packed == pytest.approx(list(spec.fparams)), f"{factory.__name__}: schema packing drifted from the factory"


def test_every_declared_slot_is_covered():
    # every kind in the registry has contiguous slots 0..n-1 and unique names
    for rid, descs in ELEMENT_PARAMS.items():
        slots = sorted(d.slot for d in descs if d.slot is not None)
        assert slots == list(range(len(slots))), f"rid {rid}: non-contiguous slots {slots}"
        names = [d.name for d in descs]
        assert len(names) == len(set(names)), f"rid {rid}: duplicate parameter names"


COMPOSITE_SAMPLES = [
    (cat.orifice, dict(throat_area=1e-3)),
    (cat.lossy_nozzle, dict(throat_area=1e-3, beta=0.8)),
    (cat.sudden_contraction, dict(cc=0.7)),
    (cat.helmholtz_resonator, dict(volume=1e-3, neck_length=0.03, neck_area=2e-4)),
    (cat.fanno_pipe, dict(length=1.0, diameter=0.04, friction_factor=0.02, n_segments=4)),
    (cat.tapered_duct, dict(area=[(0.0, 3e-3), (0.15, 1.5e-3), (0.3, 3e-3)])),
]


@pytest.mark.parametrize("factory,kwargs", COMPOSITE_SAMPLES, ids=lambda s: getattr(s, "__name__", ""))
def test_composite_schema_matches_factory_params(factory, kwargs):
    spec = factory(**kwargs)
    declared = {d.name for d in COMPOSITE_PARAMS[spec.kind]}
    assert declared == set(spec.params), f"{spec.kind}: schema names drifted from the factory params dict"


def test_descriptors_for_unknown_composite_kind_raises():
    rogue = cat.orifice(1e-3)
    rogue.kind = "mystery"
    with pytest.raises(KeyError, match="mystery"):
        descriptors_for(rogue)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _simple_network():
    """inlet -> orifice -> duct -> outlet, with named elements and edges."""
    net = nefes.Network()
    n0 = net.add(cat.mass_flow_inlet(0.3, 700.0, name="in"))
    n1 = net.add(cat.orifice(1e-3, name="ori"))
    n2 = net.add(cat.duct(length=0.5, name="d1"))
    n3 = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(n0, n1, 5e-3, name="e0")
    net.connect(n1, n2, 5e-3, name="e1")
    net.connect(n2, n3, 5e-3, name="e2")
    return net


# --------------------------------------------------------------------------- #
# Inventory and get
# --------------------------------------------------------------------------- #
def test_inventory_covers_elements_edges_and_refs():
    net = _simple_network()
    inv = net.parameters()
    assert isinstance(inv, ParameterInventory)
    addresses = inv.addresses
    assert "in.mdot" in addresses
    assert "ori.throat_area" in addresses
    assert "d1.length" in addresses
    assert "out.p" in addresses
    assert "e1.area" in addresses
    assert "p_ref" in addresses and "T_ref" in addresses
    assert inv["in.mdot"].value == pytest.approx(0.3)
    assert inv["in.mdot"].unit == "kg/s"
    # advanced knobs are hidden by default, present on request
    assert "mdot_ref" not in addresses
    assert "mdot_ref" in net.parameters(advanced=True).addresses


def test_inventory_repr_marks_layer_and_advanced():
    """Text and HTML tables expose layer (μ/∼) and advanced (*) columns."""
    net = _simple_network()
    net.add(cat.cavity(volume=1e-3, name="cav"))
    plain = net.parameters()
    text = repr(plain)
    assert "layer" in text.splitlines()[0]
    assert "adv" in text.splitlines()[0]
    assert "μ" in text
    assert "μ mean" in text and "∼ perturbation" in text
    assert plain["in.mdot"].layer == "mean" and not plain["in.mdot"].advanced
    assert plain["cav.volume"].layer == "perturbation"

    adv = net.parameters(advanced=True)
    assert adv["mdot_ref"].advanced
    assert "*" in repr(adv)

    html = plain._repr_html_()
    assert "currentColor" in html
    assert "μ" in html
    assert "title=" in html
    assert "#ccc" not in html  # theme-hardcoded border color is banned


def test_get_reads_slots_fields_composites_edges_and_refs():
    net = _simple_network()
    assert net.get("in.mdot") == pytest.approx(0.3)
    assert net.get("in.Tt") == pytest.approx(700.0)
    assert net.get("ori.throat_area") == pytest.approx(1e-3)
    assert net.get("e2.area") == pytest.approx(5e-3)
    assert net.get("d1.length") == pytest.approx(0.5)
    assert net.get("d1.area") == pytest.approx(5e-3)  # element-level read of the shared area
    assert net.get("p_ref") == pytest.approx(101325.0)
    assert net.get("in.perturbation_bc") is None


def test_element_lookup_by_name_and_index():
    net = _simple_network()
    assert net.element_index("ori") == 1
    assert net.element(1) is net.element("ori")
    with pytest.raises(KeyError, match="ory|ori"):
        net.element_index("ory")  # suggestion machinery
    with pytest.raises(KeyError, match="out of range"):
        net.element(17)


# --------------------------------------------------------------------------- #
# set / update: happy paths
# --------------------------------------------------------------------------- #
def test_set_roundtrips_and_invalidates():
    net = _simple_network()
    prob0 = net.problem
    node = net.set("in", mdot=0.5, Tt=720.0)
    assert node == 0
    assert net.get("in.mdot") == pytest.approx(0.5)
    assert net.get("in.Tt") == pytest.approx(720.0)
    assert net._compiled is None, "set must drop the compiled cache"
    assert net.problem is not prob0


def test_set_composite_rebuilds_through_factory():
    net = _simple_network()
    net.set("ori", throat_area=1.2e-3)
    assert net.get("ori.throat_area") == pytest.approx(1.2e-3)
    el = net.element("ori")
    assert is_composite(el) and el.name == "ori"
    # the derived internal edge is regenerated, never patched
    assert el.internal_edges[0][2] == pytest.approx(1.2e-3)


def test_set_composite_preserves_eps():
    net = nefes.Network()
    n0 = net.add(cat.mass_flow_inlet(0.3, 700.0, name="in"))
    n1 = net.add(cat.orifice(1e-3, name="ori", eps=1e-7))
    n2 = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(n0, n1, 5e-3)
    net.connect(n1, n2, 5e-3)
    net.set("ori", throat_area=1.1e-3)
    eps = [sub.eps for sub in net.element("ori").sub_elements if sub.eps is not None]
    assert eps == [pytest.approx(1e-7)]


def test_set_element_area_fans_out_on_constant_area_elements():
    net = _simple_network()
    net.set("d1", area=4e-3)
    assert net.get("e1.area") == pytest.approx(4e-3)
    assert net.get("e2.area") == pytest.approx(4e-3)
    # single-port element: its one incident edge
    net.set("in", area=6e-3)
    assert net.get("e0.area") == pytest.approx(6e-3)


def test_element_area_rejected_on_per_edge_elements():
    net = nefes.Network()
    n0 = net.add(cat.mass_flow_inlet(0.3, 700.0, name="in"))
    n1 = net.add(cat.isentropic_area_change(name="ac"))
    n2 = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(n0, n1, 5e-3, name="e0")
    net.connect(n1, n2, 3e-3, name="e1")
    with pytest.raises(KeyError, match="per-edge"):
        net.set("ac", area=4e-3)
    with pytest.raises(KeyError, match="per-edge"):
        net.get("ac.area")


def test_update_batches_addresses():
    net = _simple_network()
    out = net.update({"in.mdot": 0.45, "ori.throat_area": 0.9e-3, "e0.area": 6e-3, "p_ref": 90000.0})
    assert out is net
    assert net.get("in.mdot") == pytest.approx(0.45)
    assert net.get("ori.throat_area") == pytest.approx(0.9e-3)
    assert net.get("e0.area") == pytest.approx(6e-3)
    assert net.p_ref == pytest.approx(90000.0)


def test_set_object_fields_validated():
    net = _simple_network()
    net.set("out", perturbation_bc=PerturbationBC.open_end())
    assert net.get("out.perturbation_bc").kind == "open_end"
    net.set_perturbation_bc("out", None)
    assert net.get("out.perturbation_bc") is None
    with pytest.raises(ValueError, match="PerturbationBC"):
        net.set("out", perturbation_bc="open")


def test_set_dynamic_source_routes_through_schema():
    from nefes.elements.dynamic_source import n_tau_flame

    net = nefes.Network()
    n0 = net.add(cat.mass_flow_inlet(0.3, 700.0, name="in"))
    n1 = net.add(cat.heat_release_flame(5e3, name="fl"))
    n2 = net.add(cat.pressure_outlet(101325.0, name="out"))
    net.connect(n0, n1, 5e-3)
    e_ref = net.connect(n1, n2, 5e-3)
    src = n_tau_flame(1.0, 3e-3, ref_edge=e_ref)
    assert net.set_dynamic_source("fl", src) == n1
    assert net.get("fl.dynamic_source") is src
    with pytest.raises(ValueError, match="DynamicSource"):
        net.set("fl", dynamic_source=object())
    with pytest.raises(KeyError, match="no parameter"):
        net.set_dynamic_source("in", src)  # an inlet carries no S(omega) descriptor


def test_forced_splitter_fractions_vector():
    net = nefes.Network()
    n0 = net.add(cat.mass_flow_inlet(0.6, 700.0, name="in"))
    n1 = net.add(cat.forced_splitter([0.3, 0.2], name="fs"))
    n2 = net.add(cat.pressure_outlet(101325.0, name="o1"))
    n3 = net.add(cat.pressure_outlet(101325.0, name="o2"))
    n4 = net.add(cat.pressure_outlet(101325.0, name="o3"))
    net.connect(n0, n1, 5e-3)
    net.connect(n1, n2, 3e-3)
    net.connect(n1, n3, 3e-3)
    net.connect(n1, n4, 3e-3)
    assert net.get("fs.fractions") == pytest.approx([0.3, 0.2])
    net.set("fs", fractions=[0.4, 0.1])
    assert net.get("fs.fractions") == pytest.approx([0.4, 0.1])
    with pytest.raises(ValueError, match="length"):
        net.set("fs", fractions=[0.4])  # changing the count is a topology change
    with pytest.raises(ValueError, match="sum"):
        net.set("fs", fractions=[0.7, 0.5])


# --------------------------------------------------------------------------- #
# Fail-closed validation and addressing
# --------------------------------------------------------------------------- #
def test_out_of_range_set_raises_named_error():
    net = _simple_network()
    with pytest.raises(ValueError, match=r"mdot must be .*'in'"):
        net.set("in", mdot=-0.1)
    with pytest.raises(ValueError, match="throat_area"):
        net.set("ori", throat_area=-1e-3)
    with pytest.raises(ValueError, match="area"):
        net.update({"e0.area": 0.0})
    # nothing was written by the failed attempts
    assert net.get("in.mdot") == pytest.approx(0.3)
    assert net.get("ori.throat_area") == pytest.approx(1e-3)
    assert net.get("e0.area") == pytest.approx(5e-3)


def test_unknown_parameter_raises_with_inventory():
    net = _simple_network()
    with pytest.raises(KeyError, match="mdot"):
        net.set("in", mdo=0.5)
    with pytest.raises(KeyError, match="no parameter"):
        net.set("d1", volume=1.0)


def test_unknown_address_raises_with_suggestions():
    net = _simple_network()
    with pytest.raises(KeyError, match="ori"):
        net.get("orifi.throat_area")
    with pytest.raises(KeyError, match="p_ref|T_ref"):
        net.get("p_reff")
    with pytest.raises(KeyError, match="area"):
        net.get("e0.length")  # an edge's one parameter is its area


def test_marker_and_composition_validation():
    net = _simple_network()
    with pytest.raises(ValueError, match="marker"):
        net.set("in", marker=1.5)
    net.set("in", composition={"O2": 0.21, "N2": 0.79}, basis="mole")
    assert net.get("in.composition") == {"O2": 0.21, "N2": 0.79}
    with pytest.raises(ValueError, match="basis"):
        net.set("out", basis="mole")  # basis without any composition is meaningless
    with pytest.raises(ValueError, match="'mole' or 'mass'"):
        net.set("in", basis="molar")


# --------------------------------------------------------------------------- #
# copy / with_params: pristine base, preserved layout, valid warm starts
# --------------------------------------------------------------------------- #
def test_with_params_leaves_base_pristine():
    net = _simple_network()
    mod = net.with_params({"in.mdot": 0.6, "ori.throat_area": 1.4e-3, "e0.area": 7e-3})
    assert net.get("in.mdot") == pytest.approx(0.3)
    assert net.get("ori.throat_area") == pytest.approx(1e-3)
    assert net.get("e0.area") == pytest.approx(5e-3)
    assert mod.get("in.mdot") == pytest.approx(0.6)
    assert mod.get("ori.throat_area") == pytest.approx(1.4e-3)
    assert mod.get("e0.area") == pytest.approx(7e-3)


def test_copy_is_deep_for_specs_and_preserves_edge_layout():
    net = _simple_network()
    dup = net.copy()
    assert [el.name for el in dup._elements] == [el.name for el in net._elements]
    assert dup._edges == net._edges and dup._edges is not net._edges
    assert dup._edge_names == net._edge_names
    assert dup._ports == net._ports
    dup.set("in", mdot=0.9)
    assert net.get("in.mdot") == pytest.approx(0.3)
    assert dup.gas is net.gas  # the immutable gas config is shared


def test_with_params_warm_start_stays_valid():
    # A comfortably subsonic operating point: the orifice throat is near choke at the fixture's
    # default flow, and warm-start iteration counts are only cleanly ordered away from that edge.
    net = _simple_network().with_params({"in.mdot": 0.15})
    sol = net.solve()
    assert sol.converged
    mod = net.with_params({"in.mdot": 0.18})
    assert len(mod._edges) == len(net._edges)
    warm = mod.solve(x0=sol.x)
    cold = mod.solve()
    assert warm.converged and cold.converged
    assert warm.iterations <= cold.iterations
    assert warm.field("mdot") == pytest.approx(cold.field("mdot"), rel=1e-8)


def test_builder_closure_matches_with_params():
    net = _simple_network()
    build = net.builder("in.mdot")
    got = build(0.55)
    assert got.get("in.mdot") == pytest.approx(0.55)
    assert net.get("in.mdot") == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# YAML round-trip of modified values
# --------------------------------------------------------------------------- #
def test_set_then_save_load_roundtrip(tmp_path):
    net = _simple_network()
    net.update({"in.mdot": 0.42, "in.Tt": 655.0, "ori.throat_area": 1.3e-3, "e1.area": 4.5e-3, "out.p": 99000.0})
    net.set("out", perturbation_bc=PerturbationBC.open_end())
    path = str(tmp_path / "case.yaml")
    net.to_yaml(path)
    back = nefes.load_case(path)
    assert back.get(f"{back._elements[0].name}.mdot") == pytest.approx(0.42)
    assert back.get(f"{back._elements[0].name}.Tt") == pytest.approx(655.0)
    assert back.get(f"{back._elements[1].name}.throat_area") == pytest.approx(1.3e-3)
    assert float(back._edges[1][2]) == pytest.approx(4.5e-3)
    assert back.get(f"{back._elements[3].name}.p") == pytest.approx(99000.0)
    assert back.element(3).perturbation_bc.kind == "open_end"


def test_network_refs_roundtrip(tmp_path):
    net = _simple_network()
    net.update({"p_ref": 90000.0, "T_ref": 350.0})
    path = str(tmp_path / "case.yaml")
    net.to_yaml(path)
    back = nefes.load_case(path)
    assert back.p_ref == pytest.approx(90000.0)
    assert back.T_ref == pytest.approx(350.0)


# --------------------------------------------------------------------------- #
# The sweep driver
# --------------------------------------------------------------------------- #
def test_parameter_study_1d_warm_chained():
    net = _simple_network()
    mdots = np.linspace(0.15, 0.30, 4)  # kept below the orifice-throat choke so the sweep stays in scope
    res = nefes.parameter_study(
        net,
        {"in.mdot": mdots},
        probe=lambda sol: {"M_max": float(sol.field("M").max())},
    )
    assert res.shape == (4,)
    assert res.converged.all()
    assert res.grid["in.mdot"] == pytest.approx(mdots)
    assert np.all(np.diff(res.probes["M_max"]) > 0), "Mach must rise with the inflow"
    assert len(res.solutions) == 4
    # the base is pristine
    assert net.get("in.mdot") == pytest.approx(0.3)


def test_parameter_study_grid_shape_and_zip():
    net = _simple_network()
    res = nefes.parameter_study(
        net,
        {"in.mdot": [0.28, 0.30], "out.p": [99000.0, 101325.0, 103000.0]},
        probe=lambda sol: {"p_in": float(sol.field("p")[0])},
        keep_solutions=False,
    )
    assert res.shape == (2, 3)
    assert res.probes["p_in"].shape == (2, 3)
    assert res.solutions is None
    zipped = nefes.parameter_study(net, {"in.mdot": [0.28, 0.30], "out.p": [99000.0, 101325.0]}, mode="zip")
    assert zipped.shape == (2,)


def test_parameter_study_progress_prints_per_point(capsys):
    net = _simple_network()
    res = nefes.parameter_study(net, {"in.mdot": [0.20, 0.24]}, progress=True)
    assert res.converged.all()
    out = capsys.readouterr().out
    # one status line per point, carrying the index, the swept value, and the convergence result
    assert "[1/2]" in out and "[2/2]" in out
    assert "in.mdot=0.2" in out and "converged" in out


def test_parameter_study_silent_by_default(capsys):
    net = _simple_network()
    nefes.parameter_study(net, {"in.mdot": [0.20, 0.24]})
    assert capsys.readouterr().out == ""


def test_parameter_study_fail_closed_address():
    net = _simple_network()
    with pytest.raises(KeyError, match="in"):
        nefes.parameter_study(net, {"inn.mdot": [0.3, 0.4]})


def test_parameter_study_zip_length_mismatch():
    net = _simple_network()
    with pytest.raises(ValueError, match="equal-length"):
        nefes.parameter_study(net, {"in.mdot": [0.3, 0.4], "out.p": [1e5]}, mode="zip")


# --------------------------------------------------------------------------- #
# Reacting: composition writes validate against the species species_set
# --------------------------------------------------------------------------- #
def test_reacting_composition_validated_against_library():
    import os

    from nefes.thermo import SpeciesSet
    from nefes.thermo.configure import equilibrium

    mech = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data", "h2o2.yaml")
    net = nefes.Network(gas=equilibrium(SpeciesSet.from_cantera(mech)))
    n0 = net.add(cat.total_pressure_inlet(1.2e5, 300.0, composition={"O2": 0.21, "N2": 0.79}, name="air"))
    n1 = net.add(cat.pressure_outlet(1.0e5, name="out"))
    net.connect(n0, n1, 1e-2)
    net.set("air", composition={"O2": 1.0}, basis="mole")
    assert net.get("air.composition") == {"O2": 1.0}
    with pytest.raises(ValueError, match="CH4"):
        net.set("air", composition={"CH4": 1.0})  # not in the H2/O2 mechanism
