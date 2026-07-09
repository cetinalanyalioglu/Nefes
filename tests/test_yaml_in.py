"""UI-export (YAML) loader: thermo-model selection, new elements, reacting wiring.

The loader (:mod:`nefes.io.yaml_in`) is the contract between the FNetLibUI tool and the
solver.  These tests cover the additions that bring it in step with the catalog: the
``thermoModel`` selector (perfect gas vs equilibrium), the string-encoded compositions,
the per-edge frozen/equilibrium closure (auto-from-flames + explicit override), the
inlet ``inherit`` acoustic default, and the new HeatReleaseFlame / MassSource /
EquilibriumFlame elements.  The reacting cases need the bundled mechanism data.
"""

import os

import pytest
import yaml

from nefes.io import load_case, save_case
from nefes.io.yaml_in import _parse_composition
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_T, ES_RHO, ES_MDOT, ES_HT, ES_U
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL, EQ_MARKER, PERFECT_GAS

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data", "h2o2.yaml")
# Stoichiometric H2/air as a single premixed feed (mole basis).
H2_AIR = "H2:1.0, O2:0.5, N2:1.88"


# --------------------------------------------------------------------------- #
# helpers: assemble the UI save-file payload the loader consumes
# --------------------------------------------------------------------------- #
def _node(nid, ntype, index, **attrs):
    attrs["index"] = index
    return {"id": nid, "type": ntype, "attributes": attrs}


def _edge(eid, src, tgt, sp, tp, index, area, **attrs):
    attrs.update({"index": index, "area": area})
    return {
        "id": eid,
        "source": src,
        "target": tgt,
        "sourceHandle": f"{src}-port-{sp}",
        "targetHandle": f"{tgt}-port-{tp}",
        "type": "flow",
        "attributes": attrs,
    }


def _dump(tmp_path, name, global_attrs, nodes, edges):
    doc = {
        "version": "2.0.0",
        "model": {"id": "fns-flow-network", "globalAttributes": global_attrs, "nodes": nodes, "edges": edges},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def _series_reacting(tmp_path, edge0_model="auto", edge1_model="auto", name="reacting.yaml"):
    """inlet(H2/air) -> EquilibriumFlame -> outlet, on the equilibrium model."""
    g = {
        "thermoModel": "equilibrium",
        "mechanismFile": MECH,
        "species": "H2, O2, N2, H2O, OH, H, O, HO2",
        "equilibriumTInit": 2500.0,
        "frozenTInit": 300.0,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }
    nodes = [
        _node("in", "MassFlowInlet", 0, label="fuel-air", massFlowRate=1.0, totalTemperature=300.0, composition=H2_AIR),
        _node("flame", "EquilibriumFlame", 1, label="flame"),
        _node(
            "out",
            "PressureOutlet",
            2,
            label="out",
            pressure=101325.0,
            backflowTotalTemperature=300.0,
            composition=H2_AIR,
        ),
    ]
    edges = [
        _edge("e1", "in", "flame", 0, 0, 0, 0.05, thermoModel=edge0_model),
        _edge("e2", "flame", "out", 1, 0, 1, 0.05, thermoModel=edge1_model),
    ]
    return _dump(tmp_path, name, g, nodes, edges)


# --------------------------------------------------------------------------- #
# 1. composition string parser
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("O2:0.21, N2:0.79", {"O2": 0.21, "N2": 0.79}),
        ("CH4 1.0", {"CH4": 1.0}),
        ("H2=2, O2=1", {"H2": 2.0, "O2": 1.0}),
        ('{"O2": 0.21, "N2": 0.79}', {"O2": 0.21, "N2": 0.79}),
        ("", None),
        (None, None),
    ],
)
def test_parse_composition(text, expected):
    assert _parse_composition(text) == expected


def test_parse_composition_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_composition("O2 0.21 extra")


# --------------------------------------------------------------------------- #
# 2. perfect gas: new elements + inlet acoustic default
# --------------------------------------------------------------------------- #
def test_perfect_gas_heat_release_and_mass_source(tmp_path):
    g = {
        "thermoModel": "perfect_gas",
        "gasConstant": 287.0,
        "heatCapacityRatio": 1.4,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 10.0,
    }
    nodes = [
        _node("in", "MassFlowInlet", 0, label="air", massFlowRate=10.0, totalTemperature=300.0, boundaryType="inherit"),
        _node("flame", "HeatReleaseFlame", 1, label="burner", heatRelease=2.0e6),
        _node(
            "src", "MassSource", 2, label="inject", massFlowRate=0.5, injectionTemperature=400.0, injectionVelocity=0.0
        ),
        _node("out", "PressureOutlet", 3, label="out", pressure=100000.0, backflowTotalTemperature=300.0),
    ]
    edges = [
        _edge("e1", "in", "flame", 0, 0, 0, 0.05),
        _edge("e2", "flame", "src", 1, 0, 1, 0.05),
        _edge("e3", "src", "out", 1, 0, 2, 0.05),
    ]
    net = load_case(_dump(tmp_path, "pg.yaml", g, nodes, edges))
    assert net.gas.model_id == PERFECT_GAS
    # the inlet "inherit" acoustic default keeps the linearized mean BC (no stamp)
    assert net._elements[0].perturbation_bc is None
    # perfect gas: every edge follows the gas default (no per-edge override)
    assert net._edge_models == [None, None, None]
    sol = net.solve()
    assert sol.converged
    est = states_table(net.compile(), sol.x)
    cp = 1.4 * 287.0 / 0.4
    # the inlet total temperature is preserved at 300 K
    assert float(est[ES_HT, 0]) / cp == pytest.approx(300.0, rel=1e-3)
    # the steady flame raises the total temperature by Qdot/(mdot cp)
    assert float(est[ES_HT, 1]) / cp == pytest.approx(300.0 + 2.0e6 / (10.0 * cp), rel=1e-3)
    # the mass source adds 0.5 kg/s
    assert float(est[ES_MDOT, 2]) == pytest.approx(10.5, rel=1e-9)


def test_inlet_inherit_vs_explicit_bc(tmp_path):
    """The UI 'inherit' inlet default -> no perturbation stamp; an explicit kind stamps."""
    g = {
        "thermoModel": "perfect_gas",
        "gasConstant": 287.0,
        "heatCapacityRatio": 1.4,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }

    def _case(btype):
        nodes = [
            _node(
                "in",
                "TotalPressureInlet",
                0,
                label="in",
                totalPressure=120000.0,
                totalTemperature=300.0,
                boundaryType=btype,
            ),
            _node("out", "PressureOutlet", 1, label="out", pressure=101325.0, backflowTotalTemperature=300.0),
        ]
        edges = [_edge("e1", "in", "out", 0, 0, 0, 0.02)]
        return load_case(_dump(tmp_path, f"bc_{btype}.yaml", g, nodes, edges))

    assert _case("inherit")._elements[0].perturbation_bc is None
    assert _case("rigid")._elements[0].perturbation_bc.kind == "hard_wall"
    assert _case("open")._elements[0].perturbation_bc.kind == "open_end"


# --------------------------------------------------------------------------- #
# 3. reacting: composition + auto edge closure + solve
# --------------------------------------------------------------------------- #
def test_reacting_auto_edge_models_and_ignition(tmp_path):
    net = load_case(_series_reacting(tmp_path))
    assert net.gas.model_id == EQ_KERNEL
    # auto: defer to the orientation-proof marker closure (EQ_MARKER on every edge + the marker)
    assert net._edge_models == [None, None]
    prob = net.compile()
    assert prob.edge_model.tolist() == [EQ_MARKER, EQ_MARKER] and prob.marker_row == 3 + prob.n_elem
    sol = net.solve()
    assert sol.converged
    est = states_table(net.compile(), sol.x)
    assert float(est[ES_T, 0]) == pytest.approx(300.0, abs=1.0)  # unburnt (marker ~ 0 -> frozen)
    assert float(est[ES_T, 1]) > 2000.0  # burnt (marker ~ 1 -> equilibrium)
    assert float(est[ES_RHO, 0] / est[ES_RHO, 1]) > 5.0  # dilatation
    # one feed stream, discovered from the inlet composition
    assert list(net.compile().scalar_names) == ["fuel-air"]


def test_reacting_explicit_edge_override(tmp_path):
    net = load_case(_series_reacting(tmp_path, edge0_model="frozen", edge1_model="equilibrium", name="ovr.yaml"))
    assert net._edge_models == [EQ_FROZEN, EQ_KERNEL]


def test_reacting_no_flame_is_equilibrium_everywhere(tmp_path):
    """With no flame, every 'auto' edge is equilibrium (the base reacting model)."""
    g = {
        "thermoModel": "equilibrium",
        "mechanismFile": MECH,
        "species": "H2, O2, N2, H2O, OH, H, O, HO2",
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }
    nodes = [
        _node("in", "MassFlowInlet", 0, label="air", massFlowRate=1.0, totalTemperature=300.0, composition="N2:1.0"),
        _node(
            "out",
            "PressureOutlet",
            1,
            label="out",
            pressure=101325.0,
            backflowTotalTemperature=300.0,
            composition="N2:1.0",
        ),
    ]
    edges = [_edge("e1", "in", "out", 0, 0, 0, 0.05)]
    net = load_case(_dump(tmp_path, "noflame.yaml", g, nodes, edges))
    # no flame -> not marker-gated; every auto edge is plain equilibrium (the base reacting model)
    assert net._edge_models == [None]
    assert net.compile().edge_model.tolist() == [EQ_KERNEL]


def test_reacting_burnt_matches_standalone_equilibrium(tmp_path):
    from nefes.thermo import SpeciesLibrary, Thermo
    from nefes.chem.composition import resolve_composition

    net = load_case(_series_reacting(tmp_path, name="ref.yaml"))
    sol = net.solve()
    est = states_table(net.compile(), sol.x)

    lib = SpeciesLibrary.from_cantera(MECH).subset(["H2", "O2", "N2", "H2O", "OH", "H", "O", "HO2"])
    gas = Thermo(lib)
    Y, Z = resolve_composition(lib, {"H2": 1.0, "O2": 0.5, "N2": 1.88}, basis="mole")
    # the recovered burnt T is the *static* temperature, so compare against the HP
    # solve at the *static* enthalpy h = h_t - u^2/2 (the KE-coupled closure).
    h_static = float(est[ES_HT, 1]) - 0.5 * float(est[ES_U, 1]) ** 2
    ref = gas.equilibrate_HP(Z, h_static, float(est[1, 1]))  # at the burnt static pressure (ES_P row 1)
    assert float(est[ES_T, 1]) == pytest.approx(ref.T, rel=1e-3)


# --------------------------------------------------------------------------- #
# 4. reacting error paths
# --------------------------------------------------------------------------- #
def test_reacting_auto_without_feed_composition_is_rejected(tmp_path):
    # Default species="auto" derives the slate from the feed compositions; with no feed
    # composition anywhere there is nothing to derive the element pool from.
    g = {"thermoModel": "equilibrium"}  # species defaults to "auto"
    nodes = [_node("in", "MassFlowInlet", 0, massFlowRate=1.0, totalTemperature=300.0)]  # no composition
    nodes.append(_node("out", "PressureOutlet", 1, pressure=1e5))
    edges = [_edge("e1", "in", "out", 0, 0, 0, 0.05)]
    with pytest.raises(ValueError, match="composition"):
        load_case(_dump(tmp_path, "nofeed.yaml", g, nodes, edges))


def test_reacting_without_mechanism_uses_packaged_thermo_inp(tmp_path):
    # With a species list but no mechanismFile, the bundled NASA Glenn / CEA thermo.inp is used.
    g = {"thermoModel": "equilibrium", "species": "H2, O2, N2, H2O, OH, H, O"}
    nodes = [
        _node("in", "MassFlowInlet", 0, massFlowRate=1.0, totalTemperature=300.0, composition="H2:1"),
        _node("out", "PressureOutlet", 1, pressure=1e5, composition="H2:1"),
    ]
    edges = [_edge("e1", "in", "out", 0, 0, 0, 0.05)]
    net = load_case(_dump(tmp_path, "packaged.yaml", g, nodes, edges))
    assert net.gas.species_names == ["H2", "O2", "N2", "H2O", "OH", "H", "O"]


# --------------------------------------------------------------------------- #
# 4b. automatic species slate (CEA-style) over the packaged thermo.inp
# --------------------------------------------------------------------------- #
def _auto_reacting(tmp_path, composition, name, Tinit=2500.0):
    """inlet(composition) -> EquilibriumFlame -> outlet on the auto-species reacting model."""
    g = {
        "thermoModel": "equilibrium",  # species defaults to "auto"; no mechanismFile
        "equilibriumTInit": Tinit,
        "frozenTInit": 300.0,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }
    nodes = [
        _node(
            "in", "MassFlowInlet", 0, label="feed", massFlowRate=1.0, totalTemperature=300.0, composition=composition
        ),
        _node("flame", "EquilibriumFlame", 1, label="flame"),
        _node("out", "PressureOutlet", 2, label="out", pressure=101325.0, backflowTotalTemperature=300.0),
    ]
    edges = [_edge("e1", "in", "flame", 0, 0, 0, 0.05), _edge("e2", "flame", "out", 1, 0, 1, 0.05)]
    return _dump(tmp_path, name, g, nodes, edges)


def test_auto_slate_h2_air_runs_raw(tmp_path):
    # H/O/N admits ~30 gas species (below the reduce threshold) -> kept as-is.
    net = load_case(_auto_reacting(tmp_path, "H2:1.0, O2:0.5, N2:1.88", "auto_h2.yaml"))
    names = net.gas.species_names
    for feed in ("H2", "O2", "N2"):
        assert feed in names
    report = net.gas.library.reduction_report
    assert report["reducer"] == "none"
    assert report["n_candidates"] == report["n_kept"]
    net.solve()  # converges


def test_auto_slate_hydrocarbon_reduces(tmp_path):
    # CH4/air -> {C,H,O,N} admits ~115 gas species -> reduced.
    net = load_case(_auto_reacting(tmp_path, "CH4:1.0, O2:2.0, N2:7.52", "auto_ch4.yaml"))
    names = net.gas.species_names
    report = net.gas.library.reduction_report
    assert report["reducer"] == "equilibrium_sampling"
    assert report["n_candidates"] > 100
    assert report["n_kept"] < report["n_candidates"]  # actually trimmed
    # the feed and the major products survive
    for expect in ("CH4", "CO2", "H2O", "CO", "N2"):
        assert expect in names
    net.solve()


def test_auto_slate_liquid_fuel_feed(tmp_path):
    # A condensed feed (liquid Jet-A) is carried in the library but masked out of the products.
    net = load_case(_auto_reacting(tmp_path, "Jet-A(L):1.0, O2:17.75, N2:66.7", "auto_jeta.yaml"))
    lib = net.gas.library
    assert "Jet-A(L)" in lib.species_names
    assert not lib.product_mask[lib.species_index["Jet-A(L)"]]  # feed-only, never a product
    assert lib.product_mask[lib.species_index["CO2"]]
    net.solve()


def test_reacting_inlet_requires_composition(tmp_path):
    g = {
        "thermoModel": "equilibrium",
        "mechanismFile": MECH,
        "species": "H2, O2, N2, H2O, OH, H, O, HO2",
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }
    nodes = [
        _node("in", "MassFlowInlet", 0, label="in", massFlowRate=1.0, totalTemperature=300.0),  # no composition
        _node("out", "PressureOutlet", 1, label="out", pressure=101325.0, backflowTotalTemperature=300.0),
    ]
    edges = [_edge("e1", "in", "out", 0, 0, 0, 0.05)]
    with pytest.raises(ValueError, match="composition"):
        load_case(_dump(tmp_path, "nocomp.yaml", g, nodes, edges)).compile()


# --------------------------------------------------------------------------- #
# 5. round-trip (save -> load) of a reacting case
# --------------------------------------------------------------------------- #
def test_reacting_roundtrip_resolves(tmp_path):
    net = load_case(_series_reacting(tmp_path, name="rt_src.yaml"))
    sol = net.solve()
    out = str(tmp_path / "rt.yaml")
    save_case(net, out, solution=sol)
    doc = yaml.safe_load(open(out))
    ga = doc["model"]["globalAttributes"]
    assert ga["thermoModel"] == "equilibrium"
    assert ga["mechanismFile"] == MECH  # preserved through provenance
    # the auto edges stay 'auto' (the marker closure handles the split internally -- no need to
    # bake the frozen/equilibrium labels into the file); the marker rides in as a 'burnt' dataset
    assert [e["attributes"]["thermoModel"] for e in doc["model"]["edges"]] == ["auto", "auto"]
    chem = next(d for d in doc["data"]["datasets"] if d["name"] == "Mean flow chemistry")
    burnt = next(it for it in chem["items"] if it["name"] == "burnt")
    assert burnt["values"][0] == pytest.approx(0.0, abs=1e-6)  # fresh approach
    assert burnt["values"][1] == pytest.approx(1.0, abs=1e-6)  # burnt downstream

    net2 = load_case(out)
    assert net2._edge_models == [None, None] and net2.compile().edge_model.tolist() == [EQ_MARKER, EQ_MARKER]
    est0 = states_table(net.compile(), sol.x)
    sol2 = net2.solve()
    est1 = states_table(net2.compile(), sol2.x)
    assert sol2.converged
    assert float(est1[ES_T, 1]) == pytest.approx(float(est0[ES_T, 1]), rel=1e-6)


def _series_reacting_taper(tmp_path, name="reacting_taper.yaml"):
    """inlet(H2/air) -> TaperedDuct (a composite) -> EquilibriumFlame -> outlet, equilibrium model."""
    g = {
        "thermoModel": "equilibrium",
        "mechanismFile": MECH,
        "species": "H2, O2, N2, H2O, OH, H, O, HO2",
        "equilibriumTInit": 2500.0,
        "frozenTInit": 300.0,
        "referencePressure": 101325.0,
        "referenceTemperature": 300.0,
        "referenceMassFlow": 1.0,
    }
    nodes = [
        _node("in", "MassFlowInlet", 0, label="fuel-air", massFlowRate=1.0, totalTemperature=300.0, composition=H2_AIR),
        _node("tap", "TaperedDuct", 1, label="taper", areaProfile="0:0.05, 0.1:0.04"),
        _node("flame", "EquilibriumFlame", 2, label="flame"),
        _node(
            "out",
            "PressureOutlet",
            3,
            label="out",
            pressure=101325.0,
            backflowTotalTemperature=300.0,
            composition=H2_AIR,
        ),
    ]
    edges = [
        _edge("e1", "in", "tap", 0, 0, 0, 0.05, thermoModel="auto"),
        _edge("e2", "tap", "flame", 1, 0, 1, 0.04, thermoModel="auto"),
        _edge("e3", "flame", "out", 1, 0, 2, 0.04, thermoModel="auto"),
    ]
    return _dump(tmp_path, name, g, nodes, edges)


def test_reacting_composite_roundtrip(tmp_path):
    """A reacting case carrying a composite (TaperedDuct) survives save + reload + re-solve.

    Exercises two composite-aware serialization paths: the per-edge chemistry export
    (``stream_mass_fractions`` must flatten the composite to read feed compositions, else it
    reads ``residual_id`` off the composite and raises) and the taper's station areas (which
    must reload bit-identical to the external edge areas, or the boundary duct's equal-area
    check fails on reload).  The reload also asserts ``species`` was emitted as a YAML list."""
    net = load_case(_series_reacting_taper(tmp_path))
    sol = net.solve()
    assert sol.converged
    out = str(tmp_path / "rt_composite.yaml")
    save_case(net, out, solution=sol)  # composite chemistry export must not raise

    doc = yaml.safe_load(open(out))
    assert isinstance(doc["model"]["globalAttributes"]["species"], list)  # not a joined string

    net2 = load_case(out)
    sol2 = net2.solve()
    assert sol2.converged
    est0 = states_table(net.compile(), sol.x)
    est1 = states_table(net2.compile(), sol2.x)
    assert float(est1[ES_T, -1]) == pytest.approx(float(est0[ES_T, -1]), rel=1e-6)


def test_parse_species_list_and_string():
    """The species slate reads from a YAML list (verbatim, so comma-bearing CEA names survive)
    or a comma/whitespace string (a hand-written / UI convenience for simple names)."""
    from nefes.io.yaml_in import _parse_species

    names = ["C2H2,acetylene", "N2", "CH3CHO,ethanal"]
    assert _parse_species(names) == names  # a list keeps each name verbatim, commas and all
    assert _parse_species("H2, O2, N2") == ["H2", "O2", "N2"]  # legacy comma/space string
    assert _parse_species("auto") == ["auto"]
    assert _parse_species(None) is None
