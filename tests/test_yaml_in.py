"""UI-export (YAML) loader: thermo-model selection, new elements, reacting wiring.

The loader (:mod:`fns.io.yaml_in`) is the contract between the FNetLibUI tool and the
solver.  These tests cover the additions that bring it in step with the catalog: the
``thermoModel`` selector (perfect gas vs equilibrium), the string-encoded compositions,
the per-edge frozen/equilibrium closure (auto-from-flames + explicit override), the
inlet ``inherit`` acoustic default, and the new HeatReleaseFlame / MassSource /
EquilibriumFlame elements.  The reacting cases need the bundled thermolib mechanism.
"""

import os

import pytest
import yaml

from fns.io import load_case, save_case
from fns.io.yaml_in import _parse_composition
from fns.solver.control import states_table
from fns.derive import ES_T, ES_RHO, ES_MDOT, ES_HT
from fns.thermo.api import EQ_FROZEN, EQ_KERNEL, PERFECT_GAS

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")
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
    # auto: frozen (unburnt) upstream of the flame, equilibrium (burnt) downstream
    assert net._edge_models == [EQ_FROZEN, EQ_KERNEL]
    sol = net.solve()
    assert sol.converged
    est = states_table(net.compile(), sol.x)
    assert float(est[ES_T, 0]) == pytest.approx(300.0, abs=1.0)  # unburnt
    assert float(est[ES_T, 1]) > 2000.0  # burnt
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
    assert net._edge_models == [EQ_KERNEL]


def test_reacting_burnt_matches_standalone_equilibrium(tmp_path):
    from thermolib import SpeciesLibrary, Thermo
    from fns.composition import resolve_composition, enthalpy_mass

    net = load_case(_series_reacting(tmp_path, name="ref.yaml"))
    sol = net.solve()
    est = states_table(net.compile(), sol.x)

    lib = SpeciesLibrary.from_native(MECH).subset(["H2", "O2", "N2", "H2O", "OH", "H", "O", "HO2"])
    gas = Thermo(lib)
    Y, Z = resolve_composition(lib, {"H2": 1.0, "O2": 0.5, "N2": 1.88}, basis="mole")
    h = enthalpy_mass(lib, Y, 300.0)
    ref = gas.equilibrate_HP(Z, h, float(est[1, 1]))  # at the burnt static pressure (ES_P row 1)
    assert float(est[ES_T, 1]) == pytest.approx(ref.T, rel=1e-3)


# --------------------------------------------------------------------------- #
# 4. reacting error paths
# --------------------------------------------------------------------------- #
def test_reacting_without_mechanism_or_species_is_rejected(tmp_path):
    # No mechanismFile falls back to the packaged thermo.inp, which needs a species list to
    # narrow its thousands of species -- so neither given is the error.
    g = {"thermoModel": "equilibrium"}
    nodes = [_node("in", "MassFlowInlet", 0, massFlowRate=1.0, totalTemperature=300.0, composition="H2:1")]
    nodes.append(_node("out", "PressureOutlet", 1, pressure=1e5))
    edges = [_edge("e1", "in", "out", 0, 0, 0, 0.05)]
    with pytest.raises(ValueError, match="species"):
        load_case(_dump(tmp_path, "nomech.yaml", g, nodes, edges))


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
    assert [e["attributes"]["thermoModel"] for e in doc["model"]["edges"]] == ["frozen", "equilibrium"]

    net2 = load_case(out)
    assert net2._edge_models == [EQ_FROZEN, EQ_KERNEL]
    est0 = states_table(net.compile(), sol.x)
    sol2 = net2.solve()
    est1 = states_table(net2.compile(), sol2.x)
    assert sol2.converged
    assert float(est1[ES_T, 1]) == pytest.approx(float(est0[ES_T, 1]), rel=1e-6)
