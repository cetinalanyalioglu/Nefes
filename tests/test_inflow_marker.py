"""User-settable inflow burnt-marker: inject already-burnt gas at a boundary.

A boundary's ``marker`` (``0`` fresh reactant, default; ``1`` fully burnt) is the burnt-marker
value the inflow/source stream carries into the marker-gated reacting closure.  Feeding ``1``
(e.g. exhaust-gas recirculation) forces the equilibrium closure on the inflow edge -- the premix
is burnt right at the inlet instead of staying frozen until the downstream flame.

These pin: the marker value lands on the inflow edge and forces equilibrium there; the default
stays fresh/frozen; the build guard rejects a non-zero marker off a marker-gated network; the
factory range check; and the YAML round-trip.
"""

import os

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.thermo.configure import equilibrium, perfect_gas
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.shell.network import Network
from nefes.solver.control import solve
from nefes.solver.report import states_table
from nefes.assembly.recover import ES_T

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data", "h2o2.yaml")
# stoichiometric H2-air premix (2 H2 + O2 + 3.76 N2, by mole)
PREMIX = {"H2": 0.296, "O2": 0.148, "N2": 0.556}
AREA = 0.01
E_INLET, E_OUT = 0, 1  # inlet -> flame -> outlet


def _premix_datum():
    from nefes.thermo import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_cantera(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    for sp, x in PREMIX.items():
        Y[idx[sp]] = x
    Y /= Y.sum()
    return gas, gas.enthalpy_mass(Y, 300.0)


def _prob(marker=0.0, edge_models=None):
    """inlet(premix, marker) -> equilibrium flame -> outlet (a marker-gated reacting net)."""
    gas, h_ref = _premix_datum()
    els = [
        cat.total_pressure_inlet(1.2e5, 300.0, composition=PREMIX, basis="mole", marker=marker),
        cat.equilibrium_flame(),
        cat.pressure_outlet(1.0e5, 300.0, composition=PREMIX, basis="mole"),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA)]
    return build_problem(
        equilibrium(gas.mech), els, edges, mdot_ref=0.4, p_ref=1e5, h_ref=h_ref, edge_models=edge_models
    )


def test_default_inlet_is_fresh_and_frozen():
    prob = _prob(marker=0.0)
    res = solve(prob)
    assert res.converged
    # default marker = 0: the inflow edge stays fresh (cold premix), burnt only downstream of the flame
    assert res.x[prob.marker_row, E_INLET] == pytest.approx(0.0, abs=1e-6)
    est = states_table(prob, res.x)
    assert est[ES_T, E_INLET] < 400.0  # cold reactant
    assert est[ES_T, E_OUT] > 1500.0  # burnt products


def test_burnt_marker_inlet_forces_equilibrium_on_inflow_edge():
    prob = _prob(marker=1.0)
    res = solve(prob)
    assert res.converged
    # marker = 1 lands on the inflow edge and forces the equilibrium closure: the premix is burnt
    # right at the inlet, so the inflow edge is already hot (no longer the cold frozen reactant)
    assert res.x[prob.marker_row, E_INLET] == pytest.approx(1.0, abs=1e-6)
    est = states_table(prob, res.x)
    assert est[ES_T, E_INLET] > 1500.0  # burnt at the inlet
    assert est[ES_T, E_OUT] > 1500.0


def test_marker_on_non_gated_network_raises():
    # explicit per-edge closure -> not marker-gated: a non-zero marker has no transport scalar
    with pytest.raises(ValueError, match="marker-gated"):
        _prob(marker=1.0, edge_models=[EQ_FROZEN, EQ_KERNEL])


def test_marker_on_perfect_gas_raises():
    net = Network(gas=perfect_gas(287.0, 1.4))
    net.add(cat.total_pressure_inlet(1.2e5, 300.0, marker=1.0))
    net.add(cat.pressure_outlet(1.0e5, 300.0))
    net.connect(0, 1, AREA)
    with pytest.raises(ValueError, match="marker-gated"):
        net.compile()


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_marker_out_of_range_rejected_at_factory(bad):
    with pytest.raises(ValueError, match=r"marker must be in \[0, 1\]"):
        cat.mass_flow_inlet(0.4, 300.0, composition=PREMIX, marker=bad)


def test_marker_written_to_yaml():
    # yaml_out emits the marker attribute (reacting-only, only when non-zero)
    import yaml as _yaml
    from nefes.io import dump_case

    gas, h_ref = _premix_datum()
    net = Network(
        gas=equilibrium(gas.mech),
        p_ref=1e5,
        T_ref=300.0,
        mdot_ref=0.4,
        h_ref=h_ref,
        nodes=[
            cat.total_pressure_inlet(1.2e5, 300.0, composition=PREMIX, basis="mole", marker=1.0, name="egr"),
            cat.equilibrium_flame(name="flame"),
            cat.pressure_outlet(1.0e5, 300.0, composition=PREMIX, basis="mole", name="out"),
        ],
        edges=[(0, 1, AREA), (1, 2, AREA)],
    )
    doc = _yaml.safe_load(dump_case(net))
    nodes = {n["attributes"]["label"]: n["attributes"] for n in doc["model"]["nodes"]}
    assert nodes["egr"]["marker"] == pytest.approx(1.0)  # burnt feed carries the marker
    assert "marker" not in nodes["out"]  # fresh (default) boundary omits it


def test_marker_read_from_yaml():
    # yaml_in parses the marker attribute back onto the ElementSpec
    from nefes.io.yaml_in import _UI_NODE_BUILDERS

    attrs = {"totalPressure": 1.2e5, "totalTemperature": 300.0, "composition": "H2:1.0", "marker": 1.0}
    spec = _UI_NODE_BUILDERS["TotalPressureInlet"](attrs)
    assert float(spec.marker) == pytest.approx(1.0)
    # absent marker -> fresh default
    attrs2 = {"totalPressure": 1.2e5, "totalTemperature": 300.0, "composition": "H2:1.0"}
    assert float(_UI_NODE_BUILDERS["TotalPressureInlet"](attrs2).marker) == 0.0
