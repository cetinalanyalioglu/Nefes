"""Per-edge species / molar-mass / cp output for solved networks.

The mean-flow solve transports feed-stream mixture fractions, not species.  These tests
cover the post-solve recovery of the actual chemistry: molar mass ``W`` and specific heat
``cp`` as standard edge fields, the species composition per edge (equilibrium products on a
burnt edge, the feed blend on an unburnt edge), the transported mixture fractions, and the
default-on YAML emission.
"""

import os

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.shell.network import Network
from nefes.thermo.configure import equilibrium, perfect_gas

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")
AIR = {"O2": 0.21, "N2": 0.79}
R_AIR, GAMMA = 287.0, 1.4
RU = 8.314462618


def _reacting_rig():
    """air -> duct -> H2 injector -> duct -> equilibrium flame -> duct -> outlet."""
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_cantera(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    Y[idx["O2"]], Y[idx["N2"]] = 0.21, 0.79
    Y /= Y.sum()
    h_air = gas.enthalpy_mass(Y, 300.0)

    net = Network(gas=equilibrium(gas.mech), p_ref=1e5, T_ref=300.0, mdot_ref=0.4, h_ref=h_air)
    A = 0.01
    i = net.add(cat.total_pressure_inlet(1.2e5, 300.0, composition=AIR, basis="mole", name="air"))
    d1 = net.add(cat.duct(0.4))
    src = net.add(cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, basis="mole", name="H2"))
    d2 = net.add(cat.duct(0.4))
    fl = net.add(cat.equilibrium_flame())
    d3 = net.add(cat.duct(0.5))
    o = net.add(cat.mass_flow_outlet(0.406))
    for a, b in [(i, d1), (d1, src), (src, d2), (d2, fl), (fl, d3), (d3, o)]:
        net.connect(a, b, A)
    return net


def _perfect_duct():
    net = Network(gas=perfect_gas(R_AIR, GAMMA), p_ref=1e5, T_ref=300.0, mdot_ref=2.0)
    a = net.add(cat.total_pressure_inlet(1.2e5, 300.0))
    d = net.add(cat.duct(0.5))
    o = net.add(cat.pressure_outlet(1.0e5, 300.0))
    net.connect(a, d, 0.05)
    net.connect(d, o, 0.05)
    return net


# --------------------------------------------------------------------------- #
# Molar mass and cp as standard edge fields
# --------------------------------------------------------------------------- #
def test_perfect_gas_W_and_cp():
    sol = _perfect_duct().solve()
    assert sol.converged
    cp_exact = GAMMA * R_AIR / (GAMMA - 1.0)
    assert np.allclose(sol.field("W"), RU / R_AIR, rtol=1e-6)
    assert np.allclose(sol.field("cp"), cp_exact, rtol=1e-6)
    # the edge() dict surfaces them next to T, c, M
    assert "W" in sol.edge(0) and "cp" in sol.edge(0)


def test_reacting_W_and_cp_track_composition():
    sol = _reacting_rig().solve()
    assert sol.converged
    W, cp, T = sol.field("W"), sol.field("cp"), sol.field("T")
    # unburnt air ~ 28.85 g/mol, cp ~ 1.0 kJ/kg/K; burnt products are hotter and lighter
    assert W[0] == pytest.approx(0.02885, abs=5e-4)
    assert cp[0] == pytest.approx(1010.0, rel=0.05)
    assert T[-1] > 1200.0  # flame ignited
    assert W[-1] < W[0]  # combustion lightens the mixture (H2 -> H2O, dissociation)
    assert cp[-1] > cp[0]  # hot products have a higher specific heat


# --------------------------------------------------------------------------- #
# Species composition
# --------------------------------------------------------------------------- #
def test_frozen_edge_reports_feed_blend():
    sol = _reacting_rig().solve()
    x_air = sol.species(0, basis="mole")  # edge 0 is the unburnt air feed
    assert x_air["O2"] == pytest.approx(0.21, abs=1e-3)
    assert x_air["N2"] == pytest.approx(0.79, abs=1e-3)
    assert sum(x_air.values()) == pytest.approx(1.0, abs=1e-6)
    assert x_air.get("H2O", 0.0) < 1e-6  # unburnt: no products


def test_burnt_edge_reports_equilibrium_products():
    sol = _reacting_rig().solve()
    x = sol.species(5, basis="mole")  # the burnt edge
    assert x["H2O"] > 0.1  # the dominant product of H2 combustion
    assert x["N2"] > 0.5  # the inert diluent carries through
    assert sum(x.values()) == pytest.approx(1.0, abs=1e-6)
    # mass-basis fractions also sum to one and differ from mole fractions
    y = sol.species(5, basis="mass")
    assert sum(y.values()) == pytest.approx(1.0, abs=1e-6)
    assert y["H2O"] != pytest.approx(x["H2O"], abs=1e-3)


def test_perfect_gas_has_no_species():
    sol = _perfect_duct().solve()
    assert sol.species(0) == {}


def test_mixture_fractions():
    sol = _reacting_rig().solve()
    mf0 = sol.mixture_fractions(0)
    assert mf0["air"] == pytest.approx(1.0, abs=1e-6)
    assert mf0["H2"] == pytest.approx(0.0, abs=1e-6)
    # downstream of the H2 injector the fuel fraction is nonzero
    assert sol.mixture_fractions(5)["H2"] > 0.0
    assert sum(sol.mixture_fractions(5).values()) == pytest.approx(1.0, abs=1e-6)


def test_invalid_basis_raises():
    sol = _reacting_rig().solve()
    with pytest.raises(ValueError, match="basis"):
        sol.species(5, basis="volume")


# --------------------------------------------------------------------------- #
# YAML output (default-on)
# --------------------------------------------------------------------------- #
def test_yaml_includes_chemistry_by_default(tmp_path):
    sol = _reacting_rig().solve()
    path = tmp_path / "case.yaml"
    sol.to_yaml(str(path))
    txt = path.read_text()
    assert "Molar mass" in txt and "Specific heat" in txt  # W, cp fields
    assert "Chemistry" in txt  # the chemistry dataset
    assert "xi:air" in txt and "xi:H2" in txt  # mixture fractions
    assert "X:H2O" in txt and "X:N2" in txt  # species mole fractions


def test_yaml_perfect_gas_has_no_chemistry_dataset(tmp_path):
    sol = _perfect_duct().solve()
    path = tmp_path / "case.yaml"
    sol.to_yaml(str(path))
    txt = path.read_text()
    assert "Molar mass" in txt  # W still emitted
    assert "Chemistry" not in txt  # no composition transported
