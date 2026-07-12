"""Rich-quench-lean (staged) combustion on the auto marker-gated path.

A staged combustor burns fuel-rich in a primary zone (oxygen-limited, so the products
carry CO and H2), then injects fresh air to complete combustion in a lean zone.  The
lean gas is downstream of the flame, so it must stay in chemical equilibrium and burn
the leftover CO / H2 -- the quench air must *not* revert it to a frozen reactant.

The burnt marker is a sticky reachability label ("am I downstream of a flame?"),
transported by a noisy-OR so a fresh stream never dilutes a burnt one.  These pin that
the auto path re-equilibrates the lean zone with no manual per-edge closure, and that
it reproduces the explicit hard closure it is meant to automate.
"""

import numpy as np
import pytest

from nefes.assembly.recover import ES_MDOT, ES_T
from nefes.chem.chemistry import edge_species, product_moles, stream_mass_fractions
from nefes.chem.composition import enthalpy_mass, equivalence_ratio_mixture, species_mass_fractions
from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver.control import solve
from nefes.solver.report import states_table
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium

thermo = pytest.importorskip("nefes.thermo")

AIR = {"O2": 0.21, "N2": 0.79}
AREA = 0.01
MDOT_PRIMARY, MDOT_QUENCH = 0.20, 0.30
# node 0 inlet(rich premix) | 1 duct | 2 flame | 3 duct(rich) | 4 quench-air source | 5 duct(lean) | 6 outlet
E_APPROACH, E_RICH, E_LEAN = 1, 3, 5
# explicit hard closure: frozen approach, equilibrium from the flame onward (rich AND lean)
HARD_MODELS = [EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL, EQ_KERNEL, EQ_KERNEL]


def _lib():
    return thermo.SpeciesLibrary.from_cea(species=["CH4", "O2", "N2", "CO2", "H2O", "CO", "H2", "OH", "O", "H", "NO"])


def _elements(lib):
    rich = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.6)  # fuel-rich primary premix
    return [
        cat.mass_flow_inlet(MDOT_PRIMARY, 400.0, composition=rich, basis="mole"),
        cat.duct(0.3),
        cat.equilibrium_flame(),
        cat.duct(0.3),
        cat.mass_source(MDOT_QUENCH, 400.0, composition=AIR, basis="mole"),
        cat.duct(0.4),
        cat.pressure_outlet(1.0e5, 400.0, composition=AIR, basis="mole"),
    ]


def _edges():
    return [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA), (4, 5, AREA), (5, 6, AREA)]


def _prob(lib, edge_models=None):
    h_air = enthalpy_mass(lib, species_mass_fractions(lib, AIR, "mole"), 300.0)
    return build_problem(
        equilibrium(lib), _elements(lib), _edges(), mdot_ref=0.5, p_ref=1e5, h_ref=h_air, edge_models=edge_models
    )


def test_lean_zone_marker_is_sticky_not_diluted():
    lib = _lib()
    prob = _prob(lib)  # auto -> marker-gated
    res = solve(prob)
    assert res.converged
    b = res.x[prob.marker_row]
    est = states_table(prob, res.x)

    # approach fresh, rich + lean zones burnt -- the quench air does not un-burn the lean edge
    assert np.allclose(b[:2], 0.0, atol=1e-4)
    assert np.allclose(b[2:], 1.0, atol=1e-4)

    # stickiness does real work here: a mass-averaged marker at the quench node would be
    # the burnt/(burnt+fresh) mass fraction, which sits *below* the 0.5 gate crossover and
    # would wrongly report the lean edge as frozen.
    mdot_rich = abs(float(est[ES_MDOT, E_RICH]))
    naive_average = mdot_rich / (mdot_rich + MDOT_QUENCH)
    assert naive_average < 0.5  # naive averaging would gate the lean zone toward frozen
    assert b[E_LEAN] > 0.99  # the sticky OR keeps it burnt


def test_lean_zone_reequilibrates_the_rich_products():
    lib = _lib()
    prob = _prob(lib)
    res = solve(prob)
    assert res.converged

    sY = stream_mass_fractions(_elements(lib), lib)
    moles = product_moles(prob, res.x)
    rich = edge_species(prob, res.x, E_RICH, lib, basis="mole", moles=moles, stream_Y=sY)
    lean = edge_species(prob, res.x, E_LEAN, lib, basis="mole", moles=moles, stream_Y=sY)

    # rich zone: oxygen-limited, so CO and H2 survive and free O2 is essentially absent
    assert rich.get("CO", 0.0) > 0.05 and rich.get("H2", 0.0) > 0.05
    assert rich.get("O2", 0.0) < 1e-3

    # lean zone: the quench air burns the CO / H2 away and leaves excess O2
    assert lean.get("CO", 0.0) < 0.5 * rich["CO"]
    assert lean.get("H2", 0.0) < 0.5 * rich["H2"]
    assert lean.get("O2", 0.0) > 0.03
    assert lean.get("CO2", 0.0) > rich.get("CO2", 0.0)


def test_auto_marker_matches_explicit_hard_closure():
    # The sticky marker automates exactly the hard closure a user would otherwise wire by
    # hand (frozen approach, equilibrium everywhere downstream), so the mean flow matches.
    lib = _lib()
    auto = solve(_prob(lib))
    hard = solve(_prob(lib, edge_models=HARD_MODELS))
    assert auto.converged and hard.converged
    ea = states_table(_prob(lib), auto.x)
    eh = states_table(_prob(lib, edge_models=HARD_MODELS), hard.x)
    for q in (ES_MDOT, ES_T):
        assert np.allclose(ea[q], eh[q], rtol=1e-5, atol=1e-5 * np.abs(eh[q]).max())


def test_burnt_survives_a_fresh_mixing_junction():
    # The quench air here enters as its own inlet and merges at a junction (rather than an
    # inline mass source), exercising the JUNCTION noisy-OR: burnt rich products + fresh
    # air -> the merged (lean) edge stays burnt.
    lib = _lib()
    rich = equivalence_ratio_mixture(lib, {"CH4": 1.0}, AIR, 1.6)
    els = [
        cat.mass_flow_inlet(MDOT_PRIMARY, 400.0, composition=rich, basis="mole"),  # 0
        cat.equilibrium_flame(),  # 1
        cat.junction(),  # 2  rich products meet quench air
        cat.mass_flow_inlet(MDOT_QUENCH, 400.0, composition=AIR, basis="mole"),  # 3
        cat.pressure_outlet(1.0e5, 400.0, composition=AIR, basis="mole"),  # 4
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (3, 2, AREA), (2, 4, AREA)]  # e3 = merged (lean) edge
    h_air = enthalpy_mass(lib, species_mass_fractions(lib, AIR, "mole"), 300.0)
    prob = build_problem(equilibrium(lib), els, edges, mdot_ref=0.5, p_ref=1e5, h_ref=h_air)
    res = solve(prob)
    assert res.converged
    b = res.x[prob.marker_row]
    est = states_table(prob, res.x)
    # merged edge stays burnt despite the fresh-air dilution, and runs hot (re-equilibrated)
    assert b[3] > 0.99
    assert est[ES_T, 3] > 1500.0
