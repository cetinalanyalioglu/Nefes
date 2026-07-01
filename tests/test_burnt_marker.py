"""Burnt-marker scalar closure: orientation-robust frozen/equilibrium gating.

A reacting network with an equilibrium flame and no explicit per-edge closure is *marker-gated*:
every edge runs ``EQ_MARKER`` and one transported "burnt" marker scalar gates the frozen
(unburnt) / equilibrium (burnt) blend.  The marker rides the *signed* mass flow, so it labels
"downstream of a flame" robustly, regardless of how the edges were drawn -- the topology
flood-fill is demoted to the marker's initial guess.

These pin: the compiled structure, parity with the explicit hard closure (mean flow *and*
acoustics), the seed-independent self-correction, graceful degradation on a both-in flame draw,
and the marker post-processing.
"""

import os

import numpy as np
import pytest

from nefes.elements import catalog as cat
from nefes.thermo.configure import equilibrium
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL, EQ_MARKER, PERFECT_GAS
from nefes.shell.network import Network
from nefes.solver.control import solve, states_table, auto_initial_guess
from nefes.assembly.derive import ES_T, ES_RHO, ES_MDOT, ES_P
from nefes.perturbation.response.response import perturbation_response

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")
AIR = {"O2": 0.21, "N2": 0.79}
AREA = 0.01
# node 0 inlet(air) | 1 duct | 2 H2 source | 3 duct | 4 flame | 5 duct | 6 outlet
HARD_MODELS = [EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_KERNEL, EQ_KERNEL]
E_AIR, E_APPROACH, E_FLAME_OUT = 0, 3, 4


def _air_datum():
    from thermolib import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_native(MECH)
    gas = Thermo(lib)
    idx = lib.species_index
    Y = np.zeros(lib.n_species)
    Y[idx["O2"]], Y[idx["N2"]] = 0.21, 0.79
    Y /= Y.sum()
    return gas, gas.enthalpy_mass(Y, 300.0)


def _elements(gas):
    return [
        cat.total_pressure_inlet(1.2e5, 300.0, composition=AIR, basis="mole"),
        cat.duct(0.4),
        cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, basis="mole"),
        cat.duct(0.4),
        cat.equilibrium_flame(),
        cat.duct(0.5),
        cat.mass_flow_outlet(0.406),
    ]


def _edges():
    return [(0, 1, AREA), (1, 2, AREA), (2, 3, AREA), (3, 4, AREA), (4, 5, AREA), (5, 6, AREA)]


def _prob(edge_models):
    gas, h_air = _air_datum()
    return cat.build_problem(
        equilibrium(gas.mech), _elements(gas), _edges(), mdot_ref=0.4, p_ref=1e5, h_ref=h_air, edge_models=edge_models
    )


# -- compiled structure -----------------------------------------------------


def test_auto_reacting_network_is_marker_gated():
    prob = _prob(edge_models=None)  # auto -> marker-gated
    assert prob.edge_model.tolist() == [EQ_MARKER] * 6
    assert prob.n_solve == 3 + prob.n_elem + 1  # +1 for the marker
    assert prob.marker_row == 3 + prob.n_elem
    # the flood-fill seed labels the two post-flame edges burnt (declared flow-aligned)
    assert prob.marker_seed.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_explicit_models_carry_no_marker():
    prob = _prob(edge_models=HARD_MODELS)  # hard closure escape hatch
    assert prob.edge_model.tolist() == HARD_MODELS
    assert prob.n_solve == 3 + prob.n_elem  # no marker row
    assert prob.marker_row == -1 and prob.marker_seed is None


def test_perfect_gas_has_no_marker():
    from nefes.thermo.configure import perfect_gas

    net = Network(gas=perfect_gas(287.0, 1.4))
    net.add(cat.total_pressure_inlet(1.2e5, 300.0))
    net.add(cat.pressure_outlet(1.0e5, 300.0))
    net.connect(0, 1, AREA)
    prob = net.compile()
    assert prob.model_id == PERFECT_GAS and prob.marker_row == -1 and prob.n_solve == 3


# -- parity with the explicit hard closure (mean flow) ----------------------


def test_mean_flow_matches_hard_closure():
    marker = solve(_prob(edge_models=None))
    hard = solve(_prob(edge_models=HARD_MODELS))
    assert marker.converged and hard.converged
    pm, ph = _prob(edge_models=None), _prob(edge_models=HARD_MODELS)
    em, eh = states_table(pm, marker.x), states_table(ph, hard.x)
    for q in (ES_MDOT, ES_P, ES_T, ES_RHO):
        assert np.allclose(em[q], eh[q], rtol=1e-6, atol=1e-6 * np.abs(eh[q]).max())
    # the marker is exactly bimodal at convergence: fresh approach, burnt downstream
    b = marker.x[pm.marker_row]
    assert np.allclose(b[:4], 0.0, atol=1e-8)
    assert np.allclose(b[4:], 1.0, atol=1e-8)


# -- the seed is only a guess: the transport self-corrects -------------------


@pytest.mark.parametrize("seed", [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
def test_marker_self_corrects_any_seed(seed):
    prob = _prob(edge_models=None)
    x0 = auto_initial_guess(prob)
    x0[prob.marker_row, :] = seed  # deliberately wrong / scrambled marker guess
    res = solve(prob, x0=x0)
    assert res.converged
    b = res.x[prob.marker_row]
    assert np.allclose(b[:4], 0.0, atol=1e-6) and np.allclose(b[4:], 1.0, atol=1e-6)
    est = states_table(prob, res.x)
    assert est[ES_T, E_APPROACH] < 400.0 and est[ES_T, E_FLAME_OUT] > 1200.0


def test_both_in_flame_draw_degrades_gracefully():
    # both flame edges drawn INTO the flame (out-degree 0): the worst auto-start.  The marker
    # closure must not crash -- it degrades to a non-converged result instead of raising.
    gas, h_air = _air_datum()
    els = [
        cat.total_pressure_inlet(1.2e5, 300.0, composition=AIR, basis="mole"),
        cat.mass_source(0.006, 300.0, composition={"H2": 1.0}, basis="mole"),
        cat.equilibrium_flame(),
        cat.pressure_outlet(1.0e5, 300.0, composition=AIR, basis="mole"),
    ]
    edges = [(0, 1, AREA), (1, 2, AREA), (3, 2, AREA)]  # edge 2 reversed -> flame is both-in
    prob = cat.build_problem(equilibrium(gas.mech), els, edges, mdot_ref=0.4, p_ref=1e5, h_ref=h_air)
    res = solve(prob)  # must return, not raise
    assert not res.converged


# -- parity with the explicit hard closure (acoustics) ----------------------


def test_acoustic_response_matches_hard_closure():
    # the marker is acoustically passive (a decoupled convected scalar), so the acoustic transfer
    # matrix across the flame is identical to the explicit hard-closure network's.
    freqs = np.array([120.0, 350.0, 600.0])
    pm, ph = _prob(edge_models=None), _prob(edge_models=HARD_MODELS)
    xm, xh = solve(pm).x, solve(ph).x
    rm = perturbation_response(pm, xm, freqs, excite=("acoustic", "entropy"))
    rh = perturbation_response(ph, xh, freqs, excite=("acoustic", "entropy"))
    Tm = rm.transfer_matrix(E_AIR, E_FLAME_OUT)
    Th = rh.transfer_matrix(E_AIR, E_FLAME_OUT)
    assert np.allclose(Tm, Th, rtol=1e-4, atol=1e-6)


# -- marker post-processing -------------------------------------------------


def test_marker_surfaced_and_species_blended():
    net = Network(
        gas=equilibrium(_air_datum()[0].mech),
        p_ref=1e5,
        T_ref=300.0,
        mdot_ref=0.4,
        h_ref=_air_datum()[1],
        nodes=_elements(_air_datum()[0]),
        edges=_edges(),
    )
    sol = net.solve()
    assert sol.converged
    # marker accessor: fresh approach -> 0, burnt downstream -> 1
    assert sol.marker(E_APPROACH) == pytest.approx(0.0, abs=1e-6)
    assert sol.marker(E_FLAME_OUT) == pytest.approx(1.0, abs=1e-6)
    # species: the burnt edge reports equilibrium products, the fresh edge the frozen reactants
    assert sol.species(E_FLAME_OUT, basis="mole").get("H2O", 0.0) > 0.1
    assert "H2O" not in sol.species(E_APPROACH, basis="mole")
    # molar mass and cp are surfaced for every edge
    assert np.all(sol.field("W") > 0.0) and np.all(sol.field("cp") > 0.0)
