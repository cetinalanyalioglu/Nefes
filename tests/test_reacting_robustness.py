"""Overall robustness of the reacting solve from the auto-seed alone.

Every network here is solved with a bare ``solve(prob)`` -- no hand-built initial
guess, no per-case tuning.  The solver's graph-propagated seed
(``auto_initial_guess``) places each edge at its adiabatic-mixing ``(mdot, h_t,
xi)``, so convergence comes from one generic mechanism rather than per-network
hints.  The cases span a wide operating envelope, a branched (junction-mixed)
topology, and co-injection of two *carbon-bearing* fuels -- the mixture the old
elemental-``Z`` basis could not even build (it is rank-deficient over C,H,O,N),
now exact because each feed is its own transported mixture fraction.
"""

import os

import pytest

from nefes.assembly.recover import ES_HT, ES_MDOT, ES_T, ES_U
from nefes.chem.composition import enthalpy_mass, resolve_composition
from nefes.elements import catalog as cat
from nefes.shell.build import build_problem
from nefes.solver import solve
from nefes.solver.report import states_table
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.configure import equilibrium

A = 0.1
AIR = {"O2": 0.21, "N2": 0.79}
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data")
THERMO_INP = os.path.join(DATA, "thermo.inp")


def _lib():
    from nefes.thermo import ThermoInp

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    heavy = "C8H18,n-octane"
    species = ["O2", "N2", "CH4", heavy, "CO2", "H2O", "CO", "OH", "H", "O", "NO", "H2"]
    return ThermoInp(THERMO_INP).library(species), heavy


def _hp_reference(gas, lib, feeds, p):
    """Adiabatic-mixing reference: (Z, h_t) of mass-weighted feeds and its HP flame T."""
    mtot = sum(m for m, _Y, _T in feeds)
    Ymix = sum(m * Y for m, Y, _T in feeds) / mtot
    h_mix = sum(m * enthalpy_mass(lib, Y, T) for m, Y, T in feeds) / mtot
    Zmix = gas.elemental_mass_fractions(Ymix)
    ref = gas.equilibrate_HP(Zmix, h_mix, p, T_guess=2000.0)
    return h_mix, ref


@pytest.mark.parametrize("mdot_fuel", [0.030, 0.058, 0.075])
@pytest.mark.parametrize("Tin", [300.0, 650.0])
@pytest.mark.parametrize("p", [1.0e5, 6.0e5])
def test_operating_envelope_converges_from_seed(mdot_fuel, Tin, p):
    """air -> CH4 source -> flame -> outlet, swept lean..rich, cold..hot, low..high p.
    Each point solves from the auto-seed and lands on the standalone HP flame T."""
    from nefes.thermo import Thermo

    lib, _heavy = _lib()
    gas = Thermo(lib)
    mdot_air = 1.0
    Yair, _ = resolve_composition(lib, AIR, basis="mole")
    Yfuel, _ = resolve_composition(lib, {"CH4": 1.0}, basis="mole")
    h_mix, ref = _hp_reference(gas, lib, [(mdot_air, Yair, Tin), (mdot_fuel, Yfuel, Tin)], p)
    assert ref.converged

    cfg = equilibrium(lib)
    els = [
        cat.mass_flow_inlet(mdot_air, Tin, composition=AIR, basis="mole", name="air"),
        cat.mass_source(mdot_fuel, Tin, composition={"CH4": 1.0}, basis="mole", name="fuel"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=Tin, composition=AIR, basis="mole", name="out"),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A)]
    prob = build_problem(
        cfg,
        els,
        edges,
        mdot_ref=mdot_air,
        p_ref=p,
        h_ref=max(abs(h_mix), 1.0e3),
        edge_models=[EQ_FROZEN, EQ_FROZEN, EQ_KERNEL],
    )
    res = solve(prob)
    assert res.converged, (mdot_fuel, Tin, p)
    est = states_table(prob, res.x)
    assert est[ES_MDOT, 2] == pytest.approx(mdot_air + mdot_fuel, rel=1e-6)
    assert est[ES_HT, 2] == pytest.approx(h_mix, rel=1e-5)
    assert est[ES_T, 2] == pytest.approx(ref.T, rel=3e-3)


def test_two_carbon_fuels_co_injected():
    """CH4 and C8H18 co-injected into air, then burned -- a mixture the elemental
    basis could not resolve (rank-deficient over C,H,O,N).  With one mixture
    fraction per feed the network builds, solves from the seed, and the burnt T
    matches the HP equilibrium of the combined mixture."""
    from nefes.thermo import Thermo

    lib, heavy = _lib()
    gas = Thermo(lib)
    mdot_air, mdot_ch4, mdot_oct, Tin, p = 1.0, 0.025, 0.020, 350.0, 2.0e5
    Yair, _ = resolve_composition(lib, AIR, basis="mole")
    Ych4, _ = resolve_composition(lib, {"CH4": 1.0}, basis="mole")
    Yoct, _ = resolve_composition(lib, {heavy: 1.0}, basis="mole")
    h_mix, ref = _hp_reference(gas, lib, [(mdot_air, Yair, Tin), (mdot_ch4, Ych4, Tin), (mdot_oct, Yoct, Tin)], p)
    assert ref.converged

    cfg = equilibrium(lib)
    els = [
        cat.mass_flow_inlet(mdot_air, Tin, composition=AIR, basis="mole", name="air"),
        cat.mass_source(mdot_ch4, Tin, composition={"CH4": 1.0}, basis="mole", name="ch4"),
        cat.mass_source(mdot_oct, Tin, composition={heavy: 1.0}, basis="mole", name="oct"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=Tin, composition=AIR, basis="mole", name="out"),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A), (3, 4, A)]
    prob = build_problem(
        cfg,
        els,
        edges,
        mdot_ref=mdot_air,
        p_ref=p,
        h_ref=abs(h_mix),
        edge_models=[EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_KERNEL],
    )
    assert cfg.n_elem == 0  # streams are discovered at build, not at config time
    assert prob.n_elem == 3  # air + CH4 + C8H18 -> three transported mixture fractions
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    assert est[ES_MDOT, 3] == pytest.approx(mdot_air + mdot_ch4 + mdot_oct, rel=1e-6)
    assert est[ES_T, 3] == pytest.approx(ref.T, rel=3e-3)


def test_branched_mixing_converges_from_seed():
    """Two air streams at different temperatures merge at a junction, then a fuel
    source + flame.  Exercises the seed's mass-weighted propagation *through a
    junction* (not just a series chain)."""
    from nefes.thermo import Thermo

    lib, _heavy = _lib()
    gas = Thermo(lib)
    mdotA, TA = 0.7, 300.0
    mdotB, TB = 0.5, 750.0
    mdot_fuel, p = 0.05, 1.5e5
    Yair, _ = resolve_composition(lib, AIR, basis="mole")
    Yfuel, _ = resolve_composition(lib, {"CH4": 1.0}, basis="mole")
    h_mix, ref = _hp_reference(gas, lib, [(mdotA, Yair, TA), (mdotB, Yair, TB), (mdot_fuel, Yfuel, 300.0)], p)
    assert ref.converged

    cfg = equilibrium(lib)
    els = [
        cat.mass_flow_inlet(mdotA, TA, composition=AIR, basis="mole", name="airA"),
        cat.mass_flow_inlet(mdotB, TB, composition=AIR, basis="mole", name="airB"),
        cat.junction(name="mix"),
        cat.mass_source(mdot_fuel, 300.0, composition={"CH4": 1.0}, basis="mole", name="fuel"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=300.0, composition=AIR, basis="mole", name="out"),
    ]
    # airA(0) -\                          fuel(3) -> flame(4) -> out(5)
    #           >-- junction(2) --------->/
    # airB(1) -/
    edges = [(0, 2, A), (1, 2, A), (2, 3, A), (3, 4, A), (4, 5, A)]
    # frozen up to and including the fuel-laden edge; burnt after the flame
    edge_models = [EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_FROZEN, EQ_KERNEL]
    prob = build_problem(cfg, els, edges, mdot_ref=mdotA + mdotB, p_ref=p, h_ref=abs(h_mix), edge_models=edge_models)
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    # the two air streams (air is one stream) merge: mass adds through the junction
    assert est[ES_MDOT, 4] == pytest.approx(mdotA + mdotB + mdot_fuel, rel=1e-6)
    assert est[ES_T, 4] == pytest.approx(ref.T, rel=4e-3)


def test_carbonless_burn_in_carbon_library():
    """A hydrogen flame solved in a carbon-bearing library: the burnt edge's elemental
    abundance has a zero carbon entry (the parallel-branch case), so the equilibrium
    kernel must drop carbon and its species to stay non-singular.  The network solves
    from the auto-seed and the burnt static T matches a standalone HP equilibrium."""
    from nefes.thermo import Thermo

    lib, _heavy = _lib()  # library carries CH4 / CO2 / CO -> carbon is an element
    gas = Thermo(lib)
    mdot_air, mdot_h2, Tin, p = 1.0, 0.029, 300.0, 2.0e5
    Yair, _ = resolve_composition(lib, AIR, basis="mole")
    Yh2, _ = resolve_composition(lib, {"H2": 1.0}, basis="mole")
    h_mix, ref = _hp_reference(gas, lib, [(mdot_air, Yair, Tin), (mdot_h2, Yh2, Tin)], p)
    assert ref.converged

    cfg = equilibrium(lib)
    els = [
        cat.mass_flow_inlet(mdot_air, Tin, composition=AIR, basis="mole", name="air"),
        cat.mass_source(mdot_h2, Tin, composition={"H2": 1.0}, basis="mole", name="h2"),
        cat.equilibrium_flame(name="flame"),
        cat.pressure_outlet(p, Tt_backflow=Tin, composition=AIR, basis="mole", name="out"),
    ]
    edges = [(0, 1, A), (1, 2, A), (2, 3, A)]
    prob = build_problem(
        cfg, els, edges, mdot_ref=mdot_air, p_ref=p, h_ref=abs(h_mix), edge_models=[EQ_FROZEN, EQ_FROZEN, EQ_KERNEL]
    )
    res = solve(prob)
    assert res.converged
    est = states_table(prob, res.x)
    # burnt static T == HP equilibrium at the static enthalpy (carbon dropped in the masked solve)
    u = est[ES_U, 2]
    h_static = est[ES_HT, 2] - 0.5 * u * u
    Z = gas.elemental_mass_fractions((mdot_air * Yair + mdot_h2 * Yh2) / (mdot_air + mdot_h2))
    ref_static = gas.equilibrate_HP(Z, h_static, est[1, 2], T_guess=2000.0)
    assert est[ES_T, 2] == pytest.approx(ref_static.T, rel=3e-3)


def _ch4_air_lib():
    from nefes.thermo import ThermoInp

    return ThermoInp().library(["CH4", "O2", "N2", "CO2", "H2O", "CO", "OH", "H2", "H", "O", "NO"])


def test_high_pressure_reacting_converges_from_boundary_pressure_seed():
    """A 200-bar reacting solve converges from a bare cold start.

    The default seed reads the network's own boundary pressures (here a 200/190-bar
    total-pressure inlet and static outlet) rather than the gauge reference, so the very
    high-pressure operating point is reached with no warm start and no hand-built guess.
    """
    import nefes
    from nefes.chem import equivalence_ratio_mixture

    lib = _ch4_air_lib()
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    sol = nefes.Network(
        nefes.equilibrium(lib),
        nodes=[
            cat.total_pressure_inlet(2.0e7, 300.0, composition=mix, name="feed"),
            cat.equilibrium_flame(name="flame"),
            cat.pressure_outlet(1.9e7, 300.0, composition=mix, name="out"),
        ],
        edges=[(0, 1, 0.05), (1, 2, 0.05)],
        edge_models=[EQ_FROZEN, EQ_KERNEL],
    ).solve()
    assert sol.converged
    assert sol.edge(1)["p"] > 1.5e7  # actually solved at high pressure, not the 1-bar reference
    assert 2000.0 < sol.edge(1)["T"] < 2600.0  # physical adiabatic flame temperature


def test_choked_chamber_emergent_pressure_seed_converges():
    """A reacting chamber whose pressure is set implicitly by a choked nozzle converges cold.

    There is no boundary pressure to read; the seed estimates the chamber pressure from the
    nozzle's critical-mass-flux relation.  A rising mass flow raises the emergent chamber
    pressure into the multi-hundred-bar range, all from a bare ``solve``.
    """
    import nefes
    from nefes.chem import equivalence_ratio_mixture

    lib = _ch4_air_lib()
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    prev_pc = 0.0
    for mdot in (10.0, 50.0, 100.0):
        sol = nefes.Network(
            nefes.equilibrium(lib),
            nodes=[
                cat.mass_flow_inlet(mdot, 300.0, composition=mix, name="inj"),
                cat.equilibrium_flame(name="flame"),
                cat.choked_nozzle_outlet(5.0e-4, name="throat"),
            ],
            edges=[(0, 1, 0.05), (1, 2, 0.05)],
            edge_models=[EQ_FROZEN, EQ_KERNEL],
        ).solve()
        assert sol.converged, f"choked chamber failed cold at mdot={mdot}"
        pc = sol.edge(1)["p"]
        assert pc > prev_pc > -1.0  # chamber pressure rises with mass flow, monotone
        prev_pc = pc
    assert prev_pc > 1.0e8  # mdot=100 reaches >1000 bar, cold, no warm start


def test_reacting_recovery_never_raises_on_a_hard_cold_start():
    """The density-based kinetic-energy recovery returns a Solution instead of raising.

    An extreme cold start that used to raise ``kinetic-energy bracket expansion failed`` now
    yields a (possibly non-converged) Solution -- the recovery is unconditionally bracketable,
    like the perfect-gas path, so a bad intermediate iterate never kills the solve.
    """
    import nefes
    from nefes.chem import equivalence_ratio_mixture

    lib = _ch4_air_lib()
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    sol = nefes.Network(
        nefes.equilibrium(lib),
        nodes=[
            cat.mass_flow_inlet(200.0, 300.0, composition=mix, name="inj"),
            cat.equilibrium_flame(name="flame"),
            cat.choked_nozzle_outlet(5.0e-4, name="throat"),
        ],
        edges=[(0, 1, 0.05), (1, 2, 0.05)],
        edge_models=[EQ_FROZEN, EQ_KERNEL],
    ).solve()  # must not raise
    assert sol.converged  # and with the boundary-pressure seed it also converges


def _ch4_air_reacting_network():
    import nefes
    from nefes.chem import equivalence_ratio_mixture

    lib = _ch4_air_lib()
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    return nefes.Network(
        nefes.equilibrium(lib),
        nodes=[
            cat.mass_flow_inlet(1.0, 300.0, composition=mix, name="feed"),
            cat.equilibrium_flame(name="flame"),
            cat.pressure_outlet(101325.0, 300.0, composition=mix, name="out"),
        ],
        edges=[(0, 1, 0.05), (1, 2, 0.05)],
        edge_models=[EQ_FROZEN, EQ_KERNEL],
    )


def test_reacting_initial_guess_is_evaluable_and_solves():
    """``net.initial_guess()`` on a reacting network is a usable seed, not a poisoned one.

    The public initial guess must match the seed ``solve()`` itself uses: the composition
    rows carry the feed mixture (never all zero, which a reacting closure cannot evaluate),
    so ``solve(x0=net.initial_guess())`` converges to the same state as a bare ``solve()``.
    """
    import numpy as np

    net = _ch4_air_reacting_network()

    x0 = net.initial_guess()
    assert x0.shape[0] > 3  # reacting: mass, pressure, enthalpy, then composition scalars
    assert not np.allclose(x0[3:, :], 0.0)  # composition seeded from the feed, not zeros

    sol_auto = net.solve()  # bare solve (uses auto_initial_guess internally)
    sol_seed = net.solve(x0=net.initial_guess())  # must not raise, must converge
    assert sol_auto.converged and sol_seed.converged
    assert np.allclose(sol_auto.field("T"), sol_seed.field("T"), rtol=1e-8)


def test_reacting_low_level_initial_guess_never_zero_composition():
    """The low-level ``initial_guess(prob)`` is reacting-safe for any direct caller.

    With no composition supplied it returns the feed-mixing seed rather than zeros, so a
    reacting problem handed straight to the closure can be evaluated.
    """
    import numpy as np

    from nefes.solver.control import initial_guess

    prob = _ch4_air_reacting_network().compile()
    x = initial_guess(prob)
    assert not np.allclose(x[3:, :], 0.0)


def test_perfect_gas_initial_guess_unchanged():
    """The non-reacting initial guess keeps its simple ``(3, E)`` form and still solves."""
    import numpy as np

    import nefes

    net = nefes.Network(
        nodes=[cat.mass_flow_inlet(0.5, 300.0), cat.duct(0.3), cat.pressure_outlet(101325.0)],
        edges=[(0, 1, 0.05), (1, 2, 0.05)],
    )
    x0 = net.initial_guess()
    assert x0.shape[0] == 3  # perfect gas transports no composition scalars
    assert net.solve(x0=x0).converged
    assert np.all(x0[0, :] > 0.0)  # a small co-directional mass flow
