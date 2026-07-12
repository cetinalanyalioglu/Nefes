"""The flat public namespace: the common workflow is reachable from ``nefes`` directly.

A user (or an agent) should be able to build, choose a gas model, solve, and set an acoustic
termination without importing from sub-packages.  These tests pin the top-level ``nefes.*``
names and the matching re-exports on ``nefes.thermo`` so a dropped export is caught here rather
than in a stale example.
"""

import numpy as np

import nefes
import nefes.thermo as thermo
from nefes.elements import catalog as cat
from nefes.perturbation import PerturbationBC
from nefes.thermo.configure import equilibrium, perfect_gas


def test_all_names_resolve_on_package():
    """Every name advertised in ``nefes.__all__`` must be importable from the package."""
    missing = [name for name in nefes.__all__ if not hasattr(nefes, name)]
    assert not missing, f"names in __all__ not resolvable on nefes: {missing}"


def test_everyday_names_are_hoisted():
    """The everyday entry points are on the top-level namespace and are the canonical objects."""
    assert nefes.cat is cat
    assert nefes.perfect_gas is perfect_gas
    assert nefes.equilibrium is equilibrium
    assert nefes.PerturbationBC is PerturbationBC


def test_thermo_reexports_closure_constants_consistently():
    """``nefes.thermo`` surfaces the whole closure-constant / gas-model family, not a subset."""
    for name in ("PERFECT_GAS", "EQ_KERNEL", "EQ_FROZEN", "EQ_MARKER", "perfect_gas", "equilibrium"):
        assert hasattr(thermo, name), f"nefes.thermo is missing {name}"
    #  The reacting build addresses edges by these ids; they must be distinct.
    assert len({thermo.EQ_KERNEL, thermo.EQ_FROZEN, thermo.EQ_MARKER}) == 3


def test_flat_namespace_build_solve():
    """A non-reacting build, solve, and acoustic call using only ``nefes.*``."""
    net = nefes.Network(
        nefes.perfect_gas(287.0, 1.4),
        nodes=[
            nefes.cat.total_pressure_inlet(1.02e5, 300.0, perturbation_bc=nefes.PerturbationBC.hard_wall()),
            nefes.cat.duct(0.5),
            nefes.cat.pressure_outlet(1.0e5, 300.0, perturbation_bc=nefes.PerturbationBC.open_end()),
        ],
        edges=[(0, 1, 0.01), (1, 2, 0.01)],
    )
    sol = net.solve()
    assert sol.converged
    from nefes.perturbation import eigenmodes

    modes = eigenmodes(sol, freq_band=(100.0, 1500.0))
    assert modes.n_modes >= 1


def test_both_continuation_routes_live_under_perturbation():
    """The impulse-response and rational analytic-continuation routes are reachable from one place."""
    import nefes.elements.dynamic_source as dsource
    import nefes.perturbation as pert

    for name in (
        "fit_impulse_response",
        "finite_impulse_response",
        "FiniteImpulseResponse",
        "rational_fit",
        "RationalFit",
        "continuation_warning",
    ):
        assert hasattr(pert, name), f"nefes.perturbation is missing {name}"
    #  Re-exported, not re-implemented: the impulse-response route is the same object as its source.
    assert pert.fit_impulse_response is dsource.fit_impulse_response
    assert pert.finite_impulse_response is dsource.finite_impulse_response


def test_chem_reexports_composition_helpers():
    """The user-facing composition helpers are reachable one submodule away, from ``nefes.chem``."""
    import nefes.chem as chem

    for name in (
        "equivalence_ratio_mixture",
        "resolve_composition",
        "enthalpy_mass",
        "species_mass_fractions",
        "species_mole_fractions",
        "elemental_Z",
    ):
        assert hasattr(chem, name), f"nefes.chem is missing {name}"


def test_flat_namespace_reacting_build():
    """A reacting build off ``nefes.equilibrium`` + ``nefes.thermo`` closure ids."""
    from nefes.chem import equivalence_ratio_mixture
    from nefes.thermo import ThermoInp

    lib = ThermoInp().library(["CH4", "O2", "N2", "CO2", "H2O", "CO", "OH", "H2", "H", "O", "NO"])
    mix = equivalence_ratio_mixture(lib, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    net = nefes.Network(
        nefes.equilibrium(lib),
        nodes=[
            nefes.cat.mass_flow_inlet(1.0, 300.0, composition=mix, name="feed"),
            nefes.cat.equilibrium_flame(name="flame"),
            nefes.cat.pressure_outlet(101325.0, 300.0, composition=mix, name="out"),
        ],
        edges=[(0, 1, 0.05), (1, 2, 0.05)],
        edge_models=[nefes.thermo.EQ_FROZEN, nefes.thermo.EQ_KERNEL],
    )
    sol = net.solve()
    assert sol.converged
    assert sol.edge(1)["T"] > 2000.0  # burnt
    assert np.isclose(sol.edge(0)["T"], 300.0, atol=5.0)  # unburnt approach
