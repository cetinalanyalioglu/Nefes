"""Default reacting chemistry: the automatic product species_set and the bundled-data fallbacks.

A reacting network built with ``nefes.equilibrium()`` (no species_set) carries no species list;
the product slate is derived from the network's feed compositions when the network is built,
over the packaged NASA Glenn / CEA data.  ``equivalence_ratio_mixture`` likewise falls back to
that data when given no species_set.  These tests pin the behavior that lets a reacting teaser stay
short: the automatic slate reproduces an explicit one, it is feed-driven and inspectable after
the build, and the same policy drives the loader and the Python build path.
"""

import numpy as np
import pytest

import nefes
from nefes.chem import equivalence_ratio_mixture
from nefes.elements import catalog as cat
from nefes.thermo import SpeciesDatabase
from nefes.thermo.api import EQ_FROZEN, EQ_KERNEL
from nefes.thermo.autoset import auto_product_set
from nefes.thermo.configure import equilibrium

#  Stoichiometric methane/air as a plain named mole mixture (no species_set needed to write it).
CH4_AIR = {"CH4": 1.0, "O2": 2.0, "N2": 7.52}


def _flame_network(gas, composition, *, marker_gated=False):
    """A minimal premixed-inlet / flame / outlet reacting network."""
    edge_models = None if marker_gated else [EQ_FROZEN, EQ_KERNEL]
    return nefes.Network(
        gas,
        nodes=[
            cat.mass_flow_inlet(0.1, 300.0, composition=composition, name="feed"),
            cat.equilibrium_flame(name="flame"),
            cat.pressure_outlet(1.0e5, 300.0, name="out"),
        ],
        edges=[(0, 1, 0.01), (1, 2, 0.01)],
        edge_models=edge_models,
    )


def test_deferred_library_reproduces_an_explicit_slate():
    """``equilibrium()`` (auto slate) and an explicit species_set give the same flame temperature."""
    sol_auto = _flame_network(equilibrium(), CH4_AIR).solve()
    lib = SpeciesDatabase().select(["CH4", "O2", "N2", "CO2", "H2O", "CO", "OH", "H2", "H", "O", "NO"])
    sol_explicit = _flame_network(equilibrium(lib), CH4_AIR).solve()

    assert sol_auto.converged and sol_explicit.converged
    #  The adiabatic flame temperature is set by the physics, not by the slate's size, so the
    #  automatic (broader) slate and the hand-picked one agree closely.
    assert abs(sol_auto.edge(1)["T"] - sol_explicit.edge(1)["T"]) < 5.0
    assert 2000.0 < sol_auto.edge(1)["T"] < 2600.0


def test_auto_slate_is_inspectable_after_build():
    """After the build, ``net.gas`` carries the resolved species_set and its reduction report."""
    sol = _flame_network(equilibrium(), CH4_AIR).solve()
    gas = sol.network.gas

    assert gas.species_set is not None
    assert gas.n_species > 0
    #  The methane/air products are present; the report records the reduction that selected them.
    assert {"CO2", "H2O", "CO"} <= set(gas.species_names)
    report = gas.species_set.reduction_report
    assert report["n_kept"] == gas.n_species
    assert report["n_kept"] <= report["n_candidates"]


def test_network_repr_reports_auto_species_reduction():
    """Network text/HTML summaries annotate an auto-reduced slate with candidate count and threshold."""
    net = _flame_network(equilibrium(reduce_threshold=1e-4), CH4_AIR)
    net.compile()
    report = net.gas.species_set.reduction_report
    assert report["reducer"] != "none"

    text = repr(net)
    html = net._repr_html_()
    expected = (
        f"equilibrium ({net.gas.n_species} species, "
        f"auto-reduced from {report['n_candidates']}, "
        f"threshold={report['threshold']:g})"
    )
    assert expected in text
    assert expected in html

    # An unreduced auto slate (gate above the candidate count) is labelled ``auto``, not reduced.
    whole = _flame_network(equilibrium(reduce_above=10_000), CH4_AIR)
    whole.compile()
    assert whole.gas.species_set.reduction_report["reducer"] == "none"
    assert f"equilibrium ({whole.gas.n_species} species, auto)" in repr(whole)


def test_deferred_config_is_feed_driven_and_left_unmutated():
    """One deferred config, two feeds -> two feed-appropriate slates; the config itself is intact."""
    gas = equilibrium()  # a single deferred config, reused below

    ch4 = _flame_network(gas, CH4_AIR).solve()
    h2 = _flame_network(gas, {"H2": 1.0, "O2": 1.0, "N2": 3.76}).solve()

    assert ch4.converged and h2.converged
    #  Carbon products only appear when carbon is fed.
    assert {"CO2", "CO"} <= set(ch4.network.gas.species_names)
    assert not ({"CO2", "CO"} & set(h2.network.gas.species_names))
    assert "H2O" in h2.network.gas.species_names
    #  The passed-in config is never mutated: it stays deferred and reusable.
    assert gas.species_set is None and gas.auto_species_set is True


def test_rebuild_reresolves_the_slate():
    """Recompiling re-derives the slate from the current feeds (the auto flag persists)."""
    net = _flame_network(equilibrium(), CH4_AIR)
    net.compile()
    first = set(net.gas.species_names)
    #  The surfaced config keeps its automatic flag, so a second compile re-derives, not freezes.
    assert net.gas.auto_species_set is True
    net.compile()
    assert set(net.gas.species_names) == first


def test_python_build_matches_the_shared_policy():
    """The deferred build resolves to exactly what the shared ``auto_product_set`` returns."""
    net = _flame_network(equilibrium(), CH4_AIR)
    net.compile()
    feeds = [cat.mass_flow_inlet(0.1, 300.0, composition=CH4_AIR)]
    direct = auto_product_set(SpeciesDatabase(), feeds, p_ref=net.p_ref, T_init=3000.0)
    assert set(net.gas.species_names) == set(direct.species_names)


def test_marker_gated_novice_shape_converges():
    """The recommended short shape (compositions + equilibrium() + a flame, no edge_models) runs."""
    mix = equivalence_ratio_mixture({"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, phi=1.0)
    sol = _flame_network(equilibrium(), mix, marker_gated=True).solve()
    assert sol.converged
    assert 2000.0 < sol.edge(1)["T"] < 2600.0


def test_equivalence_ratio_mixture_without_library():
    """No-species_set mixing resolves species against the packaged data, matching an explicit species_set."""
    no_lib = equivalence_ratio_mixture({"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0)
    lib = SpeciesDatabase().select(["CH4", "O2", "N2"])
    with_lib = equivalence_ratio_mixture({"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 1.0, species_set=lib)

    assert set(no_lib) == set(with_lib)
    for name in no_lib:
        assert no_lib[name] == pytest.approx(with_lib[name])


def test_equivalence_ratio_mixture_array_without_library_raises():
    """Array inputs cannot be interpreted without a species_set, so ``species_set=None`` is rejected."""
    with pytest.raises(ValueError, match="named"):
        equivalence_ratio_mixture(np.array([1.0, 0.0]), {"O2": 1.0}, 1.0)


def test_deferred_library_rejects_declared_streams():
    """A deferred automatic species_set is auto-mode only; declaring a stream basis needs a species_set."""
    with pytest.raises(ValueError, match="automatic product slate"):
        equilibrium(streams={"air": {"O2": 0.21, "N2": 0.79}})


def test_auto_species_set_needs_a_feed_composition():
    """The automatic slate has nothing to build from if no feed declares a composition."""
    with pytest.raises(ValueError, match="at least one feed"):
        auto_product_set(SpeciesDatabase(), [], p_ref=1.0e5, T_init=3000.0)


def _slate_size(gas, composition=CH4_AIR):
    """Number of species the automatic slate resolves to for a compiled flame network."""
    net = _flame_network(gas, composition)
    net.compile()
    return net.gas.n_species


def test_reduce_threshold_controls_slate_size():
    """A tighter trace cutoff keeps fewer species; a looser one keeps more."""
    #  Methane/air has a large candidate set (> the gate), so the reducer runs and the cutoff bites.
    tight = _slate_size(equilibrium(reduce_threshold=1e-4))
    default = _slate_size(equilibrium())
    loose = _slate_size(equilibrium(reduce_threshold=1e-12))
    assert tight < default < loose


def test_reducer_none_keeps_every_candidate():
    """Selecting the ``none`` reducer keeps the whole candidate slate."""
    net = _flame_network(equilibrium(reducer="none"), CH4_AIR)
    net.compile()
    report = net.gas.species_set.reduction_report
    assert report["reducer"] == "none"
    assert report["n_kept"] == report["n_candidates"]
    #  It keeps strictly more than the default reduced slate.
    assert net.gas.n_species > _slate_size(equilibrium())


def test_reduce_above_gates_reduction():
    """The gate decides whether reduction runs: below it every candidate is kept, above it trims."""
    #  Hydrogen/air has a small candidate set that the default gate leaves untouched.
    h2_air = {"H2": 1.0, "O2": 1.0, "N2": 3.76}
    default = _slate_size(equilibrium(), h2_air)
    assert _flame_network(equilibrium(), h2_air).problem.gas.species_set.reduction_report["reducer"] == "none"
    #  Dropping the gate to zero forces the reducer to run and trim the lean slate.
    forced = _slate_size(equilibrium(reduce_above=0), h2_air)
    assert forced < default
    #  Raising the gate above the candidate count keeps a large slate whole.
    kept_whole = _flame_network(equilibrium(reduce_above=10_000), CH4_AIR)
    kept_whole.compile()
    r = kept_whole.gas.species_set.reduction_report
    assert r["reducer"] == "none" and r["n_kept"] == r["n_candidates"]
