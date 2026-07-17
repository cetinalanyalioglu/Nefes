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
from nefes.thermo.reduction import EquilibriumSamplingReducer, SampleState, SpeciesReductionWarning

#  Stoichiometric methane/air as a plain named mole mixture (no species_set needed to write it).
CH4_AIR = {"CH4": 1.0, "O2": 2.0, "N2": 7.52}
CH4_AIR_FEED = [cat.mass_flow_inlet(0.1, 300.0, composition=CH4_AIR)]


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


def _auto(db=None, **kw):
    """``auto_product_set`` for the methane/air feed, packaged data unless overridden."""
    return auto_product_set(db or SpeciesDatabase(), CH4_AIR_FEED, p_ref=1.0e5, T_init=3000.0, **kw)


def test_max_species_caps_the_slate_size():
    """``max_species`` bounds the kept count and keeps the highest-peaking species."""
    #  A loose threshold lets the cap alone drive the count (the size-sweep pattern).
    with pytest.warns(SpeciesReductionWarning):  # trimming below the non-trace slate is the point
        lib = _auto(max_species=8, threshold=1e-14)
    report = lib.reduction_report
    assert report["n_kept"] == 8
    assert report["max_species"] == 8
    #  The dominant methane/air products and the feed species survive the cut.
    assert {"CH4", "O2", "N2", "CO2", "H2O"} <= set(lib.species_names)


def test_max_species_is_a_ceiling_not_a_target():
    """A cap above what the threshold keeps does not pad the slate up with trace species."""
    default = _auto().reduction_report["n_kept"]
    capped_high = _auto(max_species=10_000)
    #  With the default threshold binding first, a huge cap keeps exactly the reduced slate.
    assert capped_high.reduction_report["n_kept"] == default


def test_max_species_runs_below_the_gate():
    """Setting a cap forces reduction even when the candidate pool is under ``reduce_above``."""
    h2_air = [cat.mass_flow_inlet(0.1, 300.0, composition={"H2": 1.0, "O2": 1.0, "N2": 3.76})]
    #  Hydrogen/air is small enough that the default gate keeps it whole; the cap overrides that.
    whole = auto_product_set(SpeciesDatabase(), h2_air, p_ref=1.0e5, T_init=3000.0)
    with pytest.warns(SpeciesReductionWarning):  # the cap trims the small pool below its non-trace slate
        capped = auto_product_set(SpeciesDatabase(), h2_air, p_ref=1.0e5, T_init=3000.0, max_species=4, threshold=1e-14)
    assert whole.reduction_report["reducer"] == "none"
    assert capped.reduction_report["n_kept"] == 4


def test_max_species_warns_when_it_discards_non_trace_species():
    """Capping below the above-threshold count warns that the slate may miss real species."""
    with pytest.warns(SpeciesReductionWarning, match="above the trace threshold"):
        lib = _auto(max_species=6, threshold=1e-14)
    assert lib.reduction_report["dropped_above_threshold"] > 0


def test_max_species_keeps_a_carrier_of_every_fed_element():
    """Even at an aggressive cap the kept set carries every fed-in element (else equilibrium is singular)."""
    db = SpeciesDatabase()
    with pytest.warns(SpeciesReductionWarning):  # a cap this tight necessarily drops non-trace species
        lib = _auto(db=db, max_species=5, threshold=1e-14)
    pool = {el for name in CH4_AIR for el in db[name].composition if el != "E"}
    carried = {el for name in lib.species_names for el in db[name].composition if el != "E"}
    assert pool <= carried


def test_reducer_promotes_a_carrier_for_an_uncovered_element():
    """Element coverage adds a carrier for an element the mandatory keeps miss, even below threshold."""
    db = SpeciesDatabase()
    candidates = db.select(["N2", "O2", "NO", "CO2", "CO", "H2O", "OH", "H2"])
    #  A hot lean methane/air-like elemental state (mass fractions).
    sample = SampleState({"C": 0.05, "H": 0.01, "O": 0.60, "N": 0.34}, 2400.0, 1.0e5)
    #  Threshold so high that ordinary selection keeps nothing; only mandatory + coverage remain.
    reducer = EquilibriumSamplingReducer(threshold=1.0)
    result = reducer.reduce(candidates, [sample], always_keep=["H2O"])  # H2O covers H and O only
    carried = {el for name in result.species for el in candidates.species[candidates.species_index[name]].composition}
    #  Carbon and nitrogen were uncovered by H2O, so a carrier of each was promoted.
    assert "C" in carried and "N" in carried
    assert result.report["coverage_added"]


def test_must_species_keeps_a_trace_species_and_warns():
    """``must_species`` keeps a species that is trace at equilibrium, and says so."""
    #  NO2 peaks near 3e-6 in methane/air; a 1e-3 threshold (keep_floor 1e-5) would drop it.
    with pytest.warns(SpeciesReductionWarning, match="below the trace threshold"):
        lib = _auto(must_species=["NO2"], threshold=1e-3)
    assert "NO2" in lib.species_names


def test_must_species_keeps_a_condensed_product():
    """A high-temperature condensed product (graphite) can be forced into the slate and solves."""
    #  Rich methane/air, where graphite is a genuine equilibrium product.
    rich = {"CH4": 1.0, "O2": 0.8, "N2": 3.0}
    #  At this lean-of-soot state graphite is still trace, so the force-keep is what retains it,
    #  and both the direct call and the network build re-derive it (hence a warning on each).
    with pytest.warns(SpeciesReductionWarning, match="below the trace threshold"):
        lib = auto_product_set(
            SpeciesDatabase(),
            [cat.mass_flow_inlet(0.1, 300.0, composition=rich)],
            p_ref=1.0e5,
            T_init=2200.0,
            must_species=["C(gr)"],
        )
        assert "C(gr)" in lib.species_names
        sol = _flame_network(equilibrium(must_species=["C(gr)"]), rich).solve()
    assert sol.converged and "C(gr)" in sol.network.gas.species_names


def test_must_species_with_an_unfed_element_raises():
    """A forced species naming an element no feed supplies fails predictably."""
    with pytest.raises(ValueError, match="no feed supplies"):
        _auto(must_species=["SO2"])  # sulfur is absent from methane/air


def test_must_species_rejects_a_feed_only_condensed_species():
    """A condensed species whose data does not reach combustion temperatures cannot be a product."""
    with pytest.raises(ValueError, match="feed-only condensed"):
        _auto(must_species=["Jet-A(L)"])  # a liquid fuel: feed-only, not an equilibrium product


def test_must_species_rejects_an_ion():
    """An ionic species is not carried by the subsonic combustion slate."""
    with pytest.raises(ValueError, match="ion"):
        _auto(must_species=["NO+"])


def test_must_species_absent_from_database_raises():
    """A forced species the database does not know fails as a lookup error."""
    with pytest.raises(KeyError, match="must_species not in"):
        _auto(must_species=["NOTASPECIES"])


def test_max_species_with_reducer_none_raises():
    """A cap has nothing to act on when every candidate is kept, so the pairing is rejected."""
    with pytest.raises(ValueError, match="reducer='none'"):
        equilibrium(reducer="none", max_species=5)
    with pytest.raises(ValueError, match="reducer='none'"):
        _auto(reducer_name="none", max_species=5)


def test_max_species_solves_and_is_reported_in_the_repr():
    """A capped deferred config solves and the network summary annotates the cap."""
    net = _flame_network(equilibrium(max_species=12, reduce_threshold=1e-14), CH4_AIR)
    with pytest.warns(SpeciesReductionWarning):  # the cap trims below the full non-trace slate
        sol = net.solve()
    assert sol.converged
    assert net.gas.n_species == 12
    assert f"max={12}" in repr(net)
