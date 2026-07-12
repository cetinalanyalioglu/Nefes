"""Phase tagging, gas-only product masking, and the pluggable species reducer.

These cover the automatic-slate machinery: the CEA phase flag on species, the candidate
product selection from an element pool, the equilibrium kernel masking condensed species
out of the products (so a liquid fuel is a feed, never a product), and the
:class:`~nefes.thermo.reduction.SpeciesReducer` strategy + registry.
"""

import numpy as np
import pytest

from nefes.thermo import (
    EquilibriumSamplingReducer,
    NullReducer,
    ReductionResult,
    SampleState,
    SpeciesReducer,
    ThermoInp,
    available_reducers,
    equilibrate_HP,
    equilibrate_TP,
    get_reducer,
    register_reducer,
)

# Stoichiometric CH4/air elemental mass fractions (a convenient reusable burnt state).
_Z_CH4_AIR = {"C": 0.0413, "H": 0.0139, "O": 0.2201, "N": 0.7247}


@pytest.fixture(scope="module")
def db():
    return ThermoInp()


# --------------------------------------------------------------------------- #
# phase flag
# --------------------------------------------------------------------------- #
def test_phase_flag_parsed(db):
    assert db["CO2"].phase == 0  # gas
    assert db["H2O"].phase == 0
    assert db["Jet-A(L)"].phase != 0  # liquid
    assert db["Jet-A(g)"].phase == 0  # gaseous form


def test_product_mask_excludes_condensed(db):
    lib = db.library(["Jet-A(L)", "CO2", "H2O", "N2"])
    mask = lib.product_mask
    assert mask.dtype == bool
    j = lib.species_index["Jet-A(L)"]
    assert not mask[j]  # condensed -> not a product
    assert mask[lib.species_index["CO2"]]
    assert int(mask.sum()) == 3  # the three gases


# --------------------------------------------------------------------------- #
# candidate slate selection
# --------------------------------------------------------------------------- #
def test_candidate_species_subset_of_pool(db):
    cands = db.candidate_species(["H", "O"])
    for name in cands:
        els = set(db[name].composition) - {"E"}
        assert els.issubset({"H", "O"})
    # the obvious H/O gases are present
    for expect in ("H2", "O2", "H2O", "OH", "H", "O"):
        assert expect in cands


def test_candidate_species_gas_only_and_no_ions(db):
    cands = db.candidate_species(["C", "H", "O", "N"])
    for name in cands:
        assert db[name].phase == 0  # gas only
        assert "+" not in name and "-" not in name  # no ions
        assert "E" not in db[name].composition
    assert "Jet-A(L)" not in cands  # condensed excluded
    # carbon explosion: the CHON gas pool is large (justifies reduction)
    assert len(cands) > 100


def test_candidate_species_gas_only_false_admits_condensed(db):
    with_cond = db.candidate_species(["C", "H"], gas_only=False)
    gas_only = db.candidate_species(["C", "H"], gas_only=True)
    assert len(with_cond) > len(gas_only)


# --------------------------------------------------------------------------- #
# equilibrium masking: a condensed feed never appears as a product
# --------------------------------------------------------------------------- #
def test_condensed_feed_zero_in_products(db):
    names = ["Jet-A(L)", "CO2", "H2O", "CO", "H2", "O2", "N2", "OH", "O", "H", "NO"]
    lib = db.library(names)
    Z = {"C": 0.069, "H": 0.011, "O": 0.214, "N": 0.706}
    tp = equilibrate_TP(lib, Z, 2300.0, 101325.0)
    h = float(np.real(tp.properties.h))
    res = equilibrate_HP(lib, Z, h, 101325.0, T_guess=2300.0)
    assert res.converged
    jet = float(np.real(res.X[lib.species_index["Jet-A(L)"]]))
    assert jet == 0.0  # masked out of the active product set
    # the burnt gas is a sane hydrocarbon/air product distribution
    assert 0.5 < float(np.real(res.X[lib.species_index["N2"]])) < 0.8


# --------------------------------------------------------------------------- #
# reducers
# --------------------------------------------------------------------------- #
def test_null_reducer_is_identity(db):
    lib = db.library(["CO2", "H2O", "N2"])
    res = NullReducer().reduce(lib, [])
    assert set(res.species) == {"CO2", "H2O", "N2"}
    assert res.report["n_kept"] == 3


def test_equilibrium_sampling_reducer_trims_chon(db):
    cands = db.candidate_species(["C", "H", "O", "N"])
    cand_lib = db.library(cands)
    samples = [SampleState(_Z_CH4_AIR, T, 101325.0) for T in (2200.0, 2800.0)]
    res = EquilibriumSamplingReducer().reduce(cand_lib, samples, always_keep=["N2", "O2"])
    assert isinstance(res, ReductionResult)
    assert res.report["samples_used"] == 2
    assert len(res.species) < len(cands)  # actually reduced
    # the major combustion products survive
    for expect in ("CO2", "H2O", "CO", "OH", "N2", "O2"):
        assert expect in res.species
    # always_keep is honored even if trace
    assert "N2" in res.species and "O2" in res.species


def test_reducer_always_keep_only_real_species(db):
    lib = db.library(["CO2", "H2O", "N2"])
    res = EquilibriumSamplingReducer().reduce(
        lib, [SampleState({"C": 0.3, "O": 0.7}, 1500.0, 101325.0)], always_keep=["NotASpecies"]
    )
    assert "NotASpecies" not in res.species  # silently ignored, not crashed


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_defaults_and_lookup():
    assert "equilibrium_sampling" in available_reducers()
    assert "none" in available_reducers()
    assert isinstance(get_reducer(), EquilibriumSamplingReducer)
    assert isinstance(get_reducer("none"), NullReducer)
    with pytest.raises(ValueError, match="unknown species reducer"):
        get_reducer("does-not-exist")


def test_register_custom_reducer(db):
    class KeepFirst(SpeciesReducer):
        name = "keep_first_test"

        def reduce(self, library, samples, *, always_keep=()):
            return ReductionResult(species=library.species_names[:1], report={"reducer": self.name})

    register_reducer("keep_first_test", KeepFirst)
    assert "keep_first_test" in available_reducers()
    r = get_reducer("keep_first_test")
    res = r.reduce(db.library(["CO2", "H2O", "N2"]), [])
    assert res.species == ["CO2"]

    with pytest.raises(TypeError):
        register_reducer("bad", dict)
