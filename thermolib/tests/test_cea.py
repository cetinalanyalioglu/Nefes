"""NASA Glenn / CEA ``thermo.inp`` reader (TODO #1).

The bulk is Cantera-free (parse, search, build a library, run equilibrium on
NASA-9 data).  A final cross-check against Cantera is skipped when Cantera or
``thermo.inp`` is absent (R-A8.4).
"""

import numpy as np
import pytest

from thermolib import SpeciesLibrary, Thermo, ThermoInp, default_thermo_inp, read_thermo_inp


def test_packaged_thermo_inp_is_the_default():
    """With no path, the loaders resolve to the packaged database (no naming needed)."""
    import os

    path = default_thermo_inp()
    assert os.path.isfile(path) and path.endswith("thermo.inp")
    # Every no-path entry point reads the same packaged database.
    assert len(read_thermo_inp()) == len(ThermoInp()) > 1000
    lib = SpeciesLibrary.from_cea(species=["H2", "O2", "H2O", "N2"])
    assert [s.name for s in lib.species] == ["H2", "O2", "H2O", "N2"]


def test_parse_and_search(thermo_inp):
    db = thermo_inp
    assert len(db) > 1000
    assert "H2O" in db
    assert "H2O" in db.search("h2o")  # case-insensitive
    assert db.search("ZZZnotaspecies") == []


def test_species_record_fields(thermo_inp):
    h2o = thermo_inp["H2O"]
    assert h2o.composition == {"H": 2, "O": 1}
    assert np.isclose(h2o.molar_mass, 0.018015, atol=1e-4)
    assert h2o.thermo.kind == "NASA9"
    # Multi-interval NASA-9 record.
    assert h2o.thermo.coeffs.shape[1] == 9


def test_library_build_is_nasa9_and_one_bar(thermo_inp):
    lib = thermo_inp.library(["H2", "O2", "H2O", "N2"])
    assert isinstance(lib, SpeciesLibrary)
    assert [s.name for s in lib.species] == ["H2", "O2", "H2O", "N2"]
    assert set(lib.elements) == {"H", "O", "N"}
    # CEA standard state is one bar.
    assert lib.P_ref == pytest.approx(1.0e5)


def test_library_missing_species_errors(thermo_inp):
    with pytest.raises(KeyError):
        thermo_inp.library(["H2", "NOTREAL"])


def test_from_cea_classmethod(thermo_inp, tmp_path):
    # SpeciesLibrary.from_cea is the one-call path.
    path = thermo_inp.path
    lib = SpeciesLibrary.from_cea(path, species=["H2", "O2", "H2O", "OH", "H", "O", "N2"])
    assert lib.n_species == 7


def test_cea_thermo_complex_step(thermo_inp):
    lib = thermo_inp.library(["H2", "O2", "H2O", "OH", "H", "O", "N2"])
    T = 2000.0
    eps = 1e-200
    dhRT = lib.h_RT(T + 1j * eps).imag / eps
    analytic = (lib.cp_R(T) - lib.h_RT(T)) / T
    assert np.allclose(dhRT, analytic, rtol=1e-10)


def test_cea_equilibrium_runs(thermo_inp):
    lib = thermo_inp.library(["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2", "N2"])
    gas = Thermo(lib)
    idx = lib.species_index
    X = np.zeros(lib.n_species)
    X[idx["H2"]] = 2.0
    X[idx["O2"]] = 1.0
    X[idx["N2"]] = 3.76
    Y = X * lib.molar_masses / np.sum(X * lib.molar_masses)
    Z = gas.elemental_mass_fractions(Y)
    h0 = gas.properties(Y, 300.0, 101325.0).h
    res = gas.equilibrate_HP(Z, h0, 101325.0, T_guess=2000.0)
    assert res.converged
    assert 2200.0 < res.T < 2500.0  # H2/air flame ballpark
    assert res.X[idx["H2O"]] > 0.3


def test_cea_flame_matches_cantera(thermo_inp, cantera):
    # End-to-end: NASA-9 CEA data + 1-bar reference reproduces a Cantera HP
    # flame within the thermodynamic-data provenance difference (a few K).
    ct = cantera
    lib = thermo_inp.library(["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2", "N2"])
    gas = Thermo(lib)
    idx = lib.species_index
    X = np.zeros(lib.n_species)
    X[idx["H2"]] = 2.0
    X[idx["O2"]] = 1.0
    X[idx["N2"]] = 3.76
    Y = X * lib.molar_masses / np.sum(X * lib.molar_masses)
    Z = gas.elemental_mass_fractions(Y)
    h0 = gas.properties(Y, 300.0, 101325.0).h
    res = gas.equilibrate_HP(Z, h0, 101325.0, T_guess=2000.0)

    ctg = ct.Solution("h2o2.yaml")
    ctg.TPX = 300.0, 101325.0, {"H2": 2, "O2": 1, "N2": 3.76}
    ctg.equilibrate("HP")
    assert abs(res.T - ctg.T) < 10.0
    assert np.isclose(res.X[idx["H2O"]], ctg.X[ctg.species_index("H2O")], atol=2e-3)
