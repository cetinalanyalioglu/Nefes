"""Mechanism ingestion and round-trip."""

import numpy as np

from thermolib import Mechanism


def test_native_load_structure(native_mech):
    m = native_mech
    assert m.n_species == 10
    assert set(m.elements) == {"O", "H", "Ar", "N"}
    # element matrix: H2 has 2 H atoms.
    j = m.species_index["H2"]
    i = m.element_index["H"]
    assert m.element_matrix[i, j] == 2
    # molar mass of H2O ~ 0.018 kg/mol.
    assert np.isclose(m.molar_masses[m.species_index["H2O"]], 0.018015, atol=1e-4)


def test_reactions_loaded(native_mech):
    assert len(native_mech.reactions) > 0
    r = native_mech.reactions[0]
    assert r.reactants and r.products


def test_native_roundtrip(tmp_path, native_mech):
    p = tmp_path / "rt.yaml"
    native_mech.write_native(p)
    m2 = Mechanism.from_native(p)
    assert [s.name for s in m2.species] == [s.name for s in native_mech.species]
    assert np.allclose(m2.molar_masses, native_mech.molar_masses)
    assert np.allclose(m2.element_matrix, native_mech.element_matrix)
    assert len(m2.reactions) == len(native_mech.reactions)
    # Thermo coefficients survive the round-trip.
    T = 1234.0
    assert np.allclose(m2.cp_R(T), native_mech.cp_R(T))


def test_from_cantera_matches_native(native_mech, cantera):
    m = Mechanism.from_cantera("h2o2.yaml")
    assert [s.name for s in m.species] == [s.name for s in native_mech.species]
    assert np.allclose(m.molar_masses, native_mech.molar_masses)
    T = 1500.0
    assert np.allclose(m.g_RT(T), native_mech.g_RT(T), atol=1e-10)
