"""Mechanism ingestion and round-trip."""

import numpy as np

from nefes.thermo import Mechanism


def test_cantera_load_structure(cantera_mech):
    m = cantera_mech
    assert m.n_species == 10
    assert set(m.elements) == {"O", "H", "Ar", "N"}
    # element matrix: H2 has 2 H atoms.
    j = m.species_index["H2"]
    i = m.element_index["H"]
    assert m.element_matrix[i, j] == 2
    # molar mass of H2O ~ 0.018 kg/mol.
    assert np.isclose(m.molar_masses[m.species_index["H2O"]], 0.018015, atol=1e-4)


def test_reactions_loaded(cantera_mech):
    assert len(cantera_mech.reactions) > 0
    r = cantera_mech.reactions[0]
    assert r.reactants and r.products


def test_cantera_roundtrip(tmp_path, cantera_mech):
    p = tmp_path / "rt.yaml"
    cantera_mech.write_cantera_yaml(p)
    m2 = Mechanism.from_cantera(p)
    assert [s.name for s in m2.species] == [s.name for s in cantera_mech.species]
    assert np.allclose(m2.molar_masses, cantera_mech.molar_masses)
    assert np.allclose(m2.element_matrix, cantera_mech.element_matrix)
    assert len(m2.reactions) == len(cantera_mech.reactions)
    # Thermo coefficients survive the round-trip.
    T = 1234.0
    assert np.allclose(m2.cp_R(T), cantera_mech.cp_R(T))


def test_cantera_solution_matches_file(cantera_mech, cantera):
    # Extracting from a live cantera.Solution matches parsing the packaged YAML directly.
    m = Mechanism.from_cantera(cantera.Solution("h2o2.yaml"))
    assert [s.name for s in m.species] == [s.name for s in cantera_mech.species]
    assert np.allclose(m.molar_masses, cantera_mech.molar_masses)
    T = 1500.0
    assert np.allclose(m.g_RT(T), cantera_mech.g_RT(T), atol=1e-10)
