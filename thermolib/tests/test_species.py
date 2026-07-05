"""SpeciesLibrary: vectorized thermo, NASA7/9 unification, library<->mechanism.

These tests are Cantera-free. They cover the species-library naming (separation from
'mechanism') and the array-vectorized thermo core that keeps the complex-step contract.
"""

import numpy as np
import pytest

from thermolib import (
    NASA7,
    NASA9,
    SpeciesLibrary,
    Thermo,
)


def test_library_has_no_reactions_concept(native_lib):
    # A species library is just thermo data: it has species + elements, and is all
    # equilibrium needs; there is no 'reactions' on it.
    assert native_lib.n_species == 10
    assert set(native_lib.elements) == {"O", "H", "Ar", "N"}
    assert not hasattr(native_lib, "reactions")


def test_mechanism_is_library_plus_reactions(native_mech, native_lib):
    # A mechanism *associates* a library with reactions.
    assert isinstance(native_mech.library, SpeciesLibrary)
    assert len(native_mech.reactions) > 0
    # It proxies the library's thermo/sizing surface.
    assert native_mech.n_species == native_lib.n_species
    T = 1234.0
    assert np.allclose(native_mech.cp_R(T), native_lib.cp_R(T))
    assert np.allclose(native_mech.element_matrix, native_lib.element_matrix)


def test_vectorized_matches_per_species(native_lib):
    # The library evaluates all species in one vector op; it must agree with the
    # per-species polynomials exactly.
    for T in (300.0, 1000.0, 1234.0, 3000.0):
        for fn in ("cp_R", "h_RT", "s_R", "g_RT"):
            lib_val = getattr(native_lib, fn)(T)
            per = np.array([getattr(s.thermo, fn)(T) for s in native_lib.species])
            assert np.allclose(lib_val, per), (fn, T)


def test_vectorized_complex_step_in_T(native_lib):
    # dh/dT (dimensionless) = (cp - h)/T per species, by complex step through
    # the vectorized evaluator.
    T = 1500.0
    eps = 1e-200
    dhRT = native_lib.h_RT(T + 1j * eps).imag / eps
    analytic = (native_lib.cp_R(T) - native_lib.h_RT(T)) / T
    assert np.allclose(dhRT, analytic, rtol=1e-10)


def test_nasa7_embeds_in_nasa9():
    # A NASA7 polynomial is the a1=a2=0 special case of the canonical 9-term
    # form; round-trip a known species' value.
    # cp/R of a constant-ish poly: build a trivial NASA7 and a hand NASA9 equal.
    c_low = [3.5, 1e-3, 0.0, 0.0, 0.0, -1000.0, 5.0]
    p7 = NASA7(200.0, 1000.0, 3500.0, c_low, c_low)
    # equivalent NASA9 row: [0,0, c0..c4, c5, c6]
    row = [0.0, 0.0] + c_low
    p9 = NASA9([200.0, 1000.0, 3500.0], [row, row])
    for T in (300.0, 1500.0):
        assert np.isclose(p7.cp_R(T), p9.cp_R(T))
        assert np.isclose(p7.h_RT(T), p9.h_RT(T))
        assert np.isclose(p7.s_R(T), p9.s_R(T))


def test_interval_selection_on_real_part_only():
    # Branch on Re(T): a complex perturbation must not flip the interval, so the
    # derivative is the analytic one within the chosen range.
    lo = [3.0, 1e-3, 0, 0, 0, -100.0, 1.0]
    hi = [4.0, 2e-3, 0, 0, 0, -200.0, 2.0]
    p = NASA7(200.0, 1000.0, 3500.0, lo, hi)
    # Just below the midpoint: derivative uses the LOW coefficients.
    T = 999.9999
    eps = 1e-200
    dcp = p.cp_R(T + 1j * eps).imag / eps
    assert np.isclose(dcp, lo[1], rtol=1e-9)  # d(cp/R)/dT = a4 here


def test_default_p_ref_is_one_atm(native_lib):
    assert native_lib.P_ref == pytest.approx(101325.0)


def test_subset_preserves_thermo(native_lib):
    sub = native_lib.subset(["H2", "O2", "H2O", "N2"])
    assert [s.name for s in sub.species] == ["H2", "O2", "H2O", "N2"]
    T = 1100.0
    j_full = native_lib.species_index["H2O"]
    j_sub = sub.species_index["H2O"]
    assert np.isclose(sub.cp_R(T)[j_sub], native_lib.cp_R(T)[j_full])


def test_thermo_accepts_library_or_mechanism(native_lib, native_mech):
    # Equilibrium/properties need only a library; kinetics needs reactions.
    g_lib = Thermo(native_lib)
    g_mech = Thermo(native_mech)
    Y = np.zeros(native_lib.n_species)
    Y[native_lib.species_index["H2O"]] = 1.0
    p_lib = g_lib.properties(Y, 1500.0, 101325.0)
    p_mech = g_mech.properties(Y, 1500.0, 101325.0)
    assert np.isclose(p_lib.h, p_mech.h)
    # K_c needs reactions: works from a mechanism, errors from a bare library.
    assert len(g_mech.equilibrium_constants_Kc(1500.0)) == len(native_mech.reactions)
    with pytest.raises(ValueError):
        g_lib.equilibrium_constants_Kc(1500.0)
