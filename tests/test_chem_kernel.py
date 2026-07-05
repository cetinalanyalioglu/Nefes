"""Part-A/Part-B agreement: the compiled Nefes chemistry kernel vs ``thermolib``.

``thermolib`` (Part A) is the standalone, Cantera-validated authority but is
pure-numpy.  :mod:`nefes.thermo._chem` (Part B) re-implements the same
element-potential equilibrium in numba so it runs inside the ``@njit`` residual
path.  The two MUST agree -- this test pins that, on both NASA-7 (Cantera YAML)
and NASA-9 (NASA Glenn / CEA ``thermo.inp``) data, and re-checks the complex-step
== finite-difference contract on the compiled kernel.
"""

import os

import numpy as np
import pytest

from nefes.thermo.configure import equilibrium
from nefes.thermo.equilibrium import eq_kernel_state_from_Z

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data")
H2O2 = os.path.join(DATA, "h2o2.yaml")
THERMO_INP = os.path.join(DATA, "thermo.inp")
_NASA9_SPECIES = ["H2", "O2", "H2O", "OH", "H", "O", "N2", "NO", "HO2", "H2O2"]


def _h2_air(lib, gas, phi):
    idx, W = lib.species_index, lib.molar_masses
    y = np.zeros(lib.n_species)
    for sp, m in {"H2": phi * 1.0, "O2": 0.5, "N2": 0.5 * 3.76}.items():
        y[idx[sp]] = m * W[idx[sp]]
    y /= y.sum()
    return y, gas.elemental_mass_fractions(y)


def _nasa7_lib():
    from thermolib import SpeciesLibrary

    return SpeciesLibrary.from_native(H2O2)


def _nasa9_lib():
    from thermolib import ThermoInp

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    return ThermoInp(THERMO_INP).library(_NASA9_SPECIES)


@pytest.mark.parametrize("which", ["nasa7", "nasa9"])
@pytest.mark.parametrize("phi", [0.6, 1.0, 1.3])
@pytest.mark.parametrize("p", [101325.0, 5.0e5])
def test_kernel_matches_thermolib(which, phi, p):
    from thermolib import Thermo

    lib = _nasa7_lib() if which == "nasa7" else _nasa9_lib()
    gas = Thermo(lib)
    y, Z = _h2_air(lib, gas, phi)
    h = gas.enthalpy_mass(y, 300.0)
    cfg = equilibrium(lib)  # stream-less config: the from_Z entry takes elemental Z directly

    ref = gas.equilibrate_HP(Z, h, p, T_guess=2200.0)
    assert ref.converged
    T, rho, c, W = eq_kernel_state_from_Z(cfg.tf, cfg.ti, np.ascontiguousarray(Z), h, p)

    assert T == pytest.approx(ref.T, rel=1e-6)
    assert c == pytest.approx(ref.a_equilibrium, rel=1e-6)
    # rho carries the elemental-vs-database molar-mass provenance (~1e-5 on NASA-9).
    assert rho == pytest.approx(ref.rho, rel=3e-5)


def test_kernel_complex_step_matches_fd():
    from thermolib import Thermo

    lib = _nasa7_lib()
    gas = Thermo(lib)
    y, Z = _h2_air(lib, gas, 1.0)
    h = gas.enthalpy_mass(y, 300.0)
    p = 101325.0
    cfg = equilibrium(lib)  # stream-less config: the from_Z entry takes elemental Z directly
    Zc = np.ascontiguousarray(Z)
    eps = 1e-20

    # d/dh
    cs = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc.astype(complex), complex(h, eps), complex(p, 0.0))
    dh = 1e-2
    fp = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc, h + dh, p)
    fm = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc, h - dh, p)
    for k in range(3):
        fd = (fp[k] - fm[k]) / (2 * dh)
        assert cs[k].imag / eps == pytest.approx(fd, rel=1e-5, abs=1e-10)

    # d/dp
    cs = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc.astype(complex), complex(h, 0.0), complex(p, eps))
    dp = 1.0
    fp = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc, h, p + dp)
    fm = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc, h, p - dp)
    for k in range(3):
        fd = (fp[k] - fm[k]) / (2 * dp)
        assert cs[k].imag / eps == pytest.approx(fd, rel=1e-5, abs=1e-10)

    # d/dZ[0]
    Zci = Zc.astype(complex)
    Zci[0] = complex(Z[0], eps)
    cs = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zci, complex(h, 0.0), complex(p, 0.0))
    dz = 1e-7
    Zp = Zc.copy()
    Zp[0] += dz
    Zm = Zc.copy()
    Zm[0] -= dz
    fp = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zp, h, p)
    fm = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zm, h, p)
    for k in range(3):
        fd = (fp[k] - fm[k]) / (2 * dz)
        assert cs[k].imag / eps == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_kernel_masks_condensed_feed_from_products():
    """A condensed feed species (liquid Jet-A) sets the elements + enthalpy but is excluded
    from the burnt products in the compiled kernel -- matching thermolib's masked solve, not
    the spurious low temperature you get if the liquid is (wrongly) allowed as a product."""
    from thermolib import ThermoInp, equilibrate_HP
    from nefes.chem.composition import elemental_Z, enthalpy_mass, species_mass_fractions

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    products = ["CO2", "H2O", "CO", "H2", "O2", "N2", "OH", "O", "H", "NO", "N", "NO2", "HO2", "H2O2"]
    lib = ThermoInp(THERMO_INP).library(products + ["Jet-A(L)"])
    assert not lib.product_mask[lib.species_index["Jet-A(L)"]]

    comp = {"Jet-A(L)": 1.0, "O2": 17.75, "N2": 66.7}
    Y = species_mass_fractions(lib, comp, "mole")
    Z = elemental_Z(lib, Y)
    h = enthalpy_mass(lib, Y, 300.0)  # feed enthalpy carries the liquid (latent-heat) datum
    p = 101325.0

    cfg = equilibrium(lib)
    # the bundle's product count excludes the one condensed species
    assert int(cfg.ti[6]) == lib.n_species - 1

    ref = equilibrate_HP(lib, {lib.elements[i]: float(Z[i]) for i in range(len(Z))}, h, p, T_guess=2300.0)
    assert ref.converged
    assert float(np.real(ref.X[lib.species_index["Jet-A(L)"]])) == 0.0  # masked in thermolib too
    T, rho, c, W = eq_kernel_state_from_Z(cfg.tf, cfg.ti, np.ascontiguousarray(Z), h, p)
    assert T == pytest.approx(ref.T, rel=1e-6)
    assert float(np.real(T)) > 2200.0  # not the ~1984 K spurious-product result


def test_kernel_drops_absent_element():
    """A carbonless burnt edge in a carbon-bearing library (the parallel-branch case):
    its elemental abundance ``Z`` has a zero carbon entry, so carbon's balance row is
    null -> singular.  The compiled kernel drops carbon and every carbon-bearing
    species (keep_el / keep_sp), exactly as thermolib's masked solve does -- so the
    burnt state matches and the complex-step Jacobian stays finite."""
    from thermolib import equilibrate_HP, ThermoInp
    from nefes.chem.composition import elemental_Z, enthalpy_mass, species_mass_fractions

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    # carbon-bearing library, but a carbonless (H2/air) feed
    species = ["H2", "O2", "N2", "H2O", "OH", "H", "O", "NO", "CO2", "CO", "CH4"]
    lib = ThermoInp(THERMO_INP).library(species)
    elems = [lib.elements[i] for i in range(len(lib.elements))]
    assert "C" in elems  # carbon is a library element...
    ci = elems.index("C")

    comp = {"H2": 1.0, "O2": 0.5, "N2": 0.5 * 3.76}
    Y = species_mass_fractions(lib, comp, "mole")
    Z = elemental_Z(lib, Y)
    assert float(Z[ci]) == 0.0  # ...but none is fed
    h = enthalpy_mass(lib, Y, 300.0)
    p = 3.0e5

    cfg = equilibrium(lib)
    ref = equilibrate_HP(lib, {elems[i]: float(Z[i]) for i in range(len(Z))}, h, p, T_guess=2200.0)
    assert ref.converged
    for sp in ["CO2", "CO", "CH4"]:  # carbon products absent in thermolib's compacted solve
        assert float(np.real(ref.X[lib.species_index[sp]])) == pytest.approx(0.0, abs=1e-12)

    T, rho, c, W = eq_kernel_state_from_Z(cfg.tf, cfg.ti, np.ascontiguousarray(Z), h, p)
    assert T == pytest.approx(ref.T, rel=1e-6)
    assert c == pytest.approx(ref.a_equilibrium, rel=1e-6)
    assert rho == pytest.approx(ref.rho, rel=3e-5)

    # complex-step through h stays finite and matches FD despite the dropped element
    eps, dh = 1e-20, 1e-2
    Zc = np.ascontiguousarray(Z).astype(complex)
    cs = eq_kernel_state_from_Z(cfg.tf, cfg.ti, Zc, complex(h, eps), complex(p, 0.0))
    fp = eq_kernel_state_from_Z(cfg.tf, cfg.ti, np.ascontiguousarray(Z), h + dh, p)
    fm = eq_kernel_state_from_Z(cfg.tf, cfg.ti, np.ascontiguousarray(Z), h - dh, p)
    for k in range(3):
        assert np.isfinite(cs[k].imag)
        assert cs[k].imag / eps == pytest.approx((fp[k] - fm[k]) / (2 * dh), rel=1e-5, abs=1e-10)


# ---------------------------------------------------------------------------
# Frozen (unburnt) closure: forward-blend reconstruction from feed-stream xi
# ---------------------------------------------------------------------------
def _mixed_feed_lib():
    from thermolib import ThermoInp

    if not os.path.isfile(THERMO_INP):
        pytest.skip("thermo.inp not present")
    heavy = "C8H18,n-octane"
    lib = ThermoInp(THERMO_INP).library(["O2", "N2", heavy, "H2", "CH4", "CO2", "H2O", "CO", "OH", "H", "O", "NO"])
    return lib, heavy


def _blend(lib, streams, xi):
    """Mass-weighted species mass fractions of a feed-stream blend ``Y = sum xi_k Y_k``."""
    from nefes.chem.composition import species_mass_fractions

    Y = np.zeros(lib.n_species)
    for w, spec in zip(xi, streams.values()):
        Y += w * species_mass_fractions(lib, spec, basis="mole")
    return Y


def test_frozen_from_xi_matches_thermolib():
    """The frozen closure recovers a *varying* unburnt mixture (air + 2 fuels) by the
    forward blend of feed-stream mixture fractions -- matching a thermolib frozen
    evaluation of the same mixture, with no element inversion."""
    from nefes.chem.composition import enthalpy_mass
    from nefes.thermo.equilibrium import eq_frozen_state
    from thermolib import Thermo

    lib, heavy = _mixed_feed_lib()
    gas = Thermo(lib)
    streams = {"air": {"O2": 0.21, "N2": 0.79}, "oct": {heavy: 1.0}, "h2": {"H2": 1.0}}
    cfg = equilibrium(lib, streams=streams, basis="mole")
    xi = np.array([0.94, 0.04, 0.02])  # mass fractions of stream origin (sum -> 1)
    Y = _blend(lib, streams, xi)
    Tin, p = 400.0, 2.0e5
    h = enthalpy_mass(lib, Y, Tin)
    T, rho, c, W = eq_frozen_state(cfg.tf, cfg.ti, np.ascontiguousarray(xi), h, p)

    props = gas.properties(Y, T, p)
    assert T == pytest.approx(Tin, abs=1e-2)  # frozen recovers the feed temperature
    assert rho == pytest.approx(props.rho, rel=3e-5)
    assert c == pytest.approx(props.a_frozen, rel=1e-4)


def test_frozen_from_xi_complex_step():
    """Complex-step == FD for the frozen edge through both xi (forward blend) and h
    (temperature inversion)."""
    from nefes.chem.composition import enthalpy_mass
    from nefes.thermo.equilibrium import eq_frozen_state

    lib, heavy = _mixed_feed_lib()
    streams = {"air": {"O2": 0.21, "N2": 0.79}, "oct": {heavy: 1.0}, "h2": {"H2": 1.0}}
    cfg = equilibrium(lib, streams=streams, basis="mole")
    xi = np.array([0.94, 0.04, 0.02])
    h = enthalpy_mass(lib, _blend(lib, streams, xi), 400.0)
    p = 2.0e5
    xc = np.ascontiguousarray(xi)
    eps = 1e-20

    for i in range(len(xi)):
        xci = xc.astype(complex)
        xci[i] = complex(xi[i], eps)
        cs = eq_frozen_state(cfg.tf, cfg.ti, xci, complex(h, 0.0), complex(p, 0.0))
        dz = 1e-7
        xp = xc.copy()
        xp[i] += dz
        xm = xc.copy()
        xm[i] -= dz
        fp = eq_frozen_state(cfg.tf, cfg.ti, xp, h, p)
        fm = eq_frozen_state(cfg.tf, cfg.ti, xm, h, p)
        for k in (0, 1):
            fd = (fp[k] - fm[k]) / (2 * dz)
            if abs(fd) > 1e-6:
                assert cs[k].imag / eps == pytest.approx(fd, rel=1e-4)


def test_comixed_fuels_are_resolvable():
    """Co-mixed multi-fuel (CH4 + C8H18 + H2 in air, indistinguishable at the
    C,H,O,N elemental level) is recovered exactly: each fuel is its own feed
    stream, so the forward blend is unambiguous."""
    from nefes.chem.composition import enthalpy_mass
    from nefes.thermo.equilibrium import eq_frozen_state
    from thermolib import Thermo

    lib, heavy = _mixed_feed_lib()
    gas = Thermo(lib)
    # four streams over only {C, H, O, N} -- rank-deficient for an elemental basis
    streams = {"air": {"O2": 0.21, "N2": 0.79}, "ch4": {"CH4": 1.0}, "oct": {heavy: 1.0}, "h2": {"H2": 1.0}}
    cfg = equilibrium(lib, streams=streams, basis="mole")
    assert cfg.n_elem == 4  # four transported mixture fractions, one per fuel/oxidizer
    xi = np.array([0.90, 0.03, 0.04, 0.03])
    Y = _blend(lib, streams, xi)
    Tin, p = 350.0, 1.5e5
    h = enthalpy_mass(lib, Y, Tin)
    T, rho, c, W = eq_frozen_state(cfg.tf, cfg.ti, np.ascontiguousarray(xi), h, p)
    props = gas.properties(Y, T, p)
    assert T == pytest.approx(Tin, abs=1e-2)
    assert rho == pytest.approx(props.rho, rel=3e-5)
