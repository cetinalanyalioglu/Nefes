"""Part-A/Part-B agreement: the compiled FNS chemistry kernel vs ``thermolib``.

``thermolib`` (Part A) is the standalone, Cantera-validated authority but is
pure-numpy.  :mod:`fns.thermo._chem` (Part B) re-implements the same
element-potential equilibrium in numba so it runs inside the ``@njit`` residual
path.  The two MUST agree -- this test pins that, on both NASA-7 (Cantera YAML)
and NASA-9 (NASA Glenn / CEA ``thermo.inp``) data, and re-checks the complex-step
== finite-difference contract on the compiled kernel.
"""

import os

import numpy as np
import pytest

from fns.thermo.configure import equilibrium
from fns.thermo.equilibrium import eq_kernel_state_from_Z

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
    from fns.composition import species_mass_fractions

    Y = np.zeros(lib.n_species)
    for w, spec in zip(xi, streams.values()):
        Y += w * species_mass_fractions(lib, spec, basis="mole")
    return Y


def test_frozen_from_xi_matches_thermolib():
    """The frozen closure recovers a *varying* unburnt mixture (air + 2 fuels) by the
    forward blend of feed-stream mixture fractions -- matching a thermolib frozen
    evaluation of the same mixture, with no element inversion."""
    from fns.composition import enthalpy_mass
    from fns.thermo.equilibrium import eq_frozen_state
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
    from fns.composition import enthalpy_mass
    from fns.thermo.equilibrium import eq_frozen_state

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
    """Co-mixed multi-fuel that the old elemental basis could NOT resolve (CH4 +
    C8H18 + H2 in air, indistinguishable from C,H,O,N) is now recovered exactly:
    each fuel is its own feed stream, so the forward blend is unambiguous."""
    from fns.composition import enthalpy_mass
    from fns.thermo.equilibrium import eq_frozen_state
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
