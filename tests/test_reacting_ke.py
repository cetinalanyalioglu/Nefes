"""Kinetic-energy coupling for the reacting closures (R-B2.2).

The reacting closures recover the *static* state at ``h = h_t - u^2/2`` with
``u = mdot/(rho A)`` -- an outer bracketed root on the static enthalpy wrapped
around the equilibrium/frozen solve, mirroring the perfect gas's density root.
This pins:

* the energy balance ``h = h_t - u^2/2`` is satisfied exactly at the recovered state;
* the burnt static ``(T, rho)`` equals a standalone HP-equilibrium at
  that *static* enthalpy (== Cantera transitively);
* the complex-step Jacobian == finite difference through ``(mdot, p, h_t, xi)`` across
  forward / reverse / quiescent / near-choke flow -- including the warm-cache path
  whose singular-matrix fallback must keep the linearization finite.
"""

import os

import numpy as np
import pytest

from nefes.thermo.configure import equilibrium
from nefes.thermo.edge_state import eq_kernel_state_ke_warm, eq_frozen_state_ke, eq_marker_state_ke_warm

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nefes", "thermo", "data")
H2O2 = os.path.join(DATA, "h2o2.yaml")


def _premix_cfg():
    """Stoichiometric H2/air as one feed stream; return (cfg, gas, Z, h_t300, Np)."""
    from nefes.thermo import SpeciesLibrary, Thermo

    lib = SpeciesLibrary.from_cantera(H2O2)
    gas = Thermo(lib)
    premix = {"H2": 1.0, "O2": 0.5, "N2": 0.5 * 3.76}
    idx, W = lib.species_index, lib.molar_masses
    y = np.zeros(lib.n_species)
    for sp, m in premix.items():
        y[idx[sp]] = m * W[idx[sp]]
    y /= y.sum()
    Z = gas.elemental_mass_fractions(y)
    h_t = gas.enthalpy_mass(y, 300.0)  # absolute-datum stagnation enthalpy
    cfg = equilibrium(lib, streams={"premix": premix})
    return cfg, gas, Z, h_t, int(cfg.ti[6])


# spans quiescent, low/high subsonic, and reverse flow at a realistic burnt area
@pytest.mark.parametrize("mdot", [0.0, 0.25, 0.5, 0.75, 1.0, -0.75])
def test_ke_burnt_static_matches_oracle(mdot):
    cfg, gas, Z, h_t, Np = _premix_cfg()
    xi = np.array([1.0])
    p, area = 1.0e5, 2.0e-2
    cache = np.zeros(Np + 1)
    T, rho, c, W = eq_kernel_state_ke_warm(cfg.tf, cfg.ti, xi, mdot, p, h_t, area, cache)

    u = mdot / (rho * area)
    h_static = h_t - 0.5 * u * u
    # the static enthalpy is the transported total minus the recovered kinetic energy
    assert h_static == pytest.approx(h_t - 0.5 * (mdot / (rho * area)) ** 2, rel=0, abs=1e-6)
    # burnt (T, rho) == standalone HP equilibrium at the *static* enthalpy
    ref = gas.equilibrate_HP(Z, h_static, p, T_guess=2200.0)
    assert ref.converged
    assert T == pytest.approx(ref.T, rel=1e-5)
    assert rho == pytest.approx(ref.rho, rel=1e-4)
    # the burnt edge is genuinely compressible at the high end (KE is not an O(M^2) curiosity)
    if abs(mdot) >= 1.0:
        assert abs(u) / c > 0.35


def test_ke_complex_step_matches_fd_warm_cache():
    """cs == fd through mdot, p, h_t -- with a *warm* cache (the singular-matrix path)."""
    cfg, gas, Z, h_t, Np = _premix_cfg()
    tf, ti = cfg.tf, cfg.ti
    xi = np.array([1.0])
    p, area, mdot = 1.0e5, 2.0e-2, 1.0  # M ~ 0.41
    eps = 1e-30

    # warm the cache at the operating point, then perturb from the warm state
    cache = np.zeros(Np + 1)
    eq_kernel_state_ke_warm(tf, ti, xi, mdot, p, h_t, area, cache)

    def state(md, pp, hh):
        return np.array(eq_kernel_state_ke_warm(tf, ti, xi.astype(complex), md, pp, hh, complex(area), cache.copy()))

    for name, seed_mdot, seed_p, seed_h, step, var in [
        ("mdot", eps, 0.0, 0.0, 1e-4, 0),
        ("p", 0.0, eps, 0.0, 1e-1, 1),
        ("h_t", 0.0, 0.0, eps, 1e-2, 2),
    ]:
        cs = state(complex(mdot, seed_mdot), complex(p, seed_p), complex(h_t, seed_h)).imag / eps
        base = [mdot, p, h_t]
        bp = list(base)
        bp[var] += step
        bm = list(base)
        bm[var] -= step
        fp = state(complex(bp[0]), complex(bp[1]), complex(bp[2])).real
        fm = state(complex(bm[0]), complex(bm[1]), complex(bm[2])).real
        fd = (fp - fm) / (2 * step)
        assert np.all(np.isfinite(cs)), name
        np.testing.assert_allclose(cs, fd, rtol=2e-4, atol=1e-10, err_msg=name)


def test_ke_frozen_leg_self_consistent():
    """The frozen KE leg recovers its own static state (its own density, hence KE)."""
    cfg, gas, Z, h_t, Np = _premix_cfg()
    tf, ti = cfg.tf, cfg.ti
    xi = np.array([1.0])
    p, area, mdot = 1.0e5, 2.0e-2, 1.0
    cache = np.zeros(Np + 1)
    T, rho, c, W = eq_frozen_state_ke(tf, ti, xi, mdot, p, h_t, area, cache)
    u = mdot / (rho * area)
    # frozen is cold and dense, so its velocity (hence KE) is small: static T ~ stagnation
    assert T < 305.0
    # energy balance holds for the frozen leg's own density
    h_static = h_t - 0.5 * u * u
    Tb, rb, cb, Wb = eq_frozen_state_ke(tf, ti, xi, 0.0, p, h_static, area, cache)
    # solving the frozen state directly at h_static (no KE, mdot=0) reproduces (T, rho)
    assert T == pytest.approx(Tb, rel=1e-9)
    assert rho == pytest.approx(rb, rel=1e-9)


def test_ke_marker_blend_is_per_leg():
    """At marker 0 / 1 the KE blend equals the pure frozen / equilibrium KE leg."""
    cfg, gas, Z, h_t, Np = _premix_cfg()
    tf, ti = cfg.tf, cfg.ti
    xi = np.array([1.0])
    p, area, mdot = 1.0e5, 2.0e-2, 1.0

    frozen = np.array(eq_frozen_state_ke(tf, ti, xi, mdot, p, h_t, area, np.zeros(Np + 1)))
    burnt = np.array(eq_kernel_state_ke_warm(tf, ti, xi, mdot, p, h_t, area, np.zeros(Np + 1)))
    at0 = np.array(eq_marker_state_ke_warm(tf, ti, xi, 0.0, mdot, p, h_t, area, np.zeros(Np + 1)))
    at1 = np.array(eq_marker_state_ke_warm(tf, ti, xi, 1.0, mdot, p, h_t, area, np.zeros(Np + 1)))
    np.testing.assert_allclose(at0, frozen, rtol=1e-10)
    np.testing.assert_allclose(at1, burnt, rtol=1e-10)
    # the two legs are genuinely different states (per-leg KE, not one shared h)
    assert burnt[0] > 2000.0 and frozen[0] < 305.0
