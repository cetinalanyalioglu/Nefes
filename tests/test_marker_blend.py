"""Burnt-marker blend kernel: the smooth gate and the frozen/equilibrium blend.

The reacting closure ``EQ_MARKER`` runs both the frozen (unburnt) and equilibrium (burnt)
states and blends them by a smooth gate of the transported burnt marker ``b``.  The gate is
normalized so ``g(0) = 0`` and ``g(1) = 1`` exactly -- a fresh edge is *pure* frozen, a burnt
edge *pure* equilibrium -- with the blend active only in transients.  These pin the gate
shape, the two endpoints against the standalone closures, and the complex-step contract.
"""

import os

import numpy as np
import pytest

from fns.assembly.smooth import marker_gate
from fns.thermo.equilibrium import (
    pack_equilibrium,
    eq_frozen_state,
    eq_kernel_state,
    eq_marker_state,
    eq_marker_state_warm,
    MARKER_GATE_WIDTH,
)
from fns.chem.composition import species_mass_fractions, enthalpy_mass

MECH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thermolib", "data", "h2o2.yaml")


def _bundle():
    """A two-stream (air, H2) equilibrium bundle plus a stoich-ish mixed-edge point."""
    from thermolib import SpeciesLibrary

    lib = SpeciesLibrary.from_native(MECH)
    Y_air = species_mass_fractions(lib, {"O2": 0.21, "N2": 0.79}, "mole")
    Y_h2 = species_mass_fractions(lib, {"H2": 1.0}, "mole")
    stream_Y = np.array([Y_air, Y_h2])
    tf, ti = pack_equilibrium(lib, stream_Y)
    # xi: mass fractions of each stream on the edge (near-stoich H2/air ~ 0.029 by mass)
    xi = np.array([1.0 / 1.029, 0.029 / 1.029])
    h = xi[0] * enthalpy_mass(lib, Y_air, 300.0) + xi[1] * enthalpy_mass(lib, Y_h2, 300.0)
    return tf, ti, xi, float(h)


# -- the gate ---------------------------------------------------------------


def test_gate_endpoints_exact():
    # zero leak at the bimodal converged values: pure frozen at b=0, pure equilibrium at b=1
    assert marker_gate(0.0, MARKER_GATE_WIDTH) == pytest.approx(0.0, abs=1e-14)
    assert marker_gate(1.0, MARKER_GATE_WIDTH) == pytest.approx(1.0, abs=1e-14)
    assert marker_gate(0.5, MARKER_GATE_WIDTH) == pytest.approx(0.5, abs=1e-12)


def test_gate_monotone_and_bounded():
    b = np.linspace(-0.5, 1.5, 401)
    g = np.array([marker_gate(float(x), MARKER_GATE_WIDTH) for x in b])
    assert np.all(np.diff(g) > 0.0)  # strictly increasing everywhere (no overshoot kinks)
    # bounded on the unit interval; outside it the normalized rational stays finite
    inside = (b >= 0.0) & (b <= 1.0)
    assert np.all((g[inside] >= -1e-12) & (g[inside] <= 1.0 + 1e-12))
    assert np.all(np.isfinite(g))


def test_gate_complex_step_matches_fd():
    x0, d = 0.37, 1e-6
    cs = marker_gate(x0 + 1j * 1e-30, MARKER_GATE_WIDTH).imag / 1e-30
    fd = (marker_gate(x0 + d, MARKER_GATE_WIDTH) - marker_gate(x0 - d, MARKER_GATE_WIDTH)) / (2 * d)
    assert cs == pytest.approx(fd, rel=1e-6)


# -- the blend --------------------------------------------------------------


def test_blend_endpoints_match_pure_closures():
    tf, ti, xi, h = _bundle()
    p = 101325.0
    frozen = np.array(eq_frozen_state(tf, ti, xi, h, p))
    burnt = np.array(eq_kernel_state(tf, ti, xi, h, p))
    # the flame genuinely ignites: equilibrium is much hotter than the cold reactant
    assert burnt[0] > frozen[0] + 800.0

    at0 = np.array(eq_marker_state(tf, ti, xi, 0.0, h, p))
    at1 = np.array(eq_marker_state(tf, ti, xi, 1.0, h, p))
    assert np.allclose(at0, frozen, rtol=1e-12)
    assert np.allclose(at1, burnt, rtol=1e-12)


def test_blend_is_convex_interpolation():
    tf, ti, xi, h = _bundle()
    p = 101325.0
    frozen = np.array(eq_frozen_state(tf, ti, xi, h, p))
    burnt = np.array(eq_kernel_state(tf, ti, xi, h, p))
    g = marker_gate(0.5, MARKER_GATE_WIDTH)
    mid = np.array(eq_marker_state(tf, ti, xi, 0.5, h, p))
    assert np.allclose(mid, (1.0 - g) * frozen + g * burnt, rtol=1e-12)


def test_blend_warm_matches_cold():
    tf, ti, xi, h = _bundle()
    p = 101325.0
    Np = int(ti[6])
    cache = np.zeros(Np + 1)
    for b in (0.0, 0.3, 1.0):
        cold = np.array(eq_marker_state(tf, ti, xi, b, h, p))
        warm = np.array(eq_marker_state_warm(tf, ti, xi, b, h, p, cache))
        assert np.allclose(cold, warm, rtol=1e-9)


def test_blend_complex_step_matches_fd():
    tf, ti, xi, h = _bundle()
    p, b = 101325.0, 0.42
    d = 1e-6
    # d(rho)/d(marker): complex step vs central difference
    cs = eq_marker_state(tf, ti, xi, b + 1j * 1e-30, h, p)[1].imag / 1e-30
    fd = (eq_marker_state(tf, ti, xi, b + d, h, p)[1] - eq_marker_state(tf, ti, xi, b - d, h, p)[1]) / (2 * d)
    assert cs == pytest.approx(fd, rel=1e-5)
    # d(T)/d(h_t): the blended caloric coupling is complex-step-clean
    dh = 1.0
    cs_h = eq_marker_state(tf, ti, xi, b, h + 1j * 1e-30, p)[0].imag / 1e-30
    fd_h = (eq_marker_state(tf, ti, xi, b, h + dh, p)[0] - eq_marker_state(tf, ti, xi, b, h - dh, p)[0]) / (2 * dh)
    assert cs_h == pytest.approx(fd_h, rel=1e-5)
