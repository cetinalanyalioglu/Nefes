"""Phase 0 validation: smooth primitives are analytic and complex-step-safe.

Checks, for each primitive:
  * the complex-step derivative matches the analytic derivative to ~1e-12,
  * the float64 and complex128 numba specializations both compile and agree on
    the real part,
and that Fischer-Burmeister encodes the complementarity ``a,b>=0, a*b=0``.
"""

import numpy as np
import pytest

from fns.assembly.smooth import (
    smooth_abs,
    smooth_pos,
    smooth_step,
    smooth_sign_sq,
    fischer_burmeister,
)

CS_H = 1e-30  # complex-step size


def cs_deriv(f, x, *args):
    """Complex-step derivative of f w.r.t. its first argument at real x."""
    return f(complex(x, CS_H), *args).imag / CS_H


# (function, analytic derivative w.r.t. x, delta)
CASES = [
    (smooth_abs, lambda x, d: x / np.sqrt(x * x + d * d), 0.1),
    (smooth_pos, lambda x, d: 0.5 * (1.0 + x / np.sqrt(x * x + d * d)), 0.1),
    (smooth_step, lambda x, d: 0.5 * d * d / (x * x + d * d) ** 1.5, 0.1),
    (smooth_sign_sq, lambda x, d: (2.0 * x * x + d * d) / np.sqrt(x * x + d * d), 0.1),
]


@pytest.mark.parametrize("f, dfdx, delta", CASES)
@pytest.mark.parametrize("x", [-3.0, -0.7, -0.05, 0.0, 0.05, 0.7, 3.0])
def test_complex_step_matches_analytic(f, dfdx, delta, x):
    got = cs_deriv(f, x, delta)
    expected = dfdx(x, delta)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("f, _dfdx, delta", CASES)
@pytest.mark.parametrize("x", [-2.0, 0.0, 1.3])
def test_real_and_complex_specializations_agree(f, _dfdx, delta, x):
    real_val = f(x, delta)
    complex_val = f(complex(x, 0.0), delta)
    assert isinstance(real_val, float)
    assert complex_val.real == pytest.approx(real_val, rel=1e-14, abs=1e-15)


def test_fischer_burmeister_complex_step():
    eps = 1e-5
    a, b = 0.8, 0.3
    da = fischer_burmeister(complex(a, CS_H), b, eps).imag / CS_H
    db = fischer_burmeister(a, complex(b, CS_H), eps).imag / CS_H
    denom = np.sqrt(a * a + b * b + eps * eps)
    assert da == pytest.approx(1.0 - a / denom, rel=1e-10)
    assert db == pytest.approx(1.0 - b / denom, rel=1e-10)


def test_fischer_burmeister_complementarity():
    eps = 1e-6
    # Feasible corners (one slack, the other zero) -> residual ~ 0.
    assert fischer_burmeister(2.5, 0.0, eps) == pytest.approx(0.0, abs=1e-6)
    assert fischer_burmeister(0.0, 4.0, eps) == pytest.approx(0.0, abs=1e-6)
    assert fischer_burmeister(0.0, 0.0, eps) == pytest.approx(-eps, abs=1e-9)
    # Both strictly positive -> strictly positive residual (not complementary).
    assert fischer_burmeister(1.0, 1.0, eps) > 1e-3
    # A negative argument (infeasible) -> strongly negative residual.
    assert fischer_burmeister(-1.0, 0.0, eps) == pytest.approx(-2.0, abs=1e-5)


def test_smoothness_through_zero():
    # No jump across x = 0 for any primitive (continuity at the kink).
    delta = 0.05
    for f in (smooth_abs, smooth_pos, smooth_step, smooth_sign_sq):
        left = f(-1e-9, delta)
        right = f(1e-9, delta)
        assert abs(left - right) < 1e-6
