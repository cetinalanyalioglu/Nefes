"""Smooth (branch-free, complex-analytic) helpers."""

import numpy as np

from nefes.thermo import smooth


def test_smooth_abs_recovers_abs():
    xs = np.linspace(-5, 5, 101)
    assert np.allclose(smooth.smooth_abs(xs, eps=1e-9), np.abs(xs), atol=1e-6)


def test_smooth_max_min():
    a, b = 2.0, 5.0
    assert np.isclose(smooth.smooth_max(a, b, eps=1e-9), 5.0, atol=1e-6)
    assert np.isclose(smooth.smooth_min(a, b, eps=1e-9), 2.0, atol=1e-6)


def test_smooth_functions_are_complex_analytic():
    # A real function f has a derivative recoverable by the complex step.
    eps = 1e-200
    for fn in (smooth.smooth_abs, smooth.smooth_pos):
        x = 1.3
        d_cs = fn(x + 1j * eps).imag / eps
        d_fd = (fn(x + 1e-6) - fn(x - 1e-6)) / 2e-6
        assert np.isclose(d_cs, d_fd, rtol=1e-6)


def test_smooth_heaviside_bounds():
    assert smooth.smooth_heaviside(-10.0) < 1e-3
    assert smooth.smooth_heaviside(10.0) > 1 - 1e-3
    assert np.isclose(smooth.smooth_heaviside(0.0), 0.5)
