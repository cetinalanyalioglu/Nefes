"""Smooth (branch-free) replacements for ``abs``/``min``/``max``/step.

Differentiation contract (REQUIREMENTS R-A6.2): no in-library residual or
property path may use ``abs``/``sign``/``max`` or branch on a *complex*
argument, because the consumer differentiates the whole call path by the
complex-step method.  These helpers are smooth and complex-analytic, so a
perturbation ``x + i*eps`` propagates a meaningful derivative through them.

The library carries its **own** copy of these helpers; per AD-2 it never
imports ``cbnflow`` (which has an analogous ``cbnflow.smooth``).

Each function takes a smoothing scale ``eps`` controlling the width of the
region over which the kink is rounded.  With ``eps -> 0`` they recover the
non-smooth function exactly.
"""

import numpy as np

__all__ = ["smooth_abs", "smooth_max", "smooth_min", "smooth_heaviside", "smooth_pos"]


def smooth_abs(x, eps=1e-12):
    """Smooth absolute value: ``sqrt(x**2 + eps**2)``.

    Analytic everywhere (no branch); ``-> |x|`` as ``eps -> 0``.
    """
    return np.sqrt(x * x + eps * eps)


def smooth_max(a, b, eps=1e-12):
    """Smooth maximum, ``0.5*((a+b) + smooth_abs(a-b))``."""
    return 0.5 * (a + b + smooth_abs(a - b, eps))


def smooth_min(a, b, eps=1e-12):
    """Smooth minimum, ``0.5*((a+b) - smooth_abs(a-b))``."""
    return 0.5 * (a + b - smooth_abs(a - b, eps))


def smooth_pos(x, eps=1e-12):
    """Smooth positive part ``max(x, 0)`` = ``0.5*(x + smooth_abs(x))``."""
    return 0.5 * (x + smooth_abs(x, eps))


def smooth_heaviside(x, eps=1e-3):
    """Smooth step in ``[0, 1]`` using ``0.5*(1 + tanh(x/eps))``.

    ``tanh`` is analytic, so this is complex-step safe.
    """
    return 0.5 * (1.0 + np.tanh(x / eps))
