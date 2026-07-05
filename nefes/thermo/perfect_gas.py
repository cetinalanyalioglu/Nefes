"""Calorically-perfect-gas thermo kernels.

The density recovery is the load-bearing, complex-step-critical routine of the
whole solver.  Given mass-flux ``m = mdot/A``, static pressure ``p`` and total
enthalpy ``h_t`` it returns the density ``rho`` solving

    F(rho) = rho - p*K / (h_t - m^2/(2 rho^2)) = 0,   K = cp/R

which is strictly monotone increasing with a single root for any sign of ``m``
(positive, negative, or zero), with no subsonic/supersonic branch ambiguity.

The root is found by a safeguarded Newton iteration **on the real part only**
(the iteration branches, which is not complex-step transparent).  The imaginary
part -- all that complex-step differentiation needs -- is then attached
analytically by the implicit function theorem:

    drho = -(F_m * Im(m) + F_p * Im(p) + F_h * Im(h_t)) / F_rho
"""

import numpy as np
from numba import njit, types
from numba.extending import overload


@njit(cache=True)
def _solve_density_real(m, p, ht, K):
    """Unique real root of F(rho); real scalar inputs only."""
    if not (p > 0.0) or not (ht > 0.0):
        raise ValueError("non-physical state: p <= 0 or h_t <= 0")

    m2 = m * m
    # F(p*K/ht) <= 0 and F is increasing, so the root lies in [p*K/ht, inf).
    lo = p * K / ht
    rho = lo

    # Expanding upper bracket: grow hi until F(hi) > 0 (with H > 0).
    hi = max(2.0 * lo, np.sqrt(m2 / (2.0 * ht)) * 4.0 + lo)
    bracketed = False
    for _ in range(200):
        H = ht - m2 / (2.0 * hi * hi)
        if H > 0.0 and hi - p * K / H > 0.0:
            bracketed = True
            break
        hi *= 2.0
    if not bracketed:
        raise ValueError("density bracket expansion failed")

    # Safeguarded Newton (bisection fallback keeps the iterate inside [lo, hi]).
    for _ in range(100):
        H = ht - m2 / (2.0 * rho * rho)
        if H <= 0.0:
            rho = 0.5 * (rho + hi)
            continue
        F = rho - p * K / H
        if F > 0.0:
            hi = rho
        else:
            lo = rho
        dF = 1.0 + p * K * m2 / (rho**3 * H * H)
        rho_new = rho - F / dF
        if not (lo < rho_new < hi):
            rho_new = 0.5 * (lo + hi)
        if abs(rho_new - rho) <= 1e-14 * rho:
            return rho_new
        rho = rho_new
    return rho


def _attach_density_imag(rho_r, f_rho, f_m, f_p, f_h, m, p, ht):
    """Attach the IFT-spliced imaginary part (dtype-dispatched)."""
    return rho_r  # pure-python / float fallback


@overload(_attach_density_imag, inline="always")
def _attach_density_imag_ovl(rho_r, f_rho, f_m, f_p, f_h, m, p, ht):
    any_complex = isinstance(m, types.Complex) or isinstance(p, types.Complex) or isinstance(ht, types.Complex)
    if any_complex:

        def impl(rho_r, f_rho, f_m, f_p, f_h, m, p, ht):
            drho = -(f_m * m.imag + f_p * p.imag + f_h * ht.imag) / f_rho
            return rho_r + 1j * drho

        return impl

    def impl(rho_r, f_rho, f_m, f_p, f_h, m, p, ht):
        return rho_r

    return impl


@njit(cache=True)
def pg_solve_density(m, p, ht, K):
    """Density from mass flux ``m``, static ``p`` and total enthalpy ``ht``.

    Dtype-generic: float in -> float out (real root), complex in -> complex out
    (real root + IFT-spliced imaginary part).
    """
    mr = m.real
    pr = p.real
    hr = ht.real
    rho_r = _solve_density_real(mr, pr, hr, K)

    m2 = mr * mr
    H = hr - m2 / (2.0 * rho_r * rho_r)
    f_rho = 1.0 + pr * K * m2 / (rho_r**3 * H * H)
    f_m = -pr * K * mr / (rho_r * rho_r * H * H)
    f_p = -K / H
    f_h = pr * K / (H * H)
    return _attach_density_imag(rho_r, f_rho, f_m, f_p, f_h, m, p, ht)


@njit(cache=True)
def pg_state(tf, h, p):
    """Return scalar band-2 fields ``(T, rho, c, W)`` from static ``(h, p)``.

    The scalar-tuple sibling of ``pg_update`` for the hot recover path (avoids
    allocating a dtype-matched output buffer inside the kernel).
    """
    cp = tf[0]
    R = tf[1]
    T = h / cp
    rho = p / (R * T)
    gamma = cp / (cp - R)
    c = np.sqrt(gamma * R * T)
    return T, rho, c, tf[2]


@njit(cache=True)
def pg_total_pressure(tf, M, p):
    """Isentropic total pressure ``p * (1 + (g-1)/2 M^2)^(g/(g-1))``."""
    cp = tf[0]
    R = tf[1]
    gamma = cp / (cp - R)
    return p * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))


@njit(cache=True)
def pg_update(tf, ti, Z_el, h, p, mode, out):
    """Fill band-2 thermo fields (T, rho, c, W) from static (h, p).

    ``tf = [cp, R, W]``; ``ti`` and ``Z_el`` unused for a perfect gas.  Writes
    into ``out`` (dtype follows the caller: float on the residual path, complex
    on the Jacobian-seed path).
    """
    cp = tf[0]
    R = tf[1]
    T = h / cp
    rho = p / (R * T)
    gamma = cp / (cp - R)
    c = np.sqrt(gamma * R * T)
    out[0] = T
    out[1] = rho
    out[2] = c
    out[3] = tf[2]
