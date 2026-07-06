"""Equilibrium / frozen edge-state producers: the network side of the thermochemistry.

A :class:`nefes.thermo.SpeciesLibrary` (the species *data* -- NASA polynomials and
the element matrix, all chemical equilibrium needs) is packed into the immutable
``(tf, ti)`` array bundle together with the network's **feed streams**.  A feed
stream is one distinct injected composition (an oxidizer, a diluent, a fuel,
...); the network transports one conserved band-1 scalar per stream -- the
**mixture fraction** ``xi_k`` -- and the closures reconstruct everything else by a
forward blend of ``xi`` (see :mod:`nefes.composition`).  The compiled kernels in
:mod:`nefes.thermo.kernel` unpack the flat arrays and run the element-potential
equilibrium solve, so all chemistry stays behind the single thermo boundary and
inside the network's ``@njit`` residual path.

Two per-edge models share one library + stream set:

* ``EQ_FROZEN`` -- the unburnt side: a frozen (non-reacting) real-gas state whose
  species moles are the **forward blend** ``n_feed = xi @ Nfeed`` of the feed
  streams.  No element inversion, no "element-distinguishable" restriction, so the
  unburnt composition can be any mixture of arbitrarily many co-injected fuels.
* ``EQ_KERNEL`` -- the burnt side: HP chemical equilibrium of the elemental
  composition ``Z = xi @ Zfeed`` over the full library.

The kinetic-energy coupling is exact: the closures recover the static state at
``h = h_t - u^2/2``.  Because static ``p`` is the band-1 unknown, ``G(h) = h_t -
1/2 (mdot / (rho(h) A))^2 - h`` is strictly monotone in ``h``, so the ``ke``
wrappers (:func:`eq_kernel_state_ke_warm` and friends) take a safeguarded
bracketed root of ``G`` on the real part and splice the imaginary part by the
implicit-function theorem -- the same shape as the perfect-gas density root, no
caps or floors.

``tf`` layout (all float64, ``mol`` units throughout)::

    [ p_ref, T_init, T_init_frozen,
      coeffs(Ns*MI*9), Tint(Ns*(MI-1)), A(Ne*Ns), element_weights(Ne),
      Zfeed(K*Ne), Nfeed(K*Nf), feed_coeffs(Nf*MI*9), feed_Tint(Nf*(MI-1)) ]

``ti = [Ne, Ns, MI, MI-1, Nf, K]`` (``MI`` = max NASA intervals; ``Nf`` = feed-
species union; ``K`` = number of feed streams = transported mixture fractions).
"""

import numpy as np
from numba import njit, types
from numba.extending import overload

from ..chem.composition import stream_pack_arrays
from ..assembly.smooth import marker_gate
from .kernel import RU, equil_state_cs, equilibrate_hp_cs, equilibrium_sound_speed, frozen_state_from_moles_cs

_OFF_BLOCKS = 3  # p_ref, T_init, T_init_frozen precede the flat data blocks

# Transition width of the burnt-marker blend gate (nefes.smooth.marker_gate).  Gentle (so the
# coupled solve stays Newton-friendly); the gate's zero-leak normalization makes the converged
# accuracy independent of it -- a frozen edge is pure frozen, a burnt edge pure equilibrium.
MARKER_GATE_WIDTH = 0.1

# Candidate-species count above which the automatic (CEA-style) product slate is reduced to
# its non-trace members before packing: past this the equilibrium Newton solve is expensive
# enough that reducing pays off (hydrocarbon/air admits ~115); below it, run the slate raw.
AUTO_REDUCE_THRESHOLD = 40


def pack_equilibrium(lib, stream_Y, T_init=3000.0, T_init_frozen=300.0):
    """Pack a ``nefes.thermo.SpeciesLibrary`` + its feed streams into ``(tf, ti)``.

    ``stream_Y`` is ``(K, n_species)`` mass fractions, one row per distinct feed
    stream (:func:`nefes.composition.build_streams`).  ``K = 0`` packs a stream-less
    bundle -- valid for the standalone equilibrium kernel
    (:func:`eq_kernel_state_from_Z`) but not for an ``EQ_FROZEN`` edge, which needs
    streams to reconstruct from.  ``lib`` may be a ``SpeciesLibrary`` or a
    ``Mechanism`` (which proxies the same data surface).
    """
    coeffs, Tint = lib.nasa9_arrays()  # (Ns, MI, 9), (Ns, MI-1)
    Ne = lib.n_elements
    Ns = lib.n_species
    MI = coeffs.shape[1]
    MIm1 = Tint.shape[1]
    A = np.ascontiguousarray(lib.element_matrix, dtype=np.float64)  # (Ne, Ns)
    ew = np.ascontiguousarray(lib.element_weights, dtype=np.float64)  # (Ne,) kg/mol

    stream_Y = np.atleast_2d(np.asarray(stream_Y, dtype=np.float64)) if np.size(stream_Y) else np.zeros((0, Ns))
    K = stream_Y.shape[0]
    feed_idx, Nfeed, Zfeed = stream_pack_arrays(lib, stream_Y)  # (Nf,), (K, Nf), (K, Ne)
    Nf = feed_idx.shape[0]
    feed_coeffs = np.ascontiguousarray(coeffs[feed_idx], dtype=np.float64)  # (Nf, MI, 9)
    feed_Tint = np.ascontiguousarray(Tint[feed_idx], dtype=np.float64)  # (Nf, MI-1)

    # Product subset for the HP-equilibrium (burnt) kernel: gas-phase species only.  Condensed
    # feed species (e.g. liquid fuel) stay in the full arrays -- so the frozen reconstruction and
    # the feed enthalpy datum can use them -- but never appear as equilibrium products (their
    # polynomials are invalid at flame temperature).  An all-gas library makes this the full set.
    prod_mask = np.asarray(getattr(lib, "product_mask", np.ones(Ns, bool)), dtype=bool)
    prod_idx = np.nonzero(prod_mask)[0].astype(np.int64)
    Np = prod_idx.shape[0]
    prod_A = np.ascontiguousarray(A[:, prod_idx], dtype=np.float64)  # (Ne, Np)
    prod_coeffs = np.ascontiguousarray(coeffs[prod_idx], dtype=np.float64)  # (Np, MI, 9)
    prod_Tint = np.ascontiguousarray(Tint[prod_idx], dtype=np.float64)  # (Np, MI-1)

    header = np.array([lib.P_ref, T_init, T_init_frozen], dtype=np.float64)
    tf = np.concatenate(
        [
            header,
            coeffs.ravel(),
            Tint.ravel(),
            A.ravel(),
            ew,
            np.ascontiguousarray(Zfeed, dtype=np.float64).ravel(),
            np.ascontiguousarray(Nfeed, dtype=np.float64).ravel(),
            feed_coeffs.ravel(),
            feed_Tint.ravel(),
            prod_A.ravel(),
            prod_coeffs.ravel(),
            prod_Tint.ravel(),
        ]
    ).astype(np.float64)
    ti = np.array([Ne, Ns, MI, MIm1, Nf, K, Np], dtype=np.int64)
    return np.ascontiguousarray(tf), np.ascontiguousarray(ti)


@njit(cache=True)
def _product_blocks(tf, ti):
    """Slice the gas-phase product subset ``(prod_coeffs, prod_Tint, prod_A, ew)`` from a bundle.

    The burnt HP-equilibrium kernel solves over these (Np) species, not the full library, so a
    condensed feed species never enters the product set.
    """
    Ne = ti[0]
    Ns = ti[1]
    MI = ti[2]
    MIm1 = ti[3]
    Nf = ti[4]
    K = ti[5]
    Np = ti[6]
    # element weights sit right after the full A block (used to map Z -> gram-atoms b0)
    o_ew = _OFF_BLOCKS + Ns * MI * 9 + Ns * MIm1 + Ne * Ns
    ew = tf[o_ew : o_ew + Ne]
    # product blocks are appended after Zfeed, Nfeed, feed_coeffs, feed_Tint
    o = o_ew + Ne + K * Ne + K * Nf + Nf * MI * 9 + Nf * MIm1
    prod_A = tf[o : o + Ne * Np].reshape((Ne, Np))
    o += Ne * Np
    prod_coeffs = tf[o : o + Np * MI * 9].reshape((Np, MI, 9))
    o += Np * MI * 9
    prod_Tint = tf[o : o + Np * MIm1].reshape((Np, MIm1))
    return prod_coeffs, prod_Tint, prod_A, ew


@njit(cache=True)
def eq_kernel_state_from_Z(tf, ti, Z_el, h, p):
    """HP-equilibrium state ``(T, rho, c_eq, W)`` for an explicit elemental ``Z_el``.

    The stream-free entry point: ``Z_el`` is the gram-fraction elemental vector
    directly (used by the kernel consistency tests).  The network calls
    :func:`eq_kernel_state`, which maps the transported mixture fractions to ``Z``.
    """
    Ne = ti[0]
    Np = ti[6]
    p_ref = tf[0]
    T_init = tf[1]
    coeffs, Tint, Af, ew = _product_blocks(tf, ti)
    b0 = Z_el / ew
    sb = 0.0
    for i in range(Ne):
        sb += b0[i].real
    guess = sb / (2.0 * Np)
    nj_init = np.full(Np, guess)
    # flag/nit are the solver's convergence diagnostics; unused on this path
    T, rho, c, ntot, _flag, _nit = equil_state_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_init, nj_init)
    return T, rho, c, 1.0 / ntot


@njit(cache=True)
def eq_kernel_state_from_Z_warm(tf, ti, Z_el, h, p, cache):
    """HP-equilibrium state ``(T, rho, c_eq, W)``, warm-started from ``cache``.

    Identical to :func:`eq_kernel_state_from_Z`, but the element-potential Newton seeds
    from the per-edge ``cache`` -- a length ``Ns + 1`` buffer holding the last converged
    moles (``cache[:Ns]``) and temperature (``cache[Ns]``) -- and writes the new converged
    pair back, so a nearby re-solve (the next Newton iterate, or a complex-step Jacobian
    column whose real part is unchanged) converges in a couple of steps.  **Both** the
    composition and the temperature must be warm-started together; seeding the composition
    at a stale ``T_init`` leaves the first step badly inconsistent.  ``cache`` is purely a
    *speed* hint: the HP equilibrium is unique, so the converged state is independent of it.
    An empty / mis-sized ``cache`` (e.g. a perfect-gas placeholder) disables the warm start
    and falls back to the uniform composition guess at ``T_init``.
    """
    Ne = ti[0]
    Np = ti[6]
    p_ref = tf[0]
    T_init = tf[1]
    coeffs, Tint, Af, ew = _product_blocks(tf, ti)
    b0 = Z_el / ew

    # Default: the robust uniform composition guess at the cold T_init.
    sb = 0.0
    for i in range(Ne):
        sb += b0[i].real
    nj_init = np.full(Np, sb / (2.0 * Np))
    T_start = T_init
    has_cache = cache.shape[0] == Np + 1
    if has_cache and cache[Np] > 0.0:  # a populated cache: warm-start composition *and* temperature
        T_start = cache[Np]
        for j in range(Np):
            if cache[j] > 0.0:
                nj_init[j] = cache[j]  # exactly-zero (underflowed) species keep the uniform guess

    # flag/nit are the solver's convergence diagnostics; unused on this path
    T, nj, ntot, _flag, _nit = equilibrate_hp_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_start, nj_init)
    rho = p / (RU * ntot * T)
    c = equilibrium_sound_speed(coeffs, Tint, Af, nj, ntot, T, p)
    if has_cache:  # store the converged (real) composition + temperature for the next solve
        for j in range(Np):
            cache[j] = nj[j].real
        cache[Np] = T.real
    return T, rho, c, 1.0 / ntot


@njit(cache=True)
def eq_kernel_state_warm(tf, ti, xi, h, p, cache):
    """Warm-started :func:`eq_kernel_state`: maps ``xi`` to ``Z`` then seeds from ``cache``."""
    Z = _xi_to_Z(tf, ti, xi)
    return eq_kernel_state_from_Z_warm(tf, ti, Z, h, p, cache)


@njit(cache=True)
def _xi_to_Z(tf, ti, xi):
    """Elemental ``Z = xi @ Zfeed`` from the transported mixture fractions (linear)."""
    Ne = ti[0]
    Ns = ti[1]
    MI = ti[2]
    MIm1 = ti[3]
    K = ti[5]
    o = _OFF_BLOCKS + Ns * MI * 9 + Ns * MIm1 + Ne * Ns + Ne  # skip coeffs, Tint, A, ew
    Zfeed = tf[o : o + K * Ne].reshape((K, Ne))
    Z = np.zeros(Ne, dtype=xi.dtype)
    for i in range(Ne):
        acc = xi[0] * 0.0
        for k in range(K):
            acc = acc + xi[k] * Zfeed[k, i]
        Z[i] = acc
    return Z


@njit(cache=True)
def eq_kernel_state(tf, ti, xi, h, p):
    """HP-equilibrium state ``(T, rho, c_eq, W)`` for transported mixture fractions ``xi``.

    Maps ``xi`` to the elemental composition (a linear, complex-analytic blend of
    the feed streams) and delegates to :func:`eq_kernel_state_from_Z`.
    """
    Z = _xi_to_Z(tf, ti, xi)
    return eq_kernel_state_from_Z(tf, ti, Z, h, p)


@njit(cache=True)
def eq_frozen_state(tf, ti, xi, h, p):
    """Frozen real-gas state ``(T, rho, c_frozen, W)`` of the unburnt mixture ``xi``.

    The unburnt species moles are the forward blend ``n_feed = xi @ Nfeed`` of the
    feed streams (no element inversion); the temperature follows from ``h``.
    """
    Ne = ti[0]
    Ns = ti[1]
    MI = ti[2]
    MIm1 = ti[3]
    Nf = ti[4]
    K = ti[5]
    T_init_fr = tf[2]
    o = _OFF_BLOCKS
    o += Ns * MI * 9  # skip full coeffs
    o += Ns * MIm1  # skip full Tint
    o += Ne * Ns  # skip A
    o += Ne  # skip element_weights
    o += K * Ne  # skip Zfeed
    Nfeed = tf[o : o + K * Nf].reshape((K, Nf))
    o += K * Nf
    feed_coeffs = tf[o : o + Nf * MI * 9].reshape((Nf, MI, 9))
    o += Nf * MI * 9
    feed_Tint = tf[o : o + Nf * MIm1].reshape((Nf, MIm1))

    n_feed = np.zeros(Nf, dtype=xi.dtype)
    for f in range(Nf):
        acc = xi[0] * 0.0
        for k in range(K):
            acc = acc + xi[k] * Nfeed[k, f]
        n_feed[f] = acc

    T, rho, c, ntot = frozen_state_from_moles_cs(feed_coeffs, feed_Tint, n_feed, h, p, T_init_fr)
    return T, rho, c, 1.0 / ntot


@njit(cache=True)
def eq_marker_state(tf, ti, xi, marker, h, p):
    """Marker-gated blend of the frozen and equilibrium states ``(T, rho, c, W)``.

    Runs **both** closures at the transported point ``(xi, h, p)`` and blends each recovered
    field by ``g = marker_gate(marker)``::

        field = (1 - g) * frozen + g * equilibrium

    ``g(0) = 0`` (pure frozen / unburnt) and ``g(1) = 1`` (pure equilibrium / burnt) exactly, so
    the blend is only active in transients (the marker is bimodal at convergence).  The same
    ``h_t`` is a valid enthalpy for *both* compositions -- the unburnt reactant inverts to its
    cold ``T``, the burnt mixture to the flame ``T`` -- so both solves always converge and the
    gate selects the physical one.  Complex-step-safe (smooth gate, complex-analytic closures).
    """
    Tf, rf, cf, Wf = eq_frozen_state(tf, ti, xi, h, p)
    Te, re, ce, We = eq_kernel_state(tf, ti, xi, h, p)
    g = marker_gate(marker, MARKER_GATE_WIDTH)
    return (1.0 - g) * Tf + g * Te, (1.0 - g) * rf + g * re, (1.0 - g) * cf + g * ce, (1.0 - g) * Wf + g * We


@njit(cache=True)
def eq_marker_state_warm(tf, ti, xi, marker, h, p, cache):
    """Warm-started :func:`eq_marker_state`: the equilibrium leg seeds from ``cache``.

    Identical to :func:`eq_marker_state`, but the (dominant-cost) burnt equilibrium solve is
    warm-started from the per-edge ``cache`` (moles + temperature).  The frozen leg needs no
    cache.  The equilibrium kernel runs on **every** marker-gated edge regardless of ``marker``
    (its blend weight may be ~0), so the cache stays populated and the burnt product moles are
    available for post-processing on any reacting edge.
    """
    Tf, rf, cf, Wf = eq_frozen_state(tf, ti, xi, h, p)
    Te, re, ce, We = eq_kernel_state_warm(tf, ti, xi, h, p, cache)
    g = marker_gate(marker, MARKER_GATE_WIDTH)
    return (1.0 - g) * Tf + g * Te, (1.0 - g) * rf + g * re, (1.0 - g) * cf + g * ce, (1.0 - g) * Wf + g * We


@njit(cache=True)
def eq_total_pressure(M, p, T, c, W):
    """Isentropic total pressure for a variable-gamma gas (gamma = c^2 W / (R_u T))."""
    gamma = c * c * W / (RU * T)
    return p * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))


# ---------------------------------------------------------------------------
# Kinetic-energy coupling for the reacting closures
# ---------------------------------------------------------------------------
# The ``ke`` wrappers solve the outer static-enthalpy root ``h = h_t - u^2/2``
# described in the module docstring.  ``frozen`` selects the inner closure
# (0 = HP equilibrium / burnt, 1 = frozen / unburnt) so the same machinery serves
# both legs; the marker blend runs the two separately, each with its own density
# and hence its own ``u`` and ``h``.


@njit(cache=True)
def _ke_inner_state(frozen, tf, ti, xi, h, p, cache):
    """Static reacting state ``(T, rho, c, W)`` at ``(xi, h, p)`` for the selected leg."""
    if frozen == 0:
        return eq_kernel_state_warm(tf, ti, xi, h, p, cache)
    return eq_frozen_state(tf, ti, xi, h, p)


@njit(cache=True)
def _ke_root_real(frozen, tf, ti, xi_r, mdot_r, p_r, ht_r, area_r, cache):
    """Real root ``h`` of ``G(h) = h_t - 1/2 (mdot/(rho(h) A))^2 - h`` (monotone).

    ``G`` decreases in ``h`` (a lower static enthalpy is colder, so denser, so
    slower, so less kinetic energy), giving a single root.  ``G(h_t) <= 0`` brackets
    it from above; the lower bracket is expanded until ``G > 0``, then a
    bisection-safeguarded secant converges it.  All inputs are real parts.
    """
    flux = mdot_r / area_r
    k = 0.5 * flux * flux  # kinetic energy = k / rho^2

    h_hi = ht_r
    rho_hi = _ke_inner_state(frozen, tf, ti, xi_r, h_hi, p_r, cache)[1]
    g_hi = -k / (rho_hi * rho_hi)  # = G(h_t)
    if g_hi >= -1e-300:  # quiescent / vanishing flux: the root is h_t itself
        return h_hi

    delta = k / (rho_hi * rho_hi)
    if delta < 1.0:
        delta = 1.0
    h_lo = ht_r - delta
    g_lo = 1.0
    bracketed = False
    for _ in range(200):
        rho_lo = _ke_inner_state(frozen, tf, ti, xi_r, h_lo, p_r, cache)[1]
        g_lo = ht_r - k / (rho_lo * rho_lo) - h_lo
        if g_lo > 0.0:
            bracketed = True
            break
        delta *= 2.0
        h_lo = ht_r - delta
    if not bracketed:
        raise ValueError("kinetic-energy bracket expansion failed")

    a = h_lo  # G(a) > 0
    ga = g_lo
    b = h_hi  # G(b) < 0
    gb = g_hi
    h = 0.5 * (a + b)
    for _ in range(100):
        rho = _ke_inner_state(frozen, tf, ti, xi_r, h, p_r, cache)[1]
        g = ht_r - k / (rho * rho) - h
        if g > 0.0:
            a = h
            ga = g
        else:
            b = h
            gb = g
        h_new = a - ga * (b - a) / (gb - ga)  # secant from the straddling pair
        if not (a < h_new < b):
            h_new = 0.5 * (a + b)
        if abs(h_new - h) <= 1e-13 * (abs(h) + 1.0):
            return h_new
        h = h_new
    return h


def _attach_ke_state(frozen, tf, ti, xi, mdot, p, ht, area, cache, h_s):
    """Recover ``(T, rho, c, W)`` at the converged static enthalpy ``h_s`` (dtype-dispatched).

    Pure-Python base path; the ``@overload`` below provides the compiled
    specializations.  Real inputs -> bare static state at ``h_s``; complex inputs ->
    the same plus the implicit-function imaginary part of ``h_s`` w.r.t. every input
    (so the recovered fields carry the exact complex-step seed).
    """
    return _ke_inner_state(frozen, tf, ti, xi, h_s, p, cache)


@overload(_attach_ke_state, inline="always")
def _attach_ke_state_ovl(frozen, tf, ti, xi, mdot, p, ht, area, cache, h_s):
    xi_complex = getattr(xi, "dtype", None) is not None and isinstance(xi.dtype, types.Complex)
    any_complex = (
        xi_complex
        or isinstance(mdot, types.Complex)
        or isinstance(p, types.Complex)
        or isinstance(ht, types.Complex)
        or isinstance(area, types.Complex)
    )
    if any_complex:

        def impl(frozen, tf, ti, xi, mdot, p, ht, area, cache, h_s):
            p_r = p.real
            area_r = area.real
            mdot_r = mdot.real
            xr = xi.real.astype(np.complex128)  # composition with the seed stripped
            # d rho / d h at the converged point (complex step on the static enthalpy)
            eps = 1e-30
            inner_h = _ke_inner_state(frozen, tf, ti, xr, complex(h_s, eps), complex(p_r, 0.0), cache)[1]
            rho_h = inner_h.imag / eps
            rho_r = inner_h.real
            gp = (mdot_r * mdot_r / (rho_r * rho_r * rho_r * area_r * area_r)) * rho_h - 1.0
            # input-seed contribution to G at the fixed real root (rho carries the p / xi seeds)
            rho0 = _ke_inner_state(frozen, tf, ti, xi.astype(np.complex128), complex(h_s, 0.0), p, cache)[1]
            u0 = mdot / (rho0 * area)
            g0 = ht - 0.5 * u0 * u0 - h_s
            im_hs = -g0.imag / gp
            h_complex = complex(h_s, im_hs)
            return _ke_inner_state(frozen, tf, ti, xi.astype(np.complex128), h_complex, p, cache)

        return impl

    def impl(frozen, tf, ti, xi, mdot, p, ht, area, cache, h_s):
        return _ke_inner_state(frozen, tf, ti, xi, h_s, p, cache)

    return impl


@njit(cache=True)
def _ke_solve(frozen, tf, ti, xi, mdot, p, ht, area, cache):
    """KE-coupled reacting state ``(T, rho, c, W)``; dtype-generic, complex-step-safe."""
    xi_r = xi.real.copy()
    h_s = _ke_root_real(frozen, tf, ti, xi_r, mdot.real, p.real, ht.real, area.real, cache)
    return _attach_ke_state(frozen, tf, ti, xi, mdot, p, ht, area, cache, h_s)


@njit(cache=True)
def eq_kernel_state_ke_warm(tf, ti, xi, mdot, p, ht, area, cache):
    """Burnt (HP-equilibrium) state with the kinetic-energy coupling ``h = h_t - u^2/2``.

    Warm-started from ``cache`` (moles + temperature); the outer KE root reuses it on
    every inner evaluation so the bracketed solve stays cheap.
    """
    return _ke_solve(0, tf, ti, xi, mdot, p, ht, area, cache)


@njit(cache=True)
def eq_frozen_state_ke(tf, ti, xi, mdot, p, ht, area, cache):
    """Unburnt (frozen) state with the kinetic-energy coupling ``h = h_t - u^2/2``.

    ``cache`` is accepted for signature parity with the burnt leg and ignored (the
    frozen solve needs no warm start).
    """
    return _ke_solve(1, tf, ti, xi, mdot, p, ht, area, cache)


@njit(cache=True)
def eq_marker_state_ke_warm(tf, ti, xi, marker, mdot, p, ht, area, cache):
    """Marker-gated blend of the KE-coupled frozen and equilibrium states.

    Each leg solves its **own** kinetic-energy root (its own density, hence its own
    ``u`` and static enthalpy), so the weighted-out leg always sits at a physical
    state -- the frozen leg never inherits the burnt edge's large kinetic energy.
    """
    Tf, rf, cf, Wf = _ke_solve(1, tf, ti, xi, mdot, p, ht, area, cache)
    Te, re, ce, We = _ke_solve(0, tf, ti, xi, mdot, p, ht, area, cache)
    g = marker_gate(marker, MARKER_GATE_WIDTH)
    return (1.0 - g) * Tf + g * Te, (1.0 - g) * rf + g * re, (1.0 - g) * cf + g * ce, (1.0 - g) * Wf + g * We
