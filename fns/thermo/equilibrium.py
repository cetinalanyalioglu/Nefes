"""Equilibrium / frozen thermo backend: the ``fns`` side of ``thermolib`` (AD-3).

A :class:`thermolib.SpeciesLibrary` (the species *data* -- NASA polynomials and
the element matrix, all chemical equilibrium needs) is packed into the immutable
``(tf, ti)`` array bundle together with the network's **feed streams**.  A feed
stream is one distinct injected composition (an oxidizer, a diluent, a fuel,
...); the network transports one conserved band-1 scalar per stream -- the
**mixture fraction** ``xi_k`` -- and the closures reconstruct everything else by a
forward blend of ``xi`` (see :mod:`fns.composition`).  The compiled kernels in
:mod:`fns.thermo._chem` unpack the flat arrays and run the element-potential
equilibrium solve, so all chemistry stays behind the single thermo boundary and
inside the network's ``@njit`` residual path.

Two per-edge models share one library + stream set:

* ``EQ_FROZEN`` -- the unburnt side: a frozen (non-reacting) real-gas state whose
  species moles are the **forward blend** ``n_feed = xi @ Nfeed`` of the feed
  streams.  No element inversion, no "element-distinguishable" restriction, so the
  unburnt composition can be any mixture of arbitrarily many co-injected fuels.
* ``EQ_KERNEL`` -- the burnt side: HP chemical equilibrium of the elemental
  composition ``Z = xi @ Zfeed`` over the full library.

MVP simplification: the kinetic-energy coupling ``h = h_t - u^2/2`` is dropped
(``h ~ h_t``), ``O(M^2)`` for a low-Mach combustor; the closure returns the static
density at ``h_t`` (R-B2.2 restores the exact KE fixed point later).

``tf`` layout (all float64, ``mol`` units to match thermolib)::

    [ p_ref, T_init, T_init_frozen,
      coeffs(Ns*MI*9), Tint(Ns*(MI-1)), A(Ne*Ns), element_weights(Ne),
      Zfeed(K*Ne), Nfeed(K*Nf), feed_coeffs(Nf*MI*9), feed_Tint(Nf*(MI-1)) ]

``ti = [Ne, Ns, MI, MI-1, Nf, K]`` (``MI`` = max NASA intervals; ``Nf`` = feed-
species union; ``K`` = number of feed streams = transported mixture fractions).
"""

import numpy as np
from numba import njit

from ..composition import stream_pack_arrays
from ._chem import RU, equil_state_cs, equilibrate_hp_cs, equilibrium_sound_speed, frozen_state_from_moles_cs

_OFF_BLOCKS = 3  # p_ref, T_init, T_init_frozen precede the flat data blocks


def pack_equilibrium(lib, stream_Y, T_init=3000.0, T_init_frozen=300.0):
    """Pack a ``thermolib.SpeciesLibrary`` + its feed streams into ``(tf, ti)``.

    ``stream_Y`` is ``(K, n_species)`` mass fractions, one row per distinct feed
    stream (:func:`fns.composition.build_streams`).  ``K = 0`` packs a stream-less
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
    directly (used by the kernel-vs-thermolib tests).  The network calls
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
    T, rho, c, ntot, flag, nit = equil_state_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_init, nj_init)
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

    T, nj, ntot, flag, nit = equilibrate_hp_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_start, nj_init)
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
def eq_total_pressure(M, p, T, c, W):
    """Isentropic total pressure for a variable-gamma gas (gamma = c^2 W / (R_u T))."""
    gamma = c * c * W / (RU * T)
    return p * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))
