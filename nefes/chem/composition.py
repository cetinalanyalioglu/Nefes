"""Composition descriptors: species-named mixtures -> the transported **feed-stream
mixture fractions** ``xi``.

The reacting network transports one conserved band-1 scalar per **feed stream**
(each distinct injected composition: an oxidizer, a diluent, a fuel, ...).  A feed
stream's mass is conserved through every junction, mass source and flame
(combustion conserves elemental -- hence stream-origin -- mass), so its transport
is source-free and acoustically neutral.  The mixture fractions ``xi`` reconstruct
the unburnt speciation exactly by a forward blend ``Y = sum_k xi_k Y_k``, so an
arbitrary number of co-mixed fuels is admissible.

The feed streams come from one of two modes.  In **auto** mode the streams are the
*distinct injected compositions* (auto-merged), so their number equals the distinct
feeds.  In **declared** mode the streams are named up front (``equilibrium(streams=...)``)
and each feed states its composition as a blend over them (:func:`resolve_stream_blend`);
a premixed feed then keeps its constituent streams separate, so its mixture fraction (an
equivalence ratio) stays a live degree of freedom even with a single inlet.  Either way
the two reconstructions the kernels consume are both forward linear maps of ``xi``:

* unburnt (frozen) species moles ``n_feed = sum_k xi_k n_k`` (each stream's fixed
  per-kg mole vector over the feed-species union);
* elemental ``Z = sum_k xi_k Z_k`` (the equilibrium kernel's feed).

Users think in **species** -- "inject ``C12H26``", "the air is 21% O2 / 79% N2 by
mole".  This module is the parse-time bridge: a species mixture (by **mass** or
**mole** fraction, named exactly as in the thermo data) becomes a feed stream, and
its absolute (formation-inclusive) specific enthalpy is evaluated for the
``Tt -> h_t`` datum the solver carries.

Everything here is pure Python (parse time) -- it never runs inside the compiled
residual path; only the resulting stream maps and ``h_t`` flow into the kernels.
"""

from __future__ import annotations

import numpy as np

# Moles of O2 consumed per atom of each oxidizable element on complete combustion
# (C -> CO2, H -> H2O, S -> SO2); oxygen already in the mixture *supplies* O2 at half
# an O2 per O atom.
_O2_PER_ATOM = {"C": 1.0, "H": 0.25, "S": 1.0, "O": -0.5}

# Elements that carry no oxygen demand on complete combustion (they form inert
# products: N -> N2, the noble gases).  Listed explicitly so that a present element
# in neither this set nor ``_O2_PER_ATOM`` raises in :func:`_o2_demand` instead of
# being silently dropped from the oxygen balance.
_O2_INERT = frozenset({"N", "He", "Ne", "Ar", "Kr", "Xe"})

# Two streams are "the same" if their mass fractions match to this tolerance; the
# same ``species_mass_fractions`` call is deterministic, so identical compositions
# compare exactly -- this only guards against floating-point dust.
_STREAM_ATOL = 1e-12


def species_mass_fractions(species_set, spec, basis="mole"):
    """Species **mass** fractions ``Y`` from a named mixture ``spec``.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        Provides ``species_index`` and ``molar_masses``.
    spec : dict or array_like
        Either ``{species_name: fraction}`` (unnormalized is fine), or a full
        ``(n_species,)`` array already in the chosen basis.
    basis : {"mole", "mass"}
        Whether the given fractions are mole or mass fractions.

    Returns
    -------
    Y : ndarray, shape (n_species,)
        Normalized species mass fractions in species set order.
    """
    Ns = species_set.n_species
    W = np.asarray(species_set.molar_masses, dtype=float)
    vec = np.zeros(Ns)
    if isinstance(spec, dict):
        idx = species_set.species_index
        for name, val in spec.items():
            if name not in idx:
                raise KeyError(
                    f"species {name!r} is not in the species set; available e.g. "
                    f"{list(idx)[:8]}{'...' if Ns > 8 else ''}"
                )
            vec[idx[name]] = val
    else:
        vec = np.asarray(spec, dtype=float).copy()
        if vec.shape != (Ns,):
            raise ValueError(f"composition array must have shape ({Ns},), got {vec.shape}")

    if np.any(vec < 0.0):
        raise ValueError("composition fractions must be non-negative")
    tot = vec.sum()
    if tot <= 0.0:
        raise ValueError("composition must have a positive total")

    if basis == "mole":
        Y = vec * W
    elif basis == "mass":
        Y = vec.copy()
    else:
        raise ValueError("basis must be 'mole' or 'mass'")
    return Y / Y.sum()


def elemental_Z(species_set, Y):
    """Elemental **mass** fractions ``Z`` from species mass fractions ``Y``.

    ``Z_i = W_i * Σ_j a_ij Y_j / W_j``, then renormalized to sum to one.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        Provides ``molar_masses``, ``element_matrix`` and ``element_weights``.
    Y : array_like, shape (n_species,)
        Species mass fractions (need not be normalized).

    Returns
    -------
    ndarray, shape (n_elements,)
        Normalized elemental mass fractions in element order.
    """
    Y = np.asarray(Y, dtype=float)
    Yn = Y / Y.sum()
    W = np.asarray(species_set.molar_masses, dtype=float)
    A = np.asarray(species_set.element_matrix, dtype=float)
    ew = np.asarray(species_set.element_weights, dtype=float)
    gram_atoms = A @ (Yn / W)
    Z = ew * gram_atoms
    return Z / Z.sum()


def enthalpy_mass(species_set, Y, T):
    """Absolute specific enthalpy [J/kg] of species mass fractions ``Y`` at ``T``.

    Formation-inclusive, as carried by the NASA polynomials.  Used to convert an
    inlet/source total temperature to the transported ``h_t``.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        Provides ``molar_masses`` and the dimensionless enthalpy ``h_RT(T)``.
    Y : array_like, shape (n_species,)
        Species mass fractions (need not be normalized).
    T : float
        Temperature [K] at which to evaluate the enthalpy.

    Returns
    -------
    float
        Absolute (formation-inclusive) specific enthalpy [J/kg].
    """
    # ``species_set`` is a nefes.thermo object, so nefes.thermo is importable here; the lazy
    # import keeps the universal gas constant a single source of truth without making
    # nefes.thermo a load-time dependency of this parse-time module.
    from nefes.thermo.constants import R_UNIVERSAL

    Y = np.asarray(Y, dtype=float)
    Yn = Y / Y.sum()
    W = np.asarray(species_set.molar_masses, dtype=float)
    hRT = np.asarray(species_set.h_RT(float(T)), dtype=float)
    return float(R_UNIVERSAL * T * np.sum(Yn * hRT / W))


def resolve_composition(species_set, spec, basis="mole"):
    """Convenience: a named mixture -> ``(Y, Z)`` (species and elemental mass fractions).

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        The species data.
    spec : dict or array_like
        A named species mixture ``{species: fraction}`` (unnormalized is fine) or a full
        ``(n_species,)`` array already in ``basis``.
    basis : {"mole", "mass"}, optional
        Whether the given fractions are mole or mass fractions (default ``"mole"``).

    Returns
    -------
    Y : ndarray, shape (n_species,)
        Normalized species mass fractions.
    Z : ndarray, shape (n_elements,)
        Normalized elemental mass fractions.
    """
    Y = species_mass_fractions(species_set, spec, basis)
    Z = elemental_Z(species_set, Y)
    return Y, Z


def species_mole_fractions(species_set, spec, basis="mole"):
    """Species **mole** fractions ``X`` from a named mixture ``spec``.

    The mole-fraction companion of :func:`species_mass_fractions` (same ``spec`` /
    ``basis`` conventions).

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
    spec : dict or array_like
        ``{species_name: fraction}`` (unnormalized is fine) or a full
        ``(n_species,)`` array already in ``basis``.
    basis : {"mole", "mass"}
        Whether the given fractions are mole or mass fractions.

    Returns
    -------
    X : ndarray, shape (n_species,)
        Normalized species mole fractions in species set order.
    """
    Y = species_mass_fractions(species_set, spec, basis)
    W = np.asarray(species_set.molar_masses, dtype=float)
    moles = Y / W
    return moles / moles.sum()


def _o2_demand(species_set, X):
    """Net O2 demand [mol O2 per mol mixture] of a mole-fraction vector ``X``.

    Positive for a fuel (needs oxygen), negative for an oxidizer (supplies it).  A
    present element classified in neither ``_O2_PER_ATOM`` (oxidizable) nor
    ``_O2_INERT`` raises, rather than being silently dropped from the balance and
    biasing the equivalence ratio.
    """
    A = np.asarray(species_set.element_matrix, dtype=float)  # (n_elements, n_species)
    atoms = A @ np.asarray(X, dtype=float)  # gram-atoms of each element per mole of mixture
    demand = 0.0
    for el, i in species_set.element_index.items():
        if atoms[i] <= 0.0:
            continue  # element absent from this mixture
        if el in _O2_PER_ATOM:
            demand += _O2_PER_ATOM[el] * atoms[i]
        elif el not in _O2_INERT:
            raise ValueError(
                f"element {el!r} is present but has no defined O2 combustion demand; cannot form "
                f"the equivalence-ratio blend (oxidizable {sorted(_O2_PER_ATOM)}, inert {sorted(_O2_INERT)})"
            )
    return float(demand)


def _bundled_feed_species_set(*compositions):
    """A minimal ``SpeciesSet`` over the packaged data, holding just the named species.

    The species named across the given ``{species: fraction}`` compositions are looked up in
    the packaged NASA Glenn / CEA data; enough to supply the formulae and molar masses the
    blend needs, without a product slate.  Raises if a composition is not a name-keyed mapping
    (an array cannot be interpreted without a species set).
    """
    from ..thermo import SpeciesDatabase

    names = []
    for comp in compositions:
        if not isinstance(comp, dict):
            raise ValueError(
                "species_set=None needs named {species: fraction} fuel and oxidizer; pass an explicit "
                "species_set to use full composition arrays"
            )
        for name in comp:
            if name not in names:
                names.append(name)
    return SpeciesDatabase().select(names)


def equivalence_ratio_mixture(
    fuel, oxidizer, phi, *, species_set=None, fuel_basis="mole", oxidizer_basis="mole", basis="mole"
):
    """Blend a ``fuel`` and an ``oxidizer`` to a target equivalence ratio ``phi``.

    Parameters
    ----------
    fuel, oxidizer : dict or array_like
        Compositions as ``{species_name: fraction}`` (e.g. ``{"CH4": 1.0}``,
        ``{"O2": 0.21, "N2": 0.79}``) or full ``(n_species,)`` arrays, each read in
        its own basis.
    phi : float
        Equivalence ratio (``>= 0``).  ``0`` returns the pure oxidizer.
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism, optional
        Supplies the species formulae (``element_matrix``) and molar masses.  Leave it
        ``None`` (the default) to look the named species up in the packaged NASA Glenn / CEA
        data; pass an explicit species set to use a custom mechanism or full composition arrays.
    fuel_basis, oxidizer_basis : {"mole", "mass"}
        Basis of the given ``fuel`` / ``oxidizer`` fractions.
    basis : {"mole", "mass"}
        Basis of the **returned** blend fractions.

    Returns
    -------
    dict
        ``{species_name: fraction}`` for every species present in the blend, in
        ``basis`` and normalized to sum to one.  Ready to pass straight to a
        composition-bearing element (inlet / mass source) or
        :func:`species_mass_fractions`.

    Raises
    ------
    ValueError
        If ``phi`` is negative, the ``fuel`` has no net oxygen demand, the ``oxidizer``
        supplies no oxygen, or ``species_set=None`` is used with array (not name-keyed) inputs.
    """
    if phi < 0.0:
        raise ValueError(f"equivalence ratio phi must be non-negative; got {phi}")

    if species_set is None:
        species_set = _bundled_feed_species_set(fuel, oxidizer)

    X_fuel = species_mole_fractions(species_set, fuel, fuel_basis)
    X_ox = species_mole_fractions(species_set, oxidizer, oxidizer_basis)

    d_fuel = _o2_demand(species_set, X_fuel)
    d_ox = _o2_demand(species_set, X_ox)
    if d_fuel <= 0.0:
        raise ValueError("the 'fuel' has no net oxygen demand (not a fuel for these elements C/H/S/O)")
    if d_ox >= 0.0:
        raise ValueError("the 'oxidizer' supplies no oxygen (its net O2 demand is non-negative)")

    # Stoichiometric (phi = 1) fuel-to-oxidizer mole ratio from the combined O2 balance
    # n_fuel * d_fuel + n_ox * d_ox = 0, then scale the fuel by phi (n_ox := 1 mole).
    n_fuel = phi * (-d_ox / d_fuel)
    moles = n_fuel * X_fuel + X_ox

    if basis == "mole":
        frac = moles / moles.sum()
    elif basis == "mass":
        mass = moles * np.asarray(species_set.molar_masses, dtype=float)
        frac = mass / mass.sum()
    else:
        raise ValueError("basis must be 'mole' or 'mass'")

    names = list(species_set.species_index)
    return {names[j]: float(frac[j]) for j in np.nonzero(frac > 0.0)[0]}


def build_streams(species_set, comps):
    """Distinct **feed streams** from a list of named compositions (auto-merged).

    Resolves each element's composition to species_set mass fractions and collapses
    identical ones onto a single stream, so the network transports one scalar per
    *distinct* injected composition rather than one per injecting element.  Returns
    those distinct streams together with, for every input, the index of the stream
    it maps to.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
    comps : list of (spec, basis)
        One ``(composition_spec, basis)`` per stream-introducing element (inlet,
        mass source, composition-bearing outlet), in node order.

    Returns
    -------
    stream_Y : ndarray, shape (K, n_species)
        Mass fractions of each distinct stream (species set order).
    assignment : list of int
        ``assignment[i]`` is the stream index of input ``comps[i]`` (``-1`` if its
        ``spec`` was ``None`` -- e.g. an outlet with inert backflow).

    Compositions that are identical (to ``_STREAM_ATOL``) collapse onto one stream,
    so injecting "air" in five places costs a single transported scalar.
    """
    stream_Y = []
    assignment = []
    for spec, basis in comps:
        if spec is None:
            assignment.append(-1)
            continue
        Y = species_mass_fractions(species_set, spec, basis)
        k = -1
        for j, Yj in enumerate(stream_Y):
            if np.allclose(Y, Yj, rtol=0.0, atol=_STREAM_ATOL):
                k = j
                break
        if k < 0:
            k = len(stream_Y)
            stream_Y.append(Y)
        assignment.append(k)
    Ns = species_set.n_species
    arr = np.array(stream_Y, dtype=float) if stream_Y else np.zeros((0, Ns))
    return arr, assignment


def stream_mean_molar_masses(species_set, stream_Y):
    """Mean molar mass ``[kg/mol]`` of each feed stream from its species mass fractions.

    The mass-weighted harmonic mean ``W_k = 1 / sum_j (Y_kj / W_j)`` -- the molar mass for
    which one kilogram of stream ``k`` holds ``1 / W_k`` moles.  Converts a mole-basis
    stream blend to the transported mass fractions.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        Provides ``molar_masses``.
    stream_Y : ndarray, shape (K, n_species)
        Each stream's species mass fractions (rows need not be renormalized here; each is
        already normalized by :func:`species_mass_fractions`).

    Returns
    -------
    ndarray, shape (K,)
        The per-stream mean molar mass.
    """
    stream_Y = np.atleast_2d(np.asarray(stream_Y, dtype=float))
    W = np.asarray(species_set.molar_masses, dtype=float)
    moles_per_kg = (stream_Y / W).sum(axis=1)  # 1/W_k = sum_j Y_kj / W_j
    return 1.0 / moles_per_kg


def resolve_stream_blend(species_set, stream_labels, stream_Y, blend, basis="mole"):
    """Feed donor mixture fractions ``xi`` from a blend over the declared stream basis.

    A feed injects a mixture of the network's declared streams; this resolves how much of
    each stream's *mass* it carries -- the transported mixture-fraction donor ``xi`` (one
    entry per declared stream, summing to one).  Streams named in ``blend`` contribute; the
    rest are zero.  The blend amounts are relative (need not sum to one): the user states a
    ratio and it is normalized here.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        Provides ``molar_masses`` (only used for a ``"mole"`` blend).
    stream_labels : sequence of str
        The declared stream names, in transported-scalar order.
    stream_Y : ndarray, shape (K, n_species)
        Declared stream mass fractions; row ``k`` is stream ``stream_labels[k]``.
    blend : dict of {str: float}
        ``{stream_name: amount}`` -- non-negative relative amounts of each named stream.
    basis : {"mole", "mass"}, optional
        Whether the amounts are relative moles of each stream (converted through the stream
        mean molar mass, :func:`stream_mean_molar_masses`) or relative masses (default
        ``"mole"``).

    Returns
    -------
    ndarray, shape (K,)
        The donor mixture fractions ``xi`` in declared-stream order (non-negative, sum one).

    Examples
    --------
    A twenty-to-one mole blend of ``"air"`` and ``"H2"`` yields the mass fractions of that
    premix; ``xi @ stream_Y`` reconstructs the premixed species composition exactly.
    """
    labels = list(stream_labels)
    index = {name: k for k, name in enumerate(labels)}
    amount = np.zeros(len(labels), dtype=float)
    for name, val in blend.items():
        if name not in index:
            raise KeyError(
                f"stream {name!r} is not a declared stream; the network declares {labels}. "
                f"Declare it in equilibrium(streams=...) or fix the feed's stream name."
            )
        a = float(val)
        if a < 0.0:
            raise ValueError(f"stream amount for {name!r} must be non-negative; got {a}")
        amount[index[name]] += a
    if amount.sum() <= 0.0:
        raise ValueError("a stream blend must name at least one stream with a positive amount")
    if basis == "mole":
        mass = amount * stream_mean_molar_masses(species_set, stream_Y)  # moles -> mass
    elif basis == "mass":
        mass = amount
    else:
        raise ValueError("basis must be 'mole' or 'mass'")
    return mass / mass.sum()


def stream_pack_arrays(species_set, stream_Y):
    """Forward-map arrays for the packed kernel from per-stream mass fractions.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
    stream_Y : ndarray, shape (K, n_species)
        Mass fractions of each feed stream (from :func:`build_streams`).

    Returns
    -------
    feed_idx : ndarray of int, shape (Nf,)
        Library indices of the feed-species union (every species present in any
        stream) -- the basis the frozen closure reconstructs over.
    Nfeed : ndarray, shape (K, Nf)
        Moles per kilogram of each feed species in each stream (``Y/W``).  The
        unburnt mixture is ``n_feed = xi @ Nfeed``.
    Zfeed : ndarray, shape (K, n_elements)
        Elemental mass fractions of each stream.  The equilibrium feed is
        ``Z = xi @ Zfeed``.
    """
    stream_Y = np.atleast_2d(np.asarray(stream_Y, dtype=float))
    K = stream_Y.shape[0]
    W = np.asarray(species_set.molar_masses, dtype=float)
    if K == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros((0, 0)), np.zeros((0, species_set.n_elements)))
    present = np.any(stream_Y > 0.0, axis=0)
    feed_idx = np.nonzero(present)[0].astype(np.int64)
    Nfeed = stream_Y[:, feed_idx] / W[feed_idx]
    Zfeed = np.array([elemental_Z(species_set, stream_Y[k]) for k in range(K)], dtype=float)
    return feed_idx, Nfeed, Zfeed


def network_elements(species_set, specs):
    """Elements actually present across a list of named mixtures.

    Reports which of the species set's elements are exercised by at least one mixture,
    so a caller can warn about (or trim to) the active union.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism
        The authority on the element set (``element_matrix``, ``elements``).
    specs : iterable of dict or array_like
        Named mixtures (inlet / source compositions); a ``dict`` is read on a mole
        basis, a full ``(n_species,)`` array on a mass basis.

    Returns
    -------
    list of str
        Element symbols (in species set order) whose abundance is non-zero in at least
        one mixture.
    """
    A = np.asarray(species_set.element_matrix, dtype=float)
    present = np.zeros(species_set.n_elements, dtype=bool)
    for spec in specs:
        Y = (
            species_mass_fractions(species_set, spec, basis="mass")
            if not isinstance(spec, dict)
            else species_mass_fractions(species_set, spec, basis="mole")
        )
        active_sp = Y > 0.0
        present |= np.any(A[:, active_sp] != 0.0, axis=1)
    return [e for i, e in enumerate(species_set.elements) if present[i]]
