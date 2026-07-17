"""Automatic (CEA-style) product-species set from the network feeds.

A reacting network need not carry a hand-curated species list.  Given the feed
compositions and a species database, the reachable product slate is *every* gas-phase
species buildable from the fed-in elements, reduced (when that slate is large) to the
species that are non-trace at equilibrium across the feed-mixing range.  The final
species set also carries the declared feed species so the frozen closure and the enthalpy
datum can be evaluated.

This is the single policy shared by both entry points that need it: the YAML / case
loader (:mod:`nefes.io.yaml_in`), which resolves the species set while parsing a case, and
the Python network build (:func:`nefes.shell.build.finalize_thermo`), which resolves it
when a deferred :func:`nefes.thermo.equilibrium` config meets its network.

Exports :func:`auto_product_set`.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List

import numpy as np

from ..chem.composition import elemental_Z, species_mass_fractions
from .edge_state import AUTO_REDUCE_THRESHOLD
from .reduction import SampleState, SpeciesReductionWarning, get_reducer
from .species import CONDENSED_PRODUCT_TMAX


def _dedup(seq: Iterable[str]) -> List[str]:
    """De-duplicate a sequence, preserving first-seen order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _feed_species(feed_specs) -> List[str]:
    """Feed/source species named anywhere in the network (the reactants), first-seen order.

    Each ``spec`` is any object carrying a ``composition_spec`` mapping (a node spec or an
    :class:`~nefes.shell.build.ElementSpec`); specs without a composition are skipped.
    """
    out: List[str] = []
    for sp in feed_specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        for name in comp:
            if name not in out:
                out.append(name)
    return out


def _feed_sample_states(feed_lib, feed_specs, *, p_ref: float, T_init: float):
    """Representative equilibrium probe states along the feed-mixing line.

    Each feed stream contributes its elemental composition; convex (mass) combinations of
    the distinct streams span the lean-to-rich range the network can realize, probed at a
    couple of temperatures bracketing the burnt-gas guess.  Used to drive slate reduction.
    """
    T_samples = sorted({T_init, max(1500.0, 0.7 * T_init)})

    feeds = []
    for sp in feed_specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        Y = species_mass_fractions(feed_lib, comp, getattr(sp, "basis", "mole"))
        feeds.append(elemental_Z(feed_lib, Y))

    uniq = []
    for Z in feeds:
        if not any(np.allclose(Z, U, atol=1e-9) for U in uniq):
            uniq.append(Z)

    elems = list(feed_lib.elements)

    def zdict(Z):
        return {elems[i]: float(Z[i]) for i in range(len(elems))}

    states = []
    for Z in uniq:
        states += [SampleState(zdict(Z), T, p_ref) for T in T_samples]
    for ia in range(len(uniq)):
        for ib in range(ia + 1, len(uniq)):
            for w in (0.1, 0.3, 0.5, 0.7, 0.9):
                Zm = w * uniq[ia] + (1.0 - w) * uniq[ib]
                states += [SampleState(zdict(Zm), T, p_ref) for T in T_samples]
    return states


def auto_product_set(
    db,
    feed_specs,
    *,
    p_ref: float,
    T_init: float,
    reducer_name: str = "equilibrium_sampling",
    threshold: float = None,
    reduce_above: int = None,
    max_species: int = None,
    must_species: Iterable[str] = (),
):
    """CEA-style automatic product slate over a ``SpeciesDatabase`` database ``db``.

    Declared feed species fix the reachable element pool; the candidate gas-phase slate is
    every species buildable from those elements, reduced (when large) to the species that
    are non-trace at equilibrium across the feed-mixing range.  The final species_set also
    carries the declared feed species (including condensed fuels) so the frozen closure and
    the enthalpy datum can be evaluated; the equilibrium kernel masks condensed species out
    of the products.

    The slate size has five dials: which reducer runs (``reducer_name``), how deep it trims
    (``threshold``), the candidate count above which it runs at all (``reduce_above``), a
    ceiling on the kept count (``max_species``), and species to keep regardless of their
    abundance (``must_species``).  Setting ``max_species`` runs the reduction even when the
    candidate count is below ``reduce_above``, since a cap has nothing to act on otherwise.

    Parameters
    ----------
    db : nefes.thermo.SpeciesDatabase
        The species database (the packaged NASA Glenn / CEA data, or a ``thermo.inp`` path).
    feed_specs : iterable
        The network's stream-introducing specs (inlets, sources, backflow-bearing outlets),
        each carrying a ``composition_spec`` and a ``basis`` naming its feed species.
    p_ref : float
        Reference pressure [Pa] the equilibrium probe states are evaluated at.
    T_init : float
        Burnt-gas temperature guess [K]; sets the probe-state temperatures for reduction.
    reducer_name : str, optional
        Registry key of the slate reducer (default ``"equilibrium_sampling"``); ``"none"``
        keeps every candidate.  Runs only when the candidate count exceeds ``reduce_above``,
        unless ``max_species`` is set.
    threshold : float, optional
        Trace mole-fraction threshold forwarded to the reducer: a species is kept when its
        peak equilibrium mole fraction across the feed-mixing samples clears it (subject to
        the reducer's safety margin).  Larger keeps fewer species, smaller keeps more.
        ``None`` uses the reducer's own default.
    reduce_above : int, optional
        Reduction runs only when the candidate count exceeds this; a smaller value forces
        reduction on a lean slate, a larger one keeps a broad slate whole.  ``None`` uses
        :data:`~nefes.thermo.edge_state.AUTO_REDUCE_THRESHOLD`.
    max_species : int, optional
        Ceiling on the number of kept species (``None`` for no cap).  Species are ranked by
        peak equilibrium mole fraction and the slate filled to this many, after the declared
        feed species, the ``must_species``, and one carrier of every fed-in element (which
        count against the ceiling).  Not accepted together with ``reducer_name="none"``.
    must_species : iterable of str, optional
        Species to keep regardless of abundance, for instance a marker or pollutant that is
        trace at equilibrium, or a high-temperature condensed product such as graphite
        ``C(gr)``.  Each must be present in the database, buildable from the fed-in elements,
        and an eligible equilibrium product (a gas or a condensed species whose data reaches
        combustion temperatures); an element no feed supplies, an ion, or a feed-only condensed
        species is rejected.

    Returns
    -------
    nefes.thermo.SpeciesSet
        The resolved species_set, carrying a ``reduction_report`` attribute recording which
        products were selected and why.

    Raises
    ------
    ValueError
        If no feed or source declares a composition, if ``max_species`` is combined with
        ``reducer_name="none"``, or if a ``must_species`` names an element no feed supplies,
        is an ion, or is a feed-only condensed species.
    KeyError
        If a feed species or a ``must_species`` is absent from the database.
    """
    if max_species is not None and str(reducer_name or "equilibrium_sampling") == "none":
        raise ValueError("max_species cannot be combined with reducer='none' (which keeps every candidate)")

    declared = _feed_species(feed_specs)
    if not declared:
        raise ValueError(
            "the reacting (equilibrium) model with automatic species needs at least one feed "
            "or source composition (pass an explicit species set to override)"
        )
    missing = [n for n in declared if n not in db]
    if missing:
        raise KeyError(f"feed species not in thermo.inp: {missing}")

    pool = set()
    for name in declared:
        pool.update(el for el in db[name].composition if el != "E")
    candidates = db.candidate_species(pool, gas_only=True, exclude_ions=True)
    declared_gas = [n for n in declared if db[n].phase == 0]

    must = _validate_must_species(db, must_species, pool)
    # A high-temperature condensed product (graphite C(gr), say) is a legitimate forced keep but is
    # absent from the gas-only candidate list, so add any forced species the list misses.
    candidates = _dedup(candidates + [m for m in must if m not in candidates])
    always_keep = _dedup(declared_gas + must)

    gate = AUTO_REDUCE_THRESHOLD if reduce_above is None else int(reduce_above)
    if len(candidates) <= gate and max_species is None:
        # Small candidate pool and no cap: keep every candidate (the must_species are already
        # among them) plus the declared feed species.
        report = {"reducer": "none", "n_candidates": len(candidates), "n_kept": len(candidates)}
        final = _dedup(candidates + declared)
    else:
        feed_lib = db.select(_dedup(declared))
        samples = _feed_sample_states(feed_lib, feed_specs, p_ref=p_ref, T_init=T_init)
        reducer_kwargs = {} if threshold is None else {"threshold": float(threshold)}
        if max_species is not None:
            reducer_kwargs["max_species"] = int(max_species)
        reducer = get_reducer(str(reducer_name or "equilibrium_sampling"), **reducer_kwargs)
        result = reducer.reduce(db.select(candidates), samples, always_keep=always_keep)
        report = result.report
        final = _dedup(result.species + declared)
        _warn_must_below_threshold(must, report)

    lib = db.select(final)
    lib.reduction_report = report  # auditable: which products were selected and why
    return lib


def _validate_must_species(db, must_species, pool) -> List[str]:
    """Check that every forced-keep species is an equilibrium product buildable from the feed elements.

    Returns the de-duplicated list.  A species absent from the database raises ``KeyError``; one
    naming an element no feed supplies, an ion, or a feed-only condensed species that does not
    persist as a product raises ``ValueError``, so the request fails predictably rather than
    silently doing nothing.  A gas species, or a high-temperature condensed product whose data
    reaches combustion temperatures (graphite ``C(gr)``, say), is accepted.
    """
    must = _dedup(must_species)
    for name in must:
        if name not in db:
            raise KeyError(f"must_species not in thermo.inp: {name!r}")
        sp = db[name]
        extra = {el for el in sp.composition if el != "E"} - pool
        if extra:
            raise ValueError(
                f"must_species {name!r} contains element(s) {sorted(extra)} that no feed supplies; "
                "the equilibrium cannot place an element with no feed source"
            )
        if "E" in sp.composition:  # the electron pseudo-element marks a charged (ionic) species
            raise ValueError(f"must_species {name!r} is an ion; the subsonic combustion slate carries no ions")
        if getattr(sp, "phase", 0) != 0 and float(sp.thermo.Tranges.max()) < CONDENSED_PRODUCT_TMAX:
            raise ValueError(
                f"must_species {name!r} is a feed-only condensed species; it does not persist as an "
                "equilibrium product (its data does not reach combustion temperatures)"
            )
    return must


def _warn_must_below_threshold(must, report) -> None:
    """Warn (listing them) when forced-keep species were kept despite being trace at equilibrium."""
    peaks = report.get("peak_mole_fraction")
    floor = report.get("keep_floor")
    if not must or peaks is None or floor is None:
        return
    trace = [n for n in must if peaks.get(n, 0.0) < floor]
    if trace:
        warnings.warn(
            f"must_species kept below the trace threshold ({floor:g}): {trace}",
            SpeciesReductionWarning,
            stacklevel=2,
        )
