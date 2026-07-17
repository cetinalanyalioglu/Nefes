"""Species-slate reduction: trim a candidate product set to what actually matters.

A CEA-style candidate slate (every species reachable from the fed-in elements) is correct
but bloated: a hydrocarbon/air pool admits ~100+ gas species while only ~20 are ever
non-trace at equilibrium. The bloat is pure cost, since the element-potential Newton solve
scales super-linearly in the species count. A :class:`SpeciesReducer` takes the candidate
species set plus a handful of representative thermodynamic states and returns the subset worth
keeping.

The reducer is pluggable: the orchestration code selects one by name via
:func:`get_reducer`, so a future algorithm (kinetics-informed, sensitivity-based, ...) can
be dropped in with :func:`register_reducer` without touching callers.

This runs at **setup time** on real-valued states, off the complex-step residual path, so
it is free to use thresholds and branches.

Public: :class:`SampleState`, :class:`ReductionResult`, :class:`SpeciesReducer`,
:class:`NullReducer`, :class:`EquilibriumSamplingReducer`, :class:`SpeciesReductionWarning`,
:func:`get_reducer`, :func:`register_reducer`, :func:`available_reducers`.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .equilibrate import equilibrate_TP

__all__ = [
    "SampleState",
    "ReductionResult",
    "SpeciesReducer",
    "NullReducer",
    "EquilibriumSamplingReducer",
    "SpeciesReductionWarning",
    "get_reducer",
    "register_reducer",
    "available_reducers",
]


class SpeciesReductionWarning(UserWarning):
    """Raised when a species-slate reduction cannot fully honour the requested dials.

    Emitted, for instance, when a species-count cap discards species that are above the
    trace threshold, when mandatory keeps already exceed the cap, or when no equilibrium
    sample converged so the cap cannot be applied.
    """


@dataclass
class SampleState:
    """One representative thermodynamic state for probing which species matter.

    Parameters
    ----------
    Z_elem : dict
        Elemental mass fractions ``{element: fraction}``.
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].
    """

    Z_elem: Dict[str, float]
    T: float
    p: float


@dataclass
class ReductionResult:
    """Outcome of a reduction.

    Attributes
    ----------
    species : list of str
        Kept species names.
    report : dict
        Diagnostics (candidate/kept counts, per-species peak mole fraction, samples used),
        intended to be echoed so the reduction is auditable rather than a black box.
    """

    species: List[str]
    report: dict = field(default_factory=dict)


class SpeciesReducer(ABC):
    """Strategy interface: candidate species set + sample states -> kept species subset."""

    #: registry key / human-facing name
    name = "base"

    @abstractmethod
    def reduce(self, species_set, samples, *, always_keep=()) -> ReductionResult:
        """Reduce ``species_set`` to the species worth keeping.

        Parameters
        ----------
        species_set : nefes.thermo.SpeciesSet
            The candidate product slate to trim.
        samples : sequence of SampleState
            Representative states the reduction should remain valid across.
        always_keep : iterable of str, optional
            Species that must survive regardless of the algorithm (e.g. declared feed
            species, so compositions stay expressible).

        Returns
        -------
        ReductionResult
        """
        raise NotImplementedError


class NullReducer(SpeciesReducer):
    """Identity reducer: keep every candidate species (no reduction)."""

    name = "none"

    def __init__(self, **kwargs):
        # Accept and ignore the trace-threshold dials so the registry can build any reducer
        # with the same call; keeping every species has no threshold to honour.
        pass

    def reduce(self, species_set, samples, *, always_keep=()) -> ReductionResult:
        kept = list(species_set.species_names)
        report = {"reducer": self.name, "n_candidates": len(kept), "n_kept": len(kept)}
        return ReductionResult(species=kept, report=report)


class EquilibriumSamplingReducer(SpeciesReducer):
    """Keep species that are non-trace at chemical equilibrium across sample states.

    For each :class:`SampleState` the candidate slate is equilibrated (TP) and every
    species' mole fraction recorded; the union of species exceeding a (margin-relaxed)
    trace threshold at *any* sample is kept. Sampling several states, e.g. along the
    feed-mixing line from lean to rich, guards against a single operating point missing
    species that matter elsewhere in the network.

    An optional species-count cap (``max_species``) selects the highest-peaking species
    when a bound on the slate size is wanted, for instance to test how large a set a case
    needs.  The cap is a ceiling, not a target: it only ever discards the lowest-peaking
    above-threshold species, never pads the slate up to the cap with trace species.  Two
    keeps always survive the cap and count against its budget: the mandatory species passed
    in ``always_keep`` (declared feed species and any user-forced species), and one
    highest-peaking carrier of every fed-in element, so the element-potential equilibrium
    never loses a constituent it must balance.

    Parameters
    ----------
    threshold : float, optional
        Runtime trace threshold the kept set should reproduce (default ``1e-8``).
    margin : float, optional
        Safety factor: species are kept down to ``threshold / margin`` so a species that is
        marginally trace at a sample but matters slightly off-sample is not dropped
        (default ``100``, keeping down to ``1e-10``).
    max_species : int, optional
        Ceiling on the number of kept species (``None`` for no cap).  Species are ranked by
        peak mole fraction and the slate filled to this many, after the mandatory and
        element-coverage keeps.  A trace threshold and a cap compose: a species is kept when
        it is mandatory, or when it clears the threshold *and* fits under the cap.
    """

    name = "equilibrium_sampling"

    def __init__(self, threshold=1e-8, margin=100.0, max_species=None):
        self.threshold = float(threshold)
        self.margin = float(margin)
        if max_species is not None and int(max_species) < 1:
            raise ValueError(f"max_species must be a positive integer; got {max_species!r}")
        self.max_species = None if max_species is None else int(max_species)

    def reduce(self, species_set, samples, *, always_keep=()) -> ReductionResult:
        names = list(species_set.species_names)
        index = species_set.species_index
        keep_floor = self.threshold / self.margin

        # -- 1. peak equilibrium mole fraction of every candidate across the samples
        peak = {name: 0.0 for name in names}
        n_ok = 0
        n_failed = 0
        for s in samples:
            try:
                res = equilibrate_TP(species_set, s.Z_elem, s.T, s.p)
            except Exception:
                # A pathological sample (e.g. an element with no gas product) should not
                # abort the whole reduction; skip it and note the count.
                n_failed += 1
                continue
            n_ok += 1
            X = np.real(np.asarray(res.X))
            for j, name in enumerate(names):
                if X[j] > peak[name]:
                    peak[name] = float(X[j])

        report = {
            "reducer": self.name,
            "n_candidates": len(names),
            "samples_used": n_ok,
            "samples_failed": n_failed,
            "threshold": self.threshold,
            "keep_floor": keep_floor,
        }
        if self.max_species is not None:
            report["max_species"] = self.max_species

        if n_ok == 0:
            # No sample converged: there is no ranking, so keep everything rather than
            # nothing.  A cap cannot be honoured without a ranking, so warn and drop it.
            if self.max_species is not None:
                warnings.warn(
                    "no equilibrium sample converged during species reduction; keeping every "
                    "candidate and ignoring the max_species cap",
                    SpeciesReductionWarning,
                    stacklevel=2,
                )
            kept = list(names)
            report["n_kept"] = len(kept)
            report["peak_mole_fraction"] = dict(peak)
            return ReductionResult(species=kept, report=report)

        # candidates ranked high-to-low peak, name as a deterministic tie-break
        ranked = sorted(names, key=lambda n: (-peak[n], n))
        comp = {n: set(species_set.species[index[n]].composition) - {"E"} for n in names}

        # -- 2. mandatory keeps: declared feed species and any user-forced species
        mandatory = [n for n in dict.fromkeys(always_keep) if n in index]

        # -- 3. element coverage: every fed-in element needs a gas carrier in the kept set,
        #       or the element-potential equilibrium is singular.  Promote the highest-peaking
        #       carrier of any element the mandatory keeps leave uncovered.
        pool = set().union(*(comp[n] for n in names)) if names else set()
        covered = set().union(*(comp[n] for n in mandatory)) if mandatory else set()
        coverage: List[str] = []
        for el in sorted(pool - covered):
            carrier = next((n for n in ranked if el in comp[n]), None)
            if carrier is not None and carrier not in mandatory and carrier not in coverage:
                coverage.append(carrier)
                covered |= comp[carrier]

        forced = list(dict.fromkeys(mandatory + coverage))  # non-negotiable; count against the cap
        forced_set = set(forced)
        above = [n for n in ranked if peak[n] >= keep_floor and n not in forced_set]

        # -- 4. apply the cap as a ceiling: fill with the highest-peaking above-threshold
        #       species, but never below the mandatory + coverage keeps
        dropped = 0
        if self.max_species is None:
            kept = list(dict.fromkeys(forced + above))
        elif len(forced) >= self.max_species:
            kept = list(forced)
            if len(forced) > self.max_species:
                warnings.warn(
                    f"max_species={self.max_species} is below the {len(forced)} mandatory and "
                    "element-coverage keeps; the cap is not honoured and all of them are kept",
                    SpeciesReductionWarning,
                    stacklevel=2,
                )
        else:
            budget = self.max_species - len(forced)
            kept = list(dict.fromkeys(forced + above[:budget]))
            dropped = len(above) - budget
            if dropped > 0:
                warnings.warn(
                    f"max_species={self.max_species} discarded {dropped} species above the trace "
                    f"threshold ({keep_floor:g}); the reduced slate may not reproduce the full "
                    "equilibrium",
                    SpeciesReductionWarning,
                    stacklevel=2,
                )

        report["n_kept"] = len(kept)
        report["peak_mole_fraction"] = {n: peak[n] for n in kept}
        report["coverage_added"] = list(coverage)  # carriers kept only for element coverage
        report["dropped_above_threshold"] = max(0, dropped)
        return ReductionResult(species=kept, report=report)


# -- registry ---------------------------------------------------------------
_REDUCERS = {
    EquilibriumSamplingReducer.name: EquilibriumSamplingReducer,
    NullReducer.name: NullReducer,
}


def register_reducer(name, cls):
    """Register a :class:`SpeciesReducer` subclass under ``name`` for :func:`get_reducer`."""
    if not issubclass(cls, SpeciesReducer):
        raise TypeError(f"{cls!r} is not a SpeciesReducer subclass")
    _REDUCERS[name] = cls


def available_reducers():
    """Names of the registered reducers."""
    return sorted(_REDUCERS)


def get_reducer(name="equilibrium_sampling", **kwargs):
    """Instantiate a registered reducer by name.

    Parameters
    ----------
    name : str
        Registry key (see :func:`available_reducers`).
    **kwargs
        Forwarded to the reducer constructor.

    Returns
    -------
    SpeciesReducer
    """
    try:
        cls = _REDUCERS[name]
    except KeyError:
        raise ValueError(f"unknown species reducer {name!r}; available: {available_reducers()}")
    return cls(**kwargs)
