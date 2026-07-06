"""Species-slate reduction: trim a candidate product set to what actually matters.

A CEA-style candidate slate (every species reachable from the fed-in elements) is correct
but bloated: a hydrocarbon/air pool admits ~100+ gas species while only ~20 are ever
non-trace at equilibrium. The bloat is pure cost, since the element-potential Newton solve
scales super-linearly in the species count. A :class:`SpeciesReducer` takes the candidate
library plus a handful of representative thermodynamic states and returns the subset worth
keeping.

The reducer is pluggable: the orchestration code selects one by name via
:func:`get_reducer`, so a future algorithm (kinetics-informed, sensitivity-based, ...) can
be dropped in with :func:`register_reducer` without touching callers.

This runs at **setup time** on real-valued states, off the complex-step residual path, so
it is free to use thresholds and branches.

Public: :class:`SampleState`, :class:`ReductionResult`, :class:`SpeciesReducer`,
:class:`NullReducer`, :class:`EquilibriumSamplingReducer`, :func:`get_reducer`,
:func:`register_reducer`, :func:`available_reducers`.
"""

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
    "get_reducer",
    "register_reducer",
    "available_reducers",
]


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
    """Strategy interface: candidate library + sample states -> kept species subset."""

    #: registry key / human-facing name
    name = "base"

    @abstractmethod
    def reduce(self, library, samples, *, always_keep=()) -> ReductionResult:
        """Reduce ``library`` to the species worth keeping.

        Parameters
        ----------
        library : nefes.thermo.SpeciesLibrary
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

    def reduce(self, library, samples, *, always_keep=()) -> ReductionResult:
        kept = list(library.species_names)
        report = {"reducer": self.name, "n_candidates": len(kept), "n_kept": len(kept)}
        return ReductionResult(species=kept, report=report)


class EquilibriumSamplingReducer(SpeciesReducer):
    """Keep species that are non-trace at chemical equilibrium across sample states.

    For each :class:`SampleState` the candidate slate is equilibrated (TP) and every
    species' mole fraction recorded; the union of species exceeding a (margin-relaxed)
    trace threshold at *any* sample is kept. Sampling several states, e.g. along the
    feed-mixing line from lean to rich, guards against a single operating point missing
    species that matter elsewhere in the network.

    Parameters
    ----------
    threshold : float, optional
        Runtime trace threshold the kept set should reproduce (default ``1e-8``).
    margin : float, optional
        Safety factor: species are kept down to ``threshold / margin`` so a species that is
        marginally trace at a sample but matters slightly off-sample is not dropped
        (default ``100``, keeping down to ``1e-10``).
    """

    name = "equilibrium_sampling"

    def __init__(self, threshold=1e-8, margin=100.0):
        self.threshold = float(threshold)
        self.margin = float(margin)

    def reduce(self, library, samples, *, always_keep=()) -> ReductionResult:
        names = list(library.species_names)
        keep_floor = self.threshold / self.margin
        peak = {name: 0.0 for name in names}
        n_ok = 0
        n_failed = 0
        for s in samples:
            try:
                res = equilibrate_TP(library, s.Z_elem, s.T, s.p)
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

        if n_ok == 0:
            # No sample converged; fall back to keeping everything rather than nothing.
            kept = list(names)
        else:
            kept = [name for name in names if peak[name] >= keep_floor]

        for name in always_keep:
            if name not in kept and name in library.species_index:
                kept.append(name)

        report = {
            "reducer": self.name,
            "n_candidates": len(names),
            "n_kept": len(kept),
            "samples_used": n_ok,
            "samples_failed": n_failed,
            "threshold": self.threshold,
            "keep_floor": keep_floor,
            "peak_mole_fraction": {n: peak[n] for n in kept if n in peak},
        }
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
