"""Parse-time thermo configuration (pure Python).

Builds the immutable ``(model_id, tf, ti)`` bundle and a manifest describing the
transported composition (empty for a perfect gas).  The bundle has a fixed dtype
/ contiguity signature across all models so a single compiled ``thermo_update``
serves every backend.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .api import EQ_KERNEL, PERFECT_GAS

# Universal gas constant [J/(kmol*K)]; molar mass W = RU / R_specific [kg/kmol].
RU = 8314.462618


@dataclass(frozen=True)
class ThermoConfig:
    """Immutable thermo bundle passed read-only through the kernels."""

    model_id: int
    tf: np.ndarray  # float64[::1]
    ti: np.ndarray  # int64[::1]
    element_names: List[str] = field(default_factory=list)
    species_names: List[str] = field(default_factory=list)
    # Parse-time only (never packed/compiled): the thermolib SpeciesLibrary, kept
    # so element/source builders can resolve species-named compositions to the feed
    # streams (transported mixture fractions) and the ``Tt -> h_t`` datum.
    library: object = None
    # Packing controls for a deferred equilibrium config: when no feed streams were
    # given at config time they are discovered from the network's inlet/source
    # compositions and packed at build time (see ``catalog.finalize_thermo``).
    t_init: float = 3000.0
    t_init_frozen: float = 300.0

    @property
    def n_elem(self) -> int:
        return len(self.element_names)

    @property
    def n_species(self) -> int:
        return len(self.species_names)


def perfect_gas(R: float = 287.0, gamma: float = 1.4) -> ThermoConfig:
    """Calorically-perfect-gas configuration (default: dry air)."""
    cp = gamma * R / (gamma - 1.0)
    W = RU / R
    tf = np.ascontiguousarray([cp, R, W], dtype=np.float64)
    ti = np.empty(0, dtype=np.int64)
    return ThermoConfig(model_id=PERFECT_GAS, tf=tf, ti=ti)


def perfect_gas_passive_scalars(n_scalars: int, R: float = 287.0, gamma: float = 1.4, names=None) -> ThermoConfig:
    """Perfect gas that also advects ``n_scalars`` passive conserved scalars.

    The thermodynamics are unchanged (the perfect-gas kernels ignore ``Z_el``);
    each scalar simply adds one band-1 unknown and one source-free transport
    equation per edge.  Used to exercise the composition-transport framework
    (reactive-flow D-4) without invoking chemistry.
    """
    if n_scalars < 1:
        raise ValueError("n_scalars must be >= 1")
    cfg = perfect_gas(R, gamma)
    names = list(names) if names is not None else [f"scalar{i}" for i in range(n_scalars)]
    if len(names) != n_scalars:
        raise ValueError("len(names) must equal n_scalars")
    return ThermoConfig(model_id=PERFECT_GAS, tf=cfg.tf, ti=cfg.ti, element_names=names)


def equilibrium(library, streams=None, basis: str = "mole", T_init: float = 3000.0, T_init_frozen: float = 300.0):
    """Reacting-gas config from a ``thermolib.SpeciesLibrary``.

    The transported composition is the network's **feed-stream mixture fractions**
    ``xi`` -- one conserved band-1 scalar per distinct injected composition (an
    oxidizer, a diluent, a fuel, ...).  Both per-edge closures reconstruct from
    ``xi`` by a forward blend: the unburnt (``EQ_FROZEN``) side gets its species
    moles ``n_feed = xi @ Nfeed`` and the burnt (``EQ_KERNEL``) side gets the
    elemental ``Z = xi @ Zfeed`` for the HP-equilibrium solve.  The base model is
    ``EQ_KERNEL``; mark approach edges ``EQ_FROZEN`` via
    ``build_problem(..., edge_models=...)``.

    Parameters
    ----------
    library : thermolib.SpeciesLibrary or thermolib.Mechanism
        The species data (NASA-7/9 polynomials, element matrix).  ``thermolib`` is
        the standalone authority; this packs its canonical 9-term arrays for the
        compiled kernel.
    streams : dict of {str: spec}, optional
        Explicit feed streams ``{label: composition}`` (each ``composition`` a named
        species mixture in ``basis`` units).  Useful for direct kernel/closure use
        and for pinning the scalar order/labels.  **Default ``None`` defers stream
        discovery to build time**: ``build_problem`` collects the distinct inlet /
        mass-source / outlet compositions and packs them automatically -- so a user
        only ever names species at the elements that introduce them.
    basis : {"mole", "mass"}
        Units of the explicit ``streams`` compositions (ignored when ``streams`` is
        ``None``; each element then carries its own basis).
    T_init, T_init_frozen : float
        Initial temperature guesses for the equilibrium and frozen solves [K].
    """
    from .equilibrium import pack_equilibrium

    if streams is None:
        stream_Y = np.zeros((0, library.n_species))
        labels: List[str] = []
    else:
        from ..chem.composition import species_mass_fractions

        labels = list(streams.keys())
        stream_Y = np.array([species_mass_fractions(library, streams[k], basis) for k in labels], dtype=np.float64)

    tf, ti = pack_equilibrium(library, stream_Y, T_init, T_init_frozen)
    return ThermoConfig(
        model_id=EQ_KERNEL,
        tf=tf,
        ti=ti,
        element_names=labels,
        species_names=[s.name for s in library.species],
        library=library,
        t_init=T_init,
        t_init_frozen=T_init_frozen,
    )
