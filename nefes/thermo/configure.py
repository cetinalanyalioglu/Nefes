"""Parse-time thermo configuration (pure Python).

Builds the immutable ``(model_id, tf, ti)`` bundle and a manifest describing the
transported composition (empty for a perfect gas).  The bundle has a fixed dtype
/ contiguity signature across all models so a single compiled ``thermo_update``
serves every backend.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from thermolib.constants import R_UNIVERSAL

from .api import EQ_KERNEL, PERFECT_GAS


@dataclass(frozen=True)
class ThermoConfig:
    """Immutable thermo bundle passed read-only through the kernels.

    Attributes
    ----------
    model_id : int
        Integer selecting the gas model (``PERFECT_GAS``, ``EQ_KERNEL``, ...).
    tf : numpy.ndarray
        Packed float64 data block; layout is model-specific (``[cp, R, W]`` for a
        perfect gas, the flat NASA/element arrays for equilibrium).
    ti : numpy.ndarray
        Packed int64 sizes/indices companion to ``tf`` (empty for a perfect gas).
    element_names : list of str
        Labels of the transported band-1 scalars (feed streams / passive scalars).
    species_names : list of str
        Species carried by the library (empty for a perfect gas).
    library : object
        The thermolib ``SpeciesLibrary``, kept parse-time only (never packed or
        compiled) so element/source builders can resolve species-named compositions
        to feed streams and the ``Tt -> h_t`` datum.
    t_init, t_init_frozen : float
        Initial temperature guesses [K] for a deferred equilibrium config whose feed
        streams are discovered and packed at build time.
    """

    model_id: int  # gas-model selector
    tf: np.ndarray  # float64[::1] packed real data
    ti: np.ndarray  # int64[::1] packed sizes/indices
    element_names: List[str] = field(default_factory=list)  # transported-scalar labels
    species_names: List[str] = field(default_factory=list)  # library species names
    library: object = None  # thermolib SpeciesLibrary (parse-time only)
    t_init: float = 3000.0  # equilibrium temperature guess [K]
    t_init_frozen: float = 300.0  # frozen temperature guess [K]

    @property
    def n_elem(self) -> int:
        """Number of transported band-1 scalars (feed streams / passive scalars)."""
        return len(self.element_names)

    @property
    def n_species(self) -> int:
        """Number of species carried by the library (zero for a perfect gas)."""
        return len(self.species_names)


def perfect_gas(R: float = 287.0, gamma: float = 1.4) -> ThermoConfig:
    """Calorically-perfect-gas configuration (default: dry air).

    Parameters
    ----------
    R : float, optional
        Specific gas constant [J/(kg*K)] (default 287.0, dry air).
    gamma : float, optional
        Ratio of specific heats (default 1.4).

    Returns
    -------
    ThermoConfig
        Bundle with ``tf = [cp, R, W]`` where ``W = R_u / R`` is the molar mass
        [kg/mol] and ``cp = gamma R / (gamma - 1)``.
    """
    cp = gamma * R / (gamma - 1.0)
    W = R_UNIVERSAL / R  # molar mass [kg/mol]
    tf = np.ascontiguousarray([cp, R, W], dtype=np.float64)
    ti = np.empty(0, dtype=np.int64)
    return ThermoConfig(model_id=PERFECT_GAS, tf=tf, ti=ti)


def perfect_gas_passive_scalars(n_scalars: int, R: float = 287.0, gamma: float = 1.4, names=None) -> ThermoConfig:
    """Perfect gas that also advects ``n_scalars`` passive conserved scalars.

    The thermodynamics are unchanged (the perfect-gas kernels ignore ``Z_el``);
    each scalar simply adds one band-1 unknown and one source-free transport
    equation per edge, exercising the composition-transport framework without
    invoking chemistry.

    Parameters
    ----------
    n_scalars : int
        Number of passive conserved scalars to advect (must be >= 1).
    R : float, optional
        Specific gas constant [J/(kg*K)] (default 287.0).
    gamma : float, optional
        Ratio of specific heats (default 1.4).
    names : sequence of str, optional
        Scalar labels; defaults to ``scalar0, scalar1, ...``.

    Returns
    -------
    ThermoConfig
        Perfect-gas bundle carrying ``n_scalars`` band-1 scalars.
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
