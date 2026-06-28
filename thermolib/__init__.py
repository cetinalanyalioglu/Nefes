"""thermolib -- a standalone, network-agnostic thermochemistry library.

Standalone chemical-equilibrium (HP) thermochemistry for arbitrary gaseous
mixtures.  No runtime dependency on Cantera or on any flow solver; Cantera is
used only as an *offline* importer and validation oracle (REQUIREMENTS O-3,
R-A1.1, R-A1.2).

Vocabulary: a :class:`SpeciesLibrary` is the thermochemical material database
(species + NASA polynomials -- all that equilibrium needs).  A :class:`Mechanism`
associates a species library with a set of :class:`Reaction` objects (kinetics).

Public API (REQUIREMENTS A.9)::

    from thermolib import SpeciesLibrary, Thermo
    lib   = SpeciesLibrary.from_native("h2o2.yaml")  # or ThermoInp("thermo.inp").library([...])
    gas   = Thermo(lib, backend="kernel")            # Backend D (native kernel)
    props = gas.properties(Y, T, p)                  # cp, h, s, rho, a_frozen, ...
    eq    = gas.equilibrate_HP(Z_elem, h, p)         # -> T, rho, Y, a_equilibrium
"""

from .constants import P_REF, P_REF_BAR, R_UNIVERSAL
from .species import NASA7, NASA9, Species, SpeciesLibrary, ThermoPoly
from .mechanism import Mechanism, Reaction
from .cea import ThermoInp, read_thermo_inp, default_thermo_inp
from .properties import MixtureState, mixture_properties
from .equilibrium import (
    EquilibriumResult,
    elemental_abundance,
    equilibrate_HP,
    equilibrate_TP,
)
from .thermo import Thermo
from . import smooth

__version__ = "0.2.0"

__all__ = [
    "SpeciesLibrary",
    "Species",
    "ThermoPoly",
    "NASA7",
    "NASA9",
    "Mechanism",
    "Reaction",
    "ThermoInp",
    "read_thermo_inp",
    "default_thermo_inp",
    "Thermo",
    "MixtureState",
    "mixture_properties",
    "EquilibriumResult",
    "equilibrate_HP",
    "equilibrate_TP",
    "elemental_abundance",
    "R_UNIVERSAL",
    "P_REF",
    "P_REF_BAR",
    "smooth",
    "__version__",
]
