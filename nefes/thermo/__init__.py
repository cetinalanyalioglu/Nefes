"""Thermochemistry and the pluggable gas-model boundary.

This subpackage holds both the standalone thermochemistry (NASA-Glenn / CEA species
data, the element-potential equilibrium kernel, and mixture properties) and the
solver-facing gas-model boundary (edge-state producers and the integer-tagged
``thermo_update`` dispatch).

Thermochemistry entry points::

    from nefes.thermo import SpeciesLibrary, Thermo
    lib   = SpeciesLibrary.from_cantera("h2o2.yaml")  # or ThermoInp("thermo.inp").library([...])
    gas   = Thermo(lib)
    props = gas.properties(Y, T, p)                   # cp, h, s, rho, a_frozen, ...
    eq    = gas.equilibrate_HP(Z_elem, h, p)          # -> T, rho, Y, a_equilibrium
"""

from . import smooth

# -- solver-facing gas-model boundary -----------------------------------------
from .api import (
    C_OUT,
    EQ_FROZEN,
    EQ_KERNEL,
    EQ_MARKER,
    EQ_TABLE,
    MODE_RATES,
    MODE_SPECIES,
    MODE_STATE,
    N_THERMO_OUT,
    PERFECT_GAS,
    RHO_OUT,
    T_OUT,
    W_OUT,
    thermo_update,
)
from .cea import ThermoInp, default_thermo_inp, read_thermo_inp
from .configure import ThermoConfig, equilibrium, perfect_gas

# -- thermochemistry: data layer ----------------------------------------------
from .constants import P_REF, P_REF_BAR, R_UNIVERSAL
from .equilibrate import (
    EquilibriumResult,
    elemental_abundance,
    equilibrate_HP,
    equilibrate_TP,
)
from .facade import Thermo
from .mechanism import Mechanism, Reaction
from .perfect_gas import pg_solve_density, pg_update

# -- thermochemistry: compute layer -------------------------------------------
from .properties import MixtureState, mixture_properties
from .reduction import (
    EquilibriumSamplingReducer,
    NullReducer,
    ReductionResult,
    SampleState,
    SpeciesReducer,
    available_reducers,
    get_reducer,
    register_reducer,
)
from .species import NASA7, NASA9, Species, SpeciesLibrary, ThermoPoly

__all__ = [
    # data layer
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
    "R_UNIVERSAL",
    "P_REF",
    "P_REF_BAR",
    "smooth",
    "SampleState",
    "ReductionResult",
    "SpeciesReducer",
    "NullReducer",
    "EquilibriumSamplingReducer",
    "get_reducer",
    "register_reducer",
    "available_reducers",
    # compute layer
    "MixtureState",
    "mixture_properties",
    "EquilibriumResult",
    "elemental_abundance",
    "equilibrate_HP",
    "equilibrate_TP",
    "Thermo",
    # solver-facing boundary
    "thermo_update",
    "PERFECT_GAS",
    "EQ_KERNEL",
    "EQ_FROZEN",
    "EQ_MARKER",
    "EQ_TABLE",
    "MODE_STATE",
    "MODE_SPECIES",
    "MODE_RATES",
    "T_OUT",
    "RHO_OUT",
    "C_OUT",
    "W_OUT",
    "N_THERMO_OUT",
    "ThermoConfig",
    "perfect_gas",
    "equilibrium",
    "pg_solve_density",
    "pg_update",
]
