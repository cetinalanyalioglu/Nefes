"""Network element catalog and @njit residual kernels."""

from .dynamic_source import (
    DynamicSource,
    FlameResponseTerm,
    TransferFunction,
    NTau,
    Tabulated,
    Constant,
    n_tau,
    tabulated,
    constant,
    as_transfer,
    heat_release_response,
    mass_flow_response,
    n_tau_flame,
)

__all__ = [
    "DynamicSource",
    "FlameResponseTerm",
    "TransferFunction",
    "NTau",
    "Tabulated",
    "Constant",
    "n_tau",
    "tabulated",
    "constant",
    "as_transfer",
    "heat_release_response",
    "mass_flow_response",
    "n_tau_flame",
]
