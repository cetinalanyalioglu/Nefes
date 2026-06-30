"""Network element catalog and @njit residual kernels."""

from .dynamic_source import (
    DynamicSource,
    FlameResponseTerm,
    TransferFunction,
    NTau,
    NTauLowpass,
    Tabulated,
    Constant,
    n_tau,
    n_tau_lowpass,
    tabulated,
    constant,
    as_transfer,
    heat_release_response,
    mass_flow_response,
    n_tau_flame,
)
from .continuation import RationalFit, rational_fit, continuation_warning
from .composite import CompositeElementSpec, CompositeMap, expand_composites, is_composite

__all__ = [
    "CompositeElementSpec",
    "CompositeMap",
    "expand_composites",
    "is_composite",
    "DynamicSource",
    "FlameResponseTerm",
    "TransferFunction",
    "NTau",
    "NTauLowpass",
    "Tabulated",
    "Constant",
    "n_tau",
    "n_tau_lowpass",
    "tabulated",
    "constant",
    "as_transfer",
    "heat_release_response",
    "mass_flow_response",
    "n_tau_flame",
    "RationalFit",
    "rational_fit",
    "continuation_warning",
]
