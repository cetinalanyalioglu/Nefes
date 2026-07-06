"""Selectable thermochemistry backends behind one uniform API.

Two backends are envisaged:

* **kernel**: the standalone built-in CEA-style equilibrium kernel. Its differentiation mode
  is *complex-transparent*: it is callable with complex arguments directly, and the
  equilibrium solve propagates complex perturbations via a final implicit-function step.

* **table**: an offline tabulation plus analytic surrogate. Structured here as a
  selectable option but not implemented; selecting it raises a clear error.

The consumer selects a backend by name and uses the same ``Thermo`` API regardless.

Public: :class:`KernelBackend`, :class:`TableBackend`, :func:`make_backend`.
"""

from __future__ import annotations

from . import equilibrate as _eq
from .properties import mixture_properties

__all__ = ["KernelBackend", "TableBackend", "make_backend", "DIFF_MODES"]

DIFF_MODES = {
    "complex-transparent": "callable with a complex argument directly",
    "sensitivity-providing": "returns values plus an analytic Jacobian block",
}


class KernelBackend:
    """Native element-potential equilibrium kernel."""

    name = "kernel"
    diff_mode = "complex-transparent"

    def __init__(self, lib):
        self.lib = lib

    def properties(self, Y, T, p):
        return mixture_properties(self.lib, Y, T, p)

    def equilibrate_HP(self, Z_elem, h, p, **kw):
        return _eq.equilibrate_HP(self.lib, Z_elem, h, p, **kw)

    def equilibrate_TP(self, Z_elem, T, p, **kw):
        return _eq.equilibrate_TP(self.lib, Z_elem, T, p, **kw)


class TableBackend:
    """Offline table plus analytic surrogate (not implemented)."""

    name = "table"
    diff_mode = "complex-transparent"

    def __init__(self, lib):
        raise NotImplementedError(
            "the 'table' backend (offline table/surrogate) is not implemented; use Thermo(lib, backend='kernel')."
        )


_BACKENDS = {
    "kernel": KernelBackend,
    "table": TableBackend,
}


def make_backend(name, lib):
    try:
        cls = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(_BACKENDS)}.") from exc
    return cls(lib)
