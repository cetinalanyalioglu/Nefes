"""Selectable thermochemistry backends behind one uniform API (R-A7.1).

Two backends are envisaged (REQUIREMENTS D-3):

* **Backend D -- "kernel"**: the standalone native CEA-style kernel.  Built
  first; this is the MVP backend.  Differentiation mode: *complex-transparent*
  (R-A6.1 mode 1) -- it is callable with complex arguments directly, and the
  equilibrium solve propagates complex perturbations via a final IFT step.

* **Backend A -- "table"**: an offline tabulation + analytic surrogate
  (optional/later).  Structured here as a selectable option but intentionally
  not implemented in the MVP; selecting it raises a clear error.

The consumer selects a backend by name and uses the same ``Thermo`` API
regardless (R-A7.1, A.9).
"""

from __future__ import annotations

from . import equilibrium as _eq
from .properties import mixture_properties

__all__ = ["KernelBackend", "TableBackend", "make_backend", "DIFF_MODES"]

DIFF_MODES = {
    "complex-transparent": "callable with a complex argument directly",
    "sensitivity-providing": "returns values plus an analytic Jacobian block",
}


class KernelBackend:
    """Backend D: native element-potential kernel."""

    name = "kernel"
    diff_mode = "complex-transparent"  # R-A6.1 mode 1

    def __init__(self, lib):
        self.lib = lib

    def properties(self, Y, T, p):
        return mixture_properties(self.lib, Y, T, p)

    def equilibrate_HP(self, Z_elem, h, p, **kw):
        return _eq.equilibrate_HP(self.lib, Z_elem, h, p, **kw)

    def equilibrate_TP(self, Z_elem, T, p, **kw):
        return _eq.equilibrate_TP(self.lib, Z_elem, T, p, **kw)


class TableBackend:
    """Backend A: offline table + analytic surrogate (not built in the MVP)."""

    name = "table"
    diff_mode = "complex-transparent"

    def __init__(self, lib):
        raise NotImplementedError(
            "Backend A (table/surrogate) is an optional, later backend "
            "(REQUIREMENTS D-3 / phasing step 3). The MVP ships Backend D "
            "('kernel'). Use Thermo(lib, backend='kernel')."
        )


_BACKENDS = {
    "kernel": KernelBackend,  # Backend D
    "table": TableBackend,  # Backend A
}


def make_backend(name, lib):
    try:
        cls = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(_BACKENDS)}.") from exc
    return cls(lib)
