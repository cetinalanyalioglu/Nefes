"""Lazy Plotly access for the plotting subpackage.

The core solver never draws, so Plotly is not a base dependency; it ships in the ``viz``
extra (``pip install nefes[viz]``).  Importing :mod:`nefes.plotting` must therefore succeed
even when Plotly is absent, with the missing-dependency error deferred to the moment a figure
is actually built.  Each plotting module binds its Plotly names to the proxies here instead of
importing ``plotly`` directly:

    from ._deps import go, pio, make_subplots   # module-level, import-safe

``go``/``pio`` behave like the real submodules on attribute access, and ``make_subplots`` /
``sample_colorscale`` like the real callables; the underlying import happens on first use and,
if Plotly is missing, raises a :class:`ModuleNotFoundError` pointing at the ``viz`` extra.

This module exports the proxies :data:`go`, :data:`pio`, :data:`make_subplots`, and
:data:`sample_colorscale`.
"""

import importlib
from typing import Any

_MISSING_MESSAGE = (
    "nefes.plotting requires Plotly, which is not installed. "
    "Install it with `pip install nefes[viz]` (or use the conda environment)."
)


def _import(module_name: str) -> Any:
    """Import ``module_name``, re-raising a missing Plotly as a hint to install the extra."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        #  Only rewrite the error when Plotly itself is missing; a genuine import bug elsewhere
        #  (a broken Plotly install failing on a submodule) should surface unchanged.
        if exc.name == "plotly" or (exc.name or "").startswith("plotly."):
            raise ModuleNotFoundError(_MISSING_MESSAGE) from exc
        raise


class _LazyModule:
    """Attribute proxy for a Plotly submodule, imported on first attribute access."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Any = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            #  Bypass __getattr__ recursion by writing straight to the instance dict.
            object.__setattr__(self, "_module", _import(self._module_name))
        return getattr(self._module, attr)


class _LazyCallable:
    """Call proxy for a Plotly factory, imported on first call."""

    def __init__(self, module_name: str, attr: str) -> None:
        self._module_name = module_name
        self._attr = attr

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(_import(self._module_name), self._attr)(*args, **kwargs)


go = _LazyModule("plotly.graph_objects")
pio = _LazyModule("plotly.io")
make_subplots = _LazyCallable("plotly.subplots", "make_subplots")
sample_colorscale = _LazyCallable("plotly.colors", "sample_colorscale")
