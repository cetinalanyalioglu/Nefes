"""Public-API surface guard for the :mod:`nefes.perturbation` package.

The perturbation layer is organized into four subpackages (``operator``,
``fields``, ``response``, ``stability``), but its public surface is the flat set
of names re-exported by ``nefes.perturbation.__all__``.  Most of that surface is
consumed by example notebooks, which the test suite does not execute -- so a
re-export regression (a name dropped from ``__all__``, or a subpackage import
reordered into a cycle) could otherwise slip through unnoticed.  These tests pin
the contract.
"""

import importlib

import pytest

import nefes.perturbation as P

SUBPACKAGES = ("operator", "fields", "response", "stability")


def test_all_names_resolve_on_package():
    """Every name advertised in ``__all__`` must be importable from the package."""
    missing = [name for name in P.__all__ if not hasattr(P, name)]
    assert not missing, f"names in __all__ not resolvable on nefes.perturbation: {missing}"


def test_all_is_deduplicated():
    """``__all__`` should carry no accidental duplicates from the merge of subpackages."""
    dupes = sorted({n for n in P.__all__ if P.__all__.count(n) > 1})
    assert not dupes, f"duplicate names in nefes.perturbation.__all__: {dupes}"


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackage_imports_clean(sub):
    """Each subpackage imports without triggering a circular import."""
    mod = importlib.import_module(f"nefes.perturbation.{sub}")
    assert mod.__name__ == f"nefes.perturbation.{sub}"


def test_star_import_exposes_all():
    """``from nefes.perturbation import *`` binds exactly the advertised surface."""
    ns: dict = {}
    exec("from nefes.perturbation import *", ns)
    exported = {k for k in ns if not k.startswith("__")}
    assert set(P.__all__) <= exported
