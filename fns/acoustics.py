"""Back-compatibility shim: ``fns.acoustics`` is now :mod:`fns.perturbation`.

The acoustic layer was generalised into the full perturbation network (two
acoustic characteristics plus the entropy wave -- ``N = 3``).  This module
re-exports the new surface under the old name so existing top-level imports keep
working; new code should import from :mod:`fns.perturbation`.
"""

from .perturbation import *  # noqa: F401,F403
from .perturbation import __all__  # noqa: F401
