"""Reacting-scalar bookkeeping.

The mean-flow solve transports **feed-stream mixture fractions** (one conserved
band-1 scalar per distinct injected composition), not chemical species.  This
package holds the two ends of that model:

* :mod:`~nefes.chem.composition` -- parse-time descriptors that map species-named
  mixtures to the transported feed streams (and the forward blends that
  reconstruct unburnt speciation);
* :mod:`~nefes.chem.chemistry`   -- post-solve recovery of the actual per-edge
  species composition from a converged state, for diagnostics / output.

The user-facing composition helpers are re-exported here, so a reacting setup needs
no deeper import::

    from nefes.chem import equivalence_ratio_mixture, resolve_composition, enthalpy_mass
"""

from .composition import (
    elemental_Z,
    enthalpy_mass,
    equivalence_ratio_mixture,
    resolve_composition,
    species_mass_fractions,
    species_mole_fractions,
)

__all__ = [
    "species_mass_fractions",
    "species_mole_fractions",
    "elemental_Z",
    "enthalpy_mass",
    "resolve_composition",
    "equivalence_ratio_mixture",
]
