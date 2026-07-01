"""Reacting-scalar bookkeeping.

The mean-flow solve transports **feed-stream mixture fractions** (one conserved
band-1 scalar per distinct injected composition), not chemical species.  This
package holds the two ends of that model:

* :mod:`~fns.chem.composition` -- parse-time descriptors that map species-named
  mixtures to the transported feed streams (and the forward blends that
  reconstruct unburnt speciation);
* :mod:`~fns.chem.chemistry`   -- post-solve recovery of the actual per-edge
  species composition from a converged state, for diagnostics / output.

Import submodules explicitly (``from fns.chem.composition import build_streams``);
this package ``__init__`` re-exports nothing.
"""
