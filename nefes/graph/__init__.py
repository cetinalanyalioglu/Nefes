"""Network connectivity structure.

Holds the network's static structure that the solver and the perturbation layer
read: :mod:`~nefes.graph.connectivity` -- the CSR node-row / CSC edge-column
views plus the Jacobian block-sparsity pattern.

It is a pure data structure with no intra-``nefes`` dependencies, so importing it
is side-effect-free.  Import it explicitly
(``from nefes.graph.connectivity import ...``); this package ``__init__`` re-exports
nothing to keep import order trivial.
"""
