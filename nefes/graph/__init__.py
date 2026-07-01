"""Compiled network representation.

Holds the two views of the network's static structure that the solver and the
perturbation layer read: :mod:`~nefes.graph.connectivity` (the CSR node-row / CSC
edge-column views plus the Jacobian block-sparsity pattern) and
:mod:`~nefes.graph.problem` (the immutable, solve-time :class:`CompiledProblem`
bundle built once at parse time).

Both submodules are pure data structures with no intra-``nefes`` dependencies, so
importing them is side-effect-free.  Import submodules explicitly
(``from nefes.graph.connectivity import ...``); this package ``__init__`` re-exports
nothing to keep import order trivial.
"""
