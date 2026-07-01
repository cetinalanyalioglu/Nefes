"""Residual and Jacobian evaluation kernel (below the solver control layer).

The complex-step-safe core that turns a :class:`~nefes.graph.problem.CompiledProblem`
and a solution vector into residuals and their complex-step Jacobian:

* :mod:`~nefes.assembly.smooth`   -- C-infinity, complex-step-safe primitives (a leaf
  imported across ``thermo``/``elements`` as well);
* :mod:`~nefes.assembly.closure`  -- the AD-3 thermo boundary ``(mdot, p, h_t) -> (rho, h)``;
* :mod:`~nefes.assembly.derive`   -- the edge-state recovery DAG (the ``ES_*`` slot layout);
* :mod:`~nefes.assembly.assemble` -- the node/edge residual and sparse Jacobian assembly;
* :mod:`~nefes.assembly.scaling`  -- residual / variable nondimensionalization.

This ``__init__`` deliberately re-exports **nothing** and imports no submodule, so
that a leaf like :mod:`~nefes.assembly.smooth` can be pulled in by ``thermo`` and
``elements`` without dragging in the ``thermo``-dependent submodules and forming a
package-level import cycle.  Import submodules explicitly
(``from nefes.assembly.derive import recover_edge``).
"""
