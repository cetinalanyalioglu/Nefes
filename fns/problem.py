"""The immutable, solve-time problem bundle handed to the assembly kernels.

``CompiledProblem`` is a flat struct-of-arrays: connectivity, element dispatch
ids, packed node parameters, edge areas, the field-layout scalars, the Jacobian
sparsity pattern, and the nondimensionalization scales.  Nothing here is solver
state; it is built once at parse time and threaded read-only.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledProblem:
    # thermo
    model_id: int
    tf: np.ndarray
    ti: np.ndarray
    n_elem: int
    # sizes
    n_solve: int
    n_nodes: int
    n_edges: int
    n_eq: int
    # geometry / connectivity
    area: np.ndarray  # float64[E]
    row_ptr: np.ndarray
    col_edge: np.ndarray
    orient: np.ndarray
    tail_node: np.ndarray
    head_node: np.ndarray
    # element dispatch + params
    node_rid: np.ndarray  # int64[N]
    node_acoustic_id: np.ndarray  # int64[N] -- acoustic-face dispatch (s8.3)
    npar_f: np.ndarray  # float64[...]
    npar_fptr: np.ndarray  # int64[N+1]
    # equation row layout
    node_row_ptr: np.ndarray  # int64[N+1]
    transport_row0: int
    # Jacobian CSC pattern
    indptr: np.ndarray
    indices: np.ndarray
    # nondimensionalization (filled by the solver layer; may be all-ones)
    var_scale: np.ndarray  # float64[n_solve]
    res_scale: np.ndarray  # float64[n_eq]
    # per-edge thermo model id (int64[E]); selects the closure/equilibrium kernel
    # edge-by-edge so a frozen approach edge and an equilibrium edge can coexist.
    # Defaults to the global ``model_id`` on every edge.
    edge_model: np.ndarray = None  # int64[E]
    # per-element smoothing-eps override (< 0 -> use the global eps); see ElementSpec.eps
    node_eps: np.ndarray = None  # float64[N]
    # per-node perturbation boundary condition (Python objects, read only above the
    # @njit line by the perturbation layer); None / "inherit" where unset.  See
    # fns.perturbation.boundary_bc.PerturbationBC.
    node_bc: tuple = ()  # length N (or empty -> all inherit)
    # human-readable element name per node (the YAML/UI label); for labelling only,
    # never touched by the kernels.  length N, or empty -> nameless.
    node_names: tuple = ()
    # label of each transported composition scalar (the feed-stream / mixture-fraction
    # names for the reacting model, or the passive-scalar names); length n_elem, for
    # reporting only.  Empty when there are no composition scalars.
    scalar_names: tuple = ()
    # per-node dynamic-source descriptor (Python objects; fns.elements.dynamic_source
    # .DynamicSource).  Carried for the later S(omega) perturbation phase; the mean
    # flow ignores it (a constant mean source is acoustically passive).  length N,
    # or empty -> none.
    node_dynamic_source: tuple = ()
    # band-1 row of the transported burnt marker (the last advected scalar, row 3 + n_elem),
    # or -1 when the network carries none (perfect gas, or a hard-closure reacting network).
    # Marker-gated reacting networks (auto closure with a flame) use EQ_MARKER on every edge
    # and read this row to blend frozen/equilibrium.
    marker_row: int = -1
    # per-edge initial marker (0 fresh / 1 burnt) from the compile-time flood-fill, demoted to
    # the marker transport's initial guess; None when the network carries no marker.
    marker_seed: np.ndarray = None  # float64[E]
    # composite-element expansion map (fns.elements.composite.CompositeMap), or None when the
    # network carries no composite.  Bridges the user-facing (Case) node/edge ids to the expanded
    # ids the kernels solve on; lets diagnostics hide or project composite internals.
    composite_map: object = None

    @property
    def n_col(self) -> int:
        return self.n_solve * self.n_edges
