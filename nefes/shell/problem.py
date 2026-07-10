"""The immutable, solve-time problem bundle handed to the assembly kernels.

``CompiledProblem`` is where the high-level network definition is cast into the
flat, low-level form the kernels actually solve on -- a compile step hidden from
the user.  It is a struct-of-arrays of connectivity, element dispatch ids,
packed node parameters, edge areas, field-layout scalars, the Jacobian sparsity
pattern, and the nondimensionalization scales.  It holds no solver state: built
once at parse time, then threaded read-only.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledProblem:
    # thermo
    model_id: int  # global thermo/closure model id (per-edge override via edge_model)
    tf: np.ndarray  # thermo float parameters (closure constants)
    ti: np.ndarray  # thermo integer parameters
    n_elem: int  # number of composition scalars carried per edge (0 for perfect gas)
    # sizes
    n_solve: int  # solved variables per edge (mdot, p, h_t, then n_elem composition scalars)
    n_nodes: int  # element count
    n_edges: int  # port-connection (edge) count
    n_eq: int  # total residual rows: node equations + edge transport rows
    # geometry / connectivity
    area: np.ndarray  # float64[E]
    row_ptr: np.ndarray
    col_edge: np.ndarray
    orient: np.ndarray
    tail_node: np.ndarray
    head_node: np.ndarray
    # element dispatch + params
    node_rid: np.ndarray  # int64[N]
    node_acoustic_stamp: np.ndarray  # int64[N] -- acoustic-stamp dispatch
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
    # nefes.perturbation.operator.boundary_bc.PerturbationBC.
    node_bc: tuple = ()  # length N (or empty -> all inherit)
    # human-readable element name per node (the YAML/UI label); for labelling only,
    # never touched by the kernels.  length N, or empty -> nameless.
    node_names: tuple = ()
    # label of each transported composition scalar (the feed-stream / mixture-fraction
    # names for the reacting model, or the passive-scalar names); length n_elem, for
    # reporting only.  Empty when there are no composition scalars.
    scalar_names: tuple = ()
    # per-node dynamic-source descriptor (Python objects; nefes.elements.dynamic_source
    # .DynamicSource).  Carried for the later S(omega) perturbation phase; the mean
    # flow ignores it (a constant mean source is acoustically passive).  length N,
    # or empty -> none.
    node_dynamic_source: tuple = ()
    # per-node transfer-matrix descriptor (Python objects; nefes.perturbation.matrix
    # .TransferMatrix, or an identify.UnknownTransferMatrix marker).  Carried on a
    # TRANSFER_MATRIX element for the perturbation stamp; the mean flow (an isentropic
    # area change) ignores it.  length N, or empty -> none.
    node_transfer_matrix: tuple = ()
    # band-1 row of the transported burnt marker (the last advected scalar, row 3 + n_elem),
    # or -1 when the network carries none (perfect gas, or a hard-closure reacting network).
    # Marker-gated reacting networks (auto closure with a flame) use EQ_MARKER on every edge
    # and read this row to blend frozen/equilibrium.
    marker_row: int = -1
    # per-edge initial marker (0 fresh / 1 burnt) from the compile-time flood-fill, demoted to
    # the marker transport's initial guess; None when the network carries no marker.
    marker_seed: np.ndarray = None  # float64[E]
    # composite-element expansion map (nefes.elements.composite.CompositeMap), or None when the
    # network carries no composite.  Bridges the user-facing (Case) node/edge ids to the expanded
    # ids the kernels solve on; lets diagnostics hide or project composite internals.
    composite_map: object = None

    @property
    def n_col(self) -> int:
        return self.n_solve * self.n_edges
