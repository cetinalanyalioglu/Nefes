"""Network connectivity: CSR (node->edges) and CSC (edge->endpoints) views.

State lives on edges; conservation equations are assembled at nodes.  Both
assembly directions read one of two views of the same sparsity pattern:

  * **CSR node-row view** (``row_ptr, col_edge, orient, port``): node n -> its
    incident edges, one contiguous run.
  * **CSC edge-column view** (``tail_node, tail_port, head_node, head_port``,
    length E): edge e -> its two endpoint nodes/ports in O(1).

Block-expanded, that pattern is also the Jacobian's block-sparsity.  An edge's
scalar-transport row couples wider than its two endpoint node-rows: each donor
scalar is a mass-weighted upwind mix over all edges incident to an endpoint
node, so the row depends on every edge sharing a node with e.
``build_jacobian_pattern`` encodes exactly that.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Connectivity:
    """Flat, njit-compatible connectivity arrays (all indices 0-based)."""

    n_nodes: int
    n_edges: int
    # CSR node-row view
    row_ptr: np.ndarray  # int64[N+1]
    col_edge: np.ndarray  # int64[nnz]
    orient: np.ndarray  # int8[nnz]   (+1 tail/outgoing, -1 head/incoming)
    port: np.ndarray  # int64[nnz]
    # CSC edge-column view
    tail_node: np.ndarray  # int64[E]
    head_node: np.ndarray  # int64[E]
    tail_port: np.ndarray  # int64[E]
    head_port: np.ndarray  # int64[E]

    def degree(self, n: int) -> int:
        return int(self.row_ptr[n + 1] - self.row_ptr[n])

    def incident_edges(self, n: int) -> np.ndarray:
        """Global edge indices attached to node n (in port order)."""
        return self.col_edge[self.row_ptr[n] : self.row_ptr[n + 1]]


def build_connectivity(n_nodes: int, endpoints) -> Connectivity:
    """Build connectivity from an explicit edge-endpoint table.

    ``endpoints`` is an iterable of ``(tail_node, tail_port, head_node,
    head_port)`` rows, one per edge in edge-index order (the CSC view).
    The CSR node-row view is derived from it.
    """
    rows = [tuple(int(v) for v in row) for row in endpoints]
    E = len(rows)
    tail_node = np.array([r[0] for r in rows], dtype=np.int64)
    tail_port = np.array([r[1] for r in rows], dtype=np.int64)
    head_node = np.array([r[2] for r in rows], dtype=np.int64)
    head_port = np.array([r[3] for r in rows], dtype=np.int64)

    # Collect incidences per node: (port, edge, orient).
    per_node = [[] for _ in range(n_nodes)]
    for e, (tn, tp, hn, hp) in enumerate(rows):
        per_node[tn].append((tp, e, 1))
        per_node[hn].append((hp, e, -1))

    row_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    col_edge, orient, port = [], [], []
    for n, inc in enumerate(per_node):
        inc.sort(key=lambda t: t[0])  # by local port
        ports_here = [t[0] for t in inc]
        if ports_here != list(range(len(inc))):
            raise ValueError(f"node {n} ports {ports_here} are not a 0..d-1 permutation")
        row_ptr[n + 1] = row_ptr[n] + len(inc)
        for p, e, o in inc:
            col_edge.append(e)
            orient.append(o)
            port.append(p)

    return Connectivity(
        n_nodes=n_nodes,
        n_edges=E,
        row_ptr=row_ptr,
        col_edge=np.array(col_edge, dtype=np.int64),
        orient=np.array(orient, dtype=np.int8),
        port=np.array(port, dtype=np.int64),
        tail_node=tail_node,
        head_node=head_node,
        tail_port=tail_port,
        head_port=head_port,
    )


def connectivity_from_directed_edges(n_nodes: int, edges) -> Connectivity:
    """Build connectivity from ``(tail_node, head_node)`` pairs.

    Local ports are auto-assigned in attachment order (the order edges touch a
    node, scanning edges by index) -- the convention the programmatic Network
    builder uses.
    """
    next_port = [0] * n_nodes
    endpoints = []
    for tn, hn in edges:
        tp = next_port[tn]
        next_port[tn] += 1
        hp = next_port[hn]
        next_port[hn] += 1
        endpoints.append((tn, tp, hn, hp))
    return build_connectivity(n_nodes, endpoints)


@dataclass(frozen=True)
class JacobianPattern:
    """Fixed CSC sparsity pattern of the global Jacobian, plus row layout."""

    n_eq: int  # NEQ = sum(n_eq_per_node) + n_edges
    n_col: int  # n_solve * n_edges
    n_solve: int
    indptr: np.ndarray  # int64[n_col + 1]   (CSC)
    indices: np.ndarray  # int64[nnz]         (sorted rows per column)
    node_row_ptr: np.ndarray  # int64[N+1]    node n -> its equation row block
    transport_row0: int  # first edge-transport row (= sum of node eqs)

    def edge_transport_row(self, e: int) -> int:
        return self.transport_row0 + e

    def column(self, e: int, v: int) -> int:
        return self.n_solve * e + v


def build_jacobian_pattern(conn: Connectivity, n_eq_per_node, n_solve: int) -> JacobianPattern:
    """Structural CSC pattern of the Jacobian (a superset of its true nonzeros).

    Dependency rules:
      * node n's equation rows depend on the band-1 columns of every edge
        incident to n;
      * edge e's transport row depends on the columns of e itself and of every
        edge incident to ``tail_node(e)`` or ``head_node(e)`` (the donor mix).
    """
    n_eq_per_node = np.asarray(n_eq_per_node, dtype=np.int64)
    N, E = conn.n_nodes, conn.n_edges
    # advected scalars carried per edge: h_t (always) + composition Z_el (n_solve - 3)
    n_scalars = n_solve - 2

    node_row_ptr = np.zeros(N + 1, dtype=np.int64)
    node_row_ptr[1:] = np.cumsum(n_eq_per_node)
    transport_row0 = int(node_row_ptr[-1])
    n_eq = transport_row0 + n_scalars * E
    n_col = n_solve * E

    rows_of_col = [set() for _ in range(n_col)]

    def add_block(node_rows, edge):
        for v in range(n_solve):
            c = n_solve * edge + v
            rows_of_col[c].update(node_rows)

    # Node-equation rows vs. incident edges.
    for n in range(N):
        r0, r1 = int(node_row_ptr[n]), int(node_row_ptr[n + 1])
        node_rows = range(r0, r1)
        for e in conn.incident_edges(n):
            add_block(node_rows, int(e))

    # Edge-transport rows vs. e and the edges sharing either endpoint node.
    # Each advected scalar s gets its own transport row per edge.
    for e in range(E):
        neigh = set([e])
        for nd in (int(conn.tail_node[e]), int(conn.head_node[e])):
            neigh.update(int(x) for x in conn.incident_edges(nd))
        for s in range(n_scalars):
            trow = transport_row0 + s * E + e
            for e2 in neigh:
                for v in range(n_solve):
                    rows_of_col[n_solve * e2 + v].add(trow)

    indptr = np.zeros(n_col + 1, dtype=np.int64)
    indices = []
    for c in range(n_col):
        rs = sorted(rows_of_col[c])
        indices.extend(rs)
        indptr[c + 1] = indptr[c] + len(rs)

    return JacobianPattern(
        n_eq=n_eq,
        n_col=n_col,
        n_solve=n_solve,
        indptr=indptr,
        indices=np.array(indices, dtype=np.int64),
        node_row_ptr=node_row_ptr,
        transport_row0=transport_row0,
    )


def pattern_to_csc(pat: JacobianPattern) -> sp.csc_matrix:
    """Materialize the pattern as a boolean scipy CSC matrix (for inspection)."""
    data = np.ones(len(pat.indices), dtype=bool)
    return sp.csc_matrix((data, pat.indices, pat.indptr), shape=(pat.n_eq, pat.n_col))
