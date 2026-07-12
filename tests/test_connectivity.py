"""Connectivity CSR/CSC and the Jacobian sparsity pattern."""

import pytest

from nefes.graph.connectivity import (
    build_connectivity,
    build_jacobian_pattern,
    connectivity_from_directed_edges,
    pattern_to_csc,
)

# docs/examples/ConnectivityDemonstrator.yaml, in node-index space.
DEMO = [
    (0, 0, 1, 0),
    (1, 1, 2, 0),
    (1, 2, 3, 1),
    (2, 2, 3, 0),
    (2, 1, 4, 0),
    (3, 2, 4, 1),
    (4, 2, 5, 0),
]


def test_demonstrator_csc_table():
    c = build_connectivity(6, DEMO)
    assert list(c.tail_node) == [0, 1, 1, 2, 2, 3, 4]
    assert list(c.tail_port) == [0, 1, 2, 2, 1, 2, 2]
    assert list(c.head_node) == [1, 2, 3, 3, 4, 4, 5]
    assert list(c.head_port) == [0, 0, 1, 0, 0, 1, 0]


def test_demonstrator_csr_derivation():
    c = build_connectivity(6, DEMO)
    assert list(c.row_ptr) == [0, 1, 4, 7, 10, 13, 14]
    # node 1: ports 0,1,2 -> edges e0(in), e1(out), e2(out)
    sl = slice(c.row_ptr[1], c.row_ptr[2])
    assert list(c.col_edge[sl]) == [0, 1, 2]
    assert list(c.orient[sl]) == [-1, 1, 1]
    assert list(c.port[sl]) == [0, 1, 2]
    # handshake: sum of degrees == 2E
    assert int(c.row_ptr[-1]) == 2 * c.n_edges


def test_csr_csc_round_trip():
    c = build_connectivity(6, DEMO)
    # Rebuild the endpoint table from the CSR view and compare.
    tail = {}
    head = {}
    for n in range(c.n_nodes):
        for k in range(c.row_ptr[n], c.row_ptr[n + 1]):
            e, o, p = int(c.col_edge[k]), int(c.orient[k]), int(c.port[k])
            if o == 1:
                tail[e] = (n, p)
            else:
                head[e] = (n, p)
    for e in range(c.n_edges):
        assert tail[e] == (c.tail_node[e], c.tail_port[e])
        assert head[e] == (c.head_node[e], c.head_port[e])


def test_bad_ports_rejected():
    with pytest.raises(ValueError):
        build_connectivity(2, [(0, 0, 1, 0), (0, 0, 1, 1)])  # node 0 has two port-0


def test_auto_port_assignment():
    c = connectivity_from_directed_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert list(c.tail_node) == [0, 1, 2]
    assert list(c.head_node) == [1, 2, 3]
    # node 1 sees e0 (head, port 0) then e1 (tail, port 1)
    assert list(c.tail_port) == [0, 1, 1]
    assert list(c.head_port) == [0, 0, 0]


def test_jacobian_pattern_is_square_3E():
    c = build_connectivity(6, DEMO)
    n_eq = [c.degree(n) for n in range(c.n_nodes)]
    pat = build_jacobian_pattern(c, n_eq, n_solve=3)
    assert pat.n_eq == 3 * c.n_edges
    assert pat.n_col == 3 * c.n_edges
    assert pat.transport_row0 == 2 * c.n_edges


def test_node_rows_couple_to_incident_edges():
    c = build_connectivity(6, DEMO)
    n_eq = [c.degree(n) for n in range(c.n_nodes)]
    pat = build_jacobian_pattern(c, n_eq, n_solve=3)
    M = pattern_to_csc(pat).tocsr()
    for n in range(c.n_nodes):
        incident = set(int(e) for e in c.incident_edges(n))
        for r in range(int(pat.node_row_ptr[n]), int(pat.node_row_ptr[n + 1])):
            cols = M.indices[M.indptr[r] : M.indptr[r + 1]]
            edges_touched = set(int(col) // 3 for col in cols)
            assert edges_touched == incident


def test_transport_row_donor_coupling_is_wide():
    # 3-edge chain: middle edge's transport row must couple to ALL three edges
    # (donor enthalpies at both its endpoint nodes mix every incident edge).
    c = connectivity_from_directed_edges(4, [(0, 1), (1, 2), (2, 3)])
    n_eq = [c.degree(n) for n in range(c.n_nodes)]
    pat = build_jacobian_pattern(c, n_eq, n_solve=3)
    M = pattern_to_csc(pat).tocsr()
    r = pat.edge_transport_row(1)
    cols = M.indices[M.indptr[r] : M.indptr[r + 1]]
    edges_touched = set(int(col) // 3 for col in cols)
    assert edges_touched == {0, 1, 2}
    # a boundary edge's transport row is narrower (only edges at its two nodes)
    r0 = pat.edge_transport_row(0)
    cols0 = M.indices[M.indptr[r0] : M.indptr[r0 + 1]]
    assert set(int(col) // 3 for col in cols0) == {0, 1}


def test_pattern_columns_sorted_unique():
    c = build_connectivity(6, DEMO)
    n_eq = [c.degree(n) for n in range(c.n_nodes)]
    pat = build_jacobian_pattern(c, n_eq, n_solve=3)
    for col in range(pat.n_col):
        rows = pat.indices[pat.indptr[col] : pat.indptr[col + 1]]
        assert list(rows) == sorted(set(rows))
