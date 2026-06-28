"""Spatially-resolved perturbation fields inside the ducts (theory.md s12.3).

A ``DUCT`` is the only length-bearing element, and it is **uniform and lossless**:
its two faces share one mean state ``(c, u)``, so the perturbation field *inside* it
is known in closed form -- the very phase relations the duct stamp imposes
(:func:`fns.perturbation.stamps.stamp_propagation`), now evaluated at every interior
station instead of only at the head face::

    f(s) = f_tail * exp(-i w s / (u + c))          downstream wave, from the tail
    g(s) = g_head * exp(-i w (L - s) / (c - u))     upstream wave, from the head
    h(s) = h_tail * exp(-i w s / u)                 entropy, convected from the tail

with ``s`` the axial distance from the tail face (``s in [0, L]``).  Because the
wave amplitudes ``(f, g, h)`` at every edge are already produced for any mode
(:meth:`EigenmodeResult.mode_waves`) or forced field
(:meth:`PerturbationResponse._waves`), the continuous mode shape is *exact*
post-processing -- no discretization, no extra solve, no change to the operator.

This module turns those face amplitudes into a **developed-length** field along the
network: the compact (zero-length) elements are points where the field may jump,
the ducts are the continua between them.  A serial network reduces to a single
trace; a branched one is decomposed into root->leaf paths (one per terminal), so
the shared trunk overlaps and the branches fan out.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .characteristics import basis_block_from_state, BASIS_LABELS
from .verify import duct_nodes
from ..derive import ES_C, ES_U
from ..elements.ids import MASS_FLOW_INLET, PT_INLET

# Below this |speed| a duct is treated as quiescent: the entropy wave does not
# convect (tau_0 -> inf), so it carries no interior spatial structure.
_U_FLOOR = 1e-8

# Map a friendly variable name to (flavor, component, LaTeX label); see
# characteristics.BASIS_LABELS for the flavor component order.
VARIABLE_SPEC = {
    "p": ("network", 1, r"p'"),
    "u": ("primitive", 1, r"u'"),
    "rho": ("pu_rho", 2, r"\rho'"),
    "mdot": ("network", 0, r"\dot{m}'"),
    "f": ("char", 0, r"f"),
    "g": ("char", 1, r"g"),
    "h": ("char", 2, r"h"),
}


def resolve_specs(variable=None, basis=None):
    """Resolve a variable/basis request into reconstruction specs.

    Two ways to ask for one or more quantities to reconstruct:

    * ``basis`` -- a flavor from :data:`fns.perturbation.characteristics.BASIS_LABELS`
      (``"char"``, ``"primitive"``, ``"network"``, ...).  Expands to that flavor's three
      components, labelled by ``BASIS_LABELS``.
    * ``variable`` -- a single :data:`VARIABLE_SPEC` name or a sequence of them, resolved
      one-to-one (a friendly mix across flavors, e.g. ``["p", "u"]``).

    Parameters
    ----------
    variable : str or sequence of str, optional
        Variable name(s) from :data:`VARIABLE_SPEC`.  Ignored when ``basis`` is given.
    basis : str, optional
        A flavor name; overrides ``variable`` and expands to its three components.

    Returns
    -------
    list of tuple
        ``(label, basis_flavor, component)`` triples, one per requested quantity, where
        ``label`` is the LaTeX fragment and ``(basis_flavor, component)`` feeds
        :func:`reconstruct_field` via its ``spec`` argument.

    Raises
    ------
    ValueError
        If ``basis`` or any ``variable`` name is unknown.
    """
    if basis is not None:
        if basis not in BASIS_LABELS:
            raise ValueError(f"unknown basis {basis!r}; choose from {sorted(BASIS_LABELS)}")
        labels = BASIS_LABELS[basis]
        return [(labels[c], basis, c) for c in range(len(labels))]
    names = [variable] if isinstance(variable, str) else list(variable)
    specs = []
    for name in names:
        if name not in VARIABLE_SPEC:
            raise ValueError(f"unknown variable {name!r}; choose from {sorted(VARIABLE_SPEC)}")
        flavor, component, label = VARIABLE_SPEC[name]
        specs.append((label, flavor, component))
    return specs


@dataclass(frozen=True)
class DuctSegment:
    """A length-bearing duct: its node id, the two face edges, and its length."""

    node: int  # the DUCT element id
    e_tail: int  # tail-face edge (port 0, flow enters here)
    e_head: int  # head-face edge (port 1, flow leaves here)
    length: float  # duct length L [m]


@dataclass(frozen=True)
class NetworkGeometry:
    """Topology + duct lengths needed to lay a perturbation field out in space.

    A light, picklable projection of the compiled problem (no solver state): the
    duct segments, the per-edge endpoint nodes, and per-node dispatch ids/labels.
    Attached to :class:`~fns.perturbation.EigenmodeResult` and
    :class:`~fns.perturbation.PerturbationResponse` so they can reconstruct the
    spatial mode shape without holding the whole ``CompiledProblem``.
    """

    ducts: Tuple[DuctSegment, ...]
    tail_node: np.ndarray  # int[E] -- directed edge tail element
    head_node: np.ndarray  # int[E] -- directed edge head element
    node_rid: np.ndarray  # int[N] -- mean-flow element id (for inlet detection)
    node_names: Tuple[str, ...]  # per-node label ("" where unset)
    n_nodes: int
    n_edges: int

    def duct_map(self):
        """``{node id: DuctSegment}`` for the length-bearing elements."""
        return {d.node: d for d in self.ducts}

    def __repr__(self) -> str:
        """One-line summary: node/edge/duct counts and the total developed length."""
        total = sum(d.length for d in self.ducts)
        return (
            f"NetworkGeometry: {self.n_nodes} nodes, {self.n_edges} edges, "
            f"{len(self.ducts)} duct{'' if len(self.ducts) == 1 else 's'} ({total:.4g} m total length)"
        )


def build_geometry(prob) -> NetworkGeometry:
    """Extract a :class:`NetworkGeometry` from a compiled problem.

    Parameters
    ----------
    prob : CompiledProblem
        The compiled flow network.

    Returns
    -------
    NetworkGeometry
    """
    ducts = []
    for n in duct_nodes(prob):
        base = int(prob.row_ptr[n])
        #  verify_acoustic pins orient == (-1, +1): port 0 is the tail face (flow in),
        #  port 1 the head face (flow out) -- the same e0/e1 the duct stamp uses.
        e_tail = int(prob.col_edge[base])
        e_head = int(prob.col_edge[base + 1])
        length = float(prob.npar_f[int(prob.npar_fptr[n])])
        ducts.append(DuctSegment(node=n, e_tail=e_tail, e_head=e_head, length=length))
    return NetworkGeometry(
        ducts=tuple(ducts),
        tail_node=np.asarray(prob.tail_node, dtype=np.int64).copy(),
        head_node=np.asarray(prob.head_node, dtype=np.int64).copy(),
        node_rid=np.asarray(prob.node_rid, dtype=np.int64).copy(),
        node_names=tuple(prob.node_names or ()),
        n_nodes=int(prob.n_nodes),
        n_edges=int(prob.n_edges),
    )


@dataclass
class PathField:
    """One root->leaf path's spatial field, ready to plot.

    Attributes
    ----------
    name : str
        Path label (its endpoint terminal names).
    x : ndarray
        Developed length [m] along the path (monotone non-decreasing), shape ``(n,)``.
    values : ndarray
        Complex value of the chosen variable at each station, shape ``(n,)``.
    markers : list of tuple
        ``(x, label)`` for each compact element on the path (jumps / terminals).
    """

    name: str
    x: np.ndarray
    values: np.ndarray
    markers: List[Tuple[float, str]]

    def __repr__(self) -> str:
        """One-line summary: path name, sample/marker counts, and the developed-length span."""
        x = np.asarray(self.x, dtype=float)
        span = f"0 to {x.max():.4g} m" if x.size else "empty"
        return (
            f"PathField {self.name!r}: {x.size} samples over {span}, "
            f"{len(self.markers)} marker{'' if len(self.markers) == 1 else 's'}"
        )


def _duct_chars(chars_tail, chars_head, c, u, omega, length, n_x):
    """Interior characteristic amplitudes ``(f, g, h)`` along a duct.

    ``chars_tail``/``chars_head`` are the ``(f, g, h)`` at the two faces; the
    downstream wave ``f`` and entropy ``h`` propagate from the tail, the upstream
    wave ``g`` from the head (theory.md s12.3).

    Returns ``(s, chars)`` with ``s`` shape ``(n_x,)`` and ``chars`` ``(n_x, 3)``.
    """
    s = np.linspace(0.0, length, n_x)
    f0, _g0, h0 = chars_tail
    _f1, g1, _h1 = chars_head
    f = f0 * np.exp(-1j * (omega / (u + c)) * s)
    g = g1 * np.exp(-1j * (omega / (c - u)) * (length - s))
    if abs(u) > _U_FLOOR:
        h = h0 * np.exp(-1j * (omega / u) * s)
    else:
        #  quiescent: the entropy spot is stationary, no interior phase structure.
        h = np.full(s.shape, h0, dtype=np.complex128)
    return s, np.stack([f, g, h], axis=1)


def _project(chars, est_col, K, cal, basis, component):
    """Project characteristic amplitudes onto one component of a variable flavor.

    ``chars`` is ``(n, 3)`` (or ``(3,)``); returns the chosen ``component`` of
    ``B @ (f, g, h)`` with ``B`` the flavor block at the mean state ``est_col``.
    """
    B = basis_block_from_state(basis, est_col, K, cal)
    w = np.atleast_2d(chars)
    v = np.einsum("ij,nj->ni", B, w)[:, component]
    return v if np.ndim(chars) > 1 else v[0]


def _adjacency(geo: NetworkGeometry):
    """``node -> list of (edge, neighbour node)`` (undirected over the FNS edges)."""
    adj = [[] for _ in range(geo.n_nodes)]
    for e in range(geo.n_edges):
        t, h = int(geo.tail_node[e]), int(geo.head_node[e])
        adj[t].append((e, h))
        adj[h].append((e, t))
    return adj


def _terminals(geo: NetworkGeometry):
    """Degree-1 nodes (the 1-port boundary elements)."""
    deg = np.zeros(geo.n_nodes, dtype=np.int64)
    np.add.at(deg, geo.tail_node, 1)
    np.add.at(deg, geo.head_node, 1)
    return [n for n in range(geo.n_nodes) if deg[n] == 1]


def _pick_root(geo: NetworkGeometry, terminals):
    """Choose the developed-length origin: a mean-flow inlet if present, else the lowest id."""
    inlets = [n for n in terminals if int(geo.node_rid[n]) in (MASS_FLOW_INLET, PT_INLET)]
    return min(inlets) if inlets else min(terminals)


def _node_paths(geo: NetworkGeometry, root: int):
    """Enumerate simple edge-paths from ``root`` to every other terminal.

    Each path is a list of ``(node, entry_edge)`` (``entry_edge`` is ``None`` at the
    root).  A branch point spawns one path per downstream leaf; cycles are cut by the
    used-edge set, so a network with a loop yields its spanning-tree paths.
    """
    adj = _adjacency(geo)
    terms = set(_terminals(geo))
    paths: List[List[Tuple[int, Optional[int]]]] = []

    def walk(node, entry_edge, used, trail):
        trail = trail + [(node, entry_edge)]
        if entry_edge is not None and node in terms:
            paths.append(trail)
            return
        nxt = [(e, o) for (e, o) in adj[node] if e != entry_edge and e not in used]
        if not nxt:
            paths.append(trail)  # dead end (e.g. an undriven stub)
            return
        for e, o in nxt:
            walk(o, e, used | {e}, trail)

    walk(root, None, set(), [])
    return paths


def _dedup(x, v, length_scale):
    """Drop consecutive points coincident in both developed length and value."""
    xs, vs = [x[0]], [v[0]]
    xtol = 1e-9 * max(length_scale, 1.0)
    for i in range(1, len(x)):
        dv = abs(v[i] - vs[-1])
        scale = max(abs(vs[-1]), abs(v[i]), 1e-300)
        if abs(x[i] - xs[-1]) <= xtol and dv <= 1e-9 * scale:
            continue
        xs.append(x[i])
        vs.append(v[i])
    return np.asarray(xs), np.asarray(vs, dtype=np.complex128)


def reconstruct_field(
    geometry: NetworkGeometry,
    chars_of_edge: Callable[[int], np.ndarray],
    est: np.ndarray,
    K: float,
    omega: complex,
    *,
    variable: str = "p",
    spec: Optional[Tuple[str, int]] = None,
    root: Optional[int] = None,
    n_x: int = 160,
    cals=None,
) -> List[PathField]:
    """Reconstruct the spatial perturbation field along every root->leaf path.

    Parameters
    ----------
    geometry : NetworkGeometry
        Topology and duct lengths (from :func:`build_geometry`).
    chars_of_edge : callable
        ``edge -> (f, g, h)`` complex amplitudes at that edge's face.
    est : ndarray
        Frozen mean edge-state table.
    K : float
        ``cp / R`` of the mean gas.
    omega : complex
        Angular frequency (rad/s).  Complex for an eigenmode (its small interior
        amplitude growth/decay is then captured exactly), real for a forced field.
    variable : str, optional
        Plotted quantity (:data:`VARIABLE_SPEC`): ``"p"`` (default), ``"u"``,
        ``"rho"``, ``"mdot"``, or a raw wave ``"f"``/``"g"``/``"h"``.  Ignored when
        ``spec`` is given.
    spec : tuple, optional
        A ``(basis_flavor, component)`` pair (e.g. from :func:`resolve_specs`) selecting
        any flavor component directly; overrides ``variable``.
    root : int, optional
        Developed-length origin element (default: a mean-flow inlet, else the
        lowest-id terminal).
    n_x : int, optional
        Interior samples per duct (default 160).

    Returns
    -------
    list of PathField

    Raises
    ------
    ValueError
        If ``variable`` is unknown, or the network has no terminal to root from.
    """
    if spec is not None:
        basis, component = spec
    else:
        if variable not in VARIABLE_SPEC:
            raise ValueError(f"unknown variable {variable!r}; choose from {sorted(VARIABLE_SPEC)}")
        basis, component, _label = VARIABLE_SPEC[variable]

    def cal_of(e):
        return None if cals is None else cals[e]

    def duct_fn(seg):
        e_t, e_h, L = seg.e_tail, seg.e_head, seg.length
        c = float(est[ES_C, e_t])
        u = float(est[ES_U, e_t])
        s, chars = _duct_chars(chars_of_edge(e_t), chars_of_edge(e_h), c, u, omega, L, n_x)
        return s, _project(chars, est[:, e_t], K, cal_of(e_t), basis, component)

    def point_fn(rep):
        return complex(_project(chars_of_edge(rep), est[:, rep], K, cal_of(rep), basis, component))

    return walk_paths(geometry, duct_fn, point_fn, root=root, n_x=n_x)


def walk_paths(geometry, duct_fn, point_fn, *, root=None, n_x=160):
    """Lay a per-station field out along every root->leaf developed-length path.

    The shared spatial-layout engine behind :func:`reconstruct_field` and the
    acoustic-power field diagnostics.  It walks each path from the root terminal,
    sampling ``duct_fn`` inside every length-bearing duct and ``point_fn`` at every
    compact (zero-length) node, and stitches the samples into a monotone
    developed-length trace per path.

    Parameters
    ----------
    geometry : NetworkGeometry
        Topology and duct lengths (from :func:`build_geometry`).
    duct_fn : callable
        ``DuctSegment -> (s, values)`` with ``s`` shape ``(n_x,)`` running tail->head
        and ``values`` the field at those stations.
    point_fn : callable
        ``edge -> value`` giving the field at a compact node's representative edge.
    root : int, optional
        Developed-length origin element (default: a mean-flow inlet, else the
        lowest-id terminal).
    n_x : int, optional
        Interior samples per duct (forwarded by the caller into ``duct_fn``).

    Returns
    -------
    list of PathField
    """
    terms = _terminals(geometry)
    if not terms:
        raise ValueError("network has no 1-port terminal to root the developed-length axis")
    root = _pick_root(geometry, terms) if root is None else int(root)
    dmap = geometry.duct_map()
    names = geometry.node_names

    def name_of(n):
        return names[n] if n < len(names) else ""

    fields: List[PathField] = []
    for trail in _node_paths(geometry, root):
        x_cursor = 0.0
        xs: List[float] = []
        vs: List[complex] = []
        markers: List[Tuple[float, str]] = []
        for idx, (node, entry) in enumerate(trail):
            exit_edge = trail[idx + 1][1] if idx + 1 < len(trail) else None
            if node in dmap:
                seg = dmap[node]
                s, psi = duct_fn(seg)
                L = seg.length
                if entry == seg.e_head:
                    #  traversed against the duct's flow axis: enter at the head face,
                    #  so developed length runs head -> tail; flip to stay ascending.
                    seg_x = (x_cursor + (L - s))[::-1]
                    psi = np.asarray(psi)[::-1]
                else:
                    seg_x = x_cursor + s
                xs.extend(seg_x.tolist())
                vs.extend(np.asarray(psi).tolist())
                x_cursor += L
            else:
                rep = entry if entry is not None else exit_edge
                if rep is not None:
                    xs.append(x_cursor)
                    vs.append(complex(point_fn(rep)))
                nm = name_of(node)
                if nm:
                    markers.append((x_cursor, nm))
        x_arr, v_arr = _dedup(np.asarray(xs), np.asarray(vs, dtype=np.complex128), x_cursor)
        start = name_of(trail[0][0]) or f"node {trail[0][0]}"
        end = name_of(trail[-1][0]) or f"node {trail[-1][0]}"
        fields.append(PathField(name=f"{start} → {end}", x=x_arr, values=v_arr, markers=markers))
    return fields
