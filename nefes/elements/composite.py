"""Composite elements -- convenience elements that expand to several atomic ones.

A :class:`CompositeElementSpec` presents to the user as a *single* element but
expands, at build time, into a small graph of atomic :class:`~nefes.elements.catalog.ElementSpec`
sub-elements joined by internal edges.  The expansion (:func:`expand_composites`) is a
pure **graph transformation** run once at the top of
:func:`nefes.elements.catalog.build_problem`: the solver, Jacobian assembly and
perturbation layers never see a composite, so at solve time the expanded graph is
indistinguishable from a hand-built one (no new kernels, no solver changes).

The machinery is deliberately **element-agnostic** -- it knows nothing about what its
sub-elements are, only their connectivity -- so one expander serves every composite,
from a fixed macro recipe (an orifice = ``isentropic_area_change`` + ``sudden_area_change``)
to an ``N``-segment discretization (a Fanno pipe, a tapered duct) to a future branching
sub-network.

Index policy: **append, never insert.**  A composite's first sub-element keeps the
composite's own node id; the rest are appended at the tail, and internal edges are
appended after the user edges.  So every *user* node and edge id keeps its exact
meaning after expansion -- and because the whole user-facing API is edge-indexed
(``states_table``, ``transfer_matrix``, ``scattering_matrix``), captured indices stay
valid for free.  (SuperLU re-permutes internally, so the tail append costs nothing at
solve time; a bandwidth-aware renumber is a deferred refinement -- see
``scratch/composite-elements.md`` Part III.)
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .ids import FIXED_NPORTS, RESIDUAL_NAMES


@dataclass
class CompositeElementSpec:
    """A convenience element that expands to >= 2 atomic ``ElementSpec`` sub-elements.

    Parameters
    ----------
    name : str
        Display name; sub-elements are namespaced under it (``orifice.iac``).
    sub_elements : list of ElementSpec
        The atomic elements the composite expands to (all atomic -- no nesting in v1).
    internal_edges : list of (int, int, float)
        Directed internal edges ``(tail_sub, head_sub, area)`` between sub-elements, by
        *local* sub-element index.  Each internal edge **is** an intermediate flow state
        (no junction needed for a serial chain).
    upstream_sub : int, optional
        Local index of the sub-element the inflow (external head) edge attaches to
        (default 0, the first sub-element, which keeps the composite's node id).
    downstream_sub : int, optional
        Local index of the sub-element the outflow (external tail) edge attaches to
        (default -1, the last sub-element).
    kind : str, optional
        A short type label for reporting (e.g. ``"orifice"``); defaults to ``name``.
    """

    name: str
    sub_elements: List[object]
    internal_edges: List[Tuple[int, int, float]]
    upstream_sub: int = 0
    downstream_sub: int = -1
    kind: str = ""

    @property
    def n_sub(self) -> int:
        return len(self.sub_elements)


@dataclass(frozen=True)
class CompositeMap:
    """Projects an expanded graph back to the user-facing (Case) topology.

    Built by :func:`expand_composites` and carried on the compiled problem so callers can
    hide a composite's internals by default, yet read its intra-element states on demand.

    Attributes
    ----------
    user_node_to_expanded : tuple of tuple of int
        ``user_node_to_expanded[n]`` is the expanded node ids a user node maps to (a
        single ``(n,)`` for an atomic element, the sub-element ids for a composite).
    internal_nodes : frozenset of int
        Expanded node ids that are composite internals (hidden by default).
    internal_edges : frozenset of int
        Expanded edge ids that are composite internals (hidden by default).
    composite_name : dict of int -> str
        User node id -> composite display name, for every composite element.
    composite_kind : dict of int -> str
        User node id -> composite kind label.
    """

    user_node_to_expanded: Tuple[Tuple[int, ...], ...]
    internal_nodes: FrozenSet[int]
    internal_edges: FrozenSet[int]
    composite_name: Dict[int, str] = field(default_factory=dict)
    composite_kind: Dict[int, str] = field(default_factory=dict)

    def expanded_nodes(self, user_node: int) -> Tuple[int, ...]:
        """The expanded node ids a user node expanded to (``(n,)`` if atomic)."""
        return self.user_node_to_expanded[user_node]


@dataclass
class CompositeView:
    """A solved composite element, presented as the single element the user added.

    Returned by :meth:`nefes.shell.network.Solution.composite`.  A composite expands at
    build time into a small graph of sub-elements joined by internal edges; this view lets a
    caller read those hidden internal states -- for example the throat of an orifice or a
    tapered nozzle -- without knowing the expanded node/edge layout.

    Attributes
    ----------
    name, kind : str
        The composite's display name and type label.
    node : int
        The composite's user node id (its first expanded sub-element).
    nodes : tuple of int
        All expanded node ids the composite occupies.
    internal_edges : tuple of int
        The composite's internal edge ids (both endpoints inside the composite).
    throat : int or None
        The narrowest internal edge (minimum area).  Meaningful for a contracting composite
        such as an orifice or nozzle, where it is the throat; ``None`` if the composite has no
        internal edge.
    """

    name: str
    kind: str
    node: int
    nodes: Tuple[int, ...]
    internal_edges: Tuple[int, ...]
    throat: Optional[int]
    _solution: object = None

    def state(self, e: int) -> dict:
        """``{field: value}`` of all derived quantities on edge ``e`` (an internal or boundary edge)."""
        return self._solution.edge(int(e))

    @property
    def throat_state(self) -> dict:
        """``{field: value}`` at the throat (narrowest) edge; empty if the composite has none."""
        return {} if self.throat is None else self.state(self.throat)

    def profile(self, name: str):
        """The named field along the composite's internal edges (area-ordered, narrowest first)."""
        field = self._solution.field(name)
        order = sorted(self.internal_edges, key=lambda e: float(self._solution.field("area")[e]))
        return np.array([field[e] for e in order])


def is_composite(el) -> bool:
    """True if ``el`` is a :class:`CompositeElementSpec` (vs an atomic element spec)."""
    return isinstance(el, CompositeElementSpec)


@dataclass(frozen=True)
class GridRefinement:
    """The result of refining a discretization composite from ``N`` to ``2N`` segments.

    A converged refinement *is* the verification of a Class-2 composite: if the quantities of
    interest barely move when the segment count doubles, the chain has resolved the
    continuous element (theory: ``scratch/composite-elements.md`` "Choosing N").

    Attributes
    ----------
    n_coarse, n_fine : int
        The coarse and (doubled) fine segment counts.
    coarse, fine : dict
        The probed quantities ``{name: value}`` at each resolution.
    rel_change : dict
        Relative change ``|fine - coarse| / (|fine| + tiny)`` per quantity.
    """

    n_coarse: int
    n_fine: int
    coarse: dict
    fine: dict
    rel_change: dict

    def converged(self, tol: float = 1e-2) -> bool:
        """Whether every quantity changed by less than ``tol`` (relative) under refinement."""
        return all(v < tol for v in self.rel_change.values())

    @property
    def worst(self) -> float:
        """The largest relative change across the probed quantities."""
        return max(self.rel_change.values()) if self.rel_change else 0.0


def grid_refine(build, n_coarse, probe):
    """Refine a discretization composite from ``n_coarse`` to ``2*n_coarse`` and report the change.

    The principled way to pick ``N`` for a :func:`~nefes.elements.catalog.fanno_pipe` /
    :func:`~nefes.elements.catalog.tapered_duct`: solve at two resolutions and watch the
    quantities of interest settle.  Element-agnostic -- it only calls the supplied callables.

    Parameters
    ----------
    build : callable
        ``build(N)`` -> a solved object (e.g. a ``Solution``) for ``N`` segments.
    n_coarse : int
        The coarse segment count; the fine solve uses ``2 * n_coarse``.
    probe : callable
        ``probe(solved)`` -> a ``dict`` (or mapping) of scalar quantities of interest (e.g.
        exit Mach, choke back-pressure).

    Returns
    -------
    GridRefinement
    """
    coarse = dict(probe(build(int(n_coarse))))
    fine = dict(probe(build(2 * int(n_coarse))))
    rel = {k: abs(float(fine[k]) - float(coarse[k])) / (abs(float(fine[k])) + 1e-300) for k in coarse}
    return GridRefinement(n_coarse=int(n_coarse), n_fine=2 * int(n_coarse), coarse=coarse, fine=fine, rel_change=rel)


def _implied_degree(spec: CompositeElementSpec, sub: int) -> int:
    """The wired degree a sub-element should have under the serial-composite recipe.

    Counts the sub-element's internal-edge incidences plus its one external inflow
    (if it is the ``upstream_sub``) and one external outflow (if ``downstream_sub``).
    """
    down = spec.downstream_sub if spec.downstream_sub >= 0 else spec.n_sub - 1
    deg = 0
    for ts, hs, _a in spec.internal_edges:
        if ts == sub:
            deg += 1
        if hs == sub:
            deg += 1
    if sub == spec.upstream_sub:
        deg += 1
    if sub == down:
        deg += 1
    return deg


def validate_composite(spec: CompositeElementSpec):
    """Structural validation of a composite recipe, *before* expansion.

    Errors name the composite (not a cryptic expanded node).  Checks: >= 2 atomic
    sub-elements (no nesting); ``internal_edges`` and ``upstream``/``downstream`` indices
    in range and areas positive; and that each sub-element's implied wired degree matches
    its fixed arity (``FIXED_NPORTS``) under the serial recipe.

    Raises
    ------
    ValueError
        On any structural defect, with a message naming the composite.
    """
    label = f"composite {spec.name!r}"
    n = spec.n_sub
    if n < 2:
        raise ValueError(f"{label}: a composite needs >= 2 sub-elements; got {n}")
    for k, sub in enumerate(spec.sub_elements):
        if is_composite(sub):
            raise ValueError(f"{label}: nested composites are not supported (sub-element {k} is a composite)")
        if not hasattr(sub, "residual_id"):
            raise ValueError(f"{label}: sub-element {k} is not an element spec")
    for j, (ts, hs, a) in enumerate(spec.internal_edges):
        if not (0 <= ts < n and 0 <= hs < n):
            raise ValueError(f"{label}: internal edge {j} references sub-element out of range [0, {n}): {(ts, hs)}")
        if ts == hs:
            raise ValueError(f"{label}: internal edge {j} is a self-loop on sub-element {ts}")
        if not float(a) > 0.0:
            raise ValueError(f"{label}: internal edge {j} must have a positive area; got {a}")
    if not 0 <= spec.upstream_sub < n:
        raise ValueError(f"{label}: upstream_sub {spec.upstream_sub} out of range [0, {n})")
    down = spec.downstream_sub if spec.downstream_sub >= 0 else n - 1
    if not 0 <= down < n:
        raise ValueError(f"{label}: downstream_sub {spec.downstream_sub} out of range [0, {n})")
    # each sub-element's implied wired degree must match its fixed arity, if it has one
    for k, sub in enumerate(spec.sub_elements):
        expected = FIXED_NPORTS.get(int(sub.residual_id))
        if expected is not None:
            deg = _implied_degree(spec, k)
            if deg != expected:
                tname = RESIDUAL_NAMES.get(int(sub.residual_id), f"residual#{sub.residual_id}")
                raise ValueError(
                    f"{label}: sub-element {k} ({tname}) is wired to {deg} port(s) under the recipe "
                    f"but is a {expected}-port element -- check internal_edges / upstream_sub / downstream_sub"
                )


def expand_composites(elements, edges):
    """Expand every composite in ``elements`` into atomic elements + internal edges.

    A pure build-time graph transformation (plain Python, no flow state -- complex-step
    safety is a non-issue).  The first sub-element of each composite keeps the composite's
    node id; the remaining sub-elements are appended at the tail, and internal edges are
    appended after the user edges.  External edges are rewired by **orientation**: an edge
    whose *head* is a composite enters at that composite's ``upstream_sub``; an edge whose
    *tail* is a composite leaves from its ``downstream_sub``.

    Parameters
    ----------
    elements : list
        User elements, atomic or composite.
    edges : list of (int, int, float)
        User directed edges ``(tail, head, area)``.

    Returns
    -------
    (list, list, CompositeMap or None)
        The expanded ``(elements, edges)`` -- an ordinary atomic pair ready for
        :func:`build_problem` -- and a :class:`CompositeMap` (``None`` when the network
        carries no composite, the zero-overhead fast path).
    """
    if not any(is_composite(el) for el in elements):
        return elements, edges, None

    n_user = len(elements)
    out_elements = list(elements)  # slot i may be replaced by the composite's upstream sub
    slots: List[List[int]] = [[i] for i in range(n_user)]  # user node -> expanded node ids
    next_id = n_user

    for i, el in enumerate(elements):
        if not is_composite(el):
            continue
        validate_composite(el)
        out_elements[i] = el.sub_elements[0]  # upstream sub keeps slot i
        ids = [i]
        for sub in el.sub_elements[1:]:
            out_elements.append(sub)
            ids.append(next_id)
            next_id += 1
        slots[i] = ids

    def _up(n):
        el = elements[n]
        return slots[n][el.upstream_sub] if is_composite(el) else n

    def _down(n):
        el = elements[n]
        if not is_composite(el):
            return n
        d = el.downstream_sub if el.downstream_sub >= 0 else el.n_sub - 1
        return slots[n][d]

    # rewire external edges by orientation: leave a composite at its tail, enter at its head
    new_edges = [(_down(t), _up(h), a) for (t, h, a) in edges]

    # append internal edges (each composite's own internal connectivity)
    internal_edge_ids = set()
    for i, el in enumerate(elements):
        if not is_composite(el):
            continue
        for ts, hs, a in el.internal_edges:
            internal_edge_ids.add(len(new_edges))
            new_edges.append((slots[i][ts], slots[i][hs], a))

    cmap = CompositeMap(
        user_node_to_expanded=tuple(tuple(s) for s in slots),
        internal_nodes=frozenset(n for i, el in enumerate(elements) if is_composite(el) for n in slots[i][1:]),
        internal_edges=frozenset(internal_edge_ids),
        composite_name={i: el.name for i, el in enumerate(elements) if is_composite(el)},
        composite_kind={i: (el.kind or el.name) for i, el in enumerate(elements) if is_composite(el)},
    )
    return out_elements, new_edges, cmap
