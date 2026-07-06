"""Composite elements -- convenience elements that expand to several atomic ones.

A :class:`CompositeElementSpec` presents to the user as a *single* element but
expands, at build time, into a small graph of atomic :class:`~nefes.elements.catalog.ElementSpec`
sub-elements joined by internal edges.  The expansion (:func:`expand_composites`) is a
pure **graph transformation** run once at the top of :func:`nefes.shell.build.build_problem`:
the solver, Jacobian assembly and perturbation layers never see a composite, so at solve time the
expanded graph is indistinguishable from a hand-built one (no new kernels, no solver changes).

The machinery is deliberately **element-agnostic** -- it knows nothing about what its
sub-elements are, only their connectivity -- so one expander serves every composite,
from a fixed macro recipe (an orifice = ``isentropic_area_change`` + ``sudden_area_change``)
to an ``N``-segment discretization (a Fanno pipe, a tapered duct) to a future branching
sub-network.

Index policy: **append, never insert.**  A composite's first sub-element keeps the
composite's own node id; the rest are appended at the tail, and internal edges are
appended after the user edges.  So every *user* node and edge id keeps its exact
meaning after expansion. Because the whole user-facing API is edge-indexed
(``states_table``, ``transfer_matrix``, ``scattering_matrix``), captured indices stay
valid for free.  (SuperLU re-permutes internally, so the tail append costs nothing at
solve time; a bandwidth reduction is not attempted therefore).
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .ids import FIXED_NPORTS, ELEMENT_TYPE_NAMES


@dataclass
class CompositeElementSpec:
    """A convenience element that expands to >= 2 atomic ``ElementSpec`` sub-elements.

    Parameters
    ----------
    name : str
        Display name; sub-elements are namespaced under it (``orifice.iac``).
    sub_elements : list of ElementSpec
        The atomic elements the composite expands to (all atomic -- no nesting yet).
    internal_edges : list of (int, int, float or callable)
        Directed internal edges ``(tail_sub, head_sub, area)`` between sub-elements, by
        *local* sub-element index.  Each internal edge **is** an intermediate flow state
        (no junction needed for a serial chain).  An area may be a callable
        ``f(a_up, a_down) -> float`` resolved at expansion from the areas of the
        composite's external inflow/outflow edges, so a recipe can size itself relative
        to the edges it is wired to (areas live on edges, never on elements).
    upstream_sub : int, optional
        Local index of the sub-element the inflow (external head) edge attaches to
        (default 0, the first sub-element, which keeps the composite's node id).
    downstream_sub : int, optional
        Local index of the sub-element the outflow (external tail) edge attaches to
        (default -1, the last sub-element).
    kind : str, optional
        A short type label for reporting (e.g. ``"orifice"``); defaults to ``name``.
    params : dict, optional
        The composite's own constructor parameters, as passed to its catalog factory
        (e.g. ``{"throat_area": 1e-3}``).  Purely descriptive -- the expansion never
        reads it -- but it lets reporting and the UI serialization
        (:mod:`nefes.io.yaml_out`) recover the composite as the single element the
        user specified instead of its expanded internals.
    """

    name: str
    sub_elements: List[object]
    internal_edges: List[Tuple[int, int, float]]
    upstream_sub: int = 0
    downstream_sub: int = -1
    kind: str = ""
    params: Dict[str, object] = field(default_factory=dict)

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
    """True if ``el`` is a :class:`CompositeElementSpec` (vs an atomic element spec).

    Parameters
    ----------
    el : object
        An element spec (atomic ``ElementSpec`` or a ``CompositeElementSpec``).

    Returns
    -------
    bool
    """
    return isinstance(el, CompositeElementSpec)


@dataclass(frozen=True)
class GridRefinement:
    """The result of refining a discretization composite from ``N`` to ``2N`` segments.

    A converged refinement verifies that the segment chain has resolved the continuous
    element it discretizes: if the quantities of interest barely move when the segment
    count doubles, the discretization is fine enough.

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
    quantities of interest settle.  Element-agnostic, it only calls the supplied callables.

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


@dataclass(frozen=True)
class AutoRefinement:
    """The outcome of :func:`auto_refine`: the doubling history and whether it converged.

    Attributes
    ----------
    steps : list of GridRefinement
        One entry per doubling, oldest first; ``steps[i]`` compares the ``i``-th resolution
        to the next.  Empty only if ``max_refine`` was ``0`` (not allowed).
    converged : bool
        ``True`` if every probed quantity settled below the tolerance within ``max_refine``
        doublings; ``False`` if the cap was reached first.
    n_final : int
        The finest segment count actually solved.
    final : dict
        The probed quantities at ``n_final``.
    """

    steps: List["GridRefinement"]
    converged: bool
    n_final: int
    final: dict

    @property
    def n_refine(self) -> int:
        """Number of doublings performed."""
        return len(self.steps)

    @property
    def worst(self) -> float:
        """The largest relative change at the final doubling (``0`` if no step ran)."""
        return self.steps[-1].worst if self.steps else 0.0


def auto_refine(build, n_start, probe, *, tol: float = 1e-2, max_refine: int = 6) -> AutoRefinement:
    """Refine a discretization composite until its quantities of interest stop moving.

    Repeatedly doubles the segment count -- ``n_start, 2*n_start, 4*n_start, ...`` -- and
    solves each resolution, stopping when every probed quantity changes by less than ``tol``
    (relative) from the previous resolution.  This automates the manual :func:`grid_refine`
    sweep, at the cost of one solve per doubling.

    Because a stubborn composite can demand an impractical number of segments before it
    settles, the loop is capped at ``max_refine`` doublings; if the cap is reached first the
    result is returned with ``converged = False`` (and :attr:`AutoRefinement.worst` reports how
    far it still was), so the caller can decide whether the accuracy is worth the cost.

    Parameters
    ----------
    build : callable
        ``build(N)`` -> a solved object (e.g. a ``Solution``) for ``N`` segments.
    n_start : int
        The coarsest segment count (``>= 1``).
    probe : callable
        ``probe(solved)`` -> a mapping of scalar quantities of interest (e.g. exit Mach,
        choke back-pressure).
    tol : float, optional
        Relative-change convergence tolerance (default ``1e-2``).
    max_refine : int, optional
        Maximum number of doublings to attempt (default ``6``, i.e. up to ``64 * n_start``).

    Returns
    -------
    AutoRefinement
    """
    n_start = int(n_start)
    if n_start < 1:
        raise ValueError(f"n_start must be >= 1; got {n_start}")
    if int(max_refine) < 1:
        raise ValueError(f"max_refine must be >= 1; got {max_refine}")

    prev_n = n_start
    prev = dict(probe(build(prev_n)))
    steps = []
    converged = False
    for _ in range(int(max_refine)):
        n = prev_n * 2
        cur = dict(probe(build(n)))
        rel = {k: abs(float(cur[k]) - float(prev[k])) / (abs(float(cur[k])) + 1e-300) for k in cur}
        step = GridRefinement(n_coarse=prev_n, n_fine=n, coarse=prev, fine=cur, rel_change=rel)
        steps.append(step)
        prev_n, prev = n, cur
        if step.converged(tol):
            converged = True
            break
    return AutoRefinement(steps=steps, converged=converged, n_final=prev_n, final=prev)


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
    its fixed port count (``FIXED_NPORTS``) under the serial recipe.

    Parameters
    ----------
    spec : CompositeElementSpec
        The composite recipe to validate.

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
        # a callable area is resolved (and positivity-checked) at expansion, when the
        # attached external edge areas are known
        if not callable(a) and not float(a) > 0.0:
            raise ValueError(f"{label}: internal edge {j} must have a positive area; got {a}")
    if not 0 <= spec.upstream_sub < n:
        raise ValueError(f"{label}: upstream_sub {spec.upstream_sub} out of range [0, {n})")
    down = spec.downstream_sub if spec.downstream_sub >= 0 else n - 1
    if not 0 <= down < n:
        raise ValueError(f"{label}: downstream_sub {spec.downstream_sub} out of range [0, {n})")
    # each sub-element's implied wired degree must match its fixed port count, if it has one
    for k, sub in enumerate(spec.sub_elements):
        expected = FIXED_NPORTS.get(int(sub.residual_id))
        if expected is not None:
            deg = _implied_degree(spec, k)
            if deg != expected:
                tname = ELEMENT_TYPE_NAMES.get(int(sub.residual_id), f"residual#{sub.residual_id}")
                raise ValueError(
                    f"{label}: sub-element {k} ({tname}) is wired to {deg} port(s) under the recipe "
                    f"but is a {expected}-port element -- check internal_edges / upstream_sub / downstream_sub"
                )


def expand_composites(elements, edges, ports=None):
    """Expand every composite in ``elements`` into atomic elements + internal edges.

    A pure build-time graph transformation.  The first sub-element of each composite keeps
    the composite's node id; the remaining sub-elements are appended at the tail, and internal
    edges are appended after the user edges.  External edges are rewired by **orientation**: an edge
    whose *head* is a composite enters at that composite's ``upstream_sub``; an edge whose
    *tail* is a composite leaves from its ``downstream_sub``.  Each expanded edge is emitted with
    **explicit flow-aligned ports** -- a fixed 2-port sub-element takes its inflow on port 0 and
    its outflow on port 1 -- so wiring matches a hand-built network regardless of edge index (the
    tail-appended internal edges would otherwise mis-order a sub-element's ports).

    Parameters
    ----------
    elements : list
        User elements, atomic or composite.
    edges : list of (int, int, float)
        User directed edges ``(tail, head, area)``.
    ports : list of (int or None, int or None), optional
        Per-edge user port pins ``(tail_port, head_port)`` aligned with ``edges`` (as recorded
        by :meth:`nefes.shell.network.Network.connect` / a UI export).  A pin at an **atomic**
        endpoint is preserved verbatim; a pin at a **composite** endpoint refers to a node that
        no longer exists after the expansion, so it is replaced by the rewired sub-element's
        flow-aligned port.  ``None`` (default) auto-assigns every port as before.

    Returns
    -------
    (list, list, CompositeMap or None)
        The expanded ``elements`` and ``edges``, ready for :func:`build_problem`, plus a
        :class:`CompositeMap` (``None`` when the network carries no composite -- the
        zero-overhead fast path that returns the inputs unchanged).  When a composite is
        present the edges are port-explicit 5-tuples ``(tail, head, area, tail_port, head_port)``;
        otherwise they are the input ``(tail, head, area)`` triples verbatim.
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

    # rewire external edges by orientation: leave a composite at its tail, enter at its head.
    # A user pin survives only at an atomic endpoint -- a composite endpoint is rewired to a
    # sub-element the user never addressed, so its pin is dropped and re-derived flow-aligned.
    ext_pins = ports if ports is not None else [(None, None)] * len(edges)
    new_edges = [(_down(t), _up(h), a) for (t, h, a) in edges]
    pins = [
        (None if is_composite(elements[t]) else tp, None if is_composite(elements[h]) else hp)
        for (t, h, _a), (tp, hp) in zip(edges, ext_pins)
    ]

    # append internal edges (each composite's own internal connectivity); a callable
    # area sizes itself from the composite's external inflow/outflow edge areas
    def _resolve_area(i, el, j, a):
        if not callable(a):
            return a
        label = f"composite {el.name!r}"
        inflow = [ae for (t, h, ae) in edges if h == i]
        outflow = [ae for (t, h, ae) in edges if t == i]
        if len(inflow) != 1 or len(outflow) != 1:
            raise ValueError(
                f"{label}: internal edge {j} sizes itself from the attached edges, which needs exactly "
                f"one inflow and one outflow edge; got {len(inflow)} inflow / {len(outflow)} outflow"
            )
        resolved = float(a(float(inflow[0]), float(outflow[0])))
        if not resolved > 0.0:
            raise ValueError(f"{label}: internal edge {j} resolved to a non-positive area; got {resolved}")
        return resolved

    internal_edge_ids = set()
    for i, el in enumerate(elements):
        if not is_composite(el):
            continue
        for j, (ts, hs, a) in enumerate(el.internal_edges):
            internal_edge_ids.add(len(new_edges))
            new_edges.append((slots[i][ts], slots[i][hs], _resolve_area(i, el, j, a)))
            pins.append((None, None))

    # Wire explicit ports, exactly as a user connecting a serial chain by hand: a fixed 2-port
    # element takes its inflow (the edge it heads) on port 0 and its outflow (the edge it tails)
    # on port 1, so the flow axis is always port 0 -> port 1.  This is load-bearing because the
    # internal edges are appended at the tail: without pinning, a sub-element's inflow edge can
    # outrank its outflow edge by index and land on the wrong port -- which then trips the
    # acoustic flow-alignment check on an internal duct.  Every other node auto-assigns its ports
    # in attachment order, identical to the bare-edge path.
    def _two_port(n):
        return FIXED_NPORTS.get(int(out_elements[n].residual_id)) == 2

    next_port = [0] * len(out_elements)

    def _auto(n):
        p = next_port[n]
        next_port[n] += 1
        return p

    def _tail_port(n, pin):
        if pin is not None:
            return int(pin)
        return 1 if _two_port(n) else _auto(n)

    def _head_port(n, pin):
        if pin is not None:
            return int(pin)
        return 0 if _two_port(n) else _auto(n)

    new_edges = [(t, h, a, _tail_port(t, tp), _head_port(h, hp)) for (t, h, a), (tp, hp) in zip(new_edges, pins)]

    cmap = CompositeMap(
        user_node_to_expanded=tuple(tuple(s) for s in slots),
        internal_nodes=frozenset(n for i, el in enumerate(elements) if is_composite(el) for n in slots[i][1:]),
        internal_edges=frozenset(internal_edge_ids),
        composite_name={i: el.name for i, el in enumerate(elements) if is_composite(el)},
        composite_kind={i: (el.kind or el.name) for i, el in enumerate(elements) if is_composite(el)},
    )
    return out_elements, new_edges, cmap
