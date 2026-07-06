"""CompiledProblem builder: elements + directed edges -> the immutable compiled problem.

The parse-time counterpart to the element catalog.  :func:`build_problem` (and the
port-explicit :func:`build_problem_from_connectivity`) turn a list of
:class:`~nefes.elements.catalog.ElementSpec` plus ``(tail, head, area)`` edges into a
:class:`~nefes.shell.problem.CompiledProblem`: it validates the network, discovers the
reacting feed streams, packs the per-node kernel parameters, and lays out the residual /
Jacobian scales.  The user-facing wrapper over it is :class:`~nefes.shell.network.Network`.
"""

from collections import defaultdict
from typing import List, Tuple

import numpy as np

from ..chem.composition import build_streams, enthalpy_mass, species_mass_fractions
from ..graph.connectivity import (
    build_connectivity,
    build_jacobian_pattern,
    Connectivity,
)
from . import checks
from .problem import CompiledProblem
from ..thermo.api import EQ_FROZEN, EQ_KERNEL, EQ_MARKER, PERFECT_GAS
from ..thermo.configure import ThermoConfig
from ..elements.catalog import ElementSpec, ensure_unique_names
from ..elements.composite import expand_composites
from ..elements.ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    CHOKED_NOZZLE_OUTLET,
    JUNCTION,
    SPLITTER,
    FORCED_SPLITTER,
    FLAME_EQUILIBRIUM,
    MASS_SOURCE,
    FIXED_NPORTS,
    ALLOWS_AREA_CHANGE,
    ELEMENT_TYPE_NAMES,
    KIND_MASS,
    KIND_PRESSURE,
    PORT_SOURCE,
    PORT_TARGET,
    PORT_ANY,
    port_kinds,
    row_kind_tags,
    STREAM_INTRODUCING,
)

# Relative tolerance for the equal-area check on constant-area elements.
_AREA_RTOL = 1e-9

# Boundaries that fix an *absolute pressure* (total_pressure_inlet, pressure_outlet) or tie the
# pressure level via a flow<->pressure relation (choked_nozzle_outlet).  At least one is needed or
# the mean-flow pressure level is a free gauge.
_PRESSURE_REFERENCE_RIDS = (PT_INLET, P_OUTLET, CHOKED_NOZZLE_OUTLET)

# Boundaries that prescribe an advected-scalar (h_t + composition) donor on ingestion.  The
# mass-flow / choked-nozzle outlets are outflow-only (no backflow), so they carry no such donor --
# their edge inherits the interior scalars (scalar-transparent, see node_donor).
_SCALAR_BOUNDARY_RIDS = (MASS_FLOW_INLET, PT_INLET, P_OUTLET)


def _node_label(n: int, el: ElementSpec) -> str:
    """Human-readable identifier for an element, for validation messages."""
    typ = ELEMENT_TYPE_NAMES.get(el.residual_id, f"residual {el.residual_id}")
    name = f" {el.name!r}" if el.name else ""
    return f"element {n}{name} ({typ})"


def _take_port(kinds, used, allowed):
    """Claim the lowest free local port whose kind is in ``allowed`` (or any free one).

    Returns the chosen port index and marks it used.  When ``kinds`` is ``None`` (the node's
    degree does not match its declared port count, an error :func:`validate_network` reports)
    or no kind-matching port is free, it falls back to the lowest free port so assignment still
    completes -- e.g. a boundary edge drawn against its nominal direction lands on the element's
    only port, and the mean-flow solve reverses the sign there.
    """
    if kinds is not None:
        for i, k in enumerate(kinds):
            if not used[i] and k in allowed:
                used[i] = True
                return i
    for i in range(len(used)):  # fallback: lowest free port, regardless of kind
        if not used[i]:
            used[i] = True
            return i
    raise AssertionError("no free port")  # unreachable: exactly deg edges touch each node


def _assign_ports_by_kind(elements: List[ElementSpec], edges) -> list:
    """Assign each directed edge to a kind-matching local port at both endpoints.

    The tail (source side) claims the lowest free source / bidirectional port and the head
    (target side) the lowest free target / bidirectional port, so port 0 of a two-port through
    element is always its inflow regardless of the order edges were attached.  Returns the
    endpoint table ``[(tail, tail_port, head, head_port), ...]`` for :func:`build_connectivity`.
    """
    n = len(elements)
    deg = [0] * n
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    # per-node port kinds when the degree is consistent, else None (order-based fallback so the
    # degree mismatch surfaces as validate_network's port-count error, not an assignment crash)
    node_kinds = []
    for i in range(n):
        k = port_kinds(elements[i].residual_id, deg[i])
        node_kinds.append(k if len(k) == deg[i] else None)
    used = [[False] * deg[i] for i in range(n)]
    endpoints = []
    for t, h, *_rest in edges:
        tp = _take_port(node_kinds[t], used[t], (PORT_SOURCE, PORT_ANY))
        hp = _take_port(node_kinds[h], used[h], (PORT_TARGET, PORT_ANY))
        endpoints.append((t, tp, h, hp))
    return endpoints


def validate_network(elements: List[ElementSpec], conn: Connectivity, area: np.ndarray, require_connected=None) -> None:
    """Check structural and area-consistency invariants before compiling.

    Also normalizes element display names to be unique (see
    :func:`ensure_unique_names`) -- duplicates, common with the factory defaults,
    are suffixed in place rather than rejected.

    Raises ``ValueError`` (naming the offending element) on the first violation:

    * every edge area is finite and strictly positive;
    * each element's wired port count is admissible -- exactly ``FIXED_NPORTS`` for
      an element with a fixed port count, ``>= 2`` for the variable junction/splitter;
    * elements that do not permit an area change (``ALLOWS_AREA_CHANGE`` is
      ``False`` -- the constant-area duct) carry one shared area across all their
      incident edges.  An intended area change at an area-agnostic element (e.g. a
      sudden expansion) must use an ``isentropic_area_change`` or
      ``sudden_area_change`` element;
    * a pressure reference exists (see :func:`_check_pressure_reference`);
    * unless disabled, the elements form a single connected sub-network (see
      :func:`nefes.shell.checks.assert_single_component`);
    * unless disabled, no edge joins an incompatible element-type pairing (see
      :func:`nefes.shell.checks.assert_allowed_connections`).

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements, in node order.
    conn : Connectivity
        The compiled connectivity (per-node incident edges and degrees).
    area : ndarray
        Per-edge cross-sectional area, indexed by global edge id.
    require_connected : bool, optional
        Reject a model that splits into disconnected sub-networks.  ``None`` (default) follows
        the process-wide :data:`nefes.shell.checks.CHECK_CONNECTED` toggle; pass ``True`` /
        ``False`` to force the check on / off for this call.
    """
    if require_connected is None:
        require_connected = checks.CHECK_CONNECTED
    ensure_unique_names(elements)
    area = np.asarray(area, dtype=np.float64)
    if area.size != conn.n_edges:
        raise ValueError(f"area has {area.size} entries but the network has {conn.n_edges} edges")
    bad = np.nonzero(~(np.isfinite(area) & (area > 0.0)))[0]
    if bad.size:
        raise ValueError(f"edge areas must be finite and positive; offending edge id(s): {bad.tolist()}")
    if len(elements) != conn.n_nodes:
        raise ValueError(f"{len(elements)} elements but the connectivity has {conn.n_nodes} nodes")

    for n, el in enumerate(elements):
        rid = el.residual_id
        deg = conn.degree(n)
        label = _node_label(n, el)

        expected = FIXED_NPORTS.get(rid)
        if expected is not None:
            if deg != expected:
                raise ValueError(f"{label} expects {expected} port(s) but is connected to {deg} edge(s)")
        elif rid in (JUNCTION, SPLITTER):
            if deg < 2:
                raise ValueError(f"{label} is a manifold and needs >= 2 ports but is connected to {deg} edge(s)")
        elif rid == FORCED_SPLITTER:
            if deg < 3:
                raise ValueError(
                    f"{label} is a forced splitter and needs >= 3 ports (1 inflow + >= 2 outflows) "
                    f"but is connected to {deg} edge(s)"
                )
            n_frac = len(el.fparams)
            if n_frac != deg - 2:
                raise ValueError(
                    f"{label}: a forced splitter with {deg} ports (1 inflow + {deg - 1} outflows) needs "
                    f"{deg - 2} split fraction(s) -- one per controlled outflow, the last outflow being the "
                    f"remainder -- but {n_frac} were given"
                )

        if rid == CHOKED_NOZZLE_OUTLET:
            # the compact choked nozzle is a *contraction* to a sonic throat; the throat
            # area A* must be smaller than the outlet edge area so the approach plane stays
            # subsonic (A_out/A* > 1 has a unique subsonic area-Mach root).  A* >= A_out has
            # no subsonic choked solution (that is a converging-diverging / supersonic case).
            a_out = float(area[conn.incident_edges(n)[0]])
            a_star = float(el.fparams[0])
            if not a_star < a_out:
                raise ValueError(
                    f"{label}: choked-nozzle throat area A* = {a_star:g} m^2 must be smaller than "
                    f"the outlet area {a_out:g} m^2 (a contraction). A* >= A_out has no subsonic "
                    "choked approach; a converging-diverging (supersonic) nozzle currently not "
                    "supported"
                )

        if not ALLOWS_AREA_CHANGE.get(rid, True) and deg >= 2:
            inc = conn.incident_edges(n)
            a0 = float(area[inc[0]])
            for e in inc[1:]:
                ae = float(area[e])
                if abs(ae - a0) > _AREA_RTOL * max(abs(a0), abs(ae)):
                    raise ValueError(
                        f"{label} does not permit an area change but its ports carry different "
                        f"areas ({a0:g} vs {ae:g} m^2); model the area change with an "
                        f"isentropic_area_change or sudden_area_change element"
                    )

    _check_pressure_reference(elements)
    if require_connected:
        checks.assert_single_component(conn)
    if checks.CHECK_CONNECTIONS:
        checks.assert_allowed_connections(elements, conn)


def _check_pressure_reference(elements: List[ElementSpec]) -> None:
    """Reject a boundary set with no absolute-pressure reference (a singular gauge).

    The steady residual fixes pressure only through *differences* (momentum, area-change
    and loss rows) plus the absolute pin a pressure boundary supplies.  If **every**
    boundary merely prescribes a mass flow (``mass_flow_inlet`` / ``mass_flow_outlet`` /
    ``wall``), the pressure level is undetermined: adding a constant to every pressure
    leaves the residual unchanged to leading order, so the Jacobian is singular and the
    solve cannot converge.  A ``total_pressure_inlet`` or ``pressure_outlet`` pins the
    level directly; a ``choked_nozzle_outlet`` pins it via its critical-mass-flux
    relation (the interior stagnation pressure is fixed once the flow is).
    """
    if any(el.residual_id in _PRESSURE_REFERENCE_RIDS for el in elements):
        return
    raise ValueError(
        "ill-posed boundary conditions: the network has no absolute-pressure reference. "
        "Every boundary fixes a mass flow (mass_flow_inlet / mass_flow_outlet / wall), so the "
        "pressure level is a free gauge and the steady solve is singular. Add a pressure_outlet "
        "or total_pressure_inlet (an absolute-pressure pin), or a choked_nozzle_outlet "
        "(a flow<->pressure relation), to set the level."
    )


def _row_kinds(rid: int, deg: int, mdot_ref, p_ref):
    """Residual-row scale magnitudes for one element (derived from its kind tags)."""
    scale = {KIND_MASS: mdot_ref, KIND_PRESSURE: p_ref}
    return [scale[tag] for tag in row_kind_tags(rid, deg)]


def _onehot(k: int, n: int):
    """Mixture-fraction unit vector: all mass from stream ``k`` (``[]`` if ``n==0``)."""
    xi = [0.0] * n
    if 0 <= k < n:
        xi[k] = 1.0
    return xi


def _boundary_scalars(thermo: ThermoConfig, el: ElementSpec, Tt: float, n_elem: int, label: str, stream: int):
    """Resolve a boundary's advected-scalar params ``[h_t, xi_0, ..., xi_{n_elem-1}]``.

    Converts the element's total temperature to the absolute total enthalpy
    ``h_t = h(Tt)`` and tags the stream it introduces with the mixture-fraction unit
    vector ``xi = e_stream``.  For a perfect gas ``h_t = cp*Tt`` and the composition (if
    any) is the raw passive-scalar values; for the equilibrium model the composition is
    a named species mixture whose own (formation-inclusive) enthalpy at ``Tt`` is used.
    """
    if thermo.model_id == PERFECT_GAS:
        h_t = float(thermo.tf[0]) * Tt  # cp * Tt
        if n_elem == 0:
            return [h_t]
        comp = el.composition_spec
        if comp is None:
            return [h_t] + [0.0] * n_elem
        zvals = [float(c) for c in comp]
        if len(zvals) != n_elem:
            raise ValueError(f"{label} carries {len(zvals)} scalar(s) but the model has {n_elem}")
        return [h_t] + zvals

    # equilibrium / reacting backend: an explicit species composition is required
    # for any stream that introduces mass (inlets, mass sources); an outlet may
    # omit it (its backflow scalars are used only on ingestion).
    comp = el.composition_spec
    if comp is None:
        if el.residual_id in (MASS_FLOW_INLET, PT_INLET):
            raise ValueError(
                f"{label}: the equilibrium model requires an explicit species composition "
                f"(e.g. composition={{'O2': 0.21, 'N2': 0.79}})"
            )
        return [0.0] + [0.0] * n_elem  # inert backflow placeholder
    Y = species_mass_fractions(thermo.library, comp, el.basis)
    h_t = enthalpy_mass(thermo.library, Y, Tt)
    return [h_t] + _onehot(stream, n_elem)


def _mass_source_params(thermo: ThermoConfig, el: ElementSpec, n_elem: int, label: str, stream: int):
    """Resolve a mass source's params ``[mdot_src, u_inj, h_t_src, xi_src_0, ...]``.

    The injected total enthalpy carries the stream's enthalpy at ``T_src`` plus the
    injection kinetic energy ``0.5 u_inj^2`` (D-1 datum); the injected composition is
    the mixture-fraction unit vector of its feed stream (kernel donor index
    ``pb+2+s``).
    """
    mdot_src = float(el.fparams[0])
    u_inj = float(el.fparams[1])
    T_src = float(el.fparams[2])
    ke = 0.5 * u_inj * u_inj
    if thermo.model_id == PERFECT_GAS:
        h_t_src = float(thermo.tf[0]) * T_src + ke
        if n_elem == 0:
            return [mdot_src, u_inj, h_t_src]
        comp = el.composition_spec
        zvals = [float(c) for c in comp] if comp is not None else [0.0] * n_elem
        if len(zvals) != n_elem:
            raise ValueError(f"{label} carries {len(zvals)} scalar(s) but the model has {n_elem}")
        return [mdot_src, u_inj, h_t_src] + zvals

    comp = el.composition_spec
    if comp is None:
        raise ValueError(
            f"{label}: a mass source must specify its injected species composition "
            f"(e.g. composition={{'CH4': 1.0}})"
        )
    Y = species_mass_fractions(thermo.library, comp, el.basis)
    h_t_src = enthalpy_mass(thermo.library, Y, T_src) + ke
    return [mdot_src, u_inj, h_t_src] + _onehot(stream, n_elem)


def _burnt_seed(conn: Connectivity, flame_nodes) -> np.ndarray:
    """Per-edge burnt-marker initial guess ``(0 fresh / 1 burnt)`` by a topology flood-fill.

    Seeds ``b = 1`` on every edge leaving an equilibrium flame (along the *declared* tail->head
    arrows) and floods it downstream.  This is only the marker transport's **initial guess** --
    the signed-mass-flow transport self-corrects a backward-drawn flame at convergence -- so its
    job is purely to warm the start (a correct drawing converges in one shot).
    """
    n_edges = int(conn.n_edges)
    tail = np.asarray(conn.tail_node)
    head = np.asarray(conn.head_node)
    out_edges = defaultdict(list)  # node -> outgoing edges (declared tail -> head)
    for e in range(n_edges):
        out_edges[int(tail[e])].append(e)
    burnt = np.zeros(n_edges, dtype=np.float64)
    stack = []
    for e in range(n_edges):  # seed: every edge leaving a flame is burnt
        if int(tail[e]) in flame_nodes:
            burnt[e] = 1.0
            stack.append(int(head[e]))
    while stack:  # flood downstream; each edge is marked at most once -> terminates on cycles
        for e in out_edges[stack.pop()]:
            if burnt[e] == 0.0:
                burnt[e] = 1.0
                stack.append(int(head[e]))
    return burnt


def finalize_thermo(thermo: ThermoConfig, elements: List[ElementSpec]):
    """Discover the network's feed streams and pack the equilibrium bundle.

    For the reacting (``EQ_KERNEL``) model the **streams are the distinct injected
    compositions** of the network's inlets, mass sources and (backflow-bearing)
    outlets.  This scans them in node order, auto-merges identical compositions, and
    packs the per-stream forward-blend maps -- so the user only ever names species at
    the elements that introduce them, and the transported scalar count equals the
    number of distinct feeds (never the chemical-element count, never the product
    species).

    Returns
    -------
    thermo : ThermoConfig
        The finalized config (unchanged for a perfect gas / passive-scalar model).
    node_stream : dict or None
        ``node -> stream index`` for every stream-introducing node (``-1`` if that
        element carries no composition, e.g. an inert-backflow outlet); ``None`` for
        a non-reacting model.
    """
    if thermo.model_id != EQ_KERNEL or thermo.library is None:
        return thermo, None

    from ..thermo.edge_state import pack_equilibrium

    comps = []
    comp_nodes = []
    for n, el in enumerate(elements):
        if el.residual_id in STREAM_INTRODUCING:
            comps.append((el.composition_spec, el.basis))
            comp_nodes.append(n)
    stream_Y, assignment = build_streams(thermo.library, comps)
    node_stream = {comp_nodes[i]: assignment[i] for i in range(len(comp_nodes))}

    # label each stream by the first element that introduces it (for reporting)
    K = stream_Y.shape[0]
    labels = [f"stream{k}" for k in range(K)]
    for i, k in enumerate(assignment):
        if k >= 0 and labels[k] == f"stream{k}":
            nm = elements[comp_nodes[i]].name
            if nm:
                labels[k] = nm

    tf, ti = pack_equilibrium(thermo.library, stream_Y, thermo.t_init, thermo.t_init_frozen)
    finalized = ThermoConfig(
        model_id=EQ_KERNEL,
        tf=tf,
        ti=ti,
        element_names=labels,
        species_names=thermo.species_names,
        library=thermo.library,
        t_init=thermo.t_init,
        t_init_frozen=thermo.t_init_frozen,
    )
    return finalized, node_stream


def build_problem(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    edges: List[Tuple[int, int, float]],
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
    edge_models=None,
    require_connected=None,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and directed ``(tail, head, area)`` edges.

    The lower-level functional builder; the user-facing path is
    :class:`nefes.shell.network.Network`, whose constructor accepts the same ``nodes`` /
    ``edges`` lists (and auto-derives the reference scales), then ``.solve()``.  Ports are
    auto-assigned in attachment order; use :func:`build_problem_from_connectivity` to supply
    explicit ports (e.g. a UI export where the port ordinals carry meaning).

    Parameters
    ----------
    thermo : ThermoConfig
        The thermodynamic model (perfect gas, frozen, or reacting equilibrium).
    elements : list of ElementSpec
        The network elements, in node order.
    edges : list of tuple
        Directed ``(tail, head, area)`` edges referencing node indices.
    mdot_ref, p_ref, h_ref : float
        Reference mass flow, pressure and total enthalpy for the residual/variable scaling.
    edge_models : sequence of int, optional
        Per-edge thermo-model id override aligned with ``edges``; ``None`` uses the config's
        model on every edge.
    require_connected : bool, optional
        Reject a model that splits into disconnected sub-networks; ``None`` (default) follows
        the :data:`nefes.shell.checks.CHECK_CONNECTED` toggle.

    Returns
    -------
    CompiledProblem
        The immutable compiled problem ready for the solver.
    """
    # expand any composite elements (build-time graph transform) into atomic elements +
    # internal edges; a composite-free network passes through unchanged (composite_map None).
    elements, edges, composite_map = expand_composites(elements, edges)
    n_nodes = len(elements)
    area = np.array([e[2] for e in edges], dtype=np.float64)
    if composite_map is not None:
        # the expansion pins explicit flow-aligned ports (5-tuples); honor them so each
        # sub-element's edges land on the right ports (port 0 in, port 1 out).
        conn = build_connectivity(n_nodes, [(t, tp, h, hp) for (t, h, _a, tp, hp) in edges])
    else:
        conn = build_connectivity(n_nodes, _assign_ports_by_kind(elements, edges))
    return build_problem_from_connectivity(
        thermo,
        elements,
        conn,
        area,
        mdot_ref,
        p_ref,
        h_ref,
        edge_models=edge_models,
        composite_map=composite_map,
        require_connected=require_connected,
    )


def build_problem_from_connectivity(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    conn: Connectivity,
    area: np.ndarray,
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
    edge_models=None,
    composite_map=None,
    require_connected=None,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and a prebuilt Connectivity.

    Parameters
    ----------
    thermo : ThermoConfig
        The thermodynamic model (perfect gas, frozen, or reacting equilibrium).
    elements : list of ElementSpec
        The network elements, in node order.
    conn : Connectivity
        The compiled connectivity, carrying explicit per-edge ports (``tail_port`` /
        ``head_port``), so port-ordering conventions are preserved exactly.
    area : ndarray
        Per-edge cross-sectional area, indexed by global edge id.
    mdot_ref, p_ref, h_ref : float
        Reference mass flow, pressure and total enthalpy for the residual/variable scaling.
    edge_models : sequence of int, optional
        Per-edge thermo-model id override; ``None`` uses the config's model on every edge.
    composite_map : CompositeMap, optional
        Bridges the user-facing element ids to the expanded ones when the network carried
        composite elements (set by :func:`build_problem`); ``None`` otherwise.
    require_connected : bool, optional
        Reject a model that splits into disconnected sub-networks; ``None`` (default) follows
        the :data:`nefes.shell.checks.CHECK_CONNECTED` toggle.

    Returns
    -------
    CompiledProblem
        The immutable compiled problem ready for the solver.
    """
    n_nodes = len(elements)
    area = np.ascontiguousarray(area, dtype=np.float64)
    validate_network(elements, conn, area, require_connected=require_connected)

    # discover the feed streams from the network and finalize the (reacting) thermo
    # bundle: the transported mixture fractions are the distinct injected compositions.
    thermo, node_stream = finalize_thermo(thermo, elements)

    degrees = [conn.degree(n) for n in range(n_nodes)]
    node_rid = np.array([el.residual_id for el in elements], dtype=np.int64)
    node_acoustic_id = np.array([el.acoustic_id for el in elements], dtype=np.int64)

    # Marker-gated reacting closure (the default/auto reacting path): a reacting network with at
    # least one equilibrium flame and no explicit per-edge override runs EQ_MARKER on every edge
    # and transports one extra "burnt" marker scalar (the last advected scalar) that gates the
    # frozen/equilibrium blend.  The marker rides the *signed* mass flow, so it labels "downstream
    # of a flame" robustly regardless of how the edges were drawn.  An explicit ``edge_models``
    # keeps the hard per-edge closure (EQ_FROZEN/EQ_KERNEL, no marker), the power-user escape hatch.
    flame_nodes = {n for n in range(n_nodes) if int(node_rid[n]) == FLAME_EQUILIBRIUM}
    marker_gated = thermo.model_id == EQ_KERNEL and bool(flame_nodes) and edge_models is None
    n_marker = 1 if marker_gated else 0

    # A user-set inflow marker only has a transport scalar to ride when the network is
    # marker-gated; reject a non-zero marker elsewhere rather than silently dropping it.
    if not marker_gated:
        stray = [el.name or f"node {n}" for n, el in enumerate(elements) if float(getattr(el, "marker", 0.0)) != 0.0]
        if stray:
            raise ValueError(
                "a non-zero burnt marker requires a marker-gated reacting network (an equilibrium-flame "
                "reacting model with no explicit per-edge closure); marker was set on: " + ", ".join(stray)
            )

    # pack node float params in node order.  A boundary element that prescribes
    # advected scalars carries [base, h_t, Z_el...]: slot 0 is the prescribed
    # mdot/pt/p, slot 1 the absolute total enthalpy datum (converted from Tt), and
    # the remaining n_elem slots the feed/backflow elemental composition -- so the
    # donor kernel indexes npar_f[pb + 1 + s] for advected scalar s (s = 0 is h_t).
    n_elem = thermo.n_elem
    # The burnt marker is the *last* advected scalar, so its donor param sits after the
    # composition.  A fresh feed/backflow enters with marker = 0 (the default); a boundary
    # may set ``marker = 1`` to inject already-burnt gas (e.g. exhaust-gas recirculation).
    # Appended only when the network is marker-gated.
    npar_f = []
    npar_fptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for n, el in enumerate(elements):
        fp = list(el.fparams)
        k = -1 if node_stream is None else node_stream.get(n, -1)
        marker_param = [float(el.marker)] if n_marker else []
        if el.residual_id in _SCALAR_BOUNDARY_RIDS and len(fp) >= 2:
            base, Tt = float(fp[0]), float(fp[1])
            fp = [base] + _boundary_scalars(thermo, el, Tt, n_elem, _node_label(n, el), k) + marker_param
        elif el.residual_id == MASS_SOURCE:
            fp = _mass_source_params(thermo, el, n_elem, _node_label(n, el), k) + marker_param
        npar_f.extend(fp)
        npar_fptr[n + 1] = npar_fptr[n] + len(fp)
    npar_f = np.array(npar_f, dtype=np.float64)

    # per-element smoothing-eps override (< 0 -> follow the global solve-time eps)
    node_eps = np.array([el.eps if el.eps is not None else -1.0 for el in elements], dtype=np.float64)

    # per-node perturbation BC (Python objects; read only by the perturbation layer)
    node_bc = tuple(getattr(el, "perturbation_bc", None) for el in elements)

    # per-node human-readable name (label); for plotting / reporting only
    node_names = tuple(getattr(el, "name", "") or "" for el in elements)

    # per-node dynamic-source descriptor (S(omega) provision; mean flow ignores it)
    node_dynamic_source = tuple(getattr(el, "dynamic_source", None) for el in elements)

    # per-node transfer-matrix descriptor (TRANSFER_MATRIX element; mean flow ignores it)
    node_transfer_matrix = tuple(getattr(el, "transfer_matrix", None) for el in elements)

    n_scalars = thermo.n_elem + n_marker  # composition mixture fractions + the optional burnt marker
    n_solve = 3 + n_scalars
    marker_row = (3 + thermo.n_elem) if marker_gated else -1  # the marker is the last band-1 row
    pat = build_jacobian_pattern(conn, degrees, n_solve=n_solve)

    # residual scales: node rows, then the advected-scalar transport rows (h_t for every edge,
    # then each composition scalar for every edge, then the marker for every edge).  Composition
    # mixture fractions and the marker are O(1) (in [0, 1]), so their scale = 1.
    z_scale = 1.0
    res_scale = []
    for n, el in enumerate(elements):
        res_scale.extend(_row_kinds(el.residual_id, degrees[n], mdot_ref, p_ref))
    res_scale.extend([h_ref] * conn.n_edges)
    for _ in range(n_scalars):
        res_scale.extend([z_scale] * conn.n_edges)
    res_scale = np.array(res_scale, dtype=np.float64)

    var_scale = np.array([mdot_ref, p_ref, h_ref] + [z_scale] * n_scalars, dtype=np.float64)

    # per-edge thermo model: marker-gated -> EQ_MARKER everywhere; explicit override -> verbatim;
    # otherwise the config's model on every edge.
    if edge_models is None:
        edge_model = np.full(conn.n_edges, EQ_MARKER if marker_gated else thermo.model_id, dtype=np.int64)
    else:
        edge_model = np.ascontiguousarray(edge_models, dtype=np.int64)
        if edge_model.shape[0] != conn.n_edges:
            raise ValueError(f"edge_models has {edge_model.shape[0]} entries but the network has {conn.n_edges} edges")

    # an unburnt (EQ_FROZEN) or marker-gated (EQ_MARKER, which runs the frozen leg) edge
    # reconstructs species from the feed streams; at least one stream must exist (an inlet /
    # mass source must inject a composition).
    if thermo.model_id != PERFECT_GAS and np.any((edge_model == EQ_FROZEN) | (edge_model == EQ_MARKER)):
        n_streams = int(thermo.ti[5]) if thermo.ti.shape[0] > 5 else 0
        if n_streams == 0:
            raise ValueError(
                "the network has frozen / marker-gated (unburnt-capable) edges but no feed streams "
                "were found; an inlet or mass source must inject an explicit species composition "
                "for the frozen closure to reconstruct from"
            )

    # burnt-marker initial guess by a topology flood-fill (b = 1 downstream of a flame along the
    # declared arrows, b = 0 elsewhere); only warms the start, the transport self-corrects a
    # backward-drawn flame at convergence.  None when the network carries no marker.
    marker_seed = _burnt_seed(conn, flame_nodes) if marker_gated else None

    return CompiledProblem(
        model_id=thermo.model_id,
        tf=thermo.tf,
        ti=thermo.ti,
        n_elem=thermo.n_elem,
        n_solve=n_solve,
        n_nodes=n_nodes,
        n_edges=conn.n_edges,
        n_eq=pat.n_eq,
        area=area,
        row_ptr=conn.row_ptr,
        col_edge=conn.col_edge,
        orient=conn.orient.astype(np.int64),
        tail_node=conn.tail_node,
        head_node=conn.head_node,
        node_rid=node_rid,
        node_acoustic_id=node_acoustic_id,
        npar_f=npar_f,
        npar_fptr=npar_fptr,
        node_row_ptr=pat.node_row_ptr,
        transport_row0=pat.transport_row0,
        indptr=pat.indptr,
        indices=pat.indices,
        var_scale=var_scale,
        res_scale=res_scale,
        edge_model=edge_model,
        node_eps=node_eps,
        node_bc=node_bc,
        node_names=node_names,
        node_dynamic_source=node_dynamic_source,
        node_transfer_matrix=node_transfer_matrix,
        scalar_names=tuple(thermo.element_names),
        marker_row=marker_row,
        marker_seed=marker_seed,
        composite_map=composite_map,
    )
