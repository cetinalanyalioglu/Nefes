"""Element catalog and CompiledProblem builder (Python, parse-time).

An ``ElementSpec`` names an element's residual id and its ordered float
parameters (the order the @njit kernels expect).  ``build_problem`` turns a list
of element specs plus directed edges into the immutable CompiledProblem.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..connectivity import connectivity_from_directed_edges, build_jacobian_pattern, Connectivity
from ..problem import CompiledProblem
from ..thermo.configure import ThermoConfig
from .ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    WALL,
    JUNCTION,
    SPLITTER,
    ACOUSTIC_DEFAULT,
    ACOUSTIC_DUCT,
    FIXED_NPORTS,
    ALLOWS_AREA_CHANGE,
    RESIDUAL_NAMES,
)

# Relative tolerance for the equal-area check on constant-area elements.
_AREA_RTOL = 1e-9


@dataclass
class ElementSpec:
    """One network element: residual type + ordered float parameters.

    ``acoustic_id`` (implementation-plan.md s8.3) declares the optional acoustic
    face that overrides the default CSD linearization; ``ACOUSTIC_DEFAULT`` means
    the element contributes only through ``J_alg``.

    ``eps`` optionally overrides this element's smoothing width (the smooth-step /
    complementarity regularization, in mass-flow units, i.e. ~ a fraction of
    ``mdot_ref``).  ``None`` follows the global solve-time ``eps``.  Settable at
    creation or mutated later (``spec.eps = ...``) before ``build_problem``; a
    sharper value makes the frozen perturbation linearization track the exact
    (un-regularized) jump -- see the ``SUDDEN_AREA_CHANGE`` note in ``kernels.py``.
    """

    residual_id: int
    fparams: List[float] = field(default_factory=list)
    name: str = ""
    acoustic_id: int = ACOUSTIC_DEFAULT
    eps: Optional[float] = None
    perturbation_bc: Optional[object] = None  # PerturbationBC (None -> inherit)


def mass_flow_inlet(mdot, Tt, name="inlet", perturbation_bc=None):
    return ElementSpec(MASS_FLOW_INLET, [float(mdot), float(Tt)], name, perturbation_bc=perturbation_bc)


def total_pressure_inlet(pt, Tt, name="pt-inlet", perturbation_bc=None):
    return ElementSpec(PT_INLET, [float(pt), float(Tt)], name, perturbation_bc=perturbation_bc)


def pressure_outlet(p, Tt_backflow=300.0, name="outlet", perturbation_bc=None):
    return ElementSpec(P_OUTLET, [float(p), float(Tt_backflow)], name, perturbation_bc=perturbation_bc)


def wall(name="wall", perturbation_bc=None):
    """An impermeable single-port termination: ``mdot = 0`` on its incident edge.

    The wall blocks mean flow, so the leg behind it is stagnant (``M = 0``); its
    purpose is acoustic.  By default it closes the perturbation problem as a rigid
    hard wall (``u' = 0``, ``R = +1``) -- which at the wall's ``M = 0`` state is
    identical to the inherited ``mdot' = 0`` row.  Pass ``perturbation_bc`` to model
    a non-rigid termination (e.g. a liner impedance) instead.
    """
    from ..perturbation.boundary_bc import PerturbationBC

    bc = perturbation_bc if perturbation_bc is not None else PerturbationBC.hard_wall()
    return ElementSpec(WALL, [], name, perturbation_bc=bc)


def isentropic_area_change(name="iac"):
    from .ids import ISEN_AREA_CHANGE

    return ElementSpec(ISEN_AREA_CHANGE, [], name)


def sudden_area_change(name="sac", cc=1.0, eps=None):
    """Sudden area change: Borda-Carnot expansion, vena-contracta contraction.

    Forward flow (small -> large) follows the Borda-Carnot momentum balance
    (separation at the step, mixing loss).  Reverse flow (large -> small, a
    contraction) follows a vena-contracta total-pressure loss
    ``K_c * (1/2 rho u^2)_small`` with ``K_c = (1/cc - 1)^2``, referenced to the
    downstream (small-port) dynamic head.  The small/large sides are identified
    from the attached edge areas, so ``cc`` always acts on whichever direction is
    contracting.

    Parameters
    ----------
    name : str, optional
        Element label.
    cc : float, optional
        Vena-contracta contraction coefficient for the reverse (contracting)
        flow, in ``(0, 1]``.  ``cc = 1`` (default) is the loss-free contraction:
        the reverse branch reduces to exact total-pressure continuity (the
        historical behaviour).  Use a tabulated value for the geometry (e.g.
        ~0.62 for a sharp-edged contraction at a small area ratio; Weisbach /
        Idelchik).  Forward (expanding) flow is unaffected by ``cc``.

        The loss uses the incompressible ``1/2 rho u^2`` head, so it is accurate
        only to ``O(M^2)``; a dedicated contraction element resolving the vena-
        contracta state (exact at higher Mach) is planned.
    eps : float, optional
        Optionally sharpens this element's momentum<->contraction switch (see
        ``ElementSpec.eps``); use a small value (e.g. ``1e-6 * mdot_ref``) when
        the flow is firmly one-directional and an accurate perturbation jump is
        wanted.
    """
    from .ids import SUDDEN_AREA_CHANGE

    cc = float(cc)
    if not 0.0 < cc <= 1.0:
        raise ValueError(f"sudden_area_change: contraction coefficient cc must be in (0, 1]; got {cc}")
    return ElementSpec(SUDDEN_AREA_CHANGE, [cc], name, eps=eps)


def loss(K, name="loss", ref_port=0, eps=None):
    """A concentrated total-pressure loss ``Pt_in - Pt_out = K * (1/2 rho u^2)``.

    The element conserves mass and drops total pressure by ``K`` dynamic heads,
    with the head's sign tracking the flow direction so reverse flow reverses the
    drop (modeling-guide.md s4).  The static state on each port is reconstructed
    from that port's own area, so the loss may straddle an area change: the result
    is an isentropic area change (Pt-preserving static<->dynamic conversion) with
    the concentrated ``K``-loss superposed.

    Parameters
    ----------
    K : float
        Loss coefficient, referenced to the dynamic head at port ``ref_port``.
    name : str, optional
        Element name.
    ref_port : int, optional
        Which incident port's area and velocity define the reference dynamic head
        ``1/2 rho u^2`` that ``K`` multiplies -- ``0`` (default, the upstream edge
        in the canonical orientation) or ``1``.  Only matters when the ports carry
        different areas; tabulated ``K`` values always name their reference
        section, so set this to match the source.
    eps : float, optional
        Per-element smoothing-width override (see ``ElementSpec.eps``).
    """
    from .ids import LOSS

    rp = int(ref_port)
    if rp not in (0, 1):
        raise ValueError(f"loss: ref_port must be 0 or 1; got {ref_port}")
    return ElementSpec(LOSS, [float(K), float(rp)], name, eps=eps)


def junction(name="junction"):
    return ElementSpec(JUNCTION, [], name)


def splitter(name="splitter"):
    return ElementSpec(SPLITTER, [], name)


def duct(length=0.0, name="duct"):
    """A length-bearing, lossless, constant-area duct.

    The mean face is equal-area continuity (length-independent); ``length`` is
    inert in the steady residual and read only by the acoustic phase stamp
    (theory.md s12.3).  It rides ``fparams[0]`` as ordinary acoustic metadata.
    """
    from .ids import DUCT

    return ElementSpec(DUCT, [float(length)], name, acoustic_id=ACOUSTIC_DUCT)


def _node_label(n: int, el: ElementSpec) -> str:
    """Human-readable identifier for an element, for validation messages."""
    typ = RESIDUAL_NAMES.get(el.residual_id, f"residual {el.residual_id}")
    name = f" {el.name!r}" if el.name else ""
    return f"element {n}{name} ({typ})"


def validate_network(elements: List[ElementSpec], conn: Connectivity, area: np.ndarray) -> None:
    """Check structural and area-consistency invariants before compiling.

    Raises ``ValueError`` (naming the offending element) on the first violation:

    * every edge area is finite and strictly positive;
    * each element's port count matches its arity -- exactly ``FIXED_NPORTS`` for
      fixed-arity elements, ``>= 2`` for the variable junction/splitter;
    * elements that do not permit an area change (``ALLOWS_AREA_CHANGE`` is
      ``False`` -- the constant-area duct) carry one shared area across all their
      incident edges.  An intended area change at an area-agnostic element (e.g. a
      sudden expansion) must use an ``isentropic_area_change`` or
      ``sudden_area_change`` element.

    Parameters
    ----------
    elements : list of ElementSpec
        The network elements, in node order.
    conn : Connectivity
        The compiled connectivity (per-node incident edges and degrees).
    area : ndarray
        Per-edge cross-sectional area, indexed by global edge id.
    """
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


def _row_kinds(rid: int, deg: int, mdot_ref, p_ref):
    """Residual-row scale magnitudes for one element."""
    if rid == MASS_FLOW_INLET or rid == WALL:
        return [mdot_ref]  # WALL pins mdot = 0
    if rid in (PT_INLET, P_OUTLET):
        return [p_ref]
    # interior: mass balance + (deg-1) pressure rows
    return [mdot_ref] + [p_ref] * (deg - 1)


def build_problem(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    edges: List[Tuple[int, int, float]],
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and directed (tail, head, area) edges.

    Ports are auto-assigned in attachment order.  Use
    ``build_problem_from_connectivity`` to supply explicit ports (e.g. a UI
    export where the port ordinals carry meaning).
    """
    n_nodes = len(elements)
    directed = [(t, h) for (t, h, _a) in edges]
    area = np.array([a for (_t, _h, a) in edges], dtype=np.float64)
    conn = connectivity_from_directed_edges(n_nodes, directed)
    return build_problem_from_connectivity(thermo, elements, conn, area, mdot_ref, p_ref, h_ref)


def build_problem_from_connectivity(
    thermo: ThermoConfig,
    elements: List[ElementSpec],
    conn: Connectivity,
    area: np.ndarray,
    mdot_ref: float,
    p_ref: float,
    h_ref: float,
) -> CompiledProblem:
    """Assemble a CompiledProblem from elements and a prebuilt Connectivity.

    The connectivity carries explicit per-edge ports (``tail_port``/
    ``head_port``), so port-ordering conventions are preserved exactly.
    """
    n_nodes = len(elements)
    area = np.ascontiguousarray(area, dtype=np.float64)
    validate_network(elements, conn, area)

    degrees = [conn.degree(n) for n in range(n_nodes)]
    node_rid = np.array([el.residual_id for el in elements], dtype=np.int64)
    node_acoustic_id = np.array([el.acoustic_id for el in elements], dtype=np.int64)

    # pack node float params in node order
    npar_f = []
    npar_fptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for n, el in enumerate(elements):
        npar_f.extend(el.fparams)
        npar_fptr[n + 1] = npar_fptr[n] + len(el.fparams)
    npar_f = np.array(npar_f, dtype=np.float64)

    # per-element smoothing-eps override (< 0 -> follow the global solve-time eps)
    node_eps = np.array([el.eps if el.eps is not None else -1.0 for el in elements], dtype=np.float64)

    # per-node perturbation BC (Python objects; read only by the perturbation layer)
    node_bc = tuple(getattr(el, "perturbation_bc", None) for el in elements)

    # per-node human-readable name (label); for plotting / reporting only
    node_names = tuple(getattr(el, "name", "") or "" for el in elements)

    pat = build_jacobian_pattern(conn, degrees, n_solve=3)

    # residual scales
    res_scale = []
    for n, el in enumerate(elements):
        res_scale.extend(_row_kinds(el.residual_id, degrees[n], mdot_ref, p_ref))
    res_scale.extend([h_ref] * conn.n_edges)
    res_scale = np.array(res_scale, dtype=np.float64)

    var_scale = np.array([mdot_ref, p_ref, h_ref], dtype=np.float64)

    return CompiledProblem(
        model_id=thermo.model_id,
        tf=thermo.tf,
        ti=thermo.ti,
        n_elem=thermo.n_elem,
        n_solve=3 + thermo.n_elem,
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
        node_eps=node_eps,
        node_bc=node_bc,
        node_names=node_names,
    )
