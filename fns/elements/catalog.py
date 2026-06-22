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
)


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


def sudden_area_change(name="sac", eps=None):
    """Borda/isentropic sudden area change.

    ``eps`` optionally sharpens this element's momentum<->isentropic switch (see
    ``ElementSpec.eps``); use a small value (e.g. ``1e-6 * mdot_ref``) when the
    flow is firmly one-directional and an accurate perturbation jump is wanted.
    """
    from .ids import SUDDEN_AREA_CHANGE

    return ElementSpec(SUDDEN_AREA_CHANGE, [], name, eps=eps)


def loss(K, name="loss", eps=None):
    from .ids import LOSS

    return ElementSpec(LOSS, [float(K)], name, eps=eps)


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
