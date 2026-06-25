"""Parse the node-graph UI export (YAML) into FNS connectivity.

The UI export carries each node's integer ``index`` and each edge's
``source``/``target`` node ids plus ``sourceHandle``/``targetHandle`` of the form
``...-port-<k>``.  ``load_connectivity`` resolves these into the edge-endpoint
table (CSC view), ordered by the edge's own ``index``.
"""

import re
from typing import List, Tuple

import yaml

from ..connectivity import build_connectivity, Connectivity
from ..elements import catalog as cat
from ..thermo.configure import perfect_gas
from .provenance import UIProvenance

_PORT_RE = re.compile(r"port-(\d+)$")

# The UI model id this loader targets (public/models/fns-flow-network.yaml).
MODEL_ID = "fns-flow-network"

# Model-level (globalAttributes) defaults, kept in sync with the UI model.
_GLOBAL_DEFAULTS = {
    "gasConstant": 287.0,
    "heatCapacityRatio": 1.4,
    "referencePressure": 101325.0,
    "referenceTemperature": 300.0,
    "referenceMassFlow": 0.0,
}

# UI node type -> (catalog factory, kwargs extracted from node attributes).
# Each builder takes the node's `attributes` dict and returns an ElementSpec.
_UI_NODE_BUILDERS = {
    "MassFlowInlet": lambda a: cat.mass_flow_inlet(a["massFlowRate"], a["totalTemperature"]),
    "TotalPressureInlet": lambda a: cat.total_pressure_inlet(a["totalPressure"], a["totalTemperature"]),
    "PressureOutlet": lambda a: cat.pressure_outlet(a["pressure"], a.get("backflowTotalTemperature", 300.0)),
    "MassFlowOutlet": lambda a: cat.mass_flow_outlet(a["massFlowRate"]),
    "ChokedNozzleOutlet": lambda a: cat.choked_nozzle_outlet(a["throatArea"]),
    "Wall": lambda a: cat.wall(),
    "IsentropicAreaChange": lambda a: cat.isentropic_area_change(),
    "SuddenAreaChange": lambda a: cat.sudden_area_change(cc=a.get("contractionCoefficient", 1.0)),
    "LossElement": lambda a: cat.loss(a["lossCoefficient"]),
    "Duct": lambda a: cat.duct(a.get("length", 0.0)),
    "JunctionStaticP": lambda a: cat.junction(),
    "LosslessSplitter": lambda a: cat.splitter(),
}

# Boundary types that carry a perturbation BC group in the UI schema.
_BOUNDARY_TYPES = {
    "MassFlowInlet",
    "TotalPressureInlet",
    "PressureOutlet",
    "MassFlowOutlet",
    "ChokedNozzleOutlet",
    "Wall",
}

_DEFERRED_TYPES = {"SupersonicInlet", "SupersonicOutlet"}


def _parse_perturbation_bc(attrs: dict):
    """Build a ``PerturbationBC`` from a boundary node's UI acoustic attributes.

    The UI exposes a deliberately small surface: a single ``boundaryType`` dropdown
    selecting ``"inherit"`` (the element's natural closure -- its linearized mean
    boundary row, e.g. ``mdot'=0`` for a mass-flow outlet or the compact choked-nozzle
    reflection for a choked-nozzle outlet), ``"rigid"`` (an infinite impedance / hard
    wall, ``u'=0``), ``"open"`` (an ideal pressure-release open end, ``p'=0``) or
    ``"impedance"`` (a specific acoustic impedance given by ``impedanceMagnitude``
    |Z|/rho c and ``impedancePhase`` in degrees).  Returns ``None`` for ``"inherit"`` or
    when no ``boundaryType`` is present, so the element keeps its default closure
    (``inherit`` for inlets/outlets; a hard wall for the wall element).  Richer closures
    (reflection coefficients, excitation, mean-flow open end, frequency tables) are set
    directly in Python via ``PerturbationBC``.
    """
    from ..perturbation.boundary_bc import PerturbationBC

    btype = attrs.get("boundaryType")
    if btype == "inherit":
        # the element's natural closure: its linearized mean boundary row (e.g. mdot' = 0 for
        # a mass-flow outlet, the compact choked-nozzle reflection for a choked-nozzle outlet).
        return None
    if btype is None:
        return None  # no acoustic field -> keep the element's default closure
    if btype == "rigid":
        return PerturbationBC.hard_wall()
    if btype == "open":
        return PerturbationBC.open_end()
    if btype == "impedance":
        magnitude = float(attrs.get("impedanceMagnitude", 1.0))
        phase_deg = float(attrs.get("impedancePhase", 0.0))
        return PerturbationBC.impedance_polar(magnitude, phase_deg, specific=True)
    raise ValueError(f"unknown boundaryType {btype!r} on a boundary node")


def _port_of(handle: str) -> int:
    m = _PORT_RE.search(handle)
    if not m:
        raise ValueError(f"cannot parse port from handle {handle!r}")
    return int(m.group(1))


def parse_endpoints(doc: dict) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """Return ``(n_nodes, endpoints)`` from a parsed UI-export document."""
    model = doc["model"]
    id_to_index = {}
    for node in model["nodes"]:
        id_to_index[node["id"]] = int(node["attributes"]["index"])
    n_nodes = len(id_to_index)

    rows = []
    for edge in model["edges"]:
        e_index = int(edge["attributes"]["index"])
        tn = id_to_index[edge["source"]]
        hn = id_to_index[edge["target"]]
        tp = _port_of(edge["sourceHandle"])
        hp = _port_of(edge["targetHandle"])
        rows.append((e_index, tn, tp, hn, hp))

    rows.sort(key=lambda r: r[0])
    endpoints = [(tn, tp, hn, hp) for (_e, tn, tp, hn, hp) in rows]
    return n_nodes, endpoints


def load_connectivity(path: str) -> Connectivity:
    """Load a UI-export YAML file and build its Connectivity."""
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    n_nodes, endpoints = parse_endpoints(doc)
    return build_connectivity(n_nodes, endpoints)


def _build_ui_spec(node: dict):
    ntype = node.get("type")
    attrs = node.get("attributes") or {}
    if ntype in _DEFERRED_TYPES:
        raise ValueError(f"element type {ntype!r} is deferred in v1 (subsonic scope); remove it from the case")
    try:
        builder = _UI_NODE_BUILDERS[ntype]
    except KeyError:
        raise ValueError(f"unknown FNS element type {ntype!r}")
    spec = builder(attrs)
    spec.name = str(attrs.get("label") or node.get("id") or ntype)
    if ntype in _BOUNDARY_TYPES:
        bc = _parse_perturbation_bc(attrs)
        if bc is not None:  # else keep the factory default (None=inherit; Wall=hard wall)
            spec.perturbation_bc = bc
    return spec


def load_case(path: str):
    """Load a UI-exported FNS case (``model.id == fns-flow-network``) into a ``Network``.

    Reads the native YAML the FNetLibUI tool writes out: ``model.globalAttributes``
    (gas + reference scales), ``model.nodes`` (element ``type`` + ``attributes``),
    and ``model.edges`` (``source``/``target`` nodes, ``sourceHandle``/
    ``targetHandle`` port ordinals, and ``attributes.area``).

    Ports are preserved exactly: each edge's handle ordinals are kept, and each
    element's incident ports are ordered by ordinal and densified to ``0..d-1``,
    so port-0 conventions (loss reference area, junction/splitter reference port)
    match the canvas.
    """
    from ..shell import Network  # local import to avoid an import cycle

    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    model = doc.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path}: not a UI save file (no 'model' section)")
    mid = model.get("id")
    if mid not in (None, MODEL_ID):
        raise ValueError(f"{path}: case targets model {mid!r}, expected {MODEL_ID!r}")

    g = dict(_GLOBAL_DEFAULTS)
    g.update({k: v for k, v in (model.get("globalAttributes") or {}).items() if v is not None})
    gas = perfect_gas(R=float(g["gasConstant"]), gamma=float(g["heatCapacityRatio"]))
    net = Network(
        gas,
        p_ref=float(g["referencePressure"]),
        T_ref=float(g["referenceTemperature"]),
        mdot_ref=float(g["referenceMassFlow"]) or None,
    )

    ui_nodes = model.get("nodes") or []
    ui_edges = model.get("edges") or []
    if not ui_nodes or not ui_edges:
        raise ValueError(f"{path}: the network has no nodes or no edges")

    # Elements, ordered by the UI node index so fns indices match the canvas.
    nodes_sorted = sorted(ui_nodes, key=lambda n: int((n.get("attributes") or {}).get("index", 0)))
    id_to_index = {}
    for n in nodes_sorted:
        idx = net.add(_build_ui_spec(n))
        id_to_index[n["id"]] = idx

    # Per node, gather incident (edge, side, port ordinal); densify by ordinal.
    edges_sorted = sorted(ui_edges, key=lambda e: int((e.get("attributes") or {}).get("index", 0)))
    incident = {i: [] for i in range(len(nodes_sorted))}
    parsed = []
    for ei, e in enumerate(edges_sorted):
        attrs = e.get("attributes") or {}
        for end in ("source", "target"):
            if e.get(end) not in id_to_index:
                raise ValueError(f"edge {e.get('id')!r} references unknown node {e.get(end)!r}")
        s = id_to_index[e["source"]]
        t = id_to_index[e["target"]]
        so = _port_of(e["sourceHandle"])
        to = _port_of(e["targetHandle"])
        area = float(attrs["area"])
        if area <= 0.0:
            raise ValueError(f"edge {e.get('id')!r} has non-positive area {area}")
        incident[s].append((ei, "tail", so))
        incident[t].append((ei, "head", to))
        parsed.append((ei, s, t, area, str(attrs.get("label") or e.get("id"))))

    local_port = {}  # (edge_index, side) -> dense local port at that node
    for node, lst in incident.items():
        for local, (ei, side, _ord) in enumerate(sorted(lst, key=lambda x: x[2])):
            local_port[(ei, side)] = local

    for ei, s, t, area, name in parsed:
        net.connect(s, t, area, name=name, tail_port=local_port[(ei, "tail")], head_port=local_port[(ei, "head")])

    # Retain the UI-only metadata (positions, counters, ids, title) so the case
    # can be saved back for the UI verbatim -- see fns.io.yaml_out.
    net.provenance = UIProvenance(
        doc=doc,
        node_ids=[n["id"] for n in nodes_sorted],
        edge_ids=[e.get("id", f"edge_{i + 1}") for i, e in enumerate(edges_sorted)],
    )
    return net
