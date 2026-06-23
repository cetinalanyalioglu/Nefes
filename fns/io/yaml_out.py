"""Write an FNS network (and its results) as a UI-readable YAML case.

This is the symmetric counterpart of :mod:`fns.io.yaml_in`.  It emits the native
``SaveFilePayload`` the FNetLibUI tool reads: ``version``/``timestamp``/``meta``,
the ``model`` (gas + reference scales, ``nodes``, ``edges``), the UI-only
``uiAttributes`` (canvas positions) and ``uiState`` (id counters), and an
optional ``data`` section holding result datasets.

Result data
-----------
The UI binds a dataset's ``values[i]`` to the element whose ``index`` is ``i``,
and validates that an item supplies exactly one value per element of its target
(``node`` or ``edge``).  FNS mean-flow state lives on edges, so the converged
:class:`~fns.shell.network.Solution` fields are emitted as **edge** datasets.
Pass ``node_data=True`` to additionally emit each (non-phase) edge field reduced
onto the nodes as the mean over each node's incident edges -- a provision until
genuinely nodal state exists.  Forced-response (acoustic) results are emitted as
one named dataset per selected frequency (``"<f> Hz"``), each carrying the
magnitude and phase of the requested perturbation quantities.

Layout / identity
-----------------
When the network was loaded from a UI file (``network.provenance`` is set and the
topology is unchanged) the node/edge ids, port handles, canvas positions, id
counters and title are reused verbatim, while the physical parameters are
refreshed from the live network.  Otherwise a fresh layered left-to-right layout
and synthetic ids are generated.
"""

import cmath
import copy
import math
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np
import yaml

from ..elements.ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    ISEN_AREA_CHANGE,
    SUDDEN_AREA_CHANGE,
    LOSS,
    JUNCTION,
    SPLITTER,
    DUCT,
    WALL,
)
from .yaml_in import MODEL_ID

# The UI save-file schema version this writer targets (matches yaml_in / the UI).
SAVE_FILE_VERSION = "2.0.0"
DEFAULT_TITLE = "FNS case"

# Single-port boundary elements that carry an acoustic boundary-condition group.
_BOUNDARY_RIDS = (MASS_FLOW_INLET, PT_INLET, P_OUTLET, WALL)

# Mean-flow edge field -> (UI display name, unit).  Order defines the "all" set.
_FIELD_META = {
    "mdot": ("Mass flow", "kg/s"),
    "p": ("Static pressure", "Pa"),
    "T": ("Static temperature", "K"),
    "M": ("Mach number", ""),
    "p_t": ("Total pressure", "Pa"),
    "rho": ("Density", "kg/m^3"),
    "u": ("Velocity", "m/s"),
    "c": ("Speed of sound", "m/s"),
    "h_t": ("Total enthalpy", "J/kg"),
    "area": ("Area", "m^2"),
}

# Forced-response field key -> (display label, ForcedResponse basis, component, unit).
# Each emits a magnitude item and a phase item.  "reflection" is handled specially.
_FORCED_FIELDS = {
    "p": ("Pressure", "network", 1, "Pa"),
    "u": ("Velocity", "primitive", 1, "m/s"),
    "mdot": ("Mass flow", "network", 0, "kg/s"),
}


# --------------------------------------------------------------------------- #
# Dataset value objects
# --------------------------------------------------------------------------- #
@dataclass
class MetaEntry:
    """One self-describing dataset-metadata field, rendered read-only by the UI.

    Mirrors the display fields of a model parameter so the UI can show it
    generically -- as ``label : value unit`` with an optional description --
    without hardcoding any keys (the UI never needs to know the model).

    Attributes
    ----------
    key : str
        Stable machine key (not displayed; for the UI to track entries).
    label : str
        Human-readable label shown in the UI.
    value : object
        The value (number, bool or string).
    unit : str, optional
        Unit string shown after the value.
    description : str, optional
        Longer note, surfaced behind an info affordance (Markdown/LaTeX).
    """

    key: str
    label: str
    value: object
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        out = {"key": self.key, "label": self.label, "value": _py_scalar(self.value)}
        if self.unit:
            out["unit"] = self.unit
        if self.description:
            out["description"] = self.description
        return out


@dataclass
class DataItem:
    """One result series: a value per element of ``target`` (``node``/``edge``).

    Attributes
    ----------
    name : str
        UI display name.
    target : str
        ``"node"`` or ``"edge"``; selects which elements the values color.
    values : list of float
        One value per element, ordered by element index.
    unit : str, optional
        Unit string shown in the UI legend (metadata only; no conversion).
    phase : bool, optional
        Marks a phase (degrees) series, excluded from node interpolation
        (a plain mean of wrapped phases is meaningless).
    """

    name: str
    target: str
    values: List[float]
    unit: str = ""
    phase: bool = False

    def to_dict(self, item_id: str) -> dict:
        out = {"id": item_id, "name": self.name, "target": self.target}
        if self.unit:
            out["unit"] = self.unit
        out["values"] = [float(v) for v in self.values]
        return out


@dataclass
class DataSet:
    """A named group of :class:`DataItem` s, embedded under ``data.datasets``.

    Carries optional self-describing metadata (:attr:`description` and an ordered
    list of :class:`MetaEntry` :attr:`info`) the UI renders read-only.
    """

    name: str
    items: List[DataItem] = field(default_factory=list)
    include_in_save: bool = True
    description: str = ""
    info: List[MetaEntry] = field(default_factory=list)

    def to_dict(self, ds_id: str) -> dict:
        out = {"id": ds_id, "name": self.name, "includeInSave": bool(self.include_in_save)}
        if self.description:
            out["description"] = self.description
        if self.info:
            out["info"] = [e.to_dict() for e in self.info]
        out["items"] = [it.to_dict(f"{ds_id}-item-{i}") for i, it in enumerate(self.items)]
        return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def dump_case(
    network,
    *,
    solution=None,
    fields="all",
    node_data=False,
    forced=None,
    forced_freqs=None,
    forced_fields=("p", "u"),
    title=None,
    extra_datasets=None,
    include_in_save=True,
) -> str:
    """Serialize a network (and optional results) to a UI-readable YAML string.

    Parameters
    ----------
    network : Network
        The flow network to write.
    solution : Solution, optional
        A converged mean-flow solution; when given, its result fields are
        embedded as a ``"Mean flow"`` dataset.
    fields : {"all", None} or sequence of str, optional
        Which mean-flow edge fields to include (default ``"all"``).  Names are
        the keys of :data:`_FIELD_META` (``mdot``, ``p``, ``T``, ``M``, ...).
        Ignored when ``solution`` is ``None``.
    node_data : bool, optional
        Also emit each non-phase edge field reduced onto nodes (mean over each
        node's incident edges).  Default ``False``.
    forced : ForcedResponse or list of ForcedResponse, optional
        Forced-response result(s) to snapshot as per-frequency datasets.
    forced_freqs : sequence of float, optional
        Frequencies (Hz) to snapshot from ``forced``; default is every frequency
        present.  Each must match a solved frequency.
    forced_fields : sequence of str, optional
        Perturbation quantities per snapshot (default ``("p", "u")``); keys of
        :data:`_FORCED_FIELDS`, plus ``"reflection"``.  Magnitude and phase are
        emitted for each.
    title : str, optional
        Case title (``meta.title``).  Defaults to the loaded title or
        ``"FNS case"``.
    extra_datasets : list of DataSet, optional
        Fully custom datasets appended as-is (e.g. genuinely nodal data the user
        supplies directly).
    include_in_save : bool, optional
        The ``includeInSave`` flag stamped on generated datasets (default
        ``True``).

    Returns
    -------
    str
        The YAML document.
    """
    datasets = _build_datasets(
        network, solution, fields, node_data, forced, forced_freqs, forced_fields, include_in_save
    )
    if extra_datasets:
        datasets = list(datasets) + list(extra_datasets)
    payload = build_payload(network, datasets, title)
    return _yaml_dump(payload)


def save_case(network, path: str, **kwargs) -> None:
    """Write a network (and optional results) to a UI-readable YAML file.

    Accepts the same keyword arguments as :func:`dump_case`.

    Parameters
    ----------
    network : Network
        The flow network to write.
    path : str
        Destination ``.yaml`` path.
    **kwargs
        Forwarded to :func:`dump_case`.
    """
    text = dump_case(network, **kwargs)
    with open(path, "w") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _build_datasets(network, solution, fields, node_data, forced, forced_freqs, forced_fields, include_in_save):
    datasets = []
    if solution is not None:
        items = []
        for name in _select_fields(fields):
            label, unit = _FIELD_META[name]
            items.append(DataItem(label, "edge", [float(v) for v in solution.field(name)], unit))
        if node_data:
            items += _node_items(network, items)
        info = [
            MetaEntry("kind", "Analysis", "Mean flow"),
            MetaEntry("converged", "Converged", bool(solution.converged)),
        ]
        datasets.append(DataSet("Mean flow", items, include_in_save, info=info))
    if forced is not None:
        datasets += _forced_datasets(network, forced, forced_freqs, forced_fields, node_data, include_in_save)
    return datasets


def _select_fields(fields):
    if fields is None or (isinstance(fields, str) and fields.lower() == "all"):
        return list(_FIELD_META.keys())
    if isinstance(fields, str):
        fields = [fields]
    selected = []
    for name in fields:
        if name not in _FIELD_META:
            raise ValueError(f"unknown mean-flow field {name!r}; choose from {list(_FIELD_META)}")
        selected.append(name)
    return selected


def _forced_datasets(network, forced, forced_freqs, forced_fields, node_data, include_in_save):
    responses = forced if isinstance(forced, (list, tuple)) else [forced]
    n_edges = len(network._edges)
    datasets = []
    for fr in responses:
        freqs = np.asarray(fr.freqs, dtype=float)
        for j in _select_freq_indices(freqs, forced_freqs):
            items = []
            for key in forced_fields:
                items += _forced_items(fr, j, key, n_edges)
            if node_data:
                items += _node_items(network, items)
            info = [
                MetaEntry("kind", "Analysis", "Forced response"),
                MetaEntry("frequency", "Frequency", float(freqs[j]), unit="Hz"),
            ]
            datasets.append(DataSet(f"{float(freqs[j]):g} Hz", items, include_in_save, info=info))
    return datasets


def _select_freq_indices(freqs, forced_freqs):
    if forced_freqs is None:
        return list(range(len(freqs)))
    if freqs.size == 0:
        raise ValueError("the forced response carries no frequencies")
    idxs = []
    for f in forced_freqs:
        j = int(np.argmin(np.abs(freqs - float(f))))
        if not np.isclose(freqs[j], float(f), rtol=1e-6, atol=1e-9):
            raise ValueError(f"frequency {f} Hz is not in the forced response (available: {freqs.tolist()})")
        idxs.append(j)
    return idxs


def _forced_items(fr, j, key, n_edges):
    if key == "reflection":
        cvals = [complex(fr.reflection_at(e)[j]) for e in range(n_edges)]
        return [
            DataItem("Reflection magnitude", "edge", [abs(c) for c in cvals], ""),
            DataItem("Reflection phase", "edge", [math.degrees(cmath.phase(c)) for c in cvals], "deg", phase=True),
        ]
    if key not in _FORCED_FIELDS:
        raise ValueError(f"unknown forced field {key!r}; choose from {list(_FORCED_FIELDS) + ['reflection']}")
    label, basis, comp, unit = _FORCED_FIELDS[key]
    cvals = [complex(fr.field(e, basis)[j, comp]) for e in range(n_edges)]
    return [
        DataItem(f"{label} amplitude", "edge", [abs(c) for c in cvals], unit),
        DataItem(f"{label} phase", "edge", [math.degrees(cmath.phase(c)) for c in cvals], "deg", phase=True),
    ]


def _node_items(network, items):
    """Reduce each non-phase edge item onto the nodes (mean over incident edges)."""
    incident = _incident_edges(network)
    out = []
    for it in items:
        if it.target != "edge" or it.phase:
            continue
        values = []
        for edges in incident:
            values.append(sum(it.values[e] for e in edges) / len(edges) if edges else float("nan"))
        out.append(DataItem(it.name, "node", values, it.unit))
    return out


def _incident_edges(network):
    incident = [[] for _ in network._elements]
    for ei, (t, h, _a) in enumerate(network._edges):
        incident[t].append(ei)
        incident[h].append(ei)
    return incident


# --------------------------------------------------------------------------- #
# Payload assembly
# --------------------------------------------------------------------------- #
def build_payload(network, datasets, title=None) -> dict:
    """Assemble the full ``SaveFilePayload`` dict for ``network`` plus ``datasets``."""
    prov = _matching_provenance(network)
    model = _build_model(network, prov)
    out = {
        "version": SAVE_FILE_VERSION,
        "timestamp": _now_iso(),
        "meta": {"title": title if title is not None else (_prov_title(prov) or DEFAULT_TITLE)},
        "model": model,
        "uiAttributes": _build_ui_attributes(network, prov),
        "uiState": _build_ui_state(network, prov),
    }
    if datasets:
        out["data"] = {"datasets": [ds.to_dict(f"ds-{i}-{_slug(ds.name)}") for i, ds in enumerate(datasets)]}
    return out


def _matching_provenance(network):
    """Return ``network.provenance`` only if it still matches the live topology."""
    prov = getattr(network, "provenance", None)
    if prov is None:
        return None
    if len(prov.node_ids) == len(network._elements) and len(prov.edge_ids) == len(network._edges):
        return prov
    warnings.warn(
        "network topology no longer matches its loaded provenance; "
        "synthesizing a fresh UI layout (positions/ids are not preserved)",
        stacklevel=3,
    )
    return None


def _prov_title(prov):
    if prov is None:
        return None
    return (prov.doc.get("meta") or {}).get("title")


def _build_model(network, prov):
    elements = network._elements
    edges = network._edges
    ids = _node_ids(network, prov)
    prov_nodes = {n["id"]: n for n in prov.doc["model"].get("nodes", [])} if prov else {}
    prov_edges = {e["id"]: e for e in prov.doc["model"].get("edges", [])} if prov else {}
    src_ord, tgt_ord, port_attrs = (None, None, None) if prov else _assign_ports(network)

    nodes = []
    for i, spec in enumerate(elements):
        ui_type, modeled = _spec_to_ui(spec)
        attrs = copy.deepcopy((prov_nodes.get(ids[i]) or {}).get("attributes") or {}) if prov else {}
        attrs.update(modeled)
        attrs["label"] = spec.name or attrs.get("label") or ui_type
        if spec.residual_id in _BOUNDARY_RIDS:
            attrs.update(_bc_to_ui(getattr(spec, "perturbation_bc", None)))
        if not prov:
            attrs.update(port_attrs.get(i, {}))
        attrs["index"] = i
        nodes.append({"id": ids[i], "type": ui_type, "attributes": attrs})

    edges_out = []
    for ei, (t, h, area) in enumerate(edges):
        if prov:
            eid = prov.edge_ids[ei]
            base = prov_edges.get(eid, {})
            source_handle = base.get("sourceHandle")
            target_handle = base.get("targetHandle")
            etype = base.get("type", "flow")
            eattrs = copy.deepcopy(base.get("attributes") or {})
        else:
            eid = f"edge_{ei + 1}"
            source_handle = f"{ids[t]}-port-{src_ord[ei]}"
            target_handle = f"{ids[h]}-port-{tgt_ord[ei]}"
            etype = "flow"
            eattrs = {}
        name = network._edge_names[ei] if ei < len(network._edge_names) else None
        eattrs["label"] = name or eattrs.get("label") or f"e{ei}"
        eattrs["index"] = ei
        eattrs["area"] = float(area)
        edges_out.append(
            {
                "id": eid,
                "source": ids[t],
                "target": ids[h],
                "sourceHandle": source_handle,
                "targetHandle": target_handle,
                "type": etype,
                "attributes": eattrs,
            }
        )

    return {"id": MODEL_ID, "globalAttributes": _global_attributes(network), "nodes": nodes, "edges": edges_out}


def _global_attributes(network):
    cp, R = float(network.gas.tf[0]), float(network.gas.tf[1])
    gamma = cp / (cp - R)
    return {
        "gasConstant": R,
        "heatCapacityRatio": gamma,
        "referencePressure": float(network.p_ref),
        "referenceTemperature": float(network.T_ref),
        "referenceMassFlow": float(network._mdot_ref or 0.0),
    }


def _build_ui_attributes(network, prov):
    ids = _node_ids(network, prov)
    if prov:
        existing = {u["id"]: u for u in (prov.doc.get("uiAttributes") or {}).get("nodes", [])}
        nodes = []
        for nid in ids:
            u = existing.get(nid) or {}
            nodes.append({"id": nid, "position": u.get("position", {"x": 0.0, "y": 0.0}), "data": u.get("data", {})})
        return {"nodes": nodes}
    pos = _layout(network)
    return {
        "nodes": [
            {"id": ids[i], "position": {"x": pos[i][0], "y": pos[i][1]}, "data": {}}
            for i in range(len(network._elements))
        ]
    }


def _build_ui_state(network, prov):
    if prov and prov.doc.get("uiState") is not None:
        return copy.deepcopy(prov.doc["uiState"])
    counts = {}
    for spec in network._elements:
        ui_type, _ = _spec_to_ui(spec)
        counts[ui_type] = counts.get(ui_type, 0) + 1
    return {"counters": {"nodeCounters": dict(counts), "totalNodeCounters": dict(counts)}}


# --------------------------------------------------------------------------- #
# Element / boundary-condition reverse mapping
# --------------------------------------------------------------------------- #
def _spec_to_ui(spec):
    """Map an ``ElementSpec`` to its UI ``(type, modeled-attributes)`` pair."""
    rid = spec.residual_id
    fp = spec.fparams
    if rid == MASS_FLOW_INLET:
        return "MassFlowInlet", {"massFlowRate": float(fp[0]), "totalTemperature": float(fp[1])}
    if rid == PT_INLET:
        return "TotalPressureInlet", {"totalPressure": float(fp[0]), "totalTemperature": float(fp[1])}
    if rid == P_OUTLET:
        return "PressureOutlet", {"pressure": float(fp[0]), "backflowTotalTemperature": float(fp[1])}
    if rid == ISEN_AREA_CHANGE:
        return "IsentropicAreaChange", {}
    if rid == SUDDEN_AREA_CHANGE:
        return "SuddenAreaChange", {"contractionCoefficient": float(fp[0])}
    if rid == LOSS:
        return "LossElement", {"lossCoefficient": float(fp[0])}
    if rid == JUNCTION:
        return "JunctionStaticP", {}
    if rid == SPLITTER:
        return "LosslessSplitter", {}
    if rid == DUCT:
        return "Duct", {"length": float(fp[0]) if fp else 0.0}
    if rid == WALL:
        return "Wall", {}
    raise ValueError(f"cannot serialize element with residual id {rid} to the UI format")


def _bc_to_ui(bc):
    """Map a single-port ``PerturbationBC`` back to UI acoustic attributes."""
    if bc is None:
        return {}
    kind = getattr(bc, "kind", "inherit")
    if kind == "inherit":
        return {}
    if kind == "hard_wall":
        return {"boundaryType": "rigid"}
    if kind == "open_end":
        return {"boundaryType": "open"}
    if kind == "impedance":
        Z = getattr(bc, "Z", None)
        if getattr(bc, "specific", False) and isinstance(Z, (int, float, complex)):
            z = complex(Z)
            return {
                "boundaryType": "impedance",
                "impedanceMagnitude": abs(z),
                "impedancePhase": math.degrees(cmath.phase(z)),
            }
        warnings.warn(
            "absolute or non-constant acoustic impedance cannot be expressed in the UI schema; "
            "omitting the acoustic boundary condition for this terminal",
            stacklevel=4,
        )
        return {}
    warnings.warn(
        f"perturbation BC kind {kind!r} has no UI representation; omitting its acoustic boundary condition",
        stacklevel=4,
    )
    return {}


# --------------------------------------------------------------------------- #
# Synthesized ids, ports and layout (no provenance)
# --------------------------------------------------------------------------- #
def _node_ids(network, prov):
    if prov:
        return list(prov.node_ids)
    ids = []
    counts = {}
    for spec in network._elements:
        ui_type, _ = _spec_to_ui(spec)
        counts[ui_type] = counts.get(ui_type, 0) + 1
        ids.append(f"{ui_type}_{counts[ui_type]}")
    return ids


def _assign_ports(network):
    """Assign UI port ordinals for a synthesized case (targets first, then sources).

    Mirrors the UI's port numbering (``utils/ports.ts``): a node's target
    (incoming) ports are ``0..T-1`` and its source (outgoing) ports ``T..T+S-1``.
    Returns ``(src_ord, tgt_ord, port_attrs)`` where ``port_attrs`` carries the
    dynamic-port count parameters for junctions/splitters.
    """
    elements = network._elements
    incoming = [[] for _ in elements]
    outgoing = [[] for _ in elements]
    for ei, (t, h, _a) in enumerate(network._edges):
        outgoing[t].append(ei)
        incoming[h].append(ei)

    src_ord, tgt_ord, port_attrs = {}, {}, {}
    for nd, spec in enumerate(elements):
        rid = spec.residual_id
        inc, out = incoming[nd], outgoing[nd]
        for k, ei in enumerate(inc):
            tgt_ord[ei] = k
        for k, ei in enumerate(out):
            src_ord[ei] = len(inc) + k
        if rid == JUNCTION:
            port_attrs[nd] = {"leftPorts": len(inc), "rightPorts": len(out)}
        elif rid == SPLITTER:
            port_attrs[nd] = {"rightPorts": len(out)}
    return src_ord, tgt_ord, port_attrs


def _layout(network):
    """Layered left-to-right positions: x by longest-path rank, y within rank."""
    from collections import defaultdict, deque

    n = len(network._elements)
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for t, h, _a in network._edges:
        adj[t].append(h)
        indeg[h] += 1

    rank = [0] * n
    remaining = list(indeg)
    queue = deque(i for i in range(n) if indeg[i] == 0)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            rank[v] = max(rank[v], rank[u] + 1)
            remaining[v] -= 1
            if remaining[v] == 0:
                queue.append(v)

    groups = defaultdict(list)
    for i in range(n):
        groups[rank[i]].append(i)

    dx, dy = 240.0, 140.0
    pos = {}
    for r, members in groups.items():
        for k, i in enumerate(members):
            pos[i] = (r * dx, (k - (len(members) - 1) / 2.0) * dy)
    return pos


# --------------------------------------------------------------------------- #
# YAML serialization
# --------------------------------------------------------------------------- #
def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "x"


def _py_scalar(value):
    """Coerce numpy scalars to native Python types for clean YAML output."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class _Dumper(yaml.SafeDumper):
    """SafeDumper that renders all-numeric lists inline and accepts numpy scalars."""


def _represent_list(dumper, data):
    flow = bool(data) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)
for _t in (np.float64, np.float32, np.float16):
    _Dumper.add_representer(_t, lambda d, v: d.represent_float(float(v)))
for _t in (np.int64, np.int32, np.int16, np.int8):
    _Dumper.add_representer(_t, lambda d, v: d.represent_int(int(v)))


def _yaml_dump(payload):
    return yaml.dump(
        payload,
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )
