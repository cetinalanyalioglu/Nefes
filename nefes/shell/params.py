"""Network-level parameter addressing: the inventory, ``get``/``set`` write paths and copies.

The machinery behind :meth:`nefes.shell.network.Network.parameters` / ``get`` / ``set`` /
``update`` / ``with_params``.  Addresses are dotted ``"element.param"`` strings (elements
and edges by name, plus the bare network references ``p_ref`` / ``T_ref``); every write
validates against the element's declared schema (:mod:`nefes.elements.parameters`) and
fails closed -- an unknown address raises with near-match suggestions, never a silent
no-op.

Main exports: :func:`inventory`, :func:`get_param`, :func:`set_params`,
:func:`update_params`, :func:`copy_network`, :class:`ParameterInfo`,
:class:`ParameterInventory`.
"""

import copy as _copy
import difflib
import numbers
from dataclasses import dataclass
from html import escape as _escape
from typing import Dict, List, Optional, Tuple

from ..elements.composite import is_composite
from ..elements.ids import ALLOWS_AREA_CHANGE, ELEMENT_TYPE_NAMES
from ..elements.parameters import (
    ParamDescriptor,
    descriptors_for,
    find_descriptor,
    rebuild_composite,
)
from ..elements.parametric import is_parametric

# Network-level reference parameters (dimension-preserving value knobs; the gas model is
# deliberately outside this API -- reconfiguring it reshapes the problem).
_NETWORK_PARAMS: Tuple[ParamDescriptor, ...] = (
    ParamDescriptor("p_ref", unit="Pa", lo=0.0, lo_open=True, doc="absolute-pressure gauge reference"),
    ParamDescriptor("T_ref", unit="K", lo=0.0, lo_open=True, doc="reference temperature for the initial guess"),
    ParamDescriptor(
        "mdot_ref", unit="kg/s", lo=0.0, lo_open=True, optional=True, advanced=True, doc="mass-flow scale seed override"
    ),
    ParamDescriptor("h_ref", unit="J/kg", optional=True, advanced=True, doc="absolute-enthalpy datum override"),
)

# network parameter name -> the Network attribute that stores it
_NETWORK_ATTR = {"p_ref": "p_ref", "T_ref": "T_ref", "mdot_ref": "_mdot_ref", "h_ref": "_h_ref"}

# the per-edge flow area (areas live on edges, never on elements)
_AREA = ParamDescriptor("area", unit="m^2", lo=0.0, lo_open=True, doc="edge cross-sectional flow area")

# Inventory table marks: theme-agnostic glyphs (inherit the notebook text color).
_LAYER_MARK = {"mean": "μ", "perturbation": "∼"}
_LAYER_LABEL = {"mean": "mean layer", "perturbation": "perturbation layer"}
_ADV_MARK = "*"
_INVENTORY_LEGEND = "μ mean · ∼ perturbation · * advanced"


@dataclass(frozen=True)
class ParameterInfo:
    """One row of the parameter inventory: an addressable parameter and its current state.

    Attributes
    ----------
    address : str
        The dotted address (``"inlet.mdot"``, ``"e3.area"``, ``"p_ref"``).
    value : object
        The current value.
    unit : str
        SI unit string (display only).
    bounds : str
        Human-readable admissible range (``""`` if unbounded).
    kind : str
        Value kind (``"float"``, ``"int"``, ``"vector"``, ``"object"``, ``"str"``, ``"stations"``).
    target : str
        What the address points at: ``"element"``, ``"composite"``, ``"edge"`` or ``"network"``.
    doc : str
        One-line description.
    layer : str
        Which solution layer the parameter touches: ``"mean"`` (reshapes the mean flow, and
        with it everything built on top) or ``"perturbation"`` (enters only the acoustic/
        perturbation operator -- the mean state is invariant to it).
    advanced : bool
        Whether the parameter is an advanced knob normally hidden from
        :meth:`~nefes.shell.network.Network.parameters` (still fully addressable).
    """

    address: str
    value: object
    unit: str = ""
    bounds: str = ""
    kind: str = "float"
    target: str = "element"
    doc: str = ""
    layer: str = "mean"
    advanced: bool = False


class ParameterInventory(list):
    """The list of :class:`ParameterInfo` rows ``Network.parameters()`` returns.

    A plain list with table reprs and dict-style access by address.
    The table marks each row's solution layer (``μ`` mean, ``∼`` perturbation) and
    flags advanced knobs with ``*``.

    Examples
    --------
    >>> inv = net.parameters()
    >>> inv["inlet.mdot"].value
    0.3
    >>> [p.address for p in inv if p.target == "edge"]
    ['e0.area', 'e1.area']
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            for info in self:
                if info.address == key:
                    return info
            raise KeyError(key)
        return list.__getitem__(self, key)

    @property
    def addresses(self) -> List[str]:
        """Every address in the inventory, in order."""
        return [info.address for info in self]

    @staticmethod
    def _fmt_value(v) -> str:
        if isinstance(v, float):
            return f"{v:g}"
        if v is None:
            return "-"
        text = repr(v) if isinstance(v, (dict, list, tuple)) else type(v).__name__ if hasattr(v, "__dict__") else str(v)
        return text if len(text) <= 40 else text[:37] + "..."

    @staticmethod
    def _layer_mark(layer: str) -> str:
        return _LAYER_MARK.get(layer, layer)

    @staticmethod
    def _adv_mark(advanced: bool) -> str:
        return _ADV_MARK if advanced else ""

    def __repr__(self) -> str:
        if not self:
            return "ParameterInventory (empty)"
        headers = ("address", "value", "unit", "bounds", "layer", "adv")
        rows = [
            (
                i.address,
                self._fmt_value(i.value),
                i.unit,
                i.bounds,
                self._layer_mark(i.layer),
                self._adv_mark(i.advanced),
            )
            for i in self
        ]
        widths = [max(len(r[c]) for r in rows + [headers]) for c in range(len(headers))]
        header = "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()
        lines = [header, "  ".join("-" * w for w in widths)]
        for r in rows:
            lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        lines.append("")
        lines.append(f"  {_INVENTORY_LEGEND}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        # Borders use currentColor so the table stays legible on light and dark notebook themes.
        th = "padding:2px 8px;border-bottom:1px solid currentColor;text-align:left"
        td = "padding:2px 8px;text-align:left"
        col_titles = (
            ("address", "address"),
            ("value", "value"),
            ("unit", "unit"),
            ("bounds", "bounds"),
            ("layer", "solution layer (μ mean, ∼ perturbation)"),
            ("adv", "advanced knob (*)"),
            ("doc", "doc"),
        )
        head = (
            "<tr>"
            + "".join(f"<th style='{th}' title='{_escape(title)}'>{_escape(h)}</th>" for h, title in col_titles)
            + "</tr>"
        )
        body_parts = []
        for i in self:
            layer_mark = self._layer_mark(i.layer)
            adv_mark = self._adv_mark(i.advanced)
            layer_title = _LAYER_LABEL.get(i.layer, i.layer)
            adv_title = "advanced" if i.advanced else "standard"
            cells = (
                (_escape(i.address), None),
                (_escape(self._fmt_value(i.value)), None),
                (_escape(i.unit), None),
                (_escape(i.bounds), None),
                (_escape(layer_mark), layer_title),
                (_escape(adv_mark), adv_title),
                (_escape(i.doc), None),
            )
            body_parts.append(
                "<tr>"
                + "".join(
                    (
                        f"<td style='{td}' title='{_escape(title)}'>{content}</td>"
                        if title is not None
                        else f"<td style='{td}'>{content}</td>"
                    )
                    for content, title in cells
                )
                + "</tr>"
            )
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            + head
            + "".join(body_parts)
            + "</table>"
        )
        legend = (
            f"<div style='font-family:sans-serif;font-size:0.85em;opacity:0.75;margin-top:4px'>"
            f"{_escape(_INVENTORY_LEGEND)}</div>"
        )
        return table + legend


# --------------------------------------------------------------------------- #
# Name lookup
# --------------------------------------------------------------------------- #
def element_index(net, key) -> int:
    """Resolve an element reference (index or unique display name) to its node index.

    Parameters
    ----------
    net : Network
        The network to look in.
    key : int or str
        Node index, or the element's display name.

    Returns
    -------
    int

    Raises
    ------
    KeyError
        Unknown name (with near-match suggestions) or index out of range.
    """
    if isinstance(key, numbers.Integral) and not isinstance(key, bool):
        if not 0 <= key < len(net._elements):
            raise KeyError(f"element index {key} out of range [0, {len(net._elements)})")
        return int(key)
    name = str(key)
    for n, el in enumerate(net._elements):
        if el.name == name:
            return n
    known = [el.name for el in net._elements if el.name]
    hint = _suggest(name, known)
    raise KeyError(f"no element named {name!r}{hint}")


def edge_index(net, key) -> int:
    """Resolve an edge reference (index or unique edge name) to its edge id."""
    if isinstance(key, numbers.Integral) and not isinstance(key, bool):
        if not 0 <= key < len(net._edges):
            raise KeyError(f"edge index {key} out of range [0, {len(net._edges)})")
        return int(key)
    name = str(key)
    matches = [i for i, nm in enumerate(net._edge_names) if nm == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"edge name {name!r} is ambiguous (edges {matches}); address it by index")
    hint = _suggest(name, list(net._edge_names))
    raise KeyError(f"no edge named {name!r}{hint}")


def _suggest(name: str, known: List[str]) -> str:
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    return f"; did you mean {', '.join(repr(c) for c in close)}?" if close else f"; known: {sorted(known)}"


def _element_label(el) -> str:
    typ = el.kind if is_composite(el) else ELEMENT_TYPE_NAMES.get(int(el.residual_id), "element")
    return f"{typ} {el.name!r}"


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def _vector_offset(el) -> int:
    """Index at which an element's tail vector parameter begins (past its fixed slots)."""
    return sum(1 for dd in descriptors_for(el) if dd.slot is not None)


def _read(el, d: ParamDescriptor):
    """Read one declared parameter off an element spec."""
    if is_composite(el):
        return el.params.get(d.name)
    if d.kind == "vector":
        return list(el.fparams[_vector_offset(el) :])
    if d.slot is not None:
        v = el.fparams[d.slot]
        if d.decode is not None:
            return d.decode(v)
        return int(v) if d.kind == "int" else float(v)
    return getattr(el, d.field)


def _incident_edges(net, n: int) -> List[int]:
    return [ei for ei, (t, h, _a) in enumerate(net._edges) if t == n or h == n]


def _element_area(net, n: int, for_write: bool) -> List[int]:
    """The incident edge ids behind an element-level ``area`` address (fan-out set).

    Permitted only where the address is unambiguous: a single-port element, or a
    constant-area element (``ALLOWS_AREA_CHANGE`` false) whose ports must share one area
    anyway.  Anything else (an area change, a manifold, a composite) carries genuinely
    per-edge areas, so the per-edge address must be used.
    """
    el = net._elements[n]
    inc = _incident_edges(net, n)
    if not inc:
        raise KeyError(f"{_element_label(el)} has no connected edges to carry an area")
    if len(inc) == 1:
        return inc
    if not is_composite(el) and not ALLOWS_AREA_CHANGE.get(int(el.residual_id), True):
        return inc
    verb = "set" if for_write else "read"
    names = [net._edge_names[ei] for ei in inc]
    raise KeyError(
        f"{_element_label(el)} carries per-edge areas; {verb} them on the edges instead: "
        + ", ".join(f"{nm!r}" for nm in names)
    )


def _parametric_object(net, n: int, field: str):
    """The protocol-bearing object behind an element's object-valued parameter (fail-closed)."""
    el = net._elements[n]
    obj = _read(el, find_descriptor(el, field))
    if obj is None:
        raise KeyError(f"{_element_label(el)} has no {field} attached, so its parameters are not addressable")
    if not is_parametric(obj):
        raise KeyError(f"{_element_label(el)}.{field} ({type(obj).__name__}) exposes no addressable parameters")
    return obj


def get_param(net, address: str):
    """Read one parameter by its dotted address (see :meth:`Network.get`)."""
    kind, payload = _resolve(net, address)
    if kind == "network":
        return getattr(net, _NETWORK_ATTR[payload])
    if kind == "edge":
        return float(net._edges[payload][2])
    if kind == "nested":
        n, field, tail = payload
        return _parametric_object(net, n, field).get(tail)
    n, leaf = payload
    el = net._elements[n]
    if leaf == "area":
        areas = {float(net._edges[ei][2]) for ei in _element_area(net, n, for_write=False)}
        if len(areas) > 1:
            raise ValueError(f"{_element_label(el)} edges carry different areas {sorted(areas)}; read them per edge")
        return areas.pop()
    return _read(el, find_descriptor(el, leaf))


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def _resolve(net, address: str):
    """Resolve a dotted address to ``("network", name)``, ``("edge", ei)`` or ``("element", (n, leaf))``."""
    address = str(address)
    if "." not in address:
        if address in _NETWORK_ATTR:
            return "network", address
        raise KeyError(f"unknown network parameter {address!r}{_suggest(address, list(_NETWORK_ATTR))}")
    head, leaf = address.rsplit(".", 1)
    el_names = {el.name: n for n, el in enumerate(net._elements) if el.name}
    edge_names = [nm for nm in net._edge_names]
    if head in el_names and head in edge_names and leaf == "area":
        raise KeyError(
            f"{head!r} names both an element and an edge, so {address!r} is ambiguous; "
            "rename one of them or address the edge by index"
        )
    if head in el_names:
        return "element", (el_names[head], leaf)
    if head in edge_names:
        if leaf != "area":
            raise KeyError(f"edge {head!r} has no parameter {leaf!r}; the one edge parameter is 'area'")
        return "edge", edge_index(net, head)
    # a nested address: element, then an object-valued parameter, then that object's own
    # knob (possibly dotted further, resolved by the object itself)
    first, _, rest = address.partition(".")
    if first in el_names and "." in rest:
        field, _, tail = rest.partition(".")
        return "nested", (el_names[first], field, tail)
    pool = list(el_names) + edge_names
    raise KeyError(f"unknown element or edge {head!r} in address {address!r}{_suggest(head, pool)}")


def _set_edge_area(net, ei: int, value) -> None:
    v = _AREA.validate(value, where=f"edge {net._edge_names[ei]!r}")
    t, h, _a = net._edges[ei]
    net._edges[ei] = (t, h, v)


def _validate_composition(net, el, comp) -> None:
    """Reject species the reacting species set does not carry (fail-closed at set time)."""
    lib = getattr(net.gas, "species_set", None)
    if lib is None or not isinstance(comp, dict):
        return
    known = {sp.name for sp in lib.species}
    missing = [s for s in comp if s not in known]
    if missing:
        raise ValueError(
            f"composition on {_element_label(el)} names species not in the loaded species_set: {missing}; "
            f"the species set carries {len(known)} species"
        )


def set_params(net, target, params: Dict[str, object]) -> int:
    """Validate and apply named parameters to one element, in place.

    The single element-write path: every value passes its descriptor's validation, a
    composite is rebuilt through its factory (never patched), the constant-area
    ``area`` fans out to the incident edges, and the compiled-problem cache is dropped.

    Parameters
    ----------
    net : Network
        The network holding the element.
    target : int or str
        Element node index or display name.
    params : dict
        ``{parameter name: new value}``.

    Returns
    -------
    int
        The element's node index, for chaining.
    """
    n = element_index(net, target)
    el = net._elements[n]
    where = _element_label(el)

    area_edges: List[int] = []
    area_value = None
    composite_updates: Dict[str, object] = {}
    field_writes: List[Tuple[ParamDescriptor, object]] = []

    for name, value in params.items():
        if name == "area":
            area_edges = _element_area(net, n, for_write=True)
            area_value = _AREA.validate(value, where=where)
            continue
        d = find_descriptor(el, name)
        v = d.validate(value, where=where)
        if is_composite(el):
            composite_updates[name] = v
            continue
        if d.name == "composition":
            _validate_composition(net, el, v)
        field_writes.append((d, v))

    # basis alone is meaningless: it re-interprets a composition that must exist
    new_names = set(params)
    if "basis" in new_names and "composition" not in new_names and getattr(el, "composition_spec", None) is None:
        raise ValueError(f"basis on {where} without a composition; set composition= (they are a pair)")

    # all values validated; apply atomically
    if composite_updates:
        net._elements[n] = rebuild_composite(el, composite_updates)
    for d, v in field_writes:
        if d.kind == "vector":
            offset = _vector_offset(el)
            tail_len = len(el.fparams) - offset
            if len(v) != tail_len:
                raise ValueError(
                    f"{d.name} on {where} must keep its length {tail_len} (the port count is topology); "
                    f"got {len(v)} values"
                )
            el.fparams[offset:] = [float(x) for x in v]
        elif d.slot is not None:
            stored = d.encode(v) if d.encode is not None else v
            el.fparams[d.slot] = float(stored) if d.kind != "int" else float(int(stored))
        else:
            setattr(el, d.field, v)
    for ei in area_edges:
        t, h, _a = net._edges[ei]
        net._edges[ei] = (t, h, area_value)

    net._invalidate()
    return n


def update_params(net, mapping: Dict[str, object]) -> None:
    """Apply a batch of dotted-address writes (see :meth:`Network.update`).

    Element writes are grouped per element so a composite is rebuilt once with all its
    updates merged.  Every address is resolved before anything is written, so a bad
    address leaves the network untouched.
    """
    per_element: Dict[int, Dict[str, object]] = {}
    edge_writes: List[Tuple[int, object]] = []
    network_writes: List[Tuple[str, object]] = []
    nested_writes: Dict[Tuple[int, str], List[Tuple[str, object]]] = {}
    for address, value in mapping.items():
        kind, payload = _resolve(net, address)
        if kind == "network":
            network_writes.append((payload, value))
        elif kind == "edge":
            edge_writes.append((payload, value))
        elif kind == "nested":
            n, field, tail = payload
            _parametric_object(net, n, field).get(tail)  # resolve fail-closed before any write lands
            nested_writes.setdefault((n, field), []).append((tail, value))
        else:
            n, leaf = payload
            per_element.setdefault(n, {})[leaf] = value
    # nested writes to one object chain onto a single rebuilt copy, folded into the
    # element write batch so validation and cache invalidation follow the ordinary path
    for (n, field), writes in nested_writes.items():
        obj = _parametric_object(net, n, field)
        for tail, value in writes:
            obj = obj.with_value(tail, value)
        per_element.setdefault(n, {})[field] = obj

    for name, value in network_writes:
        set_network_param(net, name, value)
    for ei, value in edge_writes:
        _set_edge_area(net, ei, value)
    for n, params in per_element.items():
        set_params(net, n, params)
    net._invalidate()


def set_network_param(net, name: str, value) -> None:
    """Validate and set one network-level reference (``p_ref`` / ``T_ref`` / seeds)."""
    for d in _NETWORK_PARAMS:
        if d.name == name:
            setattr(net, _NETWORK_ATTR[name], d.validate(value, where="the network"))
            net._invalidate()
            return
    raise KeyError(f"unknown network parameter {name!r}{_suggest(name, list(_NETWORK_ATTR))}")


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def inventory(net, advanced: bool = False, layer: Optional[str] = None) -> ParameterInventory:
    """Build the full parameter inventory of a network (see :meth:`Network.parameters`).

    Recurses into object-valued parameters whose value implements the scalar-parameter
    protocol (:mod:`nefes.elements.parametric`), emitting their knobs as ordinary float
    rows under extended dotted addresses (``flame.dynamic_source.gain``).  ``layer``
    narrows the result to ``"mean"`` or ``"perturbation"`` rows.
    """
    if layer not in (None, "mean", "perturbation"):
        raise ValueError(f"layer must be 'mean' or 'perturbation'; got {layer!r}")
    inv = ParameterInventory()
    for n, el in enumerate(net._elements):
        name = el.name or f"#{n}"
        try:
            descs = descriptors_for(el)
        except KeyError:
            descs = ()
        for d in descs:
            if d.advanced and not advanced:
                continue
            target = "composite" if is_composite(el) else "element"
            value = _read(el, d)
            inv.append(
                ParameterInfo(
                    address=f"{name}.{d.name}",
                    value=value,
                    unit=d.unit,
                    bounds=d.bounds_text,
                    kind=d.kind,
                    target=target,
                    doc=d.doc,
                    layer=d.layer,
                    advanced=d.advanced,
                )
            )
            # the object's own scalar knobs join the address space; anything attached to
            # the perturbation layer leaves the mean flow untouched by construction
            if d.kind == "object" and is_parametric(value):
                for sub in value.param_descriptors():
                    inv.append(
                        ParameterInfo(
                            address=f"{name}.{d.name}.{sub.name}",
                            value=value.get(sub.name),
                            unit=sub.unit,
                            bounds=sub.bounds_text,
                            kind="float",
                            target=target,
                            doc=sub.doc,
                            layer="perturbation",
                            advanced=sub.advanced,
                        )
                    )
    for ei, (_t, _h, a) in enumerate(net._edges):
        inv.append(
            ParameterInfo(
                address=f"{net._edge_names[ei]}.area",
                value=float(a),
                unit=_AREA.unit,
                bounds=_AREA.bounds_text,
                kind="float",
                target="edge",
                doc=_AREA.doc,
                layer=_AREA.layer,
                advanced=_AREA.advanced,
            )
        )
    for d in _NETWORK_PARAMS:
        if d.advanced and not advanced:
            continue
        inv.append(
            ParameterInfo(
                address=d.name,
                value=getattr(net, _NETWORK_ATTR[d.name]),
                unit=d.unit,
                bounds=d.bounds_text,
                kind=d.kind,
                target="network",
                doc=d.doc,
                layer=d.layer,
                advanced=d.advanced,
            )
        )
    if layer is not None:
        inv = ParameterInventory(r for r in inv if r.layer == layer)
    return inv


# --------------------------------------------------------------------------- #
# Copies
# --------------------------------------------------------------------------- #
def copy_network(net):
    """Deep-copy a network's specification (elements, edges, references), not its caches.

    Edge order, port pins and edge names are preserved by construction, so a copy's
    compiled problem has the same layout and a warm start from the original's solution
    stays valid (the precondition the continuation drivers need).  The gas model is
    shared (it is an immutable configuration); the compiled-problem cache is not copied.
    """
    cls = type(net)
    new = cls.__new__(cls)
    new.gas = net.gas
    new.require_connected = net.require_connected
    new.p_ref = net.p_ref
    new.T_ref = net.T_ref
    new._mdot_ref = net._mdot_ref
    new._h_ref = net._h_ref
    new._elements = _copy.deepcopy(net._elements)
    new._edges = list(net._edges)
    new._ports = list(net._ports)
    new._edge_names = list(net._edge_names)
    new._edge_models = list(net._edge_models)
    new.provenance = _copy.deepcopy(net.provenance)
    new._compiled = None
    return new
