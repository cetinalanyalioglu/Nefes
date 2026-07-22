"""Named-parameter descriptors for network elements (the parameter schema).

Every physical parameter an element carries is declared here once, by name: which
``fparams`` slot (or named ``ElementSpec`` field, or composite knob) it lives in, its SI
unit, its admissible range, and how to validate a new value.  The schema is the single
source of truth the generic parameter API (:meth:`nefes.shell.network.Network.get` /
``set`` / ``with_params`` / ``parameters``) reads and writes through, so names, order,
validation and packing cannot drift apart -- a consistency test
(``tests/test_parameters.py``) checks the declared packing against every factory's
actual output.

Main exports: :class:`ParamDescriptor`, :func:`descriptors_for`, :func:`pack_fparams`,
:func:`rebuild_composite`, and the :data:`ELEMENT_PARAMS` / :data:`COMPOSITE_PARAMS`
registries.

See Also
--------
nefes.elements.catalog : the factories whose signatures this schema mirrors.
nefes.shell.params : the network-level addressing built on this schema.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .composite import CompositeElementSpec, is_composite
from .ids import (
    CAVITY,
    CHOKED_NOZZLE_OUTLET,
    DUCT,
    ELEMENT_TYPE_NAMES,
    FLAME_EQUILIBRIUM,
    FLAME_HEAT_RELEASE,
    FORCED_SPLITTER,
    ISEN_AREA_CHANGE,
    JUNCTION,
    LINEAR_RESISTANCE,
    LOSS,
    MASS_FLOW_INLET,
    MASS_FLOW_OUTLET,
    MASS_SOURCE,
    MIXER,
    P_OUTLET,
    PIPE,
    PT_INLET,
    SPLITTER,
    SUDDEN_AREA_CHANGE,
    TRANSFER_MATRIX,
    WALL,
)

# Parameter value kinds a descriptor may declare.
KINDS = ("float", "int", "vector", "object", "str", "stations")

# YAML round-trip levels, purely documentary (surfaced by the inventory and the docs).
ROUNDTRIP = ("yes", "partial", "no")


@dataclass(frozen=True)
class ParamDescriptor:
    """One named element parameter: where it lives, its unit, bounds and validation.

    A descriptor's target is exactly one of: an ``fparams`` slot (``slot`` set), a named
    ``ElementSpec`` field (``field`` set), a composite knob (neither set -- it lives in
    ``CompositeElementSpec.params``), or, for ``kind="vector"``, the whole ``fparams``
    list.  Validation is fail-closed: :meth:`validate` raises a named error rather than
    accepting a value the element's factory would have rejected.

    Attributes
    ----------
    name : str
        The parameter's address leaf (``"inlet.mdot"`` -> ``"mdot"``).
    unit : str
        SI unit string, for display only (values are plain SI floats).
    lo, hi : float, optional
        Admissible range endpoints (``None`` -> unbounded on that side).
    lo_open, hi_open : bool
        Whether the corresponding endpoint is excluded (strict inequality).
    slot : int, optional
        The ``fparams`` slot this parameter occupies (atomic elements).
    field : str, optional
        The ``ElementSpec`` attribute this parameter occupies (object/field-valued).
    kind : str
        One of :data:`KINDS`; drives coercion (``"float"`` / ``"int"`` / ``"vector"`` /
        ``"object"`` / ``"str"`` / ``"stations"``).
    optional : bool
        Whether ``None`` is an admissible value (clears the field).
    check : callable, optional
        Extra validator ``check(value) -> value`` run after the generic coercion
        (e.g. the forced-splitter sum rule); raises ``ValueError`` on rejection.
    encode, decode : callable, optional
        Conversion between the public validated value and its numeric ``fparams``
        representation.  Unset means the value is stored directly.
    doc : str
        One-line human-readable description (inventory display).
    roundtrip : str
        YAML round-trip level of this parameter (:data:`ROUNDTRIP`); documentary.
    advanced : bool
        Excluded from the default :meth:`nefes.shell.network.Network.parameters`
        inventory (still fully addressable).
    layer : str
        Which solution layer the parameter touches: ``"mean"`` (default; enters the mean
        residuals, so changing it reshapes the mean flow) or ``"perturbation"`` (enters
        only the acoustic/perturbation operator -- storage volumes, inertance lengths,
        boundary and source closures -- so the mean state is invariant to it).
    """

    name: str
    unit: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_open: bool = False
    hi_open: bool = False
    slot: Optional[int] = None
    field: Optional[str] = None
    kind: str = "float"
    optional: bool = False
    check: Optional[Callable] = None
    encode: Optional[Callable] = None
    decode: Optional[Callable] = None
    doc: str = ""
    roundtrip: str = "yes"
    advanced: bool = False
    layer: str = "mean"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"descriptor {self.name!r}: unknown kind {self.kind!r}; choose from {KINDS}")
        if self.roundtrip not in ROUNDTRIP:
            raise ValueError(f"descriptor {self.name!r}: unknown roundtrip {self.roundtrip!r}")
        if self.slot is not None and self.field is not None:
            raise ValueError(f"descriptor {self.name!r}: slot and field are mutually exclusive")

    @property
    def bounds_text(self) -> str:
        """Human-readable admissible range, e.g. ``"[0, 1]"`` or ``">= 0"`` (``""`` if unbounded)."""
        if self.lo is None and self.hi is None:
            return ""
        if self.lo is not None and self.hi is not None:
            lb = "(" if self.lo_open else "["
            rb = ")" if self.hi_open else "]"
            return f"{lb}{self.lo:g}, {self.hi:g}{rb}"
        if self.lo is not None:
            return f"{'>' if self.lo_open else '>='} {self.lo:g}"
        return f"{'<' if self.hi_open else '<='} {self.hi:g}"

    def _check_range(self, v: float, where: str) -> None:
        ok_lo = self.lo is None or (v > self.lo if self.lo_open else v >= self.lo)
        ok_hi = self.hi is None or (v < self.hi if self.hi_open else v <= self.hi)
        if not (ok_lo and ok_hi):
            rng = self.bounds_text
            raise ValueError(
                f"{self.name} must be in {rng} {f'[{self.unit}] ' if self.unit else ''}(got {v}) on {where}"
            )

    def validate(self, value, where: str = "element"):
        """Coerce and range-check ``value``; return the stored form or raise ``ValueError``.

        ``where`` names the target element in the error message (fail-closed validation:
        an out-of-range or mistyped value raises immediately, it is never stored).
        """
        if value is None:
            if self.optional:
                return None
            raise ValueError(f"{self.name} on {where} cannot be None")
        if self.kind == "float":
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.name} on {where} must be a real number; got {value!r}")
            self._check_range(v, where)
            v = self.check(v) if self.check is not None else v
            return v
        if self.kind == "int":
            v = int(value)
            if v != float(value):
                raise ValueError(f"{self.name} on {where} must be an integer; got {value!r}")
            self._check_range(v, where)
            return self.check(v) if self.check is not None else v
        if self.kind == "str":
            v = str(value)
            return self.check(v) if self.check is not None else v
        if self.kind == "vector":
            try:
                v = [float(x) for x in value]
            except (TypeError, ValueError):
                raise ValueError(f"{self.name} on {where} must be a sequence of real numbers; got {value!r}")
            for x in v:
                self._check_range(x, where)
            return self.check(v) if self.check is not None else v
        if self.kind == "stations":
            try:
                v = [(float(x), float(a)) for (x, a) in value]
            except (TypeError, ValueError):
                raise ValueError(f"{self.name} on {where} must be a sequence of (x, area) pairs; got {value!r}")
            return self.check(v) if self.check is not None else v
        # kind == "object": type/constructor validation only (no numeric bounds)
        return self.check(value, where) if self.check is not None else value


# ---------------------------------------------------------------------------
# Object-field validators (type/constructor checks instead of numeric bounds)
# ---------------------------------------------------------------------------


def _check_perturbation_bc(value, where):
    from ..perturbation.operator.boundary_bc import PerturbationBC

    if not isinstance(value, PerturbationBC):
        raise ValueError(
            f"perturbation_bc on {where} must be a PerturbationBC (e.g. PerturbationBC.open_end()) or None; "
            f"got {type(value).__name__}"
        )
    return value


def _check_pipe_formulation(value):
    from .ids import PIPE_FORMULATION_CODES

    if value not in PIPE_FORMULATION_CODES:
        raise ValueError(f"formulation must be one of {sorted(PIPE_FORMULATION_CODES)}; got {value!r}")
    return value


def _encode_pipe_formulation(value):
    from .ids import PIPE_FORMULATION_CODES

    return float(PIPE_FORMULATION_CODES[value])


def _decode_pipe_formulation(value):
    from .ids import PIPE_FORMULATION_NAMES

    return PIPE_FORMULATION_NAMES[int(value)]


def _check_dynamic_source(value, where):
    from .dynamic_source import DynamicSource

    if not isinstance(value, DynamicSource):
        raise ValueError(
            f"dynamic_source on {where} must be a DynamicSource (e.g. from n_tau_flame) or None; "
            f"got {type(value).__name__}"
        )
    return value


def _check_transfer_matrix(value, where):
    from ..perturbation.identify import UnknownTransferMatrix
    from ..perturbation.matrix import TransferMatrix

    if not isinstance(value, (TransferMatrix, UnknownTransferMatrix)):
        raise ValueError(
            f"transfer_matrix on {where} must be a TransferMatrix, an UnknownTransferMatrix marker, or None; "
            f"got {type(value).__name__}"
        )
    return value


def _check_composition(value, where):
    # a named species mixture {species: fraction}; raw sequences (perfect-gas passive
    # scalars) are accepted verbatim -- the builder validates their length at compile
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            frac = float(v)
            if frac < 0.0:
                raise ValueError(f"composition on {where}: fraction for {k!r} must be >= 0; got {v}")
            out[str(k)] = frac
        if out and not any(f > 0.0 for f in out.values()):
            raise ValueError(f"composition on {where} must have at least one positive fraction")
        return out
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(
            f"composition on {where} must be a {{species: fraction}} dict (or raw passive-scalar values); "
            f"got {value!r}"
        )


def _check_basis(value):
    if value not in ("mole", "mass"):
        raise ValueError(f"basis must be 'mole' or 'mass'; got {value!r}")
    return value


def _check_fraction_sum(betas):
    import math

    if len(betas) < 1:
        raise ValueError("fractions needs at least one split fraction")
    if not math.fsum(betas) < 1.0:
        raise ValueError(
            f"fractions must sum to < 1 (the remainder branch carries 1 - sum); got sum = {math.fsum(betas):g}"
        )
    return betas


def _check_stations(pairs):
    # mirror the tapered_duct factory checks so a bad table is rejected before the rebuild
    if len(pairs) < 2:
        raise ValueError("stations needs >= 2 (x, area) pairs")
    xs = [x for x, _a in pairs]
    if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
        raise ValueError(f"station positions x must be strictly increasing; got {xs}")
    if any(a <= 0.0 for _x, a in pairs):
        raise ValueError("every station area must be positive")
    return pairs


# ---------------------------------------------------------------------------
# Shared descriptor blocks
# ---------------------------------------------------------------------------


# the optional storage lengths [l_up, l_down, end_correction] (see catalog._storage_block)
def _storage_descriptors(first_slot: int) -> Tuple[ParamDescriptor, ...]:
    docs = (
        "passage half-length on the port-0 side (acoustic storage)",
        "passage half-length on the port-1 side (acoustic storage)",
        "added-mass length (inertance only)",
    )
    names = ("l_up", "l_down", "end_correction")
    return tuple(
        ParamDescriptor(nm, unit="m", lo=0.0, slot=first_slot + i, doc=doc, layer="perturbation")
        for i, (nm, doc) in enumerate(zip(names, docs))
    )


_PERTURBATION_BC = ParamDescriptor(
    "perturbation_bc",
    field="perturbation_bc",
    kind="object",
    optional=True,
    check=_check_perturbation_bc,
    doc="acoustic termination (None inherits the linearized boundary row)",
    roundtrip="partial",
)

_DYNAMIC_SOURCE = ParamDescriptor(
    "dynamic_source",
    field="dynamic_source",
    kind="object",
    optional=True,
    check=_check_dynamic_source,
    doc="unsteady heat-release response S(omega); mean flow ignores it",
    roundtrip="no",
)

_COMPOSITION = ParamDescriptor(
    "composition",
    field="composition_spec",
    kind="object",
    optional=True,
    check=lambda v, where: _check_composition(v, where),
    doc="feed/backflow species mixture {species: fraction} in `basis` units",
    roundtrip="yes",
)

_BASIS = ParamDescriptor(
    "basis",
    field="basis",
    kind="str",
    check=_check_basis,
    doc="units of `composition`: 'mole' or 'mass'",
    roundtrip="yes",
)

_MARKER = ParamDescriptor(
    "marker",
    field="marker",
    lo=0.0,
    hi=1.0,
    doc="injected burnt marker (0 fresh, 1 burnt); marker-gated networks only",
    roundtrip="yes",
)

_EPS = ParamDescriptor(
    "eps",
    unit="kg/s",
    lo=0.0,
    lo_open=True,
    field="eps",
    optional=True,
    doc="smoothing-width override (None follows the global solve-time eps)",
    roundtrip="no",
    advanced=True,
)

_STREAM_FIELDS = (_COMPOSITION, _BASIS, _MARKER)


# ---------------------------------------------------------------------------
# Per-kind registries
# ---------------------------------------------------------------------------

# Atomic elements, keyed by residual id.  Slot order mirrors the factory packing in
# nefes.elements.catalog exactly (checked by tests/test_parameters.py).
ELEMENT_PARAMS: Dict[int, Tuple[ParamDescriptor, ...]] = {
    MASS_FLOW_INLET: (
        ParamDescriptor("mdot", unit="kg/s", lo=0.0, slot=0, doc="prescribed inflow mass rate"),
        ParamDescriptor("Tt", unit="K", lo=0.0, lo_open=True, slot=1, doc="total temperature of the feed"),
        *_STREAM_FIELDS,
        _PERTURBATION_BC,
    ),
    PT_INLET: (
        ParamDescriptor("pt", unit="Pa", lo=0.0, lo_open=True, slot=0, doc="prescribed total pressure"),
        ParamDescriptor("Tt", unit="K", lo=0.0, lo_open=True, slot=1, doc="total temperature of the feed"),
        *_STREAM_FIELDS,
        _PERTURBATION_BC,
    ),
    P_OUTLET: (
        ParamDescriptor(
            "p", unit="Pa", lo=0.0, lo_open=True, slot=0, doc="prescribed static (and backflow total) pressure"
        ),
        ParamDescriptor(
            "Tt_backflow", unit="K", lo=0.0, lo_open=True, slot=1, doc="total temperature drawn in on ingestion"
        ),
        *_STREAM_FIELDS,
        _PERTURBATION_BC,
    ),
    MASS_FLOW_OUTLET: (
        ParamDescriptor("mdot", unit="kg/s", lo=0.0, lo_open=True, slot=0, doc="prescribed outflow mass rate"),
        _PERTURBATION_BC,
    ),
    CHOKED_NOZZLE_OUTLET: (
        ParamDescriptor("throat_area", unit="m^2", lo=0.0, lo_open=True, slot=0, doc="sonic-throat area A*"),
        ParamDescriptor(
            "back_pressure",
            unit="Pa",
            lo=0.0,
            lo_open=True,
            field="back_pressure",
            optional=True,
            doc="ambient discharge pressure (diagnostic choke check only)",
        ),
        _PERTURBATION_BC,
    ),
    WALL: (_PERTURBATION_BC,),
    CAVITY: (
        ParamDescriptor(
            "volume", unit="m^3", lo=0.0, lo_open=True, slot=0, doc="enclosed cavity volume", layer="perturbation"
        ),
    ),
    ISEN_AREA_CHANGE: _storage_descriptors(0),
    TRANSFER_MATRIX: (
        ParamDescriptor(
            "transfer_matrix",
            field="transfer_matrix",
            kind="object",
            optional=True,
            check=_check_transfer_matrix,
            doc="frequency-domain 2-port descriptor (None -> isentropic area change)",
            roundtrip="no",
        ),
    ),
    SUDDEN_AREA_CHANGE: (
        ParamDescriptor("cc", lo=0.0, hi=1.0, lo_open=True, slot=0, doc="vena-contracta contraction coefficient"),
        *_storage_descriptors(1),
        _EPS,
    ),
    LOSS: (
        ParamDescriptor("K", slot=0, doc="loss coefficient (dynamic heads dropped)"),
        ParamDescriptor(
            "ref_port", lo=0, hi=1, slot=1, kind="int", doc="port whose dynamic head K references", advanced=True
        ),
        *_storage_descriptors(2),
        _EPS,
    ),
    LINEAR_RESISTANCE: (
        ParamDescriptor("R", unit="Pa/(kg/s)", lo=0.0, slot=0, doc="linear resistance (Pt drop per unit mass flow)"),
        *_storage_descriptors(1),
    ),
    FLAME_HEAT_RELEASE: (
        ParamDescriptor("Qdot", unit="W", slot=0, doc="heat-release rate added across the flame"),
        _DYNAMIC_SOURCE,
    ),
    FLAME_EQUILIBRIUM: (_DYNAMIC_SOURCE,),
    MASS_SOURCE: (
        ParamDescriptor("mdot", unit="kg/s", slot=0, doc="injected mass rate (> 0 adds mass)"),
        ParamDescriptor("u_inj", unit="m/s", slot=1, doc="axial injection velocity (0 = normal injection)"),
        ParamDescriptor("T", unit="K", lo=0.0, lo_open=True, slot=2, doc="injected stream total temperature"),
        *_STREAM_FIELDS,
        _DYNAMIC_SOURCE,
    ),
    JUNCTION: (
        ParamDescriptor(
            "volume", unit="m^3", lo=0.0, slot=0, doc="plenum chamber volume (0 = no compliance)", layer="perturbation"
        ),
    ),
    SPLITTER: (
        ParamDescriptor(
            "volume", unit="m^3", lo=0.0, slot=0, doc="plenum chamber volume (0 = no compliance)", layer="perturbation"
        ),
    ),
    MIXER: (
        ParamDescriptor(
            "recovery",
            lo=0.0,
            hi=1.0,
            slot=0,
            doc="dynamic-head recovery (0 = full dump loss / plenum, 1 = least-dissipative ideal)",
        ),
    ),
    FORCED_SPLITTER: (
        ParamDescriptor(
            "fractions",
            lo=0.0,
            hi=1.0,
            lo_open=True,
            hi_open=True,
            kind="vector",
            check=_check_fraction_sum,
            doc="controlled outflow mass fractions (port order; remainder implicit)",
        ),
    ),
    DUCT: (ParamDescriptor("length", unit="m", lo=0.0, slot=0, doc="duct length (acoustic phase)"),),
    PIPE: (
        ParamDescriptor("length", unit="m", lo=0.0, lo_open=True, slot=0, doc="pipe length"),
        ParamDescriptor("diameter", unit="m", lo=0.0, lo_open=True, slot=1, doc="hydraulic diameter (friction term)"),
        ParamDescriptor("friction_factor", lo=0.0, slot=2, doc="Darcy friction factor"),
        ParamDescriptor(
            "formulation",
            slot=3,
            kind="str",
            check=_check_pipe_formulation,
            encode=_encode_pipe_formulation,
            decode=_decode_pipe_formulation,
            doc="mean-flow closure: darcy-weisbach or momentum",
        ),
    ),
}

# Composite elements, keyed by kind.  These target the CompositeElementSpec.params dict;
# a write re-runs the factory (see rebuild_composite), never patches derived values.
COMPOSITE_PARAMS: Dict[str, Tuple[ParamDescriptor, ...]] = {
    "orifice": (
        ParamDescriptor("throat_area", unit="m^2", lo=0.0, lo_open=True, doc="throat (vena-contracta plane) area"),
    ),
    "lossy_nozzle": (
        ParamDescriptor("throat_area", unit="m^2", lo=0.0, lo_open=True, doc="throat area AT"),
        ParamDescriptor("beta", lo=0.0, hi=1.0, lo_open=True, doc="jet-to-downstream area ratio Aj/A2"),
    ),
    "sudden_contraction": (
        ParamDescriptor("cc", lo=0.0, hi=1.0, lo_open=True, doc="vena-contracta contraction coefficient"),
    ),
    "helmholtz_resonator": (
        ParamDescriptor("volume", unit="m^3", lo=0.0, lo_open=True, doc="backing cavity volume"),
        ParamDescriptor("neck_length", unit="m", lo=0.0, lo_open=True, doc="neck length (acoustic inertance)"),
        ParamDescriptor("neck_area", unit="m^2", lo=0.0, lo_open=True, doc="neck cross-sectional area"),
    ),
    "fanno_pipe": (
        ParamDescriptor("length", unit="m", lo=0.0, lo_open=True, doc="total pipe length"),
        ParamDescriptor("diameter", unit="m", lo=0.0, lo_open=True, doc="hydraulic diameter"),
        ParamDescriptor("friction_factor", lo=0.0, doc="Darcy friction factor"),
        ParamDescriptor(
            "n_segments",
            lo=2,
            kind="int",
            doc="segment count (fidelity knob; changing it re-discretizes the interior)",
            advanced=True,
        ),
        ParamDescriptor(
            "formulation",
            kind="str",
            check=_check_pipe_formulation,
            doc="segment closure: momentum or darcy-weisbach",
        ),
    ),
    "tapered_duct": (
        ParamDescriptor(
            "stations",
            kind="stations",
            check=_check_stations,
            doc="(x, area) station table; positions in m, areas in m^2",
        ),
    ),
}


def descriptors_for(el) -> Tuple[ParamDescriptor, ...]:
    """The parameter descriptors of an element spec (atomic or composite).

    Parameters
    ----------
    el : ElementSpec or CompositeElementSpec
        The element whose schema is requested.

    Returns
    -------
    tuple of ParamDescriptor
        The declared parameters, in slot order then field order.

    Raises
    ------
    KeyError
        For a composite of unknown kind (no registered schema).
    """
    if is_composite(el):
        kind = el.kind or el.name
        if kind not in COMPOSITE_PARAMS:
            raise KeyError(f"composite kind {kind!r} has no parameter schema; known kinds: {sorted(COMPOSITE_PARAMS)}")
        return COMPOSITE_PARAMS[kind]
    return ELEMENT_PARAMS.get(int(el.residual_id), ())


def find_descriptor(el, name: str) -> ParamDescriptor:
    """Look up one named parameter on an element spec; ``KeyError`` if it has none."""
    for d in descriptors_for(el):
        if d.name == name:
            return d
    label = el.kind if is_composite(el) else ELEMENT_TYPE_NAMES.get(int(el.residual_id), "element")
    known = [d.name for d in descriptors_for(el)]
    raise KeyError(f"{label} has no parameter {name!r}; it has: {known or 'none'}")


def pack_fparams(rid: int, values: Dict[str, object]):
    """Pack named slot parameters into the ``fparams`` list the kernels expect.

    The declared inverse of the factory packing: ``pack_fparams(rid, {name: value})``
    must reproduce ``ElementSpec.fparams`` exactly for every kind (the consistency test
    guards this).  A ``kind="vector"`` schema (the forced splitter) returns the vector
    itself.

    Parameters
    ----------
    rid : int
        The element's ``residual_id``.
    values : dict
        ``{parameter name: value}`` for every slot parameter of the kind.

    Returns
    -------
    list of float
    """
    slots = [d for d in ELEMENT_PARAMS.get(rid, ()) if d.slot is not None]
    vector = [d for d in ELEMENT_PARAMS.get(rid, ()) if d.kind == "vector"]
    if vector:
        return [float(v) for v in values[vector[0].name]]
    out = [0.0] * len(slots)
    for d in slots:
        value = d.encode(values[d.name]) if d.encode is not None else values[d.name]
        out[d.slot] = float(value)
    return out


# ---------------------------------------------------------------------------
# Composite rebuild (the safe write path: re-run the factory, never patch internals)
# ---------------------------------------------------------------------------


def _sub_eps(el: CompositeElementSpec):
    """Recover the eps override a composite factory threaded into its sub-elements."""
    for sub in el.sub_elements:
        eps = getattr(sub, "eps", None)
        if eps is not None:
            return eps
    return None


def rebuild_composite(el: CompositeElementSpec, updates: Dict[str, object]) -> CompositeElementSpec:
    """Re-run a composite's factory with its stored params merged with ``updates``.

    The one safe write path for a composite: editing the named knob and re-deriving the
    sub-elements and internal edges together, so they can never drift apart (patching
    ``sub_elements`` / ``internal_edges`` in place is exactly the failure mode this
    design removes).  The composite's name and any embedded smoothing override are
    preserved.

    Parameters
    ----------
    el : CompositeElementSpec
        The composite to rebuild.
    updates : dict
        ``{param name: validated value}`` to merge over the stored ``params``.

    Returns
    -------
    CompositeElementSpec
        A fresh spec from the factory; the caller swaps it into the network.
    """
    from . import catalog as cat

    kind = el.kind or el.name
    p = {**el.params, **updates}
    eps = _sub_eps(el)
    if kind == "orifice":
        return cat.orifice(p["throat_area"], name=el.name, eps=eps)
    if kind == "lossy_nozzle":
        return cat.lossy_nozzle(p["throat_area"], p["beta"], name=el.name, eps=eps)
    if kind == "sudden_contraction":
        return cat.sudden_contraction(cc=p["cc"], name=el.name, eps=eps)
    if kind == "helmholtz_resonator":
        return cat.helmholtz_resonator(p["volume"], p["neck_length"], p["neck_area"], name=el.name)
    if kind == "fanno_pipe":
        return cat.fanno_pipe(
            p["length"],
            p["diameter"],
            p["friction_factor"],
            p["n_segments"],
            name=el.name,
            formulation=p["formulation"],
        )
    if kind == "tapered_duct":
        return cat.tapered_duct(p["stations"], name=el.name)
    raise KeyError(f"composite kind {kind!r} has no registered rebuild; known kinds: {sorted(COMPOSITE_PARAMS)}")
