"""Parse the node-graph UI export (YAML) into Nefes connectivity."""

import json
import os
import re
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

import yaml

from ..elements import catalog as cat
from ..elements.ids import FLAME_EQUILIBRIUM, MASS_FLOW_INLET, MASS_SOURCE, P_OUTLET, PT_INLET
from ..graph.connectivity import Connectivity, build_connectivity
from ..thermo.api import EQ_FROZEN, EQ_KERNEL
from ..thermo.configure import equilibrium, perfect_gas
from .provenance import UIProvenance

_PORT_RE = re.compile(r"port-(\d+)$")

# The UI model id this loader targets.  Must match `id` in the Nemo model library
# (Nemo/public/models/nefes.yaml); the display name shown in the UI is set
# independently there.
MODEL_ID = "nefes"

# Model-level (globalAttributes) defaults, kept in sync with the UI model.
_GLOBAL_DEFAULTS = {
    "thermoModel": "perfect_gas",
    "gasConstant": 287.0,
    "heatCapacityRatio": 1.4,
    "mechanismFile": "",
    "species": "auto",
    "speciesReducer": "equilibrium_sampling",
    "speciesReduceThreshold": None,
    "speciesReduceAbove": None,
    "speciesMax": None,
    "speciesMust": "",
    "equilibriumTInit": 3000.0,
    "frozenTInit": 300.0,
    "streamMode": "auto",
    "streams": "",
    "streamBasis": "mole",
    "referencePressure": 101325.0,
    "referenceTemperature": 300.0,
    "referenceMassFlow": 0.0,
}


def _comp(a: dict):
    """Parse a node's UI ``composition`` string into a species mixture (or ``None``)."""
    return _parse_composition(a.get("composition"))


def _basis(a: dict) -> str:
    """A node's composition basis (``"mole"`` or ``"mass"``); defaults to ``"mole"``."""
    return str(a.get("basis") or "mole")


def _marker(a: dict) -> float:
    """A boundary's injected burnt-marker value (``0`` fresh, default; ``1`` burnt)."""
    return float(a.get("marker") or 0.0)


def _lengths(a: dict) -> dict:
    """Optional storage-length kwargs (``l_up``/``l_down``/``end_correction``) from a UI node.

    The per-port half-lengths and the end correction populate the storage block ``M`` (the
    element's compliance + inertance); all default to zero, so a node that omits them is the
    lengthless jump it was before.
    """
    return {
        "l_up": float(a.get("lengthUpstream", 0.0) or 0.0),
        "l_down": float(a.get("lengthDownstream", 0.0) or 0.0),
        "end_correction": float(a.get("endCorrection", 0.0) or 0.0),
    }


# UI node type -> (catalog factory, kwargs extracted from node attributes).
# Each builder takes the node's `attributes` dict and returns an ElementSpec.  The
# composition / basis attributes feed the reacting model; the perfect-gas model
# ignores them (no transported scalars), so they are harmless to pass always.
_UI_NODE_BUILDERS = {
    "MassFlowInlet": lambda a: cat.mass_flow_inlet(
        a["massFlowRate"], a["totalTemperature"], composition=_comp(a), basis=_basis(a), marker=_marker(a)
    ),
    "TotalPressureInlet": lambda a: cat.total_pressure_inlet(
        a["totalPressure"], a["totalTemperature"], composition=_comp(a), basis=_basis(a), marker=_marker(a)
    ),
    "PressureOutlet": lambda a: cat.pressure_outlet(
        a["pressure"],
        a.get("backflowTotalTemperature", 300.0),
        composition=_comp(a),
        basis=_basis(a),
        marker=_marker(a),
    ),
    "MassFlowOutlet": lambda a: cat.mass_flow_outlet(a["massFlowRate"]),
    "ChokedNozzleOutlet": lambda a: cat.choked_nozzle_outlet(a["throatArea"]),
    "Wall": lambda a: cat.wall(),
    "Cavity": lambda a: cat.cavity(a["volume"]),
    "IsentropicAreaChange": lambda a: cat.isentropic_area_change(**_lengths(a)),
    # The frequency-domain descriptor has no YAML form; the element loads with tm=None
    # (acoustically an isentropic area change) and is attached in Python afterwards.
    "TransferMatrix": lambda a: cat.transfer_matrix_element(),
    "SuddenAreaChange": lambda a: cat.sudden_area_change(cc=a.get("contractionCoefficient", 1.0), **_lengths(a)),
    "LossElement": lambda a: cat.loss(a["lossCoefficient"], **_lengths(a)),
    "LinearResistance": lambda a: cat.linear_resistance(a["resistance"], **_lengths(a)),
    "Duct": lambda a: cat.duct(a.get("length", 0.0)),
    "Pipe": lambda a: cat.pipe(
        a["length"],
        a["diameter"],
        a["frictionFactor"],
        formulation=a.get("formulation", "darcy-weisbach"),
    ),
    "HeatReleaseFlame": lambda a: cat.heat_release_flame(a["heatRelease"]),
    "EquilibriumFlame": lambda a: cat.equilibrium_flame(),
    "MassSource": lambda a: cat.mass_source(
        a["massFlowRate"],
        a["injectionTemperature"],
        _comp(a),
        u_inj=a.get("injectionVelocity", 0.0),
        basis=_basis(a),
        marker=_marker(a),
    ),
    "Junction": lambda a: cat.junction(
        recovery=a.get("recovery", 1.0) if a.get("recovery") is not None else 1.0,
        K=a.get("K"),
        volume=a.get("volume", 0.0) or 0.0,
        static_pressure=bool(a.get("staticPressure", False)),
    ),
    "ForcedSplitter": lambda a: cat.forced_splitter(_parse_fractions(a.get("fractions"))),
    # Composite elements: each expands at build time into its atomic recipe (see
    # nefes.elements.catalog); the UI carries only the composite's own parameters.
    "Orifice": lambda a: cat.orifice(a["throatArea"]),
    "LossyNozzle": lambda a: cat.lossy_nozzle(a["throatArea"], a["beta"]),
    "SuddenContraction": lambda a: cat.sudden_contraction(cc=a.get("contractionCoefficient", 0.62)),
    "HelmholtzResonator": lambda a: cat.helmholtz_resonator(a["volume"], a["neckLength"], a["neckArea"]),
    "FannoPipe": lambda a: cat.fanno_pipe(
        a["length"],
        a["diameter"],
        a["frictionFactor"],
        int(a.get("nSegments") or 8),
        formulation=a.get("formulation", "momentum"),
    ),
    "TaperedDuct": lambda a: cat.tapered_duct(_parse_area_profile(a.get("areaProfile"))),
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

# Per-edge thermochemistry closure tokens (UI edge ``thermoModel``) -> model id.  The writer
# (:mod:`nefes.io.yaml_out`) inverts this to emit the tokens back.
EDGE_CLOSURE = {"frozen": EQ_FROZEN, "equilibrium": EQ_KERNEL}


def _parse_composition(spec):
    """Parse a UI composition into a ``{species: fraction}`` dict (or ``None`` if empty).

    Accepts a JSON object (``{"O2": 0.21, "N2": 0.79}``) or a comma/newline separated
    list of ``species:fraction`` (``species=fraction`` / ``species fraction``) entries
    (``"O2:0.21, N2:0.79"``).  A dict is returned as-is (floats).  Returns ``None`` for an
    empty / missing value -- the reacting builders then raise their own clear error where a
    composition is mandatory.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    text = str(spec).strip()
    if not text:
        return None
    if text[0] == "{":
        return {str(k): float(v) for k, v in json.loads(text).items()}
    comp = {}
    for token in re.split(r"[,\n]+", text):
        token = token.strip()
        if not token:
            continue
        parts = re.split(r"[:=\s]+", token)
        if len(parts) != 2:
            raise ValueError(f"cannot parse composition entry {token!r}; use 'species:fraction'")
        comp[parts[0].strip()] = float(parts[1].strip())
    return comp or None


def _parse_fractions(spec) -> List[float]:
    """Parse a forced splitter's UI ``fractions`` string into the outflow betas.

    Accepts a comma/whitespace/semicolon separated list (``"0.3, 0.2"``), a JSON array
    (``"[0.3, 0.2]"``), or an already-parsed sequence.  Each value is the mass fraction of a
    controlled outflow port, in attachment order; the remainder branch is implicit (see
    :func:`nefes.elements.catalog.forced_splitter`).
    """
    if spec is None:
        raise ValueError("a ForcedSplitter node needs a 'fractions' list (one value per controlled outflow)")
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    text = str(spec).strip()
    if text.startswith("["):
        return [float(v) for v in json.loads(text)]
    values = [tok for tok in re.split(r"[,;\s]+", text) if tok]
    if not values:
        raise ValueError("a ForcedSplitter node needs a 'fractions' list (one value per controlled outflow)")
    return [float(v) for v in values]


def _parse_area_profile(spec) -> List[Tuple[float, float]]:
    """Parse a tapered duct's UI ``areaProfile`` string into ``(x, A)`` station pairs.

    Accepts ``"x:A, x:A, ..."`` entries (``=`` also works as the pair separator; comma,
    semicolon or newline between stations), a JSON array of pairs (``"[[0, 3e-3], ...]"``),
    or an already-parsed sequence of pairs.  Positions are metres, areas m^2; validation
    (>= 2 stations, strictly increasing x, positive areas) happens in the catalog factory.
    """
    if spec is None:
        raise ValueError("a TaperedDuct node needs an 'areaProfile' station table, e.g. '0:3e-3, 0.15:1.5e-3'")
    if isinstance(spec, (list, tuple)):
        return [(float(x), float(a)) for x, a in spec]
    text = str(spec).strip()
    if not text:
        raise ValueError("a TaperedDuct node needs an 'areaProfile' station table, e.g. '0:3e-3, 0.15:1.5e-3'")
    if text.startswith("["):
        return [(float(x), float(a)) for x, a in json.loads(text)]
    pairs = []
    for token in re.split(r"[,;\n]+", text):
        token = token.strip()
        if not token:
            continue
        parts = re.split(r"[:=]+", token)
        if len(parts) != 2:
            raise ValueError(f"cannot parse area-profile station {token!r}; use 'x:area'")
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def _parse_species(spec) -> Optional[List[str]]:
    """Parse a species list (``None`` if empty).

    Accepts a YAML sequence (each item one species name, kept verbatim -- the form
    :func:`nefes.io.yaml_out.save_case` emits, so CEA names carrying commas such as
    ``C2H2,acetylene`` round-trip) or a comma/whitespace separated string (a hand-written
    or UI convenience for simple, comma-free names).
    """
    if not spec:
        return None
    if isinstance(spec, (list, tuple)):
        names = [str(s).strip() for s in spec if str(s).strip()]
        return names or None
    names = [s.strip() for s in re.split(r"[,\s]+", str(spec)) if s.strip()]
    return names or None


def _resolve_path(path: str, case_dir: str) -> str:
    """Resolve a mechanism path: absolute, else relative to the case file, else the CWD."""
    if not path:
        raise ValueError("the reacting (equilibrium) thermo model needs a 'mechanismFile' set in the Model pane")
    path = str(path)
    candidates = [path] if os.path.isabs(path) else [os.path.join(case_dir, path), os.path.abspath(path)]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"mechanism file {path!r} not found (looked in {candidates})")


def _is_auto(species) -> bool:
    """True if the species spec requests the automatic (CEA-style) candidate slate."""
    return species is None or (len(species) == 1 and species[0].strip().lower() == "auto")


def _parse_streams(text, path: str) -> dict:
    """Parse the declared-stream basis string into ``{label: species mixture}``.

    The UI writes the declared streams as ``"label = species:frac, ...; label = ..."`` (streams
    separated by ``;``, each a label and a species composition joined by ``=``).  Returns an
    ordered ``{label: {species: fraction}}`` dict, the ``streams=`` argument of
    :func:`~nefes.thermo.configure.equilibrium`.
    """
    if not text or not str(text).strip():
        raise ValueError(
            f"{path}: streamMode='declared' needs a 'streams' basis, e.g. 'air = O2:0.21, N2:0.79; H2 = H2:1'"
        )
    streams: dict = {}
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"{path}: declared stream {chunk!r} must be 'label = composition'")
        label, comp = chunk.split("=", 1)
        label = label.strip()
        parsed = _parse_composition(comp)
        if not label or not parsed:
            raise ValueError(f"{path}: declared stream {chunk!r} must be 'label = species:frac, ...'")
        streams[label] = parsed
    if not streams:
        raise ValueError(f"{path}: streamMode='declared' but no streams were declared")
    return streams


def _auto_species_set(db, specs, g):
    """CEA-style automatic product slate over a ``SpeciesDatabase`` database, from the case params.

    A thin adapter that reads the reference pressure, burnt-gas temperature guess and reducer
    choice off the parsed model params ``g`` and delegates to the shared
    :func:`nefes.thermo.autoset.auto_product_set`, so the YAML loader and the Python
    network build resolve the automatic slate through one policy.
    """
    from ..thermo.autoset import auto_product_set

    thr = g.get("speciesReduceThreshold")
    above = g.get("speciesReduceAbove")
    cap = g.get("speciesMax")
    must = _parse_species(g.get("speciesMust")) or []
    return auto_product_set(
        db,
        specs,
        p_ref=float(g["referencePressure"]),
        T_init=float(g["equilibriumTInit"]),
        reducer_name=str(g.get("speciesReducer") or "equilibrium_sampling"),
        threshold=float(thr) if thr else None,
        reduce_above=int(above) if above else None,
        max_species=int(cap) if cap else None,
        must_species=must,
    )


def _build_species_set(g, specs, case_dir: str):
    """Build the reacting species set from the model params and the network feeds.

    With no ``mechanismFile`` the packaged NASA Glenn / CEA ``thermo.inp`` is the species
    database.  ``species`` may be an explicit list, or ``"auto"`` (the default) to derive a
    CEA-style candidate product slate from the feed compositions and reduce it.  An explicit
    native mechanism YAML (Cantera-subset) loads its species directly; an explicit
    ``thermo.inp`` path behaves like the packaged default.
    """
    from nefes.thermo import SpeciesDatabase, SpeciesSet

    species = _parse_species(g.get("species"))
    auto = _is_auto(species)
    mech_path = g.get("mechanismFile")

    if mech_path:
        path = _resolve_path(mech_path, case_dir)
        if path.lower().endswith((".yaml", ".yml")):
            lib = SpeciesSet.from_cantera(path)
            return lib if auto else lib.subset(species)
        return _auto_species_set(SpeciesDatabase(path), specs, g) if auto else SpeciesDatabase(path).select(species)

    db = SpeciesDatabase()
    return _auto_species_set(db, specs, g) if auto else db.select(species)


def _reacting_h_ref(gas, specs) -> float:
    """A representative absolute-enthalpy datum for the reacting solve.

    Mirrors the examples: the largest-magnitude feed/source/backflow enthalpy at its supply
    temperature, floored at ``1e4`` J/kg.  The reacting closures need this absolute datum (the
    perfect-gas ``cp * T_ref`` fallback is meaningless for a variable-composition gas).
    """
    from ..chem.composition import enthalpy_mass, species_mass_fractions
    from ..elements.composite import is_composite

    # Composites carry atomic sub-elements (a feed may live inside one), so flatten before
    # reading feed compositions -- a CompositeElementSpec has no composition of its own.
    atomic = []
    for sp in specs:
        atomic.extend(sp.sub_elements if is_composite(sp) else (sp,))

    lib = gas.species_set
    h_max = 0.0
    for sp in atomic:
        comp = sp.composition_spec
        if comp is None:
            continue
        if sp.residual_id in (MASS_FLOW_INLET, PT_INLET, P_OUTLET):
            t_supply = float(sp.fparams[1])
        elif sp.residual_id == MASS_SOURCE:
            t_supply = float(sp.fparams[2])
        else:
            continue
        Y = species_mass_fractions(lib, comp, sp.basis)
        h_max = max(h_max, abs(enthalpy_mass(lib, Y, t_supply)))
    return max(h_max, 1e4)


def _resolve_edge_models(reacting, specs, parsed, edge_tokens):
    """Per-edge thermo-model ids for the compiled problem (``None`` -> the gas/marker default).

    Perfect gas or all-``auto`` reacting edges defer to the gas/marker default (``None``); any
    explicit ``frozen`` / ``equilibrium`` edge switches to the hard per-edge closure, with the
    remaining ``auto`` edges labeled by a downstream-of-flame flood-fill.
    """
    n_edges = len(parsed)
    if not reacting:
        return [None] * n_edges

    # all-auto: defer to the marker-gated closure (orientation-robust; build_problem marker-gates).
    if all(edge_tokens[ei] == "auto" for ei in range(n_edges)):
        for tok in edge_tokens:
            if tok != "auto":  # defensive: only reachable if a token is malformed
                raise ValueError(f"unknown edge closure {tok!r}; choose 'auto', 'frozen' or 'equilibrium'")
        return [None] * n_edges

    flame_nodes = {i for i, sp in enumerate(specs) if sp.residual_id == FLAME_EQUILIBRIUM}
    # Orientation guard: the hard-closure flood-fill labels burnt along the declared arrows, so a
    # flame not drawn flow-aligned is mislabeled.  Warn per such flame (checked below).
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for ei, s, t, _area, _name in parsed:
        out_deg[s] += 1
        in_deg[t] += 1
    for f in flame_nodes:
        if out_deg[f] == 0 or in_deg[f] == 0:
            warnings.warn(
                f"flame node {f} is not drawn flow-aligned (declared in/out edges: {in_deg[f]}/{out_deg[f]}); "
                "the explicit hard-closure flood-fill labels burnt along the declared arrows, so this flame's "
                "reactant/product sides may be mislabeled. Draw its edges in the flow direction, set each "
                "incident edge's closure explicitly, or use all-'auto' edges (the orientation-proof marker "
                "closure).",
                stacklevel=2,
            )
    burnt = set()
    if flame_nodes:
        adj = defaultdict(list)  # node -> [(edge id, head node)], following the flow direction
        for ei, s, _t, _area, _name in parsed:
            adj[s].append((ei, _t))
        stack = []
        for f in flame_nodes:  # seed: every edge leaving a flame is burnt
            for ei, t in adj[f]:
                if ei not in burnt:
                    burnt.add(ei)
                    stack.append(t)
        while stack:  # flood downstream
            for ei, t in adj[stack.pop()]:
                if ei not in burnt:
                    burnt.add(ei)
                    stack.append(t)

    models = [None] * n_edges
    for ei, _s, _t, _area, _name in parsed:
        token = edge_tokens[ei]
        if token in EDGE_CLOSURE:
            models[ei] = EDGE_CLOSURE[token]
        elif token == "auto":
            models[ei] = EQ_KERNEL if (not flame_nodes or ei in burnt) else EQ_FROZEN
        else:
            raise ValueError(f"unknown edge closure {token!r}; choose 'auto', 'frozen' or 'equilibrium'")
    return models


def _parse_perturbation_bc(attrs: dict):
    """Build a ``PerturbationBC`` from a boundary node's UI acoustic attributes.

    Maps the UI ``boundaryType`` dropdown: ``"rigid"`` -> hard wall, ``"open"`` -> open end, ``"impedance"`` ->
    specific impedance from ``impedanceMagnitude`` (|Z|/rho c) and ``impedancePhase`` (degrees). Returns ``None``
    for ``"inherit"`` or when ``boundaryType`` is absent, leaving the element's default closure.
    """
    from ..perturbation.operator.boundary_bc import PerturbationBC

    btype = attrs.get("boundaryType")
    if btype == "inherit":
        return None
    if btype is None:
        return None
    if btype == "rigid":
        return PerturbationBC.hard_wall()
    if btype == "open":
        return PerturbationBC.open_end()
    if btype == "impedance":
        magnitude = float(attrs.get("impedanceMagnitude", 1.0))
        phase_deg = float(attrs.get("impedancePhase", 0.0))
        return PerturbationBC.impedance_polar(magnitude, phase_deg, specific=True)
    raise ValueError(f"unknown boundaryType {btype!r} on a boundary node")


def _port_of(handle: str, node_id: str, edge_id, side: str) -> int:
    """Port ordinal ``k`` from a UI handle ``"<node_id>-port-<k>"``, checked against ``node_id``.

    A flow edge names the ports it plugs into as ``sourceHandle`` / ``targetHandle`` of the form
    ``"<node>-port-<ordinal>"``; the node prefix must be the edge's own ``source`` / ``target``
    (named by ``side``).  A handle that names a different or absent node binds the edge to no port
    in Nemo, which then drops it silently, so it is rejected here rather than loaded as a phantom
    edge whose endpoint does not exist.
    """
    handle = str(handle)
    m = _PORT_RE.search(handle)
    if not m:
        raise ValueError(f"edge {edge_id!r}: cannot parse a port ordinal from its {side} handle {handle!r}")
    ordinal = int(m.group(1))
    expected = f"{node_id}-port-{ordinal}"
    if handle != expected:
        raise ValueError(
            f"edge {edge_id!r}: its {side} handle {handle!r} does not name its {side} node {node_id!r} "
            f"(expected {expected!r}); the edge would bind to no port and be dropped"
        )
    return ordinal


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
        tp = _port_of(edge["sourceHandle"], edge["source"], edge.get("id"), "source")
        hp = _port_of(edge["targetHandle"], edge["target"], edge.get("id"), "target")
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
    """Build an ``ElementSpec`` from a UI node, attaching its name and perturbation BC if any."""
    ntype = node.get("type")
    attrs = node.get("attributes") or {}
    if ntype in _DEFERRED_TYPES:
        raise ValueError(f"element type {ntype!r} is deferred (subsonic scope); remove it from the case")
    try:
        builder = _UI_NODE_BUILDERS[ntype]
    except KeyError:
        raise ValueError(f"unknown Nefes element type {ntype!r}")
    spec = builder(attrs)
    # The loaded label is an explicit name (from the saved case), not a factory default: assigning it
    # clears ``name_auto`` so the dedup keeps it verbatim instead of numbering it on reload.
    spec.name = str(attrs.get("label") or node.get("id") or ntype)
    spec.name_auto = False
    if ntype in _BOUNDARY_TYPES:
        bc = _parse_perturbation_bc(attrs)
        if bc is not None:  # else keep the factory default
            spec.perturbation_bc = bc
    return spec


def load_case(path: str):
    """Load a YAML file exported from the UI tool into a ``Network``.

    Parameters
    ----------
    path : str
        Path to the ``.yaml`` case written by the UI tool.

    Returns
    -------
    Network
        The reconstructed network, ready to solve.

    See Also
    --------
    case_from_dict : Build a network from an already-parsed case mapping.
    load_connectivity : Read only the topology, without boundary values.
    """
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    return case_from_dict(doc, case_dir=os.path.dirname(os.path.abspath(path)), source=path)


def case_from_dict(doc: dict, case_dir: str = None, source: str = "<dict>"):
    """Build a ``Network`` from an already-parsed UI/YAML case document.

    The file-less core of :func:`load_case`: the same schema, as an in-memory ``dict``.

    Parameters
    ----------
    doc : dict
        The parsed case document (a ``model`` section with ``nodes`` / ``edges``).
    case_dir : str, optional
        Directory used to resolve a reacting mechanism file referenced by relative path
        (default: the current working directory).
    source : str, optional
        Label used in error messages (default ``"<dict>"``).

    Returns
    -------
    Network
    """
    from ..shell import Network  # local import to avoid an import cycle

    path = source
    if case_dir is None:
        case_dir = os.getcwd()
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: case must be a mapping (dict), got {type(doc).__name__}")
    model = doc.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path}: not a UI save file (no 'model' section)")
    mid = model.get("id")
    if mid not in (None, MODEL_ID):
        raise ValueError(f"{path}: case targets model {mid!r}, expected {MODEL_ID!r}")

    g = dict(_GLOBAL_DEFAULTS)
    g.update({k: v for k, v in (model.get("globalAttributes") or {}).items() if v is not None})

    ui_nodes = model.get("nodes") or []
    ui_edges = model.get("edges") or []
    if not ui_nodes or not ui_edges:
        raise ValueError(f"{path}: the network has no nodes or no edges")

    # Thermo model: a calorically-perfect gas, or the reacting equilibrium model (transported
    # feed-stream mixture fractions constrained to HP equilibrium).
    model_kind = str(g.get("thermoModel") or "perfect_gas")
    if model_kind == "perfect_gas":
        gas = perfect_gas(R=float(g["gasConstant"]), gamma=float(g["heatCapacityRatio"]))
        reacting = False
    elif model_kind == "equilibrium":
        reacting = True
    else:
        raise ValueError(f"{path}: unknown thermoModel {model_kind!r} (expected 'perfect_gas' or 'equilibrium')")

    # Elements, ordered by the node index attribute (bandwidth-optimized in the UI).
    nodes_sorted = sorted(ui_nodes, key=lambda n: int((n.get("attributes") or {}).get("index", 0)))
    specs = [_build_ui_spec(n) for n in nodes_sorted]

    # The reacting species set and the absolute-enthalpy datum are built from the species-bearing
    # compositions, resolved after the node specs are parsed.  In "auto" mode these are the feed
    # compositions; in "declared" mode the feeds carry stream labels, not species, so the species
    # come from the declared stream basis instead (the feeds blend over it).
    if reacting:
        if str(g.get("streamMode") or "auto") == "declared":
            declared = _parse_streams(g.get("streams"), path)
            stream_basis = str(g.get("streamBasis") or "mole")
            T_ref = float(g["referenceTemperature"])
            # pseudo-feeds carrying each declared stream's species, so the species set / enthalpy
            # helpers (which read composition_spec as species) resolve from the declared basis
            stream_specs = [
                cat.mass_flow_inlet(1.0, T_ref, composition=comp, basis=stream_basis) for comp in declared.values()
            ]
            species_set = _build_species_set(g, stream_specs, case_dir)
            gas = equilibrium(
                species_set,
                streams=declared,
                basis=stream_basis,
                mode="declared",
                T_init=float(g["equilibriumTInit"]),
                T_init_frozen=float(g["frozenTInit"]),
            )
            h_ref = _reacting_h_ref(gas, stream_specs)
        else:
            species_set = _build_species_set(g, specs, case_dir)
            gas = equilibrium(species_set, T_init=float(g["equilibriumTInit"]), T_init_frozen=float(g["frozenTInit"]))
            h_ref = _reacting_h_ref(gas, specs)
    else:
        h_ref = None

    net = Network(
        gas,
        p_ref=float(g["referencePressure"]),
        T_ref=float(g["referenceTemperature"]),
        mdot_ref=float(g["referenceMassFlow"]) or None,
        h_ref=h_ref,
    )

    id_to_index = {}
    for n, spec in zip(nodes_sorted, specs):
        id_to_index[n["id"]] = net.add(spec)

    # Per node, gather incident (edge, side, port ordinal); densify by ordinal.
    edges_sorted = sorted(ui_edges, key=lambda e: int((e.get("attributes") or {}).get("index", 0)))
    incident = {i: [] for i in range(len(nodes_sorted))}
    parsed = []
    edge_tokens = []  # per-edge thermochemistry closure token (UI edge thermoModel)
    for ei, e in enumerate(edges_sorted):
        attrs = e.get("attributes") or {}
        for end in ("source", "target"):
            if e.get(end) not in id_to_index:
                raise ValueError(f"edge {e.get('id')!r} references unknown node {e.get(end)!r}")
        s = id_to_index[e["source"]]
        t = id_to_index[e["target"]]
        so = _port_of(e["sourceHandle"], e["source"], e.get("id"), "source")
        to = _port_of(e["targetHandle"], e["target"], e.get("id"), "target")
        area = float(attrs["area"])
        if area <= 0.0:
            raise ValueError(f"edge {e.get('id')!r} has non-positive area {area}")
        incident[s].append((ei, "tail", so))
        incident[t].append((ei, "head", to))
        parsed.append((ei, s, t, area, str(attrs.get("label") or e.get("id"))))
        edge_tokens.append(str(attrs.get("thermoModel") or "auto"))

    local_port = {}  # (edge_index, side) -> dense local port at that node
    for node, lst in incident.items():
        for local, (ei, side, _ord) in enumerate(sorted(lst, key=lambda x: x[2])):
            local_port[(ei, side)] = local

    # Resolve each edge's thermochemistry closure (frozen / equilibrium / auto-from-flames).
    edge_models = _resolve_edge_models(reacting, specs, parsed, edge_tokens)

    # Assemble topology from the parsed endpoints.
    for ei, s, t, area, name in parsed:
        net.connect(
            s,
            t,
            area,
            name=name,
            tail_port=local_port[(ei, "tail")],
            head_port=local_port[(ei, "head")],
            edge_model=edge_models[ei],
        )

    # Retain the UI-only metadata (positions, counters, ids, title) so the case
    # can be saved back for the UI verbatim -- see nefes.io.yaml_out.
    net.provenance = UIProvenance(
        doc=doc,
        node_ids=[n["id"] for n in nodes_sorted],
        edge_ids=[e.get("id", f"edge_{i + 1}") for i, e in enumerate(edges_sorted)],
    )
    return net


# --------------------------------------------------------------------------- #
# Restoring a saved solution
# --------------------------------------------------------------------------- #
def _dataset_items(doc: dict, name: str):
    """Return a saved dataset's series as ``{item_name: values}`` (``None`` if the dataset is absent)."""
    for ds in (doc.get("data") or {}).get("datasets") or []:
        if ds.get("name") == name:
            return {it.get("name"): it.get("values") for it in (ds.get("items") or [])}
    return None


def _dataset_meta(doc: dict, name: str, key: str):
    """Return a saved dataset's ``info`` metadata value for ``key`` (``None`` if absent)."""
    for ds in (doc.get("data") or {}).get("datasets") or []:
        if ds.get("name") == name:
            for entry in ds.get("info") or []:
                if entry.get("key") == key:
                    return entry.get("value")
    return None


def _saved_state(doc: dict, prob, dataset: str):
    """Rebuild the solver state ``x`` from a saved mean-flow dataset.

    The solved unknowns per edge are ``[mdot, p, h_t, xi_0 .. xi_{n_elem-1}, marker]``; every one
    is stored by :func:`nefes.io.yaml_out.save_case` as an edge series -- ``mdot``/``p``/``h_t``
    in the ``dataset`` (default ``"Mean flow"``) and the transported mixture fractions
    (``xi:<stream>``) and burnt marker (``burnt``) in its ``"<dataset> chemistry"`` sibling.  The
    mixture fractions are matched to their rows by *name* (``prob.scalar_names``), so a reload is
    robust to any reordering.

    Parameters
    ----------
    doc : dict
        The parsed case document.
    prob : CompiledProblem
        The freshly compiled problem the state is rebuilt for (fixes the row layout and edge count).
    dataset : str
        Name of the mean-flow dataset to restore.

    Returns
    -------
    x : ndarray
        The reconstructed state, shape ``(n_solve, n_edges)``.
    converged : bool
        The ``converged`` flag stored alongside the dataset (``False`` when not recorded).

    Raises
    ------
    ValueError
        When the requested dataset is absent, a needed field is missing, or an edge series does
        not match the network's edge count (the file was saved for a different network).
    """
    import numpy as np

    from .yaml_out import _FIELD_META

    mean = _dataset_items(doc, dataset)
    if mean is None:
        available = [ds.get("name") for ds in (doc.get("data") or {}).get("datasets") or []]
        raise ValueError(
            f"no saved dataset named {dataset!r} to restore (found {available or 'none'}); "
            "write one first with Solution.to_yaml / nefes.io.save_solution"
        )
    E = int(prob.n_edges)

    def col(items, label, where):
        vals = items.get(label)
        if vals is None:
            raise ValueError(f"the saved {where!r} dataset has no {label!r} series; cannot rebuild the state")
        arr = np.asarray(vals, dtype=np.float64)
        if arr.shape != (E,):
            raise ValueError(
                f"saved {label!r} carries {arr.size} value(s) but the network has {E} edge(s); "
                "the file was saved for a different network"
            )
        return arr

    x = np.zeros((int(prob.n_solve), E))
    x[0] = col(mean, _FIELD_META["mdot"][0], dataset)
    x[1] = col(mean, _FIELD_META["p"][0], dataset)
    x[2] = col(mean, _FIELD_META["h_t"][0], dataset)

    if prob.n_solve > 3:  # transported composition scalars live in the chemistry sibling
        chem_name = f"{dataset} chemistry"
        chem = _dataset_items(doc, chem_name)
        if chem is None:
            raise ValueError(
                f"the network transports {int(prob.n_solve) - 3} composition scalar(s) but the file has no "
                f"{chem_name!r} dataset; cannot rebuild the state"
            )
        for s, name in enumerate(prob.scalar_names):
            x[3 + s] = col(chem, f"xi:{name}", chem_name)
        mr = int(getattr(prob, "marker_row", -1))
        if mr >= 0:
            x[mr] = col(chem, "burnt", chem_name)

    converged = _dataset_meta(doc, dataset, "converged")
    return x, bool(converged) if converged is not None else False


def load_solution(path: str, method: str = "warm", dataset: str = "Mean flow", **solve_kw):
    """Load a network *and* a saved solution from a UI/YAML case, without a cold re-solve.

    Reads the network topology (as :func:`load_case`) and rebuilds the solver state from the
    solution datasets :func:`nefes.io.save_case` embeds.  The ``method`` toggle picks how much
    the loaded state is trusted:

    * ``"warm"`` (default) -- feed the stored state as the initial guess to a single ``kappa = 0``
      solve.  A faithfully-saved solution is already below tolerance, so the stage's convergence
      check returns at iteration ``0`` *before* assembling any Jacobian (the cost is one residual
      evaluation, not the full solve); a stale state simply keeps solving from the excellent start.
      The returned solution is therefore guaranteed consistent with the current network.
    * ``"deserialize"`` -- wrap the stored state verbatim in a :class:`~nefes.shell.network.Solution`
      with no solve at all (zero cost).  This trusts the file blindly: if the network was edited
      after saving, the state may not satisfy its residual, and ``residual_norm`` is reported as
      ``NaN`` (not evaluated).

    Parameters
    ----------
    path : str
        Path to a ``.yaml`` case carrying an embedded solution.
    method : {"warm", "deserialize"}, optional
        Restore strategy (default ``"warm"``).
    dataset : str, optional
        Name of the mean-flow dataset to restore (default ``"Mean flow"``); its chemistry sibling
        ``"<dataset> chemistry"`` supplies the transported scalars.
    **solve_kw
        Forwarded to :meth:`Network.solve` for ``method="warm"`` (e.g. ``tol``, ``verbose``, or an
        explicit ``kappa_stages`` to override the single ``kappa = 0`` stage).  Rejected for
        ``method="deserialize"``, which does not solve.

    Returns
    -------
    Solution
        The restored mean-flow solution.

    See Also
    --------
    load_case : Load only the network topology (no solution).
    nefes.io.save_solution : Write a network and solution to a case file.
    """
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    net = case_from_dict(doc, case_dir=os.path.dirname(os.path.abspath(path)), source=path)
    return _restore_solution(net, doc, method, dataset, solve_kw)


def _restore_solution(net, doc: dict, method: str, dataset: str, solve_kw: dict):
    """Rebuild a :class:`~nefes.shell.network.Solution` on ``net`` from the saved ``doc`` datasets."""
    from ..shell.network import Solution
    from ..solver.control import SolveResult

    prob = net.compile()
    x0, saved_converged = _saved_state(doc, prob, dataset)

    if method == "warm":
        # A faithfully-saved state is below tol, so a single kappa=0 stage returns at iteration 0
        # (the pre-Jacobian convergence check) -- a cheap verification, not a re-solve; a stale
        # state keeps solving from the excellent start.  The caller may override kappa_stages.
        kw = dict(solve_kw)
        kw.setdefault("kappa_stages", (0.0,))
        return net.solve(x0=x0, **kw)
    if method == "deserialize":
        if solve_kw:
            raise TypeError(f"method='deserialize' does not solve, so it takes no solve kwargs; got {sorted(solve_kw)}")
        # Trust the file: the stored state is returned unchecked (residual not evaluated).
        res = SolveResult(x=x0, converged=saved_converged, iterations=0, residual_norm=float("nan"))
        return Solution(net, prob, res)
    raise ValueError(
        f"unknown method {method!r}; choose 'warm' (verify by a single kappa=0 solve, the default) "
        "or 'deserialize' (return the stored state unchecked)"
    )
