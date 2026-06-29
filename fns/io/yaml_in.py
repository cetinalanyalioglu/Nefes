"""Parse the node-graph UI export (YAML) into FNS connectivity."""

import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import yaml

from ..connectivity import build_connectivity, Connectivity
from ..elements import catalog as cat
from ..elements.ids import FLAME_EQUILIBRIUM, MASS_FLOW_INLET, MASS_SOURCE, PT_INLET, P_OUTLET
from ..thermo.api import EQ_FROZEN, EQ_KERNEL
from ..thermo.configure import equilibrium, perfect_gas
from .provenance import UIProvenance

_PORT_RE = re.compile(r"port-(\d+)$")

# The UI model id this loader targets (public/models/fns-flow-network.yaml).
MODEL_ID = "fns-flow-network"

# Model-level (globalAttributes) defaults, kept in sync with the UI model.
_GLOBAL_DEFAULTS = {
    "thermoModel": "perfect_gas",
    "gasConstant": 287.0,
    "heatCapacityRatio": 1.4,
    # ``mechanismFile`` is no longer surfaced in the UI (the packaged thermo.inp is built in);
    # it stays here as an opt-in Python-side override for a native mechanism / explicit database.
    "mechanismFile": "",
    "species": "auto",
    "speciesReducer": "equilibrium_sampling",
    "equilibriumTInit": 3000.0,
    "frozenTInit": 300.0,
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


# UI node type -> (catalog factory, kwargs extracted from node attributes).
# Each builder takes the node's `attributes` dict and returns an ElementSpec.  The
# composition / basis attributes feed the reacting model; the perfect-gas model
# ignores them (no transported scalars), so they are harmless to pass always.
_UI_NODE_BUILDERS = {
    "MassFlowInlet": lambda a: cat.mass_flow_inlet(
        a["massFlowRate"], a["totalTemperature"], composition=_comp(a), basis=_basis(a)
    ),
    "TotalPressureInlet": lambda a: cat.total_pressure_inlet(
        a["totalPressure"], a["totalTemperature"], composition=_comp(a), basis=_basis(a)
    ),
    "PressureOutlet": lambda a: cat.pressure_outlet(
        a["pressure"], a.get("backflowTotalTemperature", 300.0), composition=_comp(a), basis=_basis(a)
    ),
    "MassFlowOutlet": lambda a: cat.mass_flow_outlet(a["massFlowRate"]),
    "ChokedNozzleOutlet": lambda a: cat.choked_nozzle_outlet(a["throatArea"]),
    "Wall": lambda a: cat.wall(),
    "IsentropicAreaChange": lambda a: cat.isentropic_area_change(),
    "SuddenAreaChange": lambda a: cat.sudden_area_change(cc=a.get("contractionCoefficient", 1.0)),
    "LossElement": lambda a: cat.loss(a["lossCoefficient"]),
    "Duct": lambda a: cat.duct(a.get("length", 0.0)),
    "HeatReleaseFlame": lambda a: cat.heat_release_flame(a["heatRelease"]),
    "EquilibriumFlame": lambda a: cat.equilibrium_flame(),
    "MassSource": lambda a: cat.mass_source(
        a["massFlowRate"], a["injectionTemperature"], _comp(a), u_inj=a.get("injectionVelocity", 0.0), basis=_basis(a)
    ),
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

# Per-edge thermochemistry closure tokens (UI edge ``thermoModel``) -> model id.
_EDGE_CLOSURE = {"frozen": EQ_FROZEN, "equilibrium": EQ_KERNEL}


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


def _parse_species(spec) -> Optional[List[str]]:
    """Parse a comma/whitespace separated species list (``None`` if empty)."""
    if not spec:
        return None
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


# Above ~40 candidate species the equilibrium Newton solve becomes expensive enough that
# reducing the slate pays off (hydrocarbon/air admits ~115); below it, run them raw.
_AUTO_REDUCE_THRESHOLD = 40


def _is_auto(species) -> bool:
    """True if the species spec requests the automatic (CEA-style) candidate slate."""
    return species is None or (len(species) == 1 and species[0].strip().lower() == "auto")


def _declared_species(specs) -> List[str]:
    """Feed/source species named anywhere in the network (the reactants), first-seen order."""
    out: List[str] = []
    for sp in specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        for name in comp:
            if name not in out:
                out.append(name)
    return out


def _dedup(seq) -> List[str]:
    """De-duplicate a sequence, preserving first-seen order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _feed_sample_states(feed_lib, specs, g):
    """Representative equilibrium probe states along the feed-mixing line.

    Each feed stream contributes its elemental composition; convex (mass) combinations of
    the distinct streams span the lean->rich range the network can realize, probed at a
    couple of temperatures bracketing the burnt-gas guess.  Used to drive slate reduction.
    """
    import numpy as np

    from thermolib import SampleState

    from ..composition import elemental_Z, species_mass_fractions

    p = float(g["referencePressure"])
    T_hi = float(g["equilibriumTInit"])
    T_samples = sorted({T_hi, max(1500.0, 0.7 * T_hi)})

    feeds = []
    for sp in specs:
        comp = getattr(sp, "composition_spec", None)
        if not comp:
            continue
        Y = species_mass_fractions(feed_lib, comp, getattr(sp, "basis", "mole"))
        feeds.append(elemental_Z(feed_lib, Y))

    uniq = []
    for Z in feeds:
        if not any(np.allclose(Z, U, atol=1e-9) for U in uniq):
            uniq.append(Z)

    elems = list(feed_lib.elements)

    def zdict(Z):
        return {elems[i]: float(Z[i]) for i in range(len(elems))}

    states = []
    for Z in uniq:
        states += [SampleState(zdict(Z), T, p) for T in T_samples]
    for ia in range(len(uniq)):
        for ib in range(ia + 1, len(uniq)):
            for w in (0.1, 0.3, 0.5, 0.7, 0.9):
                Zm = w * uniq[ia] + (1.0 - w) * uniq[ib]
                states += [SampleState(zdict(Zm), T, p) for T in T_samples]
    return states


def _auto_library(db, specs, g):
    """CEA-style automatic product slate over a ``ThermoInp`` database ``db``.

    Declared feed species fix the reachable element pool; the candidate gas-phase slate is
    every species buildable from those elements, reduced (when large) to the species that
    are non-trace at equilibrium across the feed-mixing range.  The final library also
    carries the declared feed species (including condensed fuels) so the frozen closure and
    the enthalpy datum can be evaluated; the equilibrium kernel masks condensed species out
    of the products.
    """
    from thermolib import get_reducer

    declared = _declared_species(specs)
    if not declared:
        raise ValueError(
            "the reacting (equilibrium) model with automatic species needs at least one feed "
            "or source composition (set 'species' to an explicit list to override)"
        )
    missing = [n for n in declared if n not in db]
    if missing:
        raise KeyError(f"feed species not in thermo.inp: {missing}")

    pool = set()
    for name in declared:
        pool.update(el for el in db[name].composition if el != "E")
    candidates = db.candidate_species(pool, gas_only=True, exclude_ions=True)
    declared_gas = [n for n in declared if db[n].phase == 0]

    if len(candidates) <= _AUTO_REDUCE_THRESHOLD:
        report = {"reducer": "none", "n_candidates": len(candidates), "n_kept": len(candidates)}
        final = _dedup(candidates + declared)
    else:
        feed_lib = db.library(_dedup(declared))
        samples = _feed_sample_states(feed_lib, specs, g)
        reducer = get_reducer(str(g.get("speciesReducer") or "equilibrium_sampling"))
        result = reducer.reduce(db.library(candidates), samples, always_keep=declared_gas)
        report = result.report
        final = _dedup(result.species + declared)

    lib = db.library(final)
    lib.reduction_report = report  # auditable: which products were selected and why
    return lib


def _build_library(g, specs, case_dir: str):
    """Build the reacting species library from the model params and the network feeds.

    With no ``mechanismFile`` the packaged NASA Glenn / CEA ``thermo.inp`` is the species
    database.  ``species`` may be an explicit list, or ``"auto"`` (the default) to derive a
    CEA-style candidate product slate from the feed compositions and reduce it.  An explicit
    native mechanism YAML (Cantera-subset) loads its species directly; an explicit
    ``thermo.inp`` path behaves like the packaged default.
    """
    from thermolib import SpeciesLibrary, ThermoInp

    species = _parse_species(g.get("species"))
    auto = _is_auto(species)
    mech_path = g.get("mechanismFile")

    if mech_path:
        path = _resolve_path(mech_path, case_dir)
        if path.lower().endswith((".yaml", ".yml")):
            lib = SpeciesLibrary.from_native(path)
            return lib if auto else lib.subset(species)
        return _auto_library(ThermoInp(path), specs, g) if auto else ThermoInp(path).library(species)

    db = ThermoInp()
    return _auto_library(db, specs, g) if auto else db.library(species)


def _reacting_h_ref(gas, specs) -> float:
    """A representative absolute-enthalpy datum for the reacting solve.

    Mirrors the examples: the largest-magnitude feed/source/backflow enthalpy at its supply
    temperature, floored at ``1e4`` J/kg.  The reacting closures need this absolute datum (the
    perfect-gas ``cp * T_ref`` fallback is meaningless for a variable-composition gas).
    """
    from ..composition import enthalpy_mass, species_mass_fractions

    lib = gas.library
    h_max = 0.0
    for sp in specs:
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
    """Per-edge thermo-model ids for the compiled problem (``None`` -> the gas default).

    Perfect gas: every edge follows the gas default (``None``).  Reacting: an explicit edge
    closure (``frozen`` / ``equilibrium``) wins; an ``auto`` edge is frozen (unburnt) upstream of
    an :func:`~fns.elements.catalog.equilibrium_flame` and equilibrium (burnt) downstream of one
    (flood-filled along the flow direction).  With no flame in the network every ``auto`` edge is
    equilibrium (equilibrium-everywhere, the base reacting model).
    """
    n_edges = len(parsed)
    if not reacting:
        return [None] * n_edges

    flame_nodes = {i for i, sp in enumerate(specs) if sp.residual_id == FLAME_EQUILIBRIUM}
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
        if token in _EDGE_CLOSURE:
            models[ei] = _EDGE_CLOSURE[token]
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
    from ..perturbation.boundary_bc import PerturbationBC

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


def _port_of(handle: str) -> int:
    """Extract the integer port ordinal from a UI handle of the form ``...-port-<k>``."""
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
    """Build an ``ElementSpec`` from a UI node, attaching its name and perturbation BC if any."""
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
        if bc is not None:  # else keep the factory default
            spec.perturbation_bc = bc
    return spec


def load_case(path: str):
    """Load a YAML file exported from the UI tool into a ``Network``."""

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

    ui_nodes = model.get("nodes") or []
    ui_edges = model.get("edges") or []
    if not ui_nodes or not ui_edges:
        raise ValueError(f"{path}: the network has no nodes or no edges")

    # Thermo model: a calorically-perfect gas, or the reacting equilibrium model (transported
    # feed-stream mixture fractions slaved to HP equilibrium).
    model_kind = str(g.get("thermoModel") or "perfect_gas")
    case_dir = os.path.dirname(os.path.abspath(path))
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

    # The reacting library is built from the network feeds (automatic slate), so it is
    # resolved after the node specs are parsed.  The closures then need an absolute-enthalpy
    # datum from the feed streams; the perfect-gas default (cp * T_ref) is meaningless for a
    # variable-composition gas.
    if reacting:
        library = _build_library(g, specs, case_dir)
        gas = equilibrium(library, T_init=float(g["equilibriumTInit"]), T_init_frozen=float(g["frozenTInit"]))
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
        so = _port_of(e["sourceHandle"])
        to = _port_of(e["targetHandle"])
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
    # can be saved back for the UI verbatim -- see fns.io.yaml_out.
    net.provenance = UIProvenance(
        doc=doc,
        node_ids=[n["id"] for n in nodes_sorted],
        edge_ids=[e.get("id", f"edge_{i + 1}") for i, e in enumerate(edges_sorted)],
    )
    return net
