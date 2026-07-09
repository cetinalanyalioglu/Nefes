"""Write a Nefes network (and its results) as a UI-readable YAML case.

This is the symmetric counterpart of :mod:`nefes.io.yaml_in`.  It emits the native
``SaveFilePayload`` the FNetLibUI tool reads: ``version``/``timestamp``/``meta``,
the ``model`` (gas + reference scales, ``nodes``, ``edges``), the UI-only
``uiAttributes`` (canvas positions) and ``uiState`` (id counters), and an
optional ``data`` section holding result datasets.

Result data
-----------
The UI binds a dataset's ``values[i]`` to the element whose ``index`` is ``i``,
and validates that an item supplies exactly one value per element of its target
(``node`` or ``edge``).  Nefes mean-flow state lives on edges, so the converged
:class:`~nefes.shell.network.Solution` fields are emitted as **edge** datasets.
Pass ``node_data=True`` to additionally emit each (non-phase) edge field reduced
onto the nodes as the mean over each node's incident edges -- a provision until
genuinely nodal state exists.  Forced-response (acoustic) results are emitted as
one named dataset per selected frequency (``"<f> Hz"``), each carrying the
magnitude and phase of the requested perturbation quantities.  Eigenmode (stability)
results are emitted as one dataset per mode (``"Mode <i>: <f> Hz"``) carrying the
per-edge mode shape, with the modal frequency, growth rate, damping ratio and
residual as dataset metadata.

Animated datasets
-----------------
A dataset may carry a ``frames`` axis (:class:`FrameAxis`): a named frame variable
(e.g. phase or frequency), its unit, and one value per frame.  Every per-frame item
then holds one row of values per frame (``values[k][i]`` binds frame ``k`` to the
element of index ``i``); an item with a flat value list inside an animated dataset
is frame-independent.  Two producers are built in: ``eig_animation=True`` sweeps each
selected eigenmode's instantaneous shape ``Re{psi e^{i theta}}`` over one phase cycle
(frame variable *Phase*, degrees), and ``forced_sweep=True`` folds a forced response's
whole frequency grid into a single dataset (frame variable *Frequency*, Hz) instead of
one snapshot dataset per frequency.  The UI shows a playback control when such a
dataset's item is displayed.

A Nyquist stability result (``nyquist=...``) is summarized as an items-free dataset
carrying the verdict, unstable-mode count, stability margin and onset frequencies as
metadata: the locus is scalar per frequency, not an element-bound field, so it has no
values to color the network with.

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
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import yaml

from ..elements.ids import (
    MASS_FLOW_INLET,
    PT_INLET,
    P_OUTLET,
    MASS_FLOW_OUTLET,
    CHOKED_NOZZLE_OUTLET,
    ISEN_AREA_CHANGE,
    TRANSFER_MATRIX,
    SUDDEN_AREA_CHANGE,
    LOSS,
    JUNCTION,
    SPLITTER,
    DUCT,
    PIPE,
    WALL,
    CAVITY,
    LINEAR_RESISTANCE,
    FLAME_HEAT_RELEASE,
    FLAME_EQUILIBRIUM,
    MASS_SOURCE,
    FORCED_SPLITTER,
    BOUNDARY_RIDS,
    STREAM_INTRODUCING,
)
from ..elements.composite import is_composite
from ..thermo.api import EQ_KERNEL
from .yaml_in import MODEL_ID, EDGE_CLOSURE

# The UI save-file schema version this writer targets (matches yaml_in / the UI).
SAVE_FILE_VERSION = "2.0.0"
DEFAULT_TITLE = "Nefes case"

# Per-edge thermo-model id -> UI edge closure token (the inverse of the reader's map).
_EDGE_CLOSURE_TOKEN = {v: k for k, v in EDGE_CLOSURE.items()}

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
    "W": ("Molar mass", "kg/mol"),
    "cp": ("Specific heat", "J/(kg K)"),
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
    generically as ``label : value unit`` with an optional description
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
class FrameAxis:
    """The frame axis of an animated dataset: a named variable with one value per frame.

    Attributes
    ----------
    variable : str
        Display name of the frame variable (e.g. ``"Phase"``, ``"Frequency"``).
    values : list of float
        The frame variable's value at each frame, in playback order.
    unit : str, optional
        Unit string shown next to the frame value in the UI player.
    """

    variable: str
    values: List[float]
    unit: str = ""

    def to_dict(self) -> dict:
        out = {"variable": self.variable}
        if self.unit:
            out["unit"] = self.unit
        out["values"] = [float(v) for v in self.values]
        return out


def _is_per_frame(values) -> bool:
    """Whether an item's ``values`` holds per-frame rows rather than a flat series."""
    return len(values) > 0 and isinstance(values[0], (list, tuple, np.ndarray))


@dataclass
class DataItem:
    """One result series: a value per element of ``target`` (``node``/``edge``).

    Attributes
    ----------
    name : str
        UI display name.
    target : str
        ``"node"`` or ``"edge"``; selects which elements the values color.
    values : list of float, or list of list of float
        One value per element, ordered by element index.  Inside an animated
        dataset (a :class:`DataSet` with a :class:`FrameAxis`) an item may
        instead hold one such row per frame.
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
        if _is_per_frame(self.values):
            out["values"] = [[float(v) for v in row] for row in self.values]
        else:
            out["values"] = [float(v) for v in self.values]
        return out


@dataclass
class DataSet:
    """A named group of :class:`DataItem` s, embedded under ``data.datasets``.

    Carries optional self-describing metadata (:attr:`description` and an ordered
    list of :class:`MetaEntry` :attr:`info`) the UI renders read-only.  With a
    :attr:`frames` axis the dataset is *animated*: each per-frame item holds one
    row of element values per frame and the UI offers playback over the frames.
    """

    name: str
    items: List[DataItem] = field(default_factory=list)
    include_in_save: bool = True
    description: str = ""
    info: List[MetaEntry] = field(default_factory=list)
    frames: Optional[FrameAxis] = None

    def _validate_frames(self):
        if self.frames is None:
            for it in self.items:
                if _is_per_frame(it.values):
                    raise ValueError(
                        f"dataset {self.name!r}: item {it.name!r} holds per-frame rows but the dataset "
                        "carries no frames axis (set DataSet.frames)"
                    )
            return
        n_frames = len(self.frames.values)
        if n_frames == 0:
            raise ValueError(f"dataset {self.name!r}: the frames axis is empty")
        for it in self.items:
            if not _is_per_frame(it.values):
                continue  # a flat series inside an animated dataset is frame-independent
            if len(it.values) != n_frames:
                raise ValueError(
                    f"dataset {self.name!r}: item {it.name!r} has {len(it.values)} frame rows "
                    f"but the frames axis has {n_frames} values"
                )
            if len({len(row) for row in it.values}) > 1:
                raise ValueError(f"dataset {self.name!r}: item {it.name!r} has frame rows of unequal length")

    def to_dict(self, ds_id: str) -> dict:
        self._validate_frames()
        out = {"id": ds_id, "name": self.name, "includeInSave": bool(self.include_in_save)}
        if self.description:
            out["description"] = self.description
        if self.info:
            out["info"] = [e.to_dict() for e in self.info]
        if self.frames is not None:
            out["frames"] = self.frames.to_dict()
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
    forced_sweep=False,
    eigenmodes=None,
    eig_modes=None,
    eig_fields=("p", "u"),
    eig_animation=False,
    eig_frames=36,
    nyquist=None,
    title=None,
    extra_datasets=None,
    include_in_save=True,
    mean_flow_name="Mean flow",
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
    forced_sweep : bool, optional
        Emit each forced response as a single *animated* dataset (one frame per
        frequency, frame variable *Frequency* in Hz) instead of one snapshot
        dataset per frequency.  Default ``False``.
    eigenmodes : EigenmodeResult, optional
        A stability (eigenmode) result to snapshot.  Each mode is emitted as its own
        ``"Mode <i>: <f> Hz"`` dataset carrying the per-edge mode-shape magnitude and
        phase, with the modal frequency, growth rate, damping ratio, residual and
        unstable flag as dataset metadata.  The mode-shape amplitude is relative (the
        eigenvector's arbitrary normalization).
    eig_modes : sequence of int, optional
        Which mode indices to snapshot (default: every mode in ``eigenmodes``).
    eig_fields : sequence of str, optional
        Perturbation quantities per mode (default ``("p", "u")``); keys of
        :data:`_FORCED_FIELDS`.  Magnitude and phase are emitted for each.
    eig_animation : bool, optional
        Additionally emit each selected mode as an *animated* dataset
        (``"Mode <i>: <f> Hz animation"``): the instantaneous shape
        ``Re{psi e^{i theta}}`` per edge, swept over one phase cycle (frame
        variable *Phase* in degrees).  Default ``False``.
    eig_frames : int, optional
        Phase frames per cycle for ``eig_animation`` (default 36; the cycle
        endpoint is excluded so looped playback is seamless).
    nyquist : NyquistResponse or list of NyquistResponse, optional
        Nyquist stability result(s) to summarize as an items-free dataset
        (verdict, unstable-mode count, margin, onset frequencies as metadata).
        The locus itself is scalar per frequency -- not an element-bound field --
        so no values are emitted.
    title : str, optional
        Case title (``meta.title``).  Defaults to the loaded title or
        ``"Nefes case"``.
    extra_datasets : list of DataSet, optional
        Fully custom datasets appended as-is (e.g. genuinely nodal data the user
        supplies directly).
    include_in_save : bool, optional
        The ``includeInSave`` flag stamped on generated datasets (default
        ``True``).
    mean_flow_name : str, optional
        Name for the mean-flow dataset built from ``solution`` (default ``"Mean flow"``); the
        companion chemistry dataset, if any, is named ``"<mean_flow_name> chemistry"``.

    Returns
    -------
    str
        The YAML document.
    """
    datasets = _build_datasets(
        network,
        solution,
        fields,
        node_data,
        forced,
        forced_freqs,
        forced_fields,
        forced_sweep,
        eigenmodes,
        eig_modes,
        eig_fields,
        eig_animation,
        eig_frames,
        nyquist,
        include_in_save,
        mean_flow_name,
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


def save_solution(network, solution, path: str, *, dataset: str = "Mean flow", **kwargs) -> None:
    """Write or append a solved network to a UI-readable YAML case.

    If ``path`` does not exist, writes a fresh case (network + this solution's datasets).  If it
    exists, the file is expected to already hold this same network; the solution's datasets are
    *appended* under the given ``dataset`` name so several solutions can be overlaid in one file.

    Parameters
    ----------
    network : Network
        The flow network (must match the network already stored in an existing file).
    solution : Solution
        The converged solution whose fields are embedded.
    path : str
        Destination ``.yaml`` path; appended to when it already exists.
    dataset : str, optional
        Name for this solution's mean-flow dataset (default ``"Mean flow"``).  Raises if a dataset
        of that name (or its companion chemistry dataset) is already present in the file.
    **kwargs
        Forwarded to :func:`dump_case` (e.g. ``fields``, ``node_data``, ``forced``).
    """
    if not os.path.exists(path):
        save_case(network, path, solution=solution, mean_flow_name=dataset, **kwargs)
        return

    import yaml as _yaml

    with open(path, "r") as fh:
        doc = _yaml.safe_load(fh)
    _require_matching_topology(doc, network, path)

    fresh = _yaml.safe_load(dump_case(network, solution=solution, mean_flow_name=dataset, **kwargs))
    new_datasets = (fresh.get("data") or {}).get("datasets", [])
    existing = doc.setdefault("data", {}).setdefault("datasets", [])
    existing_names = {d.get("name") for d in existing}
    for d in new_datasets:
        if d.get("name") in existing_names:
            raise ValueError(f"{path}: a dataset named {d.get('name')!r} already exists in this case")
    # Renumber the appended dataset / item ids so they continue after the existing ones.
    base = len(existing)
    for i, d in enumerate(new_datasets):
        d["id"] = f"ds-{base + i}-{_slug(d.get('name'))}"
        for j, item in enumerate(d.get("items", [])):
            item["id"] = f"{d['id']}-item-{j}"
    existing.extend(new_datasets)
    with open(path, "w") as fh:
        fh.write(_yaml_dump(doc))


def _require_matching_topology(doc, network, path):
    """Raise if the case in ``doc`` does not have the same node/edge counts as ``network``."""
    model = doc.get("model") if isinstance(doc, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"{path}: not a UI save file (no 'model' section) -- cannot append a solution")
    n_nodes = len(model.get("nodes") or [])
    n_edges = len(model.get("edges") or [])
    if n_nodes != len(network._elements) or n_edges != len(network._edges):
        raise ValueError(
            f"{path}: the stored case has {n_nodes} nodes / {n_edges} edges, but this network has "
            f"{len(network._elements)} / {len(network._edges)} -- append expects the same network"
        )


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _build_datasets(
    network,
    solution,
    fields,
    node_data,
    forced,
    forced_freqs,
    forced_fields,
    forced_sweep,
    eigenmodes,
    eig_modes,
    eig_fields,
    eig_animation,
    eig_frames,
    nyquist,
    include_in_save,
    mean_flow_name="Mean flow",
):
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
        datasets.append(DataSet(mean_flow_name, items, include_in_save, info=info))
        chem = _chemistry_items(network, solution)
        if chem:
            # Namespaced under the mean-flow name so several appended solutions never collide.
            datasets.append(
                DataSet(
                    f"{mean_flow_name} chemistry",
                    chem,
                    include_in_save,
                    info=[MetaEntry("kind", "Analysis", "Chemistry")],
                )
            )
    if forced is not None:
        if forced_sweep:
            datasets += _forced_sweep_datasets(network, forced, forced_freqs, forced_fields, node_data, include_in_save)
        else:
            datasets += _forced_datasets(network, forced, forced_freqs, forced_fields, node_data, include_in_save)
    if eigenmodes is not None:
        datasets += _eigenmode_datasets(network, eigenmodes, eig_modes, eig_fields, node_data, include_in_save)
        if eig_animation:
            datasets += _eigenmode_animation_datasets(
                network, eigenmodes, eig_modes, eig_fields, eig_frames, node_data, include_in_save
            )
    if nyquist is not None:
        datasets += _nyquist_datasets(nyquist, include_in_save)
    return datasets


def _chemistry_items(network, solution):
    """Per-edge composition datasets: feed-stream mixture fractions and species mole fractions.

    Emitted by default whenever the network transports a composition.  Each transported
    feed stream becomes one edge field ``xi:<stream>``; for a reacting network each species
    present anywhere becomes one edge field ``X:<species>`` (mole fraction, ``0`` where the
    species is absent).  Light enough to always include -- the per-edge chemistry the solver
    otherwise hides behind the conserved mixture fractions.
    """
    prob = solution.problem
    n_elem = int(prob.n_elem)
    if n_elem == 0:
        return []
    # Every compiled edge, matching the mean-flow dataset: the composite *internal* edges append
    # at the tail (no UI edge maps to them, so the UI simply ignores those trailing values), and
    # including them lets a saved solution be reloaded as the full solver state (see load_solution).
    n_edges = int(prob.n_edges)
    items = []
    # transported feed-stream mixture fractions (always present when composition is carried)
    for name in prob.scalar_names:
        vals = [float(solution.mixture_fractions(e).get(name, 0.0)) for e in range(n_edges)]
        items.append(DataItem(f"xi:{name}", "edge", vals, ""))
    # transported burnt marker (marker-gated reacting networks): 0 fresh / 1 burnt per edge
    if int(getattr(prob, "marker_row", -1)) >= 0:
        items.append(DataItem("burnt", "edge", [float(solution.marker(e)) for e in range(n_edges)], ""))
    # solved chemical species (reacting only -- a perfect gas carries passive scalars, no species)
    lib = network.gas.library
    if lib is not None:
        per_edge = [solution.species(e, basis="mole") for e in range(n_edges)]
        present = [s for s in (sp.name for sp in lib.species) if any(s in d for d in per_edge)]
        for s in present:
            vals = [float(d.get(s, 0.0)) for d in per_edge]
            items.append(DataItem(f"X:{s}", "edge", vals, ""))
    return items


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


def _forced_sweep_datasets(network, forced, forced_freqs, forced_fields, node_data, include_in_save):
    """One animated dataset per forced response: a frame per (selected) frequency."""
    responses = forced if isinstance(forced, (list, tuple)) else [forced]
    n_edges = len(network._edges)
    datasets = []
    for k, fr in enumerate(responses):
        freqs = np.asarray(fr.freqs, dtype=float)
        idxs = _select_freq_indices(freqs, forced_freqs)
        if not idxs:
            raise ValueError("the forced response carries no frequencies to sweep")
        frame_freqs = [float(freqs[j]) for j in idxs]
        items = []
        for key in forced_fields:
            # per-frame snapshot items (magnitude + phase per field), folded into frame rows
            per_frame = [_forced_items(fr, j, key, n_edges) for j in idxs]
            for c, proto in enumerate(per_frame[0]):
                rows = [frame[c].values for frame in per_frame]
                items.append(DataItem(proto.name, proto.target, rows, proto.unit, proto.phase))
        if node_data:
            items += _node_items(network, items)
        info = [
            MetaEntry("kind", "Analysis", "Forced response sweep"),
            MetaEntry("n_freqs", "Frequencies", len(frame_freqs)),
            MetaEntry("f_min", "Band start", frame_freqs[0], unit="Hz"),
            MetaEntry("f_max", "Band end", frame_freqs[-1], unit="Hz"),
        ]
        name = "Frequency sweep" if len(responses) == 1 else f"Frequency sweep {k + 1}"
        datasets.append(
            DataSet(
                name,
                items,
                include_in_save,
                description="Forced response over the swept frequencies; one frame per frequency.",
                info=info,
                frames=FrameAxis("Frequency", frame_freqs, "Hz"),
            )
        )
    return datasets


def _eigenmode_info(result, i):
    """The modal scalars of mode ``i`` as dataset metadata entries."""
    return [
        MetaEntry("mode", "Mode index", int(i)),
        MetaEntry("frequency", "Frequency", float(result.freqs[i]), unit="Hz"),
        MetaEntry("growth_rate", "Growth rate", float(result.growth_rates[i]), unit="1/s"),
        MetaEntry("damping_ratio", "Damping ratio", float(result.damping_ratios[i])),
        MetaEntry("unstable", "Unstable", bool(result.unstable[i])),
        MetaEntry("residual", "Residual", float(result.residuals[i])),
    ]


def _select_modes(result, eig_modes):
    n_modes = int(result.n_modes)
    modes = range(n_modes) if eig_modes is None else [int(m) for m in eig_modes]
    for i in modes:
        if not 0 <= i < n_modes:
            raise ValueError(f"eigenmode index {i} out of range [0, {n_modes})")
    return modes


def _eigenmode_datasets(network, result, eig_modes, eig_fields, node_data, include_in_save):
    """One dataset per eigenmode: per-edge mode shape + the modal scalars as metadata."""
    n_edges = len(network._edges)
    datasets = []
    for i in _select_modes(result, eig_modes):
        items = []
        for key in eig_fields:
            items += _eigenmode_items(result, i, key, n_edges)
        if node_data:
            items += _node_items(network, items)
        info = [MetaEntry("kind", "Analysis", "Eigenmode")] + _eigenmode_info(result, i)
        datasets.append(
            DataSet(
                f"Mode {i}: {float(result.freqs[i]):g} Hz",
                items,
                include_in_save,
                description="Eigenmode shape (relative amplitude; eigenvector normalization).",
                info=info,
            )
        )
    return datasets


def _eigenmode_animation_datasets(network, result, eig_modes, eig_fields, eig_frames, node_data, include_in_save):
    """One animated dataset per eigenmode: the instantaneous shape over one phase cycle.

    Each frame ``k`` holds the per-edge instantaneous physical perturbation
    ``Re{psi_e e^{i theta_k}}`` at phase ``theta_k = 2 pi k / n_frames`` (the cycle
    endpoint is excluded so looped playback is seamless).  The amplitude is relative
    (the eigenvector's arbitrary normalization), matching the snapshot datasets.
    """
    n_frames = int(eig_frames)
    if n_frames < 2:
        raise ValueError(f"eig_frames must be at least 2, got {eig_frames}")
    n_edges = len(network._edges)
    thetas = [360.0 * k / n_frames for k in range(n_frames)]
    datasets = []
    for i in _select_modes(result, eig_modes):
        items = []
        for key in eig_fields:
            if key not in _FORCED_FIELDS:
                raise ValueError(f"unknown eigenmode field {key!r}; choose from {list(_FORCED_FIELDS)}")
            label, basis, comp, unit = _FORCED_FIELDS[key]
            shape = result.mode_shape(i, basis)
            cvals = [complex(shape[e, comp]) for e in range(n_edges)]
            rows = [[(c * cmath.exp(1j * math.radians(th))).real for c in cvals] for th in thetas]
            items.append(DataItem(label, "edge", rows, unit))
        if node_data:
            items += _node_items(network, items)
        info = [
            MetaEntry("kind", "Analysis", "Eigenmode animation"),
            *_eigenmode_info(result, i),
            MetaEntry("n_frames", "Phase frames", n_frames),
        ]
        datasets.append(
            DataSet(
                f"Mode {i}: {float(result.freqs[i]):g} Hz animation",
                items,
                include_in_save,
                description=(
                    "Instantaneous mode shape Re{psi e^(i theta)} over one phase cycle "
                    "(relative amplitude; eigenvector normalization)."
                ),
                info=info,
                frames=FrameAxis("Phase", thetas, "deg"),
            )
        )
    return datasets


def _nyquist_datasets(nyquist, include_in_save):
    """Items-free summary dataset(s) for Nyquist stability results.

    The locus ``L(omega)`` / determinant ``D(omega)`` are scalar per frequency, not
    element-bound fields, so only the stability verdict and its scalars are emitted.
    """
    responses = nyquist if isinstance(nyquist, (list, tuple)) else [nyquist]
    datasets = []
    for k, ny in enumerate(responses):
        freqs = np.asarray(ny.freqs, dtype=float)
        n_unstable = int(ny.n_unstable)
        onset = ", ".join(f"{c['freq_hz']:g}" for c in ny.crossings()) or "none"
        info = [
            MetaEntry("kind", "Analysis", "Nyquist stability"),
            MetaEntry("stable", "Stable", n_unstable == 0),
            MetaEntry("n_unstable", "Unstable modes", n_unstable),
            MetaEntry("margin", "Stability margin min|D|", float(ny.margin)),
            MetaEntry("f_max", "Swept band end", float(freqs.max()) if freqs.size else 0.0, unit="Hz"),
            MetaEntry("closed", "Band edge quiet", bool(ny.closed)),
            MetaEntry("rank", "Source rank", int(ny.rank)),
            MetaEntry("isentropic", "Acoustic-only", bool(ny.isentropic)),
            MetaEntry(
                "crossings",
                "Onset frequencies",
                onset,
                unit="Hz" if onset != "none" else "",
                description="Real frequencies where the locus skims the critical point (|D| dips).",
            ),
        ]
        if ny.source_labels:
            info.append(MetaEntry("sources", "Sources", ", ".join(ny.source_labels)))
        name = "Nyquist stability" if len(responses) == 1 else f"Nyquist stability {k + 1}"
        datasets.append(
            DataSet(
                name,
                [],
                include_in_save,
                description="Open-loop (Nyquist) stability verdict over the swept real-frequency band.",
                info=info,
            )
        )
    return datasets


def _eigenmode_items(result, i, key, n_edges):
    if key not in _FORCED_FIELDS:
        raise ValueError(f"unknown eigenmode field {key!r}; choose from {list(_FORCED_FIELDS)}")
    label, basis, comp, unit = _FORCED_FIELDS[key]
    shape = result.mode_shape(i, basis)  # (n_edges, n_char)
    cvals = [complex(shape[e, comp]) for e in range(n_edges)]
    return [
        DataItem(f"{label} amplitude", "edge", [abs(c) for c in cvals], unit),
        DataItem(f"{label} phase", "edge", [math.degrees(cmath.phase(c)) for c in cvals], "deg", phase=True),
    ]


def _node_items(network, items):
    """Reduce each non-phase edge item onto the nodes (mean over incident edges).

    Per-frame items are reduced frame by frame, yielding a per-frame node item.
    """
    incident = _incident_edges(network)

    def reduce_row(row):
        return [sum(row[e] for e in edges) / len(edges) if edges else float("nan") for edges in incident]

    out = []
    for it in items:
        if it.target != "edge" or it.phase:
            continue
        if _is_per_frame(it.values):
            values = [reduce_row(row) for row in it.values]
        else:
            values = reduce_row(it.values)
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
    from ..elements.catalog import ensure_unique_names

    elements = network._elements
    # display labels must identify a node uniquely in the exported case
    ensure_unique_names(elements)
    edges = network._edges
    reacting = network.gas.model_id == EQ_KERNEL
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
        rid = getattr(spec, "residual_id", None)  # None for a composite (never a boundary)
        if rid in BOUNDARY_RIDS:
            attrs.update(_bc_to_ui(getattr(spec, "perturbation_bc", None)))
        if reacting and rid in STREAM_INTRODUCING:
            attrs.update(_composition_to_ui(spec))
        if reacting and float(getattr(spec, "marker", 0.0)) != 0.0:
            # Burnt-gas feed flag: the UI marker is a boolean, so emit a bool (True = burnt).
            # Omitted entirely when fresh (marker 0, the default) -> the UI's default false.
            attrs["marker"] = True
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
        if reacting:
            model = network._edge_models[ei] if ei < len(network._edge_models) else None
            eattrs["thermoModel"] = _EDGE_CLOSURE_TOKEN.get(model, "auto")
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

    return {
        "id": MODEL_ID,
        "globalAttributes": _global_attributes(network, prov),
        "nodes": nodes,
        "edges": edges_out,
    }


def _global_attributes(network, prov=None):
    gas = network.gas
    if gas.model_id == EQ_KERNEL:
        # The reacting model uses the built-in NASA Glenn / CEA database by default, so no file
        # is required.  The *resolved* species slate is emitted explicitly (freezing any
        # automatic reduction, so reload is deterministic and skips re-deriving it).  An explicit
        # native-mechanism path, if one was used, is recovered from the loaded provenance.
        prov_global = (prov.doc.get("model") or {}).get("globalAttributes", {}) if prov else {}
        mech = str(prov_global.get("mechanismFile") or "")
        # The Newton initial-temperature guesses (t_init / t_init_frozen) are solver
        # internals with robust defaults -- not exposed in the UI, so not emitted here.
        g = {
            "thermoModel": "equilibrium",
            # A YAML list, not a joined string: CEA species names carry commas (``C2H2,acetylene``),
            # which a comma/whitespace-delimited string cannot round-trip.
            "species": list(gas.species_names),
        }
        if mech:
            g["mechanismFile"] = mech
    else:
        cp, R = float(gas.tf[0]), float(gas.tf[1])
        g = {
            "thermoModel": "perfect_gas",
            "gasConstant": R,
            "heatCapacityRatio": cp / (cp - R),
        }
    g["referencePressure"] = float(network.p_ref)
    g["referenceTemperature"] = float(network.T_ref)
    g["referenceMassFlow"] = float(network._mdot_ref or 0.0)
    return g


def _composition_to_ui(spec):
    """UI ``composition`` / ``basis`` attributes for a stream-introducing element.

    Serializes the retained ``composition_spec`` back to the ``"species:fraction, ..."`` string
    the UI uses.  Returns ``{}`` when the element carries no composition (e.g. an inert-backflow
    outlet).  Only emitted for the reacting model (the perfect gas ignores composition).
    """
    comp = getattr(spec, "composition_spec", None)
    if comp is None:
        return {}
    if isinstance(comp, dict):
        comp_str = ", ".join(f"{k}:{float(v):g}" for k, v in comp.items())
    else:  # a raw scalar array (uncommon on the reacting backend); best-effort
        comp_str = ", ".join(f"{float(v):g}" for v in comp)
    return {"composition": comp_str, "basis": getattr(spec, "basis", "mole") or "mole"}


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
def _length_attrs(fp, off):
    """UI storage-length attributes (``lengthUpstream``/``lengthDownstream``/``endCorrection``).

    Read from the inline element's fparams storage block beginning at ``off``; emitted only
    when non-zero, so a lengthless element serializes without the extra keys.
    """
    names = ("lengthUpstream", "lengthDownstream", "endCorrection")
    out = {}
    for i, nm in enumerate(names):
        v = float(fp[off + i]) if len(fp) > off + i else 0.0
        if v != 0.0:
            out[nm] = v
    return out


def _manifold_attrs(fp):
    """UI manifold attributes: the chamber ``volume`` (emitted only when non-zero).

    ``fparams = [volume]``; a plain (volume-less) manifold serializes bare.
    """
    out = {}
    if fp and float(fp[0]) != 0.0:
        out["volume"] = float(fp[0])
    return out


# Composite kind -> (UI node type, factory-params -> UI attributes).  The factory params are
# retained on the spec (``CompositeElementSpec.params``), so a composite serializes as the single
# element the user specified -- never its expanded internals.
_COMPOSITE_TO_UI = {
    "orifice": ("Orifice", lambda p: {"throatArea": float(p["throat_area"])}),
    "lossy_nozzle": (
        "LossyNozzle",
        lambda p: {"throatArea": float(p["throat_area"]), "beta": float(p["beta"])},
    ),
    "sudden_contraction": (
        "SuddenContraction",
        lambda p: {"contractionCoefficient": float(p["cc"])},
    ),
    "helmholtz_resonator": (
        "HelmholtzResonator",
        lambda p: {
            "volume": float(p["volume"]),
            "neckLength": float(p["neck_length"]),
            "neckArea": float(p["neck_area"]),
        },
    ),
    "fanno_pipe": (
        "FannoPipe",
        lambda p: {
            "length": float(p["length"]),
            "diameter": float(p["diameter"]),
            "frictionFactor": float(p["friction_factor"]),
            "nSegments": int(p["n_segments"]),
        },
    ),
    "tapered_duct": (
        "TaperedDuct",
        # Full-precision repr, not ``%g``: the taper's downstream station area must reload
        # bit-identical to the external downstream edge area, or the boundary duct's two ports
        # disagree past the equal-area tolerance and the reloaded network fails validation.
        lambda p: {"areaProfile": ", ".join(f"{float(x)!r}:{float(a)!r}" for x, a in p["stations"])},
    ),
}


def _spec_to_ui(spec):
    """Map an ``ElementSpec`` to its UI ``(type, modeled-attributes)`` pair."""
    if is_composite(spec):
        entry = _COMPOSITE_TO_UI.get(spec.kind)
        if entry is None or not getattr(spec, "params", None):
            raise ValueError(
                f"composite element {spec.name!r} ({spec.kind!r}) cannot be serialized to the UI format; "
                "known composite kinds are: " + ", ".join(sorted(_COMPOSITE_TO_UI))
            )
        ui_type, to_attrs = entry
        return ui_type, to_attrs(spec.params)
    rid = spec.residual_id
    fp = spec.fparams
    if rid == MASS_FLOW_INLET:
        return "MassFlowInlet", {"massFlowRate": float(fp[0]), "totalTemperature": float(fp[1])}
    if rid == PT_INLET:
        return "TotalPressureInlet", {"totalPressure": float(fp[0]), "totalTemperature": float(fp[1])}
    if rid == P_OUTLET:
        return "PressureOutlet", {"pressure": float(fp[0]), "backflowTotalTemperature": float(fp[1])}
    if rid == MASS_FLOW_OUTLET:
        return "MassFlowOutlet", {"massFlowRate": float(fp[0])}
    if rid == CHOKED_NOZZLE_OUTLET:
        return "ChokedNozzleOutlet", {"throatArea": float(fp[0])}
    if rid == ISEN_AREA_CHANGE:
        return "IsentropicAreaChange", _length_attrs(fp, 0)
    if rid == TRANSFER_MATRIX:
        # The frequency-domain descriptor is a Python object with no YAML form;
        # only the element's place in the topology round-trips.
        if spec.transfer_matrix is not None:
            warnings.warn(
                f"transfer-matrix element {spec.name!r}: the attached descriptor is not serializable "
                "to the UI format; re-attach it (spec.transfer_matrix = ...) after loading",
                stacklevel=6,
            )
        return "TransferMatrix", {}
    if rid == SUDDEN_AREA_CHANGE:
        return "SuddenAreaChange", {"contractionCoefficient": float(fp[0]), **_length_attrs(fp, 1)}
    if rid == LOSS:
        return "LossElement", {"lossCoefficient": float(fp[0]), **_length_attrs(fp, 2)}
    if rid == LINEAR_RESISTANCE:
        return "LinearResistance", {"resistance": float(fp[0]), **_length_attrs(fp, 1)}
    if rid == JUNCTION:
        return "JunctionStaticP", _manifold_attrs(fp)
    if rid == SPLITTER:
        return "LosslessSplitter", _manifold_attrs(fp)
    if rid == FORCED_SPLITTER:
        # fparams are the N-1 controlled outflow betas, in port order
        return "ForcedSplitter", {"fractions": ", ".join(f"{float(b):g}" for b in fp)}
    if rid == DUCT:
        return "Duct", {"length": float(fp[0]) if fp else 0.0}
    if rid == PIPE:
        return "Pipe", {"length": float(fp[0]), "diameter": float(fp[1]), "frictionFactor": float(fp[2])}
    if rid == WALL:
        return "Wall", {}
    if rid == CAVITY:
        return "Cavity", {"volume": float(fp[0])}
    if rid == FLAME_HEAT_RELEASE:
        return "HeatReleaseFlame", {"heatRelease": float(fp[0])}
    if rid == FLAME_EQUILIBRIUM:
        return "EquilibriumFlame", {}
    if rid == MASS_SOURCE:
        # fparams = [mdot, u_inj, T]; the composition is added separately (reacting only).
        return "MassSource", {
            "massFlowRate": float(fp[0]),
            "injectionVelocity": float(fp[1]),
            "injectionTemperature": float(fp[2]),
        }
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
        rid = getattr(spec, "residual_id", None)  # None for a composite (fixed 2-port in the UI)
        inc, out = incoming[nd], outgoing[nd]
        for k, ei in enumerate(inc):
            tgt_ord[ei] = k
        for k, ei in enumerate(out):
            src_ord[ei] = len(inc) + k
        if rid == JUNCTION:
            port_attrs[nd] = {"leftPorts": len(inc), "rightPorts": len(out)}
        elif rid in (SPLITTER, FORCED_SPLITTER):
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
