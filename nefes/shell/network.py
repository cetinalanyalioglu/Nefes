"""Network builder: add elements, connect them with directed edges, then compile or solve."""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from nefes.thermo.constants import R_UNIVERSAL

from ..thermo.configure import ThermoConfig, perfect_gas
from ..thermo.api import PERFECT_GAS, EQ_KERNEL
from ..graph.connectivity import build_connectivity
from ..elements import catalog as cat
from ..elements.catalog import ElementSpec
from . import checks
from .build import build_problem, build_problem_from_connectivity
from ..elements.composite import is_composite, CompositeView
from ..elements.ids import ELEMENT_TYPE_NAMES, CHOKED_NOZZLE_OUTLET
from .problem import CompiledProblem
from ..solver import solve as _solve
from ..solver.control import initial_guess
from ..solver.report import states_table, print_states, residual_breakdown, print_residuals
from ..assembly.recover import ES_MDOT, ES_P, ES_HT, ES_RHO, ES_U, ES_T, ES_C, ES_M, ES_PT, ES_AREA, ES_W, ES_CP

# ES for "edge state"
_EDGE_FIELDS = {
    "mdot": ES_MDOT,
    "p": ES_P,
    "h_t": ES_HT,
    "rho": ES_RHO,
    "u": ES_U,
    "T": ES_T,
    "c": ES_C,
    "M": ES_M,
    "p_t": ES_PT,
    "area": ES_AREA,
    "W": ES_W,
    "cp": ES_CP,
}

# Cap on per-table rows in the Network repr (elements / edges); larger networks are truncated.
_REPR_MAX_ROWS = 20


class Network:
    """The main object for building and solving flow networks.

    A network can be built incrementally with :meth:`add` / :meth:`connect`, specified
    complete in one shot via the ``nodes`` / ``edges`` constructor arguments, or loaded from
    a saved case with :meth:`from_yaml` / :meth:`from_dict`.  Call :meth:`solve` for the
    steady mean flow (a :class:`Solution`) or :meth:`compile` / :attr:`problem` for the
    immutable compiled problem.  Write it back out with :meth:`to_yaml`.
    """

    def __init__(
        self,
        gas: Optional[ThermoConfig] = None,
        nodes=None,
        edges=None,
        *,
        edge_models=None,
        require_connected=None,
        **refs,
    ):
        """Create a network, optionally fully specified in one shot.

        The three positional arguments -- ``gas``, ``nodes`` and ``edges`` -- are the whole
        interface for the common one-shot case; the one-shot form supersedes the lower-level
        :func:`nefes.shell.build.build_problem`.

        Parameters
        ----------
        gas : ThermoConfig, optional
            The thermodynamic model (default: dry-air perfect gas).
        nodes : sequence of ElementSpec, optional
            The elements, in node order -- attached via :meth:`add`.
        edges : sequence of tuple, optional
            Directed edges referencing node indices, attached via :meth:`connect`.  Each is
            ``(tail, head, area)`` or, to pin the local ports, ``(tail, head, area, tail_port,
            head_port)``; ports left unspecified are auto-assigned in attachment order.
        edge_models : sequence of int, optional
            Advanced per-edge thermo-model id override aligned with ``edges`` (a hard
            frozen/equilibrium closure); ``None`` entries use the gas default.  Normally left
            unset -- a reacting network with an equilibrium flame gates the closure
            automatically off the transported burnt marker.
        require_connected : bool, optional
            Reject a model that splits into disconnected sub-networks.  ``None`` (default)
            follows the process-wide :data:`nefes.shell.checks.CHECK_CONNECTED` toggle; pass
            ``True`` / ``False`` to force it for this network.

        Other Parameters
        ----------------
        p_ref : float, optional
            Absolute-pressure gauge reference [Pa] (default 101325).
        T_ref : float, optional
            Reference temperature [K] for the initial guess (default 300).
        mdot_ref, h_ref : float, optional
            Seed overrides for the residual scaling; normally auto-derived and re-measured
            during the solve, so rarely set.

        Notes
        -----
        The reference scales (``p_ref``, ``T_ref``, ``mdot_ref``, ``h_ref``) are keyword-only
        advanced overrides accepted through ``**refs``: the casual user leaves them alone and
        they are auto-derived, while an advanced user can still pin any of them by name.
        """
        self.gas = gas if gas is not None else perfect_gas()
        self.require_connected = require_connected
        self.p_ref = refs.pop("p_ref", 101325.0)
        self.T_ref = refs.pop("T_ref", 300.0)
        self._mdot_ref = refs.pop("mdot_ref", None)
        # Explicit absolute-enthalpy datum; if None, falls back to ``cp * T_ref`` (perfect-gas convention).
        # Reacting closures need the gas's absolute-enthalpy reference here instead.
        self._h_ref = refs.pop("h_ref", None)
        if refs:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(refs))}")
        self._elements: List[ElementSpec] = []
        self._edges: List[Tuple[int, int, float]] = []
        self._ports: List[Tuple[Optional[int], Optional[int]]] = []
        self._edge_names: List[str] = []
        # Per-edge thermo-model override (None -> the gas config's model on that edge).
        self._edge_models: List[Optional[int]] = []
        # Provenance metadata for the network (e.g. from the UI).
        self.provenance = None
        # Lazily compiled problem, invalidated by any topology change (see ``_invalidate``).
        self._compiled: Optional[CompiledProblem] = None

        for spec in nodes or ():
            self.add(spec)
        if edge_models is not None and edges is None:
            raise ValueError("edge_models was given without edges")
        edges = list(edges or ())
        models = list(edge_models) if edge_models is not None else [None] * len(edges)
        if len(models) != len(edges):
            raise ValueError(f"edge_models has {len(models)} entries but there are {len(edges)} edges")
        for edge, model in zip(edges, models):
            # Accept a bare (tail, head, area) or a port-pinned (tail, head, area, tail_port, head_port).
            tail, head, area = edge[0], edge[1], edge[2]
            tail_port = edge[3] if len(edge) > 3 else None
            head_port = edge[4] if len(edge) > 4 else None
            self.connect(tail, head, area, tail_port=tail_port, head_port=head_port, edge_model=model)

    # -- construction -------------------------------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Network":
        """Build a network from a saved UI/YAML case file.

        A convenience so callers need not reach into :mod:`nefes.io`; equivalent to
        :func:`nefes.io.load_case`.

        Parameters
        ----------
        path : str
            Path to a ``.yaml`` case file (as written by :meth:`to_yaml`).

        Returns
        -------
        Network
        """
        from ..io import load_case

        return load_case(path)

    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        """Build a network from an in-memory case dictionary.

        The dictionary is the parsed form of the same schema :meth:`from_yaml` reads, so this
        is the file-less equivalent of :meth:`from_yaml`.

        Parameters
        ----------
        data : dict
            A case document (a ``model`` section with ``nodes`` / ``edges``, as in a YAML case).

        Returns
        -------
        Network
        """
        from ..io import case_from_dict

        return case_from_dict(data)

    def _invalidate(self) -> None:
        """Drop the cached compiled problem after a topology change."""
        self._compiled = None

    def add(self, spec: ElementSpec) -> int:
        """Add an element and return its node index.

        The element names are required to be unique. If a non-unique name is provided in the ElementSpec,
        it will be made unique by appending a number.
        """
        taken = {el.name for el in self._elements}
        base = spec.name or ""
        spec.name = cat.unique_name(base, taken, always_number=base in cat.default_name_bases())
        self._elements.append(spec)
        self._invalidate()
        return len(self._elements) - 1

    def connect(
        self, tail: int, head: int, area: float, name: str = "", *, tail_port=None, head_port=None, edge_model=None
    ) -> int:
        """Add a directed edge from element `tail` to element `head`, returning its edge id.

        The returned integer is the edge index in the compiled problem -- capture it to wire a
        dynamic source's ``ref_edge`` (e.g. the edge just upstream of a flame) without guessing.

        `tail_port`/`head_port` pin the local port indices at each endpoint; leave them `None` to let the
        compiler auto-assign ports in attachment order.

        `edge_model` is an advanced, keyword-only override of the per-edge thermo-model id (a hard
        frozen/equilibrium closure); leave it `None` to use the gas config's default model -- a reacting
        network gates the frozen/equilibrium split automatically off the transported burnt marker.
        """
        idx = len(self._edges)
        self._edges.append((tail, head, float(area)))
        self._ports.append((tail_port, head_port))
        self._edge_models.append(None if edge_model is None else int(edge_model))
        # Edge name is optional, defaulting to "e<index>".
        self._edge_names.append(name or f"e{idx}")
        self._invalidate()
        return idx

    def edge_between(self, tail: int, head: int) -> int:
        """Return the id of the directed edge from element `tail` to element `head`.

        A convenience for recovering an edge index after assembly (e.g. to set a dynamic source's
        ``ref_edge``) when the value returned by :meth:`connect` was not captured.  Raises if no such
        edge exists, or if more than one connects the same ordered pair.
        """
        matches = [i for i, (t, h, _a) in enumerate(self._edges) if t == tail and h == head]
        if not matches:
            raise ValueError(f"no edge from element {tail} to element {head}")
        if len(matches) > 1:
            raise ValueError(f"multiple edges from element {tail} to element {head}: {matches}")
        return matches[0]

    def set_dynamic_source(self, node: int, source) -> int:
        """Attach (or replace) the dynamic-source descriptor on an *already-added* element.

        Parameters
        ----------
        node : int
            Element index (as returned by :meth:`add`) to carry the source.
        source : DynamicSource or None
            The descriptor (e.g. from :func:`nefes.elements.dynamic_source.n_tau_flame`); ``None`` clears it.

        Returns
        -------
        int
            The same ``node``, for chaining.
        """
        self._elements[node].dynamic_source = source
        self._invalidate()
        return node

    def _seed_h(self) -> float:
        """Seed enthalpy scale threaded into the compiled ``var_scale`` (an explicit override or auto).

        Only the *seed* for the residual scaling -- the solve re-measures the enthalpy scale from the
        realized inflow, and the reacting initial guess seeds each edge from its feed enthalpy, so this
        need only be order-of-magnitude right.  Auto-derivation is the perfect-gas ``cp * T_ref``; an
        explicit ``h_ref=`` (as the reacting backend supplies) overrides it.
        """
        if self._h_ref is not None:
            return self._h_ref
        return self.gas.tf[0] * self.T_ref

    def _seed_mdot(self) -> float:
        """Seed mass-flow scale threaded into the compiled ``var_scale`` (an explicit override or auto).

        Only the *seed* for the residual scaling -- the solve re-measures it from the realized inflow at
        each continuation stage (``adaptive_scale``) -- so it need only be order-of-magnitude right.
        Auto-derivation: the **total** specified inflow when every inlet is a mass-flow inlet; otherwise
        a dP-based isentropic estimate ``A * sqrt(2 rho dP_max)`` from the boundary pressures; a quiescent
        / pressureless network falls back to an M=0.3 estimate.  An explicit ``mdot_ref=`` overrides it.
        """
        if self._mdot_ref is not None:
            return self._mdot_ref
        # ``getattr`` guards the composite specs (which carry no ``residual_id``); a composite
        # is never an inlet, so skipping it is correct.
        mass = [abs(el.fparams[0]) for el in self._elements if getattr(el, "residual_id", None) == cat.MASS_FLOW_INLET]
        has_pt = any(getattr(el, "residual_id", None) == cat.PT_INLET for el in self._elements)
        # every inlet a mass-flow inlet -> the total specified inflow is exactly known.
        if mass and not has_pt and sum(mass) > 0.0:
            return sum(mass)
        a_med = float(np.median([a for (_t, _h, a) in self._edges]))
        rho = self.p_ref / (self.gas.tf[1] * self.T_ref) if self.gas.model_id == PERFECT_GAS else 1.0
        # pressure-driven: an isentropic mass-flux estimate from the largest boundary dP.
        dp = self._boundary_dp()
        if dp > 0.0:
            return a_med * np.sqrt(2.0 * rho * dp)
        # quiescent / no pressure spread: the M=0.3 fallback (perfect gas) or a unit scale.
        if self.gas.model_id == PERFECT_GAS:
            cp, R = self.gas.tf[0], self.gas.tf[1]
            c = np.sqrt((cp / (cp - R)) * R * self.T_ref)
            return 0.3 * rho * c * a_med
        return max(sum(mass), 1.0)

    def _boundary_dp(self) -> float:
        """Largest a-priori pressure drop across the boundary pressure references (0 if < 2)."""
        refs = [
            el.fparams[0] for el in self._elements if getattr(el, "residual_id", None) in (cat.PT_INLET, cat.P_OUTLET)
        ]
        return (max(refs) - min(refs)) if len(refs) >= 2 else 0.0

    # -- compile / solve ----------------------------------------------------------------------------------------------

    def _resolve_edge_models(self):
        """Return the per-edge thermo-model array, or ``None`` if every edge uses the gas default."""
        if all(m is None for m in self._edge_models):
            return None
        default = int(self.gas.model_id)
        return np.array([default if m is None else m for m in self._edge_models], dtype=np.int64)

    @property
    def problem(self) -> CompiledProblem:
        """The compiled problem for the current topology, built on first access and cached.

        Most callers never need the compiled object directly -- :meth:`solve` and the
        :class:`Solution` it returns cover the common path -- but it is here for the lower-level
        routines that take a ``CompiledProblem``.  The cache is dropped whenever the network is
        mutated (:meth:`add` / :meth:`connect` / :meth:`set_dynamic_source`), so it always
        reflects the live topology.
        """
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    def compile(self) -> CompiledProblem:
        """Compile the elements and edges into an immutable ``CompiledProblem`` and cache it.

        Rebuilds unconditionally (refreshing the :attr:`problem` cache); prefer :attr:`problem`
        when a cached compile is enough.
        """
        self._compiled = self._build()
        return self._compiled

    def _build(self) -> CompiledProblem:
        """Assemble a fresh ``CompiledProblem`` from the current elements and edges."""
        edge_models = self._resolve_edge_models()
        mdot_ref, h_ref = self._seed_mdot(), self._seed_h()
        # If the ports are explicitly set, use the connectivity builder.
        explicit = self._edges and all(tp is not None and hp is not None for (tp, hp) in self._ports)
        if explicit:
            endpoints = [(t, int(tp), h, int(hp)) for (t, h, _a), (tp, hp) in zip(self._edges, self._ports)]
            conn = build_connectivity(len(self._elements), endpoints)
            area = np.array([a for (_t, _h, a) in self._edges], dtype=np.float64)
            return build_problem_from_connectivity(
                self.gas,
                self._elements,
                conn,
                area,
                mdot_ref,
                self.p_ref,
                h_ref,
                edge_models=edge_models,
                require_connected=self.require_connected,
            )
        return build_problem(
            self.gas,
            self._elements,
            self._edges,
            mdot_ref,
            self.p_ref,
            h_ref,
            edge_models=edge_models,
            require_connected=self.require_connected,
        )

    def solve(self, x0=None, **kw) -> "Solution":
        """Compile and solve the steady mean flow, returning a ``Solution``.

        Keyword arguments
        ------------------
        x0 : ndarray, optional
            Initial state, shape ``(3, E)`` (default: a uniform co-directional guess).
        tol : float, optional
            Convergence tolerance on the scaled residual 2-norm (default ``1e-10``).
        max_iter : int, optional
            Maximum Newton iterations per continuation stage (default ``80``).
        kappa_stages : sequence of float, optional
            Artificial-resistance continuation schedule, warm-started in order (default ``(0.1, 0.01, 0.0)``).
        verbose : int or bool, optional
            Progress verbosity (default ``0``). ``0``/``False`` is silent; ``1``/``True`` prints a one-line
            gross-residual summary per continuation stage; ``2`` additionally prints the scaled residual broken down by
            equation kind (mass, pressure, energy, then each composition scalar) every ``progress_interval`` iterations.
        progress_interval : int, optional
            Iteration stride for the per-iteration prints at ``verbose >= 2`` (default ``1``).

        Returns
        -------
        Solution
            The converged mean-flow result with named edge-field access.
        """
        prob = self.compile()
        res = _solve(prob, x0=x0, **kw)
        sol = Solution(self, prob, res)
        if res.converged:
            for message in sol.verify():
                warnings.warn(message, stacklevel=2)
        return sol

    def initial_guess(self, **kw):
        """Return the solver's initial state guess for the compiled problem."""
        return initial_guess(self.compile(), **kw)

    def plot(self, **kwargs):
        """Draw the network as a node/edge diagram (Plotly).

        A structural view by default: element indices/names and edge directions, with each edge's arrow
        **width scaled by its area** (``width_by="area"``), so the geometry reads at a glance. Pass
        ``width_by=None`` for uniform arrows, or another field (with a converged ``solution=``) to weight
        by it instead; ``color_by`` similarly tints the edges, and :meth:`Solution.plot` is the same diagram
        driven from a solution. Thin wrapper over :func:`nefes.plotting.plot_network_topology`; see it for
        the full keyword set.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_network_topology

        # Default the arrow width to edge area (geometry-weighted); the caller can override or disable it.
        kwargs.setdefault("width_by", "area")
        return plot_network_topology(self, **kwargs)

    def to_yaml(self, path: str, **kwargs) -> None:
        """Write this network as a UI-readable YAML case (no result data).

        The inverse of :meth:`from_yaml`.  Thin wrapper over :func:`nefes.io.save_case`; see it for
        the full set of keyword options.  To embed solved fields as well, use
        :meth:`Solution.to_yaml`.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`nefes.io.save_case`.
        """
        from ..io import save_case

        save_case(self, path, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Alias for :meth:`to_yaml` (kept for intuitive usage)."""
        self.to_yaml(path, **kwargs)

    # -- display ------------------------------------------------------------------------------------------------------

    def _gas_summary(self) -> str:
        """One-line description of the thermo model (gas, scalars/streams)."""
        g = self.gas
        if g.model_id == PERFECT_GAS:
            cp, R = float(g.tf[0]), float(g.tf[1])
            gamma = cp / (cp - R)
            text = f"perfect gas (R={R:.6g} J/kg/K, gamma={gamma:.4g})"
            if g.n_elem:
                text += f" + {g.n_elem} passive scalar(s): {', '.join(g.element_names)}"
            return text
        if g.model_id == EQ_KERNEL:
            text = f"equilibrium ({g.n_species} species)"
            # Streams are discovered at build time, so the labels may not be populated yet.
            if g.element_names:
                text += f", streams: {', '.join(g.element_names)}"
            return text
        return f"model #{g.model_id}"

    def _refs(self):
        """``(p_ref, T_ref, mdot_seed_or_None, mdot_is_explicit)`` for the repr headers."""
        try:
            # The auto-derive medians the edge areas; an edge-less network yields a quiet NaN
            # (suppress numpy's "mean of empty slice" warning -- we report it as "n/a" below).
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = float(self._seed_mdot())
            if not np.isfinite(m):
                m = None
        except Exception:
            m = None
        return self.p_ref, self.T_ref, m, self._mdot_ref is not None

    def _node_label(self, i: int) -> str:
        """Compact identifier for an element endpoint, used in the edge listing."""
        if 0 <= i < len(self._elements):
            return self._elements[i].name or f"#{i}"
        return f"#{i}"

    @staticmethod
    def _type_name(el: ElementSpec) -> str:
        """Human-readable residual-type name for an element (or the kind for a composite)."""
        if is_composite(el):
            return el.kind or "composite"
        return ELEMENT_TYPE_NAMES.get(el.residual_id, f"residual#{el.residual_id}")

    def __repr__(self) -> str:
        """Compact text summary: size, thermo model, references, and the element / edge listings.

        Elements that carry a dynamic ``S(omega)`` source are flagged with a trailing ``*``; a
        footnote reports any per-edge thermo-model overrides.  Each listing is truncated past
        ``_REPR_MAX_ROWS`` rows.
        """
        n_el, n_ed = len(self._elements), len(self._edges)
        lines = [
            f"Network: {n_el} element{'' if n_el == 1 else 's'}, {n_ed} edge{'' if n_ed == 1 else 's'}",
            f"  gas: {self._gas_summary()}",
        ]
        p, T, m, explicit = self._refs()
        mdot = "n/a" if m is None else f"{m:.4g} kg/s ({'explicit' if explicit else 'auto'})"
        lines.append(f"  refs: p={p:.6g} Pa, T={T:.6g} K, mdot={mdot}")

        n_src = sum(1 for el in self._elements if getattr(el, "dynamic_source", None) is not None)
        n_ovr = sum(1 for mdl in self._edge_models if mdl is not None)

        if n_el:
            lines.append("")
            rows = [
                (str(i), self._node_label(i), self._type_name(el) + (" *" if el.dynamic_source is not None else ""))
                for i, el in enumerate(self._elements[:_REPR_MAX_ROWS])
            ]
            lines += _text_table(("#", "name", "type"), rows, ("r", "l", "l"), indent=2)
            if n_el > _REPR_MAX_ROWS:
                lines.append(f"    ... ({n_el - _REPR_MAX_ROWS} more)")

        if n_ed:
            lines.append("")
            rows = [
                (str(i), f"{self._node_label(t)} -> {self._node_label(h)}", f"{a:.4g}", self._edge_names[i])
                for i, (t, h, a) in enumerate(self._edges[:_REPR_MAX_ROWS])
            ]
            lines += _text_table(("#", "connection", "area [m^2]", "name"), rows, ("r", "l", "r", "l"), indent=2)
            if n_ed > _REPR_MAX_ROWS:
                lines.append(f"    ... ({n_ed - _REPR_MAX_ROWS} more)")

        notes = []
        if n_src:
            notes.append(f"* = carries a dynamic S(omega) source ({n_src})")
        if n_ovr:
            notes.append(f"{n_ovr} edge(s) carry a per-edge thermo-model override")
        if notes:
            lines.append("")
            lines += [f"  {note}" for note in notes]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML summary for Jupyter: header line plus element and edge listings."""
        n_el, n_ed = len(self._elements), len(self._edges)
        p, T, m, explicit = self._refs()
        mdot = "n/a" if m is None else f"{m:.4g} kg/s ({'explicit' if explicit else 'auto'})"
        parts = [
            f"{n_el} element{'' if n_el == 1 else 's'}",
            f"{n_ed} edge{'' if n_ed == 1 else 's'}",
            self._gas_summary(),
            f"p={p:.6g} Pa, T={T:.6g} K, mdot={mdot}",
        ]
        header = (
            "<div style='font-family:sans-serif;margin-bottom:4px'>"
            "<b>Network</b> &nbsp;&middot;&nbsp; " + " &nbsp;|&nbsp; ".join(parts) + "</div>"
        )

        # Each listing becomes one flex column so the element and edge tables sit side by side.
        blocks = []
        if n_el:
            body = []
            for i, el in enumerate(self._elements[:_REPR_MAX_ROWS]):
                src = (
                    " <span style='color:#2a8a4a' title='carries a dynamic S(omega) source'>&#9733;</span>"
                    if el.dynamic_source is not None
                    else ""
                )
                body.append(
                    [str(i), self._node_label(i), self._type_name(el) + src],
                )
            block = [_caption("Elements"), _html_table(("#", "name", "type"), body, ("right", "left", "left"))]
            if n_el > _REPR_MAX_ROWS:
                block.append(f"<div style='color:#888;font-size:0.85em'>... ({n_el - _REPR_MAX_ROWS} more)</div>")
            blocks.append("".join(block))

        if n_ed:
            body = [
                [str(i), f"{self._node_label(t)} &rarr; {self._node_label(h)}", f"{a:.4g}", self._edge_names[i]]
                for i, (t, h, a) in enumerate(self._edges[:_REPR_MAX_ROWS])
            ]
            block = [
                _caption("Edges"),
                _html_table(("#", "connection", "area [m&sup2;]", "name"), body, ("right", "left", "right", "left")),
            ]
            if n_ed > _REPR_MAX_ROWS:
                block.append(f"<div style='color:#888;font-size:0.85em'>... ({n_ed - _REPR_MAX_ROWS} more)</div>")
            blocks.append("".join(block))

        columns = "".join(f"<div>{b}</div>" for b in blocks)
        flex = "display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap"
        tables = f"<div style='{flex}'>{columns}</div>" if blocks else ""
        return header + tables


def _text_table(headers, rows, align, indent=0):
    """Render an aligned fixed-width text table as a list of lines (no trailing newline)."""
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    pad = " " * indent

    def fmt(cells):
        out = []
        for cell, w, a in zip(cells, widths, align):
            out.append(str(cell).rjust(w) if a == "r" else str(cell).ljust(w))
        return pad + "  " + "  ".join(out).rstrip()

    return [fmt(headers)] + [fmt(r) for r in rows]


def _caption(text):
    """Small bold caption above a repr table (labels the side-by-side element/edge columns)."""
    return f"<div style='font-family:sans-serif;font-size:0.85em;font-weight:bold;margin-bottom:2px'>{text}</div>"


def _html_table(headers, rows, align):
    """Render an HTML table (eigenmode-repr styling) from header and row cell lists."""
    th = "padding:2px 8px;border-bottom:1px solid #ccc"
    head = "<tr>" + "".join(f"<th style='text-align:{a};{th}'>{h}</th>" for h, a in zip(headers, align)) + "</tr>"
    body = [
        "<tr>" + "".join(f"<td style='text-align:{a};padding:2px 8px'>{c}</td>" for c, a in zip(r, align)) + "</tr>"
        for r in rows
    ]
    return (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em;margin-bottom:6px'>"
        + head
        + "".join(body)
        + "</table>"
    )


@dataclass
class Solution:
    """Converged mean-flow result with named edge-field access.

    Key methods
    -----------
    field(name), edge(e)
        Read a per-edge field across the network, or every field on one edge.
    print_states(), table()
        Show / return the full per-edge mean-flow state table.
    residuals(), print_residuals()
        The converged residual broken down equation-by-equation.
    composite(key), composites
        Read a composite element's hidden interior (e.g. an orifice throat).
    species(e), mixture_fractions(e), marker(e)
        Per-edge chemistry: solved species, transported feed fractions, burnt marker.
    cuton_report()
        Per-duct plane-wave validity ceiling (higher-order-mode cut-on).
    to_yaml(path)
        Write the network and these results to a UI-readable YAML case.
    """

    network: Network
    problem: CompiledProblem
    result: object  # SolveResult

    def __repr__(self) -> str:
        """Return a string representation of the solution."""
        return f"Solution(converged={self.converged}, iterations={self.iterations}, residual_norm={self.residual_norm})"

    def _repr_html_(self) -> str:
        """Return an HTML representation of the solution."""
        return (
            f"<div>Converged: {self.converged}</div><div>Iterations: {self.iterations}</div>"
            f"<div>Residual norm: {self.residual_norm}</div>"
        )

    @property
    def converged(self) -> bool:
        """Whether the solver reached the convergence tolerance."""
        return self.result.converged

    @property
    def iterations(self) -> int:
        """Number of Newton iterations taken."""
        return self.result.iterations

    @property
    def residual_norm(self) -> float:
        """Final scaled-residual 2-norm -- the quantity the solve drives below its ``tol``."""
        return self.result.residual_norm

    @property
    def x(self) -> np.ndarray:
        """Raw converged state vector."""
        return self.result.x

    def table(self, show_internal: bool = True) -> np.ndarray:
        """Return the per-edge state table (rows are fields, columns are edges).

        Parameters
        ----------
        show_internal : bool, optional
            When ``False`` and the network carries composite elements, drop the composite
            *internal* edge columns, leaving only the user-facing edges (which keep their
            ids; internals append at the tail).  Default ``True`` (every edge).
        """
        est = states_table(self.problem, self.result.x)
        cm = self.problem.composite_map
        if show_internal or cm is None or not cm.internal_edges:
            return est
        keep = [e for e in range(self.problem.n_edges) if e not in cm.internal_edges]
        return est[:, keep]

    def composite(self, key) -> "CompositeView":
        """Read the hidden interior of a solved composite element.

        A composite (e.g. an orifice or a tapered nozzle) is added as one element but expands into
        several sub-elements joined by internal edges that :meth:`table` hides by default.  This
        returns a :class:`~nefes.elements.composite.CompositeView` over that interior -- its internal
        edges, and, for a contracting composite, its throat.

        Parameters
        ----------
        key : str or int
            The composite's name or its user node id.

        Returns
        -------
        CompositeView
            A view exposing the composite's internal edges and (where it contracts) throat state.
        """
        cm = self.problem.composite_map
        if cm is None:
            raise ValueError("this network has no composite elements")
        if isinstance(key, str):
            nodes = [n for n, nm in cm.composite_name.items() if nm == key]
            if not nodes:
                raise ValueError(f"no composite named {key!r}; have {sorted(cm.composite_name.values())}")
            node = nodes[0]
        else:
            node = int(key)
            if node not in cm.composite_name:
                raise ValueError(f"user node {node} is not a composite element")
        expanded = set(cm.expanded_nodes(node))
        tail, head, area = self.problem.tail_node, self.problem.head_node, self.problem.area
        internal = tuple(sorted(e for e in cm.internal_edges if int(tail[e]) in expanded and int(head[e]) in expanded))
        throat = min(internal, key=lambda e: float(area[e])) if internal else None
        return CompositeView(
            name=cm.composite_name[node],
            kind=cm.composite_kind.get(node, ""),
            node=node,
            nodes=tuple(cm.expanded_nodes(node)),
            internal_edges=internal,
            throat=throat,
            _solution=self,
        )

    @property
    def composites(self) -> List["CompositeView"]:
        """Every composite element in the network, as :class:`CompositeView` projections."""
        cm = self.problem.composite_map
        if cm is None:
            return []
        return [self.composite(n) for n in sorted(cm.composite_name)]

    def cuton_report(self, section: str = "circular", aspect: float = 1.0):
        """Per-duct higher-order-mode cut-on frequencies and the plane-wave ceiling.

        The Nefes acoustic layer is plane-wave (1-D); it is valid only below the first
        duct cut-on frequency.  This reports the cut-on of every edge (from its area,
        sound speed and Mach) and the network-wide ceiling
        (:attr:`~nefes.perturbation.CutOnReport.f_cuton`) -- keep any perturbation
        analysis below it.

        Parameters
        ----------
        section : {"circular", "square", "rectangular"}, optional
            Assumed duct cross-section shape (Nefes ducts store only an area).
        aspect : float, optional
            Width-to-height ratio (``>= 1``) for ``section="rectangular"``, used to recover the
            larger transverse dimension (which sets the cut-on) from the area.  Ignored for the
            circular and square sections; default ``1.0`` (a square).

        Returns
        -------
        nefes.perturbation.CutOnReport
        """
        from ..perturbation.fields.cuton import duct_cuton_frequencies

        return duct_cuton_frequencies(
            self.problem, self.result.x, section=section, aspect=aspect, names=self.network._edge_names
        )

    def plot(self, color_by=None, width_by=None, **kwargs):
        """Draw this solved network as a node/edge diagram with the solved state on the edges (Plotly).

        The same diagram as :meth:`Network.plot`, with this solution attached: the edge hover carries
        the headline state, and ``color_by`` / ``width_by`` map any solved edge field onto edge color /
        arrow width.  Shares one backend (:func:`nefes.plotting.plot_network_topology`) with the
        structural view, so topology and results read the same way.

        Parameters
        ----------
        color_by : str, optional
            Solved edge field to color the edges by, e.g. ``"T"``, ``"M"``, ``"mdot"`` (keys of the
            per-edge state; see :meth:`field`).  Adds a colorbar and labels each edge with its value.
        width_by : str, optional
            Solved edge field whose magnitude scales each edge's arrow width (e.g. ``"mdot"`` for a
            flow-weighted diagram, ``"area"`` for a geometry-weighted one).
        **kwargs
            Forwarded to :func:`nefes.plotting.plot_network_topology` (e.g. ``colorscale``,
            ``show_edge_labels``, ``show_areas``, ``title``, ``height``, ``width``).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_network_topology

        return plot_network_topology(self.network, solution=self, color_by=color_by, width_by=width_by, **kwargs)

    def print_states(self, edges=None, precision: int = 5, file=None) -> None:
        """Print the per-edge mean-flow state table to the screen.

        Thin wrapper over :func:`nefes.solver.report.print_states`; see it for the column layout.
        In a notebook (and when ``file`` is not given) the table renders as rich HTML;
        otherwise a fixed-width text table is printed.

        Parameters
        ----------
        edges : sequence of int, optional
            Edge indices to include, in the given order (default: every edge).
        precision : int, optional
            Number of significant digits printed per value (default 5).
        file : file-like, optional
            Destination stream forwarded to :func:`print` (default ``sys.stdout``).
        """
        print_states(self.problem, self.result.x, edges=edges, precision=precision, file=file)

    def residuals(self) -> dict:
        """Return the converged residual broken down equation-by-equation.

        Resolves the single :attr:`residual_norm` into its per-equation contributions,
        keyed by a human-readable label (element/equation for node rows, edge + scalar
        for transport rows).  Useful for spotting which equation a non-converged solve
        left large.

        Returns
        -------
        dict
            ``{label: scaled_residual}`` for every equation, in residual-row order.
        """
        labels, _R, R_hat = residual_breakdown(self.problem, self.result.x)
        return {label: float(v) for label, v in zip(labels, R_hat)}

    def print_residuals(self, sort: bool = True, top=None, precision: int = 4, file=None) -> None:
        """Print the residual broken down equation-by-equation.

        Thin wrapper over :func:`nefes.solver.report.print_residuals`; see it for the column layout.

        Parameters
        ----------
        sort : bool, optional
            Order rows by descending ``|scaled residual|`` (default ``True``).
        top : int, optional
            Show only the worst ``top`` equations (default: all).
        precision : int, optional
            Significant digits printed per residual value (default 4).
        file : file-like, optional
            Destination stream forwarded to :func:`print` (default ``sys.stdout``).
        """
        print_residuals(self.problem, self.result.x, sort=sort, top=top, precision=precision, file=file)

    def edge(self, e: int) -> dict:
        """Return a ``{field: value}`` dict of all derived quantities on edge ``e``."""
        col = self.table()[:, e]
        return {name: float(col[idx]) for name, idx in _EDGE_FIELDS.items()}

    def field(self, name: str) -> np.ndarray:
        """Return the named field across all edges.

        Names are the keys of the per-edge state: ``mdot, p, h_t, rho, u, T, c, M, p_t,
        area``, plus the mixture molar mass ``W`` [kg/mol] and specific heat ``cp``
        [J/(kg K)] (the latter consistent with the local sound speed -- exact for a
        perfect gas, the frozen value on an unburnt edge and the equilibrium value on a
        burnt one).
        """
        return self.table()[_EDGE_FIELDS[name], :]

    def unchoked_nozzles(self) -> list:
        """Choked-nozzle outlets whose set back pressure is too high for the throat to choke.

        A :func:`~nefes.elements.catalog.choked_nozzle_outlet` asserts a sonic throat, which
        holds only while the ambient back pressure sits below the throat's critical (sonic)
        pressure ``p* = p_t (2 / (g + 1))^(g / (g - 1))`` (``g`` the local ratio of specific
        heats, ``g = c^2 W / (R_u T)``).  For every such element given a ``back_pressure`` at
        construction, this compares it to ``p*`` at the converged state and returns the nozzles
        that would *not* actually choke -- the compact choked-nozzle model does not apply to
        them.  One of the checks :meth:`verify` runs after a solve; call it directly to inspect.

        Returns
        -------
        list of dict
            One entry per offending nozzle: ``{"node", "name", "edge", "back_pressure",
            "critical_pressure", "p_t"}``.  Empty when every nozzle with a set back pressure is
            genuinely choked (or none set one).
        """
        p_t, c, W, T = self.field("p_t"), self.field("c"), self.field("W"), self.field("T")
        cmap = getattr(self.problem, "composite_map", None)
        out = []
        for user_n, el in enumerate(self.network._elements):
            # composites expand to several atomic nodes and are never a choked nozzle; skip them
            if is_composite(el) or getattr(el, "residual_id", None) != CHOKED_NOZZLE_OUTLET:
                continue
            if el.back_pressure is None:
                continue
            # map the user node id to its compiled (expanded) node id, then to its single edge
            node = cmap.expanded_nodes(user_n)[0] if cmap is not None else user_n
            e = int(self.problem.col_edge[self.problem.row_ptr[node]])
            gamma = float(c[e]) ** 2 * float(W[e]) / (R_UNIVERSAL * float(T[e]))
            p_crit = float(p_t[e]) * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
            if float(el.back_pressure) > p_crit:
                out.append(
                    {
                        "node": node,
                        "name": el.name,
                        "edge": e,
                        "back_pressure": float(el.back_pressure),
                        "critical_pressure": p_crit,
                        "p_t": float(p_t[e]),
                    }
                )
        return out

    def verify(self) -> list:
        """Run the post-solve model-validity checks and return one message per issue found.

        The single home for checks that can only be evaluated once the mean flow is converged
        (as opposed to the structural checks :func:`nefes.shell.build.validate_network` runs at
        compile time).  Each check is gated by its ``CHECK_*`` toggle in
        :mod:`nefes.shell.checks`; currently this is the choked-nozzle back-pressure check
        (:meth:`unchoked_nozzles`, gated by ``CHECK_CHOKED_NOZZLE``).  :meth:`Network.solve`
        calls this on a converged solution and re-emits each message as a warning; call it
        directly to collect them without the warnings machinery.

        Returns
        -------
        list of str
            Human-readable messages, one per issue; empty when the solution passes every check.
        """
        messages = []
        if checks.CHECK_CHOKED_NOZZLE:
            for nz in self.unchoked_nozzles():
                messages.append(
                    f"choked_nozzle_outlet {nz['name']!r} (edge {nz['edge']}): the specified back pressure "
                    f"{nz['back_pressure']:.4g} Pa exceeds the throat's critical pressure "
                    f"{nz['critical_pressure']:.4g} Pa, so the nozzle would not choke -- the compact choked-nozzle "
                    f"model does not apply here; use a pressure_outlet, which handles the choked/unchoked "
                    f"transition against a back pressure."
                )
        return messages

    def mixture_fractions(self, e: int) -> dict:
        """Transported feed-stream mixture fractions ``{stream_label: xi}`` on edge ``e``.

        These are the conserved scalars the solver carries (one per distinct injected feed),
        not chemical species; for the actual species use :meth:`species`.  Empty for a
        perfect gas with no passive scalars.
        """
        names = self.problem.scalar_names
        xi = self.result.x[3 : 3 + self.problem.n_elem, e]
        return {name: float(v) for name, v in zip(names, xi)}

    def marker(self, e: int) -> float:
        """Transported burnt marker on edge ``e`` (``0`` fresh / unburnt, ``1`` burnt).

        The scalar that gates the reacting closure's frozen/equilibrium blend.  Bimodal at
        convergence (a flame is a discrete ``0 -> 1`` jump), so it reads ~0 on an unburnt edge
        and ~1 on a burnt one.  Returns ``0.0`` when the network carries no marker (a perfect
        gas, or a reacting network built with an explicit hard per-edge closure).
        """
        mr = int(getattr(self.problem, "marker_row", -1))
        return 0.0 if mr < 0 else float(self.result.x[mr, e])

    def _chemistry_caches(self):
        """Lazily build and cache the per-edge product moles and per-stream mass fractions."""
        if getattr(self, "_chem_cache", None) is None:
            from ..chem.chemistry import product_moles, stream_mass_fractions

            lib = self.network.gas.library
            moles = product_moles(self.problem, self.result.x)
            stream_Y = None if lib is None else stream_mass_fractions(self.network._elements, lib)
            self._chem_cache = (moles, stream_Y)
        return self._chem_cache

    def species(self, e: int, basis: str = "mole") -> dict:
        """Solved chemical species ``{name: fraction}`` on edge ``e``.

        A burnt (equilibrium) edge reports its HP-equilibrium products; an unburnt (frozen)
        edge reports the forward blend of its feed streams; a perfect-gas edge has no
        chemical species (use :meth:`mixture_fractions` for its passive scalars).

        Parameters
        ----------
        e : int
            Edge id.
        basis : {"mole", "mass"}, optional
            Mole or mass fractions (default ``"mole"``).

        Returns
        -------
        dict
            ``{species_name: fraction}`` for the species present on the edge.
        """
        from ..chem.chemistry import edge_species

        moles, stream_Y = self._chemistry_caches()
        lib = self.network.gas.library
        return edge_species(self.problem, self.result.x, e, lib, basis=basis, moles=moles, stream_Y=stream_Y)

    def to_yaml(self, path: str, dataset: str = "Mean flow", **kwargs) -> None:
        """Write the network and this solution's results to a UI-readable YAML case.

        Embeds the mean-flow fields (and any transported chemistry) as a named dataset the UI can
        load.  If ``path`` does not yet exist, a fresh case is written.  If it exists -- and already
        holds this same network -- the results are *appended* as a new dataset, so several solutions
        (e.g. operating points) can be overlaid in one file from repeated calls with distinct
        ``dataset`` names.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.  Appended to when it already exists.
        dataset : str, optional
            Name for this solution's mean-flow dataset (default ``"Mean flow"``).  Appending a
            dataset whose name is already present in the file raises ``ValueError``.
        **kwargs
            Forwarded to :func:`nefes.io.save_case` / :func:`nefes.io.dump_case` (e.g. ``fields``,
            ``node_data``, ``forced``, ``title``).
        """
        from ..io import save_solution

        save_solution(self.network, self, path, dataset=dataset, **kwargs)

    def save(self, path: str, dataset: str = "Mean flow", **kwargs) -> None:
        """Alias for :meth:`to_yaml` (kept for intuitive usage)."""
        self.to_yaml(path, dataset=dataset, **kwargs)
