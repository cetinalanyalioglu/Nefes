"""Network builder: add elements, connect them with directed edges, then compile or solve."""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..thermo.configure import ThermoConfig, perfect_gas
from ..thermo.api import PERFECT_GAS, EQ_KERNEL
from ..connectivity import build_connectivity
from ..elements import catalog as cat
from ..elements.catalog import ElementSpec
from ..elements.ids import RESIDUAL_NAMES
from ..problem import CompiledProblem
from ..solver import solve as _solve
from ..solver.control import states_table, initial_guess, print_states, residual_breakdown, print_residuals
from ..derive import ES_MDOT, ES_P, ES_HT, ES_RHO, ES_U, ES_T, ES_C, ES_M, ES_PT, ES_AREA, ES_W, ES_CP

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
    """The main object for building and solving flow networks."""

    def __init__(
        self,
        gas: Optional[ThermoConfig] = None,
        p_ref=101325.0,
        T_ref=300.0,
        mdot_ref=None,
        h_ref=None,
        nodes=None,
        edges=None,
        edge_models=None,
    ):
        """Create a network, optionally fully specified in one shot.

        The network can be built incrementally with :meth:`add` / :meth:`connect`, or
        constructed complete by passing ``nodes`` and ``edges`` -- the convenient one-shot
        form that supersedes the lower-level :func:`fns.elements.catalog.build_problem`.

        Parameters
        ----------
        gas : ThermoConfig, optional
            The thermodynamic model (default: dry-air perfect gas).
        p_ref : float, optional
            Absolute-pressure gauge reference [Pa] (default 101325).
        T_ref : float, optional
            Reference temperature [K] for the initial guess (default 300).
        mdot_ref, h_ref : float, optional
            Hidden seed overrides for the residual scaling; normally auto-derived and
            re-measured during the solve (see :attr:`mdot_ref` / :attr:`h_ref`).
        nodes : sequence of ElementSpec, optional
            The elements, in node order -- attached via :meth:`add`.
        edges : sequence of tuple, optional
            Directed edges ``(tail, head, area)`` referencing node indices, attached via
            :meth:`connect` (ports auto-assigned in attachment order).
        edge_models : sequence of int, optional
            Per-edge thermo-model id override aligned with ``edges`` (e.g. frozen-unburnt
            vs equilibrium-burnt across a flame); ``None`` entries use the gas default.
        """
        self.gas = gas if gas is not None else perfect_gas()
        self.p_ref = p_ref
        self.T_ref = T_ref
        self._mdot_ref = mdot_ref
        # Explicit absolute-enthalpy datum; if None, falls back to ``cp * T_ref`` (perfect-gas convention).
        # Reacting closures need the gas's absolute-enthalpy reference here instead.
        self._h_ref = h_ref
        self._elements: List[ElementSpec] = []
        self._edges: List[Tuple[int, int, float]] = []
        self._ports: List[Tuple[Optional[int], Optional[int]]] = []
        self._edge_names: List[str] = []
        # Per-edge thermo-model override (None -> the gas config's model on that edge).
        self._edge_models: List[Optional[int]] = []
        # Provenance metadata for the network (e.g. from the UI).
        self.provenance = None

        for spec in nodes or ():
            self.add(spec)
        if edge_models is not None and edges is None:
            raise ValueError("edge_models was given without edges")
        edges = list(edges or ())
        models = list(edge_models) if edge_models is not None else [None] * len(edges)
        if len(models) != len(edges):
            raise ValueError(f"edge_models has {len(models)} entries but there are {len(edges)} edges")
        for (tail, head, area), model in zip(edges, models):
            self.connect(tail, head, area, edge_model=model)

    # -- construction -------------------------------------------------------------------------------------------------

    def add(self, spec: ElementSpec) -> int:
        """Add an element and return its node index."""
        self._elements.append(spec)
        return len(self._elements) - 1

    def connect(
        self, tail: int, head: int, area: float, name: str = "", tail_port=None, head_port=None, edge_model=None
    ) -> int:
        """Add a directed edge from element `tail` to element `head`, returning its edge id.

        The returned integer is the edge index in the compiled problem -- capture it to wire a
        dynamic source's ``ref_edge`` (e.g. the edge just upstream of a flame) without guessing.

        `tail_port`/`head_port` pin the local port indices at each endpoint; leave them `None` to let the
        compiler auto-assign ports in attachment order.  `edge_model` overrides the per-edge thermo-model id
        (e.g. a frozen-unburnt vs. equilibrium-burnt split across a flame); leave `None` to use the gas
        config's default model on this edge.
        """
        idx = len(self._edges)
        self._edges.append((tail, head, float(area)))
        self._ports.append((tail_port, head_port))
        self._edge_models.append(None if edge_model is None else int(edge_model))
        # Edge name is optional, defaulting to "e<index>".
        self._edge_names.append(name or f"e{idx}")
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
        """Attach (or replace) the dynamic-source descriptor on an already-added element.

        Lets the network be wired up first -- so a flame's ``ref_edge`` can be taken from the edge id
        :meth:`connect` returns -- and the ``S(omega)`` source attached afterwards, rather than baking a
        guessed edge index into the element at construction time.  The mean solve ignores the source; the
        perturbation layer consumes it.

        Parameters
        ----------
        node : int
            Element index (as returned by :meth:`add`) to carry the source.
        source : DynamicSource or None
            The descriptor (e.g. from :func:`fns.elements.dynamic_source.n_tau_flame`); ``None`` clears it.

        Returns
        -------
        int
            The same ``node``, for chaining.
        """
        self._elements[node].dynamic_source = source
        return node

    @property
    def h_ref(self) -> float:
        """Reference enthalpy scale (a hidden override; normally auto-derived).

        Like :attr:`mdot_ref`, only the seed for the residual scaling -- the solve re-measures
        the enthalpy scale from the realized inflow.  Auto-derivation is the perfect-gas
        ``cp * T_ref``; the reacting backend passes its formation-inclusive datum explicitly.
        Set ``h_ref=`` only to override.
        """
        if self._h_ref is not None:
            return self._h_ref
        return self.gas.tf[0] * self.T_ref

    @property
    def mdot_ref(self) -> float:
        """Reference mass-flow scale (a hidden override; normally auto-derived).

        This is only the *seed* for the residual scaling -- the solve re-measures it from
        the realized inflow at each homotopy stage (``adaptive_scale``) -- so it need only be
        order-of-magnitude right.  Auto-derivation: the **total** specified inflow when every
        inlet is a mass-flow inlet; otherwise a dP-based isentropic estimate
        ``A * sqrt(2 rho dP_max)`` from the boundary pressures (replacing the old
        pressure-blind M=0.3 guess); a quiescent / pressureless network falls back to the
        M=0.3 estimate.  Set ``mdot_ref=`` on the network only to override it.
        """
        if self._mdot_ref is not None:
            return self._mdot_ref
        mass = [abs(el.fparams[0]) for el in self._elements if el.residual_id == cat.MASS_FLOW_INLET]
        has_pt = any(el.residual_id == cat.PT_INLET for el in self._elements)
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
        refs = [el.fparams[0] for el in self._elements if el.residual_id in (cat.PT_INLET, cat.P_OUTLET)]
        return (max(refs) - min(refs)) if len(refs) >= 2 else 0.0

    # -- compile / solve ----------------------------------------------------------------------------------------------

    def _resolve_edge_models(self):
        """Return the per-edge thermo-model array, or ``None`` if every edge uses the gas default."""
        if all(m is None for m in self._edge_models):
            return None
        default = int(self.gas.model_id)
        return np.array([default if m is None else m for m in self._edge_models], dtype=np.int64)

    def compile(self) -> CompiledProblem:
        """Compile the elements and edges into an immutable ``CompiledProblem``."""
        edge_models = self._resolve_edge_models()
        # If the ports are explicitly set, use the connectivity builder.
        explicit = self._edges and all(tp is not None and hp is not None for (tp, hp) in self._ports)
        if explicit:
            endpoints = [(t, int(tp), h, int(hp)) for (t, h, _a), (tp, hp) in zip(self._edges, self._ports)]
            conn = build_connectivity(len(self._elements), endpoints)
            area = np.array([a for (_t, _h, a) in self._edges], dtype=np.float64)
            return cat.build_problem_from_connectivity(
                self.gas, self._elements, conn, area, self.mdot_ref, self.p_ref, self.h_ref, edge_models=edge_models
            )
        return cat.build_problem(
            self.gas, self._elements, self._edges, self.mdot_ref, self.p_ref, self.h_ref, edge_models=edge_models
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
            Maximum Newton iterations per homotopy stage (default ``80``).
        kappa_stages : sequence of float, optional
            Vanishing-friction homotopy schedule, warm-started in order (default ``(0.1, 0.01, 0.0)``).
        verbose : int or bool, optional
            Progress verbosity (default ``0``). ``0``/``False`` is silent; ``1``/``True`` prints a one-line
            gross-residual summary per homotopy stage; ``2`` additionally prints the scaled residual broken down by
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
        return Solution(self, prob, res)

    def initial_guess(self, **kw):
        """Return the solver's initial state guess for the compiled problem."""
        return initial_guess(self.compile(), **kw)

    def plot_topology(self, **kwargs):
        """Draw the network topology (element indices/names and edge directions).

        A structural diagnostic -- no solve needed.  Thin wrapper over
        :func:`fns.plotting.plot_network_topology`; see it for the keyword options.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..plotting import plot_network_topology

        return plot_network_topology(self, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Write this network as a UI-readable YAML case (no result data).

        Thin wrapper over :func:`fns.io.save_case`; see it for the full set of keyword options.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`fns.io.save_case`.
        """
        from ..io import save_case

        save_case(self, path, **kwargs)

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
        """``(p_ref, T_ref, mdot_ref_or_None, mdot_is_explicit)`` for the repr headers."""
        try:
            # The auto-derive medians the edge areas; an edge-less network yields a quiet NaN
            # (suppress numpy's "mean of empty slice" warning -- we report it as "n/a" below).
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = float(self.mdot_ref)
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
        """Human-readable residual-type name for an element."""
        return RESIDUAL_NAMES.get(el.residual_id, f"residual#{el.residual_id}")

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

        out = [header]
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
            out.append(_html_table(("#", "name", "type"), body, ("right", "left", "left")))
            if n_el > _REPR_MAX_ROWS:
                out.append(f"<div style='color:#888;font-size:0.85em'>... ({n_el - _REPR_MAX_ROWS} more)</div>")

        if n_ed:
            body = [
                [str(i), f"{self._node_label(t)} &rarr; {self._node_label(h)}", f"{a:.4g}", self._edge_names[i]]
                for i, (t, h, a) in enumerate(self._edges[:_REPR_MAX_ROWS])
            ]
            out.append(
                _html_table(("#", "connection", "area [m&sup2;]", "name"), body, ("right", "left", "right", "left"))
            )
            if n_ed > _REPR_MAX_ROWS:
                out.append(f"<div style='color:#888;font-size:0.85em'>... ({n_ed - _REPR_MAX_ROWS} more)</div>")
        return "".join(out)


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
    """Converged mean-flow result with named edge-field access."""

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
        """Final residual norm."""
        return self.result.residual_norm

    @property
    def x(self) -> np.ndarray:
        """Raw converged state vector."""
        return self.result.x

    def table(self) -> np.ndarray:
        """Return the per-edge state table (rows are fields, columns are edges)."""
        return states_table(self.problem, self.result.x)

    def cuton_report(self, section: str = "circular"):
        """Per-duct higher-order-mode cut-on frequencies and the plane-wave ceiling.

        The FNS acoustic layer is plane-wave (1-D); it is valid only below the first
        duct cut-on frequency.  This reports the cut-on of every edge (from its area,
        sound speed and Mach) and the network-wide ceiling
        (:attr:`~fns.perturbation.CutOnReport.f_cuton`) -- keep any perturbation
        analysis below it.

        Parameters
        ----------
        section : {"circular", "square"}, optional
            Assumed duct cross-section shape (FNS ducts store only an area).

        Returns
        -------
        fns.perturbation.CutOnReport
        """
        from ..perturbation.cuton import duct_cuton_frequencies

        return duct_cuton_frequencies(self.problem, self.result.x, section=section, names=self.network._edge_names)

    def print_states(self, edges=None, precision: int = 5, file=None) -> None:
        """Print the per-edge mean-flow state table to the screen.

        Thin wrapper over :func:`fns.solver.control.print_states`; see it for the column layout.
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

        Thin wrapper over :func:`fns.solver.control.print_residuals`; see it for the column layout.

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

    def mixture_fractions(self, e: int) -> dict:
        """Transported feed-stream mixture fractions ``{stream_label: xi}`` on edge ``e``.

        These are the conserved scalars the solver carries (one per distinct injected feed),
        not chemical species; for the actual species use :meth:`species`.  Empty for a
        perfect gas with no passive scalars.
        """
        names = self.problem.scalar_names
        xi = self.result.x[3 : 3 + self.problem.n_elem, e]
        return {name: float(v) for name, v in zip(names, xi)}

    def _chemistry_caches(self):
        """Lazily build and cache the per-edge product moles and per-stream mass fractions."""
        if getattr(self, "_chem_cache", None) is None:
            from ..chemistry import product_moles, stream_mass_fractions

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
        from ..chemistry import edge_species

        moles, stream_Y = self._chemistry_caches()
        lib = self.network.gas.library
        return edge_species(self.problem, self.result.x, e, lib, basis=basis, moles=moles, stream_Y=stream_Y)

    def save(self, path: str, **kwargs) -> None:
        """Write the network and this solution's results as a UI-readable case.

        Wrapper over :func:`fns.io.save_case` with this solution attached, so the mean-flow result fields are
        embedded as datasets the UI can load.

        Parameters
        ----------
        path : str
            Destination ``.yaml`` path.
        **kwargs
            Forwarded to :func:`fns.io.save_case` (e.g. ``fields``, ``node_data``, ``forced``, ``title``).
        """
        from ..io import save_case

        save_case(self.network, path, solution=self, **kwargs)
